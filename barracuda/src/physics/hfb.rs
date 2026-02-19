// SPDX-License-Identifier: AGPL-3.0-only

//! Spherical Hartree-Fock + BCS solver (Level 2).
//!
//! Self-consistent mean-field solver for medium-mass nuclei (56 ≤ A ≤ 132)
//! using the Skyrme energy density functional in a spherical harmonic
//! oscillator basis with BCS pairing.
//!
//! Port of: `control/surrogate/nuclear-eos/wrapper/skyrme_hfb.py`
//! See PHYSICS.md §5 for complete equation documentation.
//!
//! References:
//!   - Ring & Schuck, *The Nuclear Many-Body Problem*, Springer (2004)
//!   - Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003)
//!   - Vautherin & Brink, Phys. Rev. C 5, 626 (1972)
//!   - Bohr & Mottelson, *Nuclear Structure* Vol. I (1969)
//!
//! Physics features (matching Python reference):
//!   - Separate proton/neutron Hamiltonians and densities
//!   - BCS pairing: constant gap Δ = 12/√A MeV (Ring & Schuck §6.2)
//!   - Coulomb: Poisson direct + Slater exchange (Slater, Phys. Rev. 81, 385)
//!   - Isospin structure in Skyrme potential (t₀/x₀, t₃/x₃)
//!   - Effective kinetic matrix T_eff (t₁/t₂ effective mass terms)
//!   - Spin-orbit splitting (W₀ parameter)
//!   - Center-of-mass correction: E_CM = -3/4 ℏω (Bohr & Mottelson §4-2)
//!
//! Uses (all BarraCUDA native — zero external dependencies):
//!   - `barracuda::special::{gamma, laguerre}` for HO basis wavefunctions
//!   - `barracuda::numerical::{trapz, gradient_1d}` for radial integrals & derivatives
//!   - `barracuda::optimize::brent` for BCS chemical potential (matches scipy.optimize.brentq)
//!   - `barracuda::linalg::eigh_f64` for symmetric eigenvalue decomposition

use super::constants::*;
use super::hfb_common::Mat;
use super::semf::semf_binding_energy;
use barracuda::linalg::eigh_f64;
use barracuda::numerical::{gradient_1d, trapz};
use barracuda::ops::grid::compute_ls_factor;
use barracuda::special::{gamma, laguerre};
use std::collections::HashMap;

use crate::tolerances::{
    BCS_DENSITY_SKIP, COULOMB_R_MIN, DENSITY_FLOOR, RHO_POWF_GUARD, SHARP_FILLING_THRESHOLD,
    SPIN_ORBIT_R_MIN,
};
use std::f64::consts::PI;

/// Basis state quantum numbers
#[derive(Debug, Clone)]
struct BasisState {
    n: usize,
    l: usize,
    j: f64,
    deg: usize, // 2j+1
}

/// Spherical HF+BCS solver
pub struct SphericalHFB {
    z: usize,
    n_neutrons: usize,
    a: usize,
    r: Vec<f64>,
    dr: f64,
    nr: usize,
    hw: f64,      // hbar*omega (MeV)
    b: f64,       // HO length parameter (fm)
    delta_p: f64, // proton pairing gap (MeV)
    delta_n: f64, // neutron pairing gap (MeV)
    states: Vec<BasisState>,
    n_states: usize,
    wf: Vec<Vec<f64>>,  // [n_states][nr] — radial wavefunctions
    dwf: Vec<Vec<f64>>, // [n_states][nr] — wavefunction derivatives
    lj_blocks: HashMap<(usize, u64), Vec<usize>>, // (l, j_x1000) → state indices
}

/// Result from HFB solve
#[derive(Debug, Clone)]
pub struct HFBResult {
    pub binding_energy_mev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub delta_e: f64,
}

impl SphericalHFB {
    pub fn new(z: usize, n: usize, n_shells: usize, r_max: f64, n_grid: usize) -> Self {
        Self::build(z, n, n_shells, r_max, n_grid)
    }

    /// Adaptive parameters matching the Python reference solver defaults
    pub fn new_adaptive(z: usize, n: usize) -> Self {
        let a = z + n;
        let a_f = a as f64;
        let n_shells = (2.0 * a_f.powf(1.0 / 3.0)) as usize + 4;
        let n_shells = n_shells.max(8).min(14);
        let r_max = (1.2 * a_f.powf(1.0 / 3.0) + 8.0_f64).max(12.0);
        let n_grid = ((r_max * 12.0) as usize).max(120);
        Self::build(z, n, n_shells, r_max, n_grid)
    }

    pub const fn n_states(&self) -> usize {
        self.n_states
    }
    pub const fn nr(&self) -> usize {
        self.nr
    }
    pub const fn z(&self) -> usize {
        self.z
    }
    pub const fn n_neutrons(&self) -> usize {
        self.n_neutrons
    }
    pub const fn dr(&self) -> f64 {
        self.dr
    }
    pub const fn hw(&self) -> f64 {
        self.hw
    }

    /// Pairing gap (same for proton and neutron in this model)
    pub const fn pairing_gap(&self) -> f64 {
        self.delta_p
    }

    /// Flat wavefunctions: [n_states × nr] row-major
    pub fn wf_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_states * self.nr);
        for s in &self.wf {
            out.extend_from_slice(s);
        }
        out
    }

    /// Flat wavefunction derivatives: [n_states × nr] row-major
    pub fn dwf_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_states * self.nr);
        for s in &self.dwf {
            out.extend_from_slice(s);
        }
        out
    }

    /// Radial grid points
    pub fn r_grid(&self) -> &[f64] {
        &self.r
    }

    /// lj_same matrix: [n_states × n_states] u32 (1 if same (l,j) block)
    pub fn lj_same_flat(&self) -> Vec<u32> {
        let ns = self.n_states;
        let mut out = vec![0u32; ns * ns];
        for indices in self.lj_blocks.values() {
            for &i in indices {
                for &j in indices {
                    out[i * ns + j] = 1;
                }
            }
        }
        out
    }

    /// l(l+1) values per state
    pub fn ll1_values(&self) -> Vec<f64> {
        self.states
            .iter()
            .map(|s| (s.l * (s.l + 1)) as f64)
            .collect()
    }

    /// (l, j) quantum numbers per state — needed for spin-orbit coupling
    pub fn lj_quantum_numbers(&self) -> Vec<(usize, f64)> {
        self.states.iter().map(|s| (s.l, s.j)).collect()
    }

    /// Per-state wavefunction access (for spin-orbit integrals)
    pub fn wf_state(&self, i: usize) -> &[f64] {
        &self.wf[i]
    }

    /// Degeneracies (2j+1) per state
    pub fn deg_values(&self) -> Vec<f64> {
        self.states.iter().map(|s| s.deg as f64).collect()
    }

    /// Build the full Hamiltonian matrix for one species (proton or neutron).
    ///
    /// Returns a flat row-major `n_states × n_states` matrix suitable for
    /// packing into `BatchedEighGpu`.
    pub fn build_hamiltonian(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        is_proton: bool,
        params: &[f64],
        w0: f64,
    ) -> Vec<f64> {
        let ns = self.n_states;
        let nr = self.nr;

        let u_sky = self.skyrme_potential(rho_p, rho_n, is_proton, params);

        let u_total: Vec<f64> = if is_proton {
            let v_c = self.coulomb_direct(rho_p);
            let v_cx = self.coulomb_exchange(rho_p);
            (0..nr).map(|k| u_sky[k] + v_c[k] + v_cx[k]).collect()
        } else {
            u_sky
        };

        let t_eff = self.build_t_eff(rho_p, rho_n, is_proton, params);

        // H = T_eff + V
        let mut h = Mat::zeros(ns);
        for i in 0..ns {
            for j in 0..ns {
                h.set(i, j, t_eff.get(i, j));
            }
        }

        // Diagonal potential matrix elements
        for i in 0..ns {
            let integ: Vec<f64> = (0..nr)
                .map(|k| self.wf[i][k].powi(2) * u_total[k] * self.r[k].powi(2))
                .collect();
            h.add(
                i,
                i,
                barracuda::numerical::trapz(&integ, &self.r).unwrap_or(0.0),
            );

            // Spin-orbit — uses barracuda::ops::grid::compute_ls_factor
            if w0 != 0.0 && self.states[i].l > 0 {
                let ls = compute_ls_factor(self.states[i].l as u32, self.states[i].j);
                let rho_total: Vec<f64> = (0..nr).map(|k| rho_p[k] + rho_n[k]).collect();
                let drho = barracuda::numerical::gradient_1d(&rho_total, self.dr);
                let so_integ: Vec<f64> = (0..nr)
                    .map(|k| {
                        self.wf[i][k].powi(2) * drho[k] / self.r[k].max(SPIN_ORBIT_R_MIN)
                            * self.r[k].powi(2)
                    })
                    .collect();
                h.add(
                    i,
                    i,
                    w0 * ls * barracuda::numerical::trapz(&so_integ, &self.r).unwrap_or(0.0),
                );
            }
        }

        // Off-diagonal: same (l, j) block
        for ((_l, _j_key), indices) in &self.lj_blocks {
            for ii in 0..indices.len() {
                for jj in (ii + 1)..indices.len() {
                    let idx_i = indices[ii];
                    let idx_j = indices[jj];
                    let integ: Vec<f64> = (0..nr)
                        .map(|k| {
                            self.wf[idx_i][k] * self.wf[idx_j][k] * u_total[k] * self.r[k].powi(2)
                        })
                        .collect();
                    let val = barracuda::numerical::trapz(&integ, &self.r).unwrap_or(0.0);
                    h.add(idx_i, idx_j, val);
                    h.add(idx_j, idx_i, val);
                }
            }
        }

        h.data
    }

    /// Compute BCS occupations from externally-provided eigenvalues.
    ///
    /// Returns (v2_occupations, chemical_potential).
    pub fn bcs_occupations_from_eigs(
        &self,
        eigenvalues: &[f64],
        num_particles: usize,
        delta: f64,
    ) -> (Vec<f64>, f64) {
        self.bcs_occupations(eigenvalues, num_particles, delta)
    }

    /// Compute density from BCS-weighted eigenstates (GPU-unpacked format).
    ///
    /// `eigvecs` is row-major [n_states × n_states] from GPU.
    /// `v2` is the BCS occupation for each state.
    /// Returns radial density profile \[nr\].
    pub fn density_from_eigenstates(&self, eigvecs: &[f64], v2: &[f64], ns: usize) -> Vec<f64> {
        let nr = self.nr;
        let degs: Vec<f64> = self.states.iter().map(|s| s.deg as f64).collect();
        let mut rho = vec![DENSITY_FLOOR; nr];

        for i in 0..ns {
            if degs[i] * v2[i] < BCS_DENSITY_SKIP {
                continue;
            }
            let mut phi = vec![0.0; nr];
            for j in 0..ns {
                let c = eigvecs[j * ns + i];
                for k in 0..nr {
                    phi[k] += c * self.wf[j][k];
                }
            }
            for k in 0..nr {
                rho[k] += degs[i] * v2[i] * phi[k].powi(2) / (4.0 * PI);
            }
        }

        for k in 0..nr {
            rho[k] = rho[k].max(DENSITY_FLOOR);
        }
        rho
    }

    /// Compute total energy from proton/neutron densities and eigendecompositions.
    ///
    /// Used by the GPU-batched solver where eigensolves happen externally.
    pub fn compute_energy_from_densities(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        eigs_p: &[f64],
        vecs_p: &[f64],
        eigs_n: &[f64],
        vecs_n: &[f64],
        params: &[f64],
    ) -> f64 {
        let ns = self.n_states;
        let delta_p = self.delta_p;
        let delta_n = self.delta_n;

        // Reconstruct v2 for energy calculation
        let (v2_p, _) = self.bcs_occupations(eigs_p, self.z, delta_p);
        let (v2_n, _) = self.bcs_occupations(eigs_n, self.n_neutrons, delta_n);

        let results_p = SpeciesResult {
            _eigenvalues: eigs_p.to_vec(),
            eigvecs: vecs_p.to_vec(),
            n: ns,
            v2: v2_p,
            _lambda: 0.0,
        };
        let results_n = SpeciesResult {
            _eigenvalues: eigs_n.to_vec(),
            eigvecs: vecs_n.to_vec(),
            n: ns,
            v2: v2_n,
            _lambda: 0.0,
        };

        self.compute_energy(rho_p, rho_n, &results_p, &results_n, params, false)
    }

    /// Fast energy calculation that accepts pre-computed v2 occupations,
    /// avoiding a redundant BCS root-solve inside compute_energy_from_densities.
    pub fn compute_energy_with_v2(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        eigs_p: &[f64],
        vecs_p: &[f64],
        eigs_n: &[f64],
        vecs_n: &[f64],
        v2_p: &[f64],
        v2_n: &[f64],
        params: &[f64],
    ) -> f64 {
        let ns = self.n_states;
        let results_p = SpeciesResult {
            _eigenvalues: eigs_p.to_vec(),
            eigvecs: vecs_p.to_vec(),
            n: ns,
            v2: v2_p.to_vec(),
            _lambda: 0.0,
        };
        let results_n = SpeciesResult {
            _eigenvalues: eigs_n.to_vec(),
            eigvecs: vecs_n.to_vec(),
            n: ns,
            v2: v2_n.to_vec(),
            _lambda: 0.0,
        };
        self.compute_energy(rho_p, rho_n, &results_p, &results_n, params, false)
    }

    fn build(z: usize, n: usize, n_shells: usize, r_max: f64, n_grid: usize) -> Self {
        let a = z + n;
        // HO frequency: ℏω = 41 A^(-1/3) MeV — Bohr & Mottelson, §2-4a
        let hw = 41.0 * (a as f64).powf(-1.0 / 3.0);
        // Oscillator length: b = ℏc / √(m_N ℏω)
        let b = HBAR_C / (M_NUCLEON * hw).sqrt();
        let dr = r_max / n_grid as f64;
        let r: Vec<f64> = (1..=n_grid).map(|i| i as f64 * dr).collect();

        // Pairing gap: Δ = 12/√A MeV — Ring & Schuck (2004), §6.2
        // Phenomenological constant-gap BCS approximation for odd-even staggering
        let delta = 12.0 / (a.max(4) as f64).sqrt();

        let mut hfb = SphericalHFB {
            z,
            n_neutrons: n,
            a,
            r,
            dr,
            nr: n_grid,
            hw,
            b,
            delta_p: delta,
            delta_n: delta,
            states: Vec::new(),
            n_states: 0,
            wf: Vec::new(),
            dwf: Vec::new(),
            lj_blocks: HashMap::new(),
        };
        hfb.build_basis(n_shells);
        hfb.compute_wavefunctions();
        hfb
    }

    fn build_basis(&mut self, n_shells: usize) {
        for n_sh in 0..n_shells {
            for l in 0..=n_sh {
                let n_rad = (n_sh - l) / 2;
                if (n_sh - l) % 2 != 0 {
                    continue;
                }
                if l > 0 {
                    for &j2 in &[2 * l as i32 - 1, 2 * l as i32 + 1] {
                        self.states.push(BasisState {
                            n: n_rad,
                            l,
                            j: j2 as f64 / 2.0,
                            deg: (j2 + 1) as usize,
                        });
                    }
                } else {
                    self.states.push(BasisState {
                        n: n_rad,
                        l: 0,
                        j: 0.5,
                        deg: 2,
                    });
                }
            }
        }
        self.n_states = self.states.len();

        // Build (l, j) block map for efficient block-diagonal operations
        for (i, s) in self.states.iter().enumerate() {
            let key = (s.l, (s.j * 1000.0) as u64);
            self.lj_blocks.entry(key).or_default().push(i);
        }
    }

    fn compute_wavefunctions(&mut self) {
        self.wf = self
            .states
            .iter()
            .map(|s| Self::ho_radial(s.n, s.l, &self.r, self.b))
            .collect();

        // Compute derivatives: dR/dr via 2nd-order finite differences (matches numpy.gradient)
        self.dwf = self
            .wf
            .iter()
            .map(|wf_i| gradient_1d(wf_i, self.dr))
            .collect();
    }

    /// HO radial wavefunction R_{nl}(r)
    fn ho_radial(n: usize, l: usize, r: &[f64], b: f64) -> Vec<f64> {
        let alpha = l as f64 + 0.5;
        let n_fact = barracuda::special::factorial(n);
        let gamma_val = gamma(n as f64 + l as f64 + 1.5).unwrap_or(1.0);
        let norm = (2.0 * n_fact / (b.powi(3) * gamma_val)).abs().sqrt();

        r.iter()
            .map(|&ri| {
                let xi = (ri / b).powi(2);
                let lag = laguerre(n, alpha, xi);
                norm * (ri / b).powi(l as i32) * (-xi / 2.0).exp() * lag
            })
            .collect()
    }

    // ─── Coulomb ─────────────────────────────────────────────────────

    /// Coulomb direct potential from proton density (spherical Poisson)
    fn coulomb_direct(&self, rho_p: &[f64]) -> Vec<f64> {
        let nr = self.nr;
        let dr = self.dr;

        let mut charge_enclosed = vec![0.0; nr];
        let mut cumsum = 0.0;
        for k in 0..nr {
            cumsum += rho_p[k] * 4.0 * PI * self.r[k].powi(2) * dr;
            charge_enclosed[k] = cumsum;
        }

        let mut phi_outer = vec![0.0; nr];
        let mut cumsum_rev = 0.0;
        for k in (0..nr).rev() {
            cumsum_rev += rho_p[k] * 4.0 * PI * self.r[k] * dr;
            phi_outer[k] = cumsum_rev;
        }

        (0..nr)
            .map(|k| E2 * (charge_enclosed[k] / self.r[k].max(COULOMB_R_MIN) + phi_outer[k]))
            .collect()
    }

    /// Coulomb exchange (Slater approximation)
    fn coulomb_exchange(&self, rho_p: &[f64]) -> Vec<f64> {
        rho_p
            .iter()
            .map(|&rp| super::hfb_common::coulomb_exchange_slater(rp))
            .collect()
    }

    // ─── Skyrme potential ────────────────────────────────────────────

    fn skyrme_potential(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        is_proton: bool,
        params: &[f64],
    ) -> Vec<f64> {
        let (t0, t3) = (params[0], params[3]);
        let (x0, x3) = (params[4], params[7]);
        let alpha = params[8];

        (0..self.nr)
            .map(|k| {
                let rho = rho_p[k] + rho_n[k];
                let rho_q = if is_proton { rho_p[k] } else { rho_n[k] };
                let rho_safe = rho.max(RHO_POWF_GUARD);

                let u_t0 = t0 * ((1.0 + x0 / 2.0) * rho - (0.5 + x0) * rho_q);

                let rho_alpha = rho_safe.powf(alpha);
                let rho_alpha_m1 = if rho > DENSITY_FLOOR {
                    rho_safe.powf(alpha - 1.0)
                } else {
                    0.0
                };
                let sum_rho2 = rho_p[k].powi(2) + rho_n[k].powi(2);

                let u_t3 = (t3 / 12.0)
                    * ((1.0 + x3 / 2.0) * (alpha + 2.0) * rho_alpha * rho
                        - (0.5 + x3) * (alpha * rho_alpha_m1 * sum_rho2 + 2.0 * rho_alpha * rho_q));

                u_t0 + u_t3
            })
            .collect()
    }

    // ─── BCS pairing ─────────────────────────────────────────────────

    fn bcs_occupations(
        &self,
        eigenvalues: &[f64],
        num_particles: usize,
        delta: f64,
    ) -> (Vec<f64>, f64) {
        if num_particles == 0 {
            return (vec![0.0; self.n_states], 0.0);
        }

        let degs: Vec<f64> = self.states.iter().map(|s| s.deg as f64).collect();

        if delta < SHARP_FILLING_THRESHOLD {
            return self.sharp_filling(eigenvalues, num_particles, &degs);
        }

        let num_p = num_particles as f64;

        let particle_number = |lam: f64| -> f64 {
            let mut total = 0.0;
            for i in 0..eigenvalues.len() {
                let ek = eigenvalues[i] - lam;
                let big_ek = (ek * ek + delta * delta).sqrt();
                let v2 = 0.5 * (1.0 - ek / big_ek);
                total += degs[i] * v2;
            }
            total - num_p
        };

        let e_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min) - 50.0;
        let e_max = eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            + 50.0;

        // Brent's method (matches scipy.optimize.brentq precision)
        let lam = match barracuda::optimize::brent(
            particle_number,
            e_min,
            e_max,
            crate::tolerances::BRENT_TOLERANCE,
            100,
        ) {
            Ok(result) => result.root,
            Err(_) => self.approx_fermi(eigenvalues, num_particles, &degs),
        };

        let v2: Vec<f64> = eigenvalues
            .iter()
            .map(|&eps| {
                let ek = eps - lam;
                let big_ek = (ek * ek + delta * delta).sqrt();
                0.5 * (1.0 - ek / big_ek)
            })
            .collect();

        (v2, lam)
    }

    fn sharp_filling(
        &self,
        eigenvalues: &[f64],
        num_particles: usize,
        degs: &[f64],
    ) -> (Vec<f64>, f64) {
        let mut idx: Vec<usize> = (0..self.n_states).collect();
        idx.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

        let mut v2 = vec![0.0; self.n_states];
        let mut remaining = num_particles as f64;
        for &i in &idx {
            let fill = remaining.min(degs[i]);
            v2[i] = fill / degs[i];
            remaining -= fill;
            if remaining <= 0.0 {
                break;
            }
        }

        let lam = self.approx_fermi(eigenvalues, num_particles, degs);
        (v2, lam)
    }

    fn approx_fermi(&self, eigenvalues: &[f64], num_particles: usize, degs: &[f64]) -> f64 {
        let mut idx: Vec<usize> = (0..self.n_states).collect();
        idx.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
        let mut count = 0.0;
        for &i in &idx {
            count += degs[i];
            if count >= num_particles as f64 {
                return eigenvalues[i];
            }
        }
        eigenvalues[*idx.last().unwrap_or(&0)]
    }

    // ─── T_eff kinetic matrix ────────────────────────────────────────

    fn build_t_eff(&self, rho_p: &[f64], rho_n: &[f64], is_proton: bool, params: &[f64]) -> Mat {
        let (t1, t2) = (params[1], params[2]);
        let (x1, x2) = (params[5], params[6]);

        let c0t = 0.25 * (t1 * (1.0 + x1 / 2.0) + t2 * (1.0 + x2 / 2.0));
        let c1n = 0.25 * (t1 * (0.5 + x1) - t2 * (0.5 + x2));

        let rho_q: &[f64] = if is_proton { rho_p } else { rho_n };
        let f_q: Vec<f64> = (0..self.nr)
            .map(|k| {
                let fval = HBAR2_2M + c0t * (rho_p[k] + rho_n[k]) - c1n * rho_q[k];
                fval.max(HBAR2_2M * 0.3)
            })
            .collect();

        let mut t_eff = Mat::zeros(self.n_states);

        for ((_l, _j_key), indices) in &self.lj_blocks {
            let l_val = self.states[indices[0]].l;
            let ll1 = (l_val * (l_val + 1)) as f64;

            for ii in 0..indices.len() {
                let idx_i = indices[ii];
                for jj in ii..indices.len() {
                    let idx_j = indices[jj];

                    let integrand: Vec<f64> = (0..self.nr)
                        .map(|k| {
                            f_q[k]
                                * (self.dwf[idx_i][k] * self.dwf[idx_j][k] * self.r[k].powi(2)
                                    + ll1 * self.wf[idx_i][k] * self.wf[idx_j][k])
                        })
                        .collect();

                    let val = trapz(&integrand, &self.r).unwrap_or(0.0);
                    t_eff.set(idx_i, idx_j, val);
                    t_eff.set(idx_j, idx_i, val);
                }
            }
        }

        t_eff
    }

    // ─── Total energy functional ─────────────────────────────────────

    fn compute_energy(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        results_p: &SpeciesResult,
        results_n: &SpeciesResult,
        params: &[f64],
        verbose: bool,
    ) -> f64 {
        let (t0, t3) = (params[0], params[3]);
        let (x0, x3) = (params[4], params[7]);
        let alpha = params[8];
        let n = self.n_states;

        let degs: Vec<f64> = self.states.iter().map(|s| s.deg as f64).collect();
        let rho: Vec<f64> = (0..self.nr).map(|k| rho_p[k] + rho_n[k]).collect();

        let mut e_kin = 0.0;
        for (is_proton, res) in [(true, results_p), (false, results_n)] {
            let t_eff = self.build_t_eff(rho_p, rho_n, is_proton, params);
            for i in 0..n {
                if degs[i] * res.v2[i] < BCS_DENSITY_SKIP {
                    continue;
                }
                let mut val = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        val += res.eigvec(a, i) * t_eff.get(a, b) * res.eigvec(b, i);
                    }
                }
                e_kin += degs[i] * res.v2[i] * val;
            }
        }

        let sum_rho2: Vec<f64> = (0..self.nr)
            .map(|k| rho_p[k].powi(2) + rho_n[k].powi(2))
            .collect();

        let integ_t0: Vec<f64> = (0..self.nr)
            .map(|k| {
                ((1.0 + x0 / 2.0) * rho[k].powi(2) - (0.5 + x0) * sum_rho2[k])
                    * 4.0
                    * PI
                    * self.r[k].powi(2)
            })
            .collect();
        let e_t0 = (t0 / 2.0) * trapz(&integ_t0, &self.r).unwrap_or(0.0);

        let integ_t3: Vec<f64> = (0..self.nr)
            .map(|k| {
                let rho_safe = rho[k].max(RHO_POWF_GUARD);
                rho_safe.powf(alpha)
                    * ((1.0 + x3 / 2.0) * rho[k].powi(2) - (0.5 + x3) * sum_rho2[k])
                    * 4.0
                    * PI
                    * self.r[k].powi(2)
            })
            .collect();
        let e_t3 = (t3 / 12.0) * trapz(&integ_t3, &self.r).unwrap_or(0.0);

        let v_c = self.coulomb_direct(rho_p);
        let integ_c_direct: Vec<f64> = (0..self.nr)
            .map(|k| v_c[k] * rho_p[k] * 4.0 * PI * self.r[k].powi(2))
            .collect();
        let e_coul_direct = 0.5 * trapz(&integ_c_direct, &self.r).unwrap_or(0.0);

        let v_cx = self.coulomb_exchange(rho_p);
        let integ_c_exch: Vec<f64> = (0..self.nr)
            .map(|k| v_cx[k] * rho_p[k] * 4.0 * PI * self.r[k].powi(2))
            .collect();
        let e_coul_exchange = trapz(&integ_c_exch, &self.r).unwrap_or(0.0);

        let mut e_pair = 0.0;
        for (delta_q, res) in [(self.delta_p, results_p), (self.delta_n, results_n)] {
            for i in 0..n {
                let v2 = res.v2[i];
                let u2 = 1.0 - v2;
                e_pair -= delta_q * degs[i] * (v2 * u2).max(0.0).sqrt();
            }
        }

        let e_cm = -0.75 * self.hw;
        let e_total = e_kin + e_t0 + e_t3 + e_coul_direct + e_coul_exchange + e_pair + e_cm;

        if verbose {
            let integ_np: Vec<f64> = (0..self.nr)
                .map(|k| rho_p[k] * 4.0 * PI * self.r[k].powi(2))
                .collect();
            let integ_nn: Vec<f64> = (0..self.nr)
                .map(|k| rho_n[k] * 4.0 * PI * self.r[k].powi(2))
                .collect();
            let n_p = trapz(&integ_np, &self.r).unwrap_or(0.0);
            let n_n = trapz(&integ_nn, &self.r).unwrap_or(0.0);
            println!("  Energy components:");
            println!("    E_kin    = {e_kin:>10.2} MeV");
            println!("    E_t0    = {e_t0:>10.2} MeV");
            println!("    E_t3    = {e_t3:>10.2} MeV");
            println!("    E_Cdir  = {e_coul_direct:>10.2} MeV");
            println!("    E_Cexch = {e_coul_exchange:>10.2} MeV");
            println!("    E_pair  = {e_pair:>10.2} MeV");
            println!("    E_cm    = {e_cm:>10.2} MeV");
            println!("    E_total = {e_total:>10.2} MeV");
            println!(
                "    N_p = {:.2}, N_n = {:.2} (target: Z={}, N={})",
                n_p, n_n, self.z, self.n_neutrons
            );
        }

        e_total
    }

    // ─── Main solver ─────────────────────────────────────────────────

    pub fn solve(&self, params: &[f64], max_iter: usize, tol: f64, mixing: f64) -> HFBResult {
        self.solve_inner(params, max_iter, tol, mixing, false)
    }

    pub fn solve_verbose(
        &self,
        params: &[f64],
        max_iter: usize,
        tol: f64,
        mixing: f64,
    ) -> HFBResult {
        self.solve_inner(params, max_iter, tol, mixing, true)
    }

    fn solve_inner(
        &self,
        params: &[f64],
        max_iter: usize,
        tol: f64,
        mixing: f64,
        verbose: bool,
    ) -> HFBResult {
        let w0 = params[9];
        let nr = self.nr;
        let ns = self.n_states;

        let r_nuc = 1.2 * (self.a as f64).powf(1.0 / 3.0);
        let rho0 = 3.0 * self.a as f64 / (4.0 * PI * r_nuc.powi(3));

        let mut rho_p: Vec<f64> = self
            .r
            .iter()
            .map(|&ri| {
                if ri < r_nuc {
                    (rho0 * self.z as f64 / self.a as f64).max(DENSITY_FLOOR)
                } else {
                    DENSITY_FLOOR
                }
            })
            .collect();
        let mut rho_n: Vec<f64> = self
            .r
            .iter()
            .map(|&ri| {
                if ri < r_nuc {
                    (rho0 * self.n_neutrons as f64 / self.a as f64).max(DENSITY_FLOOR)
                } else {
                    DENSITY_FLOOR
                }
            })
            .collect();

        let mut e_prev = 1e10_f64;
        let mut converged = false;
        let mut last_de = 0.0;
        let mut last_iter = 0;
        let mut results_p = SpeciesResult::empty(ns);
        let mut results_n = SpeciesResult::empty(ns);

        for it in 0..max_iter {
            let mut rho_p_new = vec![DENSITY_FLOOR; nr];
            let mut rho_n_new = vec![DENSITY_FLOOR; nr];

            for is_proton in [true, false] {
                let num_q = if is_proton { self.z } else { self.n_neutrons };
                let delta_q = if is_proton {
                    self.delta_p
                } else {
                    self.delta_n
                };

                let u_sky = self.skyrme_potential(&rho_p, &rho_n, is_proton, params);

                let u_total: Vec<f64> = if is_proton {
                    let v_c = self.coulomb_direct(&rho_p);
                    let v_cx = self.coulomb_exchange(&rho_p);
                    (0..nr).map(|k| u_sky[k] + v_c[k] + v_cx[k]).collect()
                } else {
                    u_sky
                };

                let t_eff = self.build_t_eff(&rho_p, &rho_n, is_proton, params);

                // Build Hamiltonian: H = T_eff + V
                let mut h = Mat::zeros(ns);
                // Copy T_eff into H
                for i in 0..ns {
                    for j in 0..ns {
                        h.set(i, j, t_eff.get(i, j));
                    }
                }

                // Add potential matrix elements
                for i in 0..ns {
                    let integ: Vec<f64> = (0..nr)
                        .map(|k| self.wf[i][k].powi(2) * u_total[k] * self.r[k].powi(2))
                        .collect();
                    h.add(i, i, trapz(&integ, &self.r).unwrap_or(0.0));

                    // Spin-orbit — uses barracuda::ops::grid::compute_ls_factor
                    if w0 != 0.0 && self.states[i].l > 0 {
                        let ls = compute_ls_factor(self.states[i].l as u32, self.states[i].j);
                        let rho_total: Vec<f64> = (0..nr).map(|k| rho_p[k] + rho_n[k]).collect();
                        let drho = gradient_1d(&rho_total, self.dr);
                        let so_integ: Vec<f64> = (0..nr)
                            .map(|k| {
                                self.wf[i][k].powi(2) * drho[k] / self.r[k].max(SPIN_ORBIT_R_MIN)
                                    * self.r[k].powi(2)
                            })
                            .collect();
                        h.add(i, i, w0 * ls * trapz(&so_integ, &self.r).unwrap_or(0.0));
                    }
                }

                // Off-diagonal: same (l, j) block
                for ((_l, _j_key), indices) in &self.lj_blocks {
                    for ii in 0..indices.len() {
                        for jj in (ii + 1)..indices.len() {
                            let idx_i = indices[ii];
                            let idx_j = indices[jj];
                            let integ: Vec<f64> = (0..nr)
                                .map(|k| {
                                    self.wf[idx_i][k]
                                        * self.wf[idx_j][k]
                                        * u_total[k]
                                        * self.r[k].powi(2)
                                })
                                .collect();
                            let val = trapz(&integ, &self.r).unwrap_or(0.0);
                            h.add(idx_i, idx_j, val);
                            h.add(idx_j, idx_i, val);
                        }
                    }
                }

                // ── Diagonalize using barracuda::linalg::eigh_f64 ──
                let eig = eigh_f64(&h.data, ns).expect("eigh_f64 failed");
                let eigenvalues = eig.eigenvalues;
                // eigenvectors in row-major flat format: V[row * ns + col]

                let (v2, lam) = self.bcs_occupations(&eigenvalues, num_q, delta_q);

                // New density from BCS-weighted eigenstates
                let degs: Vec<f64> = self.states.iter().map(|s| s.deg as f64).collect();
                let mut rho_q_new = vec![DENSITY_FLOOR; nr];

                for i in 0..ns {
                    if degs[i] * v2[i] < BCS_DENSITY_SKIP {
                        continue;
                    }
                    // phi_i(r) = sum_k c_{ki} * R_k(r)
                    // eig.eigenvectors is row-major: V[row * ns + col] = V_{row, col}
                    let mut phi = vec![0.0; nr];
                    for j in 0..ns {
                        let c = eig.eigenvectors[j * ns + i];
                        for k in 0..nr {
                            phi[k] += c * self.wf[j][k];
                        }
                    }
                    for k in 0..nr {
                        rho_q_new[k] += degs[i] * v2[i] * phi[k].powi(2) / (4.0 * PI);
                    }
                }

                for k in 0..nr {
                    rho_q_new[k] = rho_q_new[k].max(DENSITY_FLOOR);
                }

                if is_proton {
                    rho_p_new = rho_q_new;
                    results_p = SpeciesResult {
                        _eigenvalues: eigenvalues,
                        eigvecs: eig.eigenvectors,
                        n: ns,
                        v2,
                        _lambda: lam,
                    };
                } else {
                    rho_n_new = rho_q_new;
                    results_n = SpeciesResult {
                        _eigenvalues: eigenvalues,
                        eigvecs: eig.eigenvectors,
                        n: ns,
                        v2,
                        _lambda: lam,
                    };
                }
            }

            for k in 0..nr {
                rho_p[k] = (mixing * rho_p_new[k] + (1.0 - mixing) * rho_p[k]).max(DENSITY_FLOOR);
                rho_n[k] = (mixing * rho_n_new[k] + (1.0 - mixing) * rho_n[k]).max(DENSITY_FLOOR);
            }

            let e_total =
                self.compute_energy(&rho_p, &rho_n, &results_p, &results_n, params, false);

            last_de = (e_total - e_prev).abs();
            last_iter = it + 1;

            if last_de < tol && it > 5 {
                converged = true;
                e_prev = e_total;
                break;
            }
            e_prev = e_total;
        }

        if verbose {
            self.compute_energy(&rho_p, &rho_n, &results_p, &results_n, params, true);
        }

        let binding = if e_prev < 0.0 { -e_prev } else { e_prev.abs() };

        HFBResult {
            binding_energy_mev: binding,
            converged,
            iterations: last_iter,
            delta_e: last_de,
        }
    }
}

/// Per-species diagonalization results (flat storage, no nalgebra).
///
/// Fields use `_` prefix because they are stored for compute_energy access
/// but not individually read from outside. The struct is constructed and
/// passed as a whole to `compute_energy`.
#[allow(dead_code)] // fields read only within struct methods and compute_energy
struct SpeciesResult {
    _eigenvalues: Vec<f64>,
    eigvecs: Vec<f64>, // row-major n×n
    n: usize,
    v2: Vec<f64>,
    _lambda: f64,
}

impl SpeciesResult {
    fn empty(n_states: usize) -> Self {
        SpeciesResult {
            _eigenvalues: vec![0.0; n_states],
            eigvecs: vec![0.0; n_states * n_states],
            n: n_states,
            v2: vec![0.0; n_states],
            _lambda: 0.0,
        }
    }

    /// Access eigenvector element V[row, col] (row-major)
    #[inline]
    fn eigvec(&self, row: usize, col: usize) -> f64 {
        self.eigvecs[row * self.n + col]
    }
}

/// Hybrid binding energy: HFB for medium nuclei, SEMF otherwise
pub fn binding_energy_l2(z: usize, n: usize, params: &[f64]) -> (f64, bool) {
    let a = z + n;
    if (56..=132).contains(&a) {
        let hfb = SphericalHFB::new_adaptive(z, n);
        let result = hfb.solve(params, 200, 0.05, 0.3);
        (result.binding_energy_mev, result.converged)
    } else {
        (semf_binding_energy(z, n, params), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use barracuda::numerical::trapz;
    use std::f64::consts::PI;

    fn sly4_params() -> Vec<f64> {
        crate::provenance::SLY4_PARAMS.to_vec()
    }

    #[test]
    fn radial_grid_generation() {
        let hfb = SphericalHFB::new(28, 28, 8, 15.0, 120);
        let r = hfb.r_grid();
        let dr = hfb.dr();

        assert_eq!(r.len(), 120);
        assert!((dr - 15.0 / 120.0).abs() < 1e-10);
        assert!((r[0] - dr).abs() < 1e-10);
        assert!((r[r.len() - 1] - 15.0).abs() < 1e-10);

        for i in 1..r.len() {
            let spacing = r[i] - r[i - 1];
            assert!(
                (spacing - dr).abs() < 1e-8,
                "grid should be uniform: r[{}] - r[{}] = {}",
                i,
                i - 1,
                spacing
            );
        }
    }

    #[test]
    fn harmonic_oscillator_wavefunction_normalization() {
        let hfb = SphericalHFB::new(28, 28, 8, 15.0, 100);
        let r = hfb.r_grid();

        for (i, wf) in hfb.wf_flat().chunks(hfb.nr()).enumerate().take(3) {
            let integrand: Vec<f64> = r
                .iter()
                .zip(wf.iter())
                .map(|(&ri, &wi)| wi.powi(2) * ri.powi(2))
                .collect();
            let norm_sq = trapz(&integrand, r).unwrap_or(0.0);
            assert!(
                (norm_sq - 1.0).abs() < 0.05,
                "state {i}: ∫ R² r² dr = {norm_sq} (expect ~1)"
            );
        }
    }

    #[test]
    fn density_from_eigenstates_single_state() {
        let hfb = SphericalHFB::new(8, 8, 6, 12.0, 60);
        let ns = hfb.n_states();
        let r = hfb.r_grid();

        let mut eigvecs = vec![0.0; ns * ns];
        for i in 0..ns {
            eigvecs[i * ns + i] = 1.0;
        }
        let mut v2 = vec![0.0; ns];
        v2[0] = 1.0;

        let rho = hfb.density_from_eigenstates(&eigvecs, &v2, ns);
        assert_eq!(rho.len(), 60);
        assert!(rho.iter().all(|&x| x >= 1e-15));
        let deg0 = hfb.deg_values()[0];
        let integrand: Vec<f64> = r
            .iter()
            .zip(rho.iter())
            .map(|(&ri, &rhi)| rhi * 4.0 * PI * ri.powi(2))
            .collect();
        let n_total: f64 = trapz(&integrand, r).unwrap_or(0.0);
        assert!(
            (n_total - deg0).abs() < 0.5,
            "single occupied state: integral ~{n_total} (deg={deg0})"
        );
    }

    #[test]
    fn binding_energy_l2_semf_light_nucleus() {
        let params = sly4_params();
        let (b, conv) = binding_energy_l2(8, 8, &params);
        assert!(conv, "SEMF path should always converge");
        assert!(b > 0.0 && b < 200.0, "O-16 binding ~{b} MeV");
    }

    #[test]
    #[ignore = "HFB solve takes > 1s"]
    fn hfb_full_solve_ni56() {
        let params = sly4_params();
        let hfb = SphericalHFB::new_adaptive(28, 28);
        let result = hfb.solve(&params, 200, 0.05, 0.3);
        assert!(result.binding_energy_mev > 400.0);
        assert!(result.converged);
    }

    #[test]
    fn adaptive_constructor_scales_with_mass() {
        let light = SphericalHFB::new_adaptive(8, 8);
        let medium = SphericalHFB::new_adaptive(28, 28);
        let heavy = SphericalHFB::new_adaptive(50, 82);

        assert!(
            light.n_states() <= medium.n_states(),
            "O-16 should have fewer states than Ni-56"
        );
        assert!(
            medium.n_states() <= heavy.n_states(),
            "Ni-56 should have fewer states than Sn-132"
        );
        assert!(
            light.nr() <= medium.nr(),
            "lighter nucleus needs fewer grid points"
        );
    }

    #[test]
    fn build_hamiltonian_is_symmetric() {
        let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
        let ns = hfb.n_states();
        let nr = hfb.nr();
        let rho = vec![0.01; nr];
        let params = sly4_params();

        let w0 = params[9];
        let h = hfb.build_hamiltonian(&rho, &rho, true, &params, w0);
        assert_eq!(h.len(), ns * ns);

        for i in 0..ns {
            for j in i + 1..ns {
                let diff = (h[i * ns + j] - h[j * ns + i]).abs();
                assert!(
                    diff < 1e-10,
                    "H[{i},{j}]={} != H[{j},{i}]={}",
                    h[i * ns + j],
                    h[j * ns + i]
                );
            }
        }
    }

    #[test]
    fn bcs_occupations_sum_constraint() {
        let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
        let ns = hfb.n_states();
        let eigs: Vec<f64> = (0..ns).map(|i| -20.0 + 5.0 * i as f64).collect();

        let (v2, _lambda) = hfb.bcs_occupations(&eigs, 8, 12.0 / (8.0_f64 + 8.0).sqrt());
        let degs = hfb.deg_values();
        let n_total: f64 = degs.iter().zip(v2.iter()).map(|(d, v)| d * v).sum();
        assert!(
            (n_total - 8.0).abs() < 1.0,
            "BCS particle number = {n_total}, expected ~8"
        );
    }

    #[test]
    fn quantum_numbers_have_correct_degeneracy() {
        let hfb = SphericalHFB::new(20, 20, 6, 12.0, 60);
        let degs = hfb.deg_values();
        let lj = hfb.lj_quantum_numbers();

        assert_eq!(degs.len(), lj.len());
        for (i, &(l, j)) in lj.iter().enumerate() {
            let expected_deg = 2.0 * j + 1.0;
            assert!(
                (degs[i] - expected_deg).abs() < 1e-10,
                "state {i}: l={l}, j={j}, deg={}, expected {}",
                degs[i],
                expected_deg
            );
        }
    }

    #[test]
    fn wavefunction_accessor_consistency() {
        let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
        let ns = hfb.n_states();
        let nr = hfb.nr();
        let flat = hfb.wf_flat();

        assert_eq!(flat.len(), ns * nr);
        for i in 0..ns {
            let state = hfb.wf_state(i);
            assert_eq!(state.len(), nr);
            for k in 0..nr {
                assert_eq!(
                    flat[i * nr + k],
                    state[k],
                    "flat[{i}*{nr}+{k}] != state[{i}][{k}]"
                );
            }
        }
    }

    #[test]
    fn lj_same_flag_consistency() {
        let hfb = SphericalHFB::new(20, 20, 6, 12.0, 60);
        let ns = hfb.n_states();
        let lj = hfb.lj_quantum_numbers();
        let lj_same = hfb.lj_same_flat();

        assert_eq!(lj_same.len(), ns * ns);
        for i in 0..ns {
            for j in 0..ns {
                let expected = u32::from(lj[i] == lj[j]);
                assert_eq!(lj_same[i * ns + j], expected, "lj_same[{i},{j}] mismatch");
            }
        }
    }

    #[test]
    fn binding_energy_l2_determinism() {
        let params = crate::provenance::SLY4_PARAMS;
        let run = || binding_energy_l2(28, 28, &params);
        let a = run();
        let b = run();
        assert_eq!(
            a.0.to_bits(),
            b.0.to_bits(),
            "L2 binding energy not deterministic"
        );
    }

    #[test]
    fn energy_from_densities_finite_for_nucleus() {
        let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
        let ns = hfb.n_states();
        let nr = hfb.nr();
        let rho = vec![0.01; nr];
        let mut evecs = vec![0.0; ns * ns];
        for i in 0..ns {
            evecs[i * ns + i] = 1.0;
        }
        let evals: Vec<f64> = (0..ns).map(|i| -20.0 + i as f64 * 2.0).collect();
        let params = sly4_params();
        let e =
            hfb.compute_energy_from_densities(&rho, &rho, &evals, &evecs, &evals, &evecs, &params);
        assert!(e.is_finite(), "energy must be finite, got {e}");
    }

    #[test]
    fn energy_with_v2_finite() {
        let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
        let ns = hfb.n_states();
        let nr = hfb.nr();
        let rho = vec![0.01; nr];
        let mut evecs = vec![0.0; ns * ns];
        for i in 0..ns {
            evecs[i * ns + i] = 1.0;
        }
        let evals: Vec<f64> = (0..ns).map(|i| -20.0 + i as f64 * 2.0).collect();
        let v2 = vec![0.5; ns];
        let params = sly4_params();
        let e = hfb.compute_energy_with_v2(
            &rho, &rho, &evals, &evecs, &evals, &evecs, &v2, &v2, &params,
        );
        assert!(e.is_finite(), "energy_with_v2 must be finite, got {e}");
    }

    #[test]
    fn pairing_gap_scales_with_mass() {
        let light = SphericalHFB::new_adaptive(8, 8);
        let heavy = SphericalHFB::new_adaptive(50, 82);
        assert!(
            light.pairing_gap() > heavy.pairing_gap(),
            "pairing gap should decrease with mass: light={}, heavy={}",
            light.pairing_gap(),
            heavy.pairing_gap()
        );
    }

    #[test]
    fn hw_oscillator_frequency_positive() {
        let hfb = SphericalHFB::new(20, 20, 6, 12.0, 60);
        assert!(hfb.hw() > 0.0, "oscillator frequency must be positive");
        assert!(hfb.hw() < 50.0, "oscillator frequency unreasonably large");
    }

    #[test]
    fn density_floor_enforced() {
        let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
        let ns = hfb.n_states();
        let evecs = vec![0.0; ns * ns];
        let v2 = vec![0.0; ns];
        let rho = hfb.density_from_eigenstates(&evecs, &v2, ns);
        assert!(
            rho.iter().all(|&x| x >= crate::tolerances::DENSITY_FLOOR),
            "density must be >= DENSITY_FLOOR everywhere"
        );
    }

    #[test]
    fn wf_flat_and_dwf_flat_dimensions() {
        let hfb = SphericalHFB::new(8, 8, 4, 10.0, 50);
        let ns = hfb.n_states();
        let nr = hfb.nr();
        assert_eq!(hfb.wf_flat().len(), ns * nr);
        assert_eq!(hfb.dwf_flat().len(), ns * nr);
    }
}
