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
//!   - BCS pairing: constant gap Δ = 12/√A `MeV` (Ring & Schuck §6.2)
//!   - Coulomb: Poisson direct + Slater exchange (Slater, Phys. Rev. 81, 385)
//!   - Isospin structure in Skyrme potential (t₀/x₀, t₃/x₃)
//!   - Effective kinetic matrix `T_eff` (t₁/t₂ effective mass terms)
//!   - Spin-orbit splitting (W₀ parameter)
//!   - Center-of-mass correction: `E_CM` = -3/4 `ℏω` (Bohr & Mottelson §4-2)
//!
//! Uses (all `BarraCUDA` native — zero external dependencies):
//!   - `barracuda::special::{gamma, laguerre}` for HO basis wavefunctions
//!   - `barracuda::numerical::{trapz, gradient_1d}` for radial integrals & derivatives
//!   - `barracuda::optimize::brent` for BCS chemical potential (matches scipy.optimize.brentq)
//!   - `barracuda::linalg::eigh_f64` for symmetric eigenvalue decomposition
//!
//! # Module structure
//!
//! - `mod.rs` — types, basis construction, BCS, solver, energy functional
//! - `potentials.rs` — Coulomb, Skyrme, `T_eff`, Hamiltonian matrix assembly

mod potentials;
#[cfg(test)]
mod tests;

use super::constants::{HBAR_C, M_NUCLEON};
use super::hfb_common::Mat;
use super::semf::semf_binding_energy;
use barracuda::linalg::eigh_f64;
use barracuda::numerical::{gradient_1d, trapz};
use barracuda::special::{gamma, laguerre};
use std::collections::HashMap;

use crate::error::HotSpringError;
use crate::tolerances::{
    BCS_DENSITY_SKIP, DENSITY_FLOOR, FERMI_SEARCH_MARGIN, HFB_L2_MIXING, HFB_L2_TOLERANCE,
    HFB_MAX_ITER, SHARP_FILLING_THRESHOLD,
};
use std::f64::consts::PI;

/// Basis state quantum numbers
#[derive(Debug, Clone)]
pub(super) struct BasisState {
    pub n: usize,
    pub l: usize,
    pub j: f64,
    pub deg: usize, // 2j+1
}

/// Spherical HF+BCS solver
pub struct SphericalHFB {
    pub(super) z: usize,
    pub(super) n_neutrons: usize,
    pub(super) r: Vec<f64>,
    pub(super) dr: f64,
    pub(super) nr: usize,
    pub(super) hw: f64,      // hbar*omega (MeV)
    pub(super) b: f64,       // HO length parameter (fm)
    pub(super) delta_p: f64, // proton pairing gap (MeV)
    pub(super) delta_n: f64, // neutron pairing gap (MeV)
    pub(super) states: Vec<BasisState>,
    pub(super) n_states: usize,
    pub(super) wf: Vec<Vec<f64>>, // [n_states][nr] — radial wavefunctions
    pub(super) dwf: Vec<Vec<f64>>, // [n_states][nr] — wavefunction derivatives
    pub(super) lj_blocks: HashMap<(usize, u64), Vec<usize>>,
}

/// Result from HFB solve
#[derive(Debug, Clone)]
pub struct HFBResult {
    pub binding_energy_mev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub delta_e: f64,
}

// ═══════════════════════════════════════════════════════════════════
// Construction and accessors
// ═══════════════════════════════════════════════════════════════════

impl SphericalHFB {
    #[must_use]
    pub fn new(z: usize, n: usize, n_shells: usize, r_max: f64, n_grid: usize) -> Self {
        Self::build(z, n, n_shells, r_max, n_grid)
    }

    /// Adaptive parameters matching the Python reference solver defaults
    #[must_use]
    pub fn new_adaptive(z: usize, n: usize) -> Self {
        let a = z + n;
        let a_f = a as f64;
        let n_shells = (2.0 * a_f.cbrt()) as usize + 4;
        let n_shells = n_shells.clamp(8, 14);
        let r_max = 1.2f64.mul_add(a_f.cbrt(), 8.0_f64).max(12.0);
        let n_grid = ((r_max * 12.0) as usize).max(120);
        Self::build(z, n, n_shells, r_max, n_grid)
    }

    #[must_use]
    pub const fn n_states(&self) -> usize {
        self.n_states
    }
    #[must_use]
    pub const fn nr(&self) -> usize {
        self.nr
    }
    #[must_use]
    pub const fn z(&self) -> usize {
        self.z
    }
    #[must_use]
    pub const fn n_neutrons(&self) -> usize {
        self.n_neutrons
    }
    #[must_use]
    pub const fn dr(&self) -> f64 {
        self.dr
    }
    #[must_use]
    pub const fn hw(&self) -> f64 {
        self.hw
    }

    /// Pairing gap (same for proton and neutron in this model)
    #[must_use]
    pub const fn pairing_gap(&self) -> f64 {
        self.delta_p
    }

    /// Flat wavefunctions: `[n_states × nr]` row-major
    #[must_use]
    pub fn wf_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_states * self.nr);
        for s in &self.wf {
            out.extend_from_slice(s);
        }
        out
    }

    /// Flat wavefunction derivatives: `[n_states × nr]` row-major
    #[must_use]
    pub fn dwf_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_states * self.nr);
        for s in &self.dwf {
            out.extend_from_slice(s);
        }
        out
    }

    /// Radial grid points
    #[must_use]
    pub fn r_grid(&self) -> &[f64] {
        &self.r
    }

    /// `lj_same` matrix: `[n_states × n_states]` u32 (1 if same (l,j) block)
    #[must_use]
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
    #[must_use]
    pub fn ll1_values(&self) -> Vec<f64> {
        self.states
            .iter()
            .map(|s| (s.l * (s.l + 1)) as f64)
            .collect()
    }

    /// (l, j) quantum numbers per state — needed for spin-orbit coupling
    #[must_use]
    pub fn lj_quantum_numbers(&self) -> Vec<(usize, f64)> {
        self.states.iter().map(|s| (s.l, s.j)).collect()
    }

    /// Per-state wavefunction access (for spin-orbit integrals)
    #[must_use]
    pub fn wf_state(&self, i: usize) -> &[f64] {
        &self.wf[i]
    }

    /// Degeneracies (2j+1) per state
    #[must_use]
    pub fn deg_values(&self) -> Vec<f64> {
        self.states.iter().map(|s| s.deg as f64).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Basis construction and wavefunctions
// ═══════════════════════════════════════════════════════════════════

impl SphericalHFB {
    fn build(z: usize, n: usize, n_shells: usize, r_max: f64, n_grid: usize) -> Self {
        let a = z + n;
        let hw = 41.0 * (a as f64).powf(-1.0 / 3.0);
        let b = HBAR_C / (M_NUCLEON * hw).sqrt();
        let dr = r_max / n_grid as f64;
        let r: Vec<f64> = (1..=n_grid).map(|i| i as f64 * dr).collect();
        let delta = 12.0 / (a.max(4) as f64).sqrt();

        let mut hfb = Self {
            z,
            n_neutrons: n,
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

    #[allow(clippy::cast_possible_truncation)] // lj_blocks key: j < 50, (j*1000) fits u64
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
                            j: f64::from(j2) / 2.0,
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
        self.dwf = self
            .wf
            .iter()
            .map(|wf_i| gradient_1d(wf_i, self.dr))
            .collect();
    }

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
}

// ═══════════════════════════════════════════════════════════════════
// Hamiltonian, BCS, density, and energy
// ═══════════════════════════════════════════════════════════════════

impl SphericalHFB {
    /// Build the full Hamiltonian matrix for one species (proton or neutron).
    ///
    /// Returns a flat row-major `n_states × n_states` matrix suitable for
    /// packing into `BatchedEighGpu`.
    #[must_use]
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
            let v_cx = Self::coulomb_exchange(rho_p);
            (0..nr).map(|k| u_sky[k] + v_c[k] + v_cx[k]).collect()
        } else {
            u_sky
        };

        let t_eff = self.build_t_eff(rho_p, rho_n, is_proton, params);

        let mut h = Mat::zeros(ns);
        for i in 0..ns {
            for j in 0..ns {
                h.set(i, j, t_eff.get(i, j));
            }
        }

        self.add_potential_to_hamiltonian(&mut h, &u_total, rho_p, rho_n, w0);
        h.data
    }

    /// Compute BCS occupations from externally-provided eigenvalues.
    #[must_use]
    pub fn bcs_occupations_from_eigs(
        &self,
        eigenvalues: &[f64],
        num_particles: usize,
        delta: f64,
    ) -> (Vec<f64>, f64) {
        self.bcs_occupations(eigenvalues, num_particles, delta)
    }

    /// Compute density from BCS-weighted eigenstates (GPU-unpacked format).
    #[must_use]
    pub fn density_from_eigenstates(&self, eigvecs: &[f64], v2: &[f64], ns: usize) -> Vec<f64> {
        let nr = self.nr;
        let degs: Vec<f64> = self.states.iter().map(|s| s.deg as f64).collect();
        let mut rho = vec![DENSITY_FLOOR; nr];

        for (i, (&d, &v2_i)) in degs.iter().zip(v2).enumerate().take(ns) {
            if d * v2_i < BCS_DENSITY_SKIP {
                continue;
            }
            let mut phi = vec![0.0; nr];
            for j in 0..ns {
                let c = eigvecs[j * ns + i];
                for (ph, w) in phi.iter_mut().zip(self.wf[j].iter()) {
                    *ph += c * w;
                }
            }
            let fact = d * v2_i / (4.0 * PI);
            for (r, &p) in rho.iter_mut().zip(phi.iter()) {
                *r += fact * p.powi(2);
            }
        }

        for r in &mut rho {
            *r = (*r).max(DENSITY_FLOOR);
        }
        rho
    }

    /// Compute total energy from proton/neutron densities and eigendecompositions.
    #[must_use]
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
        let (v2_p, _) = self.bcs_occupations(eigs_p, self.z, self.delta_p);
        let (v2_n, _) = self.bcs_occupations(eigs_n, self.n_neutrons, self.delta_n);

        let results_p = SpeciesResult::new(eigs_p.to_vec(), vecs_p.to_vec(), ns, v2_p, 0.0);
        let results_n = SpeciesResult::new(eigs_n.to_vec(), vecs_n.to_vec(), ns, v2_n, 0.0);

        self.compute_energy(rho_p, rho_n, &results_p, &results_n, params, false)
    }

    /// Fast energy calculation that accepts pre-computed v2 occupations.
    #[must_use]
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
        let results_p =
            SpeciesResult::new(eigs_p.to_vec(), vecs_p.to_vec(), ns, v2_p.to_vec(), 0.0);
        let results_n =
            SpeciesResult::new(eigs_n.to_vec(), vecs_n.to_vec(), ns, v2_n.to_vec(), 0.0);
        self.compute_energy(rho_p, rho_n, &results_p, &results_n, params, false)
    }

    pub(super) fn bcs_occupations(
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
            eigenvalues
                .iter()
                .zip(degs.iter())
                .map(|(&ek_raw, &d)| {
                    let ek = ek_raw - lam;
                    let big_ek = ek.hypot(delta);
                    let v2 = 0.5 * (1.0 - ek / big_ek);
                    d * v2
                })
                .sum::<f64>()
                - num_p
        };

        let e_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min) - FERMI_SEARCH_MARGIN;
        let e_max = eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            + FERMI_SEARCH_MARGIN;

        let lam = barracuda::optimize::brent(
            particle_number,
            e_min,
            e_max,
            crate::tolerances::BRENT_TOLERANCE,
            100,
        )
        .map_or_else(
            |_| self.approx_fermi(eigenvalues, num_particles, &degs),
            |result| result.root,
        );

        let v2: Vec<f64> = eigenvalues
            .iter()
            .map(|&eps| {
                let ek = eps - lam;
                let big_ek = ek.hypot(delta);
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
            for (i, (&d, &v2)) in degs.iter().zip(res.v2.iter()).enumerate().take(n) {
                if d * v2 < BCS_DENSITY_SKIP {
                    continue;
                }
                let mut val = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        val += res.eigvec(a, i) * t_eff.get(a, b) * res.eigvec(b, i);
                    }
                }
                e_kin += d * v2 * val;
            }
        }

        let rho_powf_guard = crate::tolerances::RHO_POWF_GUARD;
        let sum_rho2: Vec<f64> = (0..self.nr)
            .map(|k| rho_p[k].mul_add(rho_p[k], rho_n[k].powi(2)))
            .collect();

        let integ_t0: Vec<f64> = (0..self.nr)
            .map(|k| {
                (1.0 + x0 / 2.0).mul_add(rho[k].powi(2), -((0.5 + x0) * sum_rho2[k]))
                    * 4.0
                    * PI
                    * self.r[k].powi(2)
            })
            .collect();
        let e_t0 = (t0 / 2.0) * trapz(&integ_t0, &self.r).unwrap_or(0.0);

        let integ_t3: Vec<f64> = (0..self.nr)
            .map(|k| {
                let rho_safe = rho[k].max(rho_powf_guard);
                rho_safe.powf(alpha)
                    * (1.0 + x3 / 2.0).mul_add(rho[k].powi(2), -((0.5 + x3) * sum_rho2[k]))
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

        let v_cx = Self::coulomb_exchange(rho_p);
        let integ_c_exch: Vec<f64> = (0..self.nr)
            .map(|k| v_cx[k] * rho_p[k] * 4.0 * PI * self.r[k].powi(2))
            .collect();
        let e_coul_exchange = trapz(&integ_c_exch, &self.r).unwrap_or(0.0);

        let mut e_pair = 0.0;
        for (delta_q, res) in [(self.delta_p, results_p), (self.delta_n, results_n)] {
            for (d, &v2) in degs.iter().zip(res.v2.iter()).take(n) {
                let u2 = 1.0 - v2;
                e_pair -= delta_q * d * (v2 * u2).max(0.0).sqrt();
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
}

// ═══════════════════════════════════════════════════════════════════
// SCF solver
// ═══════════════════════════════════════════════════════════════════

impl SphericalHFB {
    /// Solve the HFB equation for the given Skyrme parameters.
    ///
    /// # Errors
    ///
    /// Returns [`HotSpringError::Barracuda`] if the CPU eigensolve fails.
    pub fn solve(
        &self,
        params: &[f64],
        max_iter: usize,
        tol: f64,
        mixing: f64,
    ) -> Result<HFBResult, HotSpringError> {
        self.solve_inner(params, max_iter, tol, mixing, false)
    }

    /// Solve with per-iteration energy printout.
    ///
    /// # Errors
    ///
    /// Returns [`HotSpringError::Barracuda`] if the CPU eigensolve fails.
    pub fn solve_verbose(
        &self,
        params: &[f64],
        max_iter: usize,
        tol: f64,
        mixing: f64,
    ) -> Result<HFBResult, HotSpringError> {
        self.solve_inner(params, max_iter, tol, mixing, true)
    }

    fn solve_inner(
        &self,
        params: &[f64],
        max_iter: usize,
        tol: f64,
        mixing: f64,
        verbose: bool,
    ) -> Result<HFBResult, HotSpringError> {
        let w0 = params[9];
        let nr = self.nr;
        let ns = self.n_states;

        let (rho_p_init, rho_n_init) =
            super::hfb_common::initial_wood_saxon_density(self.z, self.n_neutrons, nr, self.dr);
        let mut rho_p = rho_p_init;
        let mut rho_n = rho_n_init;

        let mut e_prev = 1e10_f64;
        let mut converged = false;
        let mut last_de = 0.0;
        let mut last_iter = 0;
        let mut results_p = SpeciesResult::empty(ns);
        let mut results_n = SpeciesResult::empty(ns);
        let mut alpha = mixing;
        let mut e_prev2 = 1e10_f64;

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
                    let v_cx = Self::coulomb_exchange(&rho_p);
                    (0..nr).map(|k| u_sky[k] + v_c[k] + v_cx[k]).collect()
                } else {
                    u_sky
                };

                let t_eff = self.build_t_eff(&rho_p, &rho_n, is_proton, params);
                let mut h = Mat::zeros(ns);
                for i in 0..ns {
                    for j in 0..ns {
                        h.set(i, j, t_eff.get(i, j));
                    }
                }
                self.add_potential_to_hamiltonian(&mut h, &u_total, &rho_p, &rho_n, w0);

                let eig = eigh_f64(&h.data, ns)?;
                let eigenvalues = eig.eigenvalues;
                let (v2, lam) = self.bcs_occupations(&eigenvalues, num_q, delta_q);

                let degs: Vec<f64> = self.states.iter().map(|s| s.deg as f64).collect();
                let mut rho_q_new = vec![DENSITY_FLOOR; nr];

                for i in 0..ns {
                    if degs[i] * v2[i] < BCS_DENSITY_SKIP {
                        continue;
                    }
                    let mut phi = vec![0.0; nr];
                    for j in 0..ns {
                        let c = eig.eigenvectors[j * ns + i];
                        for (ph, w) in phi.iter_mut().zip(self.wf[j].iter()) {
                            *ph += c * w;
                        }
                    }
                    let fact = degs[i] * v2[i] / (4.0 * PI);
                    for (r, &p) in rho_q_new.iter_mut().zip(phi.iter()) {
                        *r += fact * p.powi(2);
                    }
                }

                for r in &mut rho_q_new {
                    *r = (*r).max(DENSITY_FLOOR);
                }

                if is_proton {
                    rho_p_new = rho_q_new;
                    results_p = SpeciesResult::new(eigenvalues, eig.eigenvectors, ns, v2, lam);
                } else {
                    rho_n_new = rho_q_new;
                    results_n = SpeciesResult::new(eigenvalues, eig.eigenvectors, ns, v2, lam);
                }
            }

            for k in 0..nr {
                rho_p[k] = alpha
                    .mul_add(rho_p_new[k], (1.0 - alpha) * rho_p[k])
                    .max(DENSITY_FLOOR);
                rho_n[k] = alpha
                    .mul_add(rho_n_new[k], (1.0 - alpha) * rho_n[k])
                    .max(DENSITY_FLOOR);
            }

            let e_total =
                self.compute_energy(&rho_p, &rho_n, &results_p, &results_n, params, false);

            last_de = (e_total - e_prev).abs();
            last_iter = it + 1;

            if it > 2 {
                let de_now = e_total - e_prev;
                let de_prev = e_prev - e_prev2;
                if de_now * de_prev < 0.0 {
                    alpha = (alpha * 0.7).max(0.05);
                }
            }
            e_prev2 = e_prev;

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

        Ok(HFBResult {
            binding_energy_mev: binding,
            converged,
            iterations: last_iter,
            delta_e: last_de,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════
// Internal types
// ═══════════════════════════════════════════════════════════════════

struct SpeciesResult {
    #[allow(dead_code)]
    _eigenvalues: Vec<f64>,
    eigvecs: Vec<f64>,
    n: usize,
    v2: Vec<f64>,
    #[allow(dead_code)]
    _lambda: f64,
}

impl SpeciesResult {
    const fn new(
        eigenvalues: Vec<f64>,
        eigvecs: Vec<f64>,
        n: usize,
        v2: Vec<f64>,
        lambda: f64,
    ) -> Self {
        Self {
            _eigenvalues: eigenvalues,
            eigvecs,
            n,
            v2,
            _lambda: lambda,
        }
    }

    fn empty(n_states: usize) -> Self {
        Self {
            _eigenvalues: vec![0.0; n_states],
            eigvecs: vec![0.0; n_states * n_states],
            n: n_states,
            v2: vec![0.0; n_states],
            _lambda: 0.0,
        }
    }

    #[inline]
    fn eigvec(&self, row: usize, col: usize) -> f64 {
        self.eigvecs[row * self.n + col]
    }
}

/// Hybrid binding energy: HFB for medium nuclei, SEMF otherwise.
///
/// # Errors
///
/// Returns [`HotSpringError::Barracuda`] if the HFB eigensolve fails.
pub fn binding_energy_l2(
    z: usize,
    n: usize,
    params: &[f64],
) -> Result<(f64, bool), HotSpringError> {
    let a = z + n;
    if (56..=132).contains(&a) {
        let hfb = SphericalHFB::new_adaptive(z, n);
        let result = hfb.solve(params, HFB_MAX_ITER, HFB_L2_TOLERANCE, HFB_L2_MIXING)?;
        Ok((result.binding_energy_mev, result.converged))
    } else {
        Ok((semf_binding_energy(z, n, params), true))
    }
}
