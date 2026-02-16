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
use super::semf::semf_binding_energy;
use barracuda::linalg::eigh_f64;
use barracuda::numerical::{gradient_1d, trapz};
use barracuda::special::{gamma, laguerre};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Basis state quantum numbers
#[derive(Debug, Clone)]
struct BasisState {
    n: usize,
    l: usize,
    j: f64,
    deg: usize, // 2j+1
}

/// Simple row-major square matrix (replaces nalgebra::DMatrix)
struct Mat {
    data: Vec<f64>,
    n: usize,
}

impl Mat {
    fn zeros(n: usize) -> Self {
        Mat { data: vec![0.0; n * n], n }
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] = v;
    }

    #[inline]
    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] += v;
    }
}

/// Spherical HF+BCS solver
pub struct SphericalHFB {
    z: usize,
    n_neutrons: usize,
    a: usize,
    r: Vec<f64>,
    dr: f64,
    nr: usize,
    hw: f64,           // hbar*omega (MeV)
    b: f64,            // HO length parameter (fm)
    delta_p: f64,      // proton pairing gap (MeV)
    delta_n: f64,      // neutron pairing gap (MeV)
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

    pub fn n_states(&self) -> usize { self.n_states }
    pub fn nr(&self) -> usize { self.nr }
    pub fn z(&self) -> usize { self.z }
    pub fn n_neutrons(&self) -> usize { self.n_neutrons }
    pub fn dr(&self) -> f64 { self.dr }
    pub fn hw(&self) -> f64 { self.hw }

    /// Pairing gap (same for proton and neutron in this model)
    pub fn pairing_gap(&self) -> f64 { self.delta_p }

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
    pub fn r_grid(&self) -> &[f64] { &self.r }

    /// lj_same matrix: [n_states × n_states] u32 (1 if same (l,j) block)
    pub fn lj_same_flat(&self) -> Vec<u32> {
        let ns = self.n_states;
        let mut out = vec![0u32; ns * ns];
        for (_, indices) in &self.lj_blocks {
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
        self.states.iter().map(|s| (s.l * (s.l + 1)) as f64).collect()
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
            h.add(i, i, barracuda::numerical::trapz(&integ, &self.r).unwrap_or(0.0));

            // Spin-orbit
            if w0 != 0.0 && self.states[i].l > 0 {
                let j = self.states[i].j;
                let l_f = self.states[i].l as f64;
                let ls = (j * (j + 1.0) - l_f * (l_f + 1.0) - 0.75) / 2.0;
                let rho_total: Vec<f64> = (0..nr).map(|k| rho_p[k] + rho_n[k]).collect();
                let drho = barracuda::numerical::gradient_1d(&rho_total, self.dr);
                let so_integ: Vec<f64> = (0..nr)
                    .map(|k| {
                        self.wf[i][k].powi(2) * drho[k] / self.r[k].max(0.1)
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
                            self.wf[idx_i][k] * self.wf[idx_j][k] * u_total[k]
                                * self.r[k].powi(2)
                        })
                        .collect();
                    let val =
                        barracuda::numerical::trapz(&integ, &self.r).unwrap_or(0.0);
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
    /// Returns radial density profile [nr].
    pub fn density_from_eigenstates(
        &self,
        eigvecs: &[f64],
        v2: &[f64],
        ns: usize,
    ) -> Vec<f64> {
        let nr = self.nr;
        let degs: Vec<f64> = self.states.iter().map(|s| s.deg as f64).collect();
        let mut rho = vec![1e-15; nr];

        for i in 0..ns {
            if degs[i] * v2[i] < 1e-12 {
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
            rho[k] = rho[k].max(1e-15);
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
            eigenvalues: eigs_p.to_vec(),
            eigvecs: vecs_p.to_vec(),
            n: ns,
            v2: v2_p,
            _lambda: 0.0,
        };
        let results_n = SpeciesResult {
            eigenvalues: eigs_n.to_vec(),
            eigvecs: vecs_n.to_vec(),
            n: ns,
            v2: v2_n,
            _lambda: 0.0,
        };

        self.compute_energy(rho_p, rho_n, &results_p, &results_n, params)
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
        self.wf = self.states.iter()
            .map(|s| Self::ho_radial(s.n, s.l, &self.r, self.b))
            .collect();

        // Compute derivatives: dR/dr via 2nd-order finite differences (matches numpy.gradient)
        self.dwf = self.wf.iter().map(|wf_i| {
            gradient_1d(wf_i, self.dr)
        }).collect();
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
            .map(|k| E2 * (charge_enclosed[k] / self.r[k].max(1e-10) + phi_outer[k]))
            .collect()
    }

    /// Coulomb exchange (Slater approximation)
    fn coulomb_exchange(&self, rho_p: &[f64]) -> Vec<f64> {
        let coeff = -E2 * (3.0 / PI).powf(1.0 / 3.0);
        rho_p.iter().map(|&rp| coeff * rp.max(0.0).powf(1.0 / 3.0)).collect()
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
                let rho_safe = rho.max(1e-20);

                let u_t0 = t0 * ((1.0 + x0 / 2.0) * rho - (0.5 + x0) * rho_q);

                let rho_alpha = rho_safe.powf(alpha);
                let rho_alpha_m1 = if rho > 1e-15 {
                    rho_safe.powf(alpha - 1.0)
                } else {
                    0.0
                };
                let sum_rho2 = rho_p[k].powi(2) + rho_n[k].powi(2);

                let u_t3 = (t3 / 12.0)
                    * ((1.0 + x3 / 2.0) * (alpha + 2.0) * rho_alpha * rho
                        - (0.5 + x3) * (alpha * rho_alpha_m1 * sum_rho2
                            + 2.0 * rho_alpha * rho_q));

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

        if delta < 0.01 {
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

        let e_min = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min) - 50.0;
        let e_max = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 50.0;

        // Brent's method (matches scipy.optimize.brentq precision)
        let lam = match barracuda::optimize::brent(&particle_number, e_min, e_max, 1e-10, 100) {
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
        idx.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

        let mut v2 = vec![0.0; self.n_states];
        let mut remaining = num_particles as f64;
        for &i in &idx {
            let fill = remaining.min(degs[i]);
            v2[i] = fill / degs[i];
            remaining -= fill;
            if remaining <= 0.0 { break; }
        }

        let lam = self.approx_fermi(eigenvalues, num_particles, degs);
        (v2, lam)
    }

    fn approx_fermi(&self, eigenvalues: &[f64], num_particles: usize, degs: &[f64]) -> f64 {
        let mut idx: Vec<usize> = (0..self.n_states).collect();
        idx.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());
        let mut count = 0.0;
        for &i in &idx {
            count += degs[i];
            if count >= num_particles as f64 { return eigenvalues[i]; }
        }
        eigenvalues[*idx.last().unwrap_or(&0)]
    }

    // ─── T_eff kinetic matrix ────────────────────────────────────────

    fn build_t_eff(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        is_proton: bool,
        params: &[f64],
    ) -> Mat {
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
                                * (self.dwf[idx_i][k] * self.dwf[idx_j][k]
                                    * self.r[k].powi(2)
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

    fn compute_energy_debug(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        results_p: &SpeciesResult,
        results_n: &SpeciesResult,
        params: &[f64],
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
                if degs[i] * res.v2[i] < 1e-12 { continue; }
                let mut val = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        val += res.eigvec(a, i) * t_eff.get(a, b) * res.eigvec(b, i);
                    }
                }
                e_kin += degs[i] * res.v2[i] * val;
            }
        }

        let sum_rho2: Vec<f64> = (0..self.nr).map(|k| rho_p[k].powi(2) + rho_n[k].powi(2)).collect();
        let integ_t0: Vec<f64> = (0..self.nr).map(|k| {
            ((1.0 + x0 / 2.0) * rho[k].powi(2) - (0.5 + x0) * sum_rho2[k]) * 4.0 * PI * self.r[k].powi(2)
        }).collect();
        let e_t0 = (t0 / 2.0) * trapz(&integ_t0, &self.r).unwrap_or(0.0);

        let integ_t3: Vec<f64> = (0..self.nr).map(|k| {
            let rho_safe = rho[k].max(1e-20);
            rho_safe.powf(alpha) * ((1.0 + x3 / 2.0) * rho[k].powi(2) - (0.5 + x3) * sum_rho2[k]) * 4.0 * PI * self.r[k].powi(2)
        }).collect();
        let e_t3 = (t3 / 12.0) * trapz(&integ_t3, &self.r).unwrap_or(0.0);

        let v_c = self.coulomb_direct(rho_p);
        let integ_cd: Vec<f64> = (0..self.nr).map(|k| v_c[k] * rho_p[k] * 4.0 * PI * self.r[k].powi(2)).collect();
        let e_coul_direct = 0.5 * trapz(&integ_cd, &self.r).unwrap_or(0.0);
        let v_cx = self.coulomb_exchange(rho_p);
        let integ_cx: Vec<f64> = (0..self.nr).map(|k| v_cx[k] * rho_p[k] * 4.0 * PI * self.r[k].powi(2)).collect();
        let e_coul_exchange = trapz(&integ_cx, &self.r).unwrap_or(0.0);

        let mut e_pair = 0.0;
        for (delta_q, res) in [(self.delta_p, results_p), (self.delta_n, results_n)] {
            for i in 0..n {
                let v2 = res.v2[i];
                let u2 = 1.0 - v2;
                e_pair -= delta_q * degs[i] * (v2 * u2).max(0.0).sqrt();
            }
        }

        let e_cm = -0.75 * self.hw;

        let integ_np: Vec<f64> = (0..self.nr).map(|k| rho_p[k] * 4.0 * PI * self.r[k].powi(2)).collect();
        let integ_nn: Vec<f64> = (0..self.nr).map(|k| rho_n[k] * 4.0 * PI * self.r[k].powi(2)).collect();
        let n_p = trapz(&integ_np, &self.r).unwrap_or(0.0);
        let n_n = trapz(&integ_nn, &self.r).unwrap_or(0.0);

        let e_total = e_kin + e_t0 + e_t3 + e_coul_direct + e_coul_exchange + e_pair + e_cm;
        println!("  Energy components:");
        println!("    E_kin    = {:>10.2} MeV", e_kin);
        println!("    E_t0    = {:>10.2} MeV", e_t0);
        println!("    E_t3    = {:>10.2} MeV", e_t3);
        println!("    E_Cdir  = {:>10.2} MeV", e_coul_direct);
        println!("    E_Cexch = {:>10.2} MeV", e_coul_exchange);
        println!("    E_pair  = {:>10.2} MeV", e_pair);
        println!("    E_cm    = {:>10.2} MeV", e_cm);
        println!("    E_total = {:>10.2} MeV", e_total);
        println!("    N_p = {:.2}, N_n = {:.2} (target: Z={}, N={})", n_p, n_n, self.z, self.n_neutrons);
        e_total
    }

    fn compute_energy(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        results_p: &SpeciesResult,
        results_n: &SpeciesResult,
        params: &[f64],
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
                if degs[i] * res.v2[i] < 1e-12 { continue; }
                let mut val = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        val += res.eigvec(a, i) * t_eff.get(a, b) * res.eigvec(b, i);
                    }
                }
                e_kin += degs[i] * res.v2[i] * val;
            }
        }

        let sum_rho2: Vec<f64> = (0..self.nr).map(|k| rho_p[k].powi(2) + rho_n[k].powi(2)).collect();

        let integ_t0: Vec<f64> = (0..self.nr)
            .map(|k| ((1.0 + x0 / 2.0) * rho[k].powi(2) - (0.5 + x0) * sum_rho2[k]) * 4.0 * PI * self.r[k].powi(2))
            .collect();
        let e_t0 = (t0 / 2.0) * trapz(&integ_t0, &self.r).unwrap_or(0.0);

        let integ_t3: Vec<f64> = (0..self.nr)
            .map(|k| {
                let rho_safe = rho[k].max(1e-20);
                rho_safe.powf(alpha) * ((1.0 + x3 / 2.0) * rho[k].powi(2) - (0.5 + x3) * sum_rho2[k]) * 4.0 * PI * self.r[k].powi(2)
            })
            .collect();
        let e_t3 = (t3 / 12.0) * trapz(&integ_t3, &self.r).unwrap_or(0.0);

        let v_c = self.coulomb_direct(rho_p);
        let integ_c_direct: Vec<f64> = (0..self.nr).map(|k| v_c[k] * rho_p[k] * 4.0 * PI * self.r[k].powi(2)).collect();
        let e_coul_direct = 0.5 * trapz(&integ_c_direct, &self.r).unwrap_or(0.0);

        let v_cx = self.coulomb_exchange(rho_p);
        let integ_c_exch: Vec<f64> = (0..self.nr).map(|k| v_cx[k] * rho_p[k] * 4.0 * PI * self.r[k].powi(2)).collect();
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
        e_kin + e_t0 + e_t3 + e_coul_direct + e_coul_exchange + e_pair + e_cm
    }

    // ─── Main solver ─────────────────────────────────────────────────

    pub fn solve(&self, params: &[f64], max_iter: usize, tol: f64, mixing: f64) -> HFBResult {
        self.solve_inner(params, max_iter, tol, mixing, false)
    }

    pub fn solve_verbose(&self, params: &[f64], max_iter: usize, tol: f64, mixing: f64) -> HFBResult {
        self.solve_inner(params, max_iter, tol, mixing, true)
    }

    fn solve_inner(&self, params: &[f64], max_iter: usize, tol: f64, mixing: f64, verbose: bool) -> HFBResult {
        let w0 = params[9];
        let nr = self.nr;
        let ns = self.n_states;

        let r_nuc = 1.2 * (self.a as f64).powf(1.0 / 3.0);
        let rho0 = 3.0 * self.a as f64 / (4.0 * PI * r_nuc.powi(3));

        let mut rho_p: Vec<f64> = self.r.iter()
            .map(|&ri| if ri < r_nuc { (rho0 * self.z as f64 / self.a as f64).max(1e-15) } else { 1e-15 })
            .collect();
        let mut rho_n: Vec<f64> = self.r.iter()
            .map(|&ri| if ri < r_nuc { (rho0 * self.n_neutrons as f64 / self.a as f64).max(1e-15) } else { 1e-15 })
            .collect();

        let mut e_prev = 1e10_f64;
        let mut converged = false;
        let mut last_de = 0.0;
        let mut last_iter = 0;
        let mut results_p = SpeciesResult::empty(ns);
        let mut results_n = SpeciesResult::empty(ns);

        for it in 0..max_iter {
            let mut rho_p_new = vec![1e-15; nr];
            let mut rho_n_new = vec![1e-15; nr];

            for is_proton in [true, false] {
                let num_q = if is_proton { self.z } else { self.n_neutrons };
                let delta_q = if is_proton { self.delta_p } else { self.delta_n };

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

                    // Spin-orbit
                    if w0 != 0.0 && self.states[i].l > 0 {
                        let j = self.states[i].j;
                        let l_f = self.states[i].l as f64;
                        let ls = (j * (j + 1.0) - l_f * (l_f + 1.0) - 0.75) / 2.0;
                        let rho_total: Vec<f64> = (0..nr).map(|k| rho_p[k] + rho_n[k]).collect();
                        let drho = gradient_1d(&rho_total, self.dr);
                        let so_integ: Vec<f64> = (0..nr)
                            .map(|k| self.wf[i][k].powi(2) * drho[k] / self.r[k].max(0.1) * self.r[k].powi(2))
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
                                .map(|k| self.wf[idx_i][k] * self.wf[idx_j][k] * u_total[k] * self.r[k].powi(2))
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
                let mut rho_q_new = vec![1e-15; nr];

                for i in 0..ns {
                    if degs[i] * v2[i] < 1e-12 { continue; }
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
                    rho_q_new[k] = rho_q_new[k].max(1e-15);
                }

                if is_proton {
                    rho_p_new = rho_q_new;
                    results_p = SpeciesResult { eigenvalues, eigvecs: eig.eigenvectors, n: ns, v2, _lambda: lam };
                } else {
                    rho_n_new = rho_q_new;
                    results_n = SpeciesResult { eigenvalues, eigvecs: eig.eigenvectors, n: ns, v2, _lambda: lam };
                }
            }

            for k in 0..nr {
                rho_p[k] = (mixing * rho_p_new[k] + (1.0 - mixing) * rho_p[k]).max(1e-15);
                rho_n[k] = (mixing * rho_n_new[k] + (1.0 - mixing) * rho_n[k]).max(1e-15);
            }

            let e_total = self.compute_energy(&rho_p, &rho_n, &results_p, &results_n, params);

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
            self.compute_energy_debug(&rho_p, &rho_n, &results_p, &results_n, params);
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

/// Per-species diagonalization results (flat storage, no nalgebra)
struct SpeciesResult {
    eigenvalues: Vec<f64>,
    eigvecs: Vec<f64>, // row-major n×n
    n: usize,
    v2: Vec<f64>,
    _lambda: f64,
}

impl SpeciesResult {
    fn empty(n_states: usize) -> Self {
        SpeciesResult {
            eigenvalues: vec![0.0; n_states],
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
    if a < 56 || a > 132 {
        (semf_binding_energy(z, n, params), true)
    } else {
        let hfb = SphericalHFB::new_adaptive(z, n);
        let result = hfb.solve(params, 200, 0.05, 0.3);
        (result.binding_energy_mev, result.converged)
    }
}
