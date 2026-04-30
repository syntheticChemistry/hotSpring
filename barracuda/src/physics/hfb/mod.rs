// SPDX-License-Identifier: AGPL-3.0-or-later

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
//! Uses (all `BarraCuda` native — zero external dependencies):
//!   - `barracuda::special::{gamma, laguerre}` for HO basis wavefunctions
//!   - `barracuda::numerical::{trapz, gradient_1d}` for radial integrals & derivatives
//!   - `barracuda::optimize::brent` for BCS chemical potential (matches scipy.optimize.brentq)
//!   - `barracuda::linalg::eigh_f64` for symmetric eigenvalue decomposition
//!
//! # Module structure
//!
//! - `mod.rs` — types, Hamiltonian/density assembly, SCF loop, public API
//! - `basis.rs` — harmonic oscillator basis and radial wavefunctions
//! - `bcs.rs` — BCS occupation factors
//! - `energy.rs` — energy functional and `SpeciesResult`
//! - `potentials.rs` — Coulomb, Skyrme, `T_eff`, Hamiltonian matrix assembly

mod basis;
mod bcs;
mod energy;
mod potentials;
#[cfg(test)]
mod tests;

use basis::BasisState;
use energy::SpeciesResult;
use super::hfb_common::Mat;
use super::semf::semf_binding_energy;
use barracuda::linalg::eigh_f64;
use std::collections::HashMap;

use crate::error::HotSpringError;
use crate::tolerances::{
    BCS_DENSITY_SKIP, DENSITY_FLOOR, HFB_L2_MIXING, HFB_L2_TOLERANCE, HFB_MAX_ITER,
};
use std::f64::consts::PI;

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

/// Result from HFB solve.
#[derive(Debug, Clone)]
pub struct HFBResult {
    /// Total binding energy (`MeV`).
    pub binding_energy_mev: f64,
    /// Whether SCF converged.
    pub converged: bool,
    /// Number of SCF iterations.
    pub iterations: usize,
    /// Final energy change (`MeV`).
    pub delta_e: f64,
}

impl SphericalHFB {
    /// Create HFB solver with given grid and basis parameters.
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

    /// Number of basis states.
    #[must_use]
    pub const fn n_states(&self) -> usize {
        self.n_states
    }
    /// Number of radial grid points.
    #[must_use]
    pub const fn nr(&self) -> usize {
        self.nr
    }
    /// Proton number.
    #[must_use]
    pub const fn z(&self) -> usize {
        self.z
    }
    /// Neutron number.
    #[must_use]
    pub const fn n_neutrons(&self) -> usize {
        self.n_neutrons
    }
    /// Radial grid spacing (fm).
    #[must_use]
    pub const fn dr(&self) -> f64 {
        self.dr
    }
    /// Oscillator energy ℏω (`MeV`).
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
}

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
                    for (r, p) in rho_q_new.iter_mut().zip(phi.iter()) {
                        let p_val: f64 = *p;
                        *r += fact * p_val.powi(2);
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
