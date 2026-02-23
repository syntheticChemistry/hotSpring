// SPDX-License-Identifier: AGPL-3.0-only

//! Axially-Deformed Hartree-Fock + BCS solver (Level 3)
//!
//! Extends the spherical L2 solver to handle nuclear deformation:
//!   - Axially-symmetric HFB in cylindrical coordinates (rho, z)
//!   - Nilsson quantum numbers: `n_z`, `n_perp`, `Lambda`, `Omega`
//!   - Deformation parameter `beta_2` from quadrupole moment `Q20`
//!   - Block-diagonal structure by Omega (K quantum number)
//!   - Self-consistent deformed densities
//!
//! Uses (all `BarraCuda` native):
//!   - `barracuda::linalg::eigh_f64` — block diagonalization
//!   - `barracuda::optimize::brent` — BCS chemical potential
//!   - `barracuda::numerical::{trapz, gradient_1d}` — integrals & gradients
//!   - `barracuda::special::{gamma, laguerre}` — basis functions
//!
//! References:
//!   - Ring & Schuck, "The Nuclear Many-Body Problem" (2004), Ch. 11
//!   - Vautherin & Brink, PRC 5, 626 (1972)
//!   - Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003)

mod basis;
mod potentials;

#[cfg(test)]
mod tests;

use super::constants::{HBAR_C, M_NUCLEON};
use super::hfb_deformed_common::{beta2_from_q20, deformation_guess, rms_radius};
use crate::error::HotSpringError;
use crate::tolerances::{
    BROYDEN_HISTORY, BROYDEN_WARMUP, DIVISION_GUARD, HFB_MAX_ITER, SCF_ENERGY_TOLERANCE,
};
use std::collections::HashMap;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// Deformed basis state
// ═══════════════════════════════════════════════════════════════════

/// Nilsson-like quantum numbers for axially-deformed harmonic oscillator
#[derive(Debug, Clone)]
pub(super) struct DeformedState {
    pub(super) n_z: usize,
    pub(super) n_perp: usize,
    pub(super) lambda: i32,
    pub(super) sigma: i32,
    pub(super) omega_x2: i32,
    #[allow(dead_code)]
    // EVOLUTION(GPU): will be used when deformed_*.wgsl shaders need parity/symmetry checks
    pub(super) _parity: i32,
    #[allow(dead_code)]
    // EVOLUTION(GPU): will be used when deformed_*.wgsl shaders wire shell truncation
    pub(super) _n_shell: usize,
}

impl DeformedState {
    #[allow(dead_code)] // EVOLUTION(GPU): omega() will be used when deformed_*.wgsl shaders need Omega as f64
    fn omega(&self) -> f64 {
        f64::from(self.omega_x2) / 2.0
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2D grid for cylindrical coordinates
// ═══════════════════════════════════════════════════════════════════

/// Cylindrical (rho, z) mesh for deformed densities
#[derive(Debug)]
pub(super) struct CylindricalGrid {
    pub(super) rho: Vec<f64>,
    pub(super) z: Vec<f64>,
    pub(super) n_rho: usize,
    pub(super) n_z: usize,
    pub(super) d_rho: f64,
    pub(super) d_z: f64,
}

impl CylindricalGrid {
    pub(super) fn new(rho_max: f64, z_max: f64, n_rho: usize, n_z: usize) -> Self {
        let d_rho = rho_max / n_rho as f64;
        let d_z = 2.0 * z_max / n_z as f64;
        let rho: Vec<f64> = (1..=n_rho).map(|i| i as f64 * d_rho).collect();
        let z: Vec<f64> = (0..n_z)
            .map(|i| (i as f64 + 0.5).mul_add(d_z, -z_max))
            .collect();

        Self {
            rho,
            z,
            n_rho,
            n_z,
            d_rho,
            d_z,
        }
    }

    /// Volume element: 2*pi*rho * `d_rho` * `d_z`
    pub(super) fn volume_element(&self, i_rho: usize, _i_z: usize) -> f64 {
        2.0 * PI * self.rho[i_rho] * self.d_rho * self.d_z
    }

    /// Flatten (`i_rho`, `i_z`) -> index
    #[inline]
    pub(super) const fn idx(&self, i_rho: usize, i_z: usize) -> usize {
        i_rho * self.n_z + i_z
    }

    /// Total grid points
    pub(super) const fn total(&self) -> usize {
        self.n_rho * self.n_z
    }
}

// ═══════════════════════════════════════════════════════════════════
// Deformed HFB solver
// ═══════════════════════════════════════════════════════════════════

/// Axially-symmetric deformed HFB solver
pub struct DeformedHFB {
    pub(super) z: usize,
    pub(super) n_neutrons: usize,
    pub(super) a: usize,
    pub(super) grid: CylindricalGrid,
    pub(super) hw_z: f64,
    pub(super) hw_perp: f64,
    pub(super) b_z: f64,
    pub(super) b_perp: f64,
    #[allow(dead_code)]
    // EVOLUTION(GPU): will be used when deformed_*.wgsl shaders track deformation during SCF
    _beta2: f64,
    pub(super) delta_p: f64,
    pub(super) delta_n: f64,
    pub(super) states: Vec<DeformedState>,
    pub(super) omega_blocks: HashMap<i32, Vec<usize>>,
}

/// Result from deformed HFB solve
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct DeformedHFBResult {
    pub binding_energy_mev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub delta_e: f64,
    pub beta2: f64,
    pub q20_fm2: f64,
    pub rms_radius_fm: f64,
}

impl DeformedHFB {
    /// Create deformed HFB solver with adaptive parameters
    #[must_use]
    pub fn new_adaptive(z: usize, n: usize) -> Self {
        let a = z + n;
        let a_f = a as f64;

        let hw0 = 41.0 * a_f.powf(-1.0 / 3.0);
        let beta2_init = deformation_guess(z, n);

        let hw_z = hw0 * (1.0 - 2.0 * beta2_init / 3.0);
        let hw_perp = hw0 * (1.0 + beta2_init / 3.0);

        let b_z = HBAR_C / (M_NUCLEON * hw_z).sqrt();
        let b_perp = HBAR_C / (M_NUCLEON * hw_perp).sqrt();

        let r0 = 1.2 * a_f.cbrt();
        let rho_max = (r0 + 8.0).max(12.0);
        let z_max = (r0 * (1.0 + beta2_init.abs()) + 8.0).max(14.0);
        let n_rho = ((rho_max * 8.0) as usize).max(60);
        let n_z = ((2.0 * z_max * 8.0) as usize).max(80);

        let grid = CylindricalGrid::new(rho_max, z_max, n_rho, n_z);

        let delta = 12.0 / a_f.max(4.0).sqrt();

        let n_shells = (2.0 * a_f.cbrt()) as usize + 5;
        let n_shells = n_shells.clamp(10, 16);

        let mut solver = Self {
            z,
            n_neutrons: n,
            a,
            grid,
            hw_z,
            hw_perp,
            b_z,
            b_perp,
            _beta2: beta2_init,
            delta_p: delta,
            delta_n: delta,
            states: Vec::new(),
            omega_blocks: HashMap::new(),
        };

        solver.build_deformed_basis(n_shells);
        solver
    }

    /// Minimal constructor for unit tests (small grid + controllable n_shells).
    #[cfg(test)]
    pub fn new_test_minimal(n_shells: usize, n_rho: usize, n_z: usize) -> Self {
        let z = 8;
        let n = 8;
        let a = z + n;
        let a_f = a as f64;
        let beta2_init = deformation_guess(z, n);
        let hw0 = 41.0 * a_f.powf(-1.0 / 3.0);
        let hw_z = hw0 * (1.0 - 2.0 * beta2_init / 3.0);
        let hw_perp = hw0 * (1.0 + beta2_init / 3.0);
        let b_z = HBAR_C / (M_NUCLEON * hw_z).sqrt();
        let b_perp = HBAR_C / (M_NUCLEON * hw_perp).sqrt();
        let rho_max = 8.0;
        let z_max = 6.0;
        let grid = CylindricalGrid::new(rho_max, z_max, n_rho, n_z);
        let delta = 12.0 / a_f.max(4.0).sqrt();
        let mut solver = DeformedHFB {
            z,
            n_neutrons: n,
            a,
            grid,
            hw_z,
            hw_perp,
            b_z,
            b_perp,
            _beta2: beta2_init,
            delta_p: delta,
            delta_n: delta,
            states: Vec::new(),
            omega_blocks: HashMap::new(),
        };
        solver.build_deformed_basis(n_shells);
        solver
    }

    /// Self-consistent HFB solve with Broyden/modified mixing.
    ///
    /// SCF iteration: density → mean field → diagonalize → new density → mix → repeat.
    /// Uses modified Broyden mixing (Johnson 1988) after initial linear mixing warmup.
    ///
    /// # Errors
    ///
    /// Returns [`HotSpringError::Barracuda`] if the CPU eigensolve fails.
    pub fn solve(&mut self, params: &[f64]) -> Result<DeformedHFBResult, HotSpringError> {
        let max_iter = HFB_MAX_ITER;
        let tol = SCF_ENERGY_TOLERANCE;
        let broyden_warmup = BROYDEN_WARMUP;
        let broyden_history = BROYDEN_HISTORY;

        let mut e_prev = 0.0;
        let mut converged = false;
        let mut binding_energy = 0.0;

        let n_grid = self.grid.total();

        let mut wavefunctions: Vec<Vec<f64>> = self
            .states
            .iter()
            .map(|s| self.evaluate_wavefunction(s))
            .collect();

        self.renormalize_wavefunctions(&mut wavefunctions);

        let mut rho_p = vec![0.0; n_grid];
        let mut rho_n = vec![0.0; n_grid];

        let vec_dim = 2 * n_grid;
        let mut broyden_dfs: Vec<Vec<f64>> = Vec::new();
        let mut broyden_dus: Vec<Vec<f64>> = Vec::new();
        let mut prev_residual: Option<Vec<f64>> = None;
        let mut prev_input: Option<Vec<f64>> = None;

        let mut iter = 0;
        let mut delta_e = 0.0;

        let mut prev_occ_p = vec![0.0; self.states.len()];
        let mut prev_occ_n = vec![0.0; self.states.len()];

        let mut v_coulomb = vec![0.0; n_grid];

        for iteration in 0..max_iter {
            iter = iteration + 1;
            let total_rho: Vec<f64> = rho_p.iter().zip(&rho_n).map(|(&p, &n)| p + n).collect();

            let tau_p_vals = self.compute_tau(&wavefunctions, &prev_occ_p);
            let tau_n_vals = self.compute_tau(&wavefunctions, &prev_occ_n);
            let j_p_vals = self.compute_spin_current(&wavefunctions, &prev_occ_p);
            let j_n_vals = self.compute_spin_current(&wavefunctions, &prev_occ_n);

            self.compute_coulomb_potential(&rho_p, &mut v_coulomb);

            let v_p = self.mean_field_potential_fast(
                params,
                &rho_p,
                &rho_n,
                &total_rho,
                true,
                &tau_p_vals,
                &tau_n_vals,
                &j_p_vals,
                &j_n_vals,
                &v_coulomb,
            );
            let v_n = self.mean_field_potential_fast(
                params,
                &rho_p,
                &rho_n,
                &total_rho,
                false,
                &tau_p_vals,
                &tau_n_vals,
                &j_p_vals,
                &j_n_vals,
                &v_coulomb,
            );

            let (eigs_p, occ_p) =
                self.diagonalize_blocks(&v_p, &wavefunctions, self.z, self.delta_p)?;
            let (eigs_n, occ_n) =
                self.diagonalize_blocks(&v_n, &wavefunctions, self.n_neutrons, self.delta_n)?;

            prev_occ_p.clone_from(&occ_p);
            prev_occ_n.clone_from(&occ_n);

            let (new_rho_p, new_rho_n) = self.compute_densities(&wavefunctions, &occ_p, &occ_n);

            if iteration < broyden_warmup {
                let alpha_mix = if iteration == 0 { 1.0 } else { 0.5 };
                for i in 0..n_grid {
                    rho_p[i] = (1.0_f64 - alpha_mix).mul_add(rho_p[i], alpha_mix * new_rho_p[i]);
                    rho_n[i] = (1.0_f64 - alpha_mix).mul_add(rho_n[i], alpha_mix * new_rho_n[i]);
                }
            } else {
                let alpha_mix = 0.4;

                let input_vec: Vec<f64> = rho_p.iter().chain(rho_n.iter()).copied().collect();
                let output_vec: Vec<f64> =
                    new_rho_p.iter().chain(new_rho_n.iter()).copied().collect();
                let residual: Vec<f64> = output_vec
                    .iter()
                    .zip(&input_vec)
                    .map(|(&out, &inp)| out - inp)
                    .collect();

                if let (Some(prev_r), Some(prev_u)) = (&prev_residual, &prev_input) {
                    let df: Vec<f64> = residual
                        .iter()
                        .zip(prev_r)
                        .map(|(&r, &pr)| r - pr)
                        .collect();
                    let du: Vec<f64> = input_vec
                        .iter()
                        .zip(prev_u)
                        .map(|(&u, &pu)| u - pu)
                        .collect();

                    if broyden_dfs.len() >= broyden_history {
                        broyden_dfs.remove(0);
                        broyden_dus.remove(0);
                    }
                    broyden_dfs.push(df);
                    broyden_dus.push(du);
                }

                let mut mixed = vec![0.0; vec_dim];
                if broyden_dfs.is_empty() {
                    for i in 0..vec_dim {
                        mixed[i] = input_vec[i] + alpha_mix * residual[i];
                    }
                } else {
                    for i in 0..vec_dim {
                        mixed[i] = input_vec[i] + alpha_mix * residual[i];
                    }
                    for m in 0..broyden_dfs.len() {
                        let df_dot_r: f64 = broyden_dfs[m]
                            .iter()
                            .zip(&residual)
                            .map(|(&a, &b)| a * b)
                            .sum();
                        let df_dot_df: f64 = broyden_dfs[m].iter().map(|&a| a * a).sum();
                        if df_dot_df > DIVISION_GUARD {
                            let gamma = df_dot_r / df_dot_df;
                            for i in 0..vec_dim {
                                mixed[i] -=
                                    gamma * (broyden_dus[m][i] + alpha_mix * broyden_dfs[m][i]);
                            }
                        }
                    }
                }

                prev_residual = Some(residual);
                prev_input = Some(input_vec);

                rho_p = mixed[..n_grid].to_vec();
                rho_n = mixed[n_grid..].to_vec();

                for v in &mut rho_p {
                    *v = v.max(0.0);
                }
                for v in &mut rho_n {
                    *v = v.max(0.0);
                }
            }

            binding_energy =
                self.total_energy(params, &rho_p, &rho_n, &eigs_p, &eigs_n, &occ_p, &occ_n);

            if !binding_energy.is_finite() || binding_energy.abs() > 1e10 {
                break;
            }

            delta_e = (binding_energy - e_prev).abs();
            if iteration > broyden_warmup && delta_e < tol {
                converged = true;
                break;
            }
            e_prev = binding_energy;

            #[cfg(test)]
            if iteration < 5 || iteration % 50 == 0 {
                let n_p_check: f64 = rho_p
                    .iter()
                    .enumerate()
                    .map(|(i, &rp)| {
                        rp * self
                            .grid
                            .volume_element(i / self.grid.n_z, i % self.grid.n_z)
                    })
                    .sum();
                let n_n_check: f64 = rho_n
                    .iter()
                    .enumerate()
                    .map(|(i, &rn)| {
                        rn * self
                            .grid
                            .volume_element(i / self.grid.n_z, i % self.grid.n_z)
                    })
                    .sum();
                eprintln!(
                    "  iter {iteration:3}: E={binding_energy:12.3} MeV  N_p={n_p_check:.2}  N_n={n_n_check:.2}  dE={delta_e:.3e}"
                );
            }
        }

        let rho_total: Vec<f64> = rho_p.iter().zip(&rho_n).map(|(&p, &n)| p + n).collect();
        let q20 = self.quadrupole_moment(&rho_total);
        let beta2 = beta2_from_q20(self.a, q20);
        let rms_r = rms_radius(
            &rho_total,
            self.grid.n_rho,
            self.grid.n_z,
            self.grid.d_rho,
            self.grid.d_z,
            0.5_f64.mul_add(-self.grid.d_z, self.grid.z[0]),
        );

        Ok(DeformedHFBResult {
            binding_energy_mev: binding_energy,
            converged,
            iterations: iter,
            delta_e,
            beta2,
            q20_fm2: q20,
            rms_radius_fm: rms_r,
        })
    }
}

/// Public API: L3 binding energy from deformed HFB.
///
/// # Errors
///
/// Returns [`HotSpringError::Barracuda`] if the deformed HFB eigensolve fails.
pub fn binding_energy_l3(
    z: usize,
    n: usize,
    params: &[f64],
) -> Result<(f64, bool, f64), HotSpringError> {
    let mut solver = DeformedHFB::new_adaptive(z, n);
    let result = solver.solve(params)?;
    Ok((result.binding_energy_mev, result.converged, result.beta2))
}
