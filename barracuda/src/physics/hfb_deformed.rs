// SPDX-License-Identifier: AGPL-3.0-only

//! Axially-Deformed Hartree-Fock + BCS solver (Level 3)
//!
//! Extends the spherical L2 solver to handle nuclear deformation:
//!   - Axially-symmetric HFB in cylindrical coordinates (rho, z)
//!   - Nilsson quantum numbers: n_z, n_perp, Lambda, Omega
//!   - Deformation parameter beta_2 from quadrupole moment Q20
//!   - Block-diagonal structure by Omega (K quantum number)
//!   - Self-consistent deformed densities
//!
//! Uses (all BarraCUDA native):
//!   - barracuda::linalg::eigh_f64 — block diagonalization
//!   - barracuda::optimize::brent — BCS chemical potential
//!   - barracuda::numerical::{trapz, gradient_1d} — integrals & gradients
//!   - barracuda::special::{gamma, laguerre} — basis functions
//!
//! References:
//!   - Ring & Schuck, "The Nuclear Many-Body Problem" (2004), Ch. 11
//!   - Vautherin & Brink, PRC 5, 626 (1972)
//!   - Bender, Heenen, Reinhard, Rev. Mod. Phys. 75, 121 (2003)

use super::constants::*;
use super::hfb_common::{factorial_f64, hermite_value, Mat};
use crate::tolerances::SPIN_ORBIT_R_MIN;
use barracuda::linalg::eigh_f64;
use barracuda::special::{gamma, laguerre};
use std::collections::HashMap;
use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════
// Deformed basis state
// ═══════════════════════════════════════════════════════════════════

/// Nilsson-like quantum numbers for axially-deformed harmonic oscillator
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DeformedState {
    n_z: usize,      // quantum number along symmetry axis
    n_perp: usize,   // perpendicular oscillator quanta
    lambda: i32,     // projection of orbital angular momentum on z-axis
    sigma: i32,      // +1 or -1 (spin projection: +1/2 or -1/2)
    omega_x2: i32,   // 2*Omega = 2*Lambda + sigma (block index, always > 0)
    _parity: i32,    // (-1)^(n_z + 2*n_perp + |lambda|) = (-1)^N
    _n_shell: usize, // major shell N = n_z + 2*n_perp + |lambda|
}

#[allow(dead_code)]
impl DeformedState {
    fn omega(&self) -> f64 {
        self.omega_x2 as f64 / 2.0
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2D grid for cylindrical coordinates
// ═══════════════════════════════════════════════════════════════════

/// Cylindrical (rho, z) mesh for deformed densities
#[derive(Debug)]
struct CylindricalGrid {
    rho: Vec<f64>, // perpendicular coordinate
    z: Vec<f64>,   // symmetry axis coordinate
    n_rho: usize,
    n_z: usize,
    d_rho: f64,
    d_z: f64,
}

impl CylindricalGrid {
    fn new(rho_max: f64, z_max: f64, n_rho: usize, n_z: usize) -> Self {
        let d_rho = rho_max / n_rho as f64;
        let d_z = 2.0 * z_max / n_z as f64;
        let rho: Vec<f64> = (1..=n_rho).map(|i| i as f64 * d_rho).collect();
        let z: Vec<f64> = (0..n_z).map(|i| -z_max + (i as f64 + 0.5) * d_z).collect();

        CylindricalGrid {
            rho,
            z,
            n_rho,
            n_z,
            d_rho,
            d_z,
        }
    }

    /// Volume element: 2*pi*rho * d_rho * d_z
    fn volume_element(&self, i_rho: usize, _i_z: usize) -> f64 {
        2.0 * PI * self.rho[i_rho] * self.d_rho * self.d_z
    }

    /// Flatten (i_rho, i_z) -> index
    #[inline]
    fn idx(&self, i_rho: usize, i_z: usize) -> usize {
        i_rho * self.n_z + i_z
    }

    /// Total grid points
    fn total(&self) -> usize {
        self.n_rho * self.n_z
    }
}

// ═══════════════════════════════════════════════════════════════════
// Deformed HFB solver
// ═══════════════════════════════════════════════════════════════════

/// Axially-symmetric deformed HFB solver
#[allow(dead_code)]
pub struct DeformedHFB {
    z: usize,
    n_neutrons: usize,
    a: usize,
    grid: CylindricalGrid,
    hw_z: f64,    // HO frequency along z (can differ from perp for deformation)
    hw_perp: f64, // HO frequency perpendicular
    b_z: f64,     // HO length parameter along z
    b_perp: f64,  // HO length parameter perpendicular
    _beta2: f64,  // deformation parameter
    delta_p: f64, // proton pairing gap
    delta_n: f64, // neutron pairing gap
    states: Vec<DeformedState>,
    /// Block structure: omega_x2 → state indices
    omega_blocks: HashMap<i32, Vec<usize>>,
}

/// Result from deformed HFB solve
#[derive(Debug, Clone)]
pub struct DeformedHFBResult {
    pub binding_energy_mev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub delta_e: f64,
    pub beta2: f64,   // final deformation
    pub q20_fm2: f64, // quadrupole moment
    pub rms_radius_fm: f64,
}

impl DeformedHFB {
    /// Create deformed HFB solver with adaptive parameters
    pub fn new_adaptive(z: usize, n: usize) -> Self {
        let a = z + n;
        let a_f = a as f64;

        // HO frequencies: axially-symmetric
        let hw0 = 41.0 * a_f.powf(-1.0 / 3.0);

        // Initial deformation guess: prolate for typical deformed nuclei
        let beta2_init = Self::initial_deformation_guess(z, n);

        // Deformed oscillator parameters
        // omega_z = omega_0 * (1 - 2*beta2/3)
        // omega_perp = omega_0 * (1 + beta2/3)
        // This preserves volume: omega_z * omega_perp^2 = omega_0^3
        let hw_z = hw0 * (1.0 - 2.0 * beta2_init / 3.0);
        let hw_perp = hw0 * (1.0 + beta2_init / 3.0);

        let b_z = HBAR_C / (M_NUCLEON * hw_z).sqrt();
        let b_perp = HBAR_C / (M_NUCLEON * hw_perp).sqrt();

        // Grid: adaptive to nucleus size
        let r0 = 1.2 * a_f.powf(1.0 / 3.0);
        let rho_max = (r0 + 8.0).max(12.0);
        let z_max = (r0 * (1.0 + beta2_init.abs()) + 8.0).max(14.0);
        let n_rho = ((rho_max * 8.0) as usize).max(60);
        let n_z = ((2.0 * z_max * 8.0) as usize).max(80);

        let grid = CylindricalGrid::new(rho_max, z_max, n_rho, n_z);

        // Pairing gaps
        let delta = 12.0 / a_f.max(4.0).sqrt();

        // Number of shells (slightly more than spherical to capture deformation)
        let n_shells = (2.0 * a_f.powf(1.0 / 3.0)) as usize + 5;
        let n_shells = n_shells.max(10).min(16);

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

    /// Heuristic initial deformation based on nuclear chart
    fn initial_deformation_guess(z: usize, n: usize) -> f64 {
        let a = z + n;
        // Known deformed regions:
        // 1. Rare earths: 150 < A < 190 (beta2 ~ 0.2-0.35)
        // 2. Actinides: A > 222 (beta2 ~ 0.2-0.3)
        // 3. Light deformed: 20 < A < 28 (beta2 ~ 0.3-0.5)
        // Magic numbers: Z,N = 2,8,20,28,50,82,126 → spherical

        let magic = [2, 8, 20, 28, 50, 82, 126];
        let z_magic = magic.iter().any(|&m| (z as i32 - m).unsigned_abs() <= 2);
        let n_magic = magic.iter().any(|&m| (n as i32 - m).unsigned_abs() <= 2);

        if z_magic && n_magic {
            0.0 // doubly magic → spherical
        } else if a > 222 {
            0.25 // actinides
        } else if a > 150 && a < 190 {
            0.28 // rare earths
        } else if a > 20 && a < 28 {
            0.35 // sd-shell deformed
        } else if z_magic || n_magic {
            0.05 // near magic → weakly deformed
        } else {
            0.15 // generic
        }
    }

    /// Build deformed HO basis: enumerate (n_z, n_perp, Lambda, sigma)
    fn build_deformed_basis(&mut self, n_shells: usize) {
        self.states.clear();
        self.omega_blocks.clear();

        for n_sh in 0..n_shells {
            // N = n_z + 2*n_perp + |Lambda|
            for n_z in 0..=n_sh {
                let remaining = n_sh - n_z;
                // remaining = 2*n_perp + |Lambda|
                for n_perp in 0..=(remaining / 2) {
                    let abs_lambda = remaining - 2 * n_perp;

                    // Lambda can be +/- abs_lambda, but we only need Lambda >= 0
                    // since (Lambda, sigma) and (-Lambda, -sigma) give same Omega
                    let lambdas = if abs_lambda == 0 {
                        vec![0_i32]
                    } else {
                        vec![abs_lambda as i32]
                    };

                    for &lambda in &lambdas {
                        for &sigma in &[1_i32, -1_i32] {
                            let omega_x2 = 2 * lambda + sigma;

                            // Only positive Omega blocks (time-reversal partner is implicit)
                            if omega_x2 <= 0 {
                                continue;
                            }

                            let parity = if n_sh % 2 == 0 { 1 } else { -1 };

                            self.states.push(DeformedState {
                                n_z,
                                n_perp,
                                lambda,
                                sigma,
                                omega_x2,
                                _parity: parity,
                                _n_shell: n_sh,
                            });
                        }
                    }
                }
            }
        }

        // Build Omega-block index
        for (i, s) in self.states.iter().enumerate() {
            self.omega_blocks.entry(s.omega_x2).or_default().push(i);
        }
    }

    /// Evaluate deformed HO wavefunction on the cylindrical grid
    /// psi(rho, z) = phi_nz(z/b_z) * phi_{n_perp,|Lambda|}(rho/b_perp)
    fn evaluate_wavefunction(&self, state: &DeformedState) -> Vec<f64> {
        let mut psi = vec![0.0; self.grid.total()];

        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];

                // z-part: Hermite oscillator
                let phi_z = Self::hermite_oscillator(state.n_z, z, self.b_z);

                // rho-part: 2D oscillator with |Lambda|
                let phi_rho = Self::laguerre_oscillator(
                    state.n_perp,
                    state.lambda.unsigned_abs() as usize,
                    rho,
                    self.b_perp,
                );

                psi[self.grid.idx(i_rho, i_z)] = phi_z * phi_rho;
            }
        }

        psi
    }

    /// 1D harmonic oscillator along z: H_n(z/b) * exp(-z²/2b²) * normalization
    fn hermite_oscillator(n: usize, z: f64, b: f64) -> f64 {
        let xi = z / b;
        // Use Laguerre-based Hermite connection for numerical stability:
        // H_{2m}(x) related to L_m^{-1/2}(x²)
        // H_{2m+1}(x) related to L_m^{1/2}(x²) * x
        // But simpler: direct Hermite via recurrence
        let h_n = hermite_value(n, xi);
        let norm = 1.0 / (b * PI.sqrt() * (1 << n) as f64 * factorial_f64(n)).sqrt();
        norm * h_n * (-xi * xi / 2.0).exp()
    }

    /// 2D oscillator radial part: Laguerre basis
    /// R_{n_perp, |Lambda|}(rho) = sqrt(n! / (pi * b² * Gamma(n+|L|+1))) * (rho/b)^|L|
    ///                              * exp(-rho²/2b²) * L_n^|L|(rho²/b²)
    ///
    /// Normalization: integral |R|² * 2*pi*rho * d_rho = 1
    /// The 1/sqrt(pi) accounts for the azimuthal 2*pi from the volume element
    /// canceling with exp(i*Lambda*phi)/sqrt(2*pi).
    fn laguerre_oscillator(n_perp: usize, abs_lambda: usize, rho: f64, b: f64) -> f64 {
        let eta = (rho / b).powi(2);
        let alpha = abs_lambda as f64;

        let n_fact = factorial_f64(n_perp);
        let gamma_val = gamma(n_perp as f64 + alpha + 1.0).unwrap_or(1.0);
        // Include 1/sqrt(2*pi) so that integral |R|^2 * 2*pi*rho*d_rho = 1
        let norm = (n_fact / (PI * b * b * gamma_val)).sqrt();

        let lag = laguerre(n_perp, alpha, eta);
        norm * (rho / b).powi(abs_lambda as i32) * (-eta / 2.0).exp() * lag
    }

    /// Self-consistent HFB solve with Broyden/modified mixing
    ///
    /// SCF iteration: density → mean field → diagonalize → new density → mix → repeat
    /// Uses modified Broyden mixing (Johnson 1988) after initial linear mixing warmup.
    pub fn solve(&mut self, params: &[f64]) -> DeformedHFBResult {
        let max_iter = 200;
        let tol = 1e-6; // Physics: SCF energy convergence threshold — 1 eV ≈ 1e-6 of total BE
        let broyden_warmup = 50; // linear mixing for first N iterations (converge density first)
        let broyden_history = 8; // max Broyden history vectors

        let mut e_prev = 0.0;
        let mut converged = false;
        let mut binding_energy = 0.0;

        // Initialize with zero mean-field → just oscillator
        let n_grid = self.grid.total();

        // Precompute all wavefunctions on the grid, then renormalize
        // to ensure integral |psi|² dV = 1 on the discrete grid
        let mut wavefunctions: Vec<Vec<f64>> = self
            .states
            .iter()
            .map(|s| self.evaluate_wavefunction(s))
            .collect();

        // Renormalize each wavefunction on the grid
        for psi in &mut wavefunctions {
            let norm2: f64 = (0..n_grid)
                .map(|k| {
                    let i_rho = k / self.grid.n_z;
                    let i_z = k % self.grid.n_z;
                    psi[k] * psi[k] * self.grid.volume_element(i_rho, i_z)
                })
                .sum();
            if norm2 > 1e-30 {
                // Physics: division-by-zero guard — well below any physical scale
                let scale = 1.0 / norm2.sqrt();
                for v in psi.iter_mut() {
                    *v *= scale;
                }
            }
        }
        let mut rho_p = vec![0.0; n_grid];
        let mut rho_n = vec![0.0; n_grid];

        // Broyden mixing state: store residual history
        // vec_dim = 2*n_grid (rho_p and rho_n concatenated)
        let vec_dim = 2 * n_grid;
        let mut broyden_dfs: Vec<Vec<f64>> = Vec::new(); // delta(F) history
        let mut broyden_dus: Vec<Vec<f64>> = Vec::new(); // delta(u) history
        let mut prev_residual: Option<Vec<f64>> = None;
        let mut prev_input: Option<Vec<f64>> = None;

        let mut iter = 0;
        let mut delta_e = 0.0;

        // Previous-iteration occupations for tau/J computation
        // (avoids double-diagonalization per iteration)
        let mut prev_occ_p = vec![0.0; self.states.len()];
        let mut prev_occ_n = vec![0.0; self.states.len()];

        // Precompute Coulomb potential as a separate step (avoids O(n²) in mean_field)
        let mut v_coulomb = vec![0.0; n_grid];

        for iteration in 0..max_iter {
            iter = iteration + 1;
            let total_rho: Vec<f64> = rho_p.iter().zip(&rho_n).map(|(&p, &n)| p + n).collect();

            // Compute tau and J from previous-iteration occupations (zero on first iter)
            let tau_p_vals = self.compute_tau(&wavefunctions, &prev_occ_p);
            let tau_n_vals = self.compute_tau(&wavefunctions, &prev_occ_n);
            let j_p_vals = self.compute_spin_current(&wavefunctions, &prev_occ_p);
            let j_n_vals = self.compute_spin_current(&wavefunctions, &prev_occ_n);

            // Precompute Coulomb potential (O(n²), but only once per iteration)
            self.compute_coulomb_potential(&rho_p, &mut v_coulomb);

            // Compute full mean-field potentials (with tau, J, precomputed Coulomb)
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

            // Diagonalize block-by-block for each Omega
            let (eigs_p, occ_p) =
                self.diagonalize_blocks(&v_p, &wavefunctions, self.z, self.delta_p);
            let (eigs_n, occ_n) =
                self.diagonalize_blocks(&v_n, &wavefunctions, self.n_neutrons, self.delta_n);

            // Store occupations for next iteration's tau/J
            prev_occ_p = occ_p.clone();
            prev_occ_n = occ_n.clone();

            // Compute output densities
            let (new_rho_p, new_rho_n) = self.compute_densities(&wavefunctions, &occ_p, &occ_n);

            // ── Density mixing (Broyden after warmup, linear during warmup) ──
            if iteration < broyden_warmup {
                // First iteration: full replacement (no previous density to mix with)
                // Subsequent warmup: conservative linear mixing
                let alpha_mix = if iteration == 0 { 1.0 } else { 0.5 };
                for i in 0..n_grid {
                    rho_p[i] = (1.0 - alpha_mix) * rho_p[i] + alpha_mix * new_rho_p[i];
                    rho_n[i] = (1.0 - alpha_mix) * rho_n[i] + alpha_mix * new_rho_n[i];
                }
            } else {
                // Modified Broyden mixing (Johnson, PRB 38, 12807, 1988)
                let alpha_mix = 0.4;

                // Pack current input and output into vectors
                let input_vec: Vec<f64> = rho_p.iter().chain(rho_n.iter()).copied().collect();
                let output_vec: Vec<f64> =
                    new_rho_p.iter().chain(new_rho_n.iter()).copied().collect();
                let residual: Vec<f64> = output_vec
                    .iter()
                    .zip(&input_vec)
                    .map(|(&out, &inp)| out - inp)
                    .collect();

                // Broyden update
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

                    // Store in history (drop oldest if full)
                    if broyden_dfs.len() >= broyden_history {
                        broyden_dfs.remove(0);
                        broyden_dus.remove(0);
                    }
                    broyden_dfs.push(df);
                    broyden_dus.push(du);
                }

                // Apply Broyden correction if we have history
                let mut mixed = vec![0.0; vec_dim];
                if broyden_dfs.is_empty() {
                    // First Broyden step: just linear mixing
                    for i in 0..vec_dim {
                        mixed[i] = input_vec[i] + alpha_mix * residual[i];
                    }
                } else {
                    // Simplified Broyden: x_new = x + alpha*F - sum_m gamma_m * (du_m + alpha*df_m)
                    // where gamma_m = <df_m | F> / <df_m | df_m>
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
                        if df_dot_df > 1e-30 {
                            // Physics: Broyden gamma denominator guard — avoid division by near-zero
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

                // Unpack
                rho_p = mixed[..n_grid].to_vec();
                rho_n = mixed[n_grid..].to_vec();

                // Ensure non-negative densities
                for v in &mut rho_p {
                    *v = v.max(0.0);
                }
                for v in &mut rho_n {
                    *v = v.max(0.0);
                }
            }

            // Compute total energy
            binding_energy =
                self.total_energy(params, &rho_p, &rho_n, &eigs_p, &eigs_n, &occ_p, &occ_n);

            // Divergence protection: if energy is unphysical, bail out
            if !binding_energy.is_finite() || binding_energy.abs() > 1e10 {
                // Physics: divergence protection — BE typically -500 to -2000 MeV
                break;
            }

            delta_e = (binding_energy - e_prev).abs();
            if iteration > broyden_warmup && delta_e < tol {
                converged = true;
                break;
            }
            e_prev = binding_energy;

            // Debug output for first few iterations (only in test/debug builds)
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

        // Compute deformation observables
        let rho_total: Vec<f64> = rho_p.iter().zip(&rho_n).map(|(&p, &n)| p + n).collect();
        let q20 = self.quadrupole_moment(&rho_total);
        let beta2 = self.beta2_from_q20(q20);
        let rms_r = self.rms_radius(&rho_total);

        DeformedHFBResult {
            binding_energy_mev: binding_energy,
            converged,
            iterations: iter,
            delta_e,
            beta2,
            q20_fm2: q20,
            rms_radius_fm: rms_r,
        }
    }

    /// Compute kinetic energy density tau(r) from wavefunctions and occupations
    /// tau = sum_i n_i |grad psi_i|^2
    fn compute_tau(&self, wavefunctions: &[Vec<f64>], occ: &[f64]) -> Vec<f64> {
        let n = self.grid.total();
        let mut tau = vec![0.0; n];

        for (i, _s) in self.states.iter().enumerate() {
            let occ_i = occ[i] * 2.0; // time-reversal degeneracy
            if occ_i < 1e-15 {
                // Physics: machine epsilon floor — skip negligible tau contributions
                continue;
            }

            let psi = &wavefunctions[i];
            // Numerical gradient |grad psi|^2 via finite differences
            for i_rho in 0..self.grid.n_rho {
                for i_z in 0..self.grid.n_z {
                    let idx = self.grid.idx(i_rho, i_z);

                    // d psi / d rho (finite difference)
                    let dpsi_drho = if i_rho == 0 {
                        (psi[self.grid.idx(1, i_z)] - psi[idx]) / self.grid.d_rho
                    } else if i_rho == self.grid.n_rho - 1 {
                        (psi[idx] - psi[self.grid.idx(i_rho - 1, i_z)]) / self.grid.d_rho
                    } else {
                        (psi[self.grid.idx(i_rho + 1, i_z)] - psi[self.grid.idx(i_rho - 1, i_z)])
                            / (2.0 * self.grid.d_rho)
                    };

                    // d psi / d z (finite difference)
                    let dpsi_dz = if i_z == 0 {
                        (psi[self.grid.idx(i_rho, 1)] - psi[idx]) / self.grid.d_z
                    } else if i_z == self.grid.n_z - 1 {
                        (psi[idx] - psi[self.grid.idx(i_rho, i_z - 1)]) / self.grid.d_z
                    } else {
                        (psi[self.grid.idx(i_rho, i_z + 1)] - psi[self.grid.idx(i_rho, i_z - 1)])
                            / (2.0 * self.grid.d_z)
                    };

                    tau[idx] += occ_i * (dpsi_drho * dpsi_drho + dpsi_dz * dpsi_dz);
                }
            }
        }

        tau
    }

    /// Compute spin-orbit density J(r) = sum_i n_i * psi_i * (l x s) * psi_i
    /// For axially-symmetric case, the relevant component is J_z ~ Lambda * sigma
    fn compute_spin_current(&self, wavefunctions: &[Vec<f64>], occ: &[f64]) -> Vec<f64> {
        let n = self.grid.total();
        let mut j_density = vec![0.0; n];

        for (i, s) in self.states.iter().enumerate() {
            let occ_i = occ[i] * 2.0;
            if occ_i < 1e-15 {
                // Physics: machine epsilon floor — skip negligible J contributions
                continue;
            }

            // Spin-current: <l·s> contribution is Lambda * sigma / 2
            let ls = s.lambda as f64 * s.sigma as f64 * 0.5;

            for k in 0..n {
                let psi2 = wavefunctions[i][k] * wavefunctions[i][k];
                j_density[k] += occ_i * ls * psi2;
            }
        }

        j_density
    }

    /// Compute |grad rho|² (squared gradient magnitude) for spin-orbit potential
    /// In cylindrical coordinates: |grad f|² = (df/d_rho)² + (df/dz)²
    /// Returns the radial derivative (df/dr) for use in the simplified spin-orbit:
    ///   V_ls ~ -W0/2 * (1/r) * d_rho/dr
    fn density_radial_derivative(&self, density: &[f64]) -> Vec<f64> {
        let n = self.grid.total();
        let mut deriv = vec![0.0; n];

        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);

                let d_drho = if i_rho == 0 {
                    (density[self.grid.idx(1, i_z)] - density[idx]) / self.grid.d_rho
                } else if i_rho == self.grid.n_rho - 1 {
                    (density[idx] - density[self.grid.idx(i_rho - 1, i_z)]) / self.grid.d_rho
                } else {
                    (density[self.grid.idx(i_rho + 1, i_z)]
                        - density[self.grid.idx(i_rho - 1, i_z)])
                        / (2.0 * self.grid.d_rho)
                };

                let d_dz = if i_z == 0 {
                    (density[self.grid.idx(i_rho, 1)] - density[idx]) / self.grid.d_z
                } else if i_z == self.grid.n_z - 1 {
                    (density[idx] - density[self.grid.idx(i_rho, i_z - 1)]) / self.grid.d_z
                } else {
                    (density[self.grid.idx(i_rho, i_z + 1)]
                        - density[self.grid.idx(i_rho, i_z - 1)])
                        / (2.0 * self.grid.d_z)
                };

                // Radial derivative: project (d/drho, d/dz) onto radial direction
                let rho_coord = self.grid.rho[i_rho];
                let z_coord = self.grid.z[i_z];
                let r = (rho_coord * rho_coord + z_coord * z_coord).sqrt().max(0.01); // Physics: minimum r for 1/r potential — avoids singularity at origin
                deriv[idx] = (d_drho * rho_coord + d_dz * z_coord) / r;
            }
        }

        deriv
    }

    /// Precompute Coulomb potential on the cylindrical grid (O(n²) but done once per SCF iter)
    ///
    /// Uses spherical monopole approximation:
    ///   V_C(r) = e² * [Q_enclosed(r) / r + V_exterior(r)]
    /// where charges are sorted by radial distance.
    fn compute_coulomb_potential(&self, rho_p: &[f64], v_coulomb: &mut [f64]) {
        let n = self.grid.total();

        // Precompute radial distances and charges for all grid points
        let mut charge_shells: Vec<(f64, f64)> = Vec::with_capacity(n); // (radius, charge)
        for i in 0..n {
            let i_rho = i / self.grid.n_z;
            let i_z = i % self.grid.n_z;
            let rho = self.grid.rho[i_rho];
            let z = self.grid.z[i_z];
            let r = (rho * rho + z * z).sqrt();
            let dv = self.grid.volume_element(i_rho, i_z);
            let charge = rho_p[i].max(0.0) * dv;
            charge_shells.push((r, charge));
        }

        // Sort by radius for efficient enclosed-charge computation
        let mut sorted_idx: Vec<usize> = (0..n).collect();
        sorted_idx.sort_by(|&a, &b| charge_shells[a].0.total_cmp(&charge_shells[b].0));

        // Prefix sums: total charge and charge/r for exterior
        let total_charge: f64 = charge_shells.iter().map(|(_, c)| c).sum();
        let total_charge_over_r: f64 = charge_shells
            .iter()
            .map(|(r, c)| if *r > 0.01 { c / r } else { 0.0 }) // Physics: Coulomb 1/r — guard at origin
            .sum();

        // For each grid point, compute V_C using sorted radial shells
        // V_C(r) = e² * (Q_<(r)/r + sum_{r'>r} q_j/r_j)
        // Build cumulative sums efficiently
        let mut cum_charge = vec![0.0; n]; // cumulative charge up to index
        let mut cum_charge_over_r = vec![0.0; n]; // cumulative charge/r up to index
        {
            let mut acc_q = 0.0;
            let mut acc_qr = 0.0;
            for (k, &si) in sorted_idx.iter().enumerate() {
                acc_q += charge_shells[si].1;
                let r = charge_shells[si].0.max(0.01); // Physics: Coulomb 1/r — avoid singularity
                acc_qr += charge_shells[si].1 / r;
                cum_charge[k] = acc_q;
                cum_charge_over_r[k] = acc_qr;
            }
        }

        // Build reverse mapping: for each grid point, its position in sorted order
        let mut rank = vec![0usize; n];
        for (k, &si) in sorted_idx.iter().enumerate() {
            rank[si] = k;
        }

        // Compute Coulomb potential
        for i in 0..n {
            let r_i = charge_shells[i].0.max(0.01); // Physics: Coulomb 1/r — avoid singularity
            let k = rank[i];

            // Q_enclosed / r_i
            let q_inner = if k > 0 { cum_charge[k - 1] } else { 0.0 };
            // Sum of q_j/r_j for exterior (r_j > r_i)
            let ext_qr = total_charge_over_r - cum_charge_over_r[k];

            // Direct + Slater exchange
            v_coulomb[i] = E2 * (q_inner / r_i + ext_qr)
                + super::hfb_common::coulomb_exchange_slater(rho_p[i]);
        }

        // Handle edge case: if no proton charge, zero out
        if total_charge < 1e-30 {
            // Physics: numerical zero — no protons, zero Coulomb
            for v in v_coulomb.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// Fast mean-field potential using precomputed Coulomb
    ///
    /// Includes:
    /// - Central Skyrme (t0, t3, x0, x3, alpha)
    /// - Effective mass / kinetic density (t1, t2, x1, x2)
    /// - Simplified spin-orbit: V_ls = -W0/2 * (1/r) * d_rho/dr
    /// - Coulomb (precomputed monopole + Slater exchange)
    fn mean_field_potential_fast(
        &self,
        params: &[f64],
        rho_p: &[f64],
        rho_n: &[f64],
        rho_total: &[f64],
        is_proton: bool,
        tau_p: &[f64],
        tau_n: &[f64],
        _j_p: &[f64],
        _j_n: &[f64],
        v_coulomb: &[f64],
    ) -> Vec<f64> {
        let n = self.grid.total();
        let mut v = vec![0.0; n];

        let t0 = params[0];
        let t1 = params[1];
        let t2 = params[2];
        let t3 = params[3];
        let x0 = params[4];
        let x1 = params[5];
        let x2 = params[6];
        let x3 = params[7];
        let alpha = params[8];
        let w0 = params[9];

        let rho_q: &[f64] = if is_proton { rho_p } else { rho_n };
        let tau_total: Vec<f64> = tau_p.iter().zip(tau_n).map(|(&a, &b)| a + b).collect();
        let tau_q: &[f64] = if is_proton { tau_p } else { tau_n };

        // Spin-orbit: simplified form using density gradient
        // V_ls ~ -W0/2 * (1/r) * d_rho_total/dr for the <l·s> diagonal part
        let d_rho_dr = self.density_radial_derivative(rho_total);
        let d_rho_q_dr = self.density_radial_derivative(rho_q);

        for i in 0..n {
            let rho = rho_total[i].max(0.0);
            let rq = rho_q[i].max(0.0);

            // ── Central Skyrme (t0, t3 terms) ──
            let v_central = t0 * ((1.0 + x0 / 2.0) * rho - (0.5 + x0) * rq)
                + t3 / 12.0
                    * rho.powf(alpha)
                    * ((2.0 + alpha) * (1.0 + x3 / 2.0) * rho
                        - (2.0 * (0.5 + x3) * rq + alpha * (1.0 + x3 / 2.0) * rho));

            // ── Effective mass terms (t1, t2) ──
            let v_eff_mass = t1 / 4.0 * ((2.0 + x1) * tau_total[i] - (1.0 + 2.0 * x1) * tau_q[i])
                + t2 / 4.0 * ((2.0 + x2) * tau_total[i] + (1.0 + 2.0 * x2) * tau_q[i]);

            // ── Spin-orbit (simplified) ──
            // V_ls = -W0/2 * (d_rho/dr + d_rho_q/dr) / r
            // This enters the Hamiltonian as <i|V_ls * l·s|i> for each state
            let i_rho = i / self.grid.n_z;
            let i_z = i % self.grid.n_z;
            let rho_coord = self.grid.rho[i_rho];
            let z_coord = self.grid.z[i_z];
            let r = (rho_coord * rho_coord + z_coord * z_coord)
                .sqrt()
                .max(SPIN_ORBIT_R_MIN);
            let v_so = -w0 / 2.0 * (d_rho_dr[i] + d_rho_q_dr[i]) / r;

            let v_total = v_central + v_eff_mass + v_so;

            // Overflow protection: clamp potential to physical range
            v[i] = v_total.clamp(-5000.0, 5000.0);

            // Coulomb (from precomputed potential, protons only)
            if is_proton {
                v[i] += v_coulomb[i].clamp(-500.0, 500.0);
            }
        }

        v
    }

    /// Diagonalize Hamiltonian block-by-block in Omega
    fn diagonalize_blocks(
        &self,
        v_potential: &[f64],
        wavefunctions: &[Vec<f64>],
        n_particles: usize,
        delta_pair: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_states = self.states.len();
        let mut all_eigenvalues = vec![0.0; n_states];
        let mut all_occupations = vec![0.0; n_states];

        // Temporary storage: eigenvalue/eigenvector for each block
        let mut block_eigs: Vec<(usize, f64)> = Vec::new(); // (state_idx, eigenvalue)

        for block_indices in self.omega_blocks.values() {
            let block_size = block_indices.len();
            if block_size == 0 {
                continue;
            }

            // Build block Hamiltonian
            let mut h = Mat::zeros(block_size);

            for bi in 0..block_size {
                let i = block_indices[bi];
                for bj in bi..block_size {
                    let j = block_indices[bj];

                    // Kinetic energy (diagonal for HO basis)
                    let t_ij = if i == j {
                        let s = &self.states[i];
                        self.hw_z * (s.n_z as f64 + 0.5)
                            + self.hw_perp
                                * (2.0 * s.n_perp as f64 + s.lambda.unsigned_abs() as f64 + 1.0)
                    } else {
                        0.0
                    };

                    // Potential matrix element: <i|V|j> = integral psi_i * V * psi_j * dV
                    let v_ij = self.potential_matrix_element(
                        &wavefunctions[i],
                        &wavefunctions[j],
                        v_potential,
                    );

                    let h_ij = t_ij + v_ij;
                    h.set(bi, bj, h_ij);
                    if bi != bj {
                        h.set(bj, bi, h_ij);
                    }
                }
            }

            // Diagonalize this block
            let eig = eigh_f64(&h.data, block_size)
                .expect("eigh_f64: eigendecomposition failed for deformed HFB block");

            for (bi, &eval) in eig.eigenvalues.iter().enumerate() {
                let state_idx = block_indices[bi];
                all_eigenvalues[state_idx] = eval;
                // Each Omega block has degeneracy 2 (time reversal: Omega and -Omega)
                block_eigs.push((state_idx, eval));
            }
        }

        // BCS occupations
        block_eigs.sort_by(|a, b| a.1.total_cmp(&b.1));

        let degs: Vec<f64> = (0..n_states).map(|_| 2.0).collect(); // time-reversal degeneracy

        if delta_pair > 1e-10 {
            // Physics: BCS pairing threshold — below ~10 keV pairing negligible
            // BCS with pairing — use proper number-conserving Fermi energy
            let fermi = Self::find_fermi_bcs(&block_eigs, n_particles, delta_pair);
            for (state_idx, eval) in &block_eigs {
                let eps = eval - fermi;
                let e_qp = (eps * eps + delta_pair * delta_pair).sqrt();
                let v2 = 0.5 * (1.0 - eps / e_qp);
                all_occupations[*state_idx] = v2.clamp(0.0, 1.0);
            }
        } else {
            // Sharp Fermi surface
            let mut particles_left = n_particles as f64;
            for (state_idx, _eval) in &block_eigs {
                let deg = degs[*state_idx];
                if particles_left >= deg {
                    all_occupations[*state_idx] = 1.0;
                    particles_left -= deg;
                } else if particles_left > 0.0 {
                    all_occupations[*state_idx] = particles_left / deg;
                    particles_left = 0.0;
                }
            }
        }

        (all_eigenvalues, all_occupations)
    }

    /// Find chemical potential that conserves particle number via bisection
    ///
    /// For BCS: N = sum_i deg_i * v²_i where v²_i = 0.5*(1 - eps_i/E_qp_i)
    /// eps_i = e_i - mu, E_qp_i = sqrt(eps_i² + Delta²)
    fn find_fermi_bcs(sorted_eigs: &[(usize, f64)], n_particles: usize, delta_pair: f64) -> f64 {
        if sorted_eigs.is_empty() {
            return 0.0;
        }

        let n_target = n_particles as f64;

        // Compute N(mu) = sum_i 2 * v²(e_i, mu, Delta)
        let particle_number = |mu: f64| -> f64 {
            let mut n = 0.0;
            for &(_, eval) in sorted_eigs {
                let eps = eval - mu;
                let e_qp = (eps * eps + delta_pair * delta_pair).sqrt();
                let v2 = 0.5 * (1.0 - eps / e_qp);
                n += 2.0 * v2; // degeneracy 2 (time-reversal)
            }
            n
        };

        // Bisection bounds: mu must be between lowest and highest eigenvalue
        // Safety: sorted_eigs is non-empty (empty check above returns 0.0)
        let e_min = sorted_eigs[0].1 - 50.0;
        let e_max = sorted_eigs[sorted_eigs.len() - 1].1 + 50.0;
        let mut mu_lo = e_min;
        let mut mu_hi = e_max;

        // N(mu) is monotonically increasing in mu, so bisection works
        for _ in 0..100 {
            let mu_mid = 0.5 * (mu_lo + mu_hi);
            let n_mid = particle_number(mu_mid);
            if n_mid < n_target {
                mu_lo = mu_mid;
            } else {
                mu_hi = mu_mid;
            }
            if (mu_hi - mu_lo) < 1e-10 {
                // Physics: Fermi energy bisection — MeV-scale precision
                break;
            }
        }

        0.5 * (mu_lo + mu_hi)
    }

    /// Potential matrix element: <i|V|j> via numerical integration
    fn potential_matrix_element(&self, psi_i: &[f64], psi_j: &[f64], v: &[f64]) -> f64 {
        let mut integral = 0.0;
        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                integral += psi_i[idx] * v[idx] * psi_j[idx] * dv;
            }
        }
        integral
    }

    /// Compute densities from wavefunctions and occupations
    fn compute_densities(
        &self,
        wavefunctions: &[Vec<f64>],
        occ_p: &[f64],
        occ_n: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.grid.total();
        let mut rho_p = vec![0.0; n];
        let mut rho_n = vec![0.0; n];

        for (i, _s) in self.states.iter().enumerate() {
            let occ_proton = occ_p[i] * 2.0; // 2 for time-reversal degeneracy
            let occ_neutron = occ_n[i] * 2.0;

            if occ_proton > 1e-15 || occ_neutron > 1e-15 {
                // Physics: numerical zero for occupation — below nuclear scale
                for k in 0..n {
                    let psi2 = wavefunctions[i][k] * wavefunctions[i][k];
                    rho_p[k] += occ_proton * psi2;
                    rho_n[k] += occ_neutron * psi2;
                }
            }
        }

        (rho_p, rho_n)
    }

    /// Compute total energy via direct EDF decomposition (no double-counting)
    ///
    /// E_total = E_kinetic + E_central_Skyrme + E_Coulomb + E_pairing - E_CM
    ///
    /// This avoids the E_sp - E_pot approach which is error-prone.
    /// Instead, each term is computed independently from the density.
    fn total_energy(
        &self,
        params: &[f64],
        rho_p: &[f64],
        rho_n: &[f64],
        _eigs_p: &[f64],
        _eigs_n: &[f64],
        occ_p: &[f64],
        occ_n: &[f64],
    ) -> f64 {
        // ── Kinetic energy: sum_i n_i * T_i (HO kinetic, known analytically) ──
        let mut e_kin = 0.0;
        for (i, s) in self.states.iter().enumerate() {
            let deg = 2.0; // time-reversal degeneracy
            let t_i = self.hw_z * (s.n_z as f64 + 0.5)
                + self.hw_perp * (2.0 * s.n_perp as f64 + s.lambda.unsigned_abs() as f64 + 1.0);
            e_kin += deg * (occ_p[i] + occ_n[i]) * t_i;
        }

        // ── Skyrme central EDF energy (from density, not mean-field) ──
        let t0 = params[0];
        let t3 = params[3];
        let x0 = params[4];
        let x3 = params[7];
        let alpha = params[8];

        let mut e_central = 0.0;
        let mut e_coul = 0.0;

        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                let rho = (rho_p[idx] + rho_n[idx]).max(0.0);
                let rp = rho_p[idx].max(0.0);
                let rn = rho_n[idx].max(0.0);

                // Skyrme central EDF: Eq. (6) of Chabanat et al. (1998)
                // H_0 = t0/4 * [(2+x0)*rho^2 - (2*x0+1)*(rho_p^2 + rho_n^2)]
                // H_3 = t3/24 * rho^alpha * [(2+x3)*rho^2 - (2*x3+1)*(rho_p^2+rho_n^2)]
                let h_0 =
                    t0 / 4.0 * ((2.0 + x0) * rho * rho - (1.0 + 2.0 * x0) * (rp * rp + rn * rn));
                let h_3 = t3 / 24.0
                    * rho.powf(alpha)
                    * ((2.0 + x3) * rho * rho - (1.0 + 2.0 * x3) * (rp * rp + rn * rn));

                e_central += (h_0 + h_3) * dv;

                // Coulomb: Slater exchange approximation (no direct integral)
                // E_Coulomb_exchange = -3/4 * e² * (3/pi)^(1/3) * integral rho_p^(4/3) dV
                let coul_exch = super::hfb_common::coulomb_exchange_energy_density(rp);
                e_coul += coul_exch * dv;
            }
        }

        // Coulomb direct (approximate: uniform sphere for protons)
        // E_Coulomb_direct ~ 3/5 * Z² * e² / R_charge
        let r_ch = 1.2 * (self.a as f64).powf(1.0 / 3.0);
        let e_coul_direct = 0.6 * (self.z as f64) * ((self.z as f64) - 1.0) * E2 / r_ch;
        e_coul += e_coul_direct;

        // ── BCS pairing energy ──
        // E_pair = -Delta² * G * N / 4, where G ~ 1/(A level density)
        let level_density = self.a as f64 / 28.0; // Rough: N_levels ~ A/28 near Fermi
        let e_pair_p = if self.delta_p > 1e-10 {
            // Physics: pairing gap threshold — negligible below ~10 keV
            -self.delta_p * self.delta_p * level_density / 4.0
        } else {
            0.0
        };
        let e_pair_n = if self.delta_n > 1e-10 {
            // Physics: pairing gap threshold — negligible below ~10 keV
            -self.delta_n * self.delta_n * level_density / 4.0
        } else {
            0.0
        };

        // ── Center of mass correction (1-body approximation) ──
        // E_CM = -<P²>/(2*m*A) ≈ -0.75 * hbar*omega_0
        let hw0 = 41.0 * (self.a as f64).powf(-1.0 / 3.0);
        let e_cm = 0.75 * hw0;

        e_kin + e_central + e_coul + e_pair_p + e_pair_n - e_cm
    }

    /// Quadrupole moment Q20 = integral rho(rho,z) * (2z² - rho²) dV
    fn quadrupole_moment(&self, rho_total: &[f64]) -> f64 {
        let mut q20 = 0.0;
        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];
                q20 += rho_total[idx] * (2.0 * z * z - rho * rho) * dv;
            }
        }
        q20
    }

    /// Extract beta2 deformation parameter from Q20
    fn beta2_from_q20(&self, q20: f64) -> f64 {
        let a = self.a as f64;
        let r0 = 1.2 * a.powf(1.0 / 3.0);
        // beta2 = sqrt(5*pi) * Q20 / (3 * A * R0²)
        (5.0 * PI).sqrt() * q20 / (3.0 * a * r0 * r0)
    }

    /// RMS radius from density
    fn rms_radius(&self, rho_total: &[f64]) -> f64 {
        let mut sum_r2 = 0.0;
        let mut sum_rho = 0.0;
        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];
                let r2 = rho * rho + z * z;
                sum_r2 += rho_total[idx] * r2 * dv;
                sum_rho += rho_total[idx] * dv;
            }
        }
        if sum_rho > 0.0 {
            (sum_r2 / sum_rho).sqrt()
        } else {
            0.0
        }
    }
}

/// Public API: L3 binding energy from deformed HFB
pub fn binding_energy_l3(z: usize, n: usize, params: &[f64]) -> (f64, bool, f64) {
    let mut solver = DeformedHFB::new_adaptive(z, n);
    let result = solver.solve(params);
    (result.binding_energy_mev, result.converged, result.beta2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::SLY4_PARAMS;

    #[test]
    fn test_hermite_values() {
        assert!((hermite_value(0, 1.0) - 1.0).abs() < 1e-10);
        assert!((hermite_value(1, 1.0) - 2.0).abs() < 1e-10);
        assert!((hermite_value(2, 1.0) - 2.0).abs() < 1e-10); // 4x²-2 = 2
        assert!((hermite_value(3, 1.0) - (-4.0)).abs() < 1e-10); // 8x³-12x = -4
    }

    #[test]
    fn test_deformed_basis_count() {
        let solver = DeformedHFB::new_adaptive(8, 8); // O-16
                                                      // Should have states in multiple Omega blocks
        assert!(solver.states.len() > 10);
        assert!(solver.omega_blocks.len() >= 2);
        println!(
            "O-16 deformed basis: {} states, {} Omega blocks",
            solver.states.len(),
            solver.omega_blocks.len()
        );
    }

    #[test]
    fn test_deformation_guess() {
        // Doubly magic
        assert_eq!(DeformedHFB::initial_deformation_guess(20, 20), 0.0);
        // Rare earth
        assert!(DeformedHFB::initial_deformation_guess(66, 96) > 0.2);
        // Actinide
        assert!(DeformedHFB::initial_deformation_guess(92, 146) > 0.2);
    }

    #[test]
    fn test_basis_construction_determinism() {
        // Basis construction (state enumeration, grid setup, Hermite/factorial)
        // must be bitwise deterministic.
        let build = || {
            let s = DeformedHFB::new_adaptive(8, 8);
            (
                s.states.len(),
                s.omega_blocks.clone(),
                s.grid.n_rho,
                s.grid.n_z,
                s.hw_z.to_bits(),
                s.hw_perp.to_bits(),
            )
        };
        let a = build();
        let b = build();
        assert_eq!(a.0, b.0, "state count mismatch");
        assert_eq!(a.1, b.1, "omega blocks mismatch");
        assert_eq!(a.2, b.2, "grid n_rho mismatch");
        assert_eq!(a.3, b.3, "grid n_z mismatch");
        assert_eq!(a.4, b.4, "hw_z bitwise mismatch");
        assert_eq!(a.5, b.5, "hw_perp bitwise mismatch");
    }

    #[test]
    #[ignore = "Heavy computation (~30s+); run with: cargo test -- --ignored test_deformed_hfb_runs"]
    fn test_deformed_hfb_runs() {
        // Just verify it doesn't crash; uses SLY4_PARAMS (provenance values match
        // Chabanat 1998 — previously used truncated -2488.91 etc., equivalent for this test)
        let (be, conv, beta2) = binding_energy_l3(8, 8, &SLY4_PARAMS);
        println!("O-16 deformed: B={be:.2} MeV, conv={conv}, beta2={beta2:.4}");
        // O-16 is doubly magic, should be nearly spherical
        assert!(beta2.abs() < 0.5, "O-16 should be nearly spherical");
    }
}
