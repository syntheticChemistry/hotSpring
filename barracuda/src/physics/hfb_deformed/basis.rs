// SPDX-License-Identifier: AGPL-3.0-only

//! Basis construction, wavefunctions, diagonalization, and density computation
//! for the axially-deformed HFB solver.
//!
//! Includes: Nilsson basis enumeration, deformed HO wavefunctions (Hermite + Laguerre),
//! block-diagonal eigensolve by Omega, BCS occupations, and density accumulation.

use super::{DeformedHFB, DeformedState};
use crate::error::HotSpringError;
use crate::physics::hfb_common::{factorial_f64, hermite_value, Mat};
use crate::tolerances::{DENSITY_FLOOR, DIVISION_GUARD, PAIRING_GAP_THRESHOLD};
use barracuda::linalg::eigh_f64;
use barracuda::special::{gamma, laguerre};
use std::f64::consts::PI;

impl DeformedHFB {
    /// Build deformed HO basis: enumerate (`n_z`, `n_perp`, `Lambda`, `sigma`)
    pub(super) fn build_deformed_basis(&mut self, n_shells: usize) {
        self.states.clear();
        self.omega_blocks.clear();

        for n_sh in 0..n_shells {
            for n_z in 0..=n_sh {
                let remaining = n_sh - n_z;
                for n_perp in 0..=(remaining / 2) {
                    let abs_lambda = remaining - 2 * n_perp;

                    let lambdas = if abs_lambda == 0 {
                        vec![0_i32]
                    } else {
                        vec![abs_lambda as i32]
                    };

                    for &lambda in &lambdas {
                        for &sigma in &[1_i32, -1_i32] {
                            let omega_x2 = 2 * lambda + sigma;

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

        for (i, s) in self.states.iter().enumerate() {
            self.omega_blocks.entry(s.omega_x2).or_default().push(i);
        }
    }

    /// Evaluate deformed HO wavefunction on the cylindrical grid.
    /// ψ(ρ, z) = `phi_nz`(z/`b_z`) × `phi_{n_perp,|Lambda|}`(ρ/`b_perp`)
    pub(super) fn evaluate_wavefunction(&self, state: &DeformedState) -> Vec<f64> {
        let mut psi = vec![0.0; self.grid.total()];

        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];

                let phi_z = Self::hermite_oscillator(state.n_z, z, self.b_z);
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

    /// 1D harmonic oscillator along z: `H_n`(z/b) × exp(-z²/2b²) × normalization
    pub(super) fn hermite_oscillator(n: usize, z: f64, b: f64) -> f64 {
        let xi = z / b;
        let h_n = hermite_value(n, xi);
        let norm = 1.0 / (b * PI.sqrt() * f64::from(1 << n) * factorial_f64(n)).sqrt();
        norm * h_n * (-xi * xi / 2.0).exp()
    }

    /// 2D oscillator radial part: Laguerre basis.
    ///
    /// `R_{n_perp, |Lambda|}`(ρ) includes 1/sqrt(π) azimuthal normalization
    /// so that integral |R|² * 2*pi*rho * `d_rho` = 1.
    pub(super) fn laguerre_oscillator(n_perp: usize, abs_lambda: usize, rho: f64, b: f64) -> f64 {
        let eta = (rho / b).powi(2);
        let alpha = abs_lambda as f64;

        let n_fact = factorial_f64(n_perp);
        let gamma_val = gamma(n_perp as f64 + alpha + 1.0).unwrap_or(1.0);
        let norm = (n_fact / (PI * b * b * gamma_val)).sqrt();

        let lag = laguerre(n_perp, alpha, eta);
        norm * (rho / b).powi(abs_lambda as i32) * (-eta / 2.0).exp() * lag
    }

    /// Diagonalize Hamiltonian block-by-block in Omega.
    pub(super) fn diagonalize_blocks(
        &self,
        v_potential: &[f64],
        wavefunctions: &[Vec<f64>],
        n_particles: usize,
        delta_pair: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), HotSpringError> {
        let n_states = self.states.len();
        let mut all_eigenvalues = vec![0.0; n_states];
        let mut all_occupations = vec![0.0; n_states];

        let mut block_eigs: Vec<(usize, f64)> = Vec::new();

        for block_indices in self.omega_blocks.values() {
            let block_size = block_indices.len();
            if block_size == 0 {
                continue;
            }

            let mut h = Mat::zeros(block_size);

            for (bi, &i) in block_indices.iter().enumerate().take(block_size) {
                for (bj, &j) in block_indices.iter().enumerate().skip(bi) {
                    let t_ij = if i == j {
                        let s = &self.states[i];
                        self.hw_z.mul_add(
                            s.n_z as f64 + 0.5,
                            self.hw_perp
                                * (2.0f64
                                    .mul_add(s.n_perp as f64, f64::from(s.lambda.unsigned_abs()))
                                    + 1.0),
                        )
                    } else {
                        0.0
                    };

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

            let eig = eigh_f64(&h.data, block_size)?;

            for (bi, &eval) in eig.eigenvalues.iter().enumerate() {
                let state_idx = block_indices[bi];
                all_eigenvalues[state_idx] = eval;
                block_eigs.push((state_idx, eval));
            }
        }

        block_eigs.sort_by(|a, b| a.1.total_cmp(&b.1));

        let degs: Vec<f64> = (0..n_states).map(|_| 2.0).collect();

        if delta_pair > PAIRING_GAP_THRESHOLD {
            let fermi = Self::find_fermi_bcs(&block_eigs, n_particles, delta_pair);
            for (state_idx, eval) in &block_eigs {
                let eps = eval - fermi;
                let e_qp = eps.hypot(delta_pair);
                let v2 = 0.5 * (1.0 - eps / e_qp);
                all_occupations[*state_idx] = v2.clamp(0.0, 1.0);
            }
        } else {
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

        Ok((all_eigenvalues, all_occupations))
    }

    /// Find chemical potential that conserves particle number via bisection.
    pub(super) fn find_fermi_bcs(
        sorted_eigs: &[(usize, f64)],
        n_particles: usize,
        delta_pair: f64,
    ) -> f64 {
        if sorted_eigs.is_empty() {
            return 0.0;
        }

        let n_target = n_particles as f64;

        let particle_number = |mu: f64| -> f64 {
            let mut n = 0.0;
            for &(_, eval) in sorted_eigs {
                let eps = eval - mu;
                let e_qp = eps.hypot(delta_pair);
                let v2 = 0.5 * (1.0 - eps / e_qp);
                n += 2.0 * v2;
            }
            n
        };

        let e_min = sorted_eigs[0].1 - 50.0;
        let e_max = sorted_eigs[sorted_eigs.len() - 1].1 + 50.0;
        let mut mu_lo = e_min;
        let mut mu_hi = e_max;

        for _ in 0..100 {
            let mu_mid = 0.5 * (mu_lo + mu_hi);
            let n_mid = particle_number(mu_mid);
            if n_mid < n_target {
                mu_lo = mu_mid;
            } else {
                mu_hi = mu_mid;
            }
            if (mu_hi - mu_lo) < PAIRING_GAP_THRESHOLD {
                break;
            }
        }

        0.5 * (mu_lo + mu_hi)
    }

    /// Potential matrix element: <i|V|j> via numerical integration.
    pub(super) fn potential_matrix_element(&self, psi_i: &[f64], psi_j: &[f64], v: &[f64]) -> f64 {
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

    /// Compute densities from wavefunctions and occupations.
    pub(super) fn compute_densities(
        &self,
        wavefunctions: &[Vec<f64>],
        occ_p: &[f64],
        occ_n: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.grid.total();
        let mut rho_p = vec![0.0; n];
        let mut rho_n = vec![0.0; n];

        for (i, _s) in self.states.iter().enumerate() {
            let occ_proton = occ_p[i] * 2.0;
            let occ_neutron = occ_n[i] * 2.0;

            if occ_proton > DENSITY_FLOOR || occ_neutron > DENSITY_FLOOR {
                for k in 0..n {
                    let psi2 = wavefunctions[i][k] * wavefunctions[i][k];
                    rho_p[k] += occ_proton * psi2;
                    rho_n[k] += occ_neutron * psi2;
                }
            }
        }

        (rho_p, rho_n)
    }

    /// Renormalize a set of wavefunctions on the grid so that each satisfies
    /// integral |psi|² dV = 1.
    pub(super) fn renormalize_wavefunctions(&self, wavefunctions: &mut [Vec<f64>]) {
        let n_grid = self.grid.total();
        for psi in wavefunctions.iter_mut() {
            let norm2: f64 = (0..n_grid)
                .map(|k| {
                    let i_rho = k / self.grid.n_z;
                    let i_z = k % self.grid.n_z;
                    psi[k] * psi[k] * self.grid.volume_element(i_rho, i_z)
                })
                .sum();
            if norm2 > DIVISION_GUARD {
                let scale = 1.0 / norm2.sqrt();
                for v in psi.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }
}
