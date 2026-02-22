// SPDX-License-Identifier: AGPL-3.0-only

//! Coulomb, Skyrme, and effective kinetic potentials for spherical HFB.
//!
//! Extracted from the monolithic `hfb.rs` for single-responsibility: this module
//! handles all potential/mean-field computations. The solver in `mod.rs` calls
//! these via `SphericalHFB` methods.

use super::SphericalHFB;
use crate::tolerances::{COULOMB_R_MIN, DENSITY_FLOOR, RHO_POWF_GUARD, SPIN_ORBIT_R_MIN};
use barracuda::numerical::{gradient_1d, trapz};
use barracuda::ops::grid::compute_ls_factor;
use std::f64::consts::PI;

use super::super::constants::{E2, HBAR2_2M};
use super::super::hfb_common::Mat;

impl SphericalHFB {
    /// Coulomb direct potential from proton density (spherical Poisson).
    pub(super) fn coulomb_direct(&self, rho_p: &[f64]) -> Vec<f64> {
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

    /// Coulomb exchange (Slater approximation).
    pub(super) fn coulomb_exchange(rho_p: &[f64]) -> Vec<f64> {
        rho_p
            .iter()
            .map(|&rp| super::super::hfb_common::coulomb_exchange_slater(rp))
            .collect()
    }

    /// Skyrme mean-field potential for one species.
    pub(super) fn skyrme_potential(
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

                let u_t0 = t0 * (1.0 + x0 / 2.0).mul_add(rho, -((0.5 + x0) * rho_q));

                let rho_alpha = rho_safe.powf(alpha);
                let rho_alpha_m1 = if rho > DENSITY_FLOOR {
                    rho_safe.powf(alpha - 1.0)
                } else {
                    0.0
                };
                let sum_rho2 = rho_p[k].mul_add(rho_p[k], rho_n[k].powi(2));

                let u_t3 = (t3 / 12.0)
                    * ((1.0 + x3 / 2.0).mul_add(
                        (alpha + 2.0) * rho_alpha * rho,
                        -(0.5 + x3)
                            * (alpha * rho_alpha_m1).mul_add(sum_rho2, 2.0 * rho_alpha * rho_q),
                    ));

                u_t0 + u_t3
            })
            .collect()
    }

    /// Effective kinetic energy matrix `T_eff` (Skyrme t₁/t₂ effective mass terms).
    pub(super) fn build_t_eff(
        &self,
        rho_p: &[f64],
        rho_n: &[f64],
        is_proton: bool,
        params: &[f64],
    ) -> Mat {
        let (t1, t2) = (params[1], params[2]);
        let (x1, x2) = (params[5], params[6]);

        let c0t = 0.25 * t1.mul_add(1.0 + x1 / 2.0, t2 * (1.0 + x2 / 2.0));
        let c1n = 0.25 * t1.mul_add(0.5 + x1, -(t2 * (0.5 + x2)));

        let rho_q: &[f64] = if is_proton { rho_p } else { rho_n };
        let f_q: Vec<f64> = (0..self.nr)
            .map(|k| {
                let fval = HBAR2_2M + c0t * (rho_p[k] + rho_n[k]) - c1n * rho_q[k];
                fval.max(HBAR2_2M * 0.3)
            })
            .collect();

        let mut t_eff = Mat::zeros(self.n_states);

        for indices in self.lj_blocks.values() {
            let l_val = self.states[indices[0]].l;
            let ll1 = (l_val * (l_val + 1)) as f64;

            for (ii, &idx_i) in indices.iter().enumerate() {
                for &idx_j in indices.iter().skip(ii) {
                    let integrand: Vec<f64> = (0..self.nr)
                        .map(|k| {
                            f_q[k]
                                * self.dwf[idx_i][k].mul_add(
                                    self.dwf[idx_j][k] * self.r[k].powi(2),
                                    ll1 * self.wf[idx_i][k] * self.wf[idx_j][k],
                                )
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

    /// Build full Hamiltonian including spin-orbit, adding to given matrix.
    ///
    /// Factored from the SCF loop and `build_hamiltonian` to avoid duplication.
    pub(super) fn add_potential_to_hamiltonian(
        &self,
        h: &mut Mat,
        u_total: &[f64],
        rho_p: &[f64],
        rho_n: &[f64],
        w0: f64,
    ) {
        let nr = self.nr;
        let ns = self.n_states;

        for i in 0..ns {
            let integ: Vec<f64> = (0..nr)
                .map(|k| self.wf[i][k].powi(2) * u_total[k] * self.r[k].powi(2))
                .collect();
            h.add(i, i, trapz(&integ, &self.r).unwrap_or(0.0));

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

        for indices in self.lj_blocks.values() {
            for (ii, &idx_i) in indices.iter().enumerate() {
                for &idx_j in indices.iter().skip(ii + 1) {
                    let integ: Vec<f64> = (0..nr)
                        .map(|k| {
                            self.wf[idx_i][k] * self.wf[idx_j][k] * u_total[k] * self.r[k].powi(2)
                        })
                        .collect();
                    let val = trapz(&integ, &self.r).unwrap_or(0.0);
                    h.add(idx_i, idx_j, val);
                    h.add(idx_j, idx_i, val);
                }
            }
        }
    }
}
