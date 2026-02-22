// SPDX-License-Identifier: AGPL-3.0-only

//! Potential, field, and energy computations for the deformed HFB solver.
//!
//! Contains: Coulomb potential, Skyrme mean field, kinetic density τ,
//! spin-orbit current J, density gradients, total energy, and quadrupole moment.

use super::DeformedHFB;
use crate::physics::constants::E2;
use crate::tolerances::{
    DEFORMED_COULOMB_R_MIN, DENSITY_FLOOR, DIVISION_GUARD, PAIRING_GAP_THRESHOLD, SPIN_ORBIT_R_MIN,
};

impl DeformedHFB {
    /// Compute kinetic energy density tau(r) from wavefunctions and occupations.
    /// tau = `sum_i` `n_i` |grad `psi_i`|^2
    pub(super) fn compute_tau(&self, wavefunctions: &[Vec<f64>], occ: &[f64]) -> Vec<f64> {
        let n = self.grid.total();
        let mut tau = vec![0.0; n];

        for (i, _s) in self.states.iter().enumerate() {
            let occ_i = occ[i] * 2.0; // time-reversal degeneracy
            if occ_i < DENSITY_FLOOR {
                continue;
            }

            let psi = &wavefunctions[i];
            for i_rho in 0..self.grid.n_rho {
                for i_z in 0..self.grid.n_z {
                    let idx = self.grid.idx(i_rho, i_z);

                    let dpsi_drho = if i_rho == 0 {
                        (psi[self.grid.idx(1, i_z)] - psi[idx]) / self.grid.d_rho
                    } else if i_rho == self.grid.n_rho - 1 {
                        (psi[idx] - psi[self.grid.idx(i_rho - 1, i_z)]) / self.grid.d_rho
                    } else {
                        (psi[self.grid.idx(i_rho + 1, i_z)] - psi[self.grid.idx(i_rho - 1, i_z)])
                            / (2.0 * self.grid.d_rho)
                    };

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

    /// Compute spin-orbit density J(r) = `sum_i` `n_i` * `psi_i` * (l x s) * `psi_i`.
    /// For axially-symmetric case, the relevant component is `J_z` ~ Lambda * sigma.
    pub(super) fn compute_spin_current(&self, wavefunctions: &[Vec<f64>], occ: &[f64]) -> Vec<f64> {
        let n = self.grid.total();
        let mut j_density = vec![0.0; n];

        for (i, s) in self.states.iter().enumerate() {
            let occ_i = occ[i] * 2.0;
            if occ_i < DENSITY_FLOOR {
                continue;
            }

            let ls = f64::from(s.lambda) * f64::from(s.sigma) * 0.5;

            for k in 0..n {
                let psi2 = wavefunctions[i][k] * wavefunctions[i][k];
                j_density[k] += occ_i * ls * psi2;
            }
        }

        j_density
    }

    /// Radial derivative of density for spin-orbit potential.
    ///
    /// In cylindrical coordinates: projects (`df/d_rho`, `df/dz`) onto the radial direction.
    pub(super) fn density_radial_derivative(&self, density: &[f64]) -> Vec<f64> {
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

                let rho_coord = self.grid.rho[i_rho];
                let z_coord = self.grid.z[i_z];
                let r = rho_coord.hypot(z_coord).max(DEFORMED_COULOMB_R_MIN);
                deriv[idx] = (d_drho * rho_coord + d_dz * z_coord) / r;
            }
        }

        deriv
    }

    /// Precompute Coulomb potential on the cylindrical grid.
    ///
    /// Uses spherical monopole approximation:
    ///   `V_C`(r) = e² * [`Q_enclosed`(r) / r + `V_exterior`(r)]
    pub(super) fn compute_coulomb_potential(&self, rho_p: &[f64], v_coulomb: &mut [f64]) {
        let n = self.grid.total();

        let mut charge_shells: Vec<(f64, f64)> = Vec::with_capacity(n);
        for (i, &rp) in rho_p.iter().enumerate().take(n) {
            let i_rho = i / self.grid.n_z;
            let i_z = i % self.grid.n_z;
            let rho = self.grid.rho[i_rho];
            let z = self.grid.z[i_z];
            let r = rho.hypot(z);
            let dv = self.grid.volume_element(i_rho, i_z);
            let charge = rp.max(0.0) * dv;
            charge_shells.push((r, charge));
        }

        let mut sorted_idx: Vec<usize> = (0..n).collect();
        sorted_idx.sort_by(|&a, &b| charge_shells[a].0.total_cmp(&charge_shells[b].0));

        let total_charge: f64 = charge_shells.iter().map(|(_, c)| c).sum();
        let total_charge_over_r: f64 = charge_shells
            .iter()
            .map(|(r, c)| {
                if *r > DEFORMED_COULOMB_R_MIN {
                    c / r
                } else {
                    0.0
                }
            })
            .sum();

        let mut cum_charge = vec![0.0; n];
        let mut cum_charge_over_r = vec![0.0; n];
        {
            let mut acc_q = 0.0;
            let mut acc_qr = 0.0;
            for (k, &si) in sorted_idx.iter().enumerate() {
                acc_q += charge_shells[si].1;
                let r = charge_shells[si].0.max(DEFORMED_COULOMB_R_MIN);
                acc_qr += charge_shells[si].1 / r;
                cum_charge[k] = acc_q;
                cum_charge_over_r[k] = acc_qr;
            }
        }

        let mut rank = vec![0usize; n];
        for (k, &si) in sorted_idx.iter().enumerate() {
            rank[si] = k;
        }

        for i in 0..n {
            let r_i = charge_shells[i].0.max(DEFORMED_COULOMB_R_MIN);
            let k = rank[i];

            let q_inner = if k > 0 { cum_charge[k - 1] } else { 0.0 };
            let ext_qr = total_charge_over_r - cum_charge_over_r[k];

            v_coulomb[i] = E2.mul_add(
                q_inner / r_i + ext_qr,
                super::super::hfb_common::coulomb_exchange_slater(rho_p[i]),
            );
        }

        if total_charge < DIVISION_GUARD {
            for v in v_coulomb.iter_mut() {
                *v = 0.0;
            }
        }
    }

    /// Fast mean-field potential using precomputed Coulomb.
    ///
    /// Includes central Skyrme (t0, t3), effective mass (t1, t2),
    /// simplified spin-orbit (W0), and Coulomb.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mean_field_potential_fast(
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

        let d_rho_dr = self.density_radial_derivative(rho_total);
        let d_rho_q_dr = self.density_radial_derivative(rho_q);

        for (i, (rt, rq)) in rho_total.iter().zip(rho_q.iter()).enumerate() {
            let rho = rt.max(0.0);
            let rq = rq.max(0.0);

            let v_central = t0.mul_add(
                (1.0 + x0 / 2.0).mul_add(rho, -((0.5 + x0) * rq)),
                t3 / 12.0
                    * rho.powf(alpha)
                    * ((2.0 + alpha) * (1.0 + x3 / 2.0)).mul_add(
                        rho,
                        -((2.0 * (0.5 + x3)).mul_add(rq, alpha * (1.0 + x3 / 2.0) * rho)),
                    ),
            );

            let v_eff_mass = (t1 / 4.0).mul_add(
                (2.0 + x1).mul_add(tau_total[i], -(2.0f64.mul_add(x1, 1.0) * tau_q[i])),
                (t2 / 4.0) * (2.0 + x2).mul_add(tau_total[i], 2.0f64.mul_add(x2, 1.0) * tau_q[i]),
            );

            let i_rho = i / self.grid.n_z;
            let i_z = i % self.grid.n_z;
            let rho_coord = self.grid.rho[i_rho];
            let z_coord = self.grid.z[i_z];
            let r = rho_coord.hypot(z_coord).max(SPIN_ORBIT_R_MIN);
            let v_so = -w0 / 2.0 * (d_rho_dr[i] + d_rho_q_dr[i]) / r;

            let v_total = v_central + v_eff_mass + v_so;
            v[i] = v_total.clamp(-5000.0, 5000.0);

            if is_proton {
                v[i] += v_coulomb[i].clamp(-500.0, 500.0);
            }
        }

        v
    }

    /// Compute total energy via direct EDF decomposition.
    ///
    /// `E_total` = `E_kinetic` + `E_central_Skyrme` + `E_Coulomb` + `E_pairing` - `E_CM`
    pub(super) fn total_energy(
        &self,
        params: &[f64],
        rho_p: &[f64],
        rho_n: &[f64],
        _eigs_p: &[f64],
        _eigs_n: &[f64],
        occ_p: &[f64],
        occ_n: &[f64],
    ) -> f64 {
        let mut e_kin = 0.0;
        for (i, s) in self.states.iter().enumerate() {
            let deg = 2.0;
            let t_i = self.hw_z.mul_add(
                s.n_z as f64 + 0.5,
                self.hw_perp
                    * (2.0f64.mul_add(s.n_perp as f64, f64::from(s.lambda.unsigned_abs())) + 1.0),
            );
            e_kin += deg * (occ_p[i] + occ_n[i]) * t_i;
        }

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

                let h_0 = (t0 / 4.0)
                    * ((2.0 + x0) * rho)
                        .mul_add(rho, -(2.0f64.mul_add(x0, 1.0) * (rp * rp + rn * rn)));
                let h_3 = t3 / 24.0
                    * rho.powf(alpha)
                    * ((2.0 + x3) * rho)
                        .mul_add(rho, -(2.0f64.mul_add(x3, 1.0) * (rp * rp + rn * rn)));

                e_central += (h_0 + h_3) * dv;

                let coul_exch = super::super::hfb_common::coulomb_exchange_energy_density(rp);
                e_coul += coul_exch * dv;
            }
        }

        let r_ch = 1.2 * (self.a as f64).cbrt();
        let e_coul_direct = 0.6 * (self.z as f64) * ((self.z as f64) - 1.0) * E2 / r_ch;
        e_coul += e_coul_direct;

        let level_density = self.a as f64 / 28.0;
        let e_pair_p = if self.delta_p > PAIRING_GAP_THRESHOLD {
            -self.delta_p * self.delta_p * level_density / 4.0
        } else {
            0.0
        };
        let e_pair_n = if self.delta_n > PAIRING_GAP_THRESHOLD {
            -self.delta_n * self.delta_n * level_density / 4.0
        } else {
            0.0
        };

        let hw0 = 41.0 * (self.a as f64).powf(-1.0 / 3.0);
        let e_cm = 0.75 * hw0;

        e_kin + e_central + e_coul + e_pair_p + e_pair_n - e_cm
    }

    /// Quadrupole moment Q20 = integral rho(rho,z) * (2z² - rho²) dV
    pub(super) fn quadrupole_moment(&self, rho_total: &[f64]) -> f64 {
        let mut q20 = 0.0;
        for i_rho in 0..self.grid.n_rho {
            for i_z in 0..self.grid.n_z {
                let idx = self.grid.idx(i_rho, i_z);
                let dv = self.grid.volume_element(i_rho, i_z);
                let rho = self.grid.rho[i_rho];
                let z = self.grid.z[i_z];
                q20 += rho_total[idx] * (2.0 * z).mul_add(z, -(rho * rho)) * dv;
            }
        }
        q20
    }
}

// ═══════════════════════════════════════════════════════════════════
// Unit tests for potentials
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::super::DeformedHFB;
    use crate::provenance::SLY4_PARAMS;

    fn make_solver() -> DeformedHFB {
        DeformedHFB::new_adaptive(8, 8)
    }

    #[test]
    fn compute_tau_zero_occupation_returns_zero() {
        let solver = make_solver();
        let n = solver.grid.total();
        let wavefunctions: Vec<Vec<f64>> = solver
            .states
            .iter()
            .map(|s| solver.evaluate_wavefunction(s))
            .collect();
        let occ: Vec<f64> = vec![0.0; solver.states.len()];

        let tau = solver.compute_tau(&wavefunctions, &occ);
        assert_eq!(tau.len(), n);
        assert!(tau.iter().all(|&t| t.abs() < 1e-15));
    }

    #[test]
    fn compute_tau_single_state_positive() {
        let solver = make_solver();
        let wavefunctions: Vec<Vec<f64>> = solver
            .states
            .iter()
            .map(|s| solver.evaluate_wavefunction(s))
            .collect();
        let mut occ = vec![0.0; solver.states.len()];
        occ[0] = 1.0; // fully occupy first state

        let tau = solver.compute_tau(&wavefunctions, &occ);
        assert!(tau.iter().all(|&t| t >= 0.0), "tau must be non-negative");
        let tau_sum: f64 = tau
            .iter()
            .enumerate()
            .map(|(k, &t)| {
                t * solver
                    .grid
                    .volume_element(k / solver.grid.n_z, k % solver.grid.n_z)
            })
            .sum();
        assert!(
            tau_sum > 0.0,
            "occupied state must yield positive tau integral"
        );
    }

    #[test]
    fn compute_spin_current_zero_occupation_returns_zero() {
        let solver = make_solver();
        let wavefunctions: Vec<Vec<f64>> = solver
            .states
            .iter()
            .map(|s| solver.evaluate_wavefunction(s))
            .collect();
        let occ: Vec<f64> = vec![0.0; solver.states.len()];

        let j = solver.compute_spin_current(&wavefunctions, &occ);
        assert!(j.iter().all(|&x| x.abs() < 1e-15));
    }

    #[test]
    fn compute_spin_current_occupation_scales_linearly() {
        let solver = make_solver();
        let wavefunctions: Vec<Vec<f64>> = solver
            .states
            .iter()
            .map(|s| solver.evaluate_wavefunction(s))
            .collect();
        let mut occ = vec![0.0; solver.states.len()];
        occ[0] = 0.5;

        let j_half = solver.compute_spin_current(&wavefunctions, &occ);
        occ[0] = 1.0;
        let j_full = solver.compute_spin_current(&wavefunctions, &occ);
        for (a, b) in j_half.iter().zip(j_full.iter()) {
            if b.abs() > 1e-12 {
                assert!(
                    (a - 0.5 * b).abs() < 1e-10,
                    "J should scale with occupation"
                );
            }
        }
    }

    #[test]
    fn density_radial_derivative_constant_density_zero() {
        let solver = make_solver();
        let n = solver.grid.total();
        let density: Vec<f64> = vec![0.16; n]; // uniform saturation density

        let deriv = solver.density_radial_derivative(&density);
        assert_eq!(deriv.len(), n);
        for (i, &d) in deriv.iter().enumerate() {
            assert!(
                d.abs() < 0.01,
                "uniform density grad should be ~0 at idx {i}"
            );
        }
    }

    #[test]
    fn density_radial_derivative_gaussian_peaked_at_origin() {
        let solver = make_solver();
        let n = solver.grid.total();
        let r_scale = 3.0;
        let mut density = vec![0.0; n];
        for i_rho in 0..solver.grid.n_rho {
            for i_z in 0..solver.grid.n_z {
                let rho = solver.grid.rho[i_rho];
                let z = solver.grid.z[i_z];
                let r2 = rho * rho + z * z;
                density[solver.grid.idx(i_rho, i_z)] = (-r2 / (r_scale * r_scale)).exp();
            }
        }

        let deriv = solver.density_radial_derivative(&density);
        assert_eq!(deriv.len(), n);
        assert!(deriv.iter().all(|&d| d.is_finite()));
    }

    #[test]
    fn compute_coulomb_potential_zero_density_zeroes_output() {
        let solver = make_solver();
        let n = solver.grid.total();
        let rho_p: Vec<f64> = vec![0.0; n];
        let mut v_coulomb = vec![1.0; n];

        solver.compute_coulomb_potential(&rho_p, &mut v_coulomb);
        assert!(v_coulomb.iter().all(|&v| v.abs() < 1e-15));
    }

    #[test]
    fn compute_coulomb_potential_uniform_positive() {
        let solver = make_solver();
        let n = solver.grid.total();
        let rho_p: Vec<f64> = vec![0.01; n]; // low proton density
        let mut v_coulomb = vec![0.0; n];

        solver.compute_coulomb_potential(&rho_p, &mut v_coulomb);
        assert!(
            v_coulomb.iter().all(|&v| v >= 0.0),
            "Coulomb potential should be repulsive (positive)"
        );
    }

    #[test]
    fn mean_field_potential_fast_proton_includes_coulomb() {
        let solver = make_solver();
        let n = solver.grid.total();
        let rho: Vec<f64> = vec![0.1; n];
        let tau: Vec<f64> = vec![0.5; n];
        let j_zero: Vec<f64> = vec![0.0; n];
        let mut v_coulomb = vec![0.0; n];
        solver.compute_coulomb_potential(&rho, &mut v_coulomb);

        let v_p = solver.mean_field_potential_fast(
            &SLY4_PARAMS,
            &rho,
            &rho,
            &rho,
            true,
            &tau,
            &tau,
            &j_zero,
            &j_zero,
            &v_coulomb,
        );
        let v_n = solver.mean_field_potential_fast(
            &SLY4_PARAMS,
            &rho,
            &rho,
            &rho,
            false,
            &tau,
            &tau,
            &j_zero,
            &j_zero,
            &v_coulomb,
        );
        let coulomb_at_origin = v_coulomb[0];
        if coulomb_at_origin > 1.0 {
            assert!(
                v_p[0] > v_n[0],
                "proton potential should exceed neutron when Coulomb present"
            );
        }
    }

    #[test]
    fn mean_field_potential_fast_symmetric_matter() {
        let solver = make_solver();
        let n = solver.grid.total();
        let rho: Vec<f64> = vec![0.16; n];
        let tau: Vec<f64> = vec![0.3; n];
        let j_zero: Vec<f64> = vec![0.0; n];
        let v_coulomb = vec![0.0; n];

        let v_p = solver.mean_field_potential_fast(
            &SLY4_PARAMS,
            &rho,
            &rho,
            &rho,
            true,
            &tau,
            &tau,
            &j_zero,
            &j_zero,
            &v_coulomb,
        );
        let v_n = solver.mean_field_potential_fast(
            &SLY4_PARAMS,
            &rho,
            &rho,
            &rho,
            false,
            &tau,
            &tau,
            &j_zero,
            &j_zero,
            &v_coulomb,
        );
        assert!(v_p[0].is_finite() && v_n[0].is_finite());
        assert!(
            v_p[0].abs() < 5000.0 && v_n[0].abs() < 5000.0,
            "potential clamped"
        );
    }

    #[test]
    fn total_energy_finite_for_sensible_density() {
        let solver = make_solver();
        let n = solver.grid.total();
        let rho: Vec<f64> = vec![0.08; n];
        let n_states = solver.states.len();
        let eigs: Vec<f64> = vec![0.0; n_states];
        let mut occ = vec![0.0; n_states];
        for o in occ.iter_mut().take(solver.z.min(n_states)) {
            *o = 1.0;
        }
        let n_end = solver.z + solver.n_neutrons.min(n_states.saturating_sub(solver.z));
        for o in &mut occ[solver.z..n_end] {
            *o = 1.0;
        }

        let e = solver.total_energy(&SLY4_PARAMS, &rho, &rho, &eigs, &eigs, &occ, &occ);
        assert!(e.is_finite());
    }

    #[test]
    fn total_energy_zero_density_near_kinetic_and_cm() {
        let solver = make_solver();
        let n = solver.grid.total();
        let rho: Vec<f64> = vec![0.0; n];
        let n_states = solver.states.len();
        let eigs: Vec<f64> = vec![-5.0; n_states];
        let occ: Vec<f64> = vec![0.0; n_states];

        let e = solver.total_energy(&SLY4_PARAMS, &rho, &rho, &eigs, &eigs, &occ, &occ);
        assert!(e.is_finite());
    }

    #[test]
    fn quadrupole_moment_prolate_positive() {
        let solver = make_solver();
        let n = solver.grid.total();
        let mut density = vec![0.0; n];
        for i_rho in 0..solver.grid.n_rho {
            for i_z in 0..solver.grid.n_z {
                let rho = solver.grid.rho[i_rho];
                let z = solver.grid.z[i_z];
                density[solver.grid.idx(i_rho, i_z)] = (-rho * rho / 16.0 - z * z / 25.0).exp();
            }
        }
        let q20 = solver.quadrupole_moment(&density);
        assert!(q20 > 0.0, "prolate (elongated along z) should give Q20 > 0");
    }

    #[test]
    fn quadrupole_moment_oblate_negative() {
        let solver = DeformedHFB::new_adaptive(8, 8);
        let n = solver.grid.total();
        let mut density = vec![0.0; n];
        for i_rho in 0..solver.grid.n_rho {
            for i_z in 0..solver.grid.n_z {
                let rho = solver.grid.rho[i_rho];
                let z = solver.grid.z[i_z];
                density[solver.grid.idx(i_rho, i_z)] = (-rho * rho / 25.0 - z * z / 16.0).exp();
            }
        }
        let q20 = solver.quadrupole_moment(&density);
        assert!(q20 < 0.0, "oblate (squashed along z) should give Q20 < 0");
    }
}
