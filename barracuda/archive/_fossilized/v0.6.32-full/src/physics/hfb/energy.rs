// SPDX-License-Identifier: AGPL-3.0-or-later

use barracuda::numerical::trapz;
use std::f64::consts::PI;

use crate::tolerances::{BCS_DENSITY_SKIP, RHO_POWF_GUARD};

use super::SphericalHFB;

pub(super) struct SpeciesResult {
    eigvecs: Vec<f64>,
    n: usize,
    v2: Vec<f64>,
}

impl SpeciesResult {
    pub(super) fn new(
        _eigenvalues: Vec<f64>,
        eigvecs: Vec<f64>,
        n: usize,
        v2: Vec<f64>,
        _lambda: f64,
    ) -> Self {
        Self { eigvecs, n, v2 }
    }

    pub(super) fn empty(n_states: usize) -> Self {
        Self {
            eigvecs: vec![0.0; n_states * n_states],
            n: n_states,
            v2: vec![0.0; n_states],
        }
    }

    #[inline]
    pub(super) fn eigvec(&self, row: usize, col: usize) -> f64 {
        self.eigvecs[row * self.n + col]
    }
}

impl SphericalHFB {
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

    pub(super) fn compute_energy(
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
                let rho_safe = rho[k].max(RHO_POWF_GUARD);
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
