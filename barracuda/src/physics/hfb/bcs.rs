// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::tolerances::{FERMI_SEARCH_MARGIN, SHARP_FILLING_THRESHOLD};

use super::SphericalHFB;

impl SphericalHFB {
    #[must_use]
    pub fn bcs_occupations_from_eigs(
        &self,
        eigenvalues: &[f64],
        num_particles: usize,
        delta: f64,
    ) -> (Vec<f64>, f64) {
        self.bcs_occupations(eigenvalues, num_particles, delta)
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
}
