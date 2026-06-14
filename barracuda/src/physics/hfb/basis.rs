// SPDX-License-Identifier: AGPL-3.0-or-later

use barracuda::numerical::gradient_1d;
use barracuda::special::{gamma, laguerre};
use std::collections::HashMap;

use super::super::constants::{HBAR_C, M_NUCLEON};
use super::SphericalHFB;

#[derive(Debug, Clone)]
pub(in crate::physics) struct BasisState {
    pub n: usize,
    pub l: usize,
    pub j: f64,
    pub deg: usize,
}

impl SphericalHFB {
    pub(super) fn build(z: usize, n: usize, n_shells: usize, r_max: f64, n_grid: usize) -> Self {
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

    pub(super) fn build_basis(&mut self, n_shells: usize) {
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

    pub(super) fn compute_wavefunctions(&mut self) {
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
