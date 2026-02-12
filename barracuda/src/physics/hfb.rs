//! Spherical Hartree-Fock-Bogoliubov solver
//!
//! Port of: `control/surrogate/nuclear-eos/wrapper/skyrme_hfb.py`
//!
//! Uses:
//!   - `barracuda::special::{gamma, laguerre}` for HO basis wavefunctions
//!   - `barracuda::numerical::{trapz, gradient_1d}` for integrals
//!   - `nalgebra::SymmetricEigen` for diagonalization (gap in barracuda::linalg)

use super::constants::*;
use super::semf::semf_binding_energy;
use barracuda::numerical::{gradient_1d, trapz};
use barracuda::special::{gamma, laguerre};
use nalgebra::{DMatrix, SymmetricEigen};
use std::f64::consts::PI;

/// Spherical HFB solver state
#[derive(Debug)]
struct BasisState {
    n: usize,
    l: usize,
    j: f64,
    deg: usize,
}

pub struct SphericalHFB {
    z: usize,
    n_neutrons: usize,
    a: usize,
    r: Vec<f64>,
    dr: f64,
    b: f64, // HO parameter
    states: Vec<BasisState>,
    wf: Vec<Vec<f64>>, // [n_states][n_grid]
}

impl SphericalHFB {
    pub fn new(z: usize, n: usize, n_shells: usize, r_max: f64, n_grid: usize) -> Self {
        let a = z + n;
        let dr = r_max / n_grid as f64;
        let r: Vec<f64> = (1..=n_grid).map(|i| i as f64 * dr).collect();
        let b = HBAR_C / (M_NUCLEON * 41.0 * (a as f64).powf(-1.0 / 3.0)).sqrt();

        let mut hfb = SphericalHFB {
            z,
            n_neutrons: n,
            a,
            r,
            dr,
            b,
            states: Vec::new(),
            wf: Vec::new(),
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
    }

    fn compute_wavefunctions(&mut self) {
        self.wf = self
            .states
            .iter()
            .map(|s| Self::ho_radial(s.n, s.l, &self.r, self.b))
            .collect();
    }

    /// HO radial wavefunction R_{nl}(r)
    /// Uses `barracuda::special::{gamma, laguerre}` for Γ function and L_n^α
    fn ho_radial(n: usize, l: usize, r: &[f64], b: f64) -> Vec<f64> {
        let alpha = l as f64 + 0.5;

        // Normalization: sqrt(2 * n! / (b³ * Γ(n + l + 3/2)))
        let n_fact = barracuda::special::factorial(n);
        let gamma_val = gamma(n as f64 + l as f64 + 1.5);
        let norm = (2.0 * n_fact / (b.powi(3) * gamma_val)).abs().sqrt();

        r.iter()
            .map(|&ri| {
                let xi = (ri / b).powi(2);
                let lag = laguerre(n, alpha, xi);
                norm * (ri / b).powi(l as i32) * (-xi / 2.0).exp() * lag
            })
            .collect()
    }

    /// Self-consistent HF iteration
    pub fn solve(&self, params: &[f64], max_iter: usize, tol: f64, mixing: f64) -> HFBResult {
        let t0 = params[0];
        let t3 = params[3];
        let x2 = params[6];
        let alpha = params[8];
        let w0 = params[9];
        let theta = 3.0 * params[1] + params[2] * (5.0 + 4.0 * x2);

        let n_states = self.states.len();
        let n_grid = self.r.len();

        // Initial density: uniform sphere
        let r_nuc = 1.2 * (self.a as f64).powf(1.0 / 3.0);
        let mut rho: Vec<f64> = self
            .r
            .iter()
            .map(|&ri| {
                if ri < r_nuc {
                    (3.0 * self.a as f64 / (4.0 * PI * r_nuc.powi(3))).max(1e-12)
                } else {
                    1e-12
                }
            })
            .collect();

        let mut e_prev = 1e10_f64;
        let mut converged = false;
        let mut last_de = 0.0_f64;
        let mut last_iter = 0;

        for it in 0..max_iter {
            // Build Hamiltonian
            let mut h = DMatrix::zeros(n_states, n_states);

            // Skyrme mean-field potential
            let u_sky: Vec<f64> = rho
                .iter()
                .map(|&rho_i| {
                    (3.0 / 4.0) * t0 * rho_i
                        + (1.0 / 8.0) * t3 * (alpha + 2.0) * rho_i.powf(alpha + 1.0)
                        + theta * (3.0 / 10.0)
                            * (3.0 * PI * PI * rho_i / 2.0).powf(2.0 / 3.0)
                            * rho_i
                            / 8.0
                })
                .collect();

            // Coulomb (uniform sphere approx)
            let r_ch = 1.2 * (self.a as f64).powf(1.0 / 3.0);
            let v_c: Vec<f64> = self
                .r
                .iter()
                .map(|&rk| {
                    if rk < r_ch {
                        E2 * self.z as f64 * (3.0 - (rk / r_ch).powi(2)) / (2.0 * r_ch)
                    } else {
                        E2 * self.z as f64 / rk
                    }
                })
                .collect();

            let u_total: Vec<f64> = u_sky
                .iter()
                .zip(v_c.iter())
                .map(|(&us, &vc)| us + vc * self.z as f64 / self.a as f64)
                .collect();

            // Diagonal elements
            let hw = 41.0 * (self.a as f64).powf(-1.0 / 3.0);
            for i in 0..n_states {
                let n_sh = 2 * self.states[i].n + self.states[i].l;
                h[(i, i)] = hw * (n_sh as f64 + 1.5);

                // Skyrme + Coulomb
                let integrand: Vec<f64> = (0..n_grid)
                    .map(|k| self.wf[i][k].powi(2) * u_total[k] * self.r[k].powi(2))
                    .collect();
                h[(i, i)] += trapz(&integrand, &self.r).unwrap_or(0.0);

                // Spin-orbit
                if w0 != 0.0 && self.states[i].l > 0 {
                    let j = self.states[i].j;
                    let l_f = self.states[i].l as f64;
                    let ls = (j * (j + 1.0) - l_f * (l_f + 1.0) - 0.75) / 2.0;
                    let drho = gradient_1d(&rho, self.dr);
                    let drho_r: Vec<f64> = drho
                        .iter()
                        .zip(self.r.iter())
                        .map(|(&d, &ri)| d / ri.max(0.1))
                        .collect();
                    let so_integ: Vec<f64> = (0..n_grid)
                        .map(|k| self.wf[i][k].powi(2) * drho_r[k] * self.r[k].powi(2))
                        .collect();
                    h[(i, i)] += w0 * ls * trapz(&so_integ, &self.r).unwrap_or(0.0);
                }
            }

            // Off-diagonal (same l, j block)
            for i in 0..n_states {
                for j_idx in (i + 1)..n_states {
                    if self.states[i].l == self.states[j_idx].l
                        && (self.states[i].j - self.states[j_idx].j).abs() < 0.01
                    {
                        let integ: Vec<f64> = (0..n_grid)
                            .map(|k| {
                                self.wf[i][k] * self.wf[j_idx][k] * u_total[k]
                                    * self.r[k].powi(2)
                            })
                            .collect();
                        let val = trapz(&integ, &self.r).unwrap_or(0.0);
                        h[(i, j_idx)] = val;
                        h[(j_idx, i)] = val;
                    }
                }
            }

            // Diagonalize using nalgebra::SymmetricEigen
            // (Gap in barracuda::linalg — documented for evolution)
            let eigen = SymmetricEigen::new(h.clone());
            let eigenvalues = eigen.eigenvalues;
            let eigvecs = eigen.eigenvectors;

            // Fill nucleons sorted by eigenvalue
            let mut idx: Vec<usize> = (0..n_states).collect();
            idx.sort_by(|&a, &b| {
                eigenvalues[a]
                    .partial_cmp(&eigenvalues[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut occ = vec![0.0_f64; n_states];
            let mut nucleons_left = self.a;
            for &i in &idx {
                let fill = nucleons_left.min(self.states[i].deg);
                occ[i] = fill as f64;
                nucleons_left -= fill;
                if nucleons_left == 0 {
                    break;
                }
            }

            // New density from eigenstates
            let mut rho_new = vec![1e-12_f64; n_grid];
            for i in 0..n_states {
                if occ[i] > 0.0 {
                    let mut phi = vec![0.0_f64; n_grid];
                    for j in 0..n_states {
                        let c = eigvecs[(j, i)];
                        for k in 0..n_grid {
                            phi[k] += c * self.wf[j][k];
                        }
                    }
                    for k in 0..n_grid {
                        rho_new[k] += occ[i] * phi[k].powi(2) / (4.0 * PI);
                    }
                }
            }

            // Mix densities
            for k in 0..n_grid {
                rho[k] = (mixing * rho_new[k] + (1.0 - mixing) * rho[k]).max(1e-12);
            }

            // Total energy
            let e_sp: f64 = (0..n_states).map(|i| occ[i] * eigenvalues[i]).sum();
            let v_integrand: Vec<f64> = (0..n_grid)
                .map(|k| u_sky[k] * rho[k] * 4.0 * PI * self.r[k].powi(2))
                .collect();
            let v_pot = trapz(&v_integrand, &self.r).unwrap_or(0.0);
            let e_total = e_sp - 0.5 * v_pot;

            last_de = (e_total - e_prev).abs();
            last_iter = it + 1;

            if last_de < tol && it > 5 {
                converged = true;
                break;
            }
            e_prev = e_total;
        }

        let binding = if e_prev < 0.0 {
            -e_prev
        } else {
            e_prev.abs()
        };

        HFBResult {
            binding_energy_mev: binding,
            converged,
            iterations: last_iter,
            delta_e: last_de,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HFBResult {
    pub binding_energy_mev: f64,
    pub converged: bool,
    pub iterations: usize,
    pub delta_e: f64,
}

/// Hybrid binding energy: HFB for medium nuclei, SEMF otherwise
/// Port of: `control/surrogate/nuclear-eos/wrapper/skyrme_hfb.py::binding_energy_l2`
pub fn binding_energy_l2(z: usize, n: usize, params: &[f64]) -> (f64, bool) {
    let a = z + n;
    if a < 56 || a > 132 {
        // SEMF for light and heavy nuclei
        (semf_binding_energy(z, n, params), true)
    } else {
        // Spherical HFB for medium nuclei
        let hfb = SphericalHFB::new(z, n, 12, 15.0, 150);
        let result = hfb.solve(params, 200, 0.1, 0.3);
        (result.binding_energy_mev, result.converged)
    }
}

