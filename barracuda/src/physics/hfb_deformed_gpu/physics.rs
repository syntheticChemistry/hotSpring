// SPDX-License-Identifier: AGPL-3.0-only

//! CPU physics kernels (Rayon-parallelized) for the deformed HFB GPU solver.
//!
//! Contains: wavefunctions, tau, spin-orbit, Coulomb, mean-field potential,
//! densities, Broyden mixing, total energy, and quadrupole moment.

use super::types::NucleusSetup;
use crate::physics::constants::E2;
use crate::physics::hfb_common::{factorial_f64, hermite_value};
use crate::tolerances::{
    BROYDEN_HISTORY, DEFORMED_COULOMB_R_MIN, DENSITY_FLOOR, DIVISION_GUARD, PAIRING_GAP_THRESHOLD,
    SPIN_ORBIT_R_MIN,
};
use barracuda::special::{gamma, laguerre};
use rayon::prelude::*;
use std::f64::consts::PI;

pub(super) fn precompute_wavefunctions(setup: &NucleusSetup) -> Vec<f64> {
    let n_grid = setup.n_grid;
    let n_states = setup.states.len();

    let wf_per_state: Vec<Vec<f64>> = (0..n_states)
        .into_par_iter()
        .map(|si| {
            let s = &setup.states[si];
            let mut wf = vec![0.0f64; n_grid];
            for i_rho in 0..setup.n_rho {
                for i_z in 0..setup.n_z {
                    let rho = (i_rho + 1) as f64 * setup.d_rho;
                    let z_c = (i_z as f64 + 0.5).mul_add(setup.d_z, setup.z_min);

                    let xi = z_c / setup.b_z;
                    let h_n = hermite_value(s.n_z as usize, xi);
                    let norm_z = 1.0
                        / (setup.b_z
                            * PI.sqrt()
                            * (1u64 << s.n_z) as f64
                            * factorial_f64(s.n_z as usize))
                        .sqrt();
                    let phi_z = norm_z * h_n * (-xi * xi / 2.0).exp();

                    let eta = (rho / setup.b_perp).powi(2);
                    let alpha = f64::from(s.abs_lambda);
                    let nf = factorial_f64(s.n_perp as usize);
                    let gv = gamma(f64::from(s.n_perp) + alpha + 1.0).unwrap_or(1.0);
                    let norm_rho = (nf / (PI * setup.b_perp * setup.b_perp * gv)).sqrt();
                    let lag = laguerre(s.n_perp as usize, alpha, eta);
                    let phi_rho = norm_rho
                        * (rho / setup.b_perp).powi(s.abs_lambda as i32)
                        * (-eta / 2.0).exp()
                        * lag;

                    wf[setup.grid_idx(i_rho, i_z)] = phi_z * phi_rho;
                }
            }
            let norm2: f64 = (0..n_grid)
                .map(|k| {
                    let ir = k / setup.n_z;
                    wf[k] * wf[k] * setup.volume_element(ir)
                })
                .sum();
            if norm2 > DIVISION_GUARD {
                let sc = 1.0 / norm2.sqrt();
                for v in &mut wf {
                    *v *= sc;
                }
            }
            wf
        })
        .collect();

    let mut wavefunctions = vec![0.0; n_states * n_grid];
    for (si, wf) in wf_per_state.into_iter().enumerate() {
        wavefunctions[si * n_grid..(si + 1) * n_grid].copy_from_slice(&wf);
    }
    wavefunctions
}

pub(super) fn compute_tau_rayon(
    setup: &NucleusSetup,
    wf: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ng = setup.n_grid;
    let n_rho = setup.n_rho;
    let n_z = setup.n_z;
    let d_rho = setup.d_rho;
    let d_z = setup.d_z;

    let tau: Vec<(f64, f64)> = (0..ng)
        .into_par_iter()
        .map(|k| {
            let i_rho = k / n_z;
            let i_z = k % n_z;
            let mut tp = 0.0;
            let mut tn = 0.0;
            for (si, _s) in setup.states.iter().enumerate() {
                let op = occ_p[si] * 2.0;
                let on = occ_n[si] * 2.0;
                if op < DENSITY_FLOOR && on < DENSITY_FLOOR {
                    continue;
                }
                let base = si * ng;
                let dpsi_drho = if i_rho == 0 {
                    (wf[base + (n_z + i_z)] - wf[base + k]) / d_rho
                } else if i_rho == n_rho - 1 {
                    (wf[base + k] - wf[base + ((i_rho - 1) * n_z + i_z)]) / d_rho
                } else {
                    (wf[base + ((i_rho + 1) * n_z + i_z)] - wf[base + ((i_rho - 1) * n_z + i_z)])
                        / (2.0 * d_rho)
                };
                let dpsi_dz = if i_z == 0 {
                    (wf[base + (i_rho * n_z + 1)] - wf[base + k]) / d_z
                } else if i_z == n_z - 1 {
                    (wf[base + k] - wf[base + (i_rho * n_z + i_z - 1)]) / d_z
                } else {
                    (wf[base + (i_rho * n_z + i_z + 1)] - wf[base + (i_rho * n_z + i_z - 1)])
                        / (2.0 * d_z)
                };
                let grad2 = dpsi_drho * dpsi_drho + dpsi_dz * dpsi_dz;
                tp += op * grad2;
                tn += on * grad2;
            }
            (tp, tn)
        })
        .collect();

    let (mut tau_p, mut tau_n) = (vec![0.0; ng], vec![0.0; ng]);
    for (k, &(tp, tn)) in tau.iter().enumerate() {
        tau_p[k] = tp;
        tau_n[k] = tn;
    }
    (tau_p, tau_n)
}

pub(super) fn compute_spin_current(
    setup: &NucleusSetup,
    wf: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ng = setup.n_grid;
    let (mut jp, mut jn) = (vec![0.0; ng], vec![0.0; ng]);
    for (si, s) in setup.states.iter().enumerate() {
        let op = occ_p[si] * 2.0;
        let on = occ_n[si] * 2.0;
        if op < DENSITY_FLOOR && on < DENSITY_FLOOR {
            continue;
        }
        let ls = f64::from(s.lambda) * f64::from(s.sigma) * 0.5;
        for k in 0..ng {
            let psi2 = wf[si * ng + k] * wf[si * ng + k];
            jp[k] += op * ls * psi2;
            jn[k] += on * ls * psi2;
        }
    }
    (jp, jn)
}

pub(super) fn compute_coulomb_cpu(setup: &NucleusSetup, rho_p: &[f64], vc: &mut [f64]) {
    let ng = setup.n_grid;
    let mut shells: Vec<(f64, f64)> = Vec::with_capacity(ng);
    for (i, &rp) in rho_p.iter().enumerate().take(ng) {
        let ir = i / setup.n_z;
        let iz = i % setup.n_z;
        let rho = (ir + 1) as f64 * setup.d_rho;
        let z = (iz as f64 + 0.5).mul_add(setup.d_z, setup.z_min);
        let r = rho.hypot(z);
        let dv = setup.volume_element(ir);
        shells.push((r, rp.max(0.0) * dv));
    }

    let mut sidx: Vec<usize> = (0..ng).collect();
    sidx.sort_by(|&a, &b| shells[a].0.total_cmp(&shells[b].0));

    let total_qr: f64 = shells
        .iter()
        .map(|(r, c)| {
            if *r > DEFORMED_COULOMB_R_MIN {
                c / r
            } else {
                0.0
            }
        })
        .sum();
    let mut cum_q = vec![0.0; ng];
    let mut cum_qr = vec![0.0; ng];
    let (mut aq, mut aqr) = (0.0, 0.0);
    for (k, &si) in sidx.iter().enumerate() {
        aq += shells[si].1;
        aqr += if shells[si].0 > DEFORMED_COULOMB_R_MIN {
            shells[si].1 / shells[si].0
        } else {
            0.0
        };
        cum_q[k] = aq;
        cum_qr[k] = aqr;
    }
    let mut rank = vec![0usize; ng];
    for (k, &si) in sidx.iter().enumerate() {
        rank[si] = k;
    }

    let tc: f64 = shells.iter().map(|(_, c)| c).sum();
    for i in 0..ng {
        let ri = shells[i].0.max(DEFORMED_COULOMB_R_MIN);
        let k = rank[i];
        let qi = if k > 0 { cum_q[k - 1] } else { 0.0 };
        let ext = total_qr - cum_qr[k];
        vc[i] = E2.mul_add(
            qi / ri + ext,
            crate::physics::hfb_common::coulomb_exchange_slater(rho_p[i]),
        );
    }
    if tc < DIVISION_GUARD {
        for v in vc.iter_mut() {
            *v = 0.0;
        }
    }
}

pub(super) fn density_radial_derivative(setup: &NucleusSetup, density: &[f64]) -> Vec<f64> {
    let ng = setup.n_grid;
    (0..ng)
        .into_par_iter()
        .map(|k| {
            let ir = k / setup.n_z;
            let iz = k % setup.n_z;
            let d_dr = if ir == 0 {
                (density[setup.n_z + iz] - density[k]) / setup.d_rho
            } else if ir == setup.n_rho - 1 {
                (density[k] - density[(ir - 1) * setup.n_z + iz]) / setup.d_rho
            } else {
                (density[(ir + 1) * setup.n_z + iz] - density[(ir - 1) * setup.n_z + iz])
                    / (2.0 * setup.d_rho)
            };
            let d_dz = if iz == 0 {
                (density[ir * setup.n_z + 1] - density[k]) / setup.d_z
            } else if iz == setup.n_z - 1 {
                (density[k] - density[ir * setup.n_z + iz - 1]) / setup.d_z
            } else {
                (density[ir * setup.n_z + iz + 1] - density[ir * setup.n_z + iz - 1])
                    / (2.0 * setup.d_z)
            };
            let rho_c = (ir + 1) as f64 * setup.d_rho;
            let z_c = (iz as f64 + 0.5).mul_add(setup.d_z, setup.z_min);
            let r = rho_c.hypot(z_c).max(DEFORMED_COULOMB_R_MIN);
            (d_dr * rho_c + d_dz * z_c) / r
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mean_field_potential(
    setup: &NucleusSetup,
    params: &[f64],
    rho_p: &[f64],
    rho_n: &[f64],
    rho_total: &[f64],
    is_proton: bool,
    tau_p: &[f64],
    tau_n: &[f64],
    v_coulomb: &[f64],
    d_rho_dr: &[f64],
) -> Vec<f64> {
    let ng = setup.n_grid;
    let (t0, t1, t2, t3) = (params[0], params[1], params[2], params[3]);
    let (x0, x1, x2, x3) = (params[4], params[5], params[6], params[7]);
    let (alpha, w0) = (params[8], params[9]);
    let rho_q: &[f64] = if is_proton { rho_p } else { rho_n };
    let tau_q: &[f64] = if is_proton { tau_p } else { tau_n };
    let d_rq_dr = density_radial_derivative(setup, rho_q);

    (0..ng)
        .into_par_iter()
        .map(|i| {
            let rho = rho_total[i].max(0.0);
            let rq = rho_q[i].max(0.0);
            let tau_tot = tau_p[i] + tau_n[i];

            let vc = t0.mul_add(
                (1.0 + x0 / 2.0).mul_add(rho, -((0.5 + x0) * rq)),
                t3 / 12.0
                    * rho.powf(alpha)
                    * ((2.0 + alpha) * (1.0 + x3 / 2.0)).mul_add(
                        rho,
                        -(2.0 * (0.5 + x3)).mul_add(rq, alpha * (1.0 + x3 / 2.0) * rho),
                    ),
            );

            let ve = t1.mul_add(
                1.0 / 4.0 * (2.0 + x1).mul_add(tau_tot, -(2.0_f64.mul_add(x1, 1.0) * tau_q[i])),
                t2 / 4.0 * (2.0 + x2).mul_add(tau_tot, 2.0_f64.mul_add(x2, 1.0) * tau_q[i]),
            );

            let ir = i / setup.n_z;
            let iz = i % setup.n_z;
            let rc = (ir + 1) as f64 * setup.d_rho;
            let zc = (iz as f64 + 0.5).mul_add(setup.d_z, setup.z_min);
            let r = rc.hypot(zc).max(SPIN_ORBIT_R_MIN);
            let vso = -w0 / 2.0 * (d_rho_dr[i] + d_rq_dr[i]) / r;

            let mut v = (vc + ve + vso).clamp(-5000.0, 5000.0);
            if is_proton {
                v += v_coulomb[i].clamp(-500.0, 500.0);
            }
            v
        })
        .collect()
}

pub(super) fn compute_densities_rayon(
    setup: &NucleusSetup,
    wf: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let ng = setup.n_grid;
    let rho: Vec<(f64, f64)> = (0..ng)
        .into_par_iter()
        .map(|k| {
            let (mut rp, mut rn) = (0.0, 0.0);
            for (i, _) in setup.states.iter().enumerate() {
                let psi2 = wf[i * ng + k] * wf[i * ng + k];
                rp += occ_p[i] * 2.0 * psi2;
                rn += occ_n[i] * 2.0 * psi2;
            }
            (rp, rn)
        })
        .collect();

    let (mut rho_p, mut rho_n) = (vec![0.0; ng], vec![0.0; ng]);
    for (k, &(rp, rn)) in rho.iter().enumerate() {
        rho_p[k] = rp;
        rho_n[k] = rn;
    }
    (rho_p, rho_n)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn density_mixing(
    rho_p: &mut [f64],
    rho_n: &mut [f64],
    new_p: &[f64],
    new_n: &[f64],
    iteration: usize,
    warmup: usize,
    ng: usize,
    _vd: usize,
    broyden_dfs: &mut Vec<Vec<f64>>,
    broyden_dus: &mut Vec<Vec<f64>>,
    prev_r: &mut Option<Vec<f64>>,
    prev_u: &mut Option<Vec<f64>>,
) {
    let vd = 2 * ng;
    if iteration < warmup {
        let am = if iteration == 0 { 1.0 } else { 0.5 };
        for i in 0..ng {
            rho_p[i] = (1.0_f64 - am).mul_add(rho_p[i], am * new_p[i]);
            rho_n[i] = (1.0_f64 - am).mul_add(rho_n[i], am * new_n[i]);
        }
    } else {
        let am = 0.4;
        let input: Vec<f64> = rho_p.iter().chain(rho_n.iter()).copied().collect();
        let output: Vec<f64> = new_p.iter().chain(new_n.iter()).copied().collect();
        let residual: Vec<f64> = output.iter().zip(&input).map(|(&o, &i)| o - i).collect();

        if let (Some(pr), Some(pu)) = (prev_r.as_ref(), prev_u.as_ref()) {
            let df: Vec<f64> = residual.iter().zip(pr).map(|(&r, &p)| r - p).collect();
            let du: Vec<f64> = input.iter().zip(pu).map(|(&u, &p)| u - p).collect();
            if broyden_dfs.len() >= BROYDEN_HISTORY {
                broyden_dfs.remove(0);
                broyden_dus.remove(0);
            }
            broyden_dfs.push(df);
            broyden_dus.push(du);
        }

        let mut mixed = vec![0.0; vd];
        for (m, (inp, res)) in input.iter().zip(residual.iter()).enumerate().take(vd) {
            mixed[m] = inp + am * res;
        }
        for m in 0..broyden_dfs.len() {
            let dfr: f64 = broyden_dfs[m]
                .iter()
                .zip(&residual)
                .map(|(&a, &b)| a * b)
                .sum();
            let dfd: f64 = broyden_dfs[m].iter().map(|&a| a * a).sum();
            if dfd > DIVISION_GUARD {
                let g = dfr / dfd;
                for i in 0..vd {
                    mixed[i] -= g * (broyden_dus[m][i] + am * broyden_dfs[m][i]);
                }
            }
        }
        *prev_r = Some(residual);
        *prev_u = Some(input);
        rho_p.copy_from_slice(&mixed[..ng]);
        rho_n.copy_from_slice(&mixed[ng..]);
        for v in rho_p.iter_mut() {
            *v = v.max(0.0);
        }
        for v in rho_n.iter_mut() {
            *v = v.max(0.0);
        }
    }
}

pub(super) fn total_energy(
    setup: &NucleusSetup,
    params: &[f64],
    rho_p: &[f64],
    rho_n: &[f64],
    occ_p: &[f64],
    occ_n: &[f64],
) -> f64 {
    let mut e_kin = 0.0;
    for (i, s) in setup.states.iter().enumerate() {
        let t = setup.hw_z.mul_add(
            f64::from(s.n_z) + 0.5,
            setup.hw_perp * (2.0_f64.mul_add(f64::from(s.n_perp), f64::from(s.abs_lambda)) + 1.0),
        );
        e_kin += 2.0 * (occ_p[i] + occ_n[i]) * t;
    }

    let (t0, t3) = (params[0], params[3]);
    let (x0, x3, alpha) = (params[4], params[7], params[8]);
    let (mut ec, mut ecl) = (0.0, 0.0);

    for ir in 0..setup.n_rho {
        let dv = setup.volume_element(ir);
        for iz in 0..setup.n_z {
            let idx = setup.grid_idx(ir, iz);
            let rho = (rho_p[idx] + rho_n[idx]).max(0.0);
            let rp = rho_p[idx].max(0.0);
            let rn = rho_n[idx].max(0.0);
            let h0 = t0 / 4.0
                * ((2.0 + x0) * rho)
                    .mul_add(rho, -(2.0_f64.mul_add(x0, 1.0) * (rp * rp + rn * rn)));
            let h3 = t3 / 24.0
                * rho.powf(alpha)
                * ((2.0 + x3) * rho)
                    .mul_add(rho, -(2.0_f64.mul_add(x3, 1.0) * (rp * rp + rn * rn)));
            ec += (h0 + h3) * dv;
            ecl += crate::physics::hfb_common::coulomb_exchange_energy_density(rp) * dv;
        }
    }

    let rch = 1.2 * (setup.a as f64).cbrt();
    ecl += 0.6 * (setup.z as f64) * (setup.z as f64 - 1.0) * E2 / rch;
    let ld = setup.a as f64 / 28.0;
    let ep = if setup.delta_p > PAIRING_GAP_THRESHOLD {
        -setup.delta_p.powi(2) * ld / 4.0
    } else {
        0.0
    } + if setup.delta_n > PAIRING_GAP_THRESHOLD {
        -setup.delta_n.powi(2) * ld / 4.0
    } else {
        0.0
    };
    let hw0 = 41.0 * (setup.a as f64).powf(-1.0 / 3.0);
    0.75_f64.mul_add(-hw0, e_kin + ec + ecl + ep)
}

pub(super) fn quadrupole(setup: &NucleusSetup, rt: &[f64]) -> f64 {
    let mut q = 0.0;
    for ir in 0..setup.n_rho {
        let dv = setup.volume_element(ir);
        for iz in 0..setup.n_z {
            let rho = (ir + 1) as f64 * setup.d_rho;
            let z = (iz as f64 + 0.5).mul_add(setup.d_z, setup.z_min);
            q += rt[setup.grid_idx(ir, iz)] * (2.0 * z).mul_add(z, -(rho * rho)) * dv;
        }
    }
    q
}
