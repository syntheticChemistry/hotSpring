// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-shift Conjugate Gradient solver.
//!
//! Solves (D†D + σᵢ)xᵢ = b for all shifts simultaneously using a single
//! Krylov space. Only one matrix-vector product per iteration regardless
//! of shift count — the key algorithmic advantage for RHMC.
//!
//! # References
//!
//! - Jegerlehner, hep-lat/9612014 — multi-shift CG algorithm
//! - Frommer et al., Numer. Math. 83 (1999) — shifted systems

use super::super::dirac::{FermionField, apply_dirac_sq};
use super::super::wilson::Lattice;

/// Result of a multi-shift CG solve.
#[derive(Clone, Debug)]
pub struct MultiShiftCgResult {
    /// Number of CG iterations (shared across all shifts).
    pub iterations: usize,
    /// Final residual norm squared (for the base system, sigma=0).
    pub residual_sq: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

/// Multi-shift Conjugate Gradient: solve (D†D + `σ_i)x_i` = b for all shifts simultaneously.
///
/// All shifted systems share the same Krylov space. Only one matrix-vector
/// product (D†D·p) per iteration, regardless of the number of shifts.
///
/// Returns solution vectors `x_i` (one per shift) and iteration count.
#[must_use]
pub fn multi_shift_cg_solve(
    lattice: &Lattice,
    b: &FermionField,
    mass: f64,
    shifts: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<FermionField>, MultiShiftCgResult) {
    let vol = lattice.volume();
    let n_shifts = shifts.len();

    let mut x: Vec<FermionField> = (0..n_shifts).map(|_| FermionField::zeros(vol)).collect();
    let mut p: Vec<FermionField> = (0..n_shifts)
        .map(|_| FermionField {
            data: b.data.clone(),
            volume: vol,
        })
        .collect();
    let mut r = FermionField {
        data: b.data.clone(),
        volume: vol,
    };

    let b_norm_sq = b.dot(b).re;
    if b_norm_sq < 1e-30 {
        return (
            x,
            MultiShiftCgResult {
                iterations: 0,
                residual_sq: 0.0,
                converged: true,
            },
        );
    }

    let tol_sq = tol * tol * b_norm_sq;
    let mut rz = b_norm_sq;
    let mut zeta_prev: Vec<f64> = vec![1.0; n_shifts];
    let mut zeta_curr: Vec<f64> = vec![1.0; n_shifts];
    let mut beta_prev: Vec<f64> = vec![0.0; n_shifts];
    let mut alpha_prev = 0.0_f64;
    let mut active: Vec<bool> = vec![true; n_shifts];

    let mut iterations = 0;

    for _iter in 0..max_iter {
        iterations += 1;

        let ap = apply_dirac_sq(lattice, &p[0], mass);

        let mut p_ap = p[0].dot(&ap).re;
        p_ap += shifts[0] * p[0].dot(&p[0]).re;

        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / p_ap;

        update_base_system(&mut x[0], &mut r, &p[0], &ap, shifts[0], alpha, vol);

        let rz_new = r.dot(&r).re;

        update_shifted_systems(
            &mut x,
            &mut p,
            &r,
            shifts,
            &mut zeta_prev,
            &mut zeta_curr,
            &mut beta_prev,
            &mut active,
            alpha,
            alpha_prev,
            rz,
            rz_new,
            vol,
        );

        let beta = if rz.abs() > 1e-30 { rz_new / rz } else { 0.0 };
        for i in 0..vol {
            for c in 0..3 {
                p[0].data[i][c] = r.data[i][c] + p[0].data[i][c].scale(beta);
            }
        }

        alpha_prev = alpha;
        rz = rz_new;

        if rz < tol_sq {
            break;
        }
    }

    (
        x,
        MultiShiftCgResult {
            iterations,
            residual_sq: rz / b_norm_sq,
            converged: rz < tol_sq,
        },
    )
}

fn update_base_system(
    x0: &mut FermionField,
    r: &mut FermionField,
    p0: &FermionField,
    ap: &FermionField,
    shift0: f64,
    alpha: f64,
    vol: usize,
) {
    for i in 0..vol {
        for c in 0..3 {
            x0.data[i][c] += p0.data[i][c].scale(alpha);
            r.data[i][c] -= (ap.data[i][c] + p0.data[i][c].scale(shift0)).scale(alpha);
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "CG state inherently requires many coupled variables"
)]
fn update_shifted_systems(
    x: &mut [FermionField],
    p: &mut [FermionField],
    r: &FermionField,
    shifts: &[f64],
    zeta_prev: &mut [f64],
    zeta_curr: &mut [f64],
    beta_prev: &mut [f64],
    active: &mut [bool],
    alpha: f64,
    alpha_prev: f64,
    rz: f64,
    rz_new: f64,
    vol: usize,
) {
    for s in 1..shifts.len() {
        if !active[s] {
            continue;
        }

        let ds = shifts[s] - shifts[0];
        let denom = alpha.mul_add(ds, 1.0)
            + alpha * alpha_prev * (1.0 - zeta_prev[s] / zeta_curr[s]) / beta_prev[s].max(1e-30);
        if denom.abs() < 1e-30 {
            active[s] = false;
            continue;
        }

        let zeta_next = zeta_curr[s] / denom;
        let alpha_s = alpha * zeta_next / zeta_curr[s];

        for i in 0..vol {
            for c in 0..3 {
                x[s].data[i][c] += p[s].data[i][c].scale(alpha_s);
            }
        }

        let beta_s = if rz.abs() > 1e-30 {
            (zeta_next / zeta_curr[s]).powi(2) * (rz_new / rz)
        } else {
            0.0
        };

        for i in 0..vol {
            for c in 0..3 {
                p[s].data[i][c] = r.data[i][c].scale(zeta_next) + p[s].data[i][c].scale(beta_s);
            }
        }

        zeta_prev[s] = zeta_curr[s];
        zeta_curr[s] = zeta_next;
        beta_prev[s] = beta_s;
    }
}
