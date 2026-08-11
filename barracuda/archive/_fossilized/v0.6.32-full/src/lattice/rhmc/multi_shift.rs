// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-shift Conjugate Gradient solver for RHMC.
//!
//! Solves (D†D + σᵢ)xᵢ = b for all shifts simultaneously using a single
//! Krylov space (Jegerlehner, hep-lat/9612014). Only one matrix-vector
//! product per iteration regardless of the number of shifts.

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
/// Returns solution vectors `x_i` (one per shift) and convergence metadata.
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

    // Base direction vector (sigma=0 system)
    let mut p0 = FermionField {
        data: b.data.clone(),
        volume: vol,
    };

    let mut active: Vec<bool> = vec![true; n_shifts];

    for iter in 0..max_iter {
        let ap0 = apply_dirac_sq(lattice, &p0, mass);
        let p0_ap0 = p0.dot(&ap0).re;

        if p0_ap0.abs() < 1e-30 {
            break;
        }

        let alpha = rz / p0_ap0;

        // Update base residual
        for i in 0..vol {
            for c in 0..3 {
                r.data[i][c] -= ap0.data[i][c].scale(alpha);
            }
        }

        let rz_new = r.dot(&r).re;
        let beta = rz_new / rz;

        // Update all shifted systems
        for s in 0..n_shifts {
            if !active[s] {
                continue;
            }
            let sigma = shifts[s];
            let denom =
                1.0 + alpha * sigma + alpha * beta_prev[s] * (1.0 - zeta_curr[s] / zeta_prev[s]);
            let zeta_next = if denom.abs() < 1e-30 {
                active[s] = false;
                continue;
            } else {
                zeta_curr[s] / denom
            };
            let alpha_s = alpha * zeta_next / zeta_curr[s];
            let beta_s = beta * (zeta_next / zeta_curr[s]).powi(2);

            // Update x_s and p_s
            for i in 0..vol {
                for c in 0..3 {
                    x[s].data[i][c] += p[s].data[i][c].scale(alpha_s);
                    p[s].data[i][c] = r.data[i][c].scale(zeta_next) + p[s].data[i][c].scale(beta_s);
                }
            }

            beta_prev[s] = beta_s;
            zeta_prev[s] = zeta_curr[s];
            zeta_curr[s] = zeta_next;
        }

        alpha_prev = alpha;

        // Update base direction
        for i in 0..vol {
            for c in 0..3 {
                p0.data[i][c] = r.data[i][c] + p0.data[i][c].scale(beta);
            }
        }

        rz = rz_new;

        if rz < tol_sq {
            return (
                x,
                MultiShiftCgResult {
                    iterations: iter + 1,
                    residual_sq: rz,
                    converged: true,
                },
            );
        }
    }

    let _ = alpha_prev;

    (
        x,
        MultiShiftCgResult {
            iterations: max_iter,
            residual_sq: rz,
            converged: false,
        },
    )
}
