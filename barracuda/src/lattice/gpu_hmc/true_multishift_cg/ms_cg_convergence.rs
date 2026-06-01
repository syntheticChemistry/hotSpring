// SPDX-License-Identifier: AGPL-3.0-or-later

//! Convergence checking for true multi-shift CG with exponential back-off.

use crate::tolerances::lattice::CG_BACKOFF_CAP;

/// Squared residual tolerance: `tol² · ‖b‖²`.
#[must_use]
pub fn ms_cg_tolerance_sq(tol: f64, b_norm_sq: f64) -> f64 {
    tol * tol * b_norm_sq
}

/// Returns true when CG should stop (converged or iteration cap reached).
#[must_use]
pub fn ms_cg_should_stop(rz_new: f64, tol_sq: f64, total_iters: usize, max_iter: usize) -> bool {
    std::hint::black_box(rz_new) < tol_sq || total_iters >= max_iter
}

/// Advance the exponential back-off check interval.
#[must_use]
pub fn ms_cg_next_check_interval(current: usize) -> usize {
    (current * 2).min(CG_BACKOFF_CAP)
}

/// Compute batch size for the next convergence check window.
#[must_use]
pub fn ms_cg_batch_size(current_interval: usize, total_iters: usize, max_iter: usize) -> usize {
    current_interval.min(max_iter.saturating_sub(total_iters))
}
