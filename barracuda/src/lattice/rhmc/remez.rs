// SPDX-License-Identifier: AGPL-3.0-or-later

//! Remez exchange algorithm for optimal rational approximation.
//!
//! Given a set of poles (shifts), finds residues that minimize the maximum
//! relative error of the partial-fraction approximation to x^p over a
//! spectral range. Used by [`super::RationalApproximation::generate`] for
//! RHMC coefficient computation.
//!
//! # References
//!
//! - Remez, "Sur la détermination des polynômes d'approximation" (1934)
//! - Clark & Kennedy, NPB 552 (1999) — rational approximation for RHMC

/// Remez exchange algorithm for fixed poles: finds residues that minimize max relative error.
///
/// Returns (coefficients, `max_relative_error`).
pub(super) fn remez_for_poles(sigma: &[f64], power: f64, eval_grid: &[f64]) -> (Vec<f64>, f64) {
    let n_poles = sigma.len();
    let ncols = n_poles + 1;
    let n_sys = ncols + 1;

    let n_eval = eval_grid.len();
    let mut ref_indices: Vec<usize> = (0..n_sys)
        .map(|k| {
            let theta = std::f64::consts::PI * k as f64 / (n_sys - 1) as f64;
            let t = 0.5 * (1.0 - theta.cos());
            ((t * (n_eval - 1) as f64) as usize).min(n_eval - 1)
        })
        .collect();
    ref_indices.sort_unstable();
    ref_indices.dedup();
    while ref_indices.len() < n_sys {
        for idx in 0..n_eval {
            if !ref_indices.contains(&idx) {
                ref_indices.push(idx);
                ref_indices.sort_unstable();
                break;
            }
        }
    }

    let mut best_coeffs = vec![0.0_f64; ncols];
    let mut best_err = f64::MAX;

    for _ in 0..60 {
        let mut mat = vec![0.0_f64; n_sys * n_sys];
        let rhs = vec![1.0_f64; n_sys];
        for k in 0..n_sys {
            let x = eval_grid[ref_indices[k]];
            let t = x.powf(power);
            mat[k * n_sys] = 1.0 / t;
            for i in 0..n_poles {
                mat[k * n_sys + i + 1] = 1.0 / ((x + sigma[i]) * t);
            }
            mat[k * n_sys + ncols] = if k % 2 == 0 { -1.0 } else { 1.0 };
        }

        let solution = solve_linear_system(&mat, &rhs, n_sys);
        let coeffs: Vec<f64> = solution[..ncols].to_vec();

        let mut max_abs = 0.0_f64;
        let mut signed_err = Vec::with_capacity(n_eval);
        for (idx, &x) in eval_grid.iter().enumerate() {
            let exact = x.powf(power);
            let mut val = coeffs[0];
            for (a, s) in coeffs[1..].iter().zip(sigma.iter()) {
                val += a / (x + s);
            }
            let se = (val - exact) / exact;
            signed_err.push((idx, se));
            if se.abs() > max_abs {
                max_abs = se.abs();
            }
        }

        if max_abs < best_err {
            best_err = max_abs;
            best_coeffs = coeffs;
        }
        if max_abs < 1e-12 {
            break;
        }

        let mut extrema: Vec<(usize, f64)> = Vec::new();
        extrema.push(signed_err[0]);
        for i in 1..n_eval - 1 {
            let (_, ep) = signed_err[i - 1];
            let (idx, e) = signed_err[i];
            let (_, en) = signed_err[i + 1];
            if (e > ep && e > en) || (e < ep && e < en) {
                extrema.push((idx, e));
            }
        }
        extrema.push(signed_err[n_eval - 1]);

        if extrema.len() < n_sys {
            break;
        }

        let mut selected: Vec<(usize, f64)> = Vec::new();
        for &(idx, e) in &extrema {
            if selected.is_empty() {
                selected.push((idx, e));
            } else {
                let Some(last) = selected.last() else {
                    continue;
                };
                let last_sign = last.1.signum();
                if (e.signum() - last_sign).abs() > f64::EPSILON {
                    selected.push((idx, e));
                } else if selected.last().is_some_and(|l| e.abs() > l.1.abs())
                    && let Some(slot) = selected.last_mut()
                {
                    *slot = (idx, e);
                }
            }
        }

        while selected.len() > n_sys {
            let Some((min_idx, _)) = selected
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.1.abs().total_cmp(&b.1.abs()))
            else {
                break;
            };
            selected.remove(min_idx);
        }

        if selected.len() < n_sys {
            break;
        }

        let new_refs: Vec<usize> = selected.iter().map(|&(idx, _)| idx).collect();
        if new_refs == ref_indices {
            break;
        }
        ref_indices = new_refs;
    }

    (best_coeffs, best_err)
}

/// Solve A·x = b via Gaussian elimination with partial pivoting.
pub(super) fn solve_linear_system(ata: &[f64], atb: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    a.copy_from_slice(&ata[..n * n]);
    let mut b = atb[..n].to_vec();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        x[i] = if diag.abs() > 1e-30 { sum / diag } else { 0.0 };
    }
    x
}
