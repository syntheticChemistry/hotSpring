// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: LTEE B2 — Anderson disorder analogy for fitness dynamics.
//!
//! Reproduces Wiser, Ribeck & Lenski (2013) "Long-Term Dynamics of
//! Adaptation in Asexual Populations" *Science* 342, 1364–1367.
//!
//! Maps LTEE fitness increments to an Anderson-like on-site disorder
//! potential and applies spectral diagnostics (level spacing ratio)
//! to test localization–delocalization transitions.
//!
//! **Default build (IPC-first):** validates power-law fitness model,
//! diminishing returns, and increment statistics.
//!
//! **`barracuda-local` build:** additionally computes eigenvalues of the
//! fitness-derived Anderson Hamiltonian and validates level spacing
//! ratio against GOE/Poisson bounds.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

/// Wiser et al. 2013 Table S2 mean parameters.
const ALPHA: f64 = 6.2e-4;
const BETA: f64 = 0.056;

/// Reference level spacing ratio bounds (Anderson localization diagnostics).
const POISSON_R: f64 = 2.0 * core::f64::consts::LN_2 - 1.0; // 0.3863
const GOE_R: f64 = 0.531;

/// LTEE generation time points where fitness was measured.
const GENERATIONS: &[f64] = &[
    500.0, 1000.0, 1500.0, 2000.0, 5000.0, 10_000.0, 15_000.0, 20_000.0, 30_000.0, 40_000.0,
    50_000.0,
];

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "ltee-anderson",
        track: Track::DomainScience,
        tier: Tier::Rust,
        provenance_crate: "notebooks/papers/13-ltee-anderson-fitness",
        provenance_date: "2026-05-11",
        description: "LTEE B2: Wiser et al. 2013 Anderson fitness analogy — power-law model + spectral diagnostics",
    },
    run,
};

/// Wiser et al. 2013 power-law fitness model.
fn wiser_fitness(t: f64) -> f64 {
    (1.0 + 2.0 * ALPHA * t).powf(BETA)
}

/// Compute fitness increments from a trajectory evaluated at `points`.
fn fitness_increments(points: &[f64]) -> Vec<f64> {
    let vals: Vec<f64> = points.iter().map(|&t| wiser_fitness(t)).collect();
    vals.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Compute mean level spacing ratio from sorted eigenvalues (pure Rust).
fn level_spacing_ratio(eigenvalues: &mut [f64]) -> f64 {
    eigenvalues.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let spacings: Vec<f64> = eigenvalues
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&s| s > 1e-14)
        .collect();
    if spacings.len() < 3 {
        return f64::NAN;
    }
    let sum: f64 = spacings
        .windows(2)
        .map(|w| w[0].min(w[1]) / w[0].max(w[1]))
        .sum();
    sum / (spacings.len() - 1) as f64
}

/// Eigenvalues of a symmetric tridiagonal matrix via implicit QL
/// with Wilkinson shift (LAPACK DSTEQR pattern). `diag` is the main
/// diagonal, `off` is the sub/super-diagonal (length = diag.len() - 1).
fn tridiag_eigenvalues(diag: &[f64], off: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![diag[0]];
    }
    let mut d = diag.to_vec();
    let mut e = vec![0.0; n];
    for (i, &val) in off.iter().enumerate() {
        e[i] = val;
    }

    let max_iter = 30 * n;
    for _ in 0..max_iter {
        let mut converged = true;
        for l in 0..n - 1 {
            let tst = d[l].abs() + d[l + 1].abs();
            if e[l].abs() > f64::EPSILON * tst {
                converged = false;
                // Find unreduced block [l..m]
                let mut m = l + 1;
                while m < n - 1 {
                    let tst2 = d[m].abs() + d[m + 1].abs();
                    if e[m].abs() <= f64::EPSILON * tst2 {
                        break;
                    }
                    m += 1;
                }
                // QL iteration from bottom of block
                let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                let r = g.hypot(1.0);
                let sign_g = if g >= 0.0 { r } else { -r };
                let mut g_val = d[m] - d[l] + e[l] / (g + sign_g);

                let mut s = 1.0;
                let mut c = 1.0;
                let mut p = 0.0;

                for i in (l..m).rev() {
                    let f = s * e[i];
                    let b = c * e[i];
                    let r2 = f.hypot(g_val);
                    e[i + 1] = r2;
                    if r2.abs() < 1e-300 {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r2;
                    c = g_val / r2;
                    g_val = d[i + 1] - p;
                    let r3 = (d[i] - g_val) * s + 2.0 * c * b;
                    p = s * r3;
                    d[i + 1] = g_val + p;
                    g_val = c * r3 - b;
                }
                d[l] -= p;
                e[l] = g_val;
                e[m] = 0.0;
                break;
            }
        }
        if converged {
            break;
        }
    }
    d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    d
}

/// Build Anderson-like Hamiltonian from fitness increments and extract
/// eigenvalues. Returns the diagonal (normalized increments), off-diagonal
/// (constant hopping), and eigenvalues.
fn fitness_anderson_eigenvalues(increments: &[f64]) -> Vec<f64> {
    let n = increments.len();
    if n < 2 {
        return vec![];
    }
    let mean = increments.iter().sum::<f64>() / n as f64;
    let var = increments.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt();

    let diag: Vec<f64> = if std > 1e-14 {
        increments.iter().map(|&x| (x - mean) / std).collect()
    } else {
        increments.to_vec()
    };
    let off = vec![-1.0; n - 1];
    tridiag_eigenvalues(&diag, &off)
}

pub fn run(v: &mut ValidationHarness) {
    // --- Power-law fitness model checks (always available) ---

    let w_500 = wiser_fitness(500.0);
    let w_5000 = wiser_fitness(5000.0);
    let w_10000 = wiser_fitness(10_000.0);
    let w_50000 = wiser_fitness(50_000.0);

    v.check_bool("ltee:fitness_500_finite", w_500.is_finite() && w_500 > 1.0);
    v.check_bool(
        "ltee:fitness_50k_finite",
        w_50000.is_finite() && w_50000 > 1.0,
    );
    v.check_bool("ltee:power_law_no_plateau", w_50000 > w_10000);
    v.check_bool("ltee:monotone_increase", w_50000 > w_5000 && w_5000 > w_500);

    // Diminishing returns: increment ratio last/first < 1
    let increments = fitness_increments(GENERATIONS);
    let first = increments[0];
    let last = increments[increments.len() - 1];
    v.check_bool(
        "ltee:diminishing_returns",
        last < first && first > 0.0 && last > 0.0,
    );

    // Increment statistics
    let mean_inc = increments.iter().sum::<f64>() / increments.len() as f64;
    v.check_bool("ltee:increments_positive_mean", mean_inc > 0.0);
    v.check_bool(
        "ltee:n_increments",
        increments.len() == GENERATIONS.len() - 1,
    );

    // --- Spectral diagnostics (eigenvalue analysis) ---

    // Fine-grained trajectory for spectral analysis
    let n_fine = 200;
    let t_min = 500.0;
    let t_max = 50_000.0;
    let fine_points: Vec<f64> = (0..n_fine)
        .map(|i| t_min + (t_max - t_min) * i as f64 / (n_fine - 1) as f64)
        .collect();
    let fine_increments = fitness_increments(&fine_points);

    let mut evals = fitness_anderson_eigenvalues(&fine_increments);
    v.check_bool("ltee:eigenvalues_computed", !evals.is_empty());
    v.check_bool(
        "ltee:eigenvalues_finite",
        evals.iter().all(|e| e.is_finite()),
    );

    let r = level_spacing_ratio(&mut evals);
    v.check_bool("ltee:r_is_finite", r.is_finite());
    v.check_bool("ltee:r_between_poisson_goe", r > POISSON_R && r < GOE_R);

    // Determinism check
    let evals2 = fitness_anderson_eigenvalues(&fine_increments);
    v.check_bool(
        "ltee:deterministic",
        evals.len() == evals2.len()
            && evals
                .iter()
                .zip(evals2.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12),
    );

    // Sliding window trend: early <r> vs late <r>
    let window = 50;
    if fine_increments.len() >= window * 2 {
        let early = &fine_increments[..window];
        let late = &fine_increments[fine_increments.len() - window..];

        let mut e_early = fitness_anderson_eigenvalues(early);
        let mut e_late = fitness_anderson_eigenvalues(late);

        let r_early = level_spacing_ratio(&mut e_early);
        let r_late = level_spacing_ratio(&mut e_late);

        v.check_bool("ltee:window_r_early_finite", r_early.is_finite());
        v.check_bool("ltee:window_r_late_finite", r_late.is_finite());
        // Both should be in the physically meaningful range
        v.check_bool(
            "ltee:window_r_early_in_range",
            r_early > 0.3 && r_early < 0.6,
        );
        v.check_bool("ltee:window_r_late_in_range", r_late > 0.3 && r_late < 0.6);
    }

    // 12-population variance (deterministic pseudo-random via parameter variation)
    let n_pops = 12;
    let mut pop_r_values = Vec::with_capacity(n_pops);
    for pop in 0..n_pops {
        let a_scale = 1.0 + 0.15 * ((pop as f64 * core::f64::consts::E).sin());
        let b_scale = 1.0 + 0.10 * ((pop as f64 * core::f64::consts::PI).cos());
        let a_pop = ALPHA * a_scale;
        let b_pop = BETA * b_scale;

        let pop_vals: Vec<f64> = fine_points
            .iter()
            .map(|&t| (1.0 + 2.0 * a_pop * t).powf(b_pop))
            .collect();
        let pop_inc: Vec<f64> = pop_vals.windows(2).map(|w| w[1] - w[0]).collect();
        let mut pop_evals = fitness_anderson_eigenvalues(&pop_inc);
        let r_pop = level_spacing_ratio(&mut pop_evals);
        if r_pop.is_finite() {
            pop_r_values.push(r_pop);
        }
    }

    v.check_bool("ltee:population_count", pop_r_values.len() == n_pops);

    if !pop_r_values.is_empty() {
        let mean_r = pop_r_values.iter().sum::<f64>() / pop_r_values.len() as f64;
        let var_r = pop_r_values
            .iter()
            .map(|&r| (r - mean_r).powi(2))
            .sum::<f64>()
            / pop_r_values.len() as f64;
        v.check_bool(
            "ltee:population_mean_r_in_range",
            mean_r > 0.3 && mean_r < 0.6,
        );
        v.check_bool("ltee:population_variance_exists", var_r > 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wiser_model_sanity() {
        let w0 = wiser_fitness(0.0);
        assert!((w0 - 1.0).abs() < 1e-12, "w(0) should be 1.0");
        assert!(wiser_fitness(50_000.0) > wiser_fitness(10_000.0));
    }

    #[test]
    fn tridiag_2x2() {
        let d = [1.0, 3.0];
        let e = [2.0];
        let evals = tridiag_eigenvalues(&d, &e);
        assert_eq!(evals.len(), 2);
        // eigenvalues of [[1,2],[2,3]] are 2 ± sqrt(5)
        let expected_lo = 2.0 - 5.0_f64.sqrt();
        let expected_hi = 2.0 + 5.0_f64.sqrt();
        assert!(
            (evals[0] - expected_lo).abs() < 1e-10,
            "low eigenvalue mismatch: {} vs {}",
            evals[0],
            expected_lo
        );
        assert!(
            (evals[1] - expected_hi).abs() < 1e-10,
            "high eigenvalue mismatch: {} vs {}",
            evals[1],
            expected_hi
        );
    }

    #[test]
    fn level_spacing_ratio_goe_range() {
        let increments =
            fitness_increments(&[500.0, 1000.0, 2000.0, 5000.0, 10_000.0, 20_000.0, 50_000.0]);
        let mut evals = fitness_anderson_eigenvalues(&increments);
        let r = level_spacing_ratio(&mut evals);
        assert!(r.is_finite(), "<r> should be finite");
        assert!(r > 0.0 && r < 1.0, "<r> should be in (0, 1)");
    }
}
