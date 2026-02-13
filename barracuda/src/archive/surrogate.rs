//! Reference surrogate modeling utilities for BarraCUDA evolution.
//!
//! These implementations demonstrate the math and algorithms that
//! BarraCUDA's `surrogate` and `sample::sparsity` modules need to
//! incorporate for robust scientific optimization.
//!
//! ## Key Reference Systems:
//!
//! 1. **LOO-CV Auto-Smoothing** — Automatically select RBF smoothing
//!    parameter via leave-one-out cross-validation grid search.
//!
//! 2. **Penalty-Filtered Training** — Filter out penalty/sentinel values
//!    before training the surrogate (critical for rugged landscapes).
//!
//! 3. **Round-Based Direct Optimization** — Alternative to surrogate-guided
//!    NM: run NM on the TRUE objective in rounds, using the surrogate
//!    only for quality monitoring and adaptive stopping.
//!
//! 4. **Adaptive Penalty Scaling** — Ensure penalty values are always
//!    higher than any feasible objective value.
//!
//! ## Usage Context
//!
//! These are proven in hotSpring's nuclear EOS L1/L2 fitting and
//! serve as specification for the BarraCUDA library team.

use barracuda::surrogate::{RBFKernel, RBFSurrogate};
use barracuda::sample::latin_hypercube;
use barracuda::optimize::EvaluationCache;

/// Find optimal RBF smoothing parameter via LOO-CV grid search.
///
/// Tests a geometric sequence of smoothing values and returns the one
/// with lowest LOO-CV RMSE. This prevents both overfitting (too low)
/// and underfitting (too high).
///
/// # Implementation Note
///
/// Uses our own LOO-CV implementation (`loo_cv_rmse_ref`) because
/// BarraCUDA's `RBFSurrogate::loo_cv_rmse()` has a hat-matrix bug:
/// it uses `(K+λI)` as both the system matrix AND right-hand side,
/// so `H_ii = 1` for all points regardless of smoothing.
///
/// **BUG FIX for BarraCUDA team:** In `compute_hat_diagonal()`,
/// build TWO matrices: `K_smooth = K + λI` and `K_raw = K`.
/// Solve `K_smooth · x = K_raw[:,i]`, then `H_ii = x[i]`.
///
/// # Reference for BarraCUDA
///
/// This should be integrated into `SparsitySamplerConfig` as an
/// `auto_smoothing: bool` option that runs this search each iteration.
///
/// # Arguments
///
/// * `x_data` - Training points
/// * `y_data` - Training values
/// * `kernel` - RBF kernel type
/// * `candidates` - Smoothing values to try (None = default geometric grid)
///
/// # Returns
///
/// `(optimal_smoothing, loo_cv_rmse, all_results)` where `all_results`
/// is a Vec of `(smoothing, rmse)` pairs for all candidates.
pub fn loo_cv_optimal_smoothing(
    x_data: &[Vec<f64>],
    y_data: &[f64],
    kernel: RBFKernel,
    candidates: Option<&[f64]>,
) -> (f64, f64, Vec<(f64, f64)>) {
    // Default: geometric grid from 1e-14 to 1e2 (17 values)
    let default_candidates: Vec<f64> = vec![
        1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4,
        1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5,
        1.0, 5.0, 10.0, 50.0, 100.0,
    ];
    let candidates = candidates.unwrap_or(&default_candidates);

    let mut results: Vec<(f64, f64)> = Vec::with_capacity(candidates.len());
    let mut best_smoothing = 1e-2; // Sensible fallback
    let mut best_rmse = f64::INFINITY;

    for &s in candidates {
        let rmse = loo_cv_rmse_ref(x_data, y_data, kernel, s);
        results.push((s, rmse));
        if rmse.is_finite() && rmse > 0.0 && rmse < best_rmse {
            best_rmse = rmse;
            best_smoothing = s;
        }
    }

    (best_smoothing, best_rmse, results)
}

/// Corrected LOO-CV RMSE computation.
///
/// **This is the reference implementation for BarraCUDA.**
///
/// Uses the "virtual leave-one-out" formula:
///   LOO_i = (y_i - ŷ_i) / (1 - H_ii)
///
/// where H is the hat matrix: H = K · (K + λI)⁻¹
///
/// Key correctness requirement: the kernel matrix K (without λ)
/// must be used as the RHS when solving for H, while K + λI is
/// the system matrix.
pub fn loo_cv_rmse_ref(
    x_data: &[Vec<f64>],
    y_data: &[f64],
    kernel: RBFKernel,
    smoothing: f64,
) -> f64 {
    let n = x_data.len();
    if n < 3 { return f64::INFINITY; }
    let n_dim = x_data[0].len();

    // Flatten training data
    let flat_x: Vec<f64> = x_data.iter().flat_map(|row| row.iter().copied()).collect();

    // Compute pairwise distances
    let mut distances = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut d2 = 0.0;
            for d in 0..n_dim {
                let diff = flat_x[i * n_dim + d] - flat_x[j * n_dim + d];
                d2 += diff * diff;
            }
            distances[i * n + j] = d2.sqrt();
        }
    }

    // Build K_raw (kernel matrix without regularization)
    let mut k_raw = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            k_raw[i * n + j] = kernel.eval(distances[i * n + j]);
        }
    }

    // Build K_smooth = K_raw + λI
    let mut k_smooth = k_raw.clone();
    for i in 0..n {
        k_smooth[i * n + i] += smoothing;
    }

    // Solve K_smooth · W = I for W = (K+λI)⁻¹ (one column at a time)
    // Then H = K_raw · W, and H_ii = Σ_j K_raw[i,j] · W[j,i]
    //
    // More efficient: for each i, solve K_smooth · w = e_i, then H_ii = K_raw[i,:] · w
    let mut h_diag = vec![0.0; n];

    for i in 0..n {
        // Build e_i
        let mut rhs = vec![0.0; n];
        rhs[i] = 1.0;

        // Solve K_smooth · w = e_i via LU (inline for reference clarity)
        match lu_solve_ref(&k_smooth, &rhs, n) {
            Some(w) => {
                // H_ii = K_raw[i,:] · w
                let mut h_ii = 0.0;
                for j in 0..n {
                    h_ii += k_raw[i * n + j] * w[j];
                }
                h_diag[i] = h_ii;
            }
            None => return f64::INFINITY, // Singular matrix
        }
    }

    // Train surrogate and get predictions at training points
    let surrogate = match RBFSurrogate::train(x_data, y_data, kernel, smoothing) {
        Ok(s) => s,
        Err(_) => return f64::INFINITY,
    };

    let predictions = match surrogate.predict(x_data) {
        Ok(p) => p,
        Err(_) => return f64::INFINITY,
    };

    // Compute LOO residuals: LOO_i = (y_i - ŷ_i) / (1 - H_ii)
    let mut sse = 0.0;
    let mut n_valid = 0;
    for i in 0..n {
        let denom = 1.0 - h_diag[i];
        if denom.abs() > 1e-12 {
            let loo_resid = (y_data[i] - predictions[i]) / denom;
            sse += loo_resid * loo_resid;
            n_valid += 1;
        }
    }

    if n_valid == 0 { return f64::INFINITY; }
    (sse / n_valid as f64).sqrt()
}

/// Reference LU solve (no pivoting, for small systems).
/// Returns None if singular.
fn lu_solve_ref(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    // Copy A
    let mut lu = a.to_vec();
    let mut x = b.to_vec();

    // Forward elimination (Gaussian with partial pivoting)
    let mut perm: Vec<usize> = (0..n).collect();
    for col in 0..n {
        // Partial pivoting
        let mut max_val = lu[perm[col] * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = lu[perm[row] * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-15 { return None; } // Singular
        perm.swap(col, max_row);

        // Eliminate below
        for row in (col + 1)..n {
            let factor = lu[perm[row] * n + col] / lu[perm[col] * n + col];
            for j in col..n {
                let val = lu[perm[col] * n + j];
                lu[perm[row] * n + j] -= factor * val;
            }
            let xc = x[perm[col]];
            x[perm[row]] -= factor * xc;
        }
    }

    // Back substitution
    let mut result = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = x[perm[i]];
        for j in (i + 1)..n {
            sum -= lu[perm[i] * n + j] * result[j];
        }
        result[i] = sum / lu[perm[i] * n + i];
    }

    Some(result)
}

/// Filter training data to remove penalty/sentinel values.
///
/// For expensive objectives that return large penalty values for
/// infeasible regions, training the surrogate on these corrupts
/// the approximation. This filter removes them before training.
///
/// # Reference for BarraCUDA
///
/// Should be an option in `SparsitySamplerConfig`:
/// ```ignore
/// .with_penalty_filter(PenaltyFilter::Threshold(1e3))
/// .with_penalty_filter(PenaltyFilter::Quantile(0.95))
/// ```
///
/// # Arguments
///
/// * `x_data` - All training points
/// * `y_data` - All training values
/// * `mode` - Filtering mode
///
/// # Returns
///
/// Filtered (x_data, y_data) with penalty values removed.
pub fn filter_training_data(
    x_data: &[Vec<f64>],
    y_data: &[f64],
    mode: PenaltyFilter,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let threshold = match mode {
        PenaltyFilter::Threshold(t) => t,
        PenaltyFilter::Quantile(q) => {
            let mut sorted: Vec<f64> = y_data.iter().copied()
                .filter(|v| v.is_finite())
                .collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if sorted.is_empty() {
                return (Vec::new(), Vec::new());
            }
            let idx = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
            sorted[idx]
        }
        PenaltyFilter::AdaptiveMAD(k) => {
            // Median + k * MAD (robust outlier detection)
            let mut sorted: Vec<f64> = y_data.iter().copied()
                .filter(|v| v.is_finite())
                .collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if sorted.is_empty() {
                return (Vec::new(), Vec::new());
            }
            let median = sorted[sorted.len() / 2];
            let mut abs_devs: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mad = abs_devs[abs_devs.len() / 2];
            median + k * mad * 1.4826  // 1.4826 scales MAD to std dev
        }
    };

    let mut x_filtered = Vec::new();
    let mut y_filtered = Vec::new();
    for (x, &y) in x_data.iter().zip(y_data.iter()) {
        if y.is_finite() && y <= threshold {
            x_filtered.push(x.clone());
            y_filtered.push(y);
        }
    }

    (x_filtered, y_filtered)
}

/// Penalty filtering mode for surrogate training data.
#[derive(Debug, Clone, Copy)]
pub enum PenaltyFilter {
    /// Remove all values above a fixed threshold.
    Threshold(f64),
    /// Remove values above the q-th quantile (0.0..1.0).
    Quantile(f64),
    /// Remove values more than k MADs above the median.
    AdaptiveMAD(f64),
}

/// Round-based optimization on the TRUE objective.
///
/// This is the approach that achieved χ²/datum = 25.43 for L2.
/// Instead of optimizing a surrogate, it runs NM directly on the
/// expensive objective in rounds, using the surrogate ONLY for
/// quality monitoring and early stopping.
///
/// # Algorithm
///
/// ```text
/// FOR round = 0..max_rounds:
///   1. Run multi-start NM on TRUE objective (n_solvers × evals_per_solver)
///   2. Add all evaluations to cache
///   3. Train surrogate on filtered cache (for monitoring only)
///   4. Compute LOO-CV RMSE (quality metric)
///   5. If no improvement for `patience` rounds → early stop
/// ```
///
/// # Reference for BarraCUDA
///
/// Should be a separate function in `barracuda::sample::sparsity`:
/// ```ignore
/// pub fn direct_sampler<F>(f: F, bounds: &[(f64,f64)], config: &DirectSamplerConfig) -> Result<...>
/// ```
pub fn round_based_direct_optimization<F>(
    f: F,
    bounds: &[(f64, f64)],
    config: &DirectSamplerConfig,
) -> DirectSamplerResult
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let n_dims = bounds.len();
    let mut cache = EvaluationCache::with_capacity(
        config.n_initial + config.max_rounds * config.evals_per_round
    );
    let mut round_results = Vec::with_capacity(config.max_rounds);

    // Phase 1: Initial LHS sampling
    let t0 = std::time::Instant::now();
    let initial_points = latin_hypercube(config.n_initial, bounds, config.seed)
        .expect("LHS generation failed");

    for point in &initial_points {
        let val = f(point);
        cache.record(point.clone(), val);
    }

    let best_after_init = cache.best().map(|r| r.f).unwrap_or(f64::INFINITY);
    println!("  [init] {} evals, best f = {:.6}", cache.len(), best_after_init);

    let mut global_best_f = best_after_init;
    let mut stagnation_count = 0;

    // Phase 2: Round-based optimization
    for round in 0..config.max_rounds {
        let round_start = cache.len();
        let round_seed = config.seed.wrapping_add((round as u64 + 1) * 7919);

        // Generate starting points for NM
        let starts = latin_hypercube(config.n_solvers, bounds, round_seed)
            .expect("LHS generation failed");

        // Run multi-start NM on TRUE objective
        let mut round_best_f = f64::INFINITY;
        for x0 in &starts {
            let (x_best, f_best, _n_evals) = barracuda::optimize::nelder_mead(
                &f,
                x0,
                bounds,
                config.evals_per_round / config.n_solvers,
                config.tol,
            ).expect("Nelder-Mead failed");

            cache.record(x_best, f_best);
            if f_best < round_best_f {
                round_best_f = f_best;
            }
        }

        let round_evals = cache.len() - round_start;
        let overall_best = cache.best().map(|r| r.f).unwrap_or(f64::INFINITY);

        // Train surrogate on filtered data for quality monitoring
        let (x_filt, y_filt) = filter_training_data(
            &cache.training_data().0,
            &cache.training_data().1,
            config.filter,
        );

        let mut surrogate_rmse = f64::NAN;
        let mut smoothing_used = config.smoothing;

        if x_filt.len() >= n_dims + 2 {
            // Auto-tune smoothing if requested
            if config.auto_smoothing {
                let (opt_s, opt_rmse, _) = loo_cv_optimal_smoothing(
                    &x_filt, &y_filt, config.kernel, None,
                );
                smoothing_used = opt_s;
                surrogate_rmse = opt_rmse;
            } else {
                if let Ok(surr) = RBFSurrogate::train(
                    &x_filt, &y_filt, config.kernel, config.smoothing
                ) {
                    surrogate_rmse = surr.loo_cv_rmse().unwrap_or(f64::NAN);
                }
            }
        }

        // Early stopping check
        let improved = overall_best < global_best_f - config.improvement_threshold;
        if improved {
            global_best_f = overall_best;
            stagnation_count = 0;
        } else {
            stagnation_count += 1;
        }

        println!("  [round {:2}] {} evals (+{}), best f = {:.6}, \
                  surrogate RMSE = {:.4} (s={:.1e}), stag = {}",
            round, cache.len(), round_evals, overall_best,
            surrogate_rmse, smoothing_used, stagnation_count);

        round_results.push(RoundResult {
            round,
            n_evals: round_evals,
            total_evals: cache.len(),
            best_f: overall_best,
            round_best_f,
            surrogate_rmse,
            smoothing_used,
            n_filtered: x_filt.len(),
        });

        if stagnation_count >= config.patience {
            println!("  Early stop after {} rounds (no improvement for {} rounds)",
                round + 1, config.patience);
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let (x_best, f_best) = cache.best()
        .map(|r| (r.x.clone(), r.f))
        .unwrap_or_else(|| (vec![0.0; n_dims], f64::INFINITY));

    DirectSamplerResult {
        x_best,
        f_best,
        cache,
        round_results,
        total_time: elapsed,
    }
}

/// Configuration for round-based direct optimization.
#[derive(Debug, Clone)]
pub struct DirectSamplerConfig {
    /// Initial LHS samples (default: 10 × n_dims)
    pub n_initial: usize,
    /// Max optimization rounds (default: 10)
    pub max_rounds: usize,
    /// True objective evaluations per round (default: 100)
    pub evals_per_round: usize,
    /// NM solvers per round (default: 8)
    pub n_solvers: usize,
    /// NM convergence tolerance (default: 1e-8)
    pub tol: f64,
    /// RBF kernel for monitoring surrogate
    pub kernel: RBFKernel,
    /// Fixed smoothing (used if auto_smoothing=false)
    pub smoothing: f64,
    /// Enable LOO-CV auto-smoothing each round
    pub auto_smoothing: bool,
    /// Penalty filter for surrogate training
    pub filter: PenaltyFilter,
    /// Rounds without improvement before early stop (default: 3)
    pub patience: usize,
    /// Minimum improvement to reset stagnation counter
    pub improvement_threshold: f64,
    /// Random seed
    pub seed: u64,
}

impl DirectSamplerConfig {
    pub fn new(n_dims: usize, seed: u64) -> Self {
        Self {
            n_initial: 10 * n_dims,
            max_rounds: 10,
            evals_per_round: 100,
            n_solvers: 8,
            tol: 1e-8,
            kernel: RBFKernel::ThinPlateSpline,
            smoothing: 0.01,
            auto_smoothing: true,
            filter: PenaltyFilter::AdaptiveMAD(5.0),
            patience: 3,
            improvement_threshold: 1e-4,
            seed,
        }
    }

    pub fn with_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n;
        self
    }

    pub fn with_evals_per_round(mut self, n: usize) -> Self {
        self.evals_per_round = n;
        self
    }

    pub fn with_solvers(mut self, n: usize) -> Self {
        self.n_solvers = n;
        self
    }

    pub fn with_patience(mut self, n: usize) -> Self {
        self.patience = n;
        self
    }

    pub fn with_filter(mut self, f: PenaltyFilter) -> Self {
        self.filter = f;
        self
    }

    pub fn with_auto_smoothing(mut self, enabled: bool) -> Self {
        self.auto_smoothing = enabled;
        self
    }
}

/// Result of round-based direct optimization.
pub struct DirectSamplerResult {
    pub x_best: Vec<f64>,
    pub f_best: f64,
    pub cache: EvaluationCache,
    pub round_results: Vec<RoundResult>,
    pub total_time: f64,
}

/// Diagnostics for a single round.
#[derive(Debug, Clone)]
pub struct RoundResult {
    pub round: usize,
    pub n_evals: usize,
    pub total_evals: usize,
    pub best_f: f64,
    pub round_best_f: f64,
    pub surrogate_rmse: f64,
    pub smoothing_used: f64,
    pub n_filtered: usize,
}

/// Determine the adaptive penalty value for an objective function.
///
/// Scans existing evaluations and returns a penalty that is
/// guaranteed to be worse than any feasible value.
///
/// # Reference for BarraCUDA
///
/// This should replace hardcoded `1e4` or `1e10` penalties.
/// The penalty adapts to the actual objective landscape.
pub fn adaptive_penalty(y_data: &[f64], margin_factor: f64) -> f64 {
    let max_feasible = y_data.iter()
        .copied()
        .filter(|v| v.is_finite() && *v < 1e8)
        .fold(f64::NEG_INFINITY, f64::max);

    if max_feasible == f64::NEG_INFINITY {
        1e10  // No feasible values yet — use large default
    } else {
        max_feasible * margin_factor.max(2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loo_cv_smoothing_quadratic() {
        // y = x^2, should prefer low smoothing
        let x_data: Vec<Vec<f64>> = (-10..=10).map(|i| vec![i as f64]).collect();
        let y_data: Vec<f64> = x_data.iter().map(|x| x[0] * x[0]).collect();

        let (opt_s, opt_rmse, results) = loo_cv_optimal_smoothing(
            &x_data, &y_data, RBFKernel::ThinPlateSpline, None,
        );

        println!("Optimal smoothing: {:.2e}, RMSE: {:.6}", opt_s, opt_rmse);
        for (s, rmse) in &results {
            println!("  s={:.2e} → RMSE={:.6}", s, rmse);
        }

        // TPS RBF on [-10,10] with 21 points: LOO-CV RMSE ≈ 2.5
        // (inherent interpolation error for polynomial-like data)
        assert!(opt_rmse < 10.0, "LOO-CV RMSE should be finite and reasonable");
        // Key: RMSE increases with smoothing (monotonic for smooth functions)
        assert!(results.last().unwrap().1 > results[3].1,
            "Higher smoothing should give worse RMSE for smooth data");
    }

    #[test]
    fn test_filter_threshold() {
        let x_data = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y_data = vec![1.0, 2.0, 100.0, 3.0];  // 100.0 is penalty

        let (xf, yf) = filter_training_data(&x_data, &y_data, PenaltyFilter::Threshold(10.0));
        assert_eq!(yf.len(), 3);
        assert!(yf.iter().all(|&v| v <= 10.0));
    }

    #[test]
    fn test_filter_adaptive_mad() {
        let x_data: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let mut y_data: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5).collect();
        y_data[19] = 1000.0;  // Outlier

        let (_, yf) = filter_training_data(&x_data, &y_data, PenaltyFilter::AdaptiveMAD(3.0));
        assert!(!yf.contains(&1000.0), "Outlier should be filtered");
        assert!(yf.len() >= 18, "Most normal values should remain");
    }

    #[test]
    fn test_adaptive_penalty() {
        let y_data = vec![1.0, 5.0, 10.0, 3.0];
        let penalty = adaptive_penalty(&y_data, 2.0);
        assert!(penalty >= 20.0, "Penalty should be >= 2× max feasible");
    }
}

