//! Reference statistical analysis utilities for BarraCUDA evolution.
//!
//! These implementations demonstrate statistical tests and methods
//! needed for scientific validation workflows. They serve as
//! specification for BarraCUDA's `stats` module expansion.
//!
//! ## Reference Systems:
//!
//! 1. **Chi-squared goodness-of-fit** — With proper error propagation
//! 2. **Bootstrap confidence intervals** — Non-parametric uncertainty
//! 3. **Residual analysis** — Systematic bias detection
//! 4. **Convergence diagnostics** — For iterative optimization

use barracuda::special::{gamma, ln_gamma};

/// Chi-squared per datum with detailed decomposition.
///
/// Returns per-nucleus contributions for identifying which
/// nuclei drive the overall χ² — critical for understanding
/// why a parameter set is good or bad.
///
/// # Reference for BarraCUDA
///
/// This should be in `barracuda::stats::chi_squared`:
/// ```ignore
/// pub fn chi2_decomposed(observed: &[f64], expected: &[f64], sigma: &[f64])
///     -> Chi2Decomposition
/// ```
pub fn chi2_decomposed(
    observed: &[f64],
    expected: &[f64],
    sigma: &[f64],
) -> Chi2Result {
    assert_eq!(observed.len(), expected.len());
    assert_eq!(observed.len(), sigma.len());
    let n = observed.len();

    let mut contributions: Vec<Chi2Contribution> = Vec::with_capacity(n);
    let mut total_chi2 = 0.0;
    let mut n_valid = 0;

    for i in 0..n {
        if sigma[i] > 0.0 {
            let residual = observed[i] - expected[i];
            let pull = residual / sigma[i];
            let chi2_i = pull * pull;

            contributions.push(Chi2Contribution {
                index: i,
                observed: observed[i],
                expected: expected[i],
                sigma: sigma[i],
                residual,
                pull,
                chi2: chi2_i,
            });

            total_chi2 += chi2_i;
            n_valid += 1;
        }
    }

    let chi2_per_datum = if n_valid > 0 { total_chi2 / n_valid as f64 } else { f64::INFINITY };

    // Sort contributions by chi2 (worst first)
    contributions.sort_by(|a, b| b.chi2.partial_cmp(&a.chi2).unwrap());

    Chi2Result {
        total_chi2,
        n_data: n_valid,
        chi2_per_datum,
        contributions,
    }
}

/// Result of decomposed chi-squared analysis.
#[derive(Debug, Clone)]
pub struct Chi2Result {
    pub total_chi2: f64,
    pub n_data: usize,
    pub chi2_per_datum: f64,
    pub contributions: Vec<Chi2Contribution>,
}

impl Chi2Result {
    /// Top-N worst-fit data points.
    pub fn worst(&self, n: usize) -> &[Chi2Contribution] {
        &self.contributions[..n.min(self.contributions.len())]
    }

    /// p-value from chi-squared CDF (approximate).
    pub fn p_value(&self) -> f64 {
        if self.n_data <= 1 { return 1.0; }
        let dof = self.n_data as f64 - 1.0;
        chi2_survival(self.total_chi2, dof)
    }

    /// Reduced chi-squared (should be ~1.0 for good fit).
    pub fn reduced_chi2(&self) -> f64 {
        if self.n_data <= 1 { return f64::INFINITY; }
        self.total_chi2 / (self.n_data as f64 - 1.0)
    }

    /// Print human-readable summary.
    pub fn print_summary(&self, label: &str) {
        println!("  {} chi-squared analysis:", label);
        println!("    χ² total:    {:.4}", self.total_chi2);
        println!("    χ²/datum:    {:.4}", self.chi2_per_datum);
        println!("    χ²/dof:      {:.4} (reduced, ideal ≈ 1.0)", self.reduced_chi2());
        println!("    N data:      {}", self.n_data);
        println!("    p-value:     {:.6}", self.p_value());
        if !self.contributions.is_empty() {
            println!("    Top-5 worst:");
            for c in self.worst(5) {
                println!("      [{}] obs={:.2}, exp={:.2}, pull={:.2}, χ²={:.2}",
                    c.index, c.observed, c.expected, c.pull, c.chi2);
            }
        }
    }
}

/// Per-datum contribution to chi-squared.
#[derive(Debug, Clone)]
pub struct Chi2Contribution {
    pub index: usize,
    pub observed: f64,
    pub expected: f64,
    pub sigma: f64,
    pub residual: f64,
    pub pull: f64,
    pub chi2: f64,
}

/// Bootstrap confidence interval for a statistic.
///
/// Resamples data with replacement and computes the statistic
/// on each bootstrap sample to estimate uncertainty.
///
/// # Reference for BarraCUDA
///
/// Should be in `barracuda::stats::bootstrap`:
/// ```ignore
/// pub fn bootstrap_ci<F>(data: &[f64], statistic: F, n_boot: usize, alpha: f64) -> BootstrapCI
/// ```
pub fn bootstrap_ci<F>(
    data: &[f64],
    statistic: F,
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> BootstrapCI
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    if n == 0 {
        return BootstrapCI {
            point_estimate: f64::NAN,
            lower: f64::NAN,
            upper: f64::NAN,
            std_error: f64::NAN,
            n_bootstrap,
        };
    }

    let point_estimate = statistic(data);

    // Simple LCG for reproducibility (not cryptographic)
    let mut rng_state = seed;
    let lcg_next = |state: &mut u64| -> usize {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % n
    };

    let mut boot_stats: Vec<f64> = Vec::with_capacity(n_bootstrap);
    let mut sample = vec![0.0; n];

    for _ in 0..n_bootstrap {
        // Resample with replacement
        for j in 0..n {
            sample[j] = data[lcg_next(&mut rng_state)];
        }
        boot_stats.push(statistic(&sample));
    }

    boot_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    let mean: f64 = boot_stats.iter().sum::<f64>() / n_bootstrap as f64;
    let variance = boot_stats.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (n_bootstrap as f64 - 1.0);

    BootstrapCI {
        point_estimate,
        lower: boot_stats[lower_idx.min(n_bootstrap - 1)],
        upper: boot_stats[upper_idx.min(n_bootstrap - 1)],
        std_error: variance.sqrt(),
        n_bootstrap,
    }
}

/// Bootstrap confidence interval result.
#[derive(Debug, Clone)]
pub struct BootstrapCI {
    pub point_estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub std_error: f64,
    pub n_bootstrap: usize,
}

impl BootstrapCI {
    pub fn print_summary(&self, label: &str) {
        println!("  {} = {:.4} [{:.4}, {:.4}] (SE={:.4}, {} bootstraps)",
            label, self.point_estimate, self.lower, self.upper,
            self.std_error, self.n_bootstrap);
    }
}

/// Convergence diagnostics for iterative optimization.
///
/// Analyzes the trajectory of objective function values to
/// determine if the optimizer has converged or is still improving.
///
/// # Reference for BarraCUDA
///
/// Should be in `barracuda::optimize::diagnostics`:
/// ```ignore
/// pub fn convergence_diagnostics(history: &[f64]) -> ConvergenceDiag
/// ```
pub fn convergence_diagnostics(history: &[f64]) -> ConvergenceDiag {
    let n = history.len();
    if n < 2 {
        return ConvergenceDiag {
            is_converged: false,
            rate: f64::NAN,
            final_improvement: f64::NAN,
            relative_change: f64::NAN,
            n_improving: 0,
            n_stagnant: n,
        };
    }

    // Count improving steps
    let mut n_improving = 0;
    let mut n_stagnant = 0;
    for i in 1..n {
        if history[i] < history[i - 1] - 1e-10 {
            n_improving += 1;
        } else {
            n_stagnant += 1;
        }
    }

    // Final improvement (last 3 rounds vs previous 3)
    let final_improvement = if n >= 6 {
        let recent: f64 = history[n-3..].iter().sum::<f64>() / 3.0;
        let earlier: f64 = history[n-6..n-3].iter().sum::<f64>() / 3.0;
        earlier - recent
    } else {
        history[0] - history[n - 1]
    };

    let relative_change = if history[0].abs() > 1e-10 {
        (history[0] - history[n - 1]).abs() / history[0].abs()
    } else {
        f64::NAN
    };

    // Convergence rate (exponential fit: f(n) = a * exp(-rate * n))
    let rate = if n >= 3 && history[0] > history[n - 1] {
        ((history[0] - history[n - 1]).max(1e-20) / history[0].abs().max(1e-20)).ln() / n as f64
    } else {
        0.0
    };

    let is_converged = n_stagnant >= n / 2 || relative_change < 1e-4;

    ConvergenceDiag {
        is_converged,
        rate,
        final_improvement,
        relative_change,
        n_improving,
        n_stagnant,
    }
}

/// Convergence diagnostic results.
#[derive(Debug, Clone)]
pub struct ConvergenceDiag {
    pub is_converged: bool,
    pub rate: f64,
    pub final_improvement: f64,
    pub relative_change: f64,
    pub n_improving: usize,
    pub n_stagnant: usize,
}

// ═══════════════════════════════════════════════════════════════════
// Internal: Chi-squared survival function (upper tail)
// Uses regularized incomplete gamma function
// ═══════════════════════════════════════════════════════════════════

fn chi2_survival(x: f64, dof: f64) -> f64 {
    // P(χ² > x | dof) = 1 - P(χ² ≤ x | dof)
    // = Q(dof/2, x/2) = upper regularized incomplete gamma
    if x <= 0.0 { return 1.0; }
    if dof <= 0.0 { return 0.0; }

    // Use BarraCUDA's regularized gamma if available
    match barracuda::special::regularized_gamma_q(dof / 2.0, x / 2.0) {
        Ok(q) => q,
        Err(_) => {
            // Fallback: use Wilson-Hilferty approximation
            let k = dof;
            let z = ((x / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) /
                    (2.0 / (9.0 * k)).sqrt();
            0.5 * barracuda::special::erfc(z / std::f64::consts::SQRT_2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi2_decomposed() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![10.5, 19.5, 30.5];
        let sigma = vec![1.0, 1.0, 1.0];

        let result = chi2_decomposed(&observed, &expected, &sigma);
        assert_eq!(result.n_data, 3);
        assert!((result.chi2_per_datum - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_mean() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let ci = bootstrap_ci(&data, |d| d.iter().sum::<f64>() / d.len() as f64,
            1000, 0.95, 42);
        assert!((ci.point_estimate - 0.495).abs() < 0.01);
        assert!(ci.lower < ci.point_estimate);
        assert!(ci.upper > ci.point_estimate);
    }

    #[test]
    fn test_convergence_improving() {
        let history = vec![10.0, 8.0, 6.0, 5.0, 4.5, 4.3, 4.2, 4.15];
        let diag = convergence_diagnostics(&history);
        assert!(diag.n_improving >= 5);
        assert!(diag.relative_change > 0.5);
    }
}


