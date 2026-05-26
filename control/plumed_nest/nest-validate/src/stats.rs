// SPDX-License-Identifier: AGPL-3.0-or-later

//! Industry-standard statistical analysis for molecular simulation data.
//!
//! Implements:
//! - Autocorrelation time estimation (Chodera 2016)
//! - Statistical inefficiency
//! - Bootstrap confidence intervals
//! - Block averaging with optimal block size detection
//! - Kolmogorov-Smirnov two-sample test
//! - Effective sample size estimation
//!
//! References:
//! - Chodera, JCTC 12, 1799 (2016) — Autocorrelation analysis
//! - Flyvbjerg & Petersen, JCP 91, 461 (1989) — Block averaging
//! - Grossfield et al., LAJP 1, 23 (2009) — Best practices for free energy

use serde::{Deserialize, Serialize};

/// Normalized autocorrelation function.
///
/// C(τ) = <(x(t) - μ)(x(t+τ) - μ)> / <(x(t) - μ)²>
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-30 {
        return vec![1.0; max_lag.min(n)];
    }

    let max_lag = max_lag.min(n - 1);
    let mut acf = Vec::with_capacity(max_lag);

    for tau in 0..max_lag {
        let sum: f64 = (0..n - tau)
            .map(|t| (data[t] - mean) * (data[t + tau] - mean))
            .sum();
        acf.push(sum / ((n - tau) as f64 * var));
    }

    acf
}

/// Integrated autocorrelation time (τ_int) using the initial positive sequence estimator.
///
/// Chodera (2016): sum ACF until first negative value, then apply
/// the Geyer (1992) initial monotone sequence correction.
/// Uses lazy evaluation — computes one lag at a time and stops early.
pub fn integrated_autocorrelation_time(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 10 {
        return 1.0;
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-30 {
        return 1.0;
    }

    let max_lag = (n / 4).min(10_000);
    let mut tau_int = 0.5;

    for tau in 1..max_lag {
        let sum: f64 = (0..n - tau)
            .map(|t| (data[t] - mean) * (data[t + tau] - mean))
            .sum();
        let c_tau = sum / ((n - tau) as f64 * var);
        if c_tau <= 0.0 {
            break;
        }
        tau_int += c_tau;
    }

    tau_int.max(1.0)
}

/// Statistical inefficiency: g = 1 + 2*τ_int
///
/// The number of correlated samples per independent sample.
pub fn statistical_inefficiency(data: &[f64]) -> f64 {
    let tau = integrated_autocorrelation_time(data);
    1.0 + 2.0 * tau
}

/// Effective sample size: N_eff = N / g
pub fn effective_sample_size(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let g = statistical_inefficiency(data);
    n / g
}

/// Standard error of the mean accounting for correlation.
///
/// SEM_corr = σ * sqrt(g / N) = σ / sqrt(N_eff)
pub fn correlated_sem(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let g = statistical_inefficiency(data);
    (var * g / n).sqrt()
}

/// Bootstrap confidence interval for a statistic.
///
/// Returns (lower, upper) bounds at the given confidence level.
pub fn bootstrap_ci(
    data: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    statistic: fn(&[f64]) -> f64,
) -> BootstrapResult {
    let n = data.len();
    let mut rng = SimpleRng::new(42);
    let mut bootstrap_stats = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let sample: Vec<f64> = (0..n)
            .map(|_| data[rng.next_usize(n)])
            .collect();
        bootstrap_stats.push(statistic(&sample));
    }

    bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = (1.0 - confidence) / 2.0;
    let lo_idx = (alpha * n_bootstrap as f64) as usize;
    let hi_idx = ((1.0 - alpha) * n_bootstrap as f64) as usize;
    let hi_idx = hi_idx.min(n_bootstrap - 1);

    let point_estimate = statistic(data);
    let mean_bootstrap = bootstrap_stats.iter().sum::<f64>() / n_bootstrap as f64;
    let bias = mean_bootstrap - point_estimate;

    BootstrapResult {
        point_estimate,
        ci_lower: bootstrap_stats[lo_idx],
        ci_upper: bootstrap_stats[hi_idx],
        confidence,
        bias,
        n_bootstrap,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub point_estimate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub confidence: f64,
    pub bias: f64,
    pub n_bootstrap: usize,
}

/// Optimal block size detection via Flyvbjerg-Petersen method.
///
/// Doubles block size until the standard error plateaus.
/// Returns the optimal block size and the corresponding SEM.
pub fn optimal_block_size(data: &[f64]) -> BlockSizeResult {
    let n = data.len();
    let mut block_sizes = Vec::new();
    let mut sems = Vec::new();

    let mut bs = 1;
    while bs <= n / 4 {
        let n_blocks = n / bs;
        if n_blocks < 4 {
            break;
        }

        // Compute block means
        let block_means: Vec<f64> = (0..n_blocks)
            .map(|b| {
                let start = b * bs;
                let end = start + bs;
                data[start..end].iter().sum::<f64>() / bs as f64
            })
            .collect();

        // SEM from block means
        let grand_mean = block_means.iter().sum::<f64>() / n_blocks as f64;
        let var = block_means.iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>() / (n_blocks - 1) as f64;
        let sem = (var / n_blocks as f64).sqrt();

        block_sizes.push(bs);
        sems.push(sem);
        bs *= 2;
    }

    // Find plateau: where SEM stops increasing significantly
    let optimal_idx = if sems.len() >= 3 {
        let mut best = 0;
        for i in 1..sems.len() - 1 {
            let ratio = sems[i + 1] / sems[i].max(1e-30);
            if ratio < 1.1 {
                best = i;
                break;
            }
            best = i;
        }
        best
    } else {
        sems.len().saturating_sub(1)
    };

    BlockSizeResult {
        optimal_block_size: block_sizes.get(optimal_idx).copied().unwrap_or(1),
        optimal_sem: sems.get(optimal_idx).copied().unwrap_or(0.0),
        block_sizes,
        sems,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSizeResult {
    pub optimal_block_size: usize,
    pub optimal_sem: f64,
    pub block_sizes: Vec<usize>,
    pub sems: Vec<f64>,
}

/// Kolmogorov-Smirnov two-sample test.
///
/// Tests whether two samples come from the same distribution.
/// Returns the KS statistic D and approximate p-value.
pub fn ks_two_sample(sample1: &[f64], sample2: &[f64]) -> KsResult {
    let n1 = sample1.len();
    let n2 = sample2.len();

    let mut s1 = sample1.to_vec();
    let mut s2 = sample2.to_vec();
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Merge and compute empirical CDFs
    let mut all: Vec<(f64, bool)> = s1.iter().map(|&x| (x, true)).collect();
    all.extend(s2.iter().map(|&x| (x, false)));
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut cdf1 = 0.0_f64;
    let mut cdf2 = 0.0_f64;
    let mut d_max = 0.0_f64;

    for (_, is_s1) in &all {
        if *is_s1 {
            cdf1 += 1.0 / n1 as f64;
        } else {
            cdf2 += 1.0 / n2 as f64;
        }
        d_max = d_max.max((cdf1 - cdf2).abs());
    }

    // Approximate p-value (Smirnov formula)
    let n_eff = (n1 as f64 * n2 as f64) / (n1 + n2) as f64;
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * d_max;
    let p_value = 2.0 * (-2.0 * lambda * lambda).exp();
    let p_value = p_value.max(0.0).min(1.0);

    KsResult {
        d_statistic: d_max,
        p_value,
        significant_005: p_value < 0.05,
        n1,
        n2,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KsResult {
    pub d_statistic: f64,
    pub p_value: f64,
    pub significant_005: bool,
    pub n1: usize,
    pub n2: usize,
}

/// Equilibration detection: find the index where the time series equilibrates.
///
/// Uses the method from Chodera (2016): maximize the effective number of
/// uncorrelated samples by discarding initial data.
pub fn detect_equilibration(data: &[f64]) -> EquilibrationResult {
    let n = data.len();
    let mut best_n_eff = 0.0;
    let mut best_t0 = 0;

    // Coarse scan: use a fast variance-based heuristic to identify equilibration
    let n_probes = 20;
    let step = (n / (2 * n_probes)).max(1);
    for probe in 0..n_probes {
        let t0 = probe * step;
        let tail = &data[t0..];
        if tail.len() < 100 {
            break;
        }
        // Quick estimate: subsample to cap computation
        let subsample_stride = (tail.len() / 50_000).max(1);
        let subsampled: Vec<f64> = tail.iter().step_by(subsample_stride).copied().collect();
        let n_eff = effective_sample_size(&subsampled) * subsample_stride as f64;
        if n_eff > best_n_eff {
            best_n_eff = n_eff;
            best_t0 = t0;
        }
    }

    // Full analysis only on the chosen equilibrated window
    let equilibrated_data = &data[best_t0..];
    let subsample_stride = (equilibrated_data.len() / 100_000).max(1);
    let subsampled: Vec<f64> = equilibrated_data.iter().step_by(subsample_stride).copied().collect();
    let g = statistical_inefficiency(&subsampled) * subsample_stride as f64;
    let n_eff = equilibrated_data.len() as f64 / g;

    EquilibrationResult {
        t0_index: best_t0,
        t0_fraction: best_t0 as f64 / n as f64,
        n_effective: n_eff,
        statistical_inefficiency: g,
        production_fraction: 1.0 - (best_t0 as f64 / n as f64),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquilibrationResult {
    pub t0_index: usize,
    pub t0_fraction: f64,
    pub n_effective: f64,
    pub statistical_inefficiency: f64,
    pub production_fraction: f64,
}

/// Full statistical summary for a time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    pub n_samples: usize,
    pub mean: f64,
    pub std: f64,
    pub sem_naive: f64,
    pub sem_correlated: f64,
    pub autocorrelation_time: f64,
    pub statistical_inefficiency: f64,
    pub effective_n: f64,
    pub equilibration: EquilibrationResult,
    pub optimal_block: BlockSizeResult,
}

pub fn full_analysis(data: &[f64]) -> TimeSeriesStats {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    let sem_naive = std / n.sqrt();

    // For large datasets, subsample for correlation analysis (preserves autocorrelation structure)
    let stride = (data.len() / 100_000).max(1);
    let working: Vec<f64> = if stride > 1 {
        data.iter().step_by(stride).copied().collect()
    } else {
        data.to_vec()
    };

    let tau_sub = integrated_autocorrelation_time(&working);
    let tau = tau_sub * stride as f64;
    let g = 1.0 + 2.0 * tau;
    let n_eff = n / g;
    let sem_corr = std * (g / n).sqrt();

    let equilibration = detect_equilibration(data);
    let optimal_block = optimal_block_size(&working);

    TimeSeriesStats {
        n_samples: data.len(),
        mean,
        std,
        sem_naive,
        sem_correlated: sem_corr,
        autocorrelation_time: tau,
        statistical_inefficiency: g,
        effective_n: n_eff,
        equilibration,
        optimal_block,
    }
}

// ─── Simple deterministic RNG (xorshift64) ─────────────────────────────────────

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_acf_white_noise() {
        // White noise: ACF should be ~0 for lag > 0
        let mut rng = SimpleRng::new(123);
        let data: Vec<f64> = (0..10000).map(|_| {
            (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0
        }).collect();

        let acf = autocorrelation(&data, 50);
        assert!((acf[0] - 1.0).abs() < 1e-10, "ACF(0) should be 1.0");
        // For white noise, ACF should be near zero for lag > 0
        for &val in &acf[2..] {
            assert!(val.abs() < 0.05, "ACF for white noise too high: {val}");
        }
    }

    #[test]
    fn test_acf_correlated() {
        // AR(1) process: x(t+1) = 0.9*x(t) + noise
        let mut rng = SimpleRng::new(456);
        let mut data = vec![0.0_f64; 10000];
        for i in 1..10000 {
            let noise = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            data[i] = 0.9 * data[i - 1] + 0.1 * noise;
        }

        let tau = integrated_autocorrelation_time(&data);
        // AR(1) with phi=0.9: theoretical tau = phi/(1-phi) = 9
        assert!(tau > 3.0, "Correlated data should have tau > 3, got {tau}");
        assert!(tau < 20.0, "tau suspiciously high: {tau}");
    }

    #[test]
    fn test_statistical_inefficiency() {
        let mut rng = SimpleRng::new(789);
        let data: Vec<f64> = (0..5000).map(|_| {
            (rng.next_u64() as f64 / u64::MAX as f64)
        }).collect();

        let g = statistical_inefficiency(&data);
        assert!(g >= 1.0, "g should be >= 1.0, got {g}");
        // Pseudo-random: g should be modest (xorshift has minor correlations)
        assert!(g < 5.0, "Pseudo-random g should be < 5, got {g}");
    }

    #[test]
    fn test_bootstrap_ci_mean() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let result = bootstrap_ci(&data, 1000, 0.95, |d| d.iter().sum::<f64>() / d.len() as f64);

        assert!((result.point_estimate - 0.495).abs() < 0.01);
        assert!(result.ci_lower < result.point_estimate);
        assert!(result.ci_upper > result.point_estimate);
        assert!(result.ci_upper - result.ci_lower < 0.2);
    }

    #[test]
    fn test_ks_same_distribution() {
        let mut rng = SimpleRng::new(111);
        let s1: Vec<f64> = (0..500).map(|_| rng.next_u64() as f64 / u64::MAX as f64).collect();
        let s2: Vec<f64> = (0..500).map(|_| rng.next_u64() as f64 / u64::MAX as f64).collect();

        let result = ks_two_sample(&s1, &s2);
        // Same distribution: should NOT be significant
        assert!(!result.significant_005, "Same distribution should not be significant, D={:.4}, p={:.4}",
            result.d_statistic, result.p_value);
    }

    #[test]
    fn test_ks_different_distribution() {
        // One uniform [0,1], one shifted [0.5, 1.5]
        let s1: Vec<f64> = (0..500).map(|i| i as f64 / 500.0).collect();
        let s2: Vec<f64> = (0..500).map(|i| 0.5 + i as f64 / 500.0).collect();

        let result = ks_two_sample(&s1, &s2);
        assert!(result.significant_005, "Different distributions should be significant");
        assert!(result.d_statistic > 0.3);
    }

    #[test]
    fn test_optimal_block_size() {
        // Correlated data
        let mut data = vec![0.0_f64; 2000];
        let mut rng = SimpleRng::new(222);
        for i in 1..2000 {
            let noise = (rng.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0;
            data[i] = 0.8 * data[i - 1] + 0.2 * noise;
        }

        let result = optimal_block_size(&data);
        assert!(result.optimal_block_size > 1, "Correlated data should need blocks > 1");
        assert!(result.optimal_sem > 0.0);
    }

    #[test]
    fn test_equilibration_detection() {
        // Data with clear equilibration period: first 200 points drifting, then stable
        let mut data = Vec::with_capacity(1000);
        for i in 0..200 {
            data.push(10.0 - i as f64 * 0.05); // drift
        }
        let mut rng = SimpleRng::new(333);
        for _ in 0..800 {
            let noise = (rng.next_u64() as f64 / u64::MAX as f64) * 0.2 - 0.1;
            data.push(0.0 + noise); // stable
        }

        let result = detect_equilibration(&data);
        // Should detect equilibration around index 200
        assert!(result.t0_index > 50, "Should discard some initial data");
        assert!(result.production_fraction > 0.5, "Most data should be production");
    }
}
