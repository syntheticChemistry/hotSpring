// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ensemble-level statistics: jackknife errors and autocorrelation estimates.

use serde::{Deserialize, Serialize};

/// Statistical analysis metadata for ensemble-level observables.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// Number of configurations in the analysis.
    pub n_configs: usize,
    /// Integrated autocorrelation time (in units of measurement interval).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tau_int: Option<f64>,
    /// Error on tau_int.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tau_int_error: Option<f64>,
    /// Error estimation method ("jackknife", "bootstrap", "binning", "autocorr").
    pub error_method: String,
    /// Number of jackknife/bootstrap samples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_samples: Option<usize>,
    /// Bin size used (for binning analysis).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bin_size: Option<usize>,
}

/// Ensemble-level summary of an observable with statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservableSummary {
    /// Observable name (e.g. "plaquette", "t0", "w0", "Q", "pbp").
    pub name: String,
    /// Central value (mean).
    pub value: f64,
    /// Statistical error.
    pub error: f64,
    /// Statistical analysis metadata.
    pub analysis: StatisticalAnalysis,
}

/// Compute jackknife error estimate for a set of measurements.
///
/// Returns (mean, jackknife_error).
pub fn jackknife_error(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n < 2 {
        return (data.first().copied().unwrap_or(0.0), 0.0);
    }

    let total: f64 = data.iter().sum();
    let mean = total / n as f64;

    let mut jk_means = Vec::with_capacity(n);
    for i in 0..n {
        let jk_sum = total - data[i];
        jk_means.push(jk_sum / (n - 1) as f64);
    }

    let jk_mean: f64 = jk_means.iter().sum::<f64>() / n as f64;
    let jk_var: f64 = jk_means
        .iter()
        .map(|&jk| (jk - jk_mean).powi(2))
        .sum::<f64>()
        * (n - 1) as f64
        / n as f64;

    (mean, jk_var.sqrt())
}

/// Estimate integrated autocorrelation time using binning analysis.
///
/// Returns (tau_int, tau_int_error) where tau_int is in units of
/// the measurement separation.
pub fn estimate_tau_int(data: &[f64]) -> (f64, f64) {
    let n = data.len();
    if n < 10 {
        return (1.0, 0.0);
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let naive_var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    if naive_var < 1e-30 {
        return (1.0, 0.0);
    }

    let mut best_tau = 1.0;
    let mut best_tau_err = 0.0;

    // Try increasing bin sizes
    let max_bin = (n / 4).max(2);
    for bin_size in 1..=max_bin {
        let n_bins = n / bin_size;
        if n_bins < 4 {
            break;
        }

        let mut bin_means = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            let start = b * bin_size;
            let end = start + bin_size;
            let bin_mean: f64 = data[start..end].iter().sum::<f64>() / bin_size as f64;
            bin_means.push(bin_mean);
        }

        let bin_var: f64 = bin_means
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (n_bins - 1) as f64;

        let tau = 0.5 * bin_size as f64 * bin_var / naive_var;
        let tau_err = tau * (2.0 / n_bins as f64).sqrt();

        if tau > best_tau {
            best_tau = tau;
            best_tau_err = tau_err;
        }
    }

    (best_tau, best_tau_err)
}
