// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::md::reservoir::NpuSimulator;
use crate::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;
use super::types::BetaResult;

/// Compute sample variance of plaquette history.
///
/// Delegates to [`barracuda::stats::correlation::variance`] (Bessel, n−1); empty or
/// single-point histories yield `0.0` to match production callers' expectations.
#[must_use]
pub fn plaquette_variance(history: &[f64]) -> f64 {
    barracuda::stats::correlation::variance(history).unwrap_or(0.0)
}

/// Statistical convergence check on plaquette window.
/// Uses variance-ratio and drift tests — mimics what the ESN therm detector learned.
#[must_use]
pub fn check_thermalization(plaq_window: &[f64], _beta: f64) -> bool {
    if plaq_window.len() < 10 {
        return false;
    }
    let n = plaq_window.len();
    let mean = plaq_window.iter().sum::<f64>() / n as f64;
    let var = plaq_window.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    let half = n / 2;
    let mean_first = plaq_window[..half].iter().sum::<f64>() / half as f64;
    let mean_second = plaq_window[half..].iter().sum::<f64>() / (n - half) as f64;
    let drift = (mean_second - mean_first).abs();

    let relative_var = if mean.abs() > 1e-12 {
        var.sqrt() / mean.abs()
    } else {
        var.sqrt()
    };

    relative_var < 0.02 && drift < 0.005
}

/// Predict trajectory rejection from observables.
/// Uses empirical heuristics: large |ΔH| and low acceptance rate predict rejection.
#[must_use]
pub fn predict_rejection(
    _beta: f64,
    _plaquette: f64,
    _action_density: f64,
    delta_h: f64,
    acceptance_rate: f64,
) -> (bool, f64) {
    let dh_mag = delta_h.abs();
    let rejection_score = if delta_h > 0.0 {
        1.0 - (-dh_mag).exp()
    } else {
        0.0
    };

    let rate_factor = if acceptance_rate < 0.3 {
        1.2
    } else if acceptance_rate < 0.5 {
        1.0
    } else {
        0.8
    };

    let confidence = (rejection_score * rate_factor).clamp(0.0, 1.0);
    let likely_rejected = confidence > 0.8;

    (likely_rejected, confidence)
}

/// Build ESN training data from accumulated β-scan results.
/// Features: [`β_norm`, plaquette, polyakov, `susceptibility_norm`]
/// Targets: [phase (0=confined, 1=deconfined), `beta_c_proximity`]
#[must_use]
pub fn build_training_data(results: &[BetaResult]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for r in results {
        let beta_norm = (r.beta - 5.0) / 2.0;
        let phase = if r.beta > KNOWN_BETA_C { 1.0 } else { 0.0 };
        let proximity = (-(r.beta - KNOWN_BETA_C).powi(2) / 0.1).exp();

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.005 * (f64::from(j) * 0.7).sin();
                vec![
                    beta_norm,
                    r.mean_plaq + noise * r.std_plaq,
                    r.polyakov + noise * 0.01,
                    r.susceptibility / 1000.0,
                ]
            })
            .collect();
        seqs.push(seq);
        targets.push(vec![phase, proximity]);
    }

    (seqs, targets)
}

/// Predict `β_c` by scanning NPU predictions and finding phase boundary.
pub fn predict_beta_c(npu: &mut NpuSimulator) -> f64 {
    let n_scan = 100;
    let mut best_beta = KNOWN_BETA_C;
    let mut best_uncertainty = 0.0_f64;

    for i in 0..n_scan {
        let beta = 5.0 + 2.0 * f64::from(i) / (f64::from(n_scan) - 1.0);
        let beta_norm = (beta - 5.0) / 2.0;
        let plaq_est = 0.35 + 0.25 * (beta - 4.5) / 2.5;
        let poly_est = if beta > 5.7 { 0.3 } else { 0.05 };
        let chi_est = if (beta - KNOWN_BETA_C).abs() < 0.2 {
            500.0
        } else {
            50.0
        };

        let seq: Vec<Vec<f64>> = vec![vec![beta_norm, plaq_est, poly_est, chi_est / 1000.0]; 10];
        let pred = npu.predict(&seq);
        if pred.len() >= 2 {
            let uncertainty = pred[1];
            if uncertainty > best_uncertainty {
                best_uncertainty = uncertainty;
                best_beta = beta;
            }
        } else if !pred.is_empty() {
            let phase_pred = pred[0];
            let u = (phase_pred - 0.5).abs().mul_add(-2.0, 1.0);
            if u > best_uncertainty {
                best_uncertainty = u;
                best_beta = beta;
            }
        }
    }

    best_beta
}

/// Find β with maximum NPU uncertainty among unmeasured regions.
pub fn find_max_uncertainty_beta(
    npu: &mut NpuSimulator,
    measured: &[f64],
    beta_min: f64,
    beta_max: f64,
    n_candidates: usize,
) -> f64 {
    let mut best_beta = f64::NAN;
    let mut best_score = 0.0_f64;

    for i in 0..n_candidates {
        let beta = beta_min + (beta_max - beta_min) * (i as f64) / (n_candidates as f64 - 1.0);

        if measured.iter().any(|&m| (m - beta).abs() < 0.08) {
            continue;
        }

        let beta_norm = (beta - 5.0) / 2.0;
        let plaq_est = 0.35 + 0.25 * (beta - 4.5) / 2.5;
        let poly_est = if beta > 5.7 { 0.3 } else { 0.05 };
        let chi_est = if (beta - KNOWN_BETA_C).abs() < 0.2 {
            500.0
        } else {
            50.0
        };

        let seq: Vec<Vec<f64>> = vec![vec![beta_norm, plaq_est, poly_est, chi_est / 1000.0]; 10];
        let pred = npu.predict(&seq);

        let uncertainty = if pred.len() >= 2 {
            pred[1]
        } else if !pred.is_empty() {
            (pred[0] - 0.5).abs().mul_add(-2.0, 1.0)
        } else {
            0.0
        };

        let proximity_bonus = (-(beta - KNOWN_BETA_C).powi(2) / 0.5).exp() * 0.3;
        let score = uncertainty + proximity_bonus;

        if score > best_score {
            best_score = score;
            best_beta = beta;
        }
    }

    best_beta
}

#[cfg(test)]
mod tests {
    use crate::production::types::BetaResult;
    use super::{
        build_training_data, check_thermalization, plaquette_variance, predict_rejection,
    };

    #[test]
    fn plaquette_variance_constant_is_zero() {
        let history = vec![0.5; 100];
        let v = plaquette_variance(&history);
        assert!(v.abs() < 1e-15, "constant → zero variance, got {v}");
    }

    #[test]
    fn plaquette_variance_known_values() {
        let history = vec![1.0, 3.0];
        let v = plaquette_variance(&history);
        assert!((v - 2.0).abs() < 1e-14, "var([1,3]) = 2, got {v}");
    }

    #[test]
    fn plaquette_variance_single_element_is_zero() {
        assert!((plaquette_variance(&[1.0])).abs() < 1e-15);
    }

    #[test]
    fn plaquette_variance_empty_is_zero() {
        assert!((plaquette_variance(&[])).abs() < 1e-15);
    }

    #[test]
    fn check_thermalization_too_short_returns_false() {
        assert!(!check_thermalization(&[0.5; 5], 6.0));
    }

    #[test]
    fn check_thermalization_stable_window_returns_true() {
        let window: Vec<f64> = (0..100)
            .map(|i| 0.5 + 0.0001 * (i as f64 * 0.1).sin())
            .collect();
        assert!(check_thermalization(&window, 6.0));
    }

    #[test]
    fn check_thermalization_drifting_returns_false() {
        let window: Vec<f64> = (0..100).map(|i| 0.3 + 0.3 * i as f64 / 100.0).collect();
        assert!(!check_thermalization(&window, 6.0));
    }

    #[test]
    fn predict_rejection_high_delta_h_low_acceptance() {
        let (rejected, confidence) = predict_rejection(6.0, 0.5, 0.5, 5.0, 0.1);
        assert!(rejected, "high |ΔH| + low acc → reject");
        assert!(confidence > 0.8, "should have high confidence");
    }

    #[test]
    fn predict_rejection_negative_delta_h() {
        let (rejected, confidence) = predict_rejection(6.0, 0.5, 0.5, -1.0, 0.7);
        assert!(!rejected, "negative ΔH → accept");
        assert!(confidence < 0.5, "low confidence for negative ΔH");
    }

    #[test]
    fn predict_rejection_zero_delta_h() {
        let (_rejected, confidence) = predict_rejection(6.0, 0.5, 0.5, 0.0, 0.7);
        assert!(confidence < 0.1, "zero ΔH → near-zero confidence");
    }

    fn make_beta_result(beta: f64) -> BetaResult {
        BetaResult {
            beta,
            mean_plaq: 0.4 + 0.02 * beta,
            std_plaq: 0.01,
            polyakov: if beta > 5.7 { 0.3 } else { 0.05 },
            susceptibility: 100.0,
            acceptance: 0.7,
            n_traj: 50,
            wall_s: 10.0,
            ..Default::default()
        }
    }

    #[test]
    fn build_training_data_shapes() {
        let results = vec![make_beta_result(5.0), make_beta_result(6.5)];
        let (seqs, targets) = build_training_data(&results);
        assert_eq!(seqs.len(), 2, "one sequence per BetaResult");
        assert_eq!(targets.len(), 2);
        assert_eq!(seqs[0].len(), 10, "sequence length = 10");
        assert_eq!(seqs[0][0].len(), 4, "4 features per frame");
        assert_eq!(targets[0].len(), 2, "2 target values: phase + proximity");
    }

    #[test]
    fn build_training_data_phase_labels() {
        let confined = make_beta_result(5.0);
        let deconfined = make_beta_result(6.5);
        let (_, targets) = build_training_data(&[confined, deconfined]);
        assert!((targets[0][0] - 0.0).abs() < 1e-10, "β=5.0 → confined");
        assert!((targets[1][0] - 1.0).abs() < 1e-10, "β=6.5 → deconfined");
    }
}
