// SPDX-License-Identifier: AGPL-3.0-only

//! L2 heterogeneous pipeline helpers
//!
//! Shared logic for the nuclear EOS L2 heterogeneous binary:
//! L1 training data generation, classifier training, and L2 objective evaluation.

use crate::physics::hfb::binding_energy_l2;
use crate::physics::{nuclear_matter_properties, semf_binding_energy};
use crate::prescreen::{nmp_objective_penalty, PreScreenClassifier};
use crate::tolerances;
use barracuda::sample::latin_hypercube;
use std::collections::HashMap;
use std::time::Instant;

/// Result of classifier training, including quality metrics.
///
/// The `usable` field indicates whether the classifier should be used in Tier 3;
/// if false, the cascade bypasses the classifier.
#[derive(Debug)]
pub struct ClassifierResult {
    /// Trained classifier
    pub classifier: PreScreenClassifier,
    /// True if recall is high enough to trust the classifier
    pub usable: bool,
}

/// Generate L1 (SEMF) training data for warm-start and classifier training.
///
/// Samples the parameter space via Latin hypercube, computes SEMF chi-squared + NMP penalty
/// for each, and returns (params, log(1+chi2)) pairs.
///
/// # Errors
///
/// Returns `HotSpringError` if Latin hypercube sampling fails.
pub fn generate_l1_training_data<S: std::hash::BuildHasher>(
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64), S>,
    n_samples: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), crate::error::HotSpringError> {
    println!("  Phase 1: Generating L1 training data ({n_samples} samples)...");
    let t0 = Instant::now();

    let samples: Vec<Vec<f64>> = latin_hypercube(n_samples, bounds, 42)
        .map_err(|e| crate::error::HotSpringError::InvalidOperation(format!("LHS: {e}")))?;
    let mut xs = Vec::with_capacity(n_samples);
    let mut ys = Vec::with_capacity(n_samples);

    for sample in &samples {
        let params: Vec<f64> = sample.clone();

        if params[8] <= 0.01 || params[8] > 1.0 {
            xs.push(params);
            ys.push(9.2); // ln(1+10000) penalty
            continue;
        }

        let Some(nmp) = nuclear_matter_properties(&params) else {
            xs.push(params);
            ys.push(9.2);
            continue;
        };

        let penalty = nmp_objective_penalty(&nmp);

        let mut chi2 = 0.0;
        let mut n_valid = 0;
        for (&(z, n), &(b_exp, _sigma)) in exp_data {
            let b_calc = semf_binding_energy(z, n, &params);
            if b_calc > 0.0 {
                let sigma_theo = tolerances::sigma_theo(b_exp);
                chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
                n_valid += 1;
            }
        }

        let obj = if n_valid > 0 {
            (chi2 / f64::from(n_valid) + penalty).ln_1p()
        } else {
            9.2
        };

        xs.push(params);
        ys.push(obj);
    }

    let n_good = ys.iter().filter(|&&y| y < 5.0).count();
    println!(
        "    Generated {} samples in {:.2}s",
        n_samples,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "    Good (log(1+χ²) < 5): {} ({:.1}%)",
        n_good,
        100.0 * n_good as f64 / n_samples as f64
    );
    println!();

    Ok((xs, ys))
}

/// Train the pre-screening classifier on L1 (params, obj) data.
///
/// Uses label threshold 7.0. If recall is poor, lowers the decision threshold
/// until recall ≥ 50% (or gives up). Returns `ClassifierResult` with `usable`
/// indicating whether Tier 3 should be used.
pub fn train_classifier(xs: &[Vec<f64>], ys: &[f64]) -> ClassifierResult {
    println!("  Phase 2: Training pre-screening classifier...");
    let t0 = Instant::now();

    let label_threshold = 7.0;
    let mut classifier = PreScreenClassifier::train(xs, ys, label_threshold);

    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_ = 0;

    for (x, &y) in xs.iter().zip(ys.iter()) {
        let pred = classifier.predict(x);
        let actual = y < label_threshold;
        if pred && actual {
            tp += 1;
        }
        if pred && !actual {
            fp += 1;
        }
        if !pred && actual {
            fn_ += 1;
        }
        if !pred && !actual {
            tn += 1;
        }
    }

    let n_positive = tp + fn_;
    let recall = if n_positive > 0 {
        f64::from(tp) / f64::from(n_positive)
    } else {
        0.0
    };
    let precision = if tp + fp > 0 {
        f64::from(tp) / f64::from(tp + fp)
    } else {
        0.0
    };
    let accuracy = f64::from(tp + tn) / xs.len().max(1) as f64;

    println!(
        "    Trained in {:.3}s on {} samples ({} positive at threshold={:.1})",
        t0.elapsed().as_secs_f64(),
        xs.len(),
        n_positive,
        label_threshold
    );
    println!("    Accuracy:  {:.1}%", accuracy * 100.0);
    println!("    Precision: {:.1}%", precision * 100.0);
    println!("    Recall:    {:.1}%", recall * 100.0);
    println!("    [TP={tp}, FP={fp}, FN={fn_}, TN={tn}]");

    let usable = if recall < 0.5 && n_positive > 5 {
        let mut best_threshold = 0.5;
        let mut best_recall = recall;
        for t in (5..50).map(|i| f64::from(i) / 100.0) {
            classifier.threshold = t;
            let mut tp2 = 0;
            let mut fn2 = 0;
            for (x, &y) in xs.iter().zip(ys.iter()) {
                let pred = classifier.predict(x);
                let actual = y < label_threshold;
                if pred && actual {
                    tp2 += 1;
                }
                if !pred && actual {
                    fn2 += 1;
                }
            }
            let r = if tp2 + fn2 > 0 {
                f64::from(tp2) / f64::from(tp2 + fn2)
            } else {
                0.0
            };
            if r > best_recall {
                best_recall = r;
                best_threshold = t;
            }
            if r >= 0.5 {
                break;
            }
        }
        classifier.threshold = best_threshold;
        println!(
            "    ⚠ Low recall — adjusted threshold to {:.2} (recall now {:.1}%)",
            best_threshold,
            best_recall * 100.0
        );
        best_recall >= 0.1
    } else if n_positive <= 5 {
        println!("    ⚠ Too few positives ({n_positive}) — classifier DISABLED (bypass Tier 3)");
        false
    } else {
        true
    };

    if !usable {
        println!("    → Cascade will skip Tier 3 (classifier), using Tiers 1+2 only");
    }
    println!();

    ClassifierResult { classifier, usable }
}

/// L2 objective: log(1 + χ²/datum + NMP_penalty) from HFB binding energies.
///
/// Evaluates spherical HFB for each nucleus, computes χ² versus experiment,
/// adds soft NMP penalty, returns ln(1+χ²+penalty).
pub fn l2_objective(params: &[f64], nuclei: &[(usize, usize, f64)]) -> f64 {
    let Some(nmp) = nuclear_matter_properties(params) else {
        return (1e4_f64).ln_1p();
    };

    let penalty = nmp_objective_penalty(&nmp);

    let results: Vec<(f64, f64)> = nuclei
        .iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, _conv) =
                binding_energy_l2(z, n, params).map_or((0.0, false), |(b, c)| (b, c));
            (b_calc, b_exp)
        })
        .collect();

    let mut chi2 = 0.0;
    let mut n_valid = 0;
    for (b_calc, b_exp) in results {
        if b_calc > 0.0 {
            let sigma_theo = tolerances::sigma_theo(b_exp);
            chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
            n_valid += 1;
        }
    }

    if n_valid == 0 {
        return (1e4_f64).ln_1p();
    }

    (chi2 / f64::from(n_valid) + penalty).ln_1p()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::provenance::SLY4_PARAMS;
    use std::collections::HashMap;

    fn skyrm_bounds_10d() -> Vec<(f64, f64)> {
        vec![
            (-3000.0, -1000.0),
            (200.0, 600.0),
            (-600.0, 0.0),
            (10000.0, 18000.0),
            (-1.0, 1.5),
            (-2.0, 1.0),
            (-2.0, 0.0),
            (-1.0, 2.5),
            (0.1, 0.5),
            (50.0, 200.0),
        ]
    }

    fn minimal_exp_data() -> HashMap<(usize, usize), (f64, f64)> {
        let mut m = HashMap::new();
        m.insert((8, 8), (127.62, 1.0));
        m.insert((28, 28), (483.99, 1.0));
        m.insert((82, 126), (1636.43, 1.0));
        m
    }

    #[test]
    fn train_classifier_produces_usable_result() {
        let xs: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                SLY4_PARAMS
                    .iter()
                    .map(|&v| v + (f64::from(i) - 10.0) * 0.01 * v.abs().max(1.0))
                    .collect()
            })
            .collect();
        let ys: Vec<f64> = (0..20).map(|i| if i < 10 { 5.0 } else { 50.0 }).collect();

        let result = train_classifier(&xs, &ys);
        assert_eq!(result.classifier.n_train, 20);
        assert!(result.classifier.weights.len() == 10);
    }

    #[test]
    fn l2_objective_sly4_finite() {
        let nuclei = vec![(8, 8, 127.62), (28, 28, 483.99), (82, 126, 1636.43)];
        let obj = l2_objective(&SLY4_PARAMS, &nuclei);
        assert!(obj.is_finite());
        assert!(obj > 0.0);
    }

    #[test]
    fn l2_objective_bad_params_returns_high_penalty() {
        let bad_params = vec![
            -5000.0, 1000.0, -2000.0, 20000.0, 2.0, -2.0, 1.0, 2.0, 0.5, 200.0,
        ];
        let nuclei = vec![(8, 8, 127.62)];
        let obj = l2_objective(&bad_params, &nuclei);
        assert!(obj.is_finite());
        assert!(obj > 5.0, "bad params should yield high objective");
    }

    #[test]
    fn l2_objective_empty_nuclei() {
        let nuclei: Vec<(usize, usize, f64)> = vec![];
        let obj = l2_objective(&SLY4_PARAMS, &nuclei);
        assert!(obj.is_finite());
        assert!(obj >= (1e4_f64).ln_1p() - 0.01);
    }

    #[test]
    fn generate_l1_training_data_small_run() {
        let bounds = skyrm_bounds_10d();
        let exp_data = minimal_exp_data();
        let result = generate_l1_training_data(&bounds, &exp_data, 10);
        assert!(result.is_ok());
        let (xs, ys) = result.unwrap();
        assert_eq!(xs.len(), 10);
        assert_eq!(ys.len(), 10);
        assert!(xs.iter().all(|x| x.len() == 10));
        assert!(ys.iter().all(|&y| y.is_finite()));
    }

    #[test]
    fn l2_objective_determinism() {
        let nuclei = vec![(28, 28, 483.99), (82, 126, 1636.43)];
        let a = l2_objective(&SLY4_PARAMS, &nuclei);
        let b = l2_objective(&SLY4_PARAMS, &nuclei);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}
