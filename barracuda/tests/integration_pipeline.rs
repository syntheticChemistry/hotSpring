// SPDX-License-Identifier: AGPL-3.0-only

//! Integration tests: L2 heterogeneous pipeline public API.
//!
//! Validates pipeline module: L1 training data generation, classifier training,
//! and L2 objective evaluation across module boundaries.

use hotspring_barracuda::pipeline::{generate_l1_training_data, l2_objective, train_classifier};
use hotspring_barracuda::provenance::SLY4_PARAMS;
use std::collections::HashMap;

fn skyrm_bounds() -> Vec<(f64, f64)> {
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
fn pipeline_generate_l1_data_succeeds() {
    let bounds = skyrm_bounds();
    let exp_data = minimal_exp_data();
    let result = generate_l1_training_data(&bounds, &exp_data, 20);
    assert!(result.is_ok());
    let (xs, ys) = result.unwrap();
    assert_eq!(xs.len(), 20);
    assert_eq!(ys.len(), 20);
}

#[test]
fn pipeline_train_classifier_succeeds() {
    let xs: Vec<Vec<f64>> = (0..15)
        .map(|i| {
            SLY4_PARAMS
                .iter()
                .map(|&v| v + (f64::from(i) - 7.0) * 0.01 * v.abs().max(1.0))
                .collect()
        })
        .collect();
    let ys: Vec<f64> = (0..15).map(|i| if i < 8 { 5.0 } else { 50.0 }).collect();
    let result = train_classifier(&xs, &ys);
    assert!(result.classifier.n_train == 15);
}

#[test]
fn pipeline_l2_objective_sly4_finite() {
    let nuclei = vec![(8, 8, 127.62), (28, 28, 483.99), (82, 126, 1636.43)];
    let obj = l2_objective(&SLY4_PARAMS, &nuclei);
    assert!(obj.is_finite());
    assert!(obj > 0.0);
}

#[test]
fn pipeline_l2_objective_deterministic() {
    let nuclei = vec![(28, 28, 483.99), (82, 126, 1636.43)];
    let a = l2_objective(&SLY4_PARAMS, &nuclei);
    let b = l2_objective(&SLY4_PARAMS, &nuclei);
    assert_eq!(a.to_bits(), b.to_bits());
}
