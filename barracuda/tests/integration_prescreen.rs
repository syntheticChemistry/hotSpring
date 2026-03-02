// SPDX-License-Identifier: AGPL-3.0-only
#![allow(clippy::unwrap_used)]

//! Integration tests: NMP cascade filter public API.
//!
//! Validates prescreen module: NMP prescreen, L1 proxy, cascade filter,
//! and classifier across module boundaries.

use hotspring_barracuda::prescreen::{
    cascade_filter, l1_proxy_prescreen, nmp_prescreen, CascadeStats, NMPConstraints,
    NMPScreenResult, PreScreenClassifier,
};
use hotspring_barracuda::provenance::SLY4_PARAMS;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[test]
fn prescreen_sly4_passes_nmp() {
    let result = nmp_prescreen(&SLY4_PARAMS, &NMPConstraints::default());
    assert!(matches!(result, NMPScreenResult::Pass(_)));
}

#[test]
fn prescreen_cascade_filter_empty() {
    let candidates: Vec<Vec<f64>> = vec![];
    let constraints = NMPConstraints::default();
    let exp_data: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let clf = PreScreenClassifier::train(&[SLY4_PARAMS.to_vec()], &[5.0], 10.0);
    let stats = Arc::new(Mutex::new(CascadeStats::default()));

    let survivors = cascade_filter(&candidates, &constraints, &exp_data, false, &clf, &stats);
    assert!(survivors.is_empty());
}

#[test]
fn prescreen_cascade_filter_sly4_passes() {
    let candidates = vec![SLY4_PARAMS.to_vec()];
    let constraints = NMPConstraints::default();
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    exp_data.insert((82, 126), (1636.43, 1.0));
    let clf = PreScreenClassifier::train(&[SLY4_PARAMS.to_vec()], &[5.0], 10.0);
    let stats = Arc::new(Mutex::new(CascadeStats::default()));

    let survivors = cascade_filter(&candidates, &constraints, &exp_data, false, &clf, &stats);
    assert_eq!(survivors.len(), 1);
}

#[test]
fn prescreen_l1_proxy_with_data() {
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    exp_data.insert((82, 126), (1636.43, 1.0));
    let result = l1_proxy_prescreen(&SLY4_PARAMS, &exp_data, 1000.0);
    assert!(result.is_some());
}
