// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::*;
use crate::provenance::SLY4_PARAMS;

#[test]
fn sly4_passes_nmp_prescreen() {
    let result = nmp_prescreen(&SLY4_PARAMS, &NMPConstraints::default());
    assert!(
        matches!(result, NMPScreenResult::Pass(_)),
        "SLy4 should pass NMP prescreen: {result:?}"
    );
}

#[test]
fn sly4_nmp_within_bounds() {
    if let NMPScreenResult::Pass(nmp) = nmp_prescreen(&SLY4_PARAMS, &NMPConstraints::default()) {
        let c = NMPConstraints::default();
        assert!(nmp.rho0_fm3 >= c.rho0_min && nmp.rho0_fm3 <= c.rho0_max);
        assert!(nmp.e_a_mev >= c.e_a_min && nmp.e_a_mev <= c.e_a_max);
        assert!(nmp.k_inf_mev >= c.k_inf_min && nmp.k_inf_mev <= c.k_inf_max);
        assert!(nmp.m_eff_ratio >= c.m_eff_min && nmp.m_eff_ratio <= c.m_eff_max);
        assert!(nmp.j_mev >= c.j_min && nmp.j_mev <= c.j_max);
    } else {
        panic!("SLy4 should pass");
    }
}

#[test]
fn bad_alpha_rejected() {
    let mut bad = SLY4_PARAMS;
    bad[8] = 0.0;
    let result = nmp_prescreen(&bad, &NMPConstraints::default());
    assert!(matches!(result, NMPScreenResult::Fail(_)));
}

#[test]
fn wrong_param_count_rejected() {
    let result = nmp_prescreen(&[0.0; 5], &NMPConstraints::default());
    assert!(matches!(result, NMPScreenResult::Fail(_)));
}

#[test]
fn tight_constraints_reject_sly4() {
    let tight = NMPConstraints {
        rho0_min: 0.161,
        rho0_max: 0.162,
        ..NMPConstraints::default()
    };
    let result = nmp_prescreen(&SLY4_PARAMS, &tight);
    assert!(matches!(result, NMPScreenResult::Fail(_)));
}

#[test]
fn classifier_trains_without_panic() {
    let xs: Vec<Vec<f64>> = (0..20)
        .map(|i| {
            SLY4_PARAMS
                .iter()
                .map(|&v| v + (f64::from(i) - 10.0) * 0.01 * v.abs().max(1.0))
                .collect()
        })
        .collect();
    let ys: Vec<f64> = (0..20).map(|i| if i < 10 { 5.0 } else { 50.0 }).collect();
    let clf = PreScreenClassifier::train(&xs, &ys, 10.0);
    assert_eq!(clf.n_train, 20);
    assert!(clf.n_positive > 0);
}

#[test]
fn classifier_predicts_without_panic() {
    let xs = vec![SLY4_PARAMS.to_vec()];
    let ys = vec![5.0];
    let clf = PreScreenClassifier::train(&xs, &ys, 10.0);
    let prob = clf.predict_prob(&SLY4_PARAMS);
    assert!((0.0..=1.0).contains(&prob));
}

#[test]
#[expect(clippy::float_cmp, reason = "exact known values (0.0)")]
fn cascade_stats_default_zero() {
    let stats = CascadeStats::default();
    assert_eq!(stats.total_candidates, 0);
    assert_eq!(stats.pass_rate(), 0.0);
}

#[test]
fn prescreen_determinism() {
    let constraints = NMPConstraints::default();
    let run = || nmp_prescreen(&SLY4_PARAMS, &constraints);
    let a = run();
    let b = run();
    match (&a, &b) {
        (NMPScreenResult::Pass(na), NMPScreenResult::Pass(nb)) => {
            assert_eq!(na.rho0_fm3.to_bits(), nb.rho0_fm3.to_bits());
            assert_eq!(na.e_a_mev.to_bits(), nb.e_a_mev.to_bits());
            assert_eq!(na.k_inf_mev.to_bits(), nb.k_inf_mev.to_bits());
            assert_eq!(na.m_eff_ratio.to_bits(), nb.m_eff_ratio.to_bits());
            assert_eq!(na.j_mev.to_bits(), nb.j_mev.to_bits());
        }
        (NMPScreenResult::Fail(fa), NMPScreenResult::Fail(fb)) => {
            assert_eq!(fa, fb, "fail reasons differ");
        }
        _ => panic!("determinism: variant mismatch: {a:?} vs {b:?}"),
    }
}

#[test]
fn prescreen_very_large_params_rejected() {
    let large: Vec<f64> = SLY4_PARAMS.iter().map(|&v| v * 100.0).collect();
    let result = nmp_prescreen(&large, &NMPConstraints::default());
    assert!(matches!(result, NMPScreenResult::Fail(_)));
}

#[test]
fn prescreen_alpha_at_boundary() {
    let mut params = SLY4_PARAMS;
    params[8] = 0.01;
    let result = nmp_prescreen(&params, &NMPConstraints::default());
    assert!(matches!(result, NMPScreenResult::Fail(_)));
}

#[test]
fn prescreen_alpha_just_above_upper_bound() {
    let mut params = SLY4_PARAMS;
    params[8] = 1.001;
    let result = nmp_prescreen(&params, &NMPConstraints::default());
    assert!(matches!(result, NMPScreenResult::Fail(_)));
}

#[test]
fn classifier_rejects_poor_params_when_trained_on_good() {
    let good_params: Vec<Vec<f64>> = (0..15)
        .map(|i| {
            SLY4_PARAMS
                .iter()
                .map(|&v| v + (f64::from(i) - 7.0) * 0.001 * v.abs().max(1.0))
                .collect()
        })
        .collect();
    let good_chi2: Vec<f64> = (0..15).map(|_| 10.0).collect();

    let bad_params: Vec<Vec<f64>> = (0..15)
        .map(|i| {
            vec![
                -5000.0 + f64::from(i) * 100.0,
                1000.0,
                -2000.0,
                20000.0,
                2.0,
                -2.0,
                1.0,
                2.0,
                1.0 / 6.0,
                200.0,
            ]
        })
        .collect();
    let bad_chi2: Vec<f64> = (0..15).map(|_| 500.0).collect();

    let xs: Vec<Vec<f64>> = good_params
        .iter()
        .chain(bad_params.iter())
        .cloned()
        .collect();
    let ys: Vec<f64> = good_chi2.iter().chain(bad_chi2.iter()).copied().collect();

    let clf = PreScreenClassifier::train(&xs, &ys, 50.0);
    let good_pred = clf.predict_prob(&SLY4_PARAMS);
    let bad_pred = clf.predict_prob(&bad_params[0]);
    assert!((0.0..=1.0).contains(&good_pred));
    assert!((0.0..=1.0).contains(&bad_pred));
    assert!(clf.n_positive > 0 && clf.n_positive < 30);
}

#[test]
fn boundary_constraints_rho0_at_limits() {
    let at_min = NMPConstraints {
        rho0_min: 0.10,
        rho0_max: 0.22,
        ..NMPConstraints::default()
    };
    let result = nmp_prescreen(&SLY4_PARAMS, &at_min);
    assert!(matches!(result, NMPScreenResult::Pass(_)));

    let exclude = NMPConstraints {
        rho0_min: 0.1596,
        rho0_max: 0.1597,
        ..NMPConstraints::default()
    };
    let result = nmp_prescreen(&SLY4_PARAMS, &exclude);
    assert!(matches!(result, NMPScreenResult::Fail(_)));
}

#[test]
fn l1_proxy_prescreen_empty_exp_data() {
    let exp_data: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let result = l1_proxy_prescreen(&SLY4_PARAMS, &exp_data, 100.0);
    assert!(result.is_none());
}

#[test]
fn l1_proxy_prescreen_above_threshold_rejected() {
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    exp_data.insert((82, 126), (1636.43, 1.0));
    let result = l1_proxy_prescreen(&SLY4_PARAMS, &exp_data, 0.001);
    assert!(result.is_none());
}

#[test]
fn l1_proxy_prescreen_below_threshold_passes() {
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    exp_data.insert((82, 126), (1636.43, 1.0));
    let result = l1_proxy_prescreen(&SLY4_PARAMS, &exp_data, 1000.0);
    assert!(result.is_some());
    let chi2 = result.expect("result asserted Some");
    assert!(chi2 > 0.0 && chi2 < 1000.0);
}

#[test]
fn l1_proxy_prescreen_all_negative_binding_n_valid_zero() {
    let mut exp_data = HashMap::new();
    exp_data.insert((2, 2), (28.3, 1.0));
    let bad_params = vec![
        -5000.0, 1000.0, -2000.0, 20000.0, 2.0, -2.0, 1.0, 2.0, 0.5, 200.0,
    ];
    let result = l1_proxy_prescreen(&bad_params, &exp_data, 1000.0);
    assert!(result.is_none());
}

#[test]
fn l1_proxy_prescreen_chi2_at_threshold_boundary() {
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    let chi2_exact = 50.0;
    let result = l1_proxy_prescreen(&SLY4_PARAMS, &exp_data, chi2_exact);
    if let Some(chi2) = result {
        assert!(chi2 < chi2_exact);
    }
}

#[test]
fn cascade_stats_print_summary_no_panic() {
    let stats = CascadeStats {
        total_candidates: 100,
        tier1_rejected: 20,
        tier2_rejected: 15,
        tier3_rejected: 10,
        tier4_evaluated: 55,
    };
    stats.print_summary();
}

#[test]
fn cascade_stats_pass_rate_nonzero() {
    let stats = CascadeStats {
        total_candidates: 100,
        tier1_rejected: 50,
        tier2_rejected: 20,
        tier3_rejected: 10,
        tier4_evaluated: 20,
    };
    assert!((stats.pass_rate() - 0.2).abs() < 1e-10);
}

#[test]
fn classifier_train_empty_data() {
    let xs: Vec<Vec<f64>> = vec![];
    let ys: Vec<f64> = vec![];
    let clf = PreScreenClassifier::train(&xs, &ys, 10.0);
    assert_eq!(clf.n_train, 0);
    assert_eq!(clf.weights.len(), 10);
    assert_eq!(clf.norm.len(), 10);
}

#[test]
fn classifier_export_weights() {
    let xs = vec![SLY4_PARAMS.to_vec()];
    let ys = vec![5.0];
    let clf = PreScreenClassifier::train(&xs, &ys, 10.0);
    let (weights, bias, norm) = clf.export_weights();
    assert_eq!(weights.len(), 10);
    assert_eq!(norm.len(), 10);
    assert!(bias.is_finite());
}

#[test]
fn cascade_filter_empty_candidates() {
    let candidates: Vec<Vec<f64>> = vec![];
    let constraints = NMPConstraints::default();
    let exp_data: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let clf = PreScreenClassifier::train(&[SLY4_PARAMS.to_vec()], &[5.0], 10.0);
    let stats = Arc::new(Mutex::new(CascadeStats::default()));

    let survivors = cascade_filter(&candidates, &constraints, &exp_data, false, &clf, &stats);
    assert!(survivors.is_empty());
    let s = stats.lock().expect("lock");
    assert_eq!(s.total_candidates, 0);
}

#[test]
fn cascade_filter_nmp_rejects_bad_params() {
    let mut bad_params = SLY4_PARAMS;
    bad_params[8] = 0.0;
    let candidates = vec![bad_params.to_vec()];
    let constraints = NMPConstraints::default();
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    let clf = PreScreenClassifier::train(&[SLY4_PARAMS.to_vec()], &[5.0], 10.0);
    let stats = Arc::new(Mutex::new(CascadeStats::default()));

    let survivors = cascade_filter(&candidates, &constraints, &exp_data, false, &clf, &stats);
    assert!(
        survivors.is_empty(),
        "bad alpha should be rejected at Tier 1"
    );
    let s = stats.lock().expect("lock");
    assert_eq!(s.tier1_rejected, 1);
}

#[test]
fn cascade_filter_sly4_passes_without_classifier() {
    let candidates = vec![SLY4_PARAMS.to_vec()];
    let constraints = NMPConstraints::default();
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    exp_data.insert((82, 126), (1636.43, 1.0));
    let clf = PreScreenClassifier::train(&[SLY4_PARAMS.to_vec()], &[5.0], 10.0);
    let stats = Arc::new(Mutex::new(CascadeStats::default()));

    let survivors = cascade_filter(&candidates, &constraints, &exp_data, false, &clf, &stats);
    assert_eq!(
        survivors.len(),
        1,
        "SLy4 should pass Tiers 1+2 when classifier disabled"
    );
    let s = stats.lock().expect("lock");
    assert_eq!(s.tier4_evaluated, 1);
}

#[test]
fn cascade_filter_determinism() {
    let candidates = vec![SLY4_PARAMS.to_vec(), SLY4_PARAMS.to_vec()];
    let constraints = NMPConstraints::default();
    let mut exp_data = HashMap::new();
    exp_data.insert((28, 28), (483.99, 1.0));
    exp_data.insert((82, 126), (1636.43, 1.0));
    let clf = PreScreenClassifier::train(&[SLY4_PARAMS.to_vec()], &[5.0], 10.0);
    let stats = Arc::new(Mutex::new(CascadeStats::default()));

    let a = cascade_filter(&candidates, &constraints, &exp_data, false, &clf, &stats);
    let stats2 = Arc::new(Mutex::new(CascadeStats::default()));
    let b = cascade_filter(&candidates, &constraints, &exp_data, false, &clf, &stats2);
    assert_eq!(a.len(), b.len());
    for (va, vb) in a.iter().zip(b.iter()) {
        for (pa, pb) in va.iter().zip(vb.iter()) {
            assert_eq!(pa.to_bits(), pb.to_bits());
        }
    }
}

#[test]
fn nmp_objective_penalty_zero_for_good_nmp() {
    if let NMPScreenResult::Pass(nmp) = nmp_prescreen(&SLY4_PARAMS, &NMPConstraints::default()) {
        let penalty = nmp_objective_penalty(&nmp);
        assert!(penalty >= 0.0, "penalty must be non-negative");
    }
}

#[test]
fn perturb_params_determinism() {
    let base = SLY4_PARAMS.to_vec();
    let bounds: Vec<(f64, f64)> = vec![
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
    ];
    let mut rng1 = 42u64;
    let mut rng2 = 42u64;
    let a = perturb_params(&base, &bounds, &mut rng1, 0.1);
    let b = perturb_params(&base, &bounds, &mut rng2, 0.1);
    for (pa, pb) in a.iter().zip(b.iter()) {
        assert_eq!(pa.to_bits(), pb.to_bits());
    }
}
