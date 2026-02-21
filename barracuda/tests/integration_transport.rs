// SPDX-License-Identifier: AGPL-3.0-only

//! Integration tests: transport coefficients and reservoir computing.
//!
//! Exercises the Daligault fit, ESN reservoir, and MD configuration pipeline
//! end-to-end.

use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig};
use hotspring_barracuda::md::transport::d_star_daligault;
use hotspring_barracuda::tolerances;

#[test]
fn daligault_d_star_positive_for_physical_params() {
    let cases = [(1.0, 10.0), (2.0, 50.0), (3.0, 100.0)];
    for (kappa, gamma) in cases {
        let d = d_star_daligault(gamma, kappa);
        assert!(
            d > 0.0,
            "D*(Γ={gamma}, κ={kappa}) must be positive, got {d}"
        );
    }
}

#[test]
fn daligault_d_star_decreases_with_coupling() {
    let d_weak = d_star_daligault(1.0, 1.0);
    let d_strong = d_star_daligault(100.0, 1.0);
    assert!(
        d_weak > d_strong,
        "D* should decrease with coupling: Γ=1 → {d_weak}, Γ=100 → {d_strong}"
    );
}

#[test]
fn esn_config_default_uses_centralized_regularization() {
    let config = EsnConfig::default();
    assert!(
        (config.regularization - tolerances::ESN_REGULARIZATION).abs() < f64::EPSILON,
        "EsnConfig default regularization should match ESN_REGULARIZATION"
    );
}

#[test]
fn esn_train_and_predict_round_trip() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 10,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.5,
        leak_rate: 0.3,
        regularization: tolerances::ESN_REGULARIZATION,
        seed: 42,
    };
    let mut esn = EchoStateNetwork::new(config);

    let train_seqs = vec![
        vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
        vec![vec![-0.1, -0.2], vec![-0.3, -0.4], vec![-0.5, -0.6]],
    ];
    let train_targets = vec![vec![1.0], vec![-1.0]];
    esn.train(&train_seqs, &train_targets);

    let test_seq = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
    let pred = esn.predict(&test_seq);
    assert!(pred.is_ok(), "prediction should succeed after training");
    let pred = pred.unwrap();
    assert_eq!(pred.len(), 1, "should predict 1 output");
    assert!(pred[0].is_finite(), "prediction must be finite");
}

#[test]
fn esn_deterministic_predictions() {
    let config = EsnConfig::default();

    let train_seqs = vec![vec![vec![0.1; 8], vec![0.2; 8], vec![0.3; 8]]];
    let train_targets = vec![vec![0.5]];

    let mut esn1 = EchoStateNetwork::new(config.clone());
    esn1.train(&train_seqs, &train_targets);
    let pred1 = esn1.predict(&train_seqs[0]).expect("predict");

    let mut esn2 = EchoStateNetwork::new(config);
    esn2.train(&train_seqs, &train_targets);
    let pred2 = esn2.predict(&train_seqs[0]).expect("predict");

    assert_eq!(
        pred1[0].to_bits(),
        pred2[0].to_bits(),
        "ESN predictions must be deterministic"
    );
}
