// SPDX-License-Identifier: AGPL-3.0-only

#![allow(clippy::expect_used)]

use crate::md::reservoir::heads;
use crate::md::reservoir::{
    spectral_radius_estimate, velocity_features, EchoStateNetwork, EsnConfig, MultiHeadNpu,
    NpuSimulator, Xoshiro256pp,
};

#[test]
fn esn_trains_and_predicts() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-4,
        seed: 42,
    };
    let mut esn = EchoStateNetwork::new(config);

    let seq1: Vec<Vec<f64>> = (0..100)
        .map(|t| {
            let x = f64::from(t) * 0.1;
            vec![x.sin(), x.cos()]
        })
        .collect();
    let seq2: Vec<Vec<f64>> = (0..100)
        .map(|t| {
            let x = f64::from(t) * 0.2;
            vec![x.sin(), x.cos()]
        })
        .collect();

    esn.train(&[seq1.clone(), seq2.clone()], &[vec![1.0], vec![2.0]]);

    let p1 = esn.predict(&seq1).expect("ESN trained");
    let p2 = esn.predict(&seq2).expect("ESN trained");

    assert!((p1[0] - 1.0).abs() < 0.5, "p1={}", p1[0]);
    assert!((p2[0] - 2.0).abs() < 0.5, "p2={}", p2[0]);
}

#[test]
fn esn_default_config() {
    let config = EsnConfig::default();
    assert_eq!(config.reservoir_size, 50);
    assert_eq!(config.input_size, 8);
    assert!((config.spectral_radius - 0.95).abs() < 1e-10);
}

#[test]
fn velocity_features_correct_shape() {
    let n = 10;
    let frame = vec![0.1; n * 3];
    let features = velocity_features(&[frame], n, 2.0, 100.0);
    assert_eq!(features.len(), 1);
    assert_eq!(features[0].len(), 8);
}

#[test]
fn export_and_npu_parity() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 30,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-4,
        seed: 42,
    };
    let mut esn = EchoStateNetwork::new(config);

    let seq1: Vec<Vec<f64>> = (0..50)
        .map(|t| {
            let x = f64::from(t) * 0.1;
            vec![x.sin(), x.cos()]
        })
        .collect();
    let seq2: Vec<Vec<f64>> = (0..50)
        .map(|t| {
            let x = f64::from(t) * 0.2;
            vec![x.sin(), x.cos()]
        })
        .collect();

    let sequences = vec![seq1, seq2];
    esn.train(&sequences, &[vec![1.0], vec![2.0]]);

    let exported = esn.export_weights().expect("ESN export_weights");
    assert_eq!(exported.w_in.len(), 30 * 2);
    assert_eq!(exported.w_res.len(), 30 * 30);
    assert_eq!(exported.w_out.len(), 30);

    let mut npu = NpuSimulator::from_exported(&exported);
    let cpu_pred = esn.predict(&sequences[0]).expect("ESN trained")[0];
    let npu_pred = npu.predict(&sequences[0])[0];
    let diff = (cpu_pred - npu_pred).abs() / cpu_pred.abs().max(1e-10);
    assert!(diff < 0.01, "CPU/NPU diff {diff} should be < 1%");
}

#[test]
fn esn_benchmark_vs_python() {
    use std::time::Instant;

    let n_features = 8;
    let n_frames = 500;
    let n_cases = 6;

    let config = EsnConfig {
        input_size: n_features,
        reservoir_size: 50,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
    };

    let mut rng = Xoshiro256pp::new(99);
    let sequences: Vec<Vec<Vec<f64>>> = (0..n_cases)
        .map(|_| {
            (0..n_frames)
                .map(|_| (0..n_features).map(|_| rng.standard_normal()).collect())
                .collect()
        })
        .collect();
    let targets: Vec<Vec<f64>> = (0..n_cases).map(|_| vec![rng.standard_normal()]).collect();

    let n_reps = if cfg!(debug_assertions) { 3 } else { 100 };
    let t0 = Instant::now();
    for _ in 0..n_reps {
        let mut esn = EchoStateNetwork::new(config.clone());
        esn.train(&sequences[..4], &targets[..4]);
        for seq in &sequences {
            let _ = esn.predict(seq).expect("ESN trained");
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_iter_ms = elapsed * 1000.0 / f64::from(n_reps);

    println!("Rust ESN: {n_reps} reps in {elapsed:.3}s");
    println!("  Per iteration (train+6 predict): {per_iter_ms:.1} ms");
    if !cfg!(debug_assertions) {
        assert!(
            per_iter_ms < 50.0,
            "ESN should complete in <50ms per iteration (release)"
        );
    }
}

#[test]
fn spectral_radius_identity() {
    let n = 5;
    let w: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let sr = spectral_radius_estimate(&w);
    assert!((sr - 1.0).abs() < 0.01, "sr={sr}");
}

#[test]
fn velocity_features_known_values() {
    let vels = vec![vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]; // 2 particles
    let feats = velocity_features(&vels, 2, 1.5, 100.0);
    assert_eq!(feats.len(), 1);
    let f = &feats[0];
    assert_eq!(f.len(), 8);
    assert!((f[0] - 0.5).abs() < 1e-10, "mean_vx: {}", f[0]); // (1+0)/2
    assert!((f[1] - 0.5).abs() < 1e-10, "mean_vy: {}", f[1]); // (0+1)/2
    assert!((f[2] - 0.0).abs() < 1e-10, "mean_vz: {}", f[2]); // (0+0)/2
    assert!((f[3] - 1.0).abs() < 1e-10, "mean_speed: {}", f[3]); // (1+1)/2
    assert!((f[4] - 0.5).abs() < 1e-10, "ke_per_particle: {}", f[4]); // (0.5+0.5)/2
    let v_rms_expected = 1.0; // sqrt((1+1)/2) = 1.0
    assert!((f[5] - v_rms_expected).abs() < 1e-10, "v_rms: {}", f[5]);
    let kappa_scaled = 1.5 / 3.0;
    assert!(
        (f[6] - kappa_scaled).abs() < 1e-10,
        "kappa_scaled: {}",
        f[6]
    );
    let gamma_scaled = 100.0_f64.log10() / 3.0;
    assert!(
        (f[7] - gamma_scaled).abs() < 1e-10,
        "gamma_scaled: {}",
        f[7]
    );
}

#[test]
fn velocity_features_multiple_frames() {
    let frame1 = vec![3.0, 0.0, 0.0]; // 1 particle, speed=3
    let frame2 = vec![0.0, 4.0, 0.0]; // 1 particle, speed=4
    let feats = velocity_features(&[frame1, frame2], 1, 1.0, 1.0);
    assert_eq!(feats.len(), 2);
    assert!((feats[0][3] - 3.0).abs() < 1e-10, "frame1 speed");
    assert!((feats[1][3] - 4.0).abs() < 1e-10, "frame2 speed");
}

#[test]
#[allow(clippy::expect_used)]
fn npu_predict_return_state_consistent() {
    let config = EsnConfig {
        input_size: 3,
        reservoir_size: 15,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.3,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 77,
    };
    let seq: Vec<Vec<f64>> = (0..5)
        .map(|i| vec![f64::from(i) * 0.1, 0.5, -0.2])
        .collect();
    let mut esn = EchoStateNetwork::new(config);
    esn.train(std::slice::from_ref(&seq), &[vec![1.0]]);
    let exported = esn.export_weights().expect("export");

    let mut npu1 = NpuSimulator::from_exported(&exported);
    let mut npu2 = NpuSimulator::from_exported(&exported);

    let pred = npu1.predict(&seq);
    let state = npu2.predict_return_state(&seq);

    assert_eq!(state.len(), exported.reservoir_size);
    let readout_from_state: f64 = exported
        .w_out
        .chunks(exported.reservoir_size)
        .next()
        .expect("w_out row")
        .iter()
        .zip(state.iter())
        .map(|(&w, &s)| f64::from(w) * f64::from(s))
        .sum();
    assert!(
        (pred[0] - readout_from_state).abs() < 1e-4,
        "predict vs state readout: {} vs {}",
        pred[0],
        readout_from_state
    );
}

#[test]
fn multi_head_esn_nine_outputs() {
    let config = EsnConfig {
        input_size: 4,
        reservoir_size: 50,
        output_size: heads::NUM_HEADS,
        spectral_radius: 0.95,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-3,
        seed: 42,
    };
    let mut esn = EchoStateNetwork::new(config);

    let seqs: Vec<Vec<Vec<f64>>> = (0..5)
        .map(|i| {
            (0..10)
                .map(|t| {
                    let x = (t as f64 + i as f64) * 0.1;
                    vec![x.sin(), x.cos(), x * 0.5, (x * 2.0).sin()]
                })
                .collect()
        })
        .collect();
    let targets: Vec<Vec<f64>> = (0..5)
        .map(|i| {
            (0..heads::NUM_HEADS)
                .map(|h| (i + h) as f64 * 0.1)
                .collect()
        })
        .collect();

    esn.train(&seqs, &targets);
    let exported = esn.export_weights().expect("export");
    assert_eq!(exported.output_size, heads::NUM_HEADS);
    assert_eq!(exported.w_out.len(), heads::NUM_HEADS * 50);

    let mut multi = MultiHeadNpu::from_exported(&exported);
    let outputs = multi.predict_all_heads(&seqs[0]);
    assert_eq!(outputs.len(), heads::NUM_HEADS);

    let head_2 = multi.predict_head(&seqs[0], heads::THERM_DETECT);
    assert!(head_2.is_finite());
}

#[test]
fn npu_readout_weight_swap() {
    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 20,
        output_size: 1,
        spectral_radius: 0.9,
        connectivity: 0.2,
        leak_rate: 0.3,
        regularization: 1e-3,
        seed: 42,
    };
    let mut esn = EchoStateNetwork::new(config);
    let seq: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1, 0.5]).collect();
    esn.train(std::slice::from_ref(&seq), &[vec![1.0]]);
    let exported = esn.export_weights().expect("export");

    let mut npu = NpuSimulator::from_exported(&exported);
    let pred_before = npu.predict(&seq)[0];

    let swapped_w_out: Vec<f32> = exported.w_out.iter().map(|w| w * 2.0).collect();
    npu.set_readout_weights(&swapped_w_out);
    let pred_after = npu.predict(&seq)[0];

    assert!(
        (pred_after - pred_before * 2.0).abs() < 0.01,
        "doubled weights should double output: {pred_before} -> {pred_after}"
    );
}

#[test]
fn esn_predict_determinism() {
    let config = EsnConfig {
        input_size: 3,
        reservoir_size: 20,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.3,
        leak_rate: 0.3,
        regularization: 1e-2,
        seed: 42,
    };
    let seqs: Vec<Vec<Vec<f64>>> = vec![
        (0..10)
            .map(|i| vec![f64::from(i) * 0.1, 0.5, -0.3])
            .collect(),
        (0..10)
            .map(|i| vec![0.0, f64::from(i) * 0.05, 0.2])
            .collect(),
    ];
    let targets = vec![vec![1.0], vec![0.0]];

    let results: Vec<Vec<f64>> = (0..2)
        .map(|_| {
            let mut esn = EchoStateNetwork::new(config.clone());
            esn.train(&seqs, &targets);
            esn.predict(&seqs[0]).expect("ESN trained")
        })
        .collect();
    assert!(
        (results[0][0] - results[1][0]).abs() < f64::EPSILON,
        "ESN predictions must be identical: {} vs {}",
        results[0][0],
        results[1][0]
    );
}
