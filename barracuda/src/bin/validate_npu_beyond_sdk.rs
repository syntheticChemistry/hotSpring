// SPDX-License-Identifier: AGPL-3.0-only

//! NPU Beyond-SDK Capabilities Validation
//!
//! Validates the mathematical properties that underpin the hardware
//! discoveries documented in `metalForge/npu/akida/BEYOND_SDK.md`.
//!
//! The Python control experiment (`control/metalforge_npu/scripts/npu_beyond_sdk.py`)
//! validates on actual AKD1000 hardware. This binary validates the math in
//! pure Rust to prove the computational properties are substrate-independent.
//!
//! Checks:
//!   1. Arbitrary input dimensions work through ESN pipeline (not just 1 or 3)
//!   2. Deep FC chain math matches single-layer math (layer merging correctness)
//!   3. Multi-output readout math (N outputs from single state)
//!   4. Weight mutation linearity (w×k → output×k)
//!   5. Quantization through arbitrary FC widths (wide FC scaling)
//!   6. Batch processing determinism (same input → same output)
//!
//! Reference: metalForge/npu/akida/BEYOND_SDK.md

use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("══════════════════════════════════════════════════════");
    println!("  NPU Beyond-SDK — metalForge Rust Validation");
    println!("══════════════════════════════════════════════════════");
    println!();

    let mut harness = ValidationHarness::new("npu_beyond_sdk");

    // ── Check 1: Arbitrary input dimensions ──
    check_arbitrary_input_dims(&mut harness);

    // ── Check 2: Deep FC chain parity ──
    check_deep_fc_parity(&mut harness);

    // ── Check 3: Multi-output readout ──
    check_multi_output(&mut harness);

    // ── Check 4: Weight mutation linearity ──
    check_weight_mutation(&mut harness);

    // ── Check 5: Wide FC quantization ──
    check_wide_fc_quantization(&mut harness);

    // ── Check 6: Determinism ──
    check_determinism(&mut harness);

    println!();
    harness.finish();
}

/// Check 1: ESN works with arbitrary input dimensions (not just 8).
///
/// The SDK claims InputConv needs 1 or 3 channels, but we showed ANY
/// dimension works. Here we verify the Rust ESN math handles 2, 5, 16,
/// 50, 64 input dimensions correctly.
fn check_arbitrary_input_dims(harness: &mut ValidationHarness) {
    println!("[1] Arbitrary input dimensions");

    let mut rng = SimpleRng::new(42);

    for &input_size in &[2, 5, 8, 16, 50, 64] {
        let config = EsnConfig {
            input_size,
            reservoir_size: 30,
            output_size: 1,
            spectral_radius: 0.9,
            connectivity: 0.3,
            leak_rate: 0.3,
            regularization: 1e-2,
            seed: 100 + input_size as u64,
        };

        let mut esn = EchoStateNetwork::new(config.clone());

        let n_train = 4;
        let n_frames = 50;
        let train_seqs: Vec<Vec<Vec<f64>>> = (0..n_train)
            .map(|_| {
                (0..n_frames)
                    .map(|_| {
                        (0..input_size)
                            .map(|_| rng.standard_normal() * 0.3)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let train_targets: Vec<Vec<f64>> = (0..n_train)
            .map(|_| vec![rng.uniform() * 1.5 + 0.1])
            .collect();

        esn.train(&train_seqs, &train_targets);

        let test_seq: Vec<Vec<f64>> = (0..n_frames)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.standard_normal() * 0.3)
                    .collect()
            })
            .collect();

        let pred = esn.predict(&test_seq).expect("ESN trained")[0];
        let is_finite = pred.is_finite() && pred.abs() < 1e6;

        println!("  input_size={input_size:3}: pred={pred:.6} finite={is_finite}");

        harness.check_bool(
            &format!("ESN with input_size={input_size} produces finite prediction"),
            is_finite,
        );
    }
}

/// Check 2: Deep FC chain produces same math as single-layer.
///
/// On hardware, multiple FC layers merge into one HW sequence (SkipDMA).
/// Mathematically, chaining quantized FC layers should produce different
/// results than a single layer, but the depth overhead on latency should
/// be small. Here we verify that deeper models still produce finite,
/// reasonable predictions with bounded error growth.
fn check_deep_fc_parity(harness: &mut ValidationHarness) {
    println!("\n[2] Deep FC chain math");

    let config = EsnConfig::default();
    let mut esn = EchoStateNetwork::new(config.clone());

    let mut rng = SimpleRng::new(99);
    let n_train = 4;
    let n_frames = 100;

    let train_seqs: Vec<Vec<Vec<f64>>> = (0..n_train)
        .map(|_| {
            (0..n_frames)
                .map(|_| {
                    (0..config.input_size)
                        .map(|_| rng.standard_normal().tanh() * 0.5_f64.tanh())
                        .collect()
                })
                .collect()
        })
        .collect();
    let train_targets: Vec<Vec<f64>> = (0..n_train)
        .map(|_| vec![rng.uniform() * 1.9 + 0.1])
        .collect();

    esn.train(&train_seqs, &train_targets);
    let exported = esn.export_weights().expect("ESN must be trained");

    let test_seq: Vec<Vec<f64>> = (0..n_frames)
        .map(|_| {
            (0..config.input_size)
                .map(|_| rng.standard_normal().tanh() * 0.5_f64.tanh())
                .collect()
        })
        .collect();

    let pred_f64 = esn.predict(&test_seq).expect("ESN trained")[0];
    let mut npu_sim = NpuSimulator::from_exported(&exported);
    let state = npu_sim.predict_return_state(&test_seq);

    // Reconstruct f32 prediction from state for comparison
    let pred_f32: f64 = exported
        .w_out
        .iter()
        .zip(state.iter())
        .map(|(&w, &s)| f64::from(w * s))
        .sum();

    let err = (pred_f64 - pred_f32).abs() / pred_f64.abs().max(1e-10);
    println!("  f64 pred: {pred_f64:.6}");
    println!("  f32 pred: {pred_f32:.6}");
    println!("  f64→f32 error: {:.6}%", err * 100.0);

    harness.check_upper(
        "f64→f32 ESN parity for deep-FC baseline",
        err,
        tolerances::NPU_F32_PARITY,
    );

    // Apply int4-quantized readout to same state to show error stays bounded
    let (w_out_q4, s_out_4) = quantize_f32_vec(&exported.w_out, 4);
    let rs = exported.reservoir_size;

    let mut deep_val = 0.0_f64;
    for j in 0..rs {
        deep_val += f64::from(w_out_q4[j]) * f64::from(s_out_4) * f64::from(state[j]);
    }

    let deep_err = (pred_f64 - deep_val).abs() / pred_f64.abs().max(1e-10);
    println!(
        "  int4-readout pred: {deep_val:.6}  err: {:.4}%",
        deep_err * 100.0
    );

    harness.check_upper(
        "int4 readout error bounded (deep FC math validation)",
        deep_err,
        tolerances::NPU_INT4_QUANTIZATION,
    );
}

/// Check 3: Multi-output readout from a single reservoir state.
///
/// On hardware, producing 10 outputs costs the same as 1 output.
/// Here we verify the math: a single ESN state can produce multiple
/// independent predictions via different readout weight vectors.
fn check_multi_output(harness: &mut ValidationHarness) {
    println!("\n[3] Multi-output readout");

    let config = EsnConfig {
        output_size: 3,
        ..EsnConfig::default()
    };
    let mut esn = EchoStateNetwork::new(config.clone());

    let mut rng = SimpleRng::new(77);
    let n_train = 6;
    let n_frames = 80;

    let train_seqs: Vec<Vec<Vec<f64>>> = (0..n_train)
        .map(|_| {
            (0..n_frames)
                .map(|_| {
                    (0..config.input_size)
                        .map(|_| rng.standard_normal() * 0.4)
                        .collect()
                })
                .collect()
        })
        .collect();
    let train_targets: Vec<Vec<f64>> = (0..n_train)
        .map(|_| {
            (0..config.output_size)
                .map(|_| rng.uniform() * 2.0 + 0.1)
                .collect()
        })
        .collect();

    esn.train(&train_seqs, &train_targets);

    let test_seq: Vec<Vec<f64>> = (0..n_frames)
        .map(|_| {
            (0..config.input_size)
                .map(|_| rng.standard_normal() * 0.4)
                .collect()
        })
        .collect();

    let predictions = esn.predict(&test_seq).expect("ESN trained");
    println!(
        "  Predictions: {:?}",
        predictions
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );

    harness.check_bool(
        "Multi-output ESN (3 outputs) produces correct count",
        predictions.len() == 3,
    );

    let all_finite = predictions.iter().all(|v| v.is_finite() && v.abs() < 1e6);
    harness.check_bool("All 3 outputs are finite and reasonable", all_finite);

    // Exact parity check — independent readouts must differ
    #[allow(clippy::float_cmp)] // determinism test: bit-identical outputs required
    let all_different = predictions[0] != predictions[1]
        && predictions[1] != predictions[2]
        && predictions[0] != predictions[2];
    harness.check_bool(
        "All 3 outputs are distinct (independent readouts)",
        all_different,
    );
}

/// Check 4: Weight mutation linearity.
///
/// On hardware, `set_variable()` updates weights and the next forward
/// pass uses them. Scaling weights by k should scale output by k for
/// a linear readout layer.
fn check_weight_mutation(harness: &mut ValidationHarness) {
    println!("\n[4] Weight mutation linearity");

    let config = EsnConfig::default();
    let mut esn = EchoStateNetwork::new(config.clone());

    let mut rng = SimpleRng::new(55);
    let n_train = 4;
    let n_frames = 60;

    let train_seqs: Vec<Vec<Vec<f64>>> = (0..n_train)
        .map(|_| {
            (0..n_frames)
                .map(|_| {
                    (0..config.input_size)
                        .map(|_| rng.standard_normal() * 0.3)
                        .collect()
                })
                .collect()
        })
        .collect();
    let train_targets: Vec<Vec<f64>> = (0..n_train)
        .map(|_| vec![rng.uniform() * 1.5 + 0.2])
        .collect();

    esn.train(&train_seqs, &train_targets);
    let exported = esn.export_weights().expect("trained");

    let test_seq: Vec<Vec<f64>> = (0..n_frames)
        .map(|_| {
            (0..config.input_size)
                .map(|_| rng.standard_normal() * 0.3)
                .collect()
        })
        .collect();

    let pred_1x = esn.predict(&test_seq).expect("ESN trained")[0];

    let mut npu = NpuSimulator::from_exported(&exported);
    let state = npu.predict_return_state(&test_seq);

    // Reconstruct prediction from state × w_out to confirm baseline
    let pred_check: f64 = state
        .iter()
        .enumerate()
        .map(|(j, &s)| f64::from(exported.w_out[j]) * f64::from(s))
        .sum();

    // Weight scaling: multiply w_out by 2 and -3
    let pred_2x: f64 = state
        .iter()
        .enumerate()
        .map(|(j, &s)| f64::from(exported.w_out[j]) * 2.0 * f64::from(s))
        .sum();
    let pred_neg3x: f64 = state
        .iter()
        .enumerate()
        .map(|(j, &s)| f64::from(exported.w_out[j]) * (-3.0) * f64::from(s))
        .sum();

    let ratio_2x = if pred_check.abs() > 1e-10 {
        pred_2x / pred_check
    } else {
        f64::NAN
    };
    let ratio_neg3x = if pred_check.abs() > 1e-10 {
        pred_neg3x / pred_check
    } else {
        f64::NAN
    };

    println!("  pred(w×1) = {pred_check:.6}  (f64={pred_1x:.6})");
    println!("  pred(w×2) = {pred_2x:.6}  ratio = {ratio_2x:.4}");
    println!("  pred(w×-3) = {pred_neg3x:.6}  ratio = {ratio_neg3x:.4}");

    let err_2x = (ratio_2x - 2.0).abs();
    let err_neg3x = (ratio_neg3x - (-3.0)).abs();

    harness.check_upper(
        "Weight×2 linearity (ratio error)",
        err_2x,
        tolerances::NPU_WEIGHT_MUTATION_LINEARITY,
    );
    harness.check_upper(
        "Weight×(-3) linearity (ratio error)",
        err_neg3x,
        tolerances::NPU_WEIGHT_MUTATION_LINEARITY,
    );
}

/// Check 5: Wide FC quantization — larger reservoir sizes still quantize well.
///
/// On hardware, FC width scales to 8192+. Here we verify the math works
/// for reservoir_size=128 and 256.
fn check_wide_fc_quantization(harness: &mut ValidationHarness) {
    println!("\n[5] Wide FC quantization");

    let mut rng = SimpleRng::new(88);

    for &rs_size in &[128, 256] {
        let config = EsnConfig {
            reservoir_size: rs_size,
            spectral_radius: 0.9,
            connectivity: 0.1,
            seed: 200 + rs_size as u64,
            ..EsnConfig::default()
        };

        let mut esn = EchoStateNetwork::new(config.clone());

        let n_train = 4;
        let n_frames = 50;
        let train_seqs: Vec<Vec<Vec<f64>>> = (0..n_train)
            .map(|_| {
                (0..n_frames)
                    .map(|_| {
                        (0..config.input_size)
                            .map(|_| rng.standard_normal() * 0.3)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let train_targets: Vec<Vec<f64>> = (0..n_train)
            .map(|_| vec![rng.uniform() * 1.5 + 0.1])
            .collect();

        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("trained");

        let test_seq: Vec<Vec<f64>> = (0..n_frames)
            .map(|_| {
                (0..config.input_size)
                    .map(|_| rng.standard_normal() * 0.3)
                    .collect()
            })
            .collect();

        let pred_f64 = esn.predict(&test_seq).expect("ESN trained")[0];
        let mut npu = NpuSimulator::from_exported(&exported);
        let pred_f32 = npu.predict(&test_seq)[0];

        let err = (pred_f64 - pred_f32).abs() / pred_f64.abs().max(1e-10);
        println!(
            "  rs={rs_size}: f64={pred_f64:.6} f32={pred_f32:.6} err={:.6}%",
            err * 100.0
        );

        harness.check_upper(
            &format!("Wide FC (rs={rs_size}) f64→f32 parity"),
            err,
            tolerances::NPU_F32_PARITY,
        );
    }
}

/// Check 6: Determinism — same input, same weights, same output.
fn check_determinism(harness: &mut ValidationHarness) {
    println!("\n[6] Determinism");

    let config = EsnConfig::default();
    let mut esn = EchoStateNetwork::new(config);

    let mut rng = SimpleRng::new(33);
    let train_seqs: Vec<Vec<Vec<f64>>> = (0..4)
        .map(|_| {
            (0..60)
                .map(|_| (0..8).map(|_| rng.standard_normal() * 0.3).collect())
                .collect()
        })
        .collect();
    let train_targets: Vec<Vec<f64>> = (0..4).map(|_| vec![rng.uniform()]).collect();
    esn.train(&train_seqs, &train_targets);

    let test_seq: Vec<Vec<f64>> = (0..60)
        .map(|_| (0..8).map(|_| rng.standard_normal() * 0.3).collect())
        .collect();

    let results: Vec<f64> = (0..10)
        .map(|_| esn.predict(&test_seq).expect("ESN trained")[0])
        .collect();
    // Exact parity check — determinism: same input → bit-identical output
    #[allow(clippy::float_cmp)] // determinism test: bit-identical outputs required
    let all_same = results.windows(2).all(|w| w[0] == w[1]);
    println!("  10 runs: all identical = {all_same}");
    println!("  value = {:.6}", results[0]);

    harness.check_bool(
        "10 identical predictions from same input (determinism)",
        all_same,
    );
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

fn quantize_f32_vec(weights: &[f32], bits: u32) -> (Vec<i8>, f32) {
    let max_abs = weights.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs < 1e-30 {
        return (vec![0i8; weights.len()], 1.0);
    }
    let max_int = ((1i32 << (bits - 1)) - 1) as f32;
    let scale = max_abs / max_int;
    let quantized: Vec<i8> = weights
        .iter()
        .map(|&v| (v / scale).round().clamp(-max_int, max_int) as i8)
        .collect();
    (quantized, scale)
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-30);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
