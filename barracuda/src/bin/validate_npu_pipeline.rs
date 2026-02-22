// SPDX-License-Identifier: AGPL-3.0-only

//! NPU Physics Pipeline Validation
//!
//! Validates the end-to-end math of the GPU→NPU transport prediction pipeline:
//!   1. Generate MD-like velocity features (simulates GPU output)
//!   2. Train ESN on CPU (f64 gold standard)
//!   3. Predict via NpuSimulator (f32 — simulates NPU deployment)
//!   4. Predict via int4-quantized readout (simulates hardware)
//!   5. Multi-output readout (D*, η*, λ* simultaneously)
//!   6. Verify pipeline composition (feature→reservoir→readout)
//!
//! The Python control (`npu_physics_pipeline.py`) validates on actual AKD1000.
//! This binary proves the math is substrate-independent.
//!
//! Key thesis: the same trained weights produce valid predictions across
//! f64 (CPU), f32 (NpuSimulator), int4 (quantized), and hardware (AKD1000).

use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("══════════════════════════════════════════════════════");
    println!("  NPU Physics Pipeline — metalForge Rust Validation");
    println!("══════════════════════════════════════════════════════");
    println!();

    let mut harness = ValidationHarness::new("npu_pipeline");

    check_pipeline_single_output(&mut harness);
    check_pipeline_multi_output(&mut harness);
    check_quantized_pipeline(&mut harness);
    check_continuous_prediction(&mut harness);

    println!();
    harness.finish();
}

/// Stage 1+2: Full pipeline — generate features, train ESN, predict D*
fn check_pipeline_single_output(harness: &mut ValidationHarness) {
    println!("[1] Single-output transport pipeline (D*)");

    let config = EsnConfig::default();
    let mut esn = EchoStateNetwork::new(config.clone());

    let mut rng = SimpleRng::new(42);
    let n_cases = 6;
    let n_train = 4;
    let n_frames = 100;

    let sequences: Vec<Vec<Vec<f64>>> = (0..n_cases)
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

    let targets: Vec<Vec<f64>> = (0..n_cases)
        .map(|_| vec![rng.uniform().mul_add(0.05, 0.001)])
        .collect();

    let train_seqs: Vec<Vec<Vec<f64>>> = sequences[..n_train].to_vec();
    let train_targets: Vec<Vec<f64>> = targets[..n_train].to_vec();

    esn.train(&train_seqs, &train_targets);

    let test_preds: Vec<f64> = sequences[n_train..]
        .iter()
        .map(|s| esn.predict(s).expect("ESN trained")[0])
        .collect();

    let all_finite = test_preds.iter().all(|v| v.is_finite());
    let all_positive = test_preds.iter().all(|v| *v > -1.0);

    println!(
        "  Test predictions: {:?}",
        test_preds
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>()
    );

    harness.check_bool("Pipeline produces finite predictions", all_finite);
    harness.check_bool(
        "Predictions are physically reasonable (> -1.0)",
        all_positive,
    );

    // NpuSimulator path
    let exported = esn.export_weights().expect("trained");
    let mut npu = NpuSimulator::from_exported(&exported);

    let npu_preds: Vec<f64> = sequences[n_train..]
        .iter()
        .map(|s| npu.predict(s)[0])
        .collect();

    let max_err = test_preds
        .iter()
        .zip(npu_preds.iter())
        .map(|(&f64_v, &f32_v)| (f64_v - f32_v).abs() / f64_v.abs().max(1e-10))
        .fold(0.0_f64, f64::max);

    println!("  f64→f32 max error: {:.6}%", max_err * 100.0);
    harness.check_upper(
        "Pipeline f64→f32 parity",
        max_err,
        tolerances::NPU_F32_PARITY,
    );
}

/// Stage 3: Multi-output readout (D*, η*, λ* simultaneously)
fn check_pipeline_multi_output(harness: &mut ValidationHarness) {
    println!("\n[2] Multi-output transport pipeline (D*, η*, λ*)");

    let config = EsnConfig {
        output_size: 3,
        ..EsnConfig::default()
    };
    let mut esn = EchoStateNetwork::new(config.clone());

    let mut rng = SimpleRng::new(77);
    let n_train = 6;
    let n_frames = 80;

    let sequences: Vec<Vec<Vec<f64>>> = (0..8)
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

    // Simulate D*, η*, λ* as correlated transport coefficients
    let targets: Vec<Vec<f64>> = (0..8)
        .map(|_| {
            let d = rng.uniform().mul_add(0.05, 0.001);
            vec![d, d * 0.8, d * 1.2]
        })
        .collect();

    esn.train(&sequences[..n_train], &targets[..n_train]);

    let multi_preds: Vec<Vec<f64>> = sequences[n_train..]
        .iter()
        .map(|s| esn.predict(s).expect("ESN trained"))
        .collect();

    harness.check_bool(
        "Multi-output pipeline produces 3 values per prediction",
        multi_preds.iter().all(|p| p.len() == 3),
    );

    // Exact parity check — independent readout weights produce distinct values
    #[allow(clippy::float_cmp)] // determinism test: bit-identical outputs required
    let all_distinct = multi_preds
        .iter()
        .all(|p| p[0] != p[1] && p[1] != p[2] && p[0] != p[2]);
    harness.check_bool(
        "D*, η*, λ* are distinct (independent readouts)",
        all_distinct,
    );

    // NpuSimulator parity for multi-output
    let exported = esn.export_weights().expect("trained");
    let mut npu = NpuSimulator::from_exported(&exported);

    let npu_multi: Vec<Vec<f64>> = sequences[n_train..]
        .iter()
        .map(|s| npu.predict(s))
        .collect();

    let max_multi_err = multi_preds
        .iter()
        .zip(npu_multi.iter())
        .flat_map(|(f64_v, f32_v)| {
            f64_v
                .iter()
                .zip(f32_v.iter())
                .map(|(&a, &b)| (a - b).abs() / a.abs().max(1e-10))
        })
        .fold(0.0_f64, f64::max);

    println!(
        "  Multi-output f64→f32 max error: {:.6}%",
        max_multi_err * 100.0
    );
    harness.check_upper(
        "Multi-output f64→f32 parity",
        max_multi_err,
        tolerances::NPU_F32_PARITY,
    );
}

/// Stage 4: Quantized pipeline (int4 readout)
fn check_quantized_pipeline(harness: &mut ValidationHarness) {
    println!("\n[3] Quantized pipeline (int4 readout — simulates NPU hardware)");

    let config = EsnConfig::default();
    let mut esn = EchoStateNetwork::new(config.clone());

    let mut rng = SimpleRng::new(99);
    let n_frames = 100;

    let sequences: Vec<Vec<Vec<f64>>> = (0..6)
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

    let targets: Vec<Vec<f64>> = (0..6)
        .map(|_| vec![rng.uniform().mul_add(0.05, 0.001)])
        .collect();

    esn.train(&sequences[..4], &targets[..4]);
    let exported = esn.export_weights().expect("trained");

    let f64_preds: Vec<f64> = sequences[4..]
        .iter()
        .map(|s| esn.predict(s).expect("ESN trained")[0])
        .collect();

    // Simulate int4 quantized readout (like AKD1000 would do)
    let (w_out_q4, s_out) = quantize_f32_vec(&exported.w_out, 4);
    let mut npu = NpuSimulator::from_exported(&exported);

    let q4_preds: Vec<f64> = sequences[4..]
        .iter()
        .map(|s| {
            let state = npu.predict_return_state(s);
            let rs = exported.reservoir_size;
            let mut val = 0.0_f64;
            for j in 0..rs {
                val += f64::from(w_out_q4[j]) * f64::from(s_out) * f64::from(state[j]);
            }
            val
        })
        .collect();

    let max_q4_err = f64_preds
        .iter()
        .zip(q4_preds.iter())
        .map(|(&a, &b)| (a - b).abs() / a.abs().max(1e-10))
        .fold(0.0_f64, f64::max);

    println!("  int4 quantized max error: {:.4}%", max_q4_err * 100.0);

    harness.check_upper(
        "Pipeline int4 readout error bounded",
        max_q4_err,
        tolerances::NPU_INT4_QUANTIZATION,
    );

    // Verify error ordering: int4 > f32 (quantization degrades)
    let mut npu2 = NpuSimulator::from_exported(&exported);
    let f32_preds: Vec<f64> = sequences[4..].iter().map(|s| npu2.predict(s)[0]).collect();

    let max_f32_err = f64_preds
        .iter()
        .zip(f32_preds.iter())
        .map(|(&a, &b)| (a - b).abs() / a.abs().max(1e-10))
        .fold(0.0_f64, f64::max);

    harness.check_bool(
        "Error ordering: int4 > f32 (quantization degrades monotonically)",
        max_q4_err > max_f32_err,
    );
}

/// Stage 5: Continuous prediction determinism (simulates streaming)
fn check_continuous_prediction(harness: &mut ValidationHarness) {
    println!("\n[4] Continuous prediction (simulates GPU→NPU streaming)");

    let config = EsnConfig::default();
    let mut esn = EchoStateNetwork::new(config.clone());

    let mut rng = SimpleRng::new(55);
    let n_frames = 60;

    let sequences: Vec<Vec<Vec<f64>>> = (0..4)
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
    let targets: Vec<Vec<f64>> = (0..4).map(|_| vec![rng.uniform() * 0.05]).collect();

    esn.train(&sequences, &targets);

    // Simulate 20 "streaming" predictions on the same data
    let test: Vec<Vec<f64>> = (0..n_frames)
        .map(|_| {
            (0..config.input_size)
                .map(|_| rng.standard_normal() * 0.3)
                .collect()
        })
        .collect();

    let preds: Vec<f64> = (0..20)
        .map(|_| esn.predict(&test).expect("ESN trained")[0])
        .collect();
    // Exact parity check — determinism: same input → bit-identical output
    #[allow(clippy::float_cmp)] // determinism test: bit-identical outputs required
    let all_same = preds.windows(2).all(|w| w[0] == w[1]);

    println!("  20 streaming predictions: all identical = {all_same}");
    harness.check_bool(
        "Streaming predictions are deterministic (20 consecutive)",
        all_same,
    );

    // Verify the NpuSimulator also gives deterministic streaming
    let exported = esn.export_weights().expect("trained");
    let mut npu = NpuSimulator::from_exported(&exported);
    let npu_preds: Vec<f64> = (0..20).map(|_| npu.predict(&test)[0]).collect();
    // Exact parity check — NpuSimulator determinism
    #[allow(clippy::float_cmp)] // determinism test: bit-identical outputs required
    let npu_all_same = npu_preds.windows(2).all(|w| w[0] == w[1]);

    harness.check_bool(
        "NpuSimulator streaming is deterministic (20 consecutive)",
        npu_all_same,
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
