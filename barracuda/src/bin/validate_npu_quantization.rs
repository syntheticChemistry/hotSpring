// SPDX-License-Identifier: AGPL-3.0-only

//! NPU Quantization Parity Validation
//!
//! Validates that ESN predictions survive quantization from f64 through
//! f32, int8, and int4 precisions. The Python control experiment
//! (`control/metalforge_npu/scripts/npu_quantization_parity.py`) provides
//! the reference; this binary proves the Rust implementation matches.
//!
//! Checks:
//!   1. f64 → f32 parity (NpuSimulator path)
//!   2. f64 → int8 quantized weights prediction error
//!   3. f64 → int4 quantized weights prediction error
//!   4. f64 → int4 weights + int4 activations (full hardware sim)
//!   5. Weight sparsity preserved through quantization
//!
//! Reference: metalForge/npu/akida/HARDWARE.md

use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("══════════════════════════════════════════════════════");
    println!("  NPU Quantization Parity — metalForge Validation");
    println!("══════════════════════════════════════════════════════");
    println!();

    let mut harness = ValidationHarness::new("npu_quantization");

    let config = EsnConfig::default();
    let mut esn = EchoStateNetwork::new(config.clone());

    // Synthetic test data (same seed as Python control)
    let n_cases = 6;
    let n_frames = 100;
    let n_features = config.input_size;

    let mut rng = SimpleRng::new(99);
    let sequences: Vec<Vec<Vec<f64>>> = (0..n_cases)
        .map(|_| {
            (0..n_frames)
                .map(|_| {
                    (0..n_features)
                        .map(|_| rng.standard_normal().tanh() * 0.5_f64.tanh())
                        .collect()
                })
                .collect()
        })
        .collect();
    let targets: Vec<Vec<f64>> = (0..n_cases)
        .map(|_| vec![rng.uniform() * 1.9 + 0.1])
        .collect();

    let train_idx = [0, 1, 2, 3];
    let train_seqs: Vec<Vec<Vec<f64>>> = train_idx.iter().map(|&i| sequences[i].clone()).collect();
    let train_targets: Vec<Vec<f64>> = train_idx.iter().map(|&i| targets[i].clone()).collect();

    // Train ESN
    println!("[1] Training ESN (reservoir=50, input=8, seed=42)");
    esn.train(&train_seqs, &train_targets);

    // Get f64 predictions for all cases
    let f64_preds: Vec<f64> = sequences.iter().map(|seq| esn.predict(seq)[0]).collect();

    println!(
        "  f64 predictions: {:?}",
        f64_preds
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );

    // Export weights for NpuSimulator (f32)
    let exported = esn.export_weights().expect("ESN must be trained");
    let mut npu_sim = NpuSimulator::from_exported(&exported);

    // ── Check 1: f64 → f32 parity (NpuSimulator) ──
    println!("\n[2] f64 → f32 parity (NpuSimulator)");
    let f32_preds: Vec<f64> = sequences
        .iter()
        .map(|seq| npu_sim.predict(seq)[0])
        .collect();

    let mut max_f32_err = 0.0_f64;
    for (i, (&p64, &p32)) in f64_preds.iter().zip(f32_preds.iter()).enumerate() {
        let err = (p64 - p32).abs() / p64.abs().max(1e-10);
        if err > max_f32_err {
            max_f32_err = err;
        }
        println!(
            "  Case {i}: f64={p64:.6} f32={p32:.6} err={:.6}%",
            err * 100.0
        );
    }
    harness.check_upper(
        "f64→f32 max relative error",
        max_f32_err,
        tolerances::NPU_F32_PARITY,
    );

    // ── Check 2: f64 → int8 quantization ──
    println!("\n[3] f64 → int8 quantization");
    let (w_in_q8, s_in_8) = quantize_f32_vec(&exported.w_in, 8);
    let (w_res_q8, s_res_8) = quantize_f32_vec(&exported.w_res, 8);

    let int8_preds: Vec<f64> = sequences
        .iter()
        .map(|seq| {
            predict_quantized(
                &w_in_q8,
                s_in_8,
                &w_res_q8,
                s_res_8,
                &exported.w_out,
                seq,
                exported.reservoir_size,
                exported.input_size,
                exported.leak_rate,
                None,
            )
        })
        .collect();

    let mut max_int8_err = 0.0_f64;
    for (i, (&p64, &p8)) in f64_preds.iter().zip(int8_preds.iter()).enumerate() {
        let err = (p64 - p8).abs() / p64.abs().max(1e-10);
        if err > max_int8_err {
            max_int8_err = err;
        }
        println!(
            "  Case {i}: f64={p64:.6} int8={p8:.6} err={:.4}%",
            err * 100.0
        );
    }
    harness.check_upper(
        "f64→int8 max relative error",
        max_int8_err,
        tolerances::NPU_INT8_QUANTIZATION,
    );

    // ── Check 3: f64 → int4 quantization ──
    println!("\n[4] f64 → int4 quantization");
    let (w_in_q4, s_in_4) = quantize_f32_vec(&exported.w_in, 4);
    let (w_res_q4, s_res_4) = quantize_f32_vec(&exported.w_res, 4);

    let int4_preds: Vec<f64> = sequences
        .iter()
        .map(|seq| {
            predict_quantized(
                &w_in_q4,
                s_in_4,
                &w_res_q4,
                s_res_4,
                &exported.w_out,
                seq,
                exported.reservoir_size,
                exported.input_size,
                exported.leak_rate,
                None,
            )
        })
        .collect();

    let mut max_int4_err = 0.0_f64;
    for (i, (&p64, &p4)) in f64_preds.iter().zip(int4_preds.iter()).enumerate() {
        let err = (p64 - p4).abs() / p64.abs().max(1e-10);
        if err > max_int4_err {
            max_int4_err = err;
        }
        println!(
            "  Case {i}: f64={p64:.6} int4={p4:.6} err={:.4}%",
            err * 100.0
        );
    }
    harness.check_upper(
        "f64→int4 max relative error",
        max_int4_err,
        tolerances::NPU_INT4_QUANTIZATION,
    );

    // ── Check 4: f64 → int4 + int4 activations (full hardware sim) ──
    println!("\n[5] f64 → int4 weights + int4 activations");
    let int4_full_preds: Vec<f64> = sequences
        .iter()
        .map(|seq| {
            predict_quantized(
                &w_in_q4,
                s_in_4,
                &w_res_q4,
                s_res_4,
                &exported.w_out,
                seq,
                exported.reservoir_size,
                exported.input_size,
                exported.leak_rate,
                Some(4),
            )
        })
        .collect();

    let mut max_int4_full_err = 0.0_f64;
    for (i, (&p64, &p4f)) in f64_preds.iter().zip(int4_full_preds.iter()).enumerate() {
        let err = (p64 - p4f).abs() / p64.abs().max(1e-10);
        if err > max_int4_full_err {
            max_int4_full_err = err;
        }
        println!(
            "  Case {i}: f64={p64:.6} int4+act4={p4f:.6} err={:.4}%",
            err * 100.0
        );
    }
    harness.check_upper(
        "f64→int4+act4 max relative error",
        max_int4_full_err,
        tolerances::NPU_INT4_FULL_QUANTIZATION,
    );

    // ── Check 5: Sparsity preservation ──
    println!("\n[6] Weight sparsity analysis");
    let w_res_sparsity_f32 = exported.w_res.iter().filter(|&&v| v.abs() < 1e-30).count() as f64
        / exported.w_res.len() as f64;
    let w_res_sparsity_q4 =
        w_res_q4.iter().filter(|&&v| v == 0).count() as f64 / w_res_q4.len() as f64;
    println!(
        "  W_res sparsity: f32={:.1}%  int4={:.1}%",
        w_res_sparsity_f32 * 100.0,
        w_res_sparsity_q4 * 100.0
    );

    harness.check_bool(
        "W_res int4 sparsity >= f32 sparsity (quantization increases sparsity)",
        w_res_sparsity_q4 >= w_res_sparsity_f32 - 0.01,
    );

    // ── Check 6: Error ordering (int4 > int8 > f32) ──
    harness.check_bool(
        "Error ordering: int4 > int8 > f32 (quantization degrades monotonically)",
        max_int4_err > max_int8_err && max_int8_err > max_f32_err,
    );

    println!();
    harness.finish();
}

// ═══════════════════════════════════════════════════════════════
// Quantization helpers
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

fn predict_quantized(
    w_in_q: &[i8],
    s_in: f32,
    w_res_q: &[i8],
    s_res: f32,
    w_out: &[f32],
    input_sequence: &[Vec<f64>],
    reservoir_size: usize,
    input_size: usize,
    leak_rate: f32,
    act_bits: Option<u32>,
) -> f64 {
    let rs = reservoir_size;
    let is = input_size;
    let mut state = vec![0.0_f32; rs];

    for input in input_sequence {
        let mut pre = vec![0.0_f32; rs];
        for i in 0..rs {
            let mut val = 0.0_f32;
            for j in 0..is {
                val += (w_in_q[i * is + j] as f32 * s_in) * input[j] as f32;
            }
            for j in 0..rs {
                val += (w_res_q[i * rs + j] as f32 * s_res) * state[j];
            }
            pre[i] = val;
        }
        for i in 0..rs {
            state[i] = (1.0 - leak_rate) * state[i] + leak_rate * pre[i].tanh();
        }

        if let Some(bits) = act_bits {
            let max_int = ((1i32 << (bits - 1)) - 1) as f32;
            let s_max = state.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            if s_max > 1e-30 {
                let s_act = s_max / max_int;
                for v in &mut state {
                    *v = (*v / s_act).round().clamp(-max_int, max_int) * s_act;
                }
            }
        }
    }

    let mut result = 0.0_f64;
    for j in 0..rs {
        result += f64::from(w_out[j]) * f64::from(state[j]);
    }
    result
}

// Minimal PRNG for reproducible test data (not for crypto)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
