// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 5: GPU as ESN reservoir (large RS + multi-output).

use hotspring_barracuda::bench::{GpuEsn, generate_test_sequence, generate_training_data, time_fn};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig};
use hotspring_barracuda::validation::ValidationHarness;

use super::INPUT_SIZE;

pub fn run(gpu: &GpuF64, harness: &mut ValidationHarness, jsonl_records: &mut Vec<String>) {
    println!();
    println!("═══ Experiment 5: GPU as ESN — Reservoir on Graphics Silicon ═══");

    let large_sizes = [256, 512, 1024];
    let long_seq = 200;
    for &rs in &large_sizes {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: rs,
            output_size: 6,
            spectral_radius: 0.95,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let (train_seqs, train_targets_raw) = generate_training_data(8, long_seq, INPUT_SIZE);
        let train_targets: Vec<Vec<f64>> = train_targets_raw
            .iter()
            .map(|t| (0..6).map(|j| f64::from(j).mul_add(0.1, t[0])).collect())
            .collect();

        let mut esn = EchoStateNetwork::new(config);
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("export");
        let test_seq = generate_test_sequence(42, long_seq, INPUT_SIZE);

        let cpu_pred = esn.predict(&test_seq).unwrap_or_default();
        let test_seq_c = test_seq.clone();
        let (cpu_us, _, _) = time_fn(|| esn.predict(&test_seq_c).unwrap_or_default(), 3, 10);

        let gpu_esn = GpuEsn::new(gpu, &exported);
        let test_seq_g = test_seq.clone();
        let (gpu_us, _, gpu_pred) = time_fn(|| gpu_esn.predict_batched(gpu, &test_seq_g), 3, 10);

        let max_diff: f64 = gpu_pred
            .iter()
            .zip(cpu_pred.iter())
            .map(|(g, c)| (g - c).abs())
            .fold(0.0, f64::max);

        let speedup = cpu_us / gpu_us.max(0.1);
        println!("    RS={rs}, OS=6, SeqLen={long_seq}:");
        println!("      CPU-f64: {cpu_us:.0} μs");
        println!("      GPU-f32: {gpu_us:.0} μs  ({speedup:.2}× vs CPU)");
        println!("      max|Δ|:  {max_diff:.4e}");

        harness.check_upper(&format!("gpu_esn_accuracy_rs{rs}"), max_diff, 1.0);

        jsonl_records.push(
            serde_json::json!({
                "experiment": "gpu_as_esn",
                "reservoir_size": rs, "output_size": 6, "seq_length": long_seq,
                "cpu_f64_us": cpu_us, "gpu_batch_us": gpu_us,
                "speedup": speedup, "max_diff": max_diff,
            })
            .to_string(),
        );
    }
}
