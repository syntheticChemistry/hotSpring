// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 3: NPU capability envelope (threshold, streaming, multi-output, mutation).

use hotspring_barracuda::bench::{GpuEsn, generate_test_sequence, generate_training_data, time_fn};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

use super::{INPUT_SIZE, N_REPS, N_WARMUP, SEQUENCE_LENGTH};

pub fn run(gpu: &GpuF64, harness: &mut ValidationHarness, jsonl_records: &mut Vec<String>) {
    println!("═══ Experiment 3: NPU Capability Envelope ═══");
    println!();

    // 3a. Threshold detection
    println!("  3a. Threshold Detection (binary classifier)");
    {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let mut esn = EchoStateNetwork::new(config);
        let n_samples = 20;
        let mut seqs = Vec::with_capacity(n_samples);
        let mut targets = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let is_above = i >= n_samples / 2;
            let base = if is_above { 0.5 } else { -0.5 };
            let seq: Vec<Vec<f64>> = (0..20)
                .map(|t| {
                    let x = f64::from(t).mul_add(0.01, base);
                    vec![x, x * 0.5, x.sin(), x.cos(), x * x, x.abs(), 0.0, 0.0]
                })
                .collect();
            seqs.push(seq);
            targets.push(vec![if is_above { 1.0 } else { 0.0 }]);
        }
        esn.train(&seqs, &targets);
        let exported = esn.export_weights().expect("export");
        let mut npu = NpuSimulator::from_exported(&exported);

        let mut correct = 0;
        for i in 0..n_samples {
            let pred = npu.predict(&seqs[i])[0];
            let expected = targets[i][0];
            let classified = if pred > 0.5 { 1.0 } else { 0.0 };
            if (classified - expected).abs() < 0.01 {
                correct += 1;
            }
        }
        let accuracy = f64::from(correct) / n_samples as f64;
        println!(
            "    Accuracy: {correct}/{n_samples} = {:.1}%",
            accuracy * 100.0
        );
        harness.check_bool("npu_threshold_detection", accuracy >= 0.7);

        jsonl_records.push(
            serde_json::json!({
                "experiment": "npu_threshold_detection",
                "n_samples": n_samples, "correct": correct, "accuracy": accuracy,
            })
            .to_string(),
        );
    }

    // 3b. Streaming inference
    println!("  3b. Streaming Inference Throughput");
    {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let (train_seqs, train_targets) = generate_training_data(4, SEQUENCE_LENGTH, INPUT_SIZE);
        let mut esn = EchoStateNetwork::new(config.clone());
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("export");
        let mut npu = NpuSimulator::from_exported(&exported);

        let stream_len = 1000;
        let stream_data = generate_test_sequence(123, stream_len, INPUT_SIZE);

        // NpuSimulator streaming
        let t0 = Instant::now();
        let mut npu_predictions = Vec::with_capacity(stream_len);
        for chunk in stream_data.chunks(1) {
            npu_predictions.push(npu.predict(chunk)[0]);
        }
        let npu_stream_us = t0.elapsed().as_secs_f64() * 1e6;
        let npu_per_step = npu_stream_us / stream_len as f64;

        // CPU f64 streaming
        let mut esn2 = EchoStateNetwork::new(config);
        esn2.train(&train_seqs, &train_targets);
        let t0 = Instant::now();
        for chunk in stream_data.chunks(1) {
            std::hint::black_box(esn2.predict(chunk).unwrap_or_default()[0]);
        }
        let cpu_stream_us = t0.elapsed().as_secs_f64() * 1e6;
        let cpu_per_step = cpu_stream_us / stream_len as f64;

        // GPU streaming
        let gpu_esn = GpuEsn::new(gpu, &exported);
        let t0 = Instant::now();
        let mut gpu_predictions = Vec::with_capacity(stream_len);
        for chunk in stream_data.chunks(1) {
            gpu_predictions.push(gpu_esn.predict(gpu, chunk)[0]);
        }
        let gpu_stream_us = t0.elapsed().as_secs_f64() * 1e6;
        let gpu_per_step = gpu_stream_us / stream_len as f64;

        println!("    {stream_len} steps streaming (RS=50):");
        println!("      CPU-f64:  {cpu_per_step:.1} μs/step  ({cpu_stream_us:.0} μs total)");
        println!("      NPU-sim:  {npu_per_step:.1} μs/step  ({npu_stream_us:.0} μs total)");
        println!("      GPU-f32:  {gpu_per_step:.1} μs/step  ({gpu_stream_us:.0} μs total)");

        harness.check_bool(
            "streaming_npu_functional",
            npu_predictions.len() == stream_len,
        );
        harness.check_bool(
            "streaming_gpu_functional",
            gpu_predictions.len() == stream_len,
        );

        jsonl_records.push(
            serde_json::json!({
                "experiment": "streaming_throughput",
                "stream_length": stream_len, "reservoir_size": 50,
                "cpu_f64_per_step_us": cpu_per_step,
                "npu_sim_per_step_us": npu_per_step,
                "gpu_f32_per_step_us": gpu_per_step,
            })
            .to_string(),
        );
    }

    // 3c. Multi-output prediction
    println!("  3c. Multi-Output ESN (NPU multi-head)");
    {
        let output_counts = [1, 2, 4, 6, 8];
        for &os in &output_counts {
            let config = EsnConfig {
                input_size: INPUT_SIZE,
                reservoir_size: 50,
                output_size: os,
                spectral_radius: 0.95,
                connectivity: 0.2,
                leak_rate: 0.3,
                regularization: 1e-4,
                seed: 42,
                ..Default::default()
            };
            let (train_seqs, train_targets_raw) =
                generate_training_data(8, SEQUENCE_LENGTH, INPUT_SIZE);
            let train_targets: Vec<Vec<f64>> = train_targets_raw
                .iter()
                .map(|t| (0..os).map(|j| (j as f64).mul_add(0.1, t[0])).collect())
                .collect();

            let mut esn = EchoStateNetwork::new(config);
            esn.train(&train_seqs, &train_targets);
            let exported = esn.export_weights().expect("export");
            let mut npu = NpuSimulator::from_exported(&exported);

            let test_seq = generate_test_sequence(555, SEQUENCE_LENGTH, INPUT_SIZE);
            let (npu_us, _, npu_pred) = time_fn(|| npu.predict(&test_seq), N_WARMUP, N_REPS);
            let cpu_pred = esn.predict(&test_seq).unwrap_or_default();

            let max_diff: f64 = npu_pred
                .iter()
                .zip(cpu_pred.iter())
                .map(|(n, c)| (n - c).abs())
                .fold(0.0, f64::max);

            println!("    OS={os}: NPU {npu_us:.1} μs, max|Δ|={max_diff:.4e}");
            harness.check_bool(&format!("multi_output_os{os}"), max_diff < 0.1);

            jsonl_records.push(
                serde_json::json!({
                    "experiment": "multi_output",
                    "output_size": os, "npu_us": npu_us, "max_diff": max_diff,
                })
                .to_string(),
            );
        }
    }

    // 3d. Weight mutation
    println!("  3d. Weight Mutation Under Load");
    {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let (train_seqs, train_targets) = generate_training_data(8, SEQUENCE_LENGTH, INPUT_SIZE);
        let mut esn = EchoStateNetwork::new(config);
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("export");
        let test_seq = generate_test_sequence(888, SEQUENCE_LENGTH, INPUT_SIZE);
        let n_mutations = 50;

        let t0 = Instant::now();
        let mut drift_values = Vec::with_capacity(n_mutations);
        let mut npu = NpuSimulator::from_exported(&exported);
        let baseline_pred = npu.predict(&test_seq)[0];

        for i in 0..n_mutations {
            let mut mutated = exported.clone();
            let scale = (i as f32).mul_add(0.001, 1.0);
            for w in &mut mutated.w_out {
                *w *= scale;
            }
            npu = NpuSimulator::from_exported(&mutated);
            let pred = npu.predict(&test_seq)[0];
            drift_values.push((pred - baseline_pred).abs());
        }
        let mutation_total_us = t0.elapsed().as_secs_f64() * 1e6;
        let per_mutation_us = mutation_total_us / n_mutations as f64;
        let max_drift = drift_values.iter().copied().fold(0.0f64, f64::max);
        let mean_drift = drift_values.iter().sum::<f64>() / drift_values.len() as f64;

        println!("    {n_mutations} mutations: {per_mutation_us:.1} μs/mutation");
        println!("    Drift: mean={mean_drift:.4e}, max={max_drift:.4e}");
        harness.check_upper("weight_mutation_latency", per_mutation_us, 100_000.0);

        jsonl_records.push(
            serde_json::json!({
                "experiment": "weight_mutation",
                "n_mutations": n_mutations,
                "per_mutation_us": per_mutation_us,
                "mean_drift": mean_drift, "max_drift": max_drift,
            })
            .to_string(),
        );
    }
}
