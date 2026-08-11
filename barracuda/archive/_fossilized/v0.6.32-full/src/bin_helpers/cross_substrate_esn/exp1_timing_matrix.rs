// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 1: Cross-substrate timing matrix at reservoir sizes 16..500.

use hotspring_barracuda::bench::{
    GpuEsn, SubstrateResult, generate_test_sequence, generate_training_data, time_fn,
};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;

use super::{INPUT_SIZE, N_REPS, N_WARMUP, SEQUENCE_LENGTH};

pub fn run(
    gpu: &GpuF64,
    harness: &mut ValidationHarness,
    all_results: &mut Vec<SubstrateResult>,
    jsonl_records: &mut Vec<String>,
    reservoir_sizes: &[usize],
) {
    println!("═══ Experiment 1: Cross-Substrate Timing Matrix ═══");
    println!("  Reservoir sizes: {reservoir_sizes:?}");
    println!("  Sequence length: {SEQUENCE_LENGTH}, Input features: {INPUT_SIZE}");
    println!("  Warmup: {N_WARMUP}, Reps: {N_REPS}");
    println!();

    let output_size = 1;

    for &rs in reservoir_sizes {
        println!("  ── Reservoir size: {rs} ──");

        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: rs,
            output_size,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };

        let (train_seqs, train_targets) = generate_training_data(8, SEQUENCE_LENGTH, INPUT_SIZE);
        let test_seq = generate_test_sequence(999, SEQUENCE_LENGTH, INPUT_SIZE);

        let mut esn = EchoStateNetwork::new(config.clone());
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("trained ESN should export");

        // CPU f64
        let mut esn_bench = EchoStateNetwork::new(config.clone());
        esn_bench.train(&train_seqs, &train_targets);
        let test_seq_cpu = test_seq.clone();
        let (cpu64_mean, cpu64_std, cpu64_pred) = time_fn(
            || esn_bench.predict(&test_seq_cpu).unwrap_or_default(),
            N_WARMUP,
            N_REPS,
        );
        println!(
            "    CPU-f64:  {:>10.1} ± {:>8.1} μs  pred={:.6}",
            cpu64_mean, cpu64_std, cpu64_pred[0]
        );
        all_results.push(SubstrateResult {
            substrate: "CPU-f64".into(),
            reservoir_size: rs,
            mean_us: cpu64_mean,
            std_us: cpu64_std,
            prediction: cpu64_pred.clone(),
        });

        // CPU f32 (NpuSimulator)
        let mut npu_sim = NpuSimulator::from_exported(&exported);
        let test_seq_npu = test_seq.clone();
        let (npu_mean, npu_std, npu_pred) =
            time_fn(|| npu_sim.predict(&test_seq_npu), N_WARMUP, N_REPS);
        println!(
            "    CPU-f32:  {:>10.1} ± {:>8.1} μs  pred={:.6}",
            npu_mean, npu_std, npu_pred[0]
        );
        all_results.push(SubstrateResult {
            substrate: "CPU-f32".into(),
            reservoir_size: rs,
            mean_us: npu_mean,
            std_us: npu_std,
            prediction: npu_pred.clone(),
        });

        // GPU f32 (per-step dispatch)
        let gpu_esn = GpuEsn::new(gpu, &exported);
        let test_seq_gpu = test_seq.clone();
        let (gpu_mean, gpu_std, gpu_pred) =
            time_fn(|| gpu_esn.predict(gpu, &test_seq_gpu), N_WARMUP, N_REPS);
        println!(
            "    GPU-f32:  {:>10.1} ± {:>8.1} μs  pred={:.6}",
            gpu_mean, gpu_std, gpu_pred[0]
        );
        all_results.push(SubstrateResult {
            substrate: "GPU-f32-step".into(),
            reservoir_size: rs,
            mean_us: gpu_mean,
            std_us: gpu_std,
            prediction: gpu_pred.clone(),
        });

        // GPU f32 (batched encoder)
        let test_seq_gpu_b = test_seq.clone();
        let (gpu_batch_mean, gpu_batch_std, gpu_batch_pred) = time_fn(
            || gpu_esn.predict_batched(gpu, &test_seq_gpu_b),
            N_WARMUP,
            N_REPS,
        );
        println!(
            "    GPU-bat:  {:>10.1} ± {:>8.1} μs  pred={:.6}",
            gpu_batch_mean, gpu_batch_std, gpu_batch_pred[0]
        );
        all_results.push(SubstrateResult {
            substrate: "GPU-f32-batch".into(),
            reservoir_size: rs,
            mean_us: gpu_batch_mean,
            std_us: gpu_batch_std,
            prediction: gpu_batch_pred.clone(),
        });

        // Accuracy parity
        let cpu_ref = cpu64_pred[0];
        for (label, pred) in [
            ("CPU-f32", &npu_pred),
            ("GPU-step", &gpu_pred),
            ("GPU-batch", &gpu_batch_pred),
        ] {
            let diff = (pred[0] - cpu_ref).abs();
            let rel = diff / cpu_ref.abs().max(1e-10);
            let ok = rel < 0.10;
            println!(
                "      {label} parity: |Δ|={diff:.6e}, rel={rel:.4e} {}",
                if ok { "OK" } else { "WARN" }
            );
            harness.check_bool(&format!("parity_{label}_rs{rs}"), ok);
        }

        let record = serde_json::json!({
            "experiment": "cross_substrate_timing",
            "reservoir_size": rs,
            "sequence_length": SEQUENCE_LENGTH,
            "cpu_f64_us": cpu64_mean,
            "cpu_f64_std": cpu64_std,
            "cpu_f32_us": npu_mean,
            "cpu_f32_std": npu_std,
            "gpu_step_us": gpu_mean,
            "gpu_step_std": gpu_std,
            "gpu_batch_us": gpu_batch_mean,
            "gpu_batch_std": gpu_batch_std,
            "cpu_f64_pred": cpu64_pred[0],
            "cpu_f32_pred": npu_pred[0],
            "gpu_step_pred": gpu_pred[0],
            "gpu_batch_pred": gpu_batch_pred[0],
        });
        jsonl_records.push(record.to_string());
        println!();
    }
}
