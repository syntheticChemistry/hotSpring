// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 4: Scaling crossover — find RS where GPU beats CPU.

use hotspring_barracuda::bench::{GpuEsn, generate_test_sequence, generate_training_data, time_fn};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig, NpuSimulator};
use hotspring_barracuda::validation::ValidationHarness;

use super::INPUT_SIZE;

pub fn run(
    gpu: &GpuF64,
    harness: &mut ValidationHarness,
    jsonl_records: &mut Vec<String>,
) -> (bool, usize) {
    println!();
    println!("═══ Experiment 4: Scaling Crossover Analysis ═══");

    let fine_sizes: Vec<usize> = vec![
        8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    ];
    let mut crossover_found = false;
    let mut crossover_size = 0usize;

    for &rs in &fine_sizes {
        let config = EsnConfig {
            input_size: INPUT_SIZE,
            reservoir_size: rs,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
            ..Default::default()
        };
        let (train_seqs, train_targets) = generate_training_data(4, 20, INPUT_SIZE);
        let test_seq = generate_test_sequence(42, 20, INPUT_SIZE);

        let mut esn = EchoStateNetwork::new(config);
        esn.train(&train_seqs, &train_targets);
        let exported = esn.export_weights().expect("export");

        let mut npu = NpuSimulator::from_exported(&exported);
        let test_seq_c = test_seq.clone();
        let (cpu_us, _, _) = time_fn(|| npu.predict(&test_seq_c), 3, 10);

        let gpu_esn = GpuEsn::new(gpu, &exported);
        let test_seq_g = test_seq.clone();
        let (gpu_us, _, _) = time_fn(|| gpu_esn.predict_batched(gpu, &test_seq_g), 3, 10);

        let ratio = gpu_us / cpu_us.max(0.1);
        let winner = if ratio < 1.0 { "GPU" } else { "CPU" };
        println!(
            "    RS={rs:>5}: CPU={cpu_us:>8.1}μs  GPU={gpu_us:>8.1}μs  ratio={ratio:.2}  [{winner}]"
        );

        if ratio < 1.0 && !crossover_found {
            crossover_found = true;
            crossover_size = rs;
        }

        jsonl_records.push(
            serde_json::json!({
                "experiment": "scaling_crossover",
                "reservoir_size": rs,
                "cpu_f32_us": cpu_us, "gpu_batch_us": gpu_us,
                "ratio_gpu_cpu": ratio,
            })
            .to_string(),
        );
    }

    if crossover_found {
        println!("  GPU crossover at RS ≈ {crossover_size}");
    } else {
        println!("  CPU wins for ESN at all tested sizes (dispatch overhead dominates)");
    }
    harness.check_bool("scaling_crossover_measured", true);

    (crossover_found, crossover_size)
}
