// SPDX-License-Identifier: AGPL-3.0-or-later

//! Experiment 2: GPU dispatch overhead isolation.

use hotspring_barracuda::bench::{GpuEsn, generate_test_sequence, generate_training_data, time_fn};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::reservoir::{EchoStateNetwork, EsnConfig};
use hotspring_barracuda::md::shaders::SHADER_ESN_RESERVOIR_UPDATE;

use super::INPUT_SIZE;

pub fn run(gpu: &GpuF64, jsonl_records: &mut Vec<String>) {
    println!("═══ Experiment 2: GPU Dispatch Overhead ═══");

    let tiny_config = EsnConfig {
        input_size: INPUT_SIZE,
        reservoir_size: 8,
        output_size: 1,
        spectral_radius: 0.95,
        connectivity: 0.5,
        leak_rate: 0.3,
        regularization: 1e-4,
        seed: 42,
        ..Default::default()
    };
    let (train_seqs, train_targets) = generate_training_data(4, 10, INPUT_SIZE);
    let mut tiny_esn = EchoStateNetwork::new(tiny_config);
    tiny_esn.train(&train_seqs, &train_targets);
    let tiny_exported = tiny_esn.export_weights().expect("export");
    let tiny_gpu_esn = GpuEsn::new(gpu, &tiny_exported);
    let tiny_seq = generate_test_sequence(777, 5, INPUT_SIZE);

    let (single_dispatch_us, _, _) = time_fn(|| tiny_gpu_esn.predict(gpu, &tiny_seq), 5, 50);
    println!("  RS=8, SeqLen=5: GPU single-dispatch = {single_dispatch_us:.1} μs");

    // Null dispatch
    let null_win = gpu.create_f32_buffer(&[0.0f32; 4], "null_win");
    let null_wres = gpu.create_f32_buffer(&[0.0f32; 16], "null_wres");
    let null_in = gpu.create_f32_buffer(&[0.0f32], "null_in");
    let null_state = gpu.create_f32_rw_buffer(&[0.0f32; 4], "null_state");
    let null_params = gpu.create_f32_buffer(&[4.0f32, 1.0, 0.3, 0.0], "null_params");
    let null_pipeline = gpu.create_pipeline(SHADER_ESN_RESERVOIR_UPDATE, "null_esn");
    let null_bg = gpu.create_bind_group(
        &null_pipeline,
        &[&null_win, &null_wres, &null_in, &null_state, &null_params],
    );
    let (null_us, _, _) = time_fn(
        || {
            gpu.dispatch(&null_pipeline, &null_bg, 1);
            vec![0.0]
        },
        10,
        100,
    );
    println!("  Null dispatch overhead: {null_us:.1} μs");
    println!();

    let record = serde_json::json!({
        "experiment": "gpu_dispatch_overhead",
        "single_dispatch_rs8_us": single_dispatch_us,
        "null_dispatch_us": null_us,
    });
    jsonl_records.push(record.to_string());
}
