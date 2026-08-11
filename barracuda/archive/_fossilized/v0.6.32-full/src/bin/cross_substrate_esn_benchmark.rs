// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-Substrate ESN Benchmark: GPU vs CPU vs NPU
//!
//! Runs identical Echo State Network workloads on every available substrate
//! to characterize where each excels, where each fails, and what the NPU
//! can actually handle versus software implementations on CPU/GPU.
//!
//! # Substrates Tested
//!
//! | Substrate | Precision | Implementation |
//! |-----------|-----------|----------------|
//! | CPU-f64   | f64       | `EchoStateNetwork::predict()` |
//! | CPU-f32   | f32       | `NpuSimulator::predict()` |
//! | GPU-f32   | f32       | WGSL `esn_reservoir_update` + `esn_readout` shaders |
//! | NPU-sim   | f32→int8  | `NpuSimulator` (Akida behavioral model) |
//!
//! # Experiments
//!
//! 1. Cross-substrate timing matrix at reservoir sizes 16..500
//! 2. GPU dispatch overhead isolation
//! 3. NPU capability envelope (threshold, streaming, multi-output, mutation)
//! 4. Scaling crossover (find RS where GPU beats CPU)
//! 5. GPU as ESN reservoir (large RS + multi-output)
//! 6. QCD-specific workload comparison across substrates

use hotspring_barracuda::bench::SubstrateResult;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

#[path = "../bin_helpers/cross_substrate_esn/mod.rs"]
mod cross_substrate_esn;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Substrate ESN Benchmark: CPU × GPU × NPU            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("cross_substrate_esn_benchmark");
    let campaign_start = Instant::now();

    // Initialize GPU
    println!("═══ Initializing GPU ═══");
    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU unavailable: {e}");
            println!("  Skipping GPU experiments — running CPU/NPU only.");
            cross_substrate_esn::cpu_only::run(&mut harness);
            harness.finish();
        }
    };
    println!("  Adapter: {}", gpu.adapter_name);
    println!("  FP64 support: {}", gpu.has_f64);
    println!();

    let reservoir_sizes: Vec<usize> = vec![16, 32, 50, 100, 200, 500];
    let mut all_results: Vec<SubstrateResult> = Vec::new();
    let mut jsonl_records: Vec<String> = Vec::new();

    cross_substrate_esn::exp1_timing_matrix::run(
        &gpu,
        &mut harness,
        &mut all_results,
        &mut jsonl_records,
        &reservoir_sizes,
    );
    cross_substrate_esn::exp2_gpu_dispatch::run(&gpu, &mut jsonl_records);
    cross_substrate_esn::exp3_npu_envelope::run(&gpu, &mut harness, &mut jsonl_records);
    let (crossover_found, crossover_size) =
        cross_substrate_esn::exp4_scaling_crossover::run(&gpu, &mut harness, &mut jsonl_records);
    cross_substrate_esn::exp5_gpu_esn::run(&gpu, &mut harness, &mut jsonl_records);
    cross_substrate_esn::exp6_qcd_workload::run(&gpu, &mut harness, &mut jsonl_records);
    cross_substrate_esn::summary::run(
        &all_results,
        &jsonl_records,
        &reservoir_sizes,
        crossover_found,
        crossover_size,
        campaign_start,
    );

    harness.finish();
}
