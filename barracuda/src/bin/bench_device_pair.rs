// SPDX-License-Identifier: AGPL-3.0-only

//! Heterogeneous dual-GPU profiling benchmark.
//!
//! Discovers and profiles both GPUs (Titan V + 3090 or equivalent), measures:
//! 1. PCIe transfer cost (predicted vs actual)
//! 2. Per-card BCS throughput
//! 3. Workload planner split predictions
//! 4. Cross-validation (redundant dispatch)
//!
//! GPU selection: uses `HOTSPRING_GPU_PRIMARY` / `HOTSPRING_GPU_SECONDARY`
//! env vars, or auto-discovers by memory/capability.
//!
//! Example:
//!   `HOTSPRING_GPU_PRIMARY=titan` `HOTSPRING_GPU_SECONDARY=3090` \
//!     cargo run --release --bin bench_device_pair

use hotspring_barracuda::device_pair::DevicePair;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::precision_routing::PhysicsDomain;
use hotspring_barracuda::workload_planner;

use std::time::Instant;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    println!("═══════════════════════════════════════════════════════════");
    println!("  Heterogeneous Dual-GPU Device Pair Benchmark");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let pair = match rt.block_on(DevicePair::discover()) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  DevicePair::discover failed: {e}");
            eprintln!("  Set HOTSPRING_GPU_PRIMARY and HOTSPRING_GPU_SECONDARY");
            return;
        }
    };

    println!("{pair}");
    println!();

    // ── Phase 1: Hardware report card ─────────────────────────────────
    println!("── Phase 1: Hardware Report Card ─────────────────────────");
    println!();

    print_card_profile("Precise", &pair.precise);
    print_card_profile("Throughput", &pair.throughput);

    println!();
    println!(
        "  Bridge: {} ↔ {}",
        pair.profile.precise_name, pair.profile.throughput_name
    );
    println!(
        "    Bandwidth: {:.1} GB/s ({})",
        pair.profile.bridge_bandwidth_gbps,
        pair.bridge.tier.bandwidth_gbps()
    );
    println!("    Latency:   {:.0} us", pair.profile.bridge_latency_us);
    println!(
        "    1 MB:      {:.1} us predicted",
        pair.profile.transfer_us(1_048_576)
    );
    println!(
        "    Split fraction: {:.1}% precise / {:.1}% throughput",
        pair.profile.precise_split_fraction() * 100.0,
        (1.0 - pair.profile.precise_split_fraction()) * 100.0
    );
    println!();

    // ── Phase 2: PCIe transfer measurement ────────────────────────────
    println!("── Phase 2: PCIe Transfer Measurement ───────────────────");
    println!();

    for &size_mb in &[0.5, 1.0, 4.0, 16.0] {
        let size_bytes = (size_mb * 1_048_576.0) as usize;
        let n_f64 = size_bytes / 8;
        let data: Vec<f64> = (0..n_f64).map(|i| i as f64 * 0.001).collect();

        let predicted_us = pair.profile.transfer_us(size_bytes);

        let t0 = Instant::now();
        let buf = pair.precise.create_f64_buffer(&data, "xfer_bench");
        pair.precise.queue().submit([]);
        let _ = pair.precise.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let actual_us = t0.elapsed().as_secs_f64() * 1e6;

        drop(buf);

        let ratio = actual_us / predicted_us.max(1.0);
        println!(
            "  {size_mb:>5.1} MB: predicted {predicted_us:>8.1} us, actual {actual_us:>8.1} us (ratio {ratio:.2}x)"
        );
    }
    println!();

    // ── Phase 3: Per-card BCS throughput ──────────────────────────────
    println!("── Phase 3: Per-Card BCS Throughput ─────────────────────");
    println!();

    let bcs_batch = 2048;
    let bcs_levels = 20;

    let time_precise = bench_bcs(&pair.precise, bcs_batch, bcs_levels);
    let time_throughput = bench_bcs(&pair.throughput, bcs_batch, bcs_levels);

    println!(
        "  {:<30} {:>10.3} ms  ({:.0} BCS/s)",
        pair.profile.precise_name,
        time_precise * 1e3,
        bcs_batch as f64 / time_precise
    );
    println!(
        "  {:<30} {:>10.3} ms  ({:.0} BCS/s)",
        pair.profile.throughput_name,
        time_throughput * 1e3,
        bcs_batch as f64 / time_throughput
    );

    let ratio = if time_precise < time_throughput {
        format!("Precise is {:.1}x faster", time_throughput / time_precise)
    } else {
        format!(
            "Throughput is {:.1}x faster",
            time_precise / time_throughput
        )
    };
    println!("  → {ratio}");
    println!();

    // ── Phase 4: Workload planner predictions ─────────────────────────
    println!("── Phase 4: Workload Planner Predictions ─────────────────");
    println!();

    let domains = [
        ("Dielectric", PhysicsDomain::Dielectric),
        ("Eigensolve", PhysicsDomain::Eigensolve),
        ("LatticeQCD", PhysicsDomain::LatticeQcd),
        ("GradientFlow", PhysicsDomain::GradientFlow),
        ("MolDynamics", PhysicsDomain::MolecularDynamics),
        ("KineticFluid", PhysicsDomain::KineticFluid),
        ("NuclearEOS", PhysicsDomain::NuclearEos),
    ];

    for (name, domain) in &domains {
        let assignment_small = workload_planner::plan_workload(
            &pair, *domain, 1_048_576, // 1MB data
            100_000.0, // 100ms compute
        );
        let assignment_large = workload_planner::plan_workload(
            &pair,
            *domain,
            1_048_576,   // 1MB data
            1_000_000.0, // 1s compute
        );
        println!("  {name:<14}  small={assignment_small:<18}  large={assignment_large:<18}");
    }
    println!();

    // ── Phase 5: Summary ──────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!(
        "  Precise brain:    {} ({:.1} TFLOPS f64 native)",
        pair.profile.precise_name, pair.profile.precise_tflops_f64
    );
    println!(
        "  Throughput brain: {} ({:.1} TFLOPS DF64 effective)",
        pair.profile.throughput_name, pair.profile.throughput_tflops_df64
    );
    println!(
        "  Optimal split:    {:.0}% precise / {:.0}% throughput",
        pair.profile.precise_split_fraction() * 100.0,
        (1.0 - pair.profile.precise_split_fraction()) * 100.0
    );
    println!("  BCS advantage:    {ratio}");
}

#[allow(deprecated)]
fn print_card_profile(role: &str, gpu: &GpuF64) {
    let strategy = gpu.driver_profile().fp64_strategy();
    let rate = gpu.driver_profile().fp64_rate;
    let routing = gpu.driver_profile().precision_routing();
    println!("  {role}: {}", gpu.adapter_name);
    println!("    FP64 rate:     {rate:?}");
    println!("    Strategy:      {strategy:?}");
    println!("    Precision:     {routing:?}");
    println!(
        "    Full DF64:     {}",
        if gpu.full_df64_mode { "YES" } else { "no" }
    );
    println!();
}

fn bench_bcs(gpu: &GpuF64, batch: usize, n_levels: usize) -> f64 {
    use hotspring_barracuda::physics::bcs_gpu::BcsBisectionGpu;

    let eigenvalues: Vec<f64> = (0..batch * n_levels)
        .map(|i| -5.0 + 10.0 * (i % n_levels) as f64 / n_levels as f64)
        .collect();
    let delta = vec![2.0; batch];
    let target_n: Vec<f64> = (0..batch).map(|i| (i % 10 + 2) as f64).collect();
    let lower = vec![-50.0_f64; batch];
    let upper = vec![200.0_f64; batch];

    let bcs = BcsBisectionGpu::new(gpu, 100, 1e-12);

    for _ in 0..3 {
        let _ = bcs.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
    }

    let n_measure = 10;
    let t0 = Instant::now();
    for _ in 0..n_measure {
        let _ = bcs.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
    }
    t0.elapsed().as_secs_f64() / n_measure as f64
}
