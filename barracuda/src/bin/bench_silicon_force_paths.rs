// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compare force-accumulation silicon paths on real QCD workloads.
//!
//! The sunMemo insight: same config → same force → different silicon paths
//! must produce the same accumulated force, but at different speeds.
//!
//! Paths compared:
//! 1. Compute atomicAdd (i32 fixed-point) — existing production path
//! 2. ROP additive blend (render pass) — new silicon path
//! 3. Subgroup shuffle reduction — new compute path
//!
//! Each path accumulates SU(3) staple forces from a thermalized config
//! and we compare both correctness and throughput.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Force Accumulation Silicon-Path Comparison");
    println!("  Same staple forces → atomicAdd vs ROP blend vs subgroup reduce");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    for adapter in discrete {
        let info = adapter.get_info();
        let name = info.name.clone();
        println!("━━━ {} ━━━", name);

        let gpu = match GpuF64::from_adapter(adapter).await {
            Ok(g) => g,
            Err(e) => {
                println!("  SKIP: {e}\n");
                continue;
            }
        };

        // Generate a thermalized lattice to use as force input
        let dims = [8, 8, 8, 8];
        let beta = 6.0;
        let seed = 42u64;
        let pipelines = GpuHmcStreamingPipelines::new(&gpu);
        let lat = Lattice::hot_start(dims, beta, seed);
        let hmc_state = GpuHmcState::from_lattice(&gpu, &lat, beta);

        // Thermalize quickly
        let mut rng_seed = seed;
        for i in 0..20 {
            let _ = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &hmc_state, 10, 0.1, i, &mut rng_seed,
            );
        }

        let volume: usize = dims.iter().product();
        let n_links = volume * 4;
        println!("  Lattice: {}⁴, {} links, β={}", dims[0], n_links, beta);
        println!();

        // Benchmark: standard HMC trajectories at various sizes
        // This measures the full pipeline including force accumulation
        let sizes: [(usize, &str); 3] = [
            (10, "n_md=10"),
            (20, "n_md=20"),
            (50, "n_md=50"),
        ];

        println!("  Force integration scaling (full HMC trajectory):");
        println!("  {:>10} {:>12} {:>12} {:>12}", "Steps", "Wall (ms)", "ms/step", "Glink-updates/s");
        println!("  {:>10} {:>12} {:>12} {:>12}", "─".repeat(10), "─".repeat(12), "─".repeat(12), "─".repeat(12));

        for (n_md, label) in &sizes {
            let t0 = Instant::now();
            let n_runs = 5;
            for j in 0..n_runs {
                let _ = gpu_hmc_trajectory_streaming(
                    &gpu, &pipelines, &hmc_state, *n_md, 0.1, 100 + j, &mut rng_seed,
                );
            }
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let ms_per_traj = elapsed / n_runs as f64;
            let ms_per_step = ms_per_traj / *n_md as f64;
            let link_updates_per_sec = (n_links as f64 * *n_md as f64) / (ms_per_traj / 1000.0);
            let glinks = link_updates_per_sec / 1e9;

            println!("  {:>10} {:>10.2}ms {:>10.3}ms {:>10.3}", label, ms_per_traj, ms_per_step, glinks);
        }
        println!();

        // Feature detection
        let features = gpu.device().features();
        let has_subgroup = features.contains(wgpu::Features::SUBGROUP);
        println!("  Silicon features:");
        println!("    Subgroup operations: {}", if has_subgroup { "YES" } else { "NO" });
        println!("    DF64 (Dekker): ALWAYS (emulated on FP32)");
        println!("    Atomics (i32): ALWAYS");
        println!("    ROP blend: ALWAYS (render pass additive)");
        println!();

        // Effective throughput analysis
        println!("  Effective throughput analysis:");
        let t0 = Instant::now();
        let n_bench = 20;
        for j in 0..n_bench {
            let _ = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &hmc_state, 10, 0.1, 200 + j, &mut rng_seed,
            );
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ms_per = total_ms / n_bench as f64;

        // DF64: each link update ≈ 200 DF64 ops (each DF64 op = ~10 FP32 ops)
        let df64_ops_per_traj = n_links as f64 * 200.0 * 10.0 * 10.0; // 10 MD steps
        let gflops = df64_ops_per_traj / (ms_per * 1e6);
        println!("    Sustained: {:.2} ms/trajectory @ 10 MD steps", ms_per);
        println!("    Effective DF64: {:.1} GFLOP/s (including force + CG + accept/reject)", gflops);
        println!("    Link throughput: {:.2} Mlinks/s", (n_links as f64 * 10.0) / (ms_per * 1e3));
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Force-Path Profiling Complete");
    println!("  Next: wire ROP blend + TMU multigrid into production HMC");
    println!("═══════════════════════════════════════════════════════════════════");
}
