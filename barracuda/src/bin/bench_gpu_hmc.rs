// SPDX-License-Identifier: AGPL-3.0-only

//! GPU HMC Scaling Benchmark — CPU vs pure GPU at production lattice sizes.
//!
//! At each lattice size: run N trajectories on CPU and GPU, report wall time
//! per trajectory and speedup ratio. All GPU math at fp64 via WGSL shaders.
//!
//! | Lattice | CPU (ms/traj) | GPU (ms/traj) | Speedup |
//! |---------|:---:|:---:|:---:|
//! | 4⁴ | — | — | — |
//! | 8⁴ | — | — | — |
//! | 8³×16 | — | — | — |
//! | 16⁴ | — | — | — |

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_hmc_trajectory, GpuHmcPipelines, GpuHmcState,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

#[allow(dead_code)]
struct BenchResult {
    label: String,
    volume: usize,
    cpu_ms: f64,
    gpu_ms: f64,
    cpu_accept: usize,
    gpu_accept: usize,
    n_traj: usize,
}

fn bench_cpu(dims: [usize; 4], beta: f64, n_traj: usize) -> (f64, usize, f64) {
    let mut lat = Lattice::hot_start(dims, beta, 42);
    let n_md = 10;
    let dt = 0.05;

    // Thermalize
    let mut cfg = HmcConfig {
        n_md_steps: n_md,
        dt,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    for _ in 0..10 {
        hmc::hmc_trajectory(&mut lat, &mut cfg);
    }

    // Benchmark
    let start = Instant::now();
    let mut accepted = 0;
    let mut plaq_sum = 0.0;
    for _ in 0..n_traj {
        let r = hmc::hmc_trajectory(&mut lat, &mut cfg);
        if r.accepted {
            accepted += 1;
        }
        plaq_sum += r.plaquette;
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    (elapsed / n_traj as f64, accepted, plaq_sum / n_traj as f64)
}

fn bench_gpu(
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    dims: [usize; 4],
    beta: f64,
    n_traj: usize,
) -> (f64, usize, f64) {
    let mut lat = Lattice::hot_start(dims, beta, 42);
    let n_md = 10;
    let dt = 0.05;

    // Thermalize on CPU (same starting point)
    let mut cfg = HmcConfig {
        n_md_steps: n_md,
        dt,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    for _ in 0..10 {
        hmc::hmc_trajectory(&mut lat, &mut cfg);
    }

    // Upload to GPU
    let state = GpuHmcState::from_lattice(gpu, &lat, beta);

    // Warmup (1 trajectory)
    let mut seed = 1000u64;
    gpu_hmc_trajectory(gpu, pipelines, &state, n_md, dt, &mut seed);

    // Benchmark
    let start = Instant::now();
    let mut accepted = 0;
    let mut plaq_sum = 0.0;
    for _ in 0..n_traj {
        let r = gpu_hmc_trajectory(gpu, pipelines, &state, n_md, dt, &mut seed);
        if r.accepted {
            accepted += 1;
        }
        plaq_sum += r.plaquette;
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    (elapsed / n_traj as f64, accepted, plaq_sum / n_traj as f64)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GPU HMC Scaling Benchmark — CPU vs Pure GPU (fp64)        ║");
    println!("║  Omelyan integrator, n_md=10, dt=0.05, β=6.0              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU not available: {e}");
            return;
        }
    };
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    let configs: Vec<(&str, [usize; 4])> = vec![
        ("4⁴", [4, 4, 4, 4]),
        ("8⁴", [8, 8, 8, 8]),
        ("8³×16", [8, 8, 8, 16]),
        ("16⁴", [16, 16, 16, 16]),
    ];

    let beta = 6.0;
    let mut results = Vec::new();

    let pipelines = GpuHmcPipelines::new(&gpu);

    for (label, dims) in &configs {
        let vol: usize = dims.iter().product();
        let n_traj = if vol <= 4096 { 20 } else { 5 };

        println!("═══ {label} (V={vol}, {n_traj} trajectories) ═══");

        let (cpu_ms, cpu_acc, cpu_plaq) = bench_cpu(*dims, beta, n_traj);
        println!("  CPU: {cpu_ms:.1} ms/traj, {cpu_acc}/{n_traj} accepted, plaq={cpu_plaq:.6}");

        let (gpu_ms, gpu_acc, gpu_plaq) = bench_gpu(&gpu, &pipelines, *dims, beta, n_traj);
        println!("  GPU: {gpu_ms:.1} ms/traj, {gpu_acc}/{n_traj} accepted, plaq={gpu_plaq:.6}");

        let speedup = cpu_ms / gpu_ms;
        if speedup >= 1.0 {
            println!("  Speedup: {speedup:.1}× (GPU faster)");
        } else {
            println!(
                "  Speedup: {:.1}× (CPU faster — dispatch overhead dominates)",
                1.0 / speedup
            );
        }
        println!();

        results.push(BenchResult {
            label: label.to_string(),
            volume: vol,
            cpu_ms,
            gpu_ms,
            cpu_accept: cpu_acc,
            gpu_accept: gpu_acc,
            n_traj,
        });
    }

    // Summary table
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary: GPU HMC Scaling                                  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Lattice  │ Volume  │ CPU ms/traj │ GPU ms/traj │ Speedup  ║");
    println!("╠══════════╪═════════╪═════════════╪═════════════╪══════════╣");
    for r in &results {
        let speedup = r.cpu_ms / r.gpu_ms;
        let tag = if speedup >= 1.0 {
            format!("{speedup:.1}×")
        } else {
            format!("1/{:.1}×", 1.0 / speedup)
        };
        println!(
            "║ {:<8} │ {:>7} │ {:>11.1} │ {:>11.1} │ {:>8} ║",
            r.label, r.volume, r.cpu_ms, r.gpu_ms, tag
        );
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}
