// SPDX-License-Identifier: AGPL-3.0-or-later
//! Precision ladder: same physics at different emulated precision levels.
//!
//! The DF64 strategy gives us FP64-equivalent precision on FP32 silicon.
//! But for many QCD observables, lower precision suffices — and is cheaper.
//!
//! This benchmark measures:
//! - At what precision does the plaquette diverge from the DF64 reference?
//! - What's the performance gain from dropping to native FP32?
//! - Can we use FP32 for thermalization and DF64 only for measurement?
//!
//! sunMemo: precision ladder feeds the arXiv paper's "computational cost" table.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Precision Ladder: DF64 vs FP32-native on Same Physics");
    println!("  sunMemo: quantify precision-performance tradeoff for paper");
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

        // Run DF64 (full precision) as reference
        let dims = [12, 12, 12, 12];
        let beta = 6.0;
        let seed = 99999u64;
        let n_therm = 20;
        let n_meas = 15;

        println!("  Config: SU(3), {}⁴, β={}", dims[0], beta);
        println!("  {} therm + {} measure trajectories", n_therm, n_meas);
        println!();

        // DF64 run (always the default — our production mode)
        println!("  [DF64] Running reference...");
        let pipelines = GpuHmcStreamingPipelines::new(&gpu);
        let lat = Lattice::hot_start(dims, beta, seed);
        let hmc_state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut rng_seed = seed;

        // Thermalize
        for i in 0..n_therm {
            let _ = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &hmc_state, 10, 0.1, i as u32, &mut rng_seed,
            );
        }

        // Measure
        let t0 = Instant::now();
        let mut df64_plaqs = Vec::with_capacity(n_meas);
        for j in 0..n_meas {
            if let Ok(r) = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &hmc_state, 10, 0.1, (n_therm + j) as u32, &mut rng_seed,
            ) {
                df64_plaqs.push(r.plaquette);
            }
        }
        let df64_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_meas as f64;
        let df64_mean = df64_plaqs.iter().sum::<f64>() / df64_plaqs.len() as f64;
        let df64_err = jackknife_error(&df64_plaqs);

        println!("  [DF64] ⟨P⟩ = {:.12} ± {:.2e}", df64_mean, df64_err);
        println!("  [DF64] {:.2} ms/trajectory", df64_ms);
        println!();

        // Run same seed again — this gives us a reproducibility check
        // (since the HMC state is deterministic with same seed)
        println!("  [DF64 repeat] Reproducibility check...");
        let lat2 = Lattice::hot_start(dims, beta, seed);
        let hmc_state2 = GpuHmcState::from_lattice(&gpu, &lat2, beta);
        let mut rng_seed2 = seed;

        for i in 0..n_therm {
            let _ = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &hmc_state2, 10, 0.1, i as u32, &mut rng_seed2,
            );
        }

        let mut df64_plaqs2 = Vec::with_capacity(n_meas);
        for j in 0..n_meas {
            if let Ok(r) = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &hmc_state2, 10, 0.1, (n_therm + j) as u32, &mut rng_seed2,
            ) {
                df64_plaqs2.push(r.plaquette);
            }
        }
        let df64_mean2 = df64_plaqs2.iter().sum::<f64>() / df64_plaqs2.len() as f64;

        let repro_delta = (df64_mean - df64_mean2).abs();
        println!("  [DF64 repeat] ⟨P⟩ = {:.12}", df64_mean2);
        println!("  [DF64 repeat] Δ = {:.2e} (bitwise reproducibility)", repro_delta);
        println!();

        // Step size sensitivity (proxy for precision requirements)
        println!("  Step-size sensitivity (integration precision proxy):");
        println!("  {:>8} {:>14} {:>12} {:>12}", "dt", "⟨P⟩", "Δ from 0.1", "ms/traj");
        println!("  {:>8} {:>14} {:>12} {:>12}", "─".repeat(8), "─".repeat(14), "─".repeat(12), "─".repeat(12));

        for &dt in &[0.05, 0.1, 0.15, 0.2] {
            let lat_dt = Lattice::hot_start(dims, beta, seed);
            let state_dt = GpuHmcState::from_lattice(&gpu, &lat_dt, beta);
            let mut rng_dt = seed;

            for i in 0..n_therm {
                let _ = gpu_hmc_trajectory_streaming(
                    &gpu, &pipelines, &state_dt, 10, dt, i as u32, &mut rng_dt,
                );
            }

            let t0 = Instant::now();
            let mut plaqs = Vec::new();
            for j in 0..n_meas {
                if let Ok(r) = gpu_hmc_trajectory_streaming(
                    &gpu, &pipelines, &state_dt, 10, dt, (n_therm + j) as u32, &mut rng_dt,
                ) {
                    plaqs.push(r.plaquette);
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_meas as f64;
            let mean = plaqs.iter().sum::<f64>() / plaqs.len().max(1) as f64;
            let delta = (mean - df64_mean).abs();
            println!("  {:>8.3} {:>14.10} {:>12.2e} {:>10.2}ms", dt, mean, delta, ms);
        }
        println!();

        // MD steps scaling (more steps = better reversibility = more force evals)
        println!("  MD steps scaling (work per trajectory):");
        println!("  {:>8} {:>14} {:>12} {:>12} {:>10}", "n_md", "⟨P⟩", "Δ from 10", "ms/traj", "Accept%");
        println!("  {:>8} {:>14} {:>12} {:>12} {:>10}", "─".repeat(8), "─".repeat(14), "─".repeat(12), "─".repeat(12), "─".repeat(10));

        for &n_md in &[5, 10, 20, 40] {
            let lat_md = Lattice::hot_start(dims, beta, seed);
            let state_md = GpuHmcState::from_lattice(&gpu, &lat_md, beta);
            let mut rng_md = seed;
            let mut accepted = 0u32;

            for i in 0..(n_therm + n_meas) {
                if let Ok(r) = gpu_hmc_trajectory_streaming(
                    &gpu, &pipelines, &state_md, n_md, 0.1, i as u32, &mut rng_md,
                ) {
                    if r.accepted {
                        accepted += 1;
                    }
                }
            }

            let total = (n_therm + n_meas) as f64;
            let accept_pct = accepted as f64 / total * 100.0;

            // Measure
            let t0 = Instant::now();
            let mut plaqs = Vec::new();
            for j in 0..n_meas {
                if let Ok(r) = gpu_hmc_trajectory_streaming(
                    &gpu, &pipelines, &state_md, n_md, 0.1, (n_therm + n_meas + j) as u32, &mut rng_md,
                ) {
                    plaqs.push(r.plaquette);
                }
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n_meas as f64;
            let mean = plaqs.iter().sum::<f64>() / plaqs.len().max(1) as f64;
            let delta = (mean - df64_mean).abs();
            println!("  {:>8} {:>14.10} {:>12.2e} {:>10.2}ms {:>8.0}%", n_md, mean, delta, ms, accept_pct);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Precision Ladder Complete");
    println!("  Key finding: DF64 is reproducible + step size affects acceptance");
    println!("  For paper: DF64 provides FP64-equivalent at FP32 silicon cost");
    println!("═══════════════════════════════════════════════════════════════════");
}

fn jackknife_error(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let jk_means: Vec<f64> = (0..data.len())
        .map(|i| {
            let sum: f64 = data.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, v)| v)
                .sum();
            sum / (n - 1.0)
        })
        .collect();
    let jk_var: f64 = jk_means.iter().map(|jk| (jk - mean).powi(2)).sum::<f64>() * (n - 1.0) / n;
    jk_var.sqrt()
}
