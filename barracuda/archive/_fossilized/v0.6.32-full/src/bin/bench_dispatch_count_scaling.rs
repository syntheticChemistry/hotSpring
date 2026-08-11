// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dispatch Count × Generation Scaling — proving multi-dispatch cache persistence.
//!
//! Theory: NVIDIA 20× slower because each pass within a command buffer
//! evicts from 6 MB L2, forcing VRAM reload. AMD's 128 MB IC retains data.
//!
//! If true: increasing n_md_steps should show:
//! - NVIDIA: time grows SUPERLINEARLY with step count (each step cold)
//! - AMD: time grows LINEARLY with step count (data stays hot)
//! - The ratio (NVIDIA/AMD) should INCREASE with more steps.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Dispatch Count × Generation — Cache Persistence Proof       ║");
    println!("║     More MD steps = more inter-pass cache pressure              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    let dims = [12, 12, 12, 12]; // 12⁴: 20736 sites, ~6 MB links → exceeds NVIDIA L2
    let beta = 6.0;
    let step_counts: &[usize] = &[2, 5, 10, 20, 40];

    let mut all_results: Vec<(String, Vec<f64>)> = Vec::new();

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

        let pipelines = GpuHmcStreamingPipelines::new(&gpu);
        let lat = Lattice::hot_start(dims, beta, 42u64);
        let hmc_state = GpuHmcState::from_lattice(&gpu, &lat, beta);

        // Warmup
        let mut seed = 42u64;
        for i in 0..3 {
            let _ = gpu_hmc_trajectory_streaming(&gpu, &pipelines, &hmc_state, 5, 0.1, i, &mut seed);
        }

        println!("  Lattice: {}⁴ = {} sites, β={}", dims[0], dims.iter().product::<usize>(), beta);
        println!("  Link buffer: {:.1} MB", dims.iter().product::<usize>() as f64 * 4.0 * 18.0 * 8.0 / 1048576.0);
        println!();
        println!("  {:>6}  {:>10}  {:>10}  {:>12}  {:>10}", "Steps", "Time (ms)", "ms/step", "Dispatches", "µs/dispatch");
        println!("  {:>6}  {:>10}  {:>10}  {:>12}  {:>10}", "─".repeat(6), "─".repeat(10), "─".repeat(10), "─".repeat(12), "─".repeat(10));

        let mut card_times: Vec<f64> = Vec::new();

        for &n_steps in step_counts {
            let n_runs = 10;
            let t0 = Instant::now();
            for j in 0..n_runs {
                let _ = gpu_hmc_trajectory_streaming(
                    &gpu, &pipelines, &hmc_state, n_steps, 0.1, 100 + j, &mut seed,
                );
            }
            let total = t0.elapsed().as_secs_f64() * 1000.0;
            let per_traj = total / n_runs as f64;
            let per_step = per_traj / n_steps as f64;
            let dispatches = n_steps * 8; // 8 passes per Omelyan step
            let us_per_dispatch = per_traj * 1000.0 / dispatches as f64;

            println!("  {:>6}  {:>8.2}ms  {:>8.3}ms  {:>12}  {:>8.1}µs",
                n_steps, per_traj, per_step, dispatches, us_per_dispatch);

            card_times.push(per_traj);
        }
        println!();

        // Scaling analysis
        let base = card_times[0]; // 2 steps
        println!("  Scaling relative to 2 steps:");
        for (i, &n) in step_counts.iter().enumerate() {
            let expected = n as f64 / step_counts[0] as f64;
            let actual = card_times[i] / base;
            let superlinear = actual / expected;
            println!("    {:>2} steps: {:.2}× time ({:.2}× expected) → scaling factor {:.3}",
                n, actual, expected, superlinear);
        }
        println!();

        all_results.push((name, card_times));
    }

    // Cross-card ratio vs step count
    if all_results.len() >= 2 {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║     Cache Persistence Analysis                                  ║");
        println!("╚══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("  If ratio INCREASES with step count → cache eviction per pass confirmed.");
        println!();
        println!("  {:>6}  {:>12}  {:>12}  {:>10}", "Steps", &all_results[0].0[..18.min(all_results[0].0.len())], &all_results[1].0[..18.min(all_results[1].0.len())], "Ratio");
        println!("  {:>6}  {:>12}  {:>12}  {:>10}", "─".repeat(6), "─".repeat(12), "─".repeat(12), "─".repeat(10));

        for (i, &n) in step_counts.iter().enumerate() {
            let t0 = all_results[0].1[i];
            let t1 = all_results[1].1[i];
            let ratio = t0 / t1;
            println!("  {:>6}  {:>10.2}ms  {:>10.2}ms  {:>8.1}×", n, t0, t1, ratio);
        }
        println!();

        let ratio_low = all_results[0].1[0] / all_results[1].1[0];
        let ratio_high = all_results[0].1.last().unwrap() / all_results[1].1.last().unwrap();
        let divergence = ratio_high / ratio_low;

        println!("  Ratio at {} steps: {:.1}×", step_counts[0], ratio_low);
        println!("  Ratio at {} steps: {:.1}×", step_counts.last().unwrap(), ratio_high);
        println!("  Divergence factor: {:.2}× (>1.0 = cache eviction confirmed)", divergence);
        println!();

        if divergence > 1.3 {
            println!("  ✓ CONFIRMED: Inter-pass cache eviction on {} grows with step count.",
                &all_results[0].0[..18.min(all_results[0].0.len())]);
            println!("    {} retains data between passes (IC persistence).",
                &all_results[1].0[..18.min(all_results[1].0.len())]);
            println!("    This is a GENERATIONAL property: any card with <12⁴ L2 will exhibit this.");
        } else {
            println!("  ✗ NOT CONFIRMED at this volume. Both cards scale similarly.");
            println!("    Try larger lattice or different working set size.");
        }
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Dispatch Count Scaling Complete                              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
