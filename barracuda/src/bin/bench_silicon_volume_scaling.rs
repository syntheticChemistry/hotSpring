// SPDX-License-Identifier: AGPL-3.0-or-later
//! Volume scaling: how does cross-GPU performance ratio change with lattice size?
//!
//! sunMemo insight: for publication, we need to know:
//! - At what volume does NVIDIA catch up to AMD?
//! - Does the 4-5× AMD advantage hold at 16⁴ and 32⁴?
//! - Where is the silicon routing crossover point?
//!
//! This informs the arXiv paper's hardware section and future card acquisition.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct ScalingPoint {
    dims: [usize; 4],
    volume: usize,
    ms_per_traj: f64,
    plaquette: f64,
}

fn measure_at_volume(gpu: &GpuF64, dims: [usize; 4], beta: f64) -> ScalingPoint {
    let seed = 12345u64;
    let pipelines = GpuHmcStreamingPipelines::new(gpu);
    let lat = Lattice::hot_start(dims, beta, seed);
    let hmc_state = GpuHmcState::from_lattice(gpu, &lat, beta);
    let mut rng_seed = seed;

    // Warm up
    for i in 0..5 {
        let _ = gpu_hmc_trajectory_streaming(
            gpu, &pipelines, &hmc_state, 10, 0.1, i, &mut rng_seed,
        );
    }

    // Measure
    let n_measure = 10;
    let t0 = Instant::now();
    let mut last_plaq = 0.0;
    for j in 0..n_measure {
        if let Ok(r) = gpu_hmc_trajectory_streaming(
            gpu, &pipelines, &hmc_state, 10, 0.1, 10 + j, &mut rng_seed,
        ) {
            last_plaq = r.plaquette;
        }
    }
    let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
    let ms_per_traj = elapsed / n_measure as f64;

    let volume: usize = dims.iter().product();
    ScalingPoint {
        dims,
        volume,
        ms_per_traj,
        plaquette: last_plaq,
    }
}

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Volume Scaling: Cross-GPU Performance vs Lattice Size");
    println!("  sunMemo: deeper results = larger volumes = need optimal routing");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let beta = 6.0;
    let volumes: Vec<[usize; 4]> = vec![
        [4, 4, 4, 4],
        [6, 6, 6, 6],
        [8, 8, 8, 8],
        [10, 10, 10, 10],
        [12, 12, 12, 12],
        [16, 16, 16, 16],
    ];

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let discrete: Vec<_> = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .collect();

    if discrete.len() < 2 {
        eprintln!("Need 2 discrete GPUs.");
        return;
    }

    let mut gpu_list: Vec<GpuF64> = Vec::new();
    for adapter in discrete {
        match GpuF64::from_adapter(adapter).await {
            Ok(g) => gpu_list.push(g),
            Err(_) => continue,
        }
    }

    println!("  GPUs:");
    for g in &gpu_list {
        println!("    • {}", g.adapter_name);
    }
    println!();
    println!("  Volumes: {:?}", volumes.iter().map(|d| format!("{}⁴", d[0])).collect::<Vec<_>>());
    println!("  β = {}, SU(3), n_md=10, dt=0.1", beta);
    println!();

    // Collect results
    let mut all_results: Vec<Vec<ScalingPoint>> = Vec::new();
    for gpu in &gpu_list {
        println!("  Profiling {} ...", gpu.adapter_name);
        let mut card_results = Vec::new();
        for dims in &volumes {
            let sp = measure_at_volume(gpu, *dims, beta);
            print!("    {}⁴: {:.2} ms  ", dims[0], sp.ms_per_traj);
            card_results.push(sp);
        }
        println!();
        all_results.push(card_results);
    }
    println!();

    // Print comparison table
    println!("━━━ Volume Scaling Comparison ━━━");
    println!();
    println!("  {:>6} {:>8} {:>14} {:>14} {:>10} {:>10}",
             "L", "Volume", &gpu_list[0].adapter_name[..14.min(gpu_list[0].adapter_name.len())],
             &gpu_list[1].adapter_name[..14.min(gpu_list[1].adapter_name.len())],
             "Ratio", "Winner");
    println!("  {:>6} {:>8} {:>14} {:>14} {:>10} {:>10}",
             "─".repeat(6), "─".repeat(8), "─".repeat(14), "─".repeat(14), "─".repeat(10), "─".repeat(10));

    for i in 0..volumes.len() {
        let t0 = all_results[0][i].ms_per_traj;
        let t1 = all_results[1][i].ms_per_traj;
        let ratio = t0 / t1;
        let winner = if t0 < t1 {
            &gpu_list[0].adapter_name
        } else {
            &gpu_list[1].adapter_name
        };
        let winner_short = if winner.contains("NVIDIA") { "NVIDIA" } else { "AMD" };
        println!("  {:>5}⁴ {:>8} {:>12.2}ms {:>12.2}ms {:>8.2}× {:>10}",
                 volumes[i][0],
                 all_results[0][i].volume,
                 t0, t1, ratio, winner_short);
    }
    println!();

    // Scaling analysis
    println!("━━━ Scaling Analysis ━━━");
    println!();
    for (g_idx, gpu) in gpu_list.iter().enumerate() {
        let results = &all_results[g_idx];
        if results.len() >= 2 {
            let small_vol = results[0].volume as f64;
            let large_vol = results.last().unwrap().volume as f64;
            let small_t = results[0].ms_per_traj;
            let large_t = results.last().unwrap().ms_per_traj;
            let scale_factor = large_vol / small_vol;
            let time_factor = large_t / small_t;
            let efficiency = scale_factor / time_factor;

            println!("  {}:", gpu.adapter_name);
            println!("    Volume scale: {:.0}× ({}⁴ → {}⁴)",
                     scale_factor, volumes[0][0], volumes.last().unwrap()[0]);
            println!("    Time scale: {:.1}×", time_factor);
            println!("    Weak scaling efficiency: {:.1}%", efficiency * 100.0);
            println!();
        }
    }

    // Cost analysis
    println!("━━━ Cost-per-Configuration Analysis ━━━");
    println!();
    println!("  At 16⁴ (production volume for arXiv paper):");
    if all_results[0].len() >= 6 && all_results[1].len() >= 6 {
        let nv_16 = all_results[0][5].ms_per_traj;
        let amd_16 = all_results[1][5].ms_per_traj;
        let n_configs = 200;
        let n_therm = 100;
        let total_traj = n_configs + n_therm;

        println!("    {}: {:.0} ms/traj → {:.1}s for {} configs",
                 gpu_list[0].adapter_name, nv_16, nv_16 * total_traj as f64 / 1000.0, n_configs);
        println!("    {}: {:.0} ms/traj → {:.1}s for {} configs",
                 gpu_list[1].adapter_name, amd_16, amd_16 * total_traj as f64 / 1000.0, n_configs);
        println!();

        let faster = nv_16.min(amd_16);
        let slower = nv_16.max(amd_16);
        let savings_pct = (1.0 - faster / slower) * 100.0;
        let faster_name = if nv_16 < amd_16 { &gpu_list[0].adapter_name } else { &gpu_list[1].adapter_name };
        println!("    → Route to {}: {:.0}% cheaper per config", faster_name, savings_pct);
        println!("    → Cooperative: run NVIDIA 16⁴ + AMD 8⁴ in parallel for maximum throughput");
    }
    println!();

    // Plaquette cross-check at each volume
    println!("━━━ Cross-Silicon Plaquette Parity ━━━");
    println!();
    println!("  {:>6} {:>14} {:>14} {:>12}", "L⁴", "GPU 0 ⟨P⟩", "GPU 1 ⟨P⟩", "Δ");
    for i in 0..volumes.len() {
        let p0 = all_results[0][i].plaquette;
        let p1 = all_results[1][i].plaquette;
        let delta = (p0 - p1).abs();
        println!("  {:>5}⁴ {:>14.10} {:>14.10} {:>12.2e}", volumes[i][0], p0, p1, delta);
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Volume Scaling Complete — silicon routing table generated");
    println!("  toadStool + barracuda absorb: route by volume to optimal card");
    println!("═══════════════════════════════════════════════════════════════════");
}
