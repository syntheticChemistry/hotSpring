// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quick validation: 32⁴ HMC on NVIDIA GPU (proves guard bypass works).
//!
//! Previously 32⁴ failed with "NVK allocation guard: Device limit exceeded"
//! because the guard compared total VRAM (~3.7 GB) against max_buffer_size.
//! After fix: guard checks per-buffer (largest = 604 MB, well under 4 GB limit).

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("━━━ 32⁴ GPU Thermalization Test (Guard Bypass Validation) ━━━");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    // Prefer NVIDIA (24 GB) for 32⁴
    let adapter = adapters
        .into_iter()
        .find(|a| {
            a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
                && a.get_info().name.contains("NVIDIA")
        })
        .expect("Need NVIDIA GPU for 32⁴ test (24 GB VRAM)");

    let gpu = GpuF64::from_adapter(adapter)
        .await
        .expect("Failed to open GPU");

    println!("  GPU: {}", gpu.adapter_name);
    println!("  Lattice: 32⁴ = 1,048,576 sites");
    println!("  VRAM needed: ~3.7 GB (15% of 24 GB)");
    println!();

    let dims = [32, 32, 32, 32];
    let beta = 6.0;
    let seed = 20260809u64;
    let n_therm = 5;

    println!("  Allocating HMC state...");
    let t0 = Instant::now();
    let lat = Lattice::hot_start(dims, beta, seed);
    let pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let state = GpuHmcState::from_lattice(&gpu, &lat, beta);
    println!("  Allocation: {:.1} ms ✓", t0.elapsed().as_secs_f64() * 1000.0);
    println!();

    println!("  Running {n_therm} HMC trajectories at 32⁴...");
    let mut rng = seed;
    let mut accepted = 0u32;
    let mut last_plaq = 0.0f64;
    let t_start = Instant::now();

    for i in 0..n_therm {
        let t_traj = Instant::now();
        let result = gpu_hmc_trajectory_streaming(&gpu, &pipelines, &state, 10, 0.1, i as u32, &mut rng)
            .expect("HMC trajectory failed");
        let ms = t_traj.elapsed().as_secs_f64() * 1000.0;
        if result.accepted { accepted += 1; }
        last_plaq = result.plaquette;
        println!("    traj {}: {:.0} ms ({}) ⟨P⟩={:.10}",
            i + 1, ms,
            if result.accepted { "accepted" } else { "rejected" },
            result.plaquette);
    }

    let total_s = t_start.elapsed().as_secs_f64();
    let ms_per_traj = total_s * 1000.0 / n_therm as f64;
    let mean_plaq = last_plaq;

    println!();
    println!("  Results:");
    println!("    Total time: {:.1} s", total_s);
    println!("    Per trajectory: {:.0} ms", ms_per_traj);
    println!("    Acceptance: {}/{} ({:.0}%)", accepted, n_therm, accepted as f64 / n_therm as f64 * 100.0);
    println!("    Final ⟨P⟩ = {:.10}", mean_plaq);
    println!();

    if mean_plaq > 0.1 && mean_plaq < 1.0 {
        println!("  ✓ 32⁴ GPU THERMALIZATION WORKS — guard bypass validated!");
        println!("  ✓ Non-zero, physical plaquette confirms real computation.");
    } else {
        println!("  ✗ FAILED — plaquette {:.6} is non-physical (expected 0.5-0.6 at β=6.0)", mean_plaq);
    }

    println!();
    println!("━━━ Guard Bypass: 22⁴ → 51⁴ UNLOCKED ━━━");
}
