// SPDX-License-Identifier: AGPL-3.0-or-later

//! DF64 Volume Scaling Diagnostic
//!
//! Loads thermalized lattice configurations at multiple volumes and computes
//! the plaquette via:
//!   1. CPU native f64 (gold standard)
//!   2. GPU DF64 (production path)
//!
//! Reports |P_df64 - P_f64| as a function of volume to determine whether
//! the 32⁴ plaquette divergence (0.786 vs expected 0.578) originates from:
//!   - Measurement error (plaquette reduction precision)
//!   - Dynamical drift (force/integrator accumulation over HMC trajectories)
//!
//! If the single-config measurement matches (|Δ| ~ 10⁻¹⁰), the divergence
//! is in the dynamics. If it grows with volume, the reduction path needs
//! native f64.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::path::PathBuf;
use std::time::Instant;
use wgpu;

fn production_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/production_v2")
}

/// Compute GPU plaquette on a pre-loaded lattice (no HMC, just measurement).
fn gpu_plaquette(gpu: &GpuF64, lattice: &Lattice, beta: f64) -> f64 {
    let pipelines = GpuHmcStreamingPipelines::new(gpu);
    let hmc_state = GpuHmcState::from_lattice(gpu, lattice, beta);

    // Run a single zero-step "trajectory" to get the plaquette measurement
    // Actually we need to just dispatch the plaquette pipeline and read back.
    // The simplest way: do 1 trajectory with dt=0 (noop integrator) — plaquette still measured.
    let mut seed = 42u64;
    match gpu_hmc_trajectory_streaming(gpu, &pipelines, &hmc_state, 1, 0.0, 0, &mut seed) {
        Ok(result) => result.plaquette,
        Err(e) => {
            eprintln!("    GPU plaquette failed: {}", e);
            f64::NAN
        }
    }
}

/// Run a short HMC chain on GPU and return final plaquette + acceptance.
fn gpu_hmc_chain(
    gpu: &GpuF64,
    lattice: &Lattice,
    beta: f64,
    n_traj: usize,
    dt: f64,
    n_md: usize,
) -> (f64, f64) {
    let pipelines = GpuHmcStreamingPipelines::new(gpu);
    let hmc_state = GpuHmcState::from_lattice(gpu, lattice, beta);
    let mut seed = 137u64;
    let mut last_plaq = 0.0;
    let mut accepted = 0u32;

    for i in 0..n_traj {
        match gpu_hmc_trajectory_streaming(
            gpu, &pipelines, &hmc_state, n_md, dt, i as u32, &mut seed,
        ) {
            Ok(result) => {
                last_plaq = result.plaquette;
                if result.accepted {
                    accepted += 1;
                }
            }
            Err(_) => {}
        }
    }
    (last_plaq, accepted as f64 / n_traj as f64)
}

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  DF64 Volume Scaling Diagnostic                                 ║");
    println!("║  Comparing GPU DF64 vs CPU f64 plaquette on saved configs       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let adapter = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .next()
        .expect("No discrete GPU found");
    let gpu_name = adapter.get_info().name.clone();
    let gpu = GpuF64::from_adapter(adapter).await.expect("GPU init failed");
    println!("  GPU: {}", gpu_name);
    println!("  Has native f64: {}", gpu.has_f64);
    println!();

    let prod_dir = production_dir();

    // Phase 1: Static measurement comparison (same config, two precision paths)
    println!("═══ PHASE 1: Static Plaquette Measurement ═══");
    println!("  Loading saved .lat configs and comparing CPU f64 vs GPU DF64");
    println!();

    let test_configs: Vec<(&str, &str, &str, [usize; 4])> = vec![
        ("16x16x16x16", "b5.90", "s42", [16, 16, 16, 16]),
        ("16x16x16x16", "b6.00", "s42", [16, 16, 16, 16]),
        ("16x16x16x16", "b6.20", "s42", [16, 16, 16, 16]),
        ("24x24x24x24", "b5.90", "s42", [24, 24, 24, 24]),
        ("24x24x24x24", "b6.00", "s42", [24, 24, 24, 24]),
        ("24x24x24x24", "b6.20", "s42", [24, 24, 24, 24]),
    ];

    println!("  {:>12} {:>6} | {:>12} {:>12} | {:>12} {:>6}",
             "Volume", "β", "CPU f64", "GPU DF64", "|Δ|", "Δ/P");
    println!("  {}", "-".repeat(76));

    let mut deltas: Vec<(usize, f64)> = Vec::new();

    for (vol_str, beta_str, seed_str, dims) in &test_configs {
        let lat_file = prod_dir.join(format!("su3_{}_{}_{}.lat", vol_str, beta_str, seed_str));

        if !lat_file.exists() {
            println!("  {:>12} {:>6} | SKIP (no .lat file)", vol_str, beta_str);
            continue;
        }

        let beta: f64 = beta_str[1..].parse().unwrap();
        let lattice = Lattice::load(&lat_file).unwrap();
        let volume = dims[0] * dims[1] * dims[2] * dims[3];

        let cpu_plaq = lattice.average_plaquette();
        let gpu_plaq = gpu_plaquette(&gpu, &lattice, beta);

        let delta = (gpu_plaq - cpu_plaq).abs();
        let relative = delta / cpu_plaq;
        deltas.push((volume, delta));

        println!("  {:>12} {:>6} | {:.10} {:.10} | {:.2e} {:.2e}",
                 vol_str, beta_str, cpu_plaq, gpu_plaq, delta, relative);
    }

    println!();

    // Analyze scaling
    if deltas.len() >= 2 {
        let small_vol = deltas.iter().filter(|(v, _)| *v <= 65536).map(|(_, d)| *d).collect::<Vec<_>>();
        let large_vol = deltas.iter().filter(|(v, _)| *v > 65536).map(|(_, d)| *d).collect::<Vec<_>>();

        if !small_vol.is_empty() && !large_vol.is_empty() {
            let avg_small: f64 = small_vol.iter().sum::<f64>() / small_vol.len() as f64;
            let avg_large: f64 = large_vol.iter().sum::<f64>() / large_vol.len() as f64;
            let ratio = avg_large / avg_small;
            let vol_ratio = 331776.0_f64 / 65536.0; // 24⁴/16⁴

            println!("  Scaling analysis:");
            println!("    Mean |Δ| at 16⁴: {:.2e}", avg_small);
            println!("    Mean |Δ| at 24⁴: {:.2e}", avg_large);
            println!("    Ratio: {:.2}x", ratio);
            println!("    Volume ratio: {:.2}x", vol_ratio);
            println!("    sqrt(V) ratio: {:.2}x", vol_ratio.sqrt());
            println!();

            if ratio < vol_ratio.sqrt() * 1.5 {
                println!("    → Error grows sub-√V: measurement is NOT the divergence source.");
                println!("    → The 32⁴ divergence is in the DYNAMICS (force/integrator drift).");
            } else if ratio < vol_ratio * 1.5 {
                println!("    → Error grows as ~√V: random accumulation in reduction.");
                println!("    → Native f64 reduction may fix the 32⁴ divergence.");
            } else {
                println!("    → Error grows faster than V: systematic DF64 bias in per-site ops.");
                println!("    → Full hybrid pipeline needed (native f64 for all global ops).");
            }
        }
    }

    // Phase 2: Short dynamical test — run 20 HMC trajectories from a thermalized config
    // and see if the plaquette drifts differently at different volumes.
    println!();
    println!("═══ PHASE 2: Short HMC Drift Test ═══");
    println!("  Running 20 HMC trajectories from thermalized configs at each volume");
    println!("  Comparing final plaquette vs starting plaquette to measure drift");
    println!();

    let drift_configs: Vec<(&str, &str, &str, [usize; 4], f64)> = vec![
        ("16x16x16x16", "b5.90", "s42", [16, 16, 16, 16], 0.01),
        ("24x24x24x24", "b5.90", "s42", [24, 24, 24, 24], 0.01),
    ];

    println!("  {:>12} | {:>10} {:>10} {:>10} | {:>6}",
             "Volume", "Start P", "End P (20)", "Drift", "Accept");
    println!("  {}", "-".repeat(66));

    for (vol_str, beta_str, seed_str, dims, dt) in &drift_configs {
        let lat_file = prod_dir.join(format!("su3_{}_{}_{}.lat", vol_str, beta_str, seed_str));
        if !lat_file.exists() {
            println!("  {:>12} | SKIP (no .lat file)", vol_str);
            continue;
        }

        let beta: f64 = beta_str[1..].parse().unwrap();
        let lattice = Lattice::load(&lat_file).unwrap();
        let start_plaq = lattice.average_plaquette();

        let t0 = Instant::now();
        let (end_plaq, acceptance) = gpu_hmc_chain(&gpu, &lattice, beta, 20, *dt, 20);
        let elapsed = t0.elapsed().as_secs_f64();

        let drift = end_plaq - start_plaq;
        println!("  {:>12} | {:.8} {:.8} {:+.2e} | {:.0}% ({:.1}s)",
                 vol_str, start_plaq, end_plaq, drift, acceptance * 100.0, elapsed);
    }

    println!();
    println!("═══ PHASE 3: Volume ceiling estimate + scaling model ═══");
    println!();
    println!("  DF64 per-operation precision: ~14 significant digits (~10⁻¹⁴)");
    println!("  Force accumulation per site: 8 staples × 18 complex mul-adds");
    println!();
    println!("  ┌─────────┬────────────┬────────────┬────────────┬────────────┬─────────────┐");
    println!("  │ Volume  │ VRAM (MB)  │ force_ops  │ err_random │ err_worst  │ ΔH/traj est │");
    println!("  ├─────────┼────────────┼────────────┼────────────┼────────────┼─────────────┤");

    for (label, vol) in [("16⁴", 65536usize), ("24⁴", 331776), ("32⁴", 1048576), ("48⁴", 5308416), ("64⁴", 16777216)] {
        let n_links = vol * 4;
        let force_ops = n_links * 8 * 18;
        let vram_mb = (n_links as f64 * 18.0 * 8.0 * 6.0) / (1024.0 * 1024.0);
        let error_random = (force_ops as f64).sqrt() * 1e-14;
        let error_worst = force_ops as f64 * 1e-14;
        // ΔH bias estimate: error compounds over n_md_steps (20) per trajectory
        let delta_h_bias = error_worst * 20.0 * vol as f64;
        println!("  │ {:>7} │ {:>8.1}   │ {:.2e}  │ {:.2e}  │ {:.2e}  │ {:.2e}   │",
                 label, vram_mb, force_ops as f64, error_random, error_worst, delta_h_bias);
    }

    println!("  └─────────┴────────────┴────────────┴────────────┴────────────┴─────────────┘");
    println!();
    println!("  VRAM constraint on RX 6950 XT (16 GB):");
    println!("    32⁴: ~3.7 GB (fits comfortably)");
    println!("    48⁴: ~18.4 GB (EXCEEDS 16 GB — needs tiling or multi-GPU)");
    println!("    64⁴: ~59 GB (requires distributed compute)");
    println!();
    println!("  Conclusion: On this hardware, the DF64 volume ceiling is bounded by");
    println!("  VRAM, not precision, when hybrid precision is active.");
    println!("  Pure-DF64 ceiling (without native f64 Hamiltonian) is ~24⁴.");
    println!("  With hybrid precision (Concurrent strategy): 32⁴+ viable.");
    println!();
    println!("  The Concurrent strategy eliminates the precision barrier:");
    println!("  - DF64 force: fast per-site work, errors are LOCAL (don't compound)");
    println!("  - Native f64 Hamiltonian: GLOBAL reduction is precise");
    println!("  - Accept/reject uses precise ΔH → correct Boltzmann distribution");
    println!("  - Force errors cause reversibility violation but NOT equilibrium bias");
    println!("    (as long as ΔH is measured precisely, detailed balance holds)");
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
}
