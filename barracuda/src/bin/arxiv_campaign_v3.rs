// SPDX-License-Identifier: AGPL-3.0-or-later

//! v3 Unified Production Campaign — Climate Shift protocol.
//!
//! Single binary for all lattice volumes, all betas, both GPUs.
//! Unified protocol: dt=0.005, n_md=20 (tau=0.1), hot start epsilon=0.2,
//! adaptive warmup with convergence detection, 500 production trajectories.
//!
//! Environment variables:
//!   BARRACUDA_GPU_ADAPTER — "AMD" or "NVIDIA" (required)
//!   CAMPAIGN_VOLUMES — comma-separated subset: "16,24,32" (default: all)
//!   CAMPAIGN_BETAS — comma-separated subset: "5.9,6.0,6.2" (default: all)
//!   CAMPAIGN_MODE — "full" (all seeds) or "xval" (seed 42 only, for cross-validation)
//!
//! Output: ~/.local/share/hotspring/production_v3/

use barracuda::device::WgpuDevice;
use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use hotspring_barracuda::node_atomic::{NodeAtomicQcd, TrajectoryRunner};
use hotspring_barracuda::spring::campaign::{CampaignConfig, CampaignGrid};
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn output_dir() -> PathBuf {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/production_v3");
    std::fs::create_dir_all(&base).ok();
    base
}

fn json_path(config: &CampaignConfig) -> PathBuf {
    output_dir().join(format!(
        "su3_{}x{}x{}x{}_b{:.2}_s{}.json",
        config.dims[0], config.dims[1], config.dims[2], config.dims[3],
        config.beta, config.seed
    ))
}

#[derive(Serialize)]
struct TrajectoryMeasurement {
    traj: usize,
    plaquette: f64,
    accepted: bool,
    delta_h: f64,
}

#[derive(Serialize)]
struct V3RunResult {
    // Protocol
    dims: [u32; 4],
    beta: f64,
    seed: u32,
    dt: f64,
    n_md: usize,
    integrator: String,
    start_type: String,
    epsilon: f64,
    protocol_version: &'static str,

    // Thermalization
    adaptive_warmup: bool,
    warmup_trajectories: usize,
    warmup_converged: bool,
    warmup_plaquette_history: Vec<f64>,

    // Production
    n_production: usize,
    measurements: Vec<TrajectoryMeasurement>,
    production_accept_rate: f64,
    mean_plaquette: f64,
    plaquette_std: f64,
    final_plaquette: f64,

    // Hardware
    gpu_name: String,
    fp64_strategy: String,
    ms_per_trajectory: f64,
    wall_time_s: f64,
}

fn build_grid() -> CampaignGrid {
    let mode = std::env::var("CAMPAIGN_MODE").unwrap_or_else(|_| "full".to_string());
    let out = output_dir();

    let mut grid = if mode == "xval" {
        CampaignGrid::arxiv_v3_xval(out)
    } else {
        CampaignGrid::arxiv_v3(out)
    };

    // Filter by volumes
    if let Ok(vol_str) = std::env::var("CAMPAIGN_VOLUMES") {
        let vols: Vec<u32> = vol_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect();
        if !vols.is_empty() {
            grid.configs.retain(|c| vols.contains(&c.dims[0]));
        }
    }

    // Filter by betas
    if let Ok(beta_str) = std::env::var("CAMPAIGN_BETAS") {
        let betas: Vec<f64> = beta_str
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        if !betas.is_empty() {
            grid.configs.retain(|c| betas.iter().any(|&b| (b - c.beta).abs() < 0.01));
        }
    }

    grid
}

fn create_device() -> Arc<WgpuDevice> {
    Arc::new(
        hotspring_barracuda::block_on::block_on(WgpuDevice::from_env())
            .expect("Failed to create GPU device (set BARRACUDA_GPU_ADAPTER=AMD or =NVIDIA)")
    )
}

fn run_single(config: &CampaignConfig, device: Arc<WgpuDevice>) -> Result<V3RunResult, String> {
    let t0 = Instant::now();

    let hmc_config = GpuHmcConfig {
        nt: config.dims[0],
        nx: config.dims[1],
        ny: config.dims[2],
        nz: config.dims[3],
        beta: config.beta,
        mass: 1.0,
        n_md_steps: config.n_md_steps,
        dt: config.dt,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 0,
    };

    let qcd = NodeAtomicQcd::with_device(device, hmc_config.clone(), config.seed as u64)
        .map_err(|e| format!("init: {e:?}"))?;

    let gpu_name = qcd.device.adapter_info().name.clone();

    qcd.upload_topology();
    qcd.seed_rng(config.seed);

    // Hot start with near-identity perturbation
    qcd.hot_start(config.epsilon).map_err(|e| format!("hot_start: {e:?}"))?;

    let volume = qcd.volume();
    let beta = config.beta;

    // Adaptive warmup
    let runner = TrajectoryRunner {
        warmup_count: config.n_warmup,
        production_count: config.n_production,
        ..Default::default()
    };

    let warmup_result = runner.run_adaptive_warmup(
        &qcd,
        config.n_warmup,
        50,
        |step, plaq, acc, conv| {
            let status = if conv { "CONVERGED" } else { "warming" };
            println!("    warmup {:4}/{}: P={:.8}, acc={:.0}%, [{}]",
                     step, config.n_warmup, plaq, acc * 100.0, status);
        },
    ).map_err(|e| format!("adaptive_warmup: {e:?}"))?;

    println!("    Warmup done: {} trajectories, converged={}",
             warmup_result.trajectories_to_thermalize, warmup_result.converged);

    // Production
    let mut measurements = Vec::with_capacity(config.n_production);
    let mut prod_accepted = 0usize;

    for t in 0..config.n_production {
        let result = qcd.run_trajectory().map_err(|e| format!("prod {t}: {e:?}"))?;
        if result.accepted {
            prod_accepted += 1;
        }

        let plaq = 1.0 - result.gauge_action / (6.0 * volume as f64 * beta);
        measurements.push(TrajectoryMeasurement {
            traj: t + 1,
            plaquette: plaq,
            accepted: result.accepted,
            delta_h: result.delta_h,
        });

        if (t + 1) % 100 == 0 {
            let acc = prod_accepted as f64 / (t + 1) as f64;
            println!("    production {}/{}: P = {:.8}, accept = {:.0}%",
                     t + 1, config.n_production, plaq, acc * 100.0);
        }
    }

    let production_accept_rate = prod_accepted as f64 / config.n_production as f64;
    let plaquettes: Vec<f64> = measurements.iter().map(|m| m.plaquette).collect();
    let mean_plaq = plaquettes.iter().sum::<f64>() / plaquettes.len() as f64;
    let variance = plaquettes.iter()
        .map(|&p| (p - mean_plaq).powi(2))
        .sum::<f64>() / (plaquettes.len() - 1) as f64;
    let std_plaq = variance.sqrt() / (plaquettes.len() as f64).sqrt();

    let wall_time = t0.elapsed().as_secs_f64();

    Ok(V3RunResult {
        dims: config.dims,
        beta: config.beta,
        seed: config.seed,
        dt: config.dt,
        n_md: config.n_md_steps,
        integrator: "Omelyan2MN".to_string(),
        start_type: "hot".to_string(),
        epsilon: config.epsilon,
        protocol_version: "v3-climate-shift",

        adaptive_warmup: config.adaptive_warmup,
        warmup_trajectories: warmup_result.trajectories_to_thermalize,
        warmup_converged: warmup_result.converged,
        warmup_plaquette_history: warmup_result.plaquette_history,

        n_production: config.n_production,
        measurements,
        production_accept_rate,
        mean_plaquette: mean_plaq,
        plaquette_std: std_plaq,
        final_plaquette: *plaquettes.last().unwrap_or(&0.0),

        gpu_name,
        fp64_strategy: warmup_result.fp64_strategy,
        ms_per_trajectory: warmup_result.ms_per_trajectory,
        wall_time_s: wall_time,
    })
}

fn main() {
    let mode = std::env::var("CAMPAIGN_MODE").unwrap_or_else(|_| "full".to_string());

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  v3 Climate Shift Campaign — Unified Protocol                   ║");
    println!("║  Protocol: hot ε=0.2, adaptive warmup, 500 production           ║");
    println!("║  HMC: Omelyan 2MN, n_md=20, dt=0.005, τ=0.1                   ║");
    println!("║  Mode: {:<57}║", mode);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let grid = build_grid();
    let remaining: Vec<&CampaignConfig> = grid.configs.iter()
        .filter(|c| !json_path(c).exists())
        .collect();
    let n_done = grid.configs.len() - remaining.len();
    let n_total = grid.configs.len();

    println!("  Grid: {n_total} total, {n_done} already complete, {} remaining",
             remaining.len());
    println!("  Output: {:?}", output_dir());
    println!();

    let device = create_device();
    let gpu_name = device.adapter_info().name.clone();
    println!("  GPU: {gpu_name}");
    println!();

    let campaign_start = Instant::now();
    let mut completed_this_run = 0;

    for (idx, config) in remaining.iter().enumerate() {
        let vol = config.dims[0];
        println!("━━━ [{}/{}] {}⁴ β={:.2} seed={} ━━━",
                 n_done + completed_this_run + 1, n_total,
                 vol, config.beta, config.seed);

        match run_single(config, device.clone()) {
            Ok(result) => {
                let json = serde_json::to_string_pretty(&result).unwrap();
                std::fs::write(json_path(config), &json).unwrap();
                println!("  ✓ Complete: ⟨P⟩ = {:.8} ± {:.2e}, accept = {:.1}%, {:.1}s",
                         result.mean_plaquette,
                         result.plaquette_std,
                         result.production_accept_rate * 100.0,
                         result.wall_time_s);
                println!("    Warmup: {} trajs (converged={})",
                         result.warmup_trajectories, result.warmup_converged);
            }
            Err(e) => {
                eprintln!("  ✗ FAILED: {}⁴ β={:.2} seed={} — {e}",
                          vol, config.beta, config.seed);
            }
        }
        println!();
        completed_this_run += 1;

        let elapsed = campaign_start.elapsed().as_secs_f64();
        let rate = elapsed / completed_this_run as f64;
        let remaining_count = remaining.len() - idx - 1;
        let eta = remaining_count as f64 * rate;
        println!("  Progress: {}/{} | Rate: {:.1}s/config | ETA: {:.1} hours",
                 n_done + completed_this_run, n_total, rate, eta / 3600.0);
        println!();
    }

    let total_time = campaign_start.elapsed().as_secs_f64();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  v3 Campaign Complete");
    println!("  Configs produced: {completed_this_run}");
    println!("  Total wall time: {:.1}s ({:.1} hours)", total_time, total_time / 3600.0);
    println!("  GPU: {gpu_name}");
    println!("═══════════════════════════════════════════════════════════════════");
}
