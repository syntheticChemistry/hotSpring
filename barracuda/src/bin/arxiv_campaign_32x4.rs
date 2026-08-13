// SPDX-License-Identifier: AGPL-3.0-or-later

//! 32⁴ Rung 1 Production Campaign — Node-Atomic path.
//!
//! The legacy `gpu_hmc_trajectory_streaming` has a GPU PRNG failure at 32⁴
//! (momenta not generated → lattice stuck at identity). This binary uses the
//! validated Node-Atomic path (upstream barraCuda HMC) with hot start.
//!
//! Grid: β = {5.9, 6.0, 6.2} × 5 seeds = 15 configurations.
//! Protocol: hot start ε=3.0, 500 warmup, 200 production.
//! HMC: Omelyan 2MN, n_md=40, dt=0.0025, τ=0.1.
//!
//! Output: ~/.local/share/hotspring/production_v2/ (same format as campaign binary)

use barracuda::device::WgpuDevice;
use barracuda::ops::lattice::gpu_hmc_types::GpuHmcConfig;
use hotspring_barracuda::node_atomic::NodeAtomicQcd;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

const N_WARMUP: usize = 500;
const N_PRODUCTION: usize = 200;
const HOT_EPSILON: f64 = 3.0;

#[derive(Clone)]
struct RunSpec {
    beta: f64,
    seed: u64,
}

fn output_dir() -> PathBuf {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hotspring/production_v2");
    std::fs::create_dir_all(&base).ok();
    base
}

fn json_path(spec: &RunSpec) -> PathBuf {
    output_dir().join(format!("su3_32x32x32x32_b{:.2}_s{}.json", spec.beta, spec.seed))
}

#[derive(serde::Serialize)]
struct TrajectoryMeasurement {
    traj: usize,
    plaquette: f64,
    accepted: bool,
    delta_h: f64,
    polyakov_re: f64,
    polyakov_im: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    wilson_loops: Option<Vec<[f64; 3]>>,
}

#[derive(serde::Serialize)]
struct RunResult {
    dims: [usize; 4],
    beta: f64,
    seed: u64,
    n_warmup: usize,
    n_production: usize,
    dt: f64,
    n_md: usize,
    integrator: String,
    start_type: String,
    gpu_name: String,
    wall_time_s: f64,
    warmup_accept_rate: f64,
    production_accept_rate: f64,
    measurements: Vec<TrajectoryMeasurement>,
    final_plaquette: f64,
    mean_plaquette: f64,
    plaquette_std: f64,
    config_blake3: String,
}

fn build_grid() -> Vec<RunSpec> {
    let betas_filter: Option<Vec<f64>> = std::env::var("CAMPAIGN_BETAS").ok().map(|s| {
        s.split(',').filter_map(|b| b.trim().parse::<f64>().ok()).collect()
    });
    let all_betas = [5.9, 6.0, 6.2];
    let seeds: &[u64] = &[42, 137, 271, 503, 719];
    let mut specs = Vec::new();
    for &beta in &all_betas {
        if let Some(ref filter) = betas_filter {
            if !filter.iter().any(|&b| (b - beta).abs() < 0.01) {
                continue;
            }
        }
        for &seed in seeds {
            specs.push(RunSpec { beta, seed });
        }
    }
    specs
}

fn create_device() -> Arc<WgpuDevice> {
    Arc::new(
        hotspring_barracuda::block_on::block_on(WgpuDevice::from_env())
            .expect("Failed to create GPU device (set BARRACUDA_GPU_ADAPTER=AMD or =NVIDIA)")
    )
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  32⁴ Production Campaign — Node-Atomic Path                    ║");
    println!("║  Protocol: hot start ε=3.0, 500 warmup, 200 production         ║");
    println!("║  HMC: Omelyan 2MN, n_md=40, dt=0.0025, τ=0.1                  ║");
    println!("║  Grid: 3 β × 5 seeds = 15 configs                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let grid = build_grid();
    let already_done: Vec<bool> = grid.iter().map(|s| json_path(s).exists()).collect();
    let n_done = already_done.iter().filter(|&&d| d).count();
    let n_total = grid.len();

    println!("  Grid: {n_total} total, {n_done} already complete, {} remaining",
             n_total - n_done);
    println!("  Output: {:?}", output_dir());
    println!();

    let device = create_device();
    let gpu_name = device.adapter_info().name.clone();
    println!("  GPU: {gpu_name}");
    println!();

    let campaign_start = Instant::now();
    let mut completed_this_run = 0;

    for (idx, spec) in grid.iter().enumerate() {
        if already_done[idx] {
            println!("  ▸ Skipping β={:.2} seed={} (already done)", spec.beta, spec.seed);
            continue;
        }

        println!("━━━ [{}/{}] 32⁴ β={:.2} seed={} ━━━",
                 n_done + completed_this_run + 1, n_total, spec.beta, spec.seed);

        match run_single(spec, device.clone()) {
            Ok(result) => {
                let json = serde_json::to_string_pretty(&result).unwrap();
                std::fs::write(json_path(spec), &json).unwrap();
                println!("  ✓ Complete: ⟨P⟩ = {:.8} ± {:.2e}, accept = {:.1}%, {:.1}s",
                         result.mean_plaquette,
                         result.plaquette_std,
                         result.production_accept_rate * 100.0,
                         result.wall_time_s);
            }
            Err(e) => {
                eprintln!("  ✗ FAILED: β={:.2} seed={} — {e}", spec.beta, spec.seed);
            }
        }
        println!();
        completed_this_run += 1;

        let elapsed = campaign_start.elapsed().as_secs_f64();
        let rate = elapsed / completed_this_run as f64;
        let remaining = (n_total - n_done - completed_this_run) as f64 * rate;
        println!("  Progress: {}/{} | ETA: {:.1} hours",
                 n_done + completed_this_run, n_total, remaining / 3600.0);
        println!();
    }

    let total_time = campaign_start.elapsed().as_secs_f64();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  32⁴ Campaign Complete");
    println!("  Configs produced: {completed_this_run}");
    println!("  Total wall time: {:.1}s ({:.1} hours)", total_time, total_time / 3600.0);
    println!("═══════════════════════════════════════════════════════════════════");
}

fn run_single(spec: &RunSpec, device: Arc<WgpuDevice>) -> Result<RunResult, String> {
    let t0 = Instant::now();

    let config = GpuHmcConfig {
        nt: 32,
        nx: 32,
        ny: 32,
        nz: 32,
        beta: spec.beta,
        mass: 1.0,
        n_md_steps: 40,
        dt: 0.0025,
        cg_tol: 1e-10,
        cg_max_iter: 1000,
        n_flavors_over_4: 0,
    };

    let qcd = NodeAtomicQcd::with_device(device, config.clone(), spec.seed)
        .map_err(|e| format!("init: {e:?}"))?;

    let gpu_name = qcd.device.adapter_info().name.clone();

    qcd.upload_topology();
    qcd.seed_rng(spec.seed as u32);
    qcd.hot_start(HOT_EPSILON).map_err(|e| format!("hot_start: {e:?}"))?;

    let volume = qcd.volume();
    let beta = spec.beta;

    // Warmup
    let mut warmup_accepted = 0usize;
    for i in 0..N_WARMUP {
        let result = qcd.run_trajectory().map_err(|e| format!("warmup {i}: {e:?}"))?;
        if result.accepted {
            warmup_accepted += 1;
        }
        if (i + 1) % 50 == 0 {
            let plaq = 1.0 - result.gauge_action / (6.0 * volume as f64 * beta);
            let acc = warmup_accepted as f64 / (i + 1) as f64;
            let rate = t0.elapsed().as_secs_f64() / (i + 1) as f64;
            println!("    warmup {:3}/{}: P={:.6}, acc={:.0}%, {:.1}s/traj",
                     i + 1, N_WARMUP, plaq, acc * 100.0, rate);
        }
    }
    let warmup_accept_rate = warmup_accepted as f64 / N_WARMUP as f64;
    println!("    Warmup done: accept = {:.1}%", warmup_accept_rate * 100.0);

    // Production
    let mut measurements = Vec::with_capacity(N_PRODUCTION);
    let mut prod_accepted = 0usize;

    for t in 0..N_PRODUCTION {
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
            polyakov_re: 0.0,
            polyakov_im: 0.0,
            wilson_loops: None,
        });

        if (t + 1) % 50 == 0 {
            let acc = prod_accepted as f64 / (t + 1) as f64;
            println!("    production {}/{}: ⟨P⟩ = {:.8}, accept = {:.0}%",
                     t + 1, N_PRODUCTION, plaq, acc * 100.0);
        }
    }

    let production_accept_rate = prod_accepted as f64 / N_PRODUCTION as f64;

    let plaquettes: Vec<f64> = measurements.iter().map(|m| m.plaquette).collect();
    let mean_plaq = plaquettes.iter().sum::<f64>() / plaquettes.len() as f64;
    let variance = plaquettes.iter()
        .map(|&p| (p - mean_plaq).powi(2))
        .sum::<f64>() / (plaquettes.len() - 1) as f64;
    let std_plaq = variance.sqrt() / (plaquettes.len() as f64).sqrt();

    let wall_time = t0.elapsed().as_secs_f64();

    Ok(RunResult {
        dims: [32, 32, 32, 32],
        beta: spec.beta,
        seed: spec.seed,
        n_warmup: N_WARMUP,
        n_production: N_PRODUCTION,
        dt: 0.0025,
        n_md: 40,
        integrator: "Omelyan2MN".to_string(),
        start_type: "hot".to_string(),
        gpu_name,
        wall_time_s: wall_time,
        warmup_accept_rate,
        production_accept_rate,
        measurements,
        final_plaquette: *plaquettes.last().unwrap_or(&0.0),
        mean_plaquette: mean_plaq,
        plaquette_std: std_plaq,
        config_blake3: "node_atomic".to_string(),
    })
}
