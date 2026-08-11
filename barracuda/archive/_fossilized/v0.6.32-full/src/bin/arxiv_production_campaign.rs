// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rung 1 Production Campaign — single binary, single protocol, clean data.
//!
//! This replaces all previous production data generation with a unified campaign:
//!   - Cold start (U=I) for unambiguous thermalization
//!   - 500 warmup trajectories (discarded)
//!   - 200 production trajectories with measurements
//!   - Fixed HMC: Omelyan 2MN, dt=0.01, n_md=20, tau=0.2
//!   - 5 seeds per (beta, volume) point
//!   - Grid: 16⁴, 24⁴, 32⁴ × β=5.9, 6.0, 6.2
//!
//! Output: ~/.local/share/hotspring/production_v2/
//!   One JSON file per run with full time series + final .lat config.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    GpuHmcState, GpuHmcStreamingPipelines, gpu_hmc_trajectory_streaming,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::lattice::complex_f64::Complex64;
use hotspring_barracuda::lattice::su3::Su3Matrix;
use std::path::PathBuf;
use std::time::Instant;

const N_WARMUP: usize = 500;
const N_PRODUCTION: usize = 200;
const DT: f64 = 0.01;
const N_MD: usize = 20;
const WILSON_EVERY: usize = 10;

#[derive(Clone)]
struct RunSpec {
    dims: [usize; 4],
    beta: f64,
    seed: u64,
}

impl RunSpec {
    fn label(&self) -> String {
        format!("{}⁴ β={:.2} seed={}", self.dims[0], self.beta, self.seed)
    }

    fn volume(&self) -> usize {
        self.dims.iter().product()
    }

    fn output_dir() -> PathBuf {
        let base = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("hotspring/production_v2");
        std::fs::create_dir_all(&base).ok();
        base
    }

    fn json_path(&self) -> PathBuf {
        let fname = format!(
            "su3_{}x{}x{}x{}_b{:.2}_s{}.json",
            self.dims[0], self.dims[1], self.dims[2], self.dims[3],
            self.beta, self.seed
        );
        Self::output_dir().join(fname)
    }

    fn lat_path(&self) -> PathBuf {
        let fname = format!(
            "su3_{}x{}x{}x{}_b{:.2}_s{}.lat",
            self.dims[0], self.dims[1], self.dims[2], self.dims[3],
            self.beta, self.seed
        );
        Self::output_dir().join(fname)
    }
}

fn build_grid() -> Vec<RunSpec> {
    let betas = [5.9, 6.0, 6.2];
    let volumes: &[usize] = &[16, 24, 32];
    let seeds: &[u64] = &[42, 137, 271, 503, 719];
    let mut specs = Vec::new();

    for &l in volumes {
        for &beta in &betas {
            for &seed in seeds {
                specs.push(RunSpec {
                    dims: [l, l, l, l],
                    beta,
                    seed,
                });
            }
        }
    }
    specs
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

#[tokio::main]
async fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Rung 1 Production Campaign — Clean Data Generation            ║");
    println!("║  Protocol: cold start, 500 warmup, 200 production              ║");
    println!("║  HMC: Omelyan 2MN, dt=0.01, n_md=20, tau=0.2                  ║");
    println!("║  Grid: 3 volumes × 3 β × 5 seeds = 45 configs                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(wgpu::Backends::all()).await;

    let gpu_hint = std::env::var("CAMPAIGN_GPU").unwrap_or_else(|_| "AMD".to_string());
    let adapter = adapters
        .into_iter()
        .filter(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .find(|a| a.get_info().name.to_uppercase().contains(&gpu_hint.to_uppercase()))
        .expect("No matching GPU found (set CAMPAIGN_GPU=AMD or CAMPAIGN_GPU=NVIDIA)");

    let gpu_name = adapter.get_info().name.clone();
    println!("  GPU: {}", gpu_name);

    let gpu: GpuF64 = GpuF64::from_adapter(adapter)
        .await
        .expect("Failed to initialize GPU");
    println!("  GPU ready");
    println!();

    let grid = build_grid();
    let already_done: Vec<bool> = grid.iter().map(|s| s.json_path().exists()).collect();
    let n_done = already_done.iter().filter(|&&d| d).count();
    let n_total = grid.len();

    println!("  Grid: {} total, {} already complete, {} remaining",
             n_total, n_done, n_total - n_done);
    println!("  Output: {:?}", RunSpec::output_dir());
    println!();

    let campaign_start = Instant::now();
    let mut completed_this_run = 0;

    for (idx, spec) in grid.iter().enumerate() {
        if already_done[idx] {
            continue;
        }

        println!("━━━ [{}/{}] {} ━━━",
                 n_done + completed_this_run + 1, n_total, spec.label());

        let result = run_single_config(&gpu, spec, &gpu_name).await;

        match result {
            Ok(run_result) => {
                let json = serde_json::to_string_pretty(&run_result).unwrap();
                std::fs::write(spec.json_path(), &json).unwrap();
                println!("  ✓ Complete: ⟨P⟩ = {:.8} ± {:.2e}, accept = {:.1}%, {:.1}s",
                         run_result.mean_plaquette,
                         run_result.plaquette_std,
                         run_result.production_accept_rate * 100.0,
                         run_result.wall_time_s);
            }
            Err(e) => {
                eprintln!("  ✗ FAILED: {} — {}", spec.label(), e);
            }
        }
        println!();
        completed_this_run += 1;

        let elapsed = campaign_start.elapsed().as_secs_f64();
        let rate = elapsed / completed_this_run as f64;
        let remaining = (n_total - n_done - completed_this_run) as f64 * rate;
        println!("  Progress: {}/{} | ETA: {:.0} min",
                 n_done + completed_this_run, n_total, remaining / 60.0);
        println!();
    }

    let total_time = campaign_start.elapsed().as_secs_f64();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Production Campaign Complete");
    println!("  GPU: {}", gpu_name);
    println!("  Configs produced: {}", completed_this_run);
    println!("  Total wall time: {:.1}s ({:.1} min)", total_time, total_time / 60.0);
    println!("═══════════════════════════════════════════════════════════════════");
}

async fn run_single_config(
    gpu: &GpuF64,
    spec: &RunSpec,
    gpu_name: &str,
) -> Result<RunResult, String> {
    let run_start = Instant::now();

    // Cold start — unambiguous thermalization baseline
    let lat = Lattice::cold_start(spec.dims, spec.beta);
    let pipelines = GpuHmcStreamingPipelines::new(gpu);
    let hmc_state = GpuHmcState::from_lattice(gpu, &lat, spec.beta);

    let mut rng_seed = spec.seed;

    // Phase 1: Staged warmup (500 trajectories, discarded)
    // Cold start thermalization requires smaller dt for larger volumes because
    // ΔH ∝ V from the ordered initial state. We ramp dt from small to production value.
    let warmup_schedule = warmup_dt_schedule(spec.volume());
    let mut warmup_accepted = 0u32;
    let mut traj_count = 0usize;

    for &(stage_dt, stage_n) in &warmup_schedule {
        for _s in 0..stage_n {
            match gpu_hmc_trajectory_streaming(
                gpu, &pipelines, &hmc_state, N_MD, stage_dt, traj_count as u32, &mut rng_seed,
            ) {
                Ok(result) => {
                    if result.accepted {
                        warmup_accepted += 1;
                    }
                    traj_count += 1;
                    if traj_count % 100 == 0 {
                        let acc = warmup_accepted as f64 / traj_count as f64 * 100.0;
                        let warmup_total: usize = warmup_schedule.iter().map(|&(_, n)| n).sum();
                        println!("    warmup {}/{}: ⟨P⟩ = {:.8}, accept = {:.0}%, dt={:.4}",
                                 traj_count, warmup_total, result.plaquette, acc, stage_dt);
                    }
                }
                Err(e) => return Err(format!("warmup step {}: {:?}", traj_count + 1, e)),
            }
        }
    }
    let warmup_accept_rate = warmup_accepted as f64 / traj_count as f64;
    println!("    Warmup done: accept = {:.1}%", warmup_accept_rate * 100.0);

    // Phase 2: Production (200 trajectories, measured)
    let prod_dt = production_dt(spec.volume());
    let mut measurements = Vec::with_capacity(N_PRODUCTION);
    let mut prod_accepted = 0u32;

    for t in 0..N_PRODUCTION {
        let traj_id = (N_WARMUP + t) as u32;
        match gpu_hmc_trajectory_streaming(
            gpu, &pipelines, &hmc_state, N_MD, prod_dt, traj_id, &mut rng_seed,
        ) {
            Ok(result) => {
                if result.accepted {
                    prod_accepted += 1;
                }

                // Read back lattice for CPU observables on Wilson loop trajectories
                let wilson_loops = if (t + 1) % WILSON_EVERY == 0 {
                    match readback_lattice(gpu, &hmc_state, spec) {
                        Ok(cpu_lat) => {
                            let mut wl = Vec::new();
                            for r in 1..=4 {
                                for tt in 1..=4 {
                                    if r == 1 && tt == 1 { continue; }
                                    let w = cpu_lat.spatial_temporal_wilson_loop(r, tt);
                                    wl.push([r as f64, tt as f64, w]);
                                }
                            }
                            let (poly_re, poly_im) = cpu_lat.complex_polyakov_average();
                            Some((wl, poly_re, poly_im))
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };

                let (poly_re, poly_im) = if let Some((_, pr, pi)) = &wilson_loops {
                    (*pr, *pi)
                } else {
                    (0.0, 0.0)
                };

                measurements.push(TrajectoryMeasurement {
                    traj: t + 1,
                    plaquette: result.plaquette,
                    accepted: result.accepted,
                    delta_h: result.delta_h,
                    polyakov_re: poly_re,
                    polyakov_im: poly_im,
                    wilson_loops: wilson_loops.map(|(wl, _, _)| wl),
                });

                if (t + 1) % 50 == 0 {
                    let acc = prod_accepted as f64 / (t + 1) as f64 * 100.0;
                    println!("    production {}/{}: ⟨P⟩ = {:.8}, accept = {:.0}%",
                             t + 1, N_PRODUCTION, result.plaquette, acc);
                }
            }
            Err(e) => return Err(format!("production step {}: {:?}", t + 1, e)),
        }
    }

    let production_accept_rate = prod_accepted as f64 / N_PRODUCTION as f64;

    // Compute statistics
    let plaquettes: Vec<f64> = measurements.iter().map(|m| m.plaquette).collect();
    let mean_plaq = plaquettes.iter().sum::<f64>() / plaquettes.len() as f64;
    let variance = plaquettes.iter()
        .map(|&p| (p - mean_plaq).powi(2))
        .sum::<f64>() / (plaquettes.len() - 1) as f64;
    let std_plaq = variance.sqrt() / (plaquettes.len() as f64).sqrt();

    // Save final config
    let config_hash = match readback_lattice(gpu, &hmc_state, spec) {
        Ok(final_lat) => {
            match final_lat.save(&spec.lat_path()) {
                Ok(h) => h.to_hex()[..16].to_string(),
                Err(_) => "save_failed".to_string(),
            }
        }
        Err(_) => "readback_failed".to_string(),
    };

    let wall_time = run_start.elapsed().as_secs_f64();

    Ok(RunResult {
        dims: spec.dims,
        beta: spec.beta,
        seed: spec.seed,
        n_warmup: warmup_dt_schedule(spec.volume()).iter().map(|&(_, n)| n).sum(),
        n_production: N_PRODUCTION,
        dt: production_dt(spec.volume()),
        n_md: N_MD,
        integrator: "Omelyan2MN".to_string(),
        start_type: "cold".to_string(),
        gpu_name: gpu_name.to_string(),
        wall_time_s: wall_time,
        warmup_accept_rate,
        production_accept_rate,
        measurements,
        final_plaquette: *plaquettes.last().unwrap_or(&0.0),
        mean_plaquette: mean_plaq,
        plaquette_std: std_plaq,
        config_blake3: config_hash,
    })
}

/// Generate a staged dt schedule for cold-start thermalization.
/// Larger volumes need smaller initial dt because ΔH ∝ V from the ordered state.
/// Returns (dt, n_trajectories) pairs that sum to N_WARMUP.
fn warmup_dt_schedule(volume: usize) -> Vec<(f64, usize)> {
    if volume <= 65536 {
        // 16⁴: single stage at production dt works fine
        vec![(DT, N_WARMUP)]
    } else if volume <= 400_000 {
        // 24⁴ (331776): ramp from dt/5 → dt/2 → dt
        vec![
            (DT / 5.0, 100),  // dt=0.002, gentle initial thermalization
            (DT / 2.0, 150),  // dt=0.005, intermediate
            (DT, 250),        // dt=0.01, production step size
        ]
    } else {
        // 32⁴ (1048576): critical slowing down requires extended thermalization.
        // Verified (Aug 2026): DF64 force is equivalent to native f64 (identical dynamics).
        // The slow convergence from 0.82→0.578 is autocorrelation, not precision.
        // Scaling: τ_auto ~ L² → from 500 at 16⁴, expect ~2000-8000 at 32⁴.
        // Using 5000 total warmup to ensure full equilibration.
        vec![
            (DT / 10.0, 50),   // dt=0.001, break cold start symmetry
            (DT / 5.0, 100),   // dt=0.002, gentle ramp
            (DT / 3.0, 200),   // dt=0.0033, bridge
            (DT / 2.0, 4650),  // dt=0.005, full thermalization (autocorrelation ~4000τ)
        ]
    }
}

/// Volume-adaptive production step size.
/// 
/// dt=0.01 gives 88% acceptance at 16⁴ but only 5-8% at 32⁴.
/// Scale dt to maintain ~60-80% acceptance across volumes:
///   16⁴: dt=0.01 (88% acceptance)
///   24⁴: dt=0.01 (64% acceptance)  
///   32⁴: dt=0.005 (target ~60% acceptance)
fn production_dt(volume: usize) -> f64 {
    if volume <= 400_000 {
        DT
    } else {
        DT / 2.0
    }
}

fn readback_lattice(
    gpu: &GpuF64,
    state: &GpuHmcState,
    spec: &RunSpec,
) -> Result<Lattice, String> {
    let n_links = spec.volume() * 4;
    let link_bytes = (n_links * 18 * 8) as u64;
    let staging = gpu.create_staging_buffer(link_bytes as usize, "readback");
    {
        let mut enc = gpu.begin_encoder("readback_copy");
        enc.copy_buffer_to_buffer(&state.link_buf, 0, &staging, 0, link_bytes);
        gpu.submit_encoder(enc);
    }
    let rx = gpu.start_async_readback(&staging);
    match gpu.finish_async_readback_f64(&staging, rx) {
        Ok(ref flat_links) => Ok(unflatten_to_lattice(flat_links, spec.dims, spec.beta)),
        Err(e) => Err(format!("readback failed: {:?}", e)),
    }
}

fn unflatten_to_lattice(flat: &[f64], dims: [usize; 4], beta: f64) -> Lattice {
    let vol: usize = dims.iter().product();
    let n_links = vol * 4;
    assert_eq!(flat.len(), n_links * 18, "flat link buffer size mismatch");

    let mut links = Vec::with_capacity(n_links);
    for i in 0..n_links {
        let base = i * 18;
        let mut m = [[Complex64 { re: 0.0, im: 0.0 }; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                m[row][col] = Complex64 {
                    re: flat[base + row * 6 + col * 2],
                    im: flat[base + row * 6 + col * 2 + 1],
                };
            }
        }
        links.push(Su3Matrix { m });
    }

    Lattice { dims, links, beta }
}
