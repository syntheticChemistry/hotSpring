// SPDX-License-Identifier: AGPL-3.0-only

//! Meta-table burst scanner: fast parameter sweeps across (lattice, beta, mass).
//!
//! Runs short scans at each parameter combination and aggregates results into
//! a JSONL meta table. The NPU trains on this table to learn the
//! (beta, lattice, mass) → (plaq, chi, acceptance) mapping.
//!
//! # Usage
//!
//! ```bash
//! # Quenched sweep
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin meta_table_scan -- \
//!   --mode=quenched --lattices=8,16,32 \
//!   --betas=5.0,5.3,5.5,5.6,5.65,5.69,5.72,5.8,6.0,6.5 \
//!   --meas=50 --therm=20 --output=meta_table_quenched.jsonl
//!
//! # Dynamical sweep
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin meta_table_scan -- \
//!   --mode=dynamical --lattices=4,8 \
//!   --betas=5.0,5.5,5.69,6.0 --masses=0.1,0.5,1.0,2.0 \
//!   --meas=50 --therm=20 --output=meta_table_dynamical.jsonl
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_dynamical_hmc_trajectory_resident, gpu_hmc_trajectory_streaming, gpu_links_to_lattice,
    GpuDynHmcState, GpuDynHmcStreamingPipelines, GpuHmcState, GpuHmcStreamingPipelines,
    GpuResidentCgBuffers, GpuResidentCgPipelines,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;

use std::io::Write;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScanMode {
    Quenched,
    Dynamical,
}

struct CliArgs {
    mode: ScanMode,
    lattices: Vec<usize>,
    betas: Vec<f64>,
    masses: Vec<f64>,
    cg_tol: f64,
    cg_max_iter: usize,
    check_interval: usize,
    n_therm: usize,
    n_meas: usize,
    seed: u64,
    output: String,
}

fn parse_args() -> CliArgs {
    let mut mode = ScanMode::Quenched;
    let mut lattices = vec![8, 16, 32];
    let mut betas = vec![5.0, 5.3, 5.5, 5.6, 5.65, 5.69, 5.72, 5.8, 6.0, 6.5];
    let mut masses = vec![0.1];
    let mut cg_tol = 1e-8;
    let mut cg_max_iter = 5000;
    let mut check_interval = 10;
    let mut n_therm = 20;
    let mut n_meas = 50;
    let mut seed = 42u64;
    let mut output = String::from("meta_table.jsonl");

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--mode=") {
            mode = match val {
                "dynamical" | "dyn" => ScanMode::Dynamical,
                _ => ScanMode::Quenched,
            };
        } else if let Some(val) = arg.strip_prefix("--lattices=") {
            lattices = val
                .split(',')
                .map(|s| s.parse().expect("lattice int"))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--betas=") {
            betas = val
                .split(',')
                .map(|s| s.parse().expect("beta float"))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--masses=") {
            masses = val
                .split(',')
                .map(|s| s.parse().expect("mass float"))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--cg-tol=") {
            cg_tol = val.parse().expect("--cg-tol=F");
        } else if let Some(val) = arg.strip_prefix("--cg-max-iter=") {
            cg_max_iter = val.parse().expect("--cg-max-iter=N");
        } else if let Some(val) = arg.strip_prefix("--check-interval=") {
            check_interval = val.parse().expect("--check-interval=N");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            n_meas = val.parse().expect("--meas=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = val.to_string();
        }
    }

    CliArgs {
        mode,
        lattices,
        betas,
        masses,
        cg_tol,
        cg_max_iter,
        check_interval,
        n_therm,
        n_meas,
        seed,
        output,
    }
}

fn main() {
    let args = parse_args();
    let mode_str = match args.mode {
        ScanMode::Quenched => "quenched",
        ScanMode::Dynamical => "dynamical",
    };

    let n_combos = match args.mode {
        ScanMode::Quenched => args.lattices.len() * args.betas.len(),
        ScanMode::Dynamical => args.lattices.len() * args.betas.len() * args.masses.len(),
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Meta-Table Burst Scanner — {mode_str} mode                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattices: {:?}", args.lattices);
    println!("  β values: {:?}", args.betas);
    if args.mode == ScanMode::Dynamical {
        println!("  Masses:   {:?}", args.masses);
    }
    println!("  Therm: {}, Meas: {}", args.n_therm, args.n_meas);
    println!("  Combinations: {n_combos}");
    println!("  Output: {}", args.output);
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out_file = std::io::BufWriter::new(
        std::fs::File::create(&args.output)
            .unwrap_or_else(|e| panic!("Cannot create {}: {e}", args.output)),
    );

    let total_start = Instant::now();
    let mut combo_idx = 0usize;

    for &lattice_size in &args.lattices {
        let dims = [lattice_size, lattice_size, lattice_size, lattice_size];
        let vol: usize = dims.iter().product();
        let vol_f = vol as f64;
        let scale = (4096.0_f64 / vol_f).powf(0.25);
        let dt = (0.05 * scale).max(0.001);
        let n_md = ((0.5 / dt).round() as usize).max(10);

        println!("── Lattice {lattice_size}⁴ (dt={dt:.4}, n_md={n_md}) ──");

        let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
        let dyn_streaming_pipelines;
        let resident_cg_pipelines;

        let (dyn_sp, res_cg) = if args.mode == ScanMode::Dynamical {
            (
                Some(GpuDynHmcStreamingPipelines::new(&gpu)),
                Some(GpuResidentCgPipelines::new(&gpu)),
            )
        } else {
            (None, None)
        };
        dyn_streaming_pipelines = dyn_sp;
        resident_cg_pipelines = res_cg;

        let mass_list: &[f64] = match args.mode {
            ScanMode::Quenched => &[0.0], // placeholder, not used
            ScanMode::Dynamical => &args.masses,
        };

        for &beta in &args.betas {
            for &mass in mass_list {
                combo_idx += 1;
                let point_start = Instant::now();

                let mass_val = if args.mode == ScanMode::Dynamical {
                    Some(mass)
                } else {
                    None
                };

                let (mean_plaq, chi, acceptance, mean_cg, wall_per_traj) = match args.mode {
                    ScanMode::Quenched => run_quenched_point(
                        &gpu,
                        &quenched_pipelines,
                        dims,
                        beta,
                        n_md,
                        dt,
                        args.n_therm,
                        args.n_meas,
                        args.seed + combo_idx as u64,
                    ),
                    ScanMode::Dynamical => run_dynamical_point(
                        &gpu,
                        dyn_streaming_pipelines.as_ref().unwrap(),
                        resident_cg_pipelines.as_ref().unwrap(),
                        &quenched_pipelines,
                        dims,
                        beta,
                        mass,
                        args.cg_tol,
                        args.cg_max_iter,
                        args.check_interval,
                        n_md,
                        dt,
                        args.n_therm,
                        args.n_meas,
                        args.seed + combo_idx as u64,
                    ),
                };

                let wall_s = point_start.elapsed().as_secs_f64();
                let timestamp = chrono_timestamp();

                let row = serde_json::json!({
                    "lattice": lattice_size,
                    "beta": beta,
                    "mass": mass_val,
                    "mode": mode_str,
                    "mean_plaq": mean_plaq,
                    "chi": chi,
                    "acceptance": acceptance,
                    "mean_cg_iters": mean_cg,
                    "wall_s_per_traj": wall_per_traj,
                    "n_meas": args.n_meas,
                    "timestamp": timestamp,
                });
                writeln!(out_file, "{row}").ok();
                out_file.flush().ok();

                let mass_str = mass_val.map(|m| format!(" m={m}")).unwrap_or_default();
                print!(
                    "  [{combo_idx}/{n_combos}] {}⁴ β={beta:.4}{mass_str}: ⟨P⟩={mean_plaq:.6} χ={chi:.2} acc={:.0}%",
                    lattice_size,
                    acceptance * 100.0,
                );
                if args.mode == ScanMode::Dynamical {
                    print!(" ⟨CG⟩={mean_cg:.0}");
                }
                println!(" ({wall_s:.1}s)");
            }
        }
    }

    let total_wall = total_start.elapsed().as_secs_f64();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Meta-table complete: {} rows in {:.1}s ({:.1} min)",
        combo_idx,
        total_wall,
        total_wall / 60.0,
    );
    println!("  Output: {}", args.output);
    println!("  GPU: {}", gpu.adapter_name);
}

fn chrono_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", dur.as_secs())
}

/// Run a single quenched parameter point. Returns (mean_plaq, chi, acceptance, 0.0, wall_per_traj).
#[allow(clippy::too_many_arguments)]
fn run_quenched_point(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    dims: [usize; 4],
    beta: f64,
    n_md: usize,
    dt: f64,
    n_therm: usize,
    n_meas: usize,
    seed: u64,
) -> (f64, f64, f64, f64, f64) {
    let vol: usize = dims.iter().product();
    let mut lat = Lattice::hot_start(dims, beta, seed);

    if vol <= 65536 {
        let mut cfg = HmcConfig {
            n_md_steps: n_md,
            dt,
            seed: seed * 1000,
            integrator: IntegratorType::Omelyan,
        };
        for _ in 0..5 {
            hmc::hmc_trajectory(&mut lat, &mut cfg);
        }
    }

    let state = GpuHmcState::from_lattice(gpu, &lat, beta);
    let mut s = seed * 100;

    for i in 0..n_therm {
        gpu_hmc_trajectory_streaming(gpu, pipelines, &state, n_md, dt, i as u32, &mut s);
    }

    let mut plaq_vals = Vec::with_capacity(n_meas);
    let mut n_accepted = 0usize;
    let meas_start = Instant::now();

    for i in 0..n_meas {
        let r = gpu_hmc_trajectory_streaming(
            gpu,
            pipelines,
            &state,
            n_md,
            dt,
            (n_therm + i) as u32,
            &mut s,
        );
        plaq_vals.push(r.plaquette);
        if r.accepted {
            n_accepted += 1;
        }
    }

    let meas_wall = meas_start.elapsed().as_secs_f64();
    let wall_per_traj = meas_wall / n_meas as f64;

    let mean_plaq: f64 = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
    let var_plaq: f64 = plaq_vals
        .iter()
        .map(|p| (p - mean_plaq).powi(2))
        .sum::<f64>()
        / (plaq_vals.len() - 1).max(1) as f64;
    let chi = var_plaq * vol as f64;
    let acceptance = n_accepted as f64 / n_meas as f64;

    (mean_plaq, chi, acceptance, 0.0, wall_per_traj)
}

/// Run a single dynamical parameter point. Returns (mean_plaq, chi, acceptance, mean_cg, wall_per_traj).
#[allow(clippy::too_many_arguments)]
fn run_dynamical_point(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcStreamingPipelines,
    rcg_pipelines: &GpuResidentCgPipelines,
    quenched_pipelines: &GpuHmcStreamingPipelines,
    dims: [usize; 4],
    beta: f64,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    check_interval: usize,
    n_md: usize,
    dt: f64,
    n_therm: usize,
    n_meas: usize,
    seed: u64,
) -> (f64, f64, f64, f64, f64) {
    let vol: usize = dims.iter().product();
    let mut lat = Lattice::hot_start(dims, beta, seed);

    // CPU pre-thermalization for small volumes
    if vol <= 65536 {
        let mut cfg = HmcConfig {
            n_md_steps: n_md,
            dt,
            seed: seed * 1000,
            integrator: IntegratorType::Omelyan,
        };
        for _ in 0..5 {
            hmc::hmc_trajectory(&mut lat, &mut cfg);
        }
    }

    // Quenched pre-thermalization (20 trajectories)
    let quenched_pretherm = 20.min(n_therm);
    let quenched_state = GpuHmcState::from_lattice(gpu, &lat, beta);
    let mut s = seed * 100;
    for i in 0..quenched_pretherm {
        gpu_hmc_trajectory_streaming(
            gpu,
            quenched_pipelines,
            &quenched_state,
            n_md,
            dt,
            i as u32,
            &mut s,
        );
    }
    gpu_links_to_lattice(gpu, &quenched_state, &mut lat);

    // Dynamical thermalization + measurement
    let dyn_state = GpuDynHmcState::from_lattice(gpu, &lat, beta, mass, cg_tol, cg_max_iter);
    let cg_bufs = GpuResidentCgBuffers::new(gpu, &dyn_pipelines.dyn_hmc, rcg_pipelines, &dyn_state);

    let dyn_therm = n_therm.saturating_sub(quenched_pretherm);
    for i in 0..dyn_therm {
        let traj_idx = quenched_pretherm + i;
        gpu_dynamical_hmc_trajectory_resident(
            gpu,
            dyn_pipelines,
            rcg_pipelines,
            &dyn_state,
            &cg_bufs,
            n_md,
            dt,
            traj_idx as u32,
            &mut s,
            check_interval,
        );
    }

    let mut plaq_vals = Vec::with_capacity(n_meas);
    let mut n_accepted = 0usize;
    let mut cg_total = 0usize;
    let meas_start = Instant::now();

    for i in 0..n_meas {
        let traj_idx = n_therm + i;
        let r = gpu_dynamical_hmc_trajectory_resident(
            gpu,
            dyn_pipelines,
            rcg_pipelines,
            &dyn_state,
            &cg_bufs,
            n_md,
            dt,
            traj_idx as u32,
            &mut s,
            check_interval,
        );
        plaq_vals.push(r.plaquette);
        if r.accepted {
            n_accepted += 1;
        }
        cg_total += r.cg_iterations;
    }

    let meas_wall = meas_start.elapsed().as_secs_f64();
    let wall_per_traj = meas_wall / n_meas as f64;

    let mean_plaq: f64 = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
    let var_plaq: f64 = plaq_vals
        .iter()
        .map(|p| (p - mean_plaq).powi(2))
        .sum::<f64>()
        / (plaq_vals.len() - 1).max(1) as f64;
    let chi = var_plaq * vol as f64;
    let acceptance = n_accepted as f64 / n_meas as f64;
    let mean_cg = cg_total as f64 / n_meas as f64;

    (mean_plaq, chi, acceptance, mean_cg, wall_per_traj)
}
