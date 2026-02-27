// SPDX-License-Identifier: AGPL-3.0-only

//! Production dynamical fermion β-scan at arbitrary lattice size.
//!
//! Runs GPU-resident CG dynamical HMC at each β value, measuring plaquette,
//! Polyakov loop, susceptibility, action density, and CG convergence.
//! Uses hot start + quenched pre-thermalization for reliable acceptance.
//!
//! # Usage
//!
//! ```bash
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_scan -- \
//!   --lattice=8 --betas=5.0,5.5,5.69,6.0 --mass=0.1 --therm=200 --meas=500
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

struct CliArgs {
    lattice: usize,
    betas: Vec<f64>,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    check_interval: usize,
    n_therm: usize,
    n_quenched_pretherm: usize,
    n_meas: usize,
    seed: u64,
    output: Option<String>,
    trajectory_log: Option<String>,
}

fn parse_args() -> CliArgs {
    let mut lattice = 8;
    let mut betas = vec![5.0, 5.5, 5.69, 6.0];
    let mut mass = 0.1;
    let mut cg_tol = 1e-8;
    let mut cg_max_iter = 5000;
    let mut check_interval = 10;
    let mut n_therm = 200;
    let mut n_quenched_pretherm = 50;
    let mut n_meas = 500;
    let mut seed = 42u64;
    let mut output = None;
    let mut trajectory_log = None;

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--betas=") {
            betas = val
                .split(',')
                .map(|s| s.parse().expect("beta float"))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            mass = val.parse().expect("--mass=F");
        } else if let Some(val) = arg.strip_prefix("--cg-tol=") {
            cg_tol = val.parse().expect("--cg-tol=F");
        } else if let Some(val) = arg.strip_prefix("--cg-max-iter=") {
            cg_max_iter = val.parse().expect("--cg-max-iter=N");
        } else if let Some(val) = arg.strip_prefix("--check-interval=") {
            check_interval = val.parse().expect("--check-interval=N");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--quenched-pretherm=") {
            n_quenched_pretherm = val.parse().expect("--quenched-pretherm=N");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            n_meas = val.parse().expect("--meas=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--trajectory-log=") {
            trajectory_log = Some(val.to_string());
        }
    }

    CliArgs {
        lattice,
        betas,
        mass,
        cg_tol,
        cg_max_iter,
        check_interval,
        n_therm,
        n_quenched_pretherm,
        n_meas,
        seed,
        output,
        trajectory_log,
    }
}

#[derive(Clone, Debug)]
struct BetaResult {
    beta: f64,
    mass: f64,
    mean_plaq: f64,
    std_plaq: f64,
    polyakov: f64,
    susceptibility: f64,
    action_density: f64,
    acceptance: f64,
    mean_cg_iters: f64,
    n_traj: usize,
    wall_s: f64,
}

fn plaquette_variance(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    history
        .iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>()
        / (history.len() - 1) as f64
}

fn complex_polyakov_average(lat: &Lattice) -> (f64, f64) {
    let ns = [lat.dims[0], lat.dims[1], lat.dims[2]];
    let spatial_vol = ns[0] * ns[1] * ns[2];
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for ix in 0..ns[0] {
        for iy in 0..ns[1] {
            for iz in 0..ns[2] {
                let c = lat.polyakov_loop([ix, iy, iz]);
                sum_re += c.re;
                sum_im += c.im;
            }
        }
    }
    let avg_re = sum_re / spatial_vol as f64;
    let avg_im = sum_im / spatial_vol as f64;
    let mag = (avg_re * avg_re + avg_im * avg_im).sqrt();
    let phase = avg_im.atan2(avg_re);
    (mag, phase)
}

fn main() {
    let args = parse_args();
    let dims = [args.lattice, args.lattice, args.lattice, args.lattice];
    let vol: usize = dims.iter().product();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Production Dynamical β-Scan — GPU Resident CG (fp64)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:  {}⁴ ({} sites)", args.lattice, vol);
    println!(
        "  VRAM est: {:.1} GB (dynamical)",
        vol as f64 * 4.0 * 18.0 * 8.0 * 5.0 / 1e9
    );
    println!("  β values: {:?}", args.betas);
    println!("  Mass:     {}", args.mass);
    println!("  CG:       tol={:.0e}, max_iter={}, check_interval={}", args.cg_tol, args.cg_max_iter, args.check_interval);
    println!("  Therm:    {} dynamical + {} quenched pre-therm", args.n_therm, args.n_quenched_pretherm);
    println!("  Meas:     {}", args.n_meas);
    println!("  Seed:     {}", args.seed);
    if args.trajectory_log.is_some() {
        println!("  Trajectory log: ENABLED (JSONL per-trajectory)");
    }
    println!();

    let vol_f = vol as f64;
    let ref_vol = 4096.0_f64;
    let scale = (ref_vol / vol_f).powf(0.25);
    let dt = (0.05 * scale).max(0.001);
    let n_md = ((0.5 / dt).round() as usize).max(20);
    println!(
        "  HMC:      dt={:.4}, n_md={}, traj_length={:.3}",
        dt, n_md, dt * n_md as f64
    );
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

    let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let dyn_streaming_pipelines = GpuDynHmcStreamingPipelines::new(&gpu);
    let resident_cg_pipelines = GpuResidentCgPipelines::new(&gpu);

    let mut traj_writer: Option<std::io::BufWriter<std::fs::File>> =
        args.trajectory_log.as_ref().map(|path| {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| panic!("Cannot create trajectory log {path}: {e}"));
            std::io::BufWriter::new(f)
        });

    let total_start = Instant::now();
    let mut results = Vec::new();

    for (bi, &beta) in args.betas.iter().enumerate() {
        println!("── β = {:.4}, m = {} ({}/{}) ──", beta, args.mass, bi + 1, args.betas.len());

        let start = Instant::now();
        let mut lat = Lattice::hot_start(dims, beta, args.seed + bi as u64);

        // CPU pre-thermalization for small volumes
        if vol <= 65536 {
            let mut cfg = HmcConfig {
                n_md_steps: n_md,
                dt,
                seed: args.seed + bi as u64 * 1000,
                integrator: IntegratorType::Omelyan,
            };
            for _ in 0..5 {
                hmc::hmc_trajectory(&mut lat, &mut cfg);
            }
        }

        // Phase 1: Quenched pre-thermalization on GPU
        let quenched_state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut seed = args.seed * 100 + bi as u64;

        if args.n_quenched_pretherm > 0 {
            print!("  Quenched pre-therm ({} traj)...", args.n_quenched_pretherm);
            std::io::stdout().flush().ok();
            for i in 0..args.n_quenched_pretherm {
                let r = gpu_hmc_trajectory_streaming(
                    &gpu, &quenched_pipelines, &quenched_state, n_md, dt, i as u32, &mut seed,
                );
                if let Some(ref mut w) = traj_writer {
                    let line = serde_json::json!({
                        "beta": beta,
                        "mass": args.mass,
                        "traj_idx": i,
                        "phase": "quenched_pretherm",
                        "accepted": r.accepted,
                        "plaquette": r.plaquette,
                        "delta_h": r.delta_h,
                        "cg_iters": 0,
                    });
                    writeln!(w, "{line}").ok();
                }
                if (i + 1) % 10 == 0 {
                    print!(" {}", i + 1);
                    std::io::stdout().flush().ok();
                }
            }
            println!(" done");
        }

        // Read back thermalized links for the dynamical state
        gpu_links_to_lattice(&gpu, &quenched_state, &mut lat);

        // Phase 2: Dynamical HMC with resident CG
        let dyn_state = GpuDynHmcState::from_lattice(
            &gpu, &lat, beta, args.mass, args.cg_tol, args.cg_max_iter,
        );
        let cg_bufs = GpuResidentCgBuffers::new(
            &gpu,
            &dyn_streaming_pipelines.dyn_hmc,
            &resident_cg_pipelines,
            &dyn_state,
        );

        let mut plaq_history: Vec<f64> = Vec::with_capacity(32);

        print!("  Dynamical therm ({} traj)...", args.n_therm);
        std::io::stdout().flush().ok();
        for i in 0..args.n_therm {
            let traj_start = Instant::now();
            let traj_idx = args.n_quenched_pretherm + i;
            let r = gpu_dynamical_hmc_trajectory_resident(
                &gpu,
                &dyn_streaming_pipelines,
                &resident_cg_pipelines,
                &dyn_state,
                &cg_bufs,
                n_md,
                dt,
                traj_idx as u32,
                &mut seed,
                args.check_interval,
            );
            let wall_us = traj_start.elapsed().as_micros() as u64;

            plaq_history.push(r.plaquette);
            if plaq_history.len() > 32 {
                plaq_history.remove(0);
            }

            if let Some(ref mut w) = traj_writer {
                let pvar = plaquette_variance(&plaq_history);
                let line = serde_json::json!({
                    "beta": beta,
                    "mass": args.mass,
                    "traj_idx": traj_idx,
                    "phase": "dynamical_therm",
                    "accepted": r.accepted,
                    "plaquette": r.plaquette,
                    "delta_h": r.delta_h,
                    "cg_iters": r.cg_iterations,
                    "plaquette_var": pvar,
                    "wall_us": wall_us,
                });
                writeln!(w, "{line}").ok();
            }

            if (i + 1) % 50 == 0 {
                print!(" {}", i + 1);
                std::io::stdout().flush().ok();
            }
        }
        println!(" done");

        // Phase 3: Measurement
        let mut plaq_vals = Vec::with_capacity(args.n_meas);
        let mut poly_vals = Vec::new();
        let mut cg_iters_total = 0usize;
        let mut n_accepted = 0usize;
        plaq_history.clear();

        print!("  Measuring ({} traj)...", args.n_meas);
        std::io::stdout().flush().ok();
        for i in 0..args.n_meas {
            let traj_start = Instant::now();
            let traj_idx = args.n_quenched_pretherm + args.n_therm + i;
            let r = gpu_dynamical_hmc_trajectory_resident(
                &gpu,
                &dyn_streaming_pipelines,
                &resident_cg_pipelines,
                &dyn_state,
                &cg_bufs,
                n_md,
                dt,
                traj_idx as u32,
                &mut seed,
                args.check_interval,
            );
            let wall_us = traj_start.elapsed().as_micros() as u64;

            plaq_vals.push(r.plaquette);
            plaq_history.push(r.plaquette);
            if plaq_history.len() > 32 {
                plaq_history.remove(0);
            }
            cg_iters_total += r.cg_iterations;
            if r.accepted {
                n_accepted += 1;
            }

            let do_poly_readback = traj_writer.is_some() || (i + 1) % 100 == 0;
            let mut poly_mag = 0.0;
            let mut poly_phase = 0.0;
            if do_poly_readback {
                gpu_links_to_lattice(&gpu, &dyn_state.gauge, &mut lat);
                let (m, p) = complex_polyakov_average(&lat);
                poly_mag = m;
                poly_phase = p;
                if (i + 1) % 100 == 0 {
                    poly_vals.push(poly_mag);
                }
            }

            if let Some(ref mut w) = traj_writer {
                let pvar = plaquette_variance(&plaq_history);
                let line = serde_json::json!({
                    "beta": beta,
                    "mass": args.mass,
                    "traj_idx": traj_idx,
                    "phase": "measurement",
                    "accepted": r.accepted,
                    "plaquette": r.plaquette,
                    "delta_h": r.delta_h,
                    "cg_iters": r.cg_iterations,
                    "polyakov_re": poly_mag,
                    "polyakov_phase": poly_phase,
                    "action_density": 6.0 * (1.0 - r.plaquette),
                    "plaquette_var": pvar,
                    "wall_us": wall_us,
                });
                writeln!(w, "{line}").ok();
            }

            if (i + 1) % 200 == 0 {
                print!(" {}", i + 1);
                std::io::stdout().flush().ok();
            }
        }
        println!(" done");

        if let Some(ref mut w) = traj_writer {
            w.flush().ok();
        }

        let mean_plaq: f64 = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
        let var_plaq: f64 = plaq_vals
            .iter()
            .map(|p| (p - mean_plaq).powi(2))
            .sum::<f64>()
            / (plaq_vals.len() - 1).max(1) as f64;
        let std_plaq = var_plaq.sqrt();

        let mean_poly: f64 = if poly_vals.is_empty() {
            gpu_links_to_lattice(&gpu, &dyn_state.gauge, &mut lat);
            lat.average_polyakov_loop()
        } else {
            poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
        };

        let susceptibility = var_plaq * vol as f64;
        let action_density = 6.0 * (1.0 - mean_plaq);
        let acceptance = n_accepted as f64 / args.n_meas as f64;
        let mean_cg = cg_iters_total as f64 / args.n_meas as f64;
        let wall_s = start.elapsed().as_secs_f64();

        let result = BetaResult {
            beta,
            mass: args.mass,
            mean_plaq,
            std_plaq,
            polyakov: mean_poly,
            susceptibility,
            action_density,
            acceptance,
            mean_cg_iters: mean_cg,
            n_traj: args.n_meas,
            wall_s,
        };
        results.push(result.clone());

        println!(
            "  ⟨P⟩ = {:.6} ± {:.6}  |L| = {:.4}  χ = {:.4}  acc = {:.0}%  ⟨CG⟩ = {:.0}  ({:.1}s)",
            mean_plaq, std_plaq, mean_poly, susceptibility,
            acceptance * 100.0, mean_cg, wall_s,
        );
        println!();
    }

    let total_wall = total_start.elapsed().as_secs_f64();

    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Production β-Scan Summary: {}⁴ Dynamical SU(3), m={}",
        args.lattice, args.mass
    );
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "β", "⟨P⟩", "σ(P)", "|L|", "χ", "acc%", "⟨CG⟩", "time"
    );
    for r in &results {
        println!(
            "  {:>6.4} {:>10.6} {:>10.6} {:>10.4} {:>10.4} {:>7.1}% {:>8.0} {:>7.1}s",
            r.beta, r.mean_plaq, r.std_plaq, r.polyakov, r.susceptibility,
            r.acceptance * 100.0, r.mean_cg_iters, r.wall_s
        );
    }
    println!();
    println!(
        "  Total wall time: {:.1}s ({:.1} min)",
        total_wall, total_wall / 60.0
    );
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    if let Some(ref path) = args.trajectory_log {
        println!("  Trajectory log: {path}");
    }

    if let Some(path) = args.output {
        let json = serde_json::json!({
            "experiment": "023_DYNAMICAL_PRODUCTION",
            "lattice": args.lattice,
            "dims": dims,
            "volume": vol,
            "mass": args.mass,
            "cg_tol": args.cg_tol,
            "cg_max_iter": args.cg_max_iter,
            "check_interval": args.check_interval,
            "gpu": gpu.adapter_name,
            "dt": dt,
            "n_md": n_md,
            "n_quenched_pretherm": args.n_quenched_pretherm,
            "n_therm": args.n_therm,
            "n_meas": args.n_meas,
            "seed": args.seed,
            "total_wall_s": total_wall,
            "points": results.iter().map(|r| serde_json::json!({
                "beta": r.beta,
                "mass": r.mass,
                "mean_plaquette": r.mean_plaq,
                "std_plaquette": r.std_plaq,
                "polyakov": r.polyakov,
                "susceptibility": r.susceptibility,
                "action_density": r.action_density,
                "acceptance": r.acceptance,
                "mean_cg_iterations": r.mean_cg_iters,
                "n_trajectories": r.n_traj,
                "wall_s": r.wall_s,
            })).collect::<Vec<_>>(),
        });
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
            .unwrap_or_else(|e| eprintln!("  Failed to write {path}: {e}"));
        println!("  Results saved to: {path}");
    }
}
