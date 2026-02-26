// SPDX-License-Identifier: AGPL-3.0-only

//! Production quenched β-scan at arbitrary lattice size.
//!
//! Runs GPU streaming Omelyan HMC at each β value, measuring plaquette,
//! Polyakov loop, susceptibility, and action density. Output streams to
//! stdout and a JSON results file.
//!
//! # Usage
//!
//! ```bash
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_beta_scan -- \
//!   --lattice=32 --betas=5.5,5.69,5.8,6.0 --therm=200 --meas=1000
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_links_to_lattice, GpuHmcState, GpuHmcStreamingPipelines,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;

use std::io::Write;
use std::time::Instant;

struct CliArgs {
    lattice: usize,
    betas: Vec<f64>,
    n_therm: usize,
    n_meas: usize,
    seed: u64,
    output: Option<String>,
    trajectory_log: Option<String>,
}

fn parse_args() -> CliArgs {
    let mut lattice = 32;
    let mut betas = vec![5.5, 5.69, 5.8, 6.0, 6.5];
    let mut n_therm = 200;
    let mut n_meas = 1000;
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
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("--therm=N");
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
        n_therm,
        n_meas,
        seed,
        output,
        trajectory_log,
    }
}

#[derive(Clone, Debug)]
struct BetaResult {
    beta: f64,
    mean_plaq: f64,
    std_plaq: f64,
    polyakov: f64,
    susceptibility: f64,
    action_density: f64,
    acceptance: f64,
    n_traj: usize,
    wall_s: f64,
}

/// Compute running plaquette variance from a history window.
fn plaquette_variance(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    history.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (history.len() - 1) as f64
}

/// Compute complex Polyakov loop average (returns (|L|, phase)).
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
    println!("║  Production Quenched β-Scan — GPU Streaming HMC (fp64)     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:  {}⁴ ({} sites)", args.lattice, vol);
    println!(
        "  VRAM est: {:.1} GB (quenched)",
        vol as f64 * 4.0 * 18.0 * 8.0 * 3.0 / 1e9
    );
    println!("  β values: {:?}", args.betas);
    println!("  Therm:    {}, Meas: {}", args.n_therm, args.n_meas);
    println!("  Seed:     {}", args.seed);
    if args.trajectory_log.is_some() {
        println!("  Trajectory log: ENABLED (JSONL per-trajectory)");
    }
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

    let vol_f = vol as f64;
    let ref_vol = 4096.0_f64;
    let scale = (ref_vol / vol_f).powf(0.25);
    let dt = (0.05 * scale).max(0.002);
    let n_md = ((0.5 / dt).round() as usize).max(10);
    println!(
        "  HMC:      dt={:.4}, n_md={}, traj_length={:.3}",
        dt,
        n_md,
        dt * n_md as f64
    );
    println!();

    let mut traj_writer: Option<std::io::BufWriter<std::fs::File>> =
        args.trajectory_log.as_ref().map(|path| {
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| panic!("Cannot create trajectory log {path}: {e}"));
            std::io::BufWriter::new(f)
        });

    let total_start = Instant::now();
    let mut results = Vec::new();

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);

    for (bi, &beta) in args.betas.iter().enumerate() {
        println!("── β = {:.4} ({}/{}) ──", beta, bi + 1, args.betas.len());

        let start = Instant::now();
        let mut lat = Lattice::hot_start(dims, beta, args.seed + bi as u64);

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

        let state = GpuHmcState::from_lattice(&gpu, &lat, beta);
        let mut seed = args.seed * 100 + bi as u64;
        let mut plaq_history: Vec<f64> = Vec::with_capacity(32);

        print!("  Thermalizing ({} traj)...", args.n_therm);
        std::io::stdout().flush().ok();
        for i in 0..args.n_therm {
            let traj_start = Instant::now();
            let r = gpu_hmc_trajectory_streaming(
                &gpu, &pipelines, &state, n_md, dt, i as u32, &mut seed,
            );
            let wall_us = traj_start.elapsed().as_micros() as u64;

            plaq_history.push(r.plaquette);
            if plaq_history.len() > 32 {
                plaq_history.remove(0);
            }

            if let Some(ref mut w) = traj_writer {
                gpu_links_to_lattice(&gpu, &state, &mut lat);
                let (poly_mag, poly_phase) = complex_polyakov_average(&lat);
                let pvar = plaquette_variance(&plaq_history);
                let line = serde_json::json!({
                    "beta": beta,
                    "traj_idx": i,
                    "is_therm": true,
                    "accepted": r.accepted,
                    "plaquette": r.plaquette,
                    "polyakov_re": poly_mag,
                    "delta_h": r.delta_h,
                    "cg_iters": 0,
                    "plaquette_var": pvar,
                    "polyakov_phase": poly_phase,
                    "action_density": 6.0 * (1.0 - r.plaquette),
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

        let mut plaq_vals = Vec::with_capacity(args.n_meas);
        let mut poly_vals = Vec::with_capacity(args.n_meas);
        let mut n_accepted = 0usize;
        plaq_history.clear();

        print!("  Measuring ({} traj)...", args.n_meas);
        std::io::stdout().flush().ok();
        for i in 0..args.n_meas {
            let traj_start = Instant::now();
            let r = gpu_hmc_trajectory_streaming(
                &gpu,
                &pipelines,
                &state,
                n_md,
                dt,
                (args.n_therm + i) as u32,
                &mut seed,
            );
            let wall_us = traj_start.elapsed().as_micros() as u64;

            plaq_vals.push(r.plaquette);
            plaq_history.push(r.plaquette);
            if plaq_history.len() > 32 {
                plaq_history.remove(0);
            }
            if r.accepted {
                n_accepted += 1;
            }

            let do_poly_readback = traj_writer.is_some() || (i + 1) % 100 == 0;
            let mut poly_mag = 0.0;
            let mut poly_phase = 0.0;
            if do_poly_readback {
                gpu_links_to_lattice(&gpu, &state, &mut lat);
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
                    "traj_idx": args.n_therm + i,
                    "is_therm": false,
                    "accepted": r.accepted,
                    "plaquette": r.plaquette,
                    "polyakov_re": poly_mag,
                    "delta_h": r.delta_h,
                    "cg_iters": 0,
                    "plaquette_var": pvar,
                    "polyakov_phase": poly_phase,
                    "action_density": 6.0 * (1.0 - r.plaquette),
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
            gpu_links_to_lattice(&gpu, &state, &mut lat);
            lat.average_polyakov_loop()
        } else {
            poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
        };

        let susceptibility = var_plaq * vol as f64;
        let action_density = 6.0 * (1.0 - mean_plaq);
        let acceptance = n_accepted as f64 / args.n_meas as f64;
        let wall_s = start.elapsed().as_secs_f64();

        let result = BetaResult {
            beta,
            mean_plaq,
            std_plaq,
            polyakov: mean_poly,
            susceptibility,
            action_density,
            acceptance,
            n_traj: args.n_meas,
            wall_s,
        };
        results.push(result.clone());

        println!(
            "  ⟨P⟩ = {:.6} ± {:.6}  |L| = {:.4}  χ = {:.4}  acc = {:.0}%  ({:.1}s)",
            mean_plaq,
            std_plaq,
            mean_poly,
            susceptibility,
            acceptance * 100.0,
            wall_s,
        );
        println!();
    }

    let total_wall = total_start.elapsed().as_secs_f64();

    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Production β-Scan Summary: {}⁴ Quenched SU(3)",
        args.lattice
    );
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "β", "⟨P⟩", "σ(P)", "|L|", "χ", "acc%", "time"
    );
    for r in &results {
        println!(
            "  {:>6.4} {:>10.6} {:>10.6} {:>10.4} {:>10.4} {:>7.1}% {:>7.1}s",
            r.beta,
            r.mean_plaq,
            r.std_plaq,
            r.polyakov,
            r.susceptibility,
            r.acceptance * 100.0,
            r.wall_s
        );
    }
    println!();
    println!(
        "  Total wall time: {:.1}s ({:.1} min)",
        total_wall,
        total_wall / 60.0
    );
    println!("  GPU: {}", gpu.adapter_name);
    println!();

    if let Some(ref path) = args.trajectory_log {
        println!("  Trajectory log: {path}");
    }

    if let Some(path) = args.output {
        let json = serde_json::json!({
            "lattice": args.lattice,
            "dims": dims,
            "volume": vol,
            "gpu": gpu.adapter_name,
            "n_therm": args.n_therm,
            "n_meas": args.n_meas,
            "seed": args.seed,
            "total_wall_s": total_wall,
            "points": results.iter().map(|r| serde_json::json!({
                "beta": r.beta,
                "mean_plaquette": r.mean_plaq,
                "std_plaquette": r.std_plaq,
                "polyakov": r.polyakov,
                "susceptibility": r.susceptibility,
                "action_density": r.action_density,
                "acceptance": r.acceptance,
                "n_trajectories": r.n_traj,
                "wall_s": r.wall_s,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
            .unwrap_or_else(|e| eprintln!("  Failed to write {path}: {e}"));
        println!("  Results saved to: {path}");
    }
}
