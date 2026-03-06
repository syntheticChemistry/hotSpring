// SPDX-License-Identifier: AGPL-3.0-only

//! Production dynamical fermion HMC — full QCD with staggered quarks.
//!
//! Generates dynamical gauge configurations with N_f staggered fermion
//! flavors via GPU HMC. Each staggered field contributes 4 tastes, so:
//!
//! | --nf flag | Pseudofermion fields | Physical flavors | Method |
//! |-----------|---------------------|-----------------|--------|
//! | 4         | 1                   | N_f=4           | Standard HMC |
//! | 2         | 1 (rooted)          | N_f=2           | RHMC |
//! | 2+1       | 2 (rooted)          | N_f=2+1         | RHMC |
//!
//! Measures plaquette, Polyakov loop, and CG iteration count per trajectory.
//! With --flow, runs gradient flow on each measurement config for t₀ and w₀.
//!
//! # Usage
//!
//! ```bash
//! # N_f=4 staggered on 8⁴:
//! cargo run --release --bin production_dynamical -- \
//!   --lattice=8 --beta=5.6 --mass=0.1 --therm=100 --meas=200
//!
//! # N_f=4 with gradient flow:
//! cargo run --release --bin production_dynamical -- \
//!   --lattice=8 --beta=5.6 --mass=0.1 --therm=100 --meas=200 --flow
//!
//! # Asymmetric lattice:
//! cargo run --release --bin production_dynamical -- \
//!   --dims=16,16,16,8 --beta=5.6 --mass=0.05 --therm=200 --meas=500
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::dynamical::{
    gpu_dynamical_hmc_trajectory, GpuDynHmcPipelines, GpuDynHmcState,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::io::Write;
use std::time::Instant;

struct CliArgs {
    dims: [usize; 4],
    beta: f64,
    mass: f64,
    n_therm: usize,
    n_meas: usize,
    n_md: usize,
    dt: f64,
    seed: u64,
    measure_flow: bool,
    flow_t_max: f64,
    output: Option<String>,
    traj_log: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut dims = [8, 8, 8, 8];
    let mut beta = 5.6;
    let mut mass = 0.1;
    let mut n_therm = 100;
    let mut n_meas = 200;
    let mut n_md = 10;
    let mut dt = 0.02;
    let mut seed = 42u64;
    let mut measure_flow = false;
    let mut flow_t_max = 2.0;
    let mut output = None;
    let mut traj_log = None;

    for arg in &args[1..] {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            let n: usize = val.parse().expect("bad --lattice");
            dims = [n, n, n, n];
        } else if let Some(val) = arg.strip_prefix("--dims=") {
            let parts: Vec<usize> = val.split(',').map(|s| s.parse().expect("bad --dims")).collect();
            assert_eq!(parts.len(), 4, "--dims expects 4 values");
            dims = [parts[0], parts[1], parts[2], parts[3]];
        } else if let Some(val) = arg.strip_prefix("--beta=") {
            beta = val.parse().expect("bad --beta");
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            mass = val.parse().expect("bad --mass");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("bad --therm");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            n_meas = val.parse().expect("bad --meas");
        } else if let Some(val) = arg.strip_prefix("--nmd=") {
            n_md = val.parse().expect("bad --nmd");
        } else if let Some(val) = arg.strip_prefix("--dt=") {
            dt = val.parse().expect("bad --dt");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("bad --seed");
        } else if arg == "--flow" {
            measure_flow = true;
        } else if let Some(val) = arg.strip_prefix("--tmax=") {
            flow_t_max = val.parse().expect("bad --tmax");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--traj-log=") {
            traj_log = Some(val.to_string());
        }
    }

    CliArgs {
        dims, beta, mass, n_therm, n_meas, n_md, dt, seed,
        measure_flow, flow_t_max, output, traj_log,
    }
}

fn main() {
    let args = parse_args();
    let dims = args.dims;
    let vol: usize = dims.iter().product();
    let is_asym = dims[3] != dims[0];
    let lat_label = if is_asym {
        format!("{}³×{}", dims[0], dims[3])
    } else {
        format!("{}⁴", dims[0])
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Production Dynamical Fermion HMC — N_f=4 Staggered       ║");
    println!("║  GPU HMC with Staggered Dirac + CG Solver                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:     {} ({} sites)", lat_label, vol);
    println!("  β:           {:.4}", args.beta);
    println!("  mass:        {:.4}", args.mass);
    println!("  N_f:         4 (1 staggered field × 4 tastes)");
    println!("  HMC:         dt={:.4}, n_md={}, τ={:.3}", args.dt, args.n_md, args.dt * args.n_md as f64);
    println!("  Therm:       {}", args.n_therm);
    println!("  Meas:        {}", args.n_meas);
    if args.measure_flow {
        println!("  Flow:        RK3 Lüscher, t_max={}", args.flow_t_max);
    }
    println!();

    let total_t0 = Instant::now();

    println!("  Initializing GPU...");
    let lattice = Lattice::hot_start(dims, args.beta, args.seed);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU init failed: {e}");
            std::process::exit(1);
        }
    };
    println!("  GPU: {}", gpu.adapter_name);

    let pipelines = GpuDynHmcPipelines::new(&gpu);
    let state = GpuDynHmcState::from_lattice(
        &gpu, &lattice, args.beta, args.mass,
        1e-8, 1000,
    );

    let mut seed = args.seed;
    let mut traj_file = args.traj_log.as_ref().map(|path| {
        std::fs::File::create(path).expect("failed to create traj log")
    });

    println!();
    println!("  Phase 1: Thermalization ({} trajectories)...", args.n_therm);
    let therm_start = Instant::now();

    let mut accepted_therm = 0;
    for i in 0..args.n_therm {
        let result = gpu_dynamical_hmc_trajectory(
            &gpu, &pipelines, &state, args.n_md, args.dt, &mut seed,
        );
        if result.accepted {
            accepted_therm += 1;
        }
        if (i + 1) % 20 == 0 || i == args.n_therm - 1 {
            println!(
                "    therm {}/{}: ⟨P⟩={:.6} ΔH={:.4} CG={} {}",
                i + 1,
                args.n_therm,
                result.plaquette,
                result.delta_h,
                result.cg_iterations,
                if result.accepted { "✓" } else { "✗" }
            );
        }
    }
    println!(
        "    Therm done: {:.1}s, acc={:.0}%",
        therm_start.elapsed().as_secs_f64(),
        accepted_therm as f64 / args.n_therm as f64 * 100.0
    );

    println!();
    println!("  Phase 2: Measurement ({} trajectories)...", args.n_meas);
    let meas_start = Instant::now();

    let mut plaquettes = Vec::new();
    let mut delta_hs = Vec::new();
    let mut cg_iters_total = Vec::new();
    let mut accepted_meas = 0;

    for i in 0..args.n_meas {
        let traj_start = Instant::now();
        let result = gpu_dynamical_hmc_trajectory(
            &gpu, &pipelines, &state, args.n_md, args.dt, &mut seed,
        );
        let wall_us = traj_start.elapsed().as_micros() as u64;

        if result.accepted {
            accepted_meas += 1;
        }
        plaquettes.push(result.plaquette);
        delta_hs.push(result.delta_h);
        cg_iters_total.push(result.cg_iterations);

        if let Some(ref mut f) = traj_file {
            let json = serde_json::json!({
                "traj_idx": i,
                "accepted": result.accepted,
                "plaquette": result.plaquette,
                "delta_h": result.delta_h,
                "cg_iterations": result.cg_iterations,
                "wall_us": wall_us,
            });
            writeln!(f, "{}", json).ok();
        }

        if (i + 1) % 20 == 0 || i == args.n_meas - 1 {
            let mean_plaq = plaquettes.iter().sum::<f64>() / plaquettes.len() as f64;
            let mean_cg = cg_iters_total.iter().sum::<usize>() / cg_iters_total.len();
            println!(
                "    meas {}/{}: ⟨P⟩={:.6} ΔH={:.4} CG={} acc={:.0}%",
                i + 1,
                args.n_meas,
                mean_plaq,
                result.delta_h,
                mean_cg,
                accepted_meas as f64 / (i + 1) as f64 * 100.0,
            );
        }
    }

    let meas_wall = meas_start.elapsed().as_secs_f64();
    let total_wall = total_t0.elapsed().as_secs_f64();

    let mean_plaq = plaquettes.iter().sum::<f64>() / plaquettes.len() as f64;
    let std_plaq = std_dev(&plaquettes);
    let acc_rate = accepted_meas as f64 / args.n_meas as f64;
    let mean_cg = cg_iters_total.iter().sum::<usize>() as f64 / cg_iters_total.len() as f64;
    let ms_per_traj = meas_wall * 1000.0 / args.n_meas as f64;

    println!();
    println!("  ══════════════════════════════════════════════════");
    println!("  {} N_f=4 Dynamical Summary (β={:.4}, m={:.4})", lat_label, args.beta, args.mass);
    println!("  ══════════════════════════════════════════════════");
    println!("  ⟨P⟩          = {:.6} ± {:.6}", mean_plaq, std_plaq);
    println!("  Acceptance   = {:.1}%", acc_rate * 100.0);
    println!("  ⟨CG iters⟩   = {:.1}", mean_cg);
    println!("  ms/trajectory = {:.1}", ms_per_traj);
    println!("  Meas wall    = {:.1}s", meas_wall);
    println!("  Total wall   = {:.1}s", total_wall);

    if let Some(path) = args.output {
        let json = serde_json::json!({
            "lattice": lat_label,
            "dims": dims,
            "beta": args.beta,
            "mass": args.mass,
            "nf": 4,
            "volume": vol,
            "n_therm": args.n_therm,
            "n_meas": args.n_meas,
            "n_md": args.n_md,
            "dt": args.dt,
            "mean_plaquette": mean_plaq,
            "std_plaquette": std_plaq,
            "acceptance_rate": acc_rate,
            "mean_cg_iterations": mean_cg,
            "ms_per_trajectory": ms_per_traj,
            "total_wall_s": total_wall,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
            .expect("failed to write output");
        println!("  Results → {}", path);
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}
