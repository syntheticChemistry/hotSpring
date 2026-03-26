// SPDX-License-Identifier: AGPL-3.0-only

//! Production RHMC β-scan: all-flavors dynamical QCD (Nf=2, 2+1).
//!
//! Runs GPU RHMC trajectories at each β value with rational approximation
//! of fractional determinant powers. Measures plaquette, acceptance rate,
//! ΔH, and CG iterations per trajectory.
//!
//! Designed for weekend production runs at 32^4.
//!
//! # Usage
//!
//! ```bash
//! HOTSPRING_GPU_ADAPTER=6950 cargo run --release --bin production_rhmc_scan -- \
//!   --lattice=8 --betas=5.0,5.5,6.0 --nf=2 --mass=0.1 --therm=50 --meas=200
//!
//! # Nf=2+1 with light + strange mass:
//! cargo run --release --bin production_rhmc_scan -- \
//!   --lattice=16 --nf=2+1 --mass=0.05 --strange-mass=0.5 --therm=100 --meas=500
//!
//! # Weekend 32^4 production:
//! cargo run --release --bin production_rhmc_scan -- \
//!   --lattice=32 --betas=5.5,5.6,5.7,5.8,5.9,6.0 --nf=2 --mass=0.1 \
//!   --therm=100 --meas=400 --n-md=30 --dt=0.008
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_rhmc_trajectory, GpuDynHmcPipelines, GpuDynHmcState,
    GpuHmcState, GpuHmcStreamingPipelines, GpuRhmcPipelines, GpuRhmcState,
};
use hotspring_barracuda::lattice::rhmc::RhmcConfig;
use hotspring_barracuda::lattice::wilson::Lattice;

use std::io::Write;
use std::time::Instant;

struct CliArgs {
    lattice: usize,
    betas: Vec<f64>,
    nf: String,
    mass: f64,
    strange_mass: f64,
    n_therm: usize,
    n_quenched_pretherm: usize,
    n_meas: usize,
    n_md_steps: usize,
    dt: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: u64,
    output: Option<String>,
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        lattice: 8,
        betas: vec![5.5, 5.69, 6.0],
        nf: "2".to_string(),
        mass: 0.1,
        strange_mass: 0.5,
        n_therm: 50,
        n_quenched_pretherm: 30,
        n_meas: 200,
        n_md_steps: 30,
        dt: 0.01,
        cg_tol: 1e-8,
        cg_max_iter: 5000,
        seed: 42,
        output: None,
    };

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            args.lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--betas=") {
            args.betas = val
                .split(',')
                .map(|s| s.parse().expect("beta float"))
                .collect();
        } else if let Some(val) = arg.strip_prefix("--nf=") {
            args.nf = val.to_string();
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            args.mass = val.parse().expect("--mass=F");
        } else if let Some(val) = arg.strip_prefix("--strange-mass=") {
            args.strange_mass = val.parse().expect("--strange-mass=F");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            args.n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--quenched-pretherm=") {
            args.n_quenched_pretherm = val.parse().expect("--quenched-pretherm=N");
        } else if let Some(val) = arg.strip_prefix("--meas=") {
            args.n_meas = val.parse().expect("--meas=N");
        } else if let Some(val) = arg.strip_prefix("--n-md=") {
            args.n_md_steps = val.parse().expect("--n-md=N");
        } else if let Some(val) = arg.strip_prefix("--dt=") {
            args.dt = val.parse().expect("--dt=F");
        } else if let Some(val) = arg.strip_prefix("--cg-tol=") {
            args.cg_tol = val.parse().expect("--cg-tol=F");
        } else if let Some(val) = arg.strip_prefix("--cg-max-iter=") {
            args.cg_max_iter = val.parse().expect("--cg-max-iter=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            args.seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            args.output = Some(val.to_string());
        }
    }

    args
}

fn main() {
    let args = parse_args();
    let l = args.lattice;
    let dims = [l, l, l, l];

    eprintln!("═══════════════════════════════════════════════════════════");
    eprintln!("  Production RHMC β-scan");
    eprintln!("  Lattice: {l}^4 ({} sites)", l * l * l * l);
    eprintln!("  Nf: {}  mass: {}  strange: {}", args.nf, args.mass, args.strange_mass);
    eprintln!("  MD: {} steps × dt={}", args.n_md_steps, args.dt);
    eprintln!("  β points: {:?}", args.betas);
    eprintln!("  Therm: {} (+ {} quenched pre-therm)", args.n_therm, args.n_quenched_pretherm);
    eprintln!("  Meas: {} per β", args.n_meas);
    eprintln!("═══════════════════════════════════════════════════════════");

    let total_traj = args.betas.len() * (args.n_therm + args.n_meas);
    eprintln!("  Total trajectories: {total_traj}");

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("  GPU: {}", gpu.adapter_name);

    let mut seed = args.seed;

    // Open output file if requested
    let mut out_file = args.output.as_ref().map(|path| {
        let f = std::fs::File::create(path).expect("create output file");
        std::io::BufWriter::new(f)
    });

    // Write header
    let header = "beta,traj,accepted,delta_h,plaquette,cg_iters,ms_per_traj,phase";
    if let Some(ref mut f) = out_file {
        writeln!(f, "{header}").ok();
    }
    println!("{header}");

    let run_start = Instant::now();

    for &beta in &args.betas {
        eprintln!("\n━━━ β = {beta:.4} ━━━");

        // Create lattice with hot start
        let lattice = Lattice::hot_start(dims, beta, seed);

        // Build RHMC config
        let mut rhmc_config = match args.nf.as_str() {
            "2" => RhmcConfig::nf2(args.mass, beta),
            "2+1" | "3" => RhmcConfig::nf2p1(args.mass, args.strange_mass, beta),
            _ => {
                eprintln!("Unknown --nf={}, defaulting to Nf=2", args.nf);
                RhmcConfig::nf2(args.mass, beta)
            }
        };
        rhmc_config.dt = args.dt;
        rhmc_config.n_md_steps = args.n_md_steps;
        rhmc_config.cg_tol = args.cg_tol;
        rhmc_config.cg_max_iter = args.cg_max_iter;

        // Quenched pre-thermalization on a separate state (fast, warms gauge field)
        let quenched_state = GpuHmcState::from_lattice(&gpu, &lattice, beta);
        if args.n_quenched_pretherm > 0 {
            let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
            for i in 0..args.n_quenched_pretherm {
                let r = gpu_hmc_trajectory_streaming(
                    &gpu,
                    &quenched_pipelines,
                    &quenched_state,
                    20,
                    0.1,
                    i as u32,
                    &mut seed,
                );
                if (i + 1) % 10 == 0 {
                    eprintln!(
                        "  quenched pretherm {}/{}: P={:.6} ΔH={:.4} {}",
                        i + 1,
                        args.n_quenched_pretherm,
                        r.plaquette,
                        r.delta_h,
                        if r.accepted { "✓" } else { "✗" }
                    );
                }
            }
        }

        // GPU dynamical state built from the pre-thermalized gauge config.
        // GpuDynHmcState wraps the quenched state's link buffer (already warm).
        // For RHMC, we use from_lattice which re-uploads; we could optimize
        // to reuse the buffer later.
        let dyn_state = GpuDynHmcState::from_lattice(
            &gpu,
            &lattice,
            beta,
            rhmc_config.sectors[0].mass,
            args.cg_tol,
            args.cg_max_iter,
        );
        let rhmc_state = GpuRhmcState::new(&gpu, &rhmc_config, dyn_state);

        // Compile pipelines
        let dyn_pipelines = GpuDynHmcPipelines::new(&gpu);
        let rhmc_pipelines = GpuRhmcPipelines::new(&gpu);

        // RHMC thermalization
        let mut therm_accepted = 0;
        for i in 0..args.n_therm {
            let t0 = Instant::now();
            let r = gpu_rhmc_trajectory(
                &gpu,
                &dyn_pipelines,
                &rhmc_pipelines,
                &rhmc_state,
                &rhmc_config,
                &mut seed,
            );
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if r.accepted {
                therm_accepted += 1;
            }

            if (i + 1) % 10 == 0 || i == 0 {
                let rate = therm_accepted as f64 / (i + 1) as f64 * 100.0;
                eprintln!(
                    "  therm {}/{}: P={:.6} ΔH={:.4e} CG={} {:.0}ms acc={:.0}%",
                    i + 1,
                    args.n_therm,
                    r.plaquette,
                    r.delta_h,
                    r.total_cg_iterations,
                    ms,
                    rate
                );
            }
        }

        // RHMC measurement
        let mut meas_accepted = 0;
        let mut plaq_sum = 0.0;
        let mut plaq_sq_sum = 0.0;

        for i in 0..args.n_meas {
            let t0 = Instant::now();
            let r = gpu_rhmc_trajectory(
                &gpu,
                &dyn_pipelines,
                &rhmc_pipelines,
                &rhmc_state,
                &rhmc_config,
                &mut seed,
            );
            let ms = t0.elapsed().as_secs_f64() * 1000.0;

            if r.accepted {
                meas_accepted += 1;
            }
            plaq_sum += r.plaquette;
            plaq_sq_sum += r.plaquette * r.plaquette;

            let line = format!(
                "{beta:.4},{},{},{:.6e},{:.8},{},{:.1},meas",
                i,
                i32::from(r.accepted),
                r.delta_h,
                r.plaquette,
                r.total_cg_iterations,
                ms
            );
            println!("{line}");
            if let Some(ref mut f) = out_file {
                writeln!(f, "{line}").ok();
                f.flush().ok();
            }

            if (i + 1) % 50 == 0 || i + 1 == args.n_meas {
                let n = (i + 1) as f64;
                let mean_p = plaq_sum / n;
                let var_p = plaq_sq_sum / n - mean_p * mean_p;
                let rate = meas_accepted as f64 / n * 100.0;
                eprintln!(
                    "  meas {}/{}: ⟨P⟩={:.6} σ(P)={:.2e} acc={:.0}% {:.0}ms/traj",
                    i + 1,
                    args.n_meas,
                    mean_p,
                    var_p.max(0.0).sqrt(),
                    rate,
                    ms
                );
            }
        }

        // Summary for this β
        let n = args.n_meas as f64;
        let mean_p = plaq_sum / n;
        let var_p = (plaq_sq_sum / n - mean_p * mean_p).max(0.0);
        let chi = var_p * (l * l * l * l) as f64 * 6.0;
        let rate = meas_accepted as f64 / n * 100.0;

        eprintln!("\n  β={beta:.4} summary:");
        eprintln!("    ⟨P⟩ = {mean_p:.6} ± {:.2e}", var_p.sqrt());
        eprintln!("    χ   = {chi:.2}");
        eprintln!("    acc = {rate:.1}% ({meas_accepted}/{} trajectories)", args.n_meas);
    }

    let total_time = run_start.elapsed();
    eprintln!("\n═══════════════════════════════════════════════════════════");
    eprintln!("  Production RHMC Complete");
    eprintln!("  Total time: {:.1}s ({:.1}h)", total_time.as_secs_f64(), total_time.as_secs_f64() / 3600.0);
    eprintln!("═══════════════════════════════════════════════════════════");
}
