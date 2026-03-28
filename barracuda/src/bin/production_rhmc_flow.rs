// SPDX-License-Identifier: AGPL-3.0-only

//! Production RHMC + gradient flow — Chuna Paper 43 continuation.
//!
//! Thermalizes Nf=2 or Nf=2+1 dynamical configs via GPU RHMC, then runs
//! Wilson gradient flow on each measurement config to extract t₀ and w₀.
//!
//! This is the experiment that closes the loop: Chuna's LSCFRK integrators
//! running on GPU-generated dynamical fermion configs at consumer scale.
//!
//! # Usage
//!
//! ```bash
//! # Nf=2 at 8^4 with gradient flow:
//! cargo run --release --bin production_rhmc_flow -- \
//!   --lattice=8 --beta=6.0 --nf=2 --mass=0.1 --therm=50 --configs=10 --skip=5
//!
//! # Nf=2+1 at 8^4:
//! cargo run --release --bin production_rhmc_flow -- \
//!   --lattice=8 --beta=6.0 --nf=2+1 --mass=0.05 --strange-mass=0.5 \
//!   --therm=50 --configs=5 --skip=5
//! ```

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_rhmc_trajectory_unidirectional,
    GpuDynHmcPipelines, GpuDynHmcState, GpuHmcState, GpuHmcStreamingPipelines,
    GpuRhmcPipelines, GpuRhmcState, UniHamiltonianBuffers, UniPipelines,
};
use hotspring_barracuda::lattice::gpu_hmc::resident_shifted_cg::GpuResidentShiftedCgBuffers;
use hotspring_barracuda::lattice::gradient_flow::{find_t0, find_w0, FlowIntegrator};
use hotspring_barracuda::lattice::gpu_flow::{
    GpuFlowPipelines, GpuFlowState, FlowReduceBuffers, gpu_gradient_flow_resident,
};
use hotspring_barracuda::lattice::rhmc::RhmcConfig;
use hotspring_barracuda::lattice::wilson::Lattice;

use std::time::Instant;

struct CliArgs {
    lattice: usize,
    beta: f64,
    nf: String,
    mass: f64,
    strange_mass: f64,
    n_quenched_pretherm: usize,
    n_therm: usize,
    n_configs: usize,
    n_skip: usize,
    n_md_steps: usize,
    dt: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    flow_epsilon: f64,
    flow_t_max: f64,
    flow_integrator: FlowIntegrator,
    seed: u64,
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        lattice: 8,
        beta: 6.0,
        nf: "2".to_string(),
        mass: 0.1,
        strange_mass: 0.5,
        n_quenched_pretherm: 30,
        n_therm: 50,
        n_configs: 10,
        n_skip: 5,
        n_md_steps: 2,
        dt: 0.005,
        cg_tol: 1e-10,
        cg_max_iter: 5000,
        flow_epsilon: 0.01,
        flow_t_max: 3.0,
        flow_integrator: FlowIntegrator::Lscfrk3w7,
        seed: 42,
    };

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            args.lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--beta=") {
            args.beta = val.parse().expect("--beta=F");
        } else if let Some(val) = arg.strip_prefix("--nf=") {
            args.nf = val.to_string();
        } else if let Some(val) = arg.strip_prefix("--mass=") {
            args.mass = val.parse().expect("--mass=F");
        } else if let Some(val) = arg.strip_prefix("--strange-mass=") {
            args.strange_mass = val.parse().expect("--strange-mass=F");
        } else if let Some(val) = arg.strip_prefix("--quenched-pretherm=") {
            args.n_quenched_pretherm = val.parse().expect("--quenched-pretherm=N");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            args.n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--configs=") {
            args.n_configs = val.parse().expect("--configs=N");
        } else if let Some(val) = arg.strip_prefix("--skip=") {
            args.n_skip = val.parse().expect("--skip=N");
        } else if let Some(val) = arg.strip_prefix("--n-md=") {
            args.n_md_steps = val.parse().expect("--n-md=N");
        } else if let Some(val) = arg.strip_prefix("--dt=") {
            args.dt = val.parse().expect("--dt=F");
        } else if let Some(val) = arg.strip_prefix("--cg-tol=") {
            args.cg_tol = val.parse().expect("--cg-tol=F");
        } else if let Some(val) = arg.strip_prefix("--flow-epsilon=") {
            args.flow_epsilon = val.parse().expect("--flow-epsilon=F");
        } else if let Some(val) = arg.strip_prefix("--flow-tmax=") {
            args.flow_t_max = val.parse().expect("--flow-tmax=F");
        } else if let Some(val) = arg.strip_prefix("--integrator=") {
            args.flow_integrator = match val {
                "euler" => FlowIntegrator::Euler,
                "rk2" => FlowIntegrator::Rk2,
                "luscher" | "rk3" | "w6" => FlowIntegrator::Rk3Luscher,
                "w7" | "lscfrk3w7" => FlowIntegrator::Lscfrk3w7,
                "ck" | "lscfrk4ck" => FlowIntegrator::Lscfrk4ck,
                _ => panic!("Unknown integrator: {val}. Use: w7, ck, luscher"),
            };
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            args.seed = val.parse().expect("--seed=N");
        }
    }

    args
}

fn integrator_name(i: FlowIntegrator) -> &'static str {
    match i {
        FlowIntegrator::Euler => "Euler",
        FlowIntegrator::Rk2 => "RK2",
        FlowIntegrator::Rk3Luscher => "LSCFRK3W6 (Lüscher)",
        FlowIntegrator::Lscfrk3w7 => "LSCFRK3W7 (Chuna)",
        FlowIntegrator::Lscfrk4ck => "LSCFRK4CK",
    }
}

fn main() {
    let args = parse_args();
    let l = args.lattice;
    let dims = [l, l, l, l];
    let vol = l * l * l * l;
    let nf_label = match args.nf.as_str() {
        "2+1" | "3" => format!("Nf=2+1 (m_l={}, m_s={})", args.mass, args.strange_mass),
        _ => format!("Nf=2 (m={})", args.mass),
    };

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Production RHMC + Gradient Flow — Chuna Paper 43 Cont.   ║");
    eprintln!("║  Dynamical fermion configs → Wilson flow → t₀, w₀          ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Lattice:     {l}⁴ ({vol} sites)");
    eprintln!("  β:           {:.4}", args.beta);
    eprintln!("  Fermions:    {nf_label}");
    eprintln!("  RHMC:        dt={}, n_md={}, τ={:.4}", args.dt, args.n_md_steps, args.dt * args.n_md_steps as f64);
    eprintln!("  Pre-therm:   {} quenched HMC", args.n_quenched_pretherm);
    eprintln!("  Therm:       {} RHMC trajectories", args.n_therm);
    eprintln!("  Configs:     {} (skip {} RHMC between)", args.n_configs, args.n_skip);
    eprintln!("  Flow:        {} ε={} t_max={}", integrator_name(args.flow_integrator), args.flow_epsilon, args.flow_t_max);
    eprintln!();

    let total_start = Instant::now();

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

    // Phase 0: Quenched pre-thermalization
    let lattice = Lattice::hot_start(dims, args.beta, args.seed);
    let quenched_state = GpuHmcState::from_lattice(&gpu, &lattice, args.beta);

    if args.n_quenched_pretherm > 0 {
        eprintln!("\n  Phase 0: Quenched pre-thermalization ({} HMC)...", args.n_quenched_pretherm);
        let quenched_pipelines = GpuHmcStreamingPipelines::new(&gpu);
        for i in 0..args.n_quenched_pretherm {
            let r = gpu_hmc_trajectory_streaming(
                &gpu, &quenched_pipelines, &quenched_state,
                20, 0.1, i as u32, &mut seed,
            );
            if (i + 1) % 10 == 0 {
                eprintln!("    pretherm {}/{}: P={:.6} ΔH={:.4} {}",
                    i + 1, args.n_quenched_pretherm, r.plaquette, r.delta_h,
                    if r.accepted { "✓" } else { "✗" });
            }
        }
    }

    // Build RHMC config
    let mut rhmc_config = match args.nf.as_str() {
        "2+1" | "3" => RhmcConfig::nf2p1(args.mass, args.strange_mass, args.beta),
        _ => RhmcConfig::nf2(args.mass, args.beta),
    };
    rhmc_config.dt = args.dt;
    rhmc_config.n_md_steps = args.n_md_steps;
    rhmc_config.cg_tol = args.cg_tol;
    rhmc_config.cg_max_iter = args.cg_max_iter;

    // Build RHMC state from lattice, copy pre-thermalized links
    let dyn_state = GpuDynHmcState::from_lattice(
        &gpu, &lattice, args.beta,
        rhmc_config.sectors[0].mass, args.cg_tol, args.cg_max_iter,
    );
    let rhmc_state = GpuRhmcState::new(&gpu, &rhmc_config, dyn_state);

    if args.n_quenched_pretherm > 0 {
        let n_bytes = (quenched_state.n_links * 18 * 8) as u64;
        let mut enc = gpu.begin_encoder("copy_therm_links");
        enc.copy_buffer_to_buffer(
            &quenched_state.link_buf, 0,
            &rhmc_state.gauge.gauge.link_buf, 0,
            n_bytes,
        );
        gpu.submit_encoder(enc);
    }

    let dyn_pipelines = GpuDynHmcPipelines::new(&gpu);
    let rhmc_pipelines = GpuRhmcPipelines::new(&gpu);
    let uni_pipelines = UniPipelines::new(&gpu);
    let scg_bufs = GpuResidentShiftedCgBuffers::new(
        &gpu, &dyn_pipelines, &uni_pipelines.shifted_cg, &rhmc_state.gauge,
    );
    let ham_bufs = UniHamiltonianBuffers::new(
        &gpu, &uni_pipelines.shifted_cg.base.reduce_pipeline,
        &rhmc_state.gauge.gauge, &rhmc_state.gauge,
    );

    // Phase 1: RHMC thermalization
    eprintln!("\n  Phase 1: RHMC thermalization ({} trajectories)...", args.n_therm);
    let therm_start = Instant::now();
    let mut therm_accepted = 0;

    for i in 0..args.n_therm {
        let r = gpu_rhmc_trajectory_unidirectional(
            &gpu, &dyn_pipelines, &rhmc_pipelines, &uni_pipelines,
            &rhmc_state, &scg_bufs, None, &ham_bufs, &rhmc_config, &mut seed,
        );
        if r.accepted { therm_accepted += 1; }
        if (i + 1) % 10 == 0 || i == 0 {
            eprintln!("    therm {}/{}: P={:.6} ΔH={:.4e} acc={:.0}%",
                i + 1, args.n_therm, r.plaquette, r.delta_h,
                therm_accepted as f64 / (i + 1) as f64 * 100.0);
        }
    }
    let therm_time = therm_start.elapsed().as_secs_f64();
    eprintln!("    Thermalization: {therm_time:.1}s ({:.0}% acceptance)",
        therm_accepted as f64 / args.n_therm as f64 * 100.0);

    // Phase 2: RHMC measurement configs + gradient flow
    eprintln!("\n  Phase 2: RHMC → Gradient Flow ({} configs, skip {})", args.n_configs, args.n_skip);
    eprintln!("  ────────────────────────────────────────────────────────");

    let flow_pipelines = GpuFlowPipelines::new(&gpu);
    let mut all_plaq = Vec::new();
    let mut all_t0 = Vec::new();
    let mut all_w0 = Vec::new();

    for cfg_idx in 0..args.n_configs {
        // Skip trajectories for decorrelation
        for _ in 0..args.n_skip {
            gpu_rhmc_trajectory_unidirectional(
                &gpu, &dyn_pipelines, &rhmc_pipelines, &uni_pipelines,
                &rhmc_state, &scg_bufs, None, &ham_bufs, &rhmc_config, &mut seed,
            );
        }

        // GPU-resident gradient flow: B4+B5 fully eliminated
        let flow_state = GpuFlowState::from_gpu_gauge(&gpu, &rhmc_state.gauge.gauge);
        let flow_reduce = FlowReduceBuffers::new(&gpu, &flow_pipelines.reduce_pipeline, &flow_state);

        let flow_result = gpu_gradient_flow_resident(
            &gpu, &flow_pipelines, &flow_state, &flow_reduce,
            args.flow_integrator,
            args.flow_epsilon,
            args.flow_t_max,
            1,
        );

        // Pre-flow plaquette from the t=0 measurement (first entry)
        let plaq = flow_result.measurements.first().map_or(f64::NAN, |m| m.plaquette);
        all_plaq.push(plaq);

        let t0_val = find_t0(&flow_result.measurements);
        let w0_val = find_w0(&flow_result.measurements);

        if let Some(t0) = t0_val { all_t0.push(t0); }
        if let Some(w0) = w0_val { all_w0.push(w0); }

        let e_final = flow_result.measurements.last().map_or(f64::NAN, |m| m.energy_density);

        eprintln!("    cfg {:>3}/{}: ⟨P⟩={:.6} t₀={} w₀={} E(t_max)={:.4} ({:.1}s GPU flow)",
            cfg_idx + 1, args.n_configs, plaq,
            t0_val.map_or("N/A".to_string(), |v| format!("{v:.4}")),
            w0_val.map_or("N/A".to_string(), |v| format!("{v:.4}")),
            e_final, flow_result.wall_seconds);
    }

    // Summary
    let total_time = total_start.elapsed().as_secs_f64();
    let mean_plaq = all_plaq.iter().sum::<f64>() / all_plaq.len() as f64;
    let std_plaq = std_dev(&all_plaq);

    eprintln!();
    eprintln!("  ══════════════════════════════════════════════════════════");
    eprintln!("  {l}⁴ {} RHMC + Flow Summary (β={:.4})", args.nf, args.beta);
    eprintln!("  ══════════════════════════════════════════════════════════");
    eprintln!("  ⟨P⟩          = {mean_plaq:.6} ± {std_plaq:.6}");

    if !all_t0.is_empty() {
        let mean_t0 = all_t0.iter().sum::<f64>() / all_t0.len() as f64;
        let std_t0 = std_dev(&all_t0);
        eprintln!("  t₀           = {mean_t0:.4} ± {std_t0:.4} ({}/{} configs)", all_t0.len(), args.n_configs);
    } else {
        eprintln!("  t₀           = not found (increase --flow-tmax or lattice size)");
    }

    if !all_w0.is_empty() {
        let mean_w0 = all_w0.iter().sum::<f64>() / all_w0.len() as f64;
        let std_w0 = std_dev(&all_w0);
        eprintln!("  w₀           = {mean_w0:.4} ± {std_w0:.4} ({}/{} configs)", all_w0.len(), args.n_configs);
    } else {
        eprintln!("  w₀           = not found (increase --flow-tmax or lattice size)");
    }

    eprintln!("  Total wall   = {total_time:.1}s ({:.2}h)", total_time / 3600.0);
    eprintln!();

    // CSV output to stdout
    println!("cfg,plaquette,t0,w0");
    for (i, &p) in all_plaq.iter().enumerate() {
        let t0 = all_t0.get(i).copied().unwrap_or(f64::NAN);
        let w0 = all_w0.get(i).copied().unwrap_or(f64::NAN);
        println!("{i},{p:.8},{t0:.6},{w0:.6}");
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 { return 0.0; }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    var.sqrt()
}
