// SPDX-License-Identifier: AGPL-3.0-only

//! Production gradient flow measurement — SU(3) quenched.
//!
//! Generates quenched gauge configurations via GPU HMC, then runs Wilson
//! gradient flow (RK3 Lüscher) on each configuration to measure:
//!
//! - E(t): gauge energy density at flow time t
//! - t₀: flow scale defined by t²⟨E(t)⟩ = 0.3 (Lüscher 2010)
//! - w₀: flow scale defined by t d/dt[t²E(t)] = 0.3 (BMW 2012)
//!
//! This reproduces the core observable from Bazavov & Chuna,
//! arXiv:2101.05320, on local consumer hardware.
//!
//! # Usage
//!
//! ```bash
//! # Quick validation:
//! cargo run --release --bin production_gradient_flow -- \
//!   --lattice=8 --beta=6.0 --therm=100 --configs=10 --skip=10
//!
//! # Production 16⁴:
//! cargo run --release --bin production_gradient_flow -- \
//!   --lattice=16 --beta=6.0 --therm=200 --configs=50 --skip=20
//!
//! # Asymmetric lattice:
//! cargo run --release --bin production_gradient_flow -- \
//!   --dims=32,32,32,8 --beta=6.0 --therm=100 --configs=20 --skip=10
//! ```

use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, compute_w_function, find_t0, find_w0, run_flow,
};
use hotspring_barracuda::lattice::hmc::{HmcConfig, IntegratorType, hmc_trajectory};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

struct CliArgs {
    dims: [usize; 4],
    beta: f64,
    n_therm: usize,
    n_configs: usize,
    n_skip: usize,
    flow_t_max: f64,
    flow_epsilon: f64,
    integrator: FlowIntegrator,
    seed: u64,
    output: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut dims = [8, 8, 8, 8];
    let mut beta = 6.0;
    let mut n_therm = 200;
    let mut n_configs = 20;
    let mut n_skip = 10;
    let mut flow_t_max = 3.0;
    let mut flow_epsilon = 0.01;
    let mut integrator = FlowIntegrator::Lscfrk3w7;
    let mut seed = 42u64;
    let mut output = None;

    for arg in &args[1..] {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            let n: usize = val.parse().expect("bad --lattice");
            dims = [n, n, n, n];
        } else if let Some(val) = arg.strip_prefix("--dims=") {
            let parts: Vec<usize> = val
                .split(',')
                .map(|s| s.parse().expect("bad --dims"))
                .collect();
            assert_eq!(parts.len(), 4, "--dims expects 4 values");
            dims = [parts[0], parts[1], parts[2], parts[3]];
        } else if let Some(val) = arg.strip_prefix("--beta=") {
            beta = val.parse().expect("bad --beta");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("bad --therm");
        } else if let Some(val) = arg.strip_prefix("--configs=") {
            n_configs = val.parse().expect("bad --configs");
        } else if let Some(val) = arg.strip_prefix("--skip=") {
            n_skip = val.parse().expect("bad --skip");
        } else if let Some(val) = arg.strip_prefix("--tmax=") {
            flow_t_max = val.parse().expect("bad --tmax");
        } else if let Some(val) = arg.strip_prefix("--epsilon=") {
            flow_epsilon = val.parse().expect("bad --epsilon");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("bad --seed");
        } else if let Some(val) = arg.strip_prefix("--integrator=") {
            integrator = match val {
                "euler" => FlowIntegrator::Euler,
                "rk2" => FlowIntegrator::Rk2,
                "luscher" | "rk3" | "w6" => FlowIntegrator::Rk3Luscher,
                "w7" | "lscfrk3w7" => FlowIntegrator::Lscfrk3w7,
                "ck" | "lscfrk4ck" => FlowIntegrator::Lscfrk4ck,
                _ => panic!("Unknown integrator: {val}. Use: luscher, w7, ck"),
            };
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        }
    }

    CliArgs {
        dims,
        beta,
        n_therm,
        n_configs,
        n_skip,
        flow_t_max,
        flow_epsilon,
        integrator,
        seed,
        output,
    }
}

fn label(dims: &[usize; 4]) -> String {
    if dims[0] == dims[1] && dims[1] == dims[2] && dims[2] == dims[3] {
        format!("{}⁴", dims[0])
    } else {
        format!("{}³×{}", dims[0], dims[3])
    }
}

fn main() {
    let args = parse_args();
    let dims = args.dims;
    let vol: usize = dims.iter().product();
    let lat_label = label(&dims);
    let total_hmc_traj = args.n_therm + args.n_configs * args.n_skip;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Production Gradient Flow — SU(3) Quenched                 ║");
    println!("║  Wilson Flow + RK3 (Lüscher) → t₀, w₀                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:     {lat_label} ({vol} sites)");
    println!("  β:           {:.4}", args.beta);
    println!("  Therm:       {} HMC trajectories", args.n_therm);
    println!(
        "  Configs:     {} (skip {} between)",
        args.n_configs, args.n_skip
    );
    println!("  Total HMC:   {total_hmc_traj} trajectories");
    let integrator_name = match args.integrator {
        FlowIntegrator::Euler => "Euler (1st order)",
        FlowIntegrator::Rk2 => "RK2 (2nd order)",
        FlowIntegrator::Rk3Luscher => "LSCFRK3W6 / Lüscher (3rd order, 3-stage)",
        FlowIntegrator::Lscfrk3w7 => "LSCFRK3W7 / Chuna (3rd order, 3-stage, optimized)",
        FlowIntegrator::Lscfrk4ck => "LSCFRK4CK / Carpenter-Kennedy (4th order, 5-stage)",
    };
    println!(
        "  Flow:        {}, ε={}, t_max={}",
        integrator_name, args.flow_epsilon, args.flow_t_max
    );
    println!();

    let total_t0 = Instant::now();

    let mut hmc_cfg = HmcConfig {
        n_md_steps: 20,
        dt: 0.05,
        seed: args.seed,
        integrator: IntegratorType::Omelyan,
    };

    println!(
        "  Phase 1: Thermalization ({} HMC trajectories)...",
        args.n_therm
    );
    let therm_start = Instant::now();

    let mut lattice = Lattice::hot_start(dims, args.beta, args.seed);

    for i in 0..args.n_therm {
        let result = hmc_trajectory(&mut lattice, &mut hmc_cfg);
        if (i + 1) % 50 == 0 || i == args.n_therm - 1 {
            println!(
                "    therm {}/{}: ⟨P⟩={:.6} ΔH={:.4} {}",
                i + 1,
                args.n_therm,
                lattice.average_plaquette(),
                result.delta_h,
                if result.accepted { "✓" } else { "✗" }
            );
        }
    }

    println!(
        "    Thermalization: {:.1}s",
        therm_start.elapsed().as_secs_f64()
    );
    println!();

    println!(
        "  Phase 2: Gradient Flow Measurements ({} configs)",
        args.n_configs
    );
    println!("  ────────────────────────────────────────────────");

    let mut all_t0 = Vec::new();
    let mut all_w0 = Vec::new();
    let mut all_plaq = Vec::new();
    let mut config_results = Vec::new();

    for cfg in 0..args.n_configs {
        for _ in 0..args.n_skip {
            hmc_trajectory(&mut lattice, &mut hmc_cfg);
        }

        let plaq = lattice.average_plaquette();
        all_plaq.push(plaq);

        let mut flow_lattice = Lattice {
            dims: lattice.dims,
            links: lattice.links.clone(),
            beta: lattice.beta,
        };
        let flow_start = Instant::now();
        let measurements = run_flow(
            &mut flow_lattice,
            args.integrator,
            args.flow_epsilon,
            args.flow_t_max,
            1,
        );
        let flow_time = flow_start.elapsed().as_secs_f64();

        let t0_val = find_t0(&measurements);
        let w0_val = find_w0(&measurements);

        if let Some(t0) = t0_val {
            all_t0.push(t0);
        }
        if let Some(w0) = w0_val {
            all_w0.push(w0);
        }

        let e_final = measurements.last().map_or(0.0, |m| m.energy_density);

        println!(
            "    cfg {:3}/{}: ⟨P⟩={:.6} t₀={} w₀={} E(t_max)={:.4} ({:.1}s)",
            cfg + 1,
            args.n_configs,
            plaq,
            t0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            w0_val.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}")),
            e_final,
            flow_time,
        );

        let w_func = compute_w_function(&measurements);

        config_results.push(serde_json::json!({
            "config_idx": cfg,
            "plaquette": plaq,
            "t0": t0_val,
            "w0": w0_val,
            "e_final": e_final,
            "flow_time_s": flow_time,
            "flow_measurements": measurements.iter().map(|m| {
                serde_json::json!({
                    "t": m.t,
                    "e": m.energy_density,
                    "t2e": m.t2_e,
                    "plaq": m.plaquette,
                })
            }).collect::<Vec<_>>(),
            "w_function": w_func.iter().map(|(t, w)| {
                serde_json::json!({"t": t, "w": w})
            }).collect::<Vec<_>>(),
        }));
    }

    println!();
    println!("  ══════════════════════════════════════════════════");
    println!("  {} Gradient Flow Summary (β={:.4})", lat_label, args.beta);
    println!("  ══════════════════════════════════════════════════");

    let mean_plaq = all_plaq.iter().sum::<f64>() / all_plaq.len() as f64;
    println!("  ⟨P⟩ = {:.6} ± {:.6}", mean_plaq, std_dev(&all_plaq));

    if all_t0.is_empty() {
        println!("  t₀  = not found (increase --tmax)");
    } else {
        let mean_t0 = all_t0.iter().sum::<f64>() / all_t0.len() as f64;
        let std_t0 = std_dev(&all_t0);
        println!(
            "  t₀  = {:.4} ± {:.4} ({}/{} configs)",
            mean_t0,
            std_t0,
            all_t0.len(),
            args.n_configs
        );
    }

    if all_w0.is_empty() {
        println!("  w₀  = not found (increase --tmax)");
    } else {
        let mean_w0 = all_w0.iter().sum::<f64>() / all_w0.len() as f64;
        let std_w0 = std_dev(&all_w0);
        println!(
            "  w₀  = {:.4} ± {:.4} ({}/{} configs)",
            mean_w0,
            std_w0,
            all_w0.len(),
            args.n_configs
        );
    }

    let total_elapsed = total_t0.elapsed();
    println!("  Total wall time: {:.1}s", total_elapsed.as_secs_f64());

    if let Some(path) = args.output {
        let json = serde_json::json!({
            "lattice": lat_label,
            "dims": dims,
            "beta": args.beta,
            "volume": vol,
            "n_therm": args.n_therm,
            "n_configs": args.n_configs,
            "n_skip": args.n_skip,
            "flow_integrator": integrator_name,
            "flow_epsilon": args.flow_epsilon,
            "flow_t_max": args.flow_t_max,
            "mean_plaquette": mean_plaq,
            "mean_t0": if all_t0.is_empty() { None } else { Some(all_t0.iter().sum::<f64>() / all_t0.len() as f64) },
            "std_t0": if all_t0.is_empty() { None } else { Some(std_dev(&all_t0)) },
            "mean_w0": if all_w0.is_empty() { None } else { Some(all_w0.iter().sum::<f64>() / all_w0.len() as f64) },
            "std_w0": if all_w0.is_empty() { None } else { Some(std_dev(&all_w0)) },
            "total_wall_s": total_elapsed.as_secs_f64(),
            "configs": config_results,
        });

        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
            .expect("failed to write output");
        println!("  Results → {path}");
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
