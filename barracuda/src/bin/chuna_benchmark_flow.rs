// SPDX-License-Identifier: AGPL-3.0-only

//! Chuna Engine: Benchmark Flow — integrator efficiency comparison workbench.
//!
//! This is Chuna's primary research tool. Compares gradient flow integrators
//! (Euler, RK2, W6/Lüscher, W7/Chuna, CK4/Carpenter-Kennedy) on the same
//! gauge configuration across multiple step sizes, reporting:
//!
//!   - Accuracy: t₀, w₀, Q at each step size vs reference (smallest ε)
//!   - Efficiency: wall time, force evaluations, cost per unit accuracy
//!   - Convergence order: log-log error vs step size
//!
//! This directly extends the analysis from Bazavov & Chuna, arXiv:2101.05320,
//! and provides the workbench for optimizing integrator coefficients.
//!
//! # Usage
//!
//! ```bash
//! # Benchmark with default parameters:
//! cargo run --release --bin chuna_benchmark_flow -- \
//!   --input=data/conf_000100.lime
//!
//! # Custom step size range:
//! cargo run --release --bin chuna_benchmark_flow -- \
//!   --input=data/conf_000100.lime --eps-range=0.002,0.005,0.01,0.02,0.04
//!
//! # Quick self-contained run (generates its own config):
//! cargo run --release --bin chuna_benchmark_flow -- \
//!   --self-generate --lattice=8 --beta=6.0 --therm=100
//!
//! # Self-generate with asymmetric geometry:
//! cargo run --release --bin chuna_benchmark_flow -- \
//!   --self-generate --ns=16 --nt=32 --beta=6.0
//! ```

use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, LscfrkCoefficients, derive_lscfrk3, find_t0, find_w0, run_flow,
    run_flow_custom, topological_charge,
};
use hotspring_barracuda::lattice::hmc::{HmcConfig, IntegratorType, hmc_trajectory};
use hotspring_barracuda::lattice::ildg::read_gauge_config_file;
use hotspring_barracuda::lattice::measurement::{RunManifest, format_dims, parse_dims_from_args};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::TelemetryWriter;

use std::time::Instant;

struct CliArgs {
    input: Option<String>,
    self_generate: bool,
    dims: [usize; 4],
    beta: f64,
    therm: usize,
    tmax: f64,
    eps_range: Vec<f64>,
    outdir: Option<String>,
    custom_c2: Option<f64>,
    custom_c3: Option<f64>,
    telemetry: Option<String>,
}

fn parse_args() -> CliArgs {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let dims = parse_dims_from_args(&raw_args).unwrap_or([8, 8, 8, 8]);

    let mut args = CliArgs {
        input: None,
        self_generate: false,
        dims,
        beta: 6.0,
        therm: 100,
        tmax: 4.0,
        eps_range: vec![0.04, 0.02, 0.01, 0.005],
        outdir: None,
        custom_c2: None,
        custom_c3: None,
        telemetry: None,
    };

    for arg in &raw_args {
        if let Some(v) = arg.strip_prefix("--input=") {
            args.input = Some(v.to_string());
        } else if arg == "--self-generate" {
            args.self_generate = true;
        } else if let Some(v) = arg.strip_prefix("--beta=") {
            args.beta = v.parse().expect("--beta=F");
        } else if let Some(v) = arg.strip_prefix("--therm=") {
            args.therm = v.parse().expect("--therm=N");
        } else if let Some(v) = arg.strip_prefix("--tmax=") {
            args.tmax = v.parse().expect("--tmax=F");
        } else if let Some(v) = arg.strip_prefix("--eps-range=") {
            args.eps_range = v.split(',').map(|s| s.parse().expect("eps value")).collect();
        } else if let Some(v) = arg.strip_prefix("--outdir=") {
            args.outdir = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--custom-c2=") {
            args.custom_c2 = Some(v.parse().expect("--custom-c2=F"));
        } else if let Some(v) = arg.strip_prefix("--custom-c3=") {
            args.custom_c3 = Some(v.parse().expect("--custom-c3=F"));
        } else if let Some(v) = arg.strip_prefix("--telemetry=") {
            args.telemetry = Some(v.to_string());
        }
    }
    args
}

/// Number of force evaluations per flow step for each integrator.
fn force_evals(int: FlowIntegrator) -> usize {
    match int {
        FlowIntegrator::Euler => 1,
        FlowIntegrator::Rk2 => 2,
        FlowIntegrator::Rk3Luscher | FlowIntegrator::Lscfrk3w7 => 3,
        FlowIntegrator::Lscfrk4ck => 5,
    }
}

fn integrator_label(int: FlowIntegrator) -> &'static str {
    match int {
        FlowIntegrator::Euler => "Euler    (O1,1s)",
        FlowIntegrator::Rk2 => "RK2      (O2,2s)",
        FlowIntegrator::Rk3Luscher => "W6/Lüsch (O3,3s)",
        FlowIntegrator::Lscfrk3w7 => "W7/Chuna (O3,3s)",
        FlowIntegrator::Lscfrk4ck => "CK4      (O4,5s)",
    }
}

struct FlowBenchResult {
    integrator: FlowIntegrator,
    epsilon: f64,
    t0: Option<f64>,
    w0: Option<f64>,
    q: f64,
    wall_secs: f64,
    n_steps: usize,
    n_force_evals: usize,
}

fn main() {
    let args = parse_args();
    let run_manifest = RunManifest::capture("chuna_benchmark_flow");
    let mut telemetry = match &args.telemetry {
        Some(p) => TelemetryWriter::new(p),
        None => TelemetryWriter::disabled(),
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Engine: Flow Integrator Benchmark Workbench         ║");
    println!("║  hotSpring-barracuda — Bazavov & Chuna (2021)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let lattice = if let Some(ref path) = args.input {
        println!("\n  Loading: {path}");
        let (lat, _meta) = read_gauge_config_file(path).expect("read ILDG");
        let [nx, ny, nz, nt] = lat.dims;
        println!(
            "  Loaded: {}×{}×{}×{}, β={:.4}, ⟨P⟩={:.6}",
            nx, ny, nz, nt, lat.beta, lat.average_plaquette()
        );
        lat
    } else if args.self_generate {
        println!("\n  Self-generating {} config at β={:.2}...", format_dims(args.dims), args.beta);
        let dims = args.dims;
        let mut lat = Lattice::hot_start(dims, args.beta, 42);
        let mut hmc_cfg = HmcConfig {
            n_md_steps: 20,
            dt: 0.05,
            seed: 42,
            integrator: IntegratorType::Omelyan,
        };
        let therm_start = Instant::now();
        for _ in 0..args.therm {
            hmc_trajectory(&mut lat, &mut hmc_cfg);
        }
        println!(
            "  Thermalized: ⟨P⟩={:.6} ({} HMC, {:.1}s)",
            lat.average_plaquette(),
            args.therm,
            therm_start.elapsed().as_secs_f64()
        );
        lat
    } else {
        eprintln!("Provide --input=FILE.lime or --self-generate");
        std::process::exit(1);
    };

    if let Some(ref od) = args.outdir {
        std::fs::create_dir_all(od).expect("create output dir");
    }

    let integrators = [
        FlowIntegrator::Euler,
        FlowIntegrator::Rk2,
        FlowIntegrator::Rk3Luscher,
        FlowIntegrator::Lscfrk3w7,
        FlowIntegrator::Lscfrk4ck,
    ];

    // Sort eps_range descending (largest step first)
    let mut eps_range = args.eps_range.clone();
    eps_range.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Derive custom coefficients if requested
    let custom_coeffs: Option<LscfrkCoefficients> =
        if let (Some(c2), Some(c3)) = (args.custom_c2, args.custom_c3) {
            let (a, b) = derive_lscfrk3(c2, c3);
            println!("\n  Custom LSCFRK3 (c₂={c2}, c₃={c3}):");
            println!("    A = {:?}", a);
            println!("    B = {:?}", b);
            let a_static: &'static [f64] = Box::leak(a.to_vec().into_boxed_slice());
            let b_static: &'static [f64] = Box::leak(b.to_vec().into_boxed_slice());
            Some(LscfrkCoefficients {
                a: a_static,
                b: b_static,
            })
        } else {
            None
        };

    let n_integrators = integrators.len() + if custom_coeffs.is_some() { 1 } else { 0 };

    println!("\n  Step sizes: {:?}", eps_range);
    println!("  t_max: {}", args.tmax);
    println!("  Integrators: {} ({}custom)", n_integrators, if custom_coeffs.is_some() { "incl. " } else { "" });

    // Collect all results
    let mut all_results: Vec<FlowBenchResult> = Vec::new();

    for &eps in &eps_range {
        println!("\n══════════════════════════════════════════════════════════════");
        println!("  ε = {eps}");
        println!("══════════════════════════════════════════════════════════════");

        for &int in &integrators {
            let start = Instant::now();
            let mut flow_lat = lattice.clone();
            let measure_interval = (0.05 / eps).max(1.0) as usize;
            let results = run_flow(&mut flow_lat, int, eps, args.tmax, measure_interval);
            let t0 = find_t0(&results);
            let w0 = find_w0(&results);
            let q = topological_charge(&flow_lat);
            let wall_secs = start.elapsed().as_secs_f64();
            let n_steps = (args.tmax / eps).round() as usize;
            let n_force = n_steps * force_evals(int);

            all_results.push(FlowBenchResult {
                integrator: int,
                epsilon: eps,
                t0,
                w0,
                q,
                wall_secs,
                n_steps,
                n_force_evals: n_force,
            });

            let label = integrator_label(int);
            let section = format!("bench/{label}/eps={eps}");
            if let Some(t0_val) = t0 {
                telemetry.log(&section, "t0", t0_val);
            }
            if let Some(w0_val) = w0 {
                telemetry.log(&section, "w0", w0_val);
            }
            telemetry.log(&section, "Q", q);
            telemetry.log(&section, "wall_secs", wall_secs);
            telemetry.log(&section, "n_force_evals", n_force as f64);

            let t0s = t0.map_or("  N/A   ".to_string(), |v| format!("{v:8.5}"));
            let w0s = w0.map_or("  N/A   ".to_string(), |v| format!("{v:8.5}"));
            println!(
                "  {:<18}  t₀={t0s}  w₀={w0s}  Q={q:>7.3}  {n_force:>6} evals  {wall_secs:>7.3}s",
                label
            );
        }

        // Custom coefficients run
        if let Some(ref coeffs) = custom_coeffs {
            let start = Instant::now();
            let mut flow_lat = lattice.clone();
            let measure_interval = (0.05 / eps).max(1.0) as usize;
            let results =
                run_flow_custom(&mut flow_lat, coeffs, eps, args.tmax, measure_interval);
            let t0 = find_t0(&results);
            let w0 = find_w0(&results);
            let q = topological_charge(&flow_lat);
            let wall_secs = start.elapsed().as_secs_f64();
            let n_steps = (args.tmax / eps).round() as usize;
            let n_stages = coeffs.a.len();
            let n_force = n_steps * n_stages;

            all_results.push(FlowBenchResult {
                integrator: FlowIntegrator::Lscfrk3w7, // placeholder for custom
                epsilon: eps,
                t0,
                w0,
                q,
                wall_secs,
                n_steps,
                n_force_evals: n_force,
            });

            let t0s = t0.map_or("  N/A   ".to_string(), |v| format!("{v:8.5}"));
            let w0s = w0.map_or("  N/A   ".to_string(), |v| format!("{v:8.5}"));
            println!(
                "  {:<18}  t₀={t0s}  w₀={w0s}  Q={q:>7.3}  {n_force:>6} evals  {wall_secs:>7.3}s",
                "Custom (user)"
            );
        }
    }

    // Summary table: accuracy relative to finest step size
    let finest_eps = eps_range.last().copied().unwrap_or(0.005);
    println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Summary: Relative accuracy vs reference (ε={finest_eps})                        ║");
    println!("╠══════════════════╦══════╦══════════╦══════════╦══════════╦══════════╦════════════╣");
    println!("║ Integrator       ║  ε   ║ Δt₀/t₀  ║ Δw₀/w₀  ║   ΔQ     ║  Evals   ║  Time (s)  ║");
    println!("╠══════════════════╬══════╬══════════╬══════════╬══════════╬══════════╬════════════╣");

    for &int in &integrators {
        let ref_result = all_results
            .iter()
            .find(|r| r.integrator == int && (r.epsilon - finest_eps).abs() < 1e-10);
        let ref_t0 = ref_result.and_then(|r| r.t0);
        let ref_w0 = ref_result.and_then(|r| r.w0);
        let ref_q = ref_result.map(|r| r.q).unwrap_or(0.0);

        for res in all_results.iter().filter(|r| r.integrator == int) {
            let dt0 = match (res.t0, ref_t0) {
                (Some(a), Some(b)) if b.abs() > 1e-10 => format!("{:>8.2e}", (a - b).abs() / b),
                _ => "   ref  ".to_string(),
            };
            let dw0 = match (res.w0, ref_w0) {
                (Some(a), Some(b)) if b.abs() > 1e-10 => format!("{:>8.2e}", (a - b).abs() / b),
                _ => "   ref  ".to_string(),
            };
            let dq = if (res.epsilon - finest_eps).abs() < 1e-10 {
                "   ref  ".to_string()
            } else {
                format!("{:>8.3}", (res.q - ref_q).abs())
            };

            let label = if (res.epsilon - finest_eps).abs() < 1e-10 {
                format!("{} *", &integrator_label(int)[..16])
            } else {
                integrator_label(int)[..18].to_string()
            };

            println!(
                "║ {label:<18}║{:>5.3} ║ {dt0} ║ {dw0} ║ {dq} ║ {:>8} ║ {:>10.3} ║",
                res.epsilon, res.n_force_evals, res.wall_secs
            );
        }
    }
    println!("╚══════════════════╩══════╩══════════╩══════════╩══════════╩══════════╩════════════╝");

    // Convergence order analysis
    println!("\n═══ Convergence Order Analysis ═══");
    println!("  (log₂(error ratio) between successive step sizes → effective order)");
    println!();

    for &int in &integrators {
        let results_for_int: Vec<&FlowBenchResult> = all_results
            .iter()
            .filter(|r| r.integrator == int && r.t0.is_some())
            .collect();

        if results_for_int.len() < 2 {
            continue;
        }

        let ref_t0 = ref_result_for(int, finest_eps, &all_results)
            .and_then(|r| r.t0)
            .unwrap_or(0.0);

        if ref_t0.abs() < 1e-10 {
            continue;
        }

        print!("  {:<20}", integrator_label(int));
        let mut prev_err: Option<f64> = None;
        let mut prev_eps: Option<f64> = None;
        for res in &results_for_int {
            if (res.epsilon - finest_eps).abs() < 1e-10 {
                continue;
            }
            let err = (res.t0.unwrap_or(0.0) - ref_t0).abs() / ref_t0;
            if let (Some(pe), Some(peps)) = (prev_err, prev_eps) {
                if err > 1e-15 && pe > 1e-15 {
                    let order = (pe / err).ln() / (peps / res.epsilon).ln();
                    print!("  O≈{order:.1}(ε={:.3})", res.epsilon);
                }
            }
            prev_err = Some(err);
            prev_eps = Some(res.epsilon);
        }
        println!();
    }

    // Write JSON summary
    if let Some(ref od) = args.outdir {
        let json = build_json_summary(&all_results, finest_eps, args.tmax, &run_manifest);
        let path = format!("{od}/benchmark_flow.json");
        std::fs::write(&path, json).expect("write summary JSON");
        println!("\n  → Summary: {path}");
    }
    drop(telemetry);

    println!("\n═══ Complete ═══");
}

fn ref_result_for<'a>(
    int: FlowIntegrator,
    eps: f64,
    results: &'a [FlowBenchResult],
) -> Option<&'a FlowBenchResult> {
    results
        .iter()
        .find(|r| r.integrator == int && (r.epsilon - eps).abs() < 1e-10)
}

fn build_json_summary(results: &[FlowBenchResult], ref_eps: f64, tmax: f64, manifest: &RunManifest) -> String {
    let mut json = String::from("{\n  \"benchmark\": \"flow_integrator_comparison\",\n");
    json.push_str(&format!("  \"run\": {},\n", manifest.to_json_value()));
    json.push_str(&format!("  \"reference_epsilon\": {ref_eps},\n"));
    json.push_str(&format!("  \"t_max\": {tmax},\n"));
    json.push_str("  \"results\": [\n");

    for (i, res) in results.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!(
            "      \"integrator\": \"{:?}\",\n",
            res.integrator
        ));
        json.push_str(&format!("      \"epsilon\": {},\n", res.epsilon));
        json.push_str(&format!(
            "      \"t0\": {},\n",
            res.t0.map_or("null".to_string(), |v| format!("{v}"))
        ));
        json.push_str(&format!(
            "      \"w0\": {},\n",
            res.w0.map_or("null".to_string(), |v| format!("{v}"))
        ));
        json.push_str(&format!("      \"Q\": {},\n", res.q));
        json.push_str(&format!("      \"wall_seconds\": {},\n", res.wall_secs));
        json.push_str(&format!("      \"n_steps\": {},\n", res.n_steps));
        json.push_str(&format!(
            "      \"n_force_evals\": {}\n",
            res.n_force_evals
        ));
        json.push_str("    }");
        if i + 1 < results.len() {
            json.push(',');
        }
        json.push('\n');
    }

    json.push_str("  ]\n}\n");
    json
}
