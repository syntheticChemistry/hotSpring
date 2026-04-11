// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chuna Engine: Flow — run gradient flow on gauge configurations.
//!
//! Loads ILDG/LIME gauge configurations (ours or external MILC) and runs
//! gradient flow with selectable integrator. Reports t₀, w₀, topological
//! charge, and the full flow curve. Designed as the measurement front-end
//! for Chuna's flow optimization research.
//!
//! # Usage
//!
//! ```bash
//! # Flow a single ILDG config with W7 integrator:
//! cargo run --release --bin chuna_flow -- \
//!   --input=data/conf_000100.lime --integrator=w7 --tmax=4.0 --eps=0.01
//!
//! # Flow all configs in a directory:
//! cargo run --release --bin chuna_flow -- \
//!   --dir=data/b6.0_L8/ --integrator=ck4 --tmax=8.0 --eps=0.005
//!
//! # Compare integrators on same config (quick benchmark):
//! cargo run --release --bin chuna_flow -- \
//!   --input=data/conf_000100.lime --compare-all --tmax=4.0
//! ```

use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, find_t0, find_w0, run_flow, topological_charge,
};
use hotspring_barracuda::lattice::ildg::read_gauge_config_file;
use hotspring_barracuda::lattice::measurement::{
    ConfigMeasurement, FlowPoint, FlowResults, GaugeObservables, ImplementationInfo, RunManifest,
    TopologyResults,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::validation::TelemetryWriter;

use std::time::Instant;

struct CliArgs {
    input: Option<String>,
    dir: Option<String>,
    integrator: String,
    tmax: f64,
    eps: f64,
    compare_all: bool,
    outdir: Option<String>,
    telemetry: Option<String>,
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        input: None,
        dir: None,
        integrator: "w7".to_string(),
        tmax: 4.0,
        eps: 0.01,
        compare_all: false,
        outdir: None,
        telemetry: None,
    };

    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--input=") {
            args.input = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--dir=") {
            args.dir = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--integrator=") {
            args.integrator = v.to_string();
        } else if let Some(v) = arg.strip_prefix("--tmax=") {
            args.tmax = v.parse().expect("--tmax=F");
        } else if let Some(v) = arg.strip_prefix("--eps=") {
            args.eps = v.parse().expect("--eps=F");
        } else if arg == "--compare-all" {
            args.compare_all = true;
        } else if let Some(v) = arg.strip_prefix("--outdir=") {
            args.outdir = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--telemetry=") {
            args.telemetry = Some(v.to_string());
        }
    }
    args
}

fn parse_integrator(name: &str) -> FlowIntegrator {
    match name.to_lowercase().as_str() {
        "euler" => FlowIntegrator::Euler,
        "rk2" => FlowIntegrator::Rk2,
        "w6" | "luscher" | "rk3" => FlowIntegrator::Rk3Luscher,
        "w7" | "chuna" | "lscfrk3w7" => FlowIntegrator::Lscfrk3w7,
        "ck4" | "lscfrk4ck" | "ck45" => FlowIntegrator::Lscfrk4ck,
        _ => {
            eprintln!("Unknown integrator: {name}. Options: euler, rk2, w6, w7, ck4");
            std::process::exit(1);
        }
    }
}

fn integrator_name(int: FlowIntegrator) -> &'static str {
    match int {
        FlowIntegrator::Euler => "Euler",
        FlowIntegrator::Rk2 => "RK2",
        FlowIntegrator::Rk3Luscher => "LSCFRK3W6 (Lüscher)",
        FlowIntegrator::Lscfrk3w7 => "LSCFRK3W7 (Chuna)",
        FlowIntegrator::Lscfrk4ck => "LSCFRK4CK (CK45)",
    }
}

fn main() {
    let args = parse_args();
    let run_manifest = RunManifest::capture("chuna_flow");
    let mut telemetry = match &args.telemetry {
        Some(p) => TelemetryWriter::new(p),
        None => TelemetryWriter::disabled(),
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Engine: Gradient Flow Measurement                   ║");
    println!("║  hotSpring-barracuda — Bazavov & Chuna (2021) integrators  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let configs = collect_configs(&args);
    if configs.is_empty() {
        eprintln!("No input configs. Use --input=FILE.lime or --dir=DIR/");
        std::process::exit(1);
    }

    if let Some(ref od) = args.outdir {
        std::fs::create_dir_all(od).expect("create output dir");
    }

    for path in &configs {
        println!("\n═══ Loading: {path} ═══");
        let load_start = Instant::now();
        let (lattice, meta) = read_gauge_config_file(path).expect("read ILDG config");
        let [nx, ny, nz, nt] = lattice.dims;
        println!(
            "  Loaded: {}×{}×{}×{}, β={:.4}, plaq={:.6} ({:.2}s)",
            nx,
            ny,
            nz,
            nt,
            lattice.beta,
            lattice.average_plaquette(),
            load_start.elapsed().as_secs_f64()
        );

        if args.compare_all {
            run_comparison(
                &lattice,
                &args,
                &meta.lfn,
                &meta.ensemble_id,
                meta.trajectory,
            );
        } else {
            let int = parse_integrator(&args.integrator);
            let mut measurement = run_single_flow(
                &lattice,
                int,
                args.eps,
                args.tmax,
                &meta.lfn,
                &meta.ensemble_id,
                meta.trajectory,
                &mut telemetry,
            );
            measurement.implementation = Some(ImplementationInfo::auto_detect_cpu_only());
            measurement.run = Some(run_manifest.clone());
            if let Some(ref od) = args.outdir {
                let fname = format!("{od}/flow_{:06}.json", meta.trajectory);
                std::fs::write(&fname, measurement.to_json()).expect("write flow result");
                println!("  → Saved: {fname}");
            }
        }
    }
}

fn collect_configs(args: &CliArgs) -> Vec<String> {
    let mut paths = Vec::new();

    if let Some(ref input) = args.input {
        paths.push(input.clone());
    }

    if let Some(ref dir) = args.dir {
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut lime_files: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path().display().to_string())
                .filter(|p| p.ends_with(".lime"))
                .collect();
            lime_files.sort();
            paths.extend(lime_files);
        }
    }

    paths
}

fn run_single_flow(
    lattice: &Lattice,
    integrator: FlowIntegrator,
    eps: f64,
    tmax: f64,
    lfn: &str,
    ensemble_id: &str,
    trajectory: usize,
    telemetry: &mut TelemetryWriter,
) -> ConfigMeasurement {
    let flow_start = Instant::now();

    let mut flow_lat = lattice.clone();
    let measure_interval = (0.05 / eps).max(1.0) as usize;
    let results = run_flow(&mut flow_lat, integrator, eps, tmax, measure_interval);
    let t0 = find_t0(&results);
    let w0 = find_w0(&results);
    let q = topological_charge(&flow_lat);
    let flow_secs = flow_start.elapsed().as_secs_f64();

    for fm in &results {
        telemetry.log("flow", &format!("E(t={:.4})", fm.t), fm.energy_density);
        telemetry.log("flow", &format!("t2E(t={:.4})", fm.t), fm.t2_e);
    }
    if let Some(t0_val) = t0 {
        telemetry.log("flow", "t0", t0_val);
    }
    if let Some(w0_val) = w0 {
        telemetry.log("flow", "w0", w0_val);
    }
    telemetry.log("flow", "Q", q);

    println!(
        "  {} (ε={eps}): t₀={}, w₀={}, Q={q:.3} ({flow_secs:.2}s)",
        integrator_name(integrator),
        t0.map_or("N/A".to_string(), |v| format!("{v:.6}")),
        w0.map_or("N/A".to_string(), |v| format!("{v:.6}")),
    );

    let plaq = lattice.average_plaquette();
    let ns = [lattice.dims[0], lattice.dims[1], lattice.dims[2]];
    let spatial_vol = ns[0] * ns[1] * ns[2];
    let (mut pre, mut pim) = (0.0, 0.0);
    for ix in 0..ns[0] {
        for iy in 0..ns[1] {
            for iz in 0..ns[2] {
                let l = lattice.polyakov_loop([ix, iy, iz]);
                pre += l.re;
                pim += l.im;
            }
        }
    }
    pre /= spatial_vol as f64;
    pim /= spatial_vol as f64;

    let mut meas = ConfigMeasurement::new(ensemble_id, trajectory, lfn);
    meas.gauge = GaugeObservables {
        plaquette: plaq,
        polyakov_abs: (pre * pre + pim * pim).sqrt(),
        polyakov_re: pre,
        polyakov_im: pim,
        action_density: 6.0 * (1.0 - plaq),
    };
    meas.flow = Some(FlowResults {
        integrator: format!("{integrator:?}"),
        epsilon: eps,
        t_max: tmax,
        t0,
        w0,
        flow_curve: results
            .iter()
            .map(|fm| FlowPoint {
                t: fm.t,
                energy_density: fm.energy_density,
                t2_e: fm.t2_e,
            })
            .collect(),
    });
    meas.topology = Some(TopologyResults {
        charge: q,
        flow_time: tmax,
    });
    meas.wall_seconds = flow_secs;
    meas
}

fn run_comparison(
    lattice: &Lattice,
    args: &CliArgs,
    _lfn: &str,
    _ensemble_id: &str,
    _trajectory: usize,
) {
    println!("\n  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!(
        "  │  Integrator Comparison (same config, ε={}, t_max={})              │",
        args.eps, args.tmax
    );
    println!("  ├──────────────────────┬──────────┬──────────┬─────────┬────────────────┤");
    println!("  │ Integrator           │    t₀    │    w₀    │    Q    │   Time (s)     │");
    println!("  ├──────────────────────┼──────────┼──────────┼─────────┼────────────────┤");

    let integrators = [
        ("Euler", FlowIntegrator::Euler),
        ("RK2", FlowIntegrator::Rk2),
        ("LSCFRK3W6 (Lüscher)", FlowIntegrator::Rk3Luscher),
        ("LSCFRK3W7 (Chuna)", FlowIntegrator::Lscfrk3w7),
        ("LSCFRK4CK (CK45)", FlowIntegrator::Lscfrk4ck),
    ];

    for (name, int) in &integrators {
        let start = Instant::now();
        let mut flow_lat = lattice.clone();
        let measure_interval = (0.05 / args.eps).max(1.0) as usize;
        let results = run_flow(&mut flow_lat, *int, args.eps, args.tmax, measure_interval);
        let t0 = find_t0(&results);
        let w0 = find_w0(&results);
        let q = topological_charge(&flow_lat);
        let secs = start.elapsed().as_secs_f64();

        let t0s = t0.map_or("   N/A  ".to_string(), |v| format!("{v:8.5}"));
        let w0s = w0.map_or("   N/A  ".to_string(), |v| format!("{v:8.5}"));
        println!("  │ {name:<20} │ {t0s} │ {w0s} │ {q:>7.3} │ {secs:>14.3} │");
    }
    println!("  └──────────────────────┴──────────┴──────────┴─────────┴────────────────┘");
}
