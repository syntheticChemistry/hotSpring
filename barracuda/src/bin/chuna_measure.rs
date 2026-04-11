// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chuna Engine: Measure — compute observables on gauge configurations.
//!
//! Loads ILDG/LIME gauge configurations and computes a comprehensive set
//! of observables, writing formalized JSON measurements compatible with
//! the Bazavov/MILC analysis ecosystem.
//!
//! Observables:
//!   - Gauge: plaquette, Polyakov loop, action density
//!   - Wilson loops: W(R,T) grid up to R_max
//!   - Gradient flow: t₀, w₀, flow curve, topological charge
//!   - Fermion: chiral condensate (stochastic estimator)
//!
//! # Usage
//!
//! ```bash
//! # Measure a single ILDG config:
//! cargo run --release --bin chuna_measure -- \
//!   --input=data/conf_000100.lime --mass=0.1 --outdir=measurements/
//!
//! # Batch measure all configs in a directory:
//! cargo run --release --bin chuna_measure -- \
//!   --dir=data/b6.0_L8/ --mass=0.1 --wilson-rmax=4 --no-fermion
//!
//! # HVP with CG convergence history (for solver comparison):
//! cargo run --release --bin chuna_measure -- \
//!   --dir=data/ --hvp --cg-history --outdir=measurements/
//! # produces cg_history_NNNNNN.tsv per config (iteration, residual)
//! ```

use hotspring_barracuda::dag_provenance::{DagEvent, DagSession};
use hotspring_barracuda::lattice::correlator::{
    chiral_condensate_stochastic, hvp_integral, point_propagator_correlator,
    point_propagator_correlator_with_history,
};
use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, find_t0, find_w0, run_flow, topological_charge,
};
use hotspring_barracuda::lattice::ildg::read_gauge_config_file;
use hotspring_barracuda::lattice::measurement::{
    ConfigMeasurement, FermionObservables, FlowPoint, FlowResults, GaugeObservables, HvpResults,
    ImplementationInfo, RunManifest, TopologyResults, WilsonLoopEntry, min_spatial_dim,
};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::receipt_signing;
use hotspring_barracuda::validation::TelemetryWriter;

use std::time::Instant;

struct CliArgs {
    input: Option<String>,
    dir: Option<String>,
    outdir: String,
    wilson_rmax: usize,
    flow: bool,
    flow_tmax: f64,
    flow_eps: f64,
    fermion: bool,
    mass: f64,
    cg_tol: f64,
    cg_max_iter: usize,
    n_sources: usize,
    seed: u64,
    telemetry: Option<String>,
    hvp: bool,
    cg_history: bool,
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        input: None,
        dir: None,
        outdir: "measurements".to_string(),
        wilson_rmax: 4,
        flow: true,
        flow_tmax: 4.0,
        flow_eps: 0.01,
        fermion: true,
        mass: 0.1,
        cg_tol: 1e-8,
        cg_max_iter: 5000,
        n_sources: 10,
        seed: 42,
        telemetry: None,
        hvp: false,
        cg_history: false,
    };

    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--input=") {
            args.input = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--dir=") {
            args.dir = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--outdir=") {
            args.outdir = v.to_string();
        } else if let Some(v) = arg.strip_prefix("--wilson-rmax=") {
            args.wilson_rmax = v.parse().expect("--wilson-rmax=N");
        } else if arg == "--no-flow" {
            args.flow = false;
        } else if let Some(v) = arg.strip_prefix("--flow-tmax=") {
            args.flow_tmax = v.parse().expect("--flow-tmax=F");
        } else if let Some(v) = arg.strip_prefix("--flow-eps=") {
            args.flow_eps = v.parse().expect("--flow-eps=F");
        } else if arg == "--no-fermion" {
            args.fermion = false;
        } else if let Some(v) = arg.strip_prefix("--mass=") {
            args.mass = v.parse().expect("--mass=F");
        } else if let Some(v) = arg.strip_prefix("--cg-tol=") {
            args.cg_tol = v.parse().expect("--cg-tol=F");
        } else if let Some(v) = arg.strip_prefix("--n-sources=") {
            args.n_sources = v.parse().expect("--n-sources=N");
        } else if let Some(v) = arg.strip_prefix("--seed=") {
            args.seed = v.parse().expect("--seed=N");
        } else if let Some(v) = arg.strip_prefix("--telemetry=") {
            args.telemetry = Some(v.to_string());
        } else if arg == "--hvp" {
            args.hvp = true;
        } else if arg == "--cg-history" {
            args.cg_history = true;
        }
    }
    args
}

fn main() {
    let args = parse_args();
    let run_manifest = RunManifest::capture("chuna_measure");
    let mut telemetry = match &args.telemetry {
        Some(p) => TelemetryWriter::new(p),
        None => TelemetryWriter::disabled(),
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Engine: Observable Measurement                      ║");
    println!("║  hotSpring-barracuda — formalized JSON output              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let nucleus = NucleusContext::detect();
    nucleus.print_banner();
    let mut run_manifest = if nucleus.any_alive() {
        run_manifest.with_nucleus(&nucleus)
    } else {
        run_manifest
    };

    let mut dag_session = DagSession::begin(&nucleus, "chuna_measure");

    let configs = collect_configs(&args);
    if configs.is_empty() {
        eprintln!("No input configs. Use --input=FILE.lime or --dir=DIR/");
        std::process::exit(1);
    }

    std::fs::create_dir_all(&args.outdir).expect("create output dir");
    println!("  Output: {}", args.outdir);
    println!(
        "  Wilson loops: R_max={}, Flow: {}, Fermion: {}",
        args.wilson_rmax,
        if args.flow { "yes" } else { "no" },
        if args.fermion {
            format!("yes (m={}, {} sources)", args.mass, args.n_sources)
        } else {
            "no".to_string()
        }
    );

    let total_start = Instant::now();

    for (i, path) in configs.iter().enumerate() {
        let cfg_start = Instant::now();
        println!("\n═══ [{}/{}] {path} ═══", i + 1, configs.len());

        let (lattice, meta) = read_gauge_config_file(path).expect("read ILDG config");
        let [nx, ny, nz, nt] = lattice.dims;
        println!(
            "  Loaded: {}×{}×{}×{}, β={:.4}",
            nx, ny, nz, nt, lattice.beta
        );

        let mut measurement = measure_config(
            &lattice,
            &meta.ensemble_id,
            meta.trajectory,
            &meta.lfn,
            &args,
            &mut telemetry,
        );
        measurement.run = Some(run_manifest.clone());

        let meas_path = format!("{}/meas_{:06}.json", args.outdir, meta.trajectory);
        let meas_json = measurement.to_json();
        std::fs::write(&meas_path, &meas_json).expect("write measurement");

        // Sign each measurement receipt if bearDog is available
        if let Ok(mut receipt_val) = serde_json::from_str::<serde_json::Value>(&meas_json) {
            receipt_signing::sign_and_embed(
                &nucleus,
                &mut receipt_val,
                std::path::Path::new(&meas_path),
            );
        }

        // DAG event per config measurement
        if let Some(ref mut dag) = dag_session {
            dag.append(
                &nucleus,
                DagEvent {
                    phase: "measure".to_string(),
                    input_hash: None,
                    output_hash: None,
                    wall_seconds: cfg_start.elapsed().as_secs_f64(),
                    summary: serde_json::json!({
                        "trajectory": meta.trajectory,
                        "plaquette": measurement.gauge.plaquette,
                    }),
                },
            );
        }

        println!(
            "  → {meas_path} ({:.2}s)",
            cfg_start.elapsed().as_secs_f64()
        );
    }

    // Finalize DAG session
    if let Some(dag) = dag_session {
        let prov = dag.dehydrate(&nucleus);
        run_manifest.set_dag_provenance(&prov);
    }

    println!(
        "\n═══ Complete: {} configs measured in {:.1}s ═══",
        configs.len(),
        total_start.elapsed().as_secs_f64()
    );
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

fn measure_config(
    lattice: &Lattice,
    ensemble_id: &str,
    trajectory: usize,
    lfn: &str,
    args: &CliArgs,
    telemetry: &mut TelemetryWriter,
) -> ConfigMeasurement {
    let total_start = Instant::now();
    let mut meas = ConfigMeasurement::new(ensemble_id, trajectory, lfn);
    meas.implementation = Some(ImplementationInfo::auto_detect_cpu_only());

    // Gauge observables
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
    let poly_abs = (pre * pre + pim * pim).sqrt();

    meas.gauge = GaugeObservables {
        plaquette: plaq,
        polyakov_abs: poly_abs,
        polyakov_re: pre,
        polyakov_im: pim,
        action_density: 6.0 * (1.0 - plaq),
    };
    telemetry.log("gauge", "plaquette", plaq);
    telemetry.log("gauge", "polyakov_abs", poly_abs);
    telemetry.log("gauge", "action_density", 6.0 * (1.0 - plaq));
    println!(
        "  Gauge:   ⟨P⟩={plaq:.6}  |L|={poly_abs:.4}  S_d={:.4}",
        6.0 * (1.0 - plaq)
    );

    // Wilson loops
    let r_max = args.wilson_rmax.min(min_spatial_dim(lattice.dims) / 2);
    let t_max_wl = r_max.min(lattice.dims[3] / 2);
    if r_max > 0 {
        let mut wl_entries = Vec::new();
        for r in 1..=r_max {
            for t in 1..=t_max_wl {
                let val = lattice.spatial_temporal_wilson_loop(r, t);
                wl_entries.push(WilsonLoopEntry { r, t, value: val });
            }
        }
        println!(
            "  Wilson:  {}×{} grid, W(1,1)={:.6}",
            r_max,
            t_max_wl,
            wl_entries.first().map_or(0.0, |e| e.value)
        );
        meas.wilson_loops = Some(wl_entries);
    }

    // Gradient flow
    if args.flow {
        let flow_start = Instant::now();
        let mut flow_lat = lattice.clone();
        let measure_interval = (0.05 / args.flow_eps).max(1.0) as usize;
        let results = run_flow(
            &mut flow_lat,
            FlowIntegrator::Lscfrk3w7,
            args.flow_eps,
            args.flow_tmax,
            measure_interval,
        );
        let t0 = find_t0(&results);
        let w0 = find_w0(&results);
        let q = topological_charge(&flow_lat);

        meas.flow = Some(FlowResults {
            integrator: "Lscfrk3w7".to_string(),
            epsilon: args.flow_eps,
            t_max: args.flow_tmax,
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
            flow_time: args.flow_tmax,
        });

        if let Some(t0_val) = t0 {
            telemetry.log("flow", "t0", t0_val);
        }
        if let Some(w0_val) = w0 {
            telemetry.log("flow", "w0", w0_val);
        }
        telemetry.log("flow", "Q", q);

        let t0s = t0.map_or("N/A".to_string(), |v| format!("{v:.4}"));
        let w0s = w0.map_or("N/A".to_string(), |v| format!("{v:.4}"));
        println!(
            "  Flow:    t₀={t0s}  w₀={w0s}  Q={q:.3}  ({:.2}s)",
            flow_start.elapsed().as_secs_f64()
        );
    }

    // Fermion observables
    if args.fermion {
        let ferm_start = Instant::now();
        let result = chiral_condensate_stochastic(
            lattice,
            args.mass,
            args.cg_tol,
            args.cg_max_iter,
            args.n_sources,
            args.seed,
        );
        meas.fermion = Some(FermionObservables {
            chiral_condensate: result.condensate,
            chiral_condensate_error: result.error,
            n_sources: result.n_sources,
            mass: args.mass,
        });
        telemetry.log("fermion", "chiral_condensate", result.condensate);
        telemetry.log("fermion", "chiral_condensate_error", result.error);
        println!(
            "  Fermion: ⟨ψ̄ψ⟩={:.6} ± {:.6} (m={}, {} src, {:.2}s)",
            result.condensate,
            result.error,
            args.mass,
            args.n_sources,
            ferm_start.elapsed().as_secs_f64()
        );
    }

    // HVP correlator
    if args.hvp {
        let hvp_start = Instant::now();
        let corr_result = if args.cg_history {
            point_propagator_correlator_with_history(
                lattice,
                args.mass,
                args.cg_tol,
                args.cg_max_iter,
            )
        } else {
            point_propagator_correlator(lattice, args.mass, args.cg_tol, args.cg_max_iter)
        };
        let a_mu = hvp_integral(&corr_result.correlator);
        telemetry.log("hvp", "hvp_integral", a_mu);
        telemetry.log("hvp", "cg_iterations", corr_result.cg.iterations as f64);
        meas.hvp = Some(HvpResults {
            mass: args.mass,
            cg_residual: corr_result.cg.final_residual,
            cg_iterations: corr_result.cg.iterations,
            correlator: corr_result.correlator.clone(),
            hvp_integral: a_mu,
        });
        println!(
            "  HVP:    a_μ={:.6e}  ({} CG iters, {:.2}s)",
            a_mu,
            corr_result.cg.iterations,
            hvp_start.elapsed().as_secs_f64()
        );

        // Write CG convergence history if requested
        if args.cg_history && !corr_result.cg.residual_history.is_empty() {
            let hist_path = format!("{}/cg_history_{:06}.tsv", args.outdir, trajectory);
            let mut hist_lines = Vec::with_capacity(corr_result.cg.residual_history.len() + 1);
            hist_lines.push("iteration\tresidual".to_string());
            for (i, res) in corr_result.cg.residual_history.iter().enumerate() {
                hist_lines.push(format!("{i}\t{res:.12e}"));
            }
            std::fs::write(&hist_path, hist_lines.join("\n") + "\n").expect("write CG history");
            println!(
                "  CG history → {hist_path} ({} iters)",
                corr_result.cg.residual_history.len()
            );
        }
    }

    meas.wall_seconds = total_start.elapsed().as_secs_f64();
    meas
}
