// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chuna Engine: Analyze — ensemble-level statistical analysis of measurement JSONs.
//!
//! Reads a directory of per-config measurement JSONs (produced by `chuna_generate`
//! or `chuna_measure`) and computes ensemble-level statistics: jackknife errors,
//! integrated autocorrelation times, susceptibilities, and summary tables.
//!
//! # Usage
//!
//! ```bash
//! # Analyze an ensemble directory:
//! cargo run --release --bin chuna_analyze -- --dir=data/b6.0_L8/measurements/
//!
//! # Analyze with JSON output:
//! cargo run --release --bin chuna_analyze -- --dir=data/ --output=analysis.json
//!
//! # Analyze with TSV output (paper tables / numpy.loadtxt):
//! cargo run --release --bin chuna_analyze -- --dir=data/ --format=tsv --output=analysis.tsv
//!
//! # Analyze with asymmetric geometry support:
//! cargo run --release --bin chuna_analyze -- --dir=data/b6.0_16x32/measurements/
//! ```

use hotspring_barracuda::dag_provenance::{DagEvent, DagSession};
use hotspring_barracuda::lattice::correlator::{plaquette_susceptibility, polyakov_susceptibility};
use hotspring_barracuda::lattice::gradient_flow::topological_susceptibility;
use hotspring_barracuda::lattice::measurement::{
    ConfigMeasurement, RunManifest, estimate_tau_int, jackknife_error,
};
use hotspring_barracuda::primal_bridge::NucleusContext;
use hotspring_barracuda::receipt_signing;

use std::path::Path;

struct CliArgs {
    dir: String,
    output: Option<String>,
    skip: usize,
    format: OutputFormat,
}

#[derive(Clone, Copy, PartialEq)]
enum OutputFormat {
    Json,
    Tsv,
}

fn parse_args() -> CliArgs {
    let mut args = CliArgs {
        dir: ".".to_string(),
        output: None,
        skip: 0,
        format: OutputFormat::Json,
    };

    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--dir=") {
            args.dir = v.to_string();
        } else if let Some(v) = arg.strip_prefix("--output=") {
            args.output = Some(v.to_string());
        } else if let Some(v) = arg.strip_prefix("--skip=") {
            args.skip = v.parse().expect("--skip=N");
        } else if let Some(v) = arg.strip_prefix("--format=") {
            args.format = match v {
                "tsv" | "TSV" => OutputFormat::Tsv,
                "json" | "JSON" => OutputFormat::Json,
                _ => {
                    eprintln!("Unknown format: {v} (use json or tsv)");
                    std::process::exit(1);
                }
            };
        }
    }
    args
}

fn load_measurements(dir: &str) -> Vec<ConfigMeasurement> {
    let path = Path::new(dir);
    if !path.is_dir() {
        eprintln!("Not a directory: {dir}");
        std::process::exit(1);
    }

    let mut files: Vec<_> = std::fs::read_dir(path)
        .expect("read directory")
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "json"))
        .collect();

    files.sort_by_key(std::fs::DirEntry::file_name);

    let mut measurements = Vec::new();
    for entry in &files {
        let Ok(data) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(m) = serde_json::from_str::<ConfigMeasurement>(&data) else {
            continue;
        };
        measurements.push(m);
    }

    measurements
}

#[derive(serde::Serialize)]
struct EnsembleAnalysis {
    schema_version: String,
    ensemble_id: String,
    n_configs: usize,
    n_skipped: usize,

    plaquette: ObservableStats,
    polyakov_abs: ObservableStats,

    #[serde(skip_serializing_if = "Option::is_none")]
    topological_charge: Option<ObservableStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    topo_susceptibility: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    t0: Option<ObservableStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    w0: Option<ObservableStats>,

    #[serde(skip_serializing_if = "Option::is_none")]
    plaquette_susceptibility: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    polyakov_susceptibility: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    condensate: Option<ObservableStats>,

    #[serde(skip_serializing_if = "Option::is_none")]
    run: Option<serde_json::Value>,
}

#[derive(serde::Serialize)]
struct ObservableStats {
    mean: f64,
    error: f64,
    tau_int: f64,
    tau_int_error: f64,
    n_values: usize,
}

fn analyze_observable(values: &[f64]) -> ObservableStats {
    let (mean, error) = jackknife_error(values);
    let (tau, tau_err) = estimate_tau_int(values);
    ObservableStats {
        mean,
        error,
        tau_int: tau,
        tau_int_error: tau_err,
        n_values: values.len(),
    }
}

fn main() {
    let args = parse_args();
    let run_manifest = RunManifest::capture("chuna_analyze");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Engine: Ensemble Analysis                           ║");
    println!("║  hotSpring-barracuda — statistical analysis pipeline       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let nucleus = NucleusContext::detect();
    nucleus.print_banner();
    let mut run_manifest = if nucleus.any_alive() {
        run_manifest.with_nucleus(&nucleus)
    } else {
        run_manifest
    };

    let mut dag_session = DagSession::begin(&nucleus, "chuna_analyze");

    println!();
    println!("  Directory: {}", args.dir);

    let all_measurements = load_measurements(&args.dir);
    if all_measurements.is_empty() {
        eprintln!("  No valid measurement JSONs found in {}", args.dir);
        std::process::exit(1);
    }

    let measurements: Vec<_> = all_measurements.into_iter().skip(args.skip).collect();
    let n_total = measurements.len() + args.skip;
    let n_used = measurements.len();

    println!(
        "  Configs:   {} loaded, {} skipped, {} analyzed",
        n_total, args.skip, n_used
    );

    let ensemble_id = measurements
        .first()
        .map(|m| m.ensemble_id.clone())
        .unwrap_or_default();
    println!("  Ensemble:  {ensemble_id}");
    println!();

    let plaquettes: Vec<f64> = measurements.iter().map(|m| m.gauge.plaquette).collect();
    let poly_abs: Vec<f64> = measurements.iter().map(|m| m.gauge.polyakov_abs).collect();

    let topo_charges: Vec<f64> = measurements
        .iter()
        .filter_map(|m| m.topology.as_ref().map(|t| t.charge))
        .collect();

    let t0_values: Vec<f64> = measurements
        .iter()
        .filter_map(|m| m.flow.as_ref().and_then(|f| f.t0))
        .collect();

    let w0_values: Vec<f64> = measurements
        .iter()
        .filter_map(|m| m.flow.as_ref().and_then(|f| f.w0))
        .collect();

    let condensate_values: Vec<f64> = measurements
        .iter()
        .filter_map(|m| m.fermion.as_ref().map(|f| f.chiral_condensate))
        .collect();

    let plaq_stats = analyze_observable(&plaquettes);
    let poly_stats = analyze_observable(&poly_abs);

    println!("═══ Gauge Observables ═══");
    println!(
        "  ⟨P⟩       = {:.8} ± {:.2e}  (τ_int={:.1})",
        plaq_stats.mean, plaq_stats.error, plaq_stats.tau_int
    );
    println!(
        "  |L|       = {:.6} ± {:.2e}  (τ_int={:.1})",
        poly_stats.mean, poly_stats.error, poly_stats.tau_int
    );

    let vol: usize = measurements.first().map_or(1, |m| {
        let lfn = &m.ildg_lfn;
        lfn.split('/')
            .find_map(|s| s.strip_prefix("L"))
            .and_then(|s| s.parse::<usize>().ok())
            .map_or(1, |l| l * l * l * l)
    });

    let plaq_chi = if plaquettes.len() >= 2 {
        Some(plaquette_susceptibility(&plaquettes, vol.max(1)))
    } else {
        None
    };
    let poly_chi = if poly_abs.len() >= 2 {
        Some(polyakov_susceptibility(&poly_abs, vol.max(1)))
    } else {
        None
    };

    if let Some(chi) = plaq_chi {
        println!("  χ_P       = {chi:.6e}");
    }
    if let Some(chi) = poly_chi {
        println!("  χ_L       = {chi:.6e}");
    }

    let topo_stats = if topo_charges.len() >= 2 {
        let stats = analyze_observable(&topo_charges);
        println!();
        println!("═══ Topology ═══");
        println!(
            "  ⟨Q⟩       = {:.4} ± {:.4}  (τ_int={:.1})",
            stats.mean, stats.error, stats.tau_int
        );
        Some(stats)
    } else {
        None
    };

    let topo_chi = if topo_charges.len() >= 2 {
        let chi = topological_susceptibility(&topo_charges, vol.max(1));
        println!("  χ_t       = {chi:.6e}");
        Some(chi)
    } else {
        None
    };

    let t0_stats = if t0_values.len() >= 2 {
        let stats = analyze_observable(&t0_values);
        println!();
        println!("═══ Scale Setting ═══");
        println!(
            "  t₀        = {:.6} ± {:.2e}  (τ_int={:.1})",
            stats.mean, stats.error, stats.tau_int
        );
        Some(stats)
    } else {
        None
    };

    let w0_stats = if w0_values.len() >= 2 {
        let stats = analyze_observable(&w0_values);
        println!(
            "  w₀        = {:.6} ± {:.2e}  (τ_int={:.1})",
            stats.mean, stats.error, stats.tau_int
        );
        Some(stats)
    } else {
        None
    };

    let condensate_stats = if condensate_values.len() >= 2 {
        let stats = analyze_observable(&condensate_values);
        println!();
        println!("═══ Fermion ═══");
        println!(
            "  ⟨ψ̄ψ⟩     = {:.6e} ± {:.2e}  (τ_int={:.1})",
            stats.mean, stats.error, stats.tau_int
        );
        Some(stats)
    } else {
        None
    };

    println!();

    let analysis = EnsembleAnalysis {
        schema_version: "1.0".to_string(),
        ensemble_id,
        n_configs: n_used,
        n_skipped: args.skip,
        plaquette: plaq_stats,
        polyakov_abs: poly_stats,
        topological_charge: topo_stats,
        topo_susceptibility: topo_chi,
        t0: t0_stats,
        w0: w0_stats,
        plaquette_susceptibility: plaq_chi,
        polyakov_susceptibility: poly_chi,
        condensate: condensate_stats,
        run: serde_json::from_str(&run_manifest.to_json_value()).ok(),
    };

    // DAG event for the analysis phase
    if let Some(ref mut dag) = dag_session {
        dag.append(
            &nucleus,
            DagEvent {
                phase: "analyze".to_string(),
                input_hash: None,
                output_hash: None,
                wall_seconds: 0.0,
                summary: serde_json::json!({
                    "n_configs": n_used,
                    "plaquette_mean": analysis.plaquette.mean,
                }),
            },
        );
    }

    // Finalize DAG session
    if let Some(dag) = dag_session {
        let prov = dag.dehydrate(&nucleus);
        run_manifest.set_dag_provenance(&prov);
    }

    let json = serde_json::to_string_pretty(&analysis).expect("serialize");

    if args.format == OutputFormat::Tsv {
        let tsv = analysis_to_tsv(&analysis);
        if let Some(ref path) = args.output {
            std::fs::write(path, &tsv).expect("write TSV output");
            println!("  TSV → {path}");
        } else {
            print!("{tsv}");
        }
    } else if let Some(ref path) = args.output {
        std::fs::write(path, &json).expect("write output");

        // Sign the analysis receipt if bearDog is available
        if let Ok(mut receipt_val) = serde_json::from_str::<serde_json::Value>(&json) {
            receipt_signing::sign_and_embed(&nucleus, &mut receipt_val, std::path::Path::new(path));
        }

        println!("  Results → {path}");
    } else {
        println!("{json}");
    }
}

fn analysis_to_tsv(a: &EnsembleAnalysis) -> String {
    let mut lines = Vec::new();
    lines.push("observable\tmean\terror\ttau_int\ttau_int_error\tn_values".to_string());

    fn push_obs(lines: &mut Vec<String>, name: &str, s: &ObservableStats) {
        lines.push(format!(
            "{}\t{:.12e}\t{:.6e}\t{:.4}\t{:.4}\t{}",
            name, s.mean, s.error, s.tau_int, s.tau_int_error, s.n_values
        ));
    }

    push_obs(&mut lines, "plaquette", &a.plaquette);
    push_obs(&mut lines, "polyakov_abs", &a.polyakov_abs);

    if let Some(ref s) = a.topological_charge {
        push_obs(&mut lines, "topological_charge", s);
    }
    if let Some(ref s) = a.t0 {
        push_obs(&mut lines, "t0", s);
    }
    if let Some(ref s) = a.w0 {
        push_obs(&mut lines, "w0", s);
    }
    if let Some(ref s) = a.condensate {
        push_obs(&mut lines, "condensate", s);
    }
    if let Some(v) = a.topo_susceptibility {
        lines.push(format!("topo_susceptibility\t{v:.12e}\t0\t0\t0\t0"));
    }
    if let Some(v) = a.plaquette_susceptibility {
        lines.push(format!("plaquette_susceptibility\t{v:.12e}\t0\t0\t0\t0"));
    }
    if let Some(v) = a.polyakov_susceptibility {
        lines.push(format!("polyakov_susceptibility\t{v:.12e}\t0\t0\t0\t0"));
    }

    lines.push(String::new());
    lines.join("\n")
}
