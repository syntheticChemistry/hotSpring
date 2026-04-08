// SPDX-License-Identifier: AGPL-3.0-or-later

//! 3-Month Validation Matrix Runner
//!
//! Orchestrates the full validation matrix for Chuna collaboration:
//!
//! - **Month 1**: Quenched gradient flow ladder (8⁴→32⁴) + dynamical RHMC scaling
//! - **Month 2**: Observable expansion + beta scans + mass scans
//! - **Month 3**: Chuna-directed runs + artifact assembly
//!
//! Each cell in the matrix is a (lattice, beta, mass, Nf, observables) tuple.
//! The runner selects cells via CLI flags and produces structured JSON output.
//!
//! # Usage
//!
//! ```bash
//! # Run the full quenched ladder (Month 1, Weeks 1-2):
//! cargo run --release --bin validation_matrix -- --phase=quenched-ladder
//!
//! # Run dynamical scaling (Month 1, Weeks 3-4):
//! cargo run --release --bin validation_matrix -- --phase=dynamical-scaling
//!
//! # Run a single cell:
//! cargo run --release --bin validation_matrix -- --lattice=16 --beta=6.0 --mode=quenched
//!
//! # Run beta scan at volume (Month 2):
//! cargo run --release --bin validation_matrix -- --phase=beta-scan
//!
//! # Run mass scan (Month 2):
//! cargo run --release --bin validation_matrix -- --phase=mass-scan
//!
//! # Run everything (full matrix):
//! cargo run --release --bin validation_matrix -- --phase=all
//! ```

use hotspring_barracuda::bin_helpers::validation_matrix::{
    beta_scan_cells, custom_cell, dynamical_scaling_cells, mass_scan_cells, parse_args,
    print_planning_matrix, quenched_ladder_cells, run_cell, CliArgs, MatrixCell,
};
use hotspring_barracuda::lattice::measurement::RunManifest;
use hotspring_barracuda::validation::TelemetryWriter;

use std::time::Instant;

fn main() {
    let args = parse_args();
    let run_manifest = RunManifest::capture("validation_matrix");
    let mut telemetry = match &args.telemetry {
        Some(p) => TelemetryWriter::new(p),
        None => TelemetryWriter::disabled(),
    };

    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Validation Matrix — 3-Month Roadmap Runner                ║");
    eprintln!("║  hotSpring × Chuna — Ns³×Nt geometry + parameter mixing   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");
    eprintln!();
    eprintln!("  Phase: {}", args.phase);
    eprintln!("  Seed:  {}", args.seed);
    if let Some(ref l) = args.lattice_raw {
        eprintln!("  Geometry: ns={l}");
    }
    if let Some(ref nt) = args.nt_raw {
        eprintln!("  Geometry: nt={nt}");
    }
    if let Some(ref b) = args.beta_raw {
        eprintln!("  Params:   beta={b}");
    }
    if let Some(ref nf) = args.nf_raw {
        eprintln!("  Params:   nf={nf}");
    }
    if let Some(ref m) = args.mass_raw {
        eprintln!("  Params:   mass={m}");
    }
    eprintln!();

    if args.phase == "print-matrix" {
        print_planning_matrix();
        return;
    }

    let cells: Vec<MatrixCell> = match args.phase.as_str() {
        "quenched-ladder" => quenched_ladder_cells(),
        "dynamical-scaling" => dynamical_scaling_cells(),
        "beta-scan" => beta_scan_cells(args.first_lattice().unwrap_or(16)),
        "mass-scan" => mass_scan_cells(args.first_lattice().unwrap_or(16), args.first_beta().unwrap_or(6.0)),
        "custom" => custom_cell(&args),
        "all" => {
            let mut all = quenched_ladder_cells();
            all.extend(dynamical_scaling_cells());
            all.extend(beta_scan_cells(16));
            all.extend(mass_scan_cells(16, 6.0));
            all
        }
        other => {
            eprintln!("Unknown phase: {other}");
            eprintln!("Valid phases: quenched-ladder, dynamical-scaling, beta-scan, mass-scan, custom, all, print-matrix");
            std::process::exit(1);
        }
    };

    // Filter by lattice/beta if specified (uses first value from comma-separated list)
    let filter_lattice = args.first_lattice();
    let filter_beta = args.first_beta();
    let cells: Vec<MatrixCell> = if args.phase == "custom" {
        cells
    } else {
        cells
            .into_iter()
            .filter(|c| filter_lattice.is_none() || c.dims[0] == filter_lattice.unwrap())
            .filter(|c| filter_beta.is_none() || (c.beta - filter_beta.unwrap()).abs() < 1e-6)
            .collect()
    };

    eprintln!("  Matrix cells to run: {}", cells.len());
    for (i, cell) in cells.iter().enumerate() {
        eprintln!(
            "    [{:>2}] {} — {} β={:.2} {}",
            i + 1,
            cell.label,
            cell.lattice_label(),
            cell.beta,
            if cell.nf > 0 {
                format!("Nf={} m={:.3}", cell.nf, cell.mass.unwrap_or(0.0))
            } else {
                "quenched".to_string()
            }
        );
    }
    eprintln!();

    let total_start = Instant::now();
    let mut results = Vec::with_capacity(cells.len());

    for (i, cell) in cells.iter().enumerate() {
        eprintln!(
            "═══ Cell {}/{} ═══",
            i + 1,
            cells.len()
        );
        let result = run_cell(cell, args.seed + i as u64, args.max_flow_time);
        telemetry.log(&result.label, "plaquette", result.mean_plaquette);
        telemetry.log(&result.label, "acceptance", result.acceptance);
        if let Some(t0) = result.t0 {
            telemetry.log(&result.label, "t0", t0);
        }
        if let Some(w0) = result.w0 {
            telemetry.log(&result.label, "w0", w0);
        }
        if let Some(q) = result.topo_charge {
            telemetry.log(&result.label, "Q", q);
        }
        if let Some(pbp) = result.chiral_condensate {
            telemetry.log(&result.label, "pbp", pbp);
        }
        telemetry.log(&result.label, "wall_seconds", result.wall_seconds);
        results.push(result);
    }

    let total_wall = total_start.elapsed().as_secs_f64();

    // Summary table
    eprintln!();
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  Validation Matrix Summary");
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  {:>28} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "Label", "⟨P⟩", "σ(P)", "|L|", "t₀", "w₀", "time"
    );
    for r in &results {
        eprintln!(
            "  {:>28} {:>8.6} {:>10.2e} {:>8.4} {:>8} {:>8} {:>7.1}s",
            r.label,
            r.mean_plaquette,
            r.std_plaquette,
            r.mean_polyakov,
            r.t0.map(|v| format!("{v:.3}")).unwrap_or("—".into()),
            r.w0.map(|v| format!("{v:.3}")).unwrap_or("—".into()),
            r.wall_seconds,
        );
    }
    eprintln!();
    eprintln!(
        "  Total: {:.1}s ({:.1} min) — {} cells completed",
        total_wall,
        total_wall / 60.0,
        results.len()
    );

    // Print to stdout as CSV summary
    println!(
        "label,lattice,beta,mass,nf,plaquette,std_plaq,polyakov,t0,w0,topo_q,pbp,acceptance,wall_s"
    );
    for r in &results {
        println!(
            "{},{},{:.4},{},{},{:.8},{:.4e},{:.6},{},{},{},{},{:.3},{:.1}",
            r.label,
            r.lattice,
            r.beta,
            r.mass.map(|m| format!("{m:.4}")).unwrap_or_default(),
            r.nf,
            r.mean_plaquette,
            r.std_plaquette,
            r.mean_polyakov,
            r.t0.map(|v| format!("{v:.6}")).unwrap_or_default(),
            r.w0.map(|v| format!("{v:.6}")).unwrap_or_default(),
            r.topo_charge
                .map(|v| format!("{v:.4}"))
                .unwrap_or_default(),
            r.chiral_condensate
                .map(|v| format!("{v:.6e}"))
                .unwrap_or_default(),
            r.acceptance,
            r.wall_seconds,
        );
    }

    drop(telemetry);

    // JSON output
    if let Some(path) = &args.output {
        let report = serde_json::json!({
            "run": run_manifest,
            "phase": args.phase,
            "seed": args.seed,
            "total_cells": results.len(),
            "total_wall_seconds": total_wall,
            "results": results,
        });
        match serde_json::to_string_pretty(&report) {
            Ok(json) => {
                std::fs::write(path, json)
                    .unwrap_or_else(|e| eprintln!("Failed to write {path}: {e}"));
                eprintln!("  Results written to: {path}");
            }
            Err(e) => eprintln!("JSON error: {e}"),
        }
    }
}
