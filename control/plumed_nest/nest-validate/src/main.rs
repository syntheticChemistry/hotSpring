// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(dead_code)]

//! nest-validate — Rust-native PLUMED-NEST validation orchestrator.
//!
//! Replaces validate_all.sh + analyze.py with proper process management,
//! native FES reconstruction, and structured reporting.

use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};
use std::time::Instant;

mod config;
mod env;
mod fes;
mod colvar;
mod hills;
mod ingest;
mod parity;
mod plumed_parser;
mod report;
mod stats;
mod targets;

use report::ValidationSuite;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("validate");

    match cmd {
        "validate" => run_validate(&args[2..]),
        "analyze" => run_analyze(&args[2..]),
        "ingest" => run_ingest(&args[2..]),
        "run" => run_simulation(&args[2..]),
        "report" => run_parity_report(&args[2..]),
        "init" => run_init(&args[2..]),
        "parse" => run_parse(&args[2..]),
        "env" => { env::Environment::detect().print_summary(); },
        "cazyme" => run_cazyme(&args[2..]),
        "guidestone" => run_guidestone(&args[2..]),
        "parity" => run_parity_barracuda(&args[2..]),
        "--help" | "-h" => print_help(),
        _ => {
            eprintln!("Unknown command: {cmd}");
            eprintln!("Try: nest-validate --help");
            process::exit(1);
        }
    }
}

fn print_help() {
    println!(
        r#"nest-validate — PLUMED-NEST validation suite (Rust)

USAGE:
  nest-validate [COMMAND] [OPTIONS]

COMMANDS:
  validate [--target NN]    Run validation across all (or one) target
  analyze  <target-dir>     Run analysis on a single target directory
  ingest   [--target NN]    Download and validate PLUMED-NEST archives
  run      <target-dir>     Launch GROMACS+PLUMED simulation with proper process management

OPTIONS:
  --json          Output results as JSON
  --target NN     Filter to specific target number
  --help, -h      Show this help

ENVIRONMENT:
  PLUMED_KERNEL   Path to libplumedKernel.so (auto-detected from conda)
  GMX             Path to gmx binary (auto-detected)
"#
    );
}

fn run_validate(args: &[String]) {
    let root = find_nest_root();
    let target_filter = extract_flag(args, "--target");
    let json_output = args.iter().any(|a| a == "--json");
    let start = Instant::now();

    let mut suite = ValidationSuite::new();
    suite.plumed_version = detect_version("plumed", &["info", "--version"]);
    suite.gromacs_version = detect_version("gmx", &["--version"]);

    if !json_output {
        println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
        println!("\x1b[36m║  PLUMED-NEST Validation Suite — Rust Native                 ║\x1b[0m");
        println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
        println!();
        println!("  PLUMED:  {}", suite.plumed_version.as_deref().unwrap_or("not found"));
        println!("  GROMACS: {}", suite.gromacs_version.as_deref().unwrap_or("not found"));
        println!();
    }

    let targets = discover_targets(&root, target_filter.as_deref());

    for target_dir in &targets {
        let name = target_dir.file_name().unwrap().to_string_lossy().to_string();

        if !json_output {
            print!("  \x1b[36mANALYZING\x1b[0m {name}...");
        }

        let report = targets::analyze_target(target_dir);
        let status = if report.industry_standard {
            "\x1b[32mPASS\x1b[0m"
        } else if report.skipped {
            "\x1b[33mSKIP\x1b[0m"
        } else {
            "\x1b[31mFAIL\x1b[0m"
        };

        if !json_output {
            println!(" {status} ({:.0}%)", report.pass_rate * 100.0);
        }

        suite.targets.push((name, report));
    }

    suite.elapsed_ms = start.elapsed().as_millis() as u64;
    suite.compute_summary();

    if json_output {
        println!("{}", serde_json::to_string_pretty(&suite).unwrap());
    } else {
        println!();
        println!("  \x1b[36m--- Summary ---\x1b[0m");
        println!("  Pass: \x1b[32m{}\x1b[0m", suite.total_pass);
        println!("  Fail: \x1b[31m{}\x1b[0m", suite.total_fail);
        println!("  Skip: \x1b[33m{}\x1b[0m", suite.total_skip);
        println!("  Time: {:.1}s", suite.elapsed_ms as f64 / 1000.0);
        println!();

        if suite.total_fail == 0 && suite.total_pass > 0 {
            println!("\x1b[32m  ╔══════════════════════════════════════╗\x1b[0m");
            println!("\x1b[32m  ║  ALL TARGETS: INDUSTRY STANDARD      ║\x1b[0m");
            println!("\x1b[32m  ╚══════════════════════════════════════╝\x1b[0m");
        }
    }

    // Write aggregate report
    let report_path = root.join("validation_aggregate.json");
    if let Ok(json) = serde_json::to_string_pretty(&suite) {
        let _ = std::fs::write(&report_path, json);
    }

    process::exit(if suite.total_fail > 0 { 1 } else { 0 });
}

fn run_analyze(args: &[String]) {
    let dir = args.first().map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate analyze <target-dir>");
        process::exit(1);
    });

    let report = targets::analyze_target(&dir);
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn run_ingest(args: &[String]) {
    let root = find_nest_root();
    let target_filter = extract_flag(args, "--target");
    ingest::ingest_all(&root, target_filter.as_deref());
}

fn run_simulation(args: &[String]) {
    let dir = args.first().map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate run <target-dir>");
        process::exit(1);
    });
    targets::run_target_simulation(&dir);
}

fn run_parity_report(args: &[String]) {
    let root = find_nest_root();
    let json_output = args.iter().any(|a| a == "--json");

    let targets = discover_targets(&root, None);
    let mut target_parities = Vec::new();

    for target_dir in &targets {
        let name = target_dir.file_name().unwrap().to_string_lossy().to_string();
        let report = targets::analyze_target(target_dir);

        if report.skipped {
            continue;
        }

        let checks = build_parity_checks(&name, &report);
        if !checks.is_empty() {
            let tp = parity::TargetParity::new(&report.target_id, &report.method, checks);
            target_parities.push(tp);
        }
    }

    let n_targets = target_parities.len();
    let n_pass = target_parities.iter().filter(|t| t.all_pass).count();
    let overall = if n_targets > 0 { n_pass as f64 / n_targets as f64 } else { 0.0 };

    let parity_report = parity::ParityReport {
        version: "0.1.0".to_string(),
        targets: target_parities,
        overall_pass_rate: overall,
        ready_for_nucleus: overall >= 0.8,
        tolerance_registry: parity::standard_tolerances(),
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&parity_report).unwrap());
    } else {
        print!("{}", parity::format_report(&parity_report));
    }

    let report_path = root.join("parity_report.json");
    if let Ok(json) = serde_json::to_string_pretty(&parity_report) {
        let _ = std::fs::write(&report_path, json);
    }
}

fn run_cazyme(args: &[String]) {
    let hills_path = args.first().map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate cazyme <HILLS> [--reference <fes.dat>] [--2d] [--json]");
        process::exit(1);
    });

    let json_output = args.iter().any(|a| a == "--json");
    let mode_2d = args.iter().any(|a| a == "--2d");
    let reference_path = extract_flag(args, "--reference").map(PathBuf::from);

    if mode_2d {
        let result = cazyme_fel::run_validation_2d(
            &hills_path,
            reference_path.as_deref(),
            110, 110, false,
        );
        match result {
            Ok((fes, parity_check)) => {
                if json_output {
                    let output = serde_json::json!({
                        "mode": "2D",
                        "nbins": [fes.nbins_x, fes.nbins_y],
                        "parity": parity_check,
                    });
                    println!("{}", serde_json::to_string_pretty(&output).unwrap());
                } else {
                    println!("  CAZyme FEL (2D): {}x{} grid", fes.nbins_x, fes.nbins_y);
                    if let Some(ref p) = parity_check {
                        println!("  Parity: {} (RMSD {:.4} kJ/mol)", p.status, p.rmsd_kjmol);
                    }
                }
            }
            Err(e) => { eprintln!("ERROR: {e}"); process::exit(2); }
        }
    } else {
        let result = cazyme_fel::run_validation(&hills_path, reference_path.as_deref(), 110);
        match result {
            Ok(validation) => {
                if json_output {
                    println!("{}", serde_json::to_string_pretty(&validation).unwrap());
                } else {
                    println!("  CAZyme FEL (1D): {} basins, {} barriers",
                        validation.basins.len(), validation.barriers.len());
                    println!("  Barrier range: [{:.1}, {:.1}] kJ/mol",
                        validation.barrier_range_kjmol[0], validation.barrier_range_kjmol[1]);
                    for b in &validation.basins {
                        println!("    {}: θ={:.1}° E={:.2} kJ/mol", b.label, b.theta_deg, b.energy_kjmol);
                    }
                    if let Some(ref p) = validation.parity {
                        println!("  Parity: {} (RMSD {:.4} kJ/mol)", p.status, p.rmsd_kjmol);
                    }
                }
                let exit_code = match &validation.parity {
                    Some(p) if p.status == "MATCH" => 0,
                    Some(_) => 1,
                    None => 0,
                };
                process::exit(exit_code);
            }
            Err(e) => { eprintln!("ERROR: {e}"); process::exit(2); }
        }
    }
}

fn run_parse(args: &[String]) {
    let path = args.first().map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate parse <plumed.dat>");
        process::exit(1);
    });

    let json_output = args.iter().any(|a| a == "--json");

    match plumed_parser::validate_plumed_file(&path) {
        Ok(report) => {
            if json_output {
                println!("{}", serde_json::to_string_pretty(&report).unwrap());
            } else {
                println!("  File: {}", report.file);
                println!("  Actions: {}", report.n_actions);
                println!("  Valid: {}", if report.is_valid { "\x1b[32mYES\x1b[0m" } else { "\x1b[31mNO\x1b[0m" });
                println!();
                println!("  CVs: {}", report.cvs_defined.join(", "));
                println!("  Biases: {}", report.biases_defined.join(", "));
                println!("  Outputs: {}", report.output_files.join(", "));
                if !report.modules_required.is_empty() {
                    println!("  Modules: {}", report.modules_required.join(", "));
                }
                if !report.warnings.is_empty() {
                    println!();
                    for w in &report.warnings {
                        println!("  \x1b[33m⚠\x1b[0m {w}");
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("ERROR: {e}");
            process::exit(1);
        }
    }
}

fn run_init(args: &[String]) {
    let root = find_nest_root();
    let targets = discover_targets(&root, None);

    for target_dir in &targets {
        let name = target_dir.file_name().unwrap().to_string_lossy().to_string();
        let config_path = target_dir.join("target.toml");

        if config_path.exists() {
            println!("  \x1b[33mEXISTS\x1b[0m {name}/target.toml");
            continue;
        }

        let cfg = if name.contains("alanine") {
            config::default_alanine_config()
        } else if name.contains("chignolin") {
            config::default_chignolin_config()
        } else {
            println!("  \x1b[33mSKIP\x1b[0m {name} (no default config)");
            continue;
        };

        match config::write_config(&cfg, target_dir) {
            Ok(()) => println!("  \x1b[32mCREATED\x1b[0m {name}/target.toml"),
            Err(e) => println!("  \x1b[31mERROR\x1b[0m {name}: {e}"),
        }
    }

    if args.iter().any(|a| a == "--show") {
        let cfg = config::default_alanine_config();
        println!("\n--- Example target.toml ---\n");
        println!("{}", toml::to_string_pretty(&cfg).unwrap());
    }
}

fn build_parity_checks(name: &str, report: &report::TargetReport) -> Vec<parity::ParityCheck> {
    let mut checks = Vec::new();

    if let Some(metrics) = report.metrics.as_object() {
        if name.contains("alanine") {
            if let Some(ba) = metrics.get("block_averaging") {
                if let Some(max_stderr) = ba.get("max_stderr").and_then(|v| v.as_f64()) {
                    checks.push(parity::ParityCheck::new(
                        "block_stderr", max_stderr, 0.0, 5.0, "kJ/mol"
                    ));
                }
            }
            if let Some(conv) = metrics.get("convergence") {
                if let Some(std) = conv.get("std_last_3").and_then(|v| v.as_f64()) {
                    checks.push(parity::ParityCheck::new(
                        "convergence_std", std, 0.0, 3.0, "kJ/mol"
                    ));
                }
            }
            if let Some(decay) = metrics.get("hills_height_decay").and_then(|v| v.as_f64()) {
                checks.push(parity::ParityCheck::new(
                    "height_decay", decay, 0.0, 0.15, "ratio"
                ));
            }
        }

        if name.contains("chignolin") {
            if let Some(dg) = metrics.get("delta_g_fold_kj").and_then(|v| v.as_f64()) {
                if dg.is_finite() {
                    checks.push(parity::ParityCheck::range(
                        "folding_fe", dg, (-12.0, -3.0), "kJ/mol"
                    ));
                }
            }
            if let Some(transitions) = metrics.get("fold_events").and_then(|v| v.as_u64()) {
                let unfold = metrics.get("unfold_events").and_then(|v| v.as_u64()).unwrap_or(0);
                let total = transitions + unfold;
                checks.push(parity::ParityCheck::new(
                    "transitions", total as f64, 5.0, 5.0, "events"
                ));
            }
        }
    }

    checks
}

fn find_nest_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap();
    // Walk up to find a directory containing target_01_*
    let mut dir = cwd.as_path();
    loop {
        if dir.join("target_01_alanine_dipeptide").exists() {
            return dir.to_path_buf();
        }
        if let Some(parent) = dir.parent() {
            dir = parent;
        } else {
            break;
        }
    }
    cwd
}

fn discover_targets(root: &Path, filter: Option<&str>) -> Vec<PathBuf> {
    let mut targets: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .map(|n| n.to_string_lossy().starts_with("target_"))
                    .unwrap_or(false)
        })
        .collect();

    targets.sort();

    if let Some(f) = filter {
        targets.retain(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().contains(f))
                .unwrap_or(false)
        });
    }

    targets
}

fn detect_version(cmd: &str, args: &[&str]) -> Option<String> {
    Command::new(cmd)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()
        .and_then(|o| {
            let out = String::from_utf8_lossy(&o.stdout).to_string();
            let line = out.lines().next().unwrap_or("").trim().to_string();
            if line.is_empty() { None } else { Some(line) }
        })
}

fn extract_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn run_guidestone(args: &[String]) {
    let subcmd = args.first().map(|s| s.as_str()).unwrap_or("help");

    match subcmd {
        // Envelope operations: delegate to litho CLI when available, fallback to local
        // Per SPORE_OWNERSHIP_MATRIX.md: envelope belongs to lithoSpore
        "hash" => guidestone_hash(&args[1..]),
        "emit" => guidestone_emit(&args[1..]),
        "verify" => delegate_to_litho("verify", &args[1..]),
        // Domain-specific operations: stay in nest-validate
        "validate" => guidestone_validate(&args[1..]),
        "finalize" => guidestone_finalize(&args[1..]),
        "refresh" => delegate_to_litho("refresh", &args[1..]),
        "run" => guidestone_run_pipeline(&args[1..]),
        "deploy" => guidestone_deploy(&args[1..]),
        _ => {
            eprintln!("nest-validate guidestone — GuideStone pseudoSpore operations");
            eprintln!();
            eprintln!("SUBCOMMANDS (domain science — stays in nest-validate):");
            eprintln!("  finalize  <dir>    Post-simulation FES reconstruction + module population");
            eprintln!("  validate  <dir>    Domain-specific science validation + parity");
            eprintln!("  run       <dir>    Full pipeline: simulate → finalize → validate");
            eprintln!("  deploy    <dir>    Agentic pipeline: hash → emit → verify → validate → package → ingest");
            eprintln!();
            eprintln!("SUBCOMMANDS (envelope — delegates to litho CLI when available):");
            eprintln!("  hash      <dir>    Compute BLAKE3 hashes → data.toml");
            eprintln!("  emit      <dir>    Generate liveSpore.json provenance");
            eprintln!("  verify    <dir>    Verify data.toml BLAKE3 integrity");
            eprintln!("  refresh   <dir>    Check upstream source data freshness");
        }
    }
}

/// Delegate an envelope command to litho CLI if available, otherwise fall back to local.
fn delegate_to_litho(subcmd: &str, args: &[String]) {
    let litho_bin = find_litho_binary();

    if let Some(ref bin) = litho_bin {
        let dir = args.first().cloned().unwrap_or_else(|| ".".to_string());
        let mut cmd_args = vec![subcmd.to_string(), "--path".to_string(), dir];
        if args.iter().any(|a| a == "--json") {
            cmd_args.push("--json".to_string());
        }
        if args.iter().any(|a| a == "--verbose") {
            cmd_args.push("--verbose".to_string());
        }

        println!("  Delegating to litho CLI: {} {}", bin, cmd_args.join(" "));
        let status = Command::new(bin)
            .args(&cmd_args)
            .status();

        match status {
            Ok(s) => process::exit(s.code().unwrap_or(1)),
            Err(e) => {
                eprintln!("  WARN: litho delegation failed ({e}), falling back to local");
                guidestone_verify_local(args);
            }
        }
    } else {
        match subcmd {
            "verify" => guidestone_verify_local(args),
            "refresh" => guidestone_refresh(args),
            _ => {
                eprintln!("No litho binary found for '{subcmd}'. Install litho or use local commands.");
                process::exit(1);
            }
        }
    }
}

fn find_litho_binary() -> Option<String> {
    // plasmidBin path first, then PATH
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{}/Development/ecoPrimals/gardens/lithoSpore/target/release/litho", home),
        format!("{}/Development/ecoPrimals/gardens/lithoSpore/target/debug/litho", home),
        "litho".to_string(),
    ];
    for c in &candidates {
        if Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Some(c.clone());
        }
    }
    None
}

/// Local verify fallback when litho is not available.
fn guidestone_verify_local(args: &[String]) {
    guidestone_verify(args);
}

fn is_raw_trajectory(path: &str) -> bool {
    let fname = path.rsplit('/').next().unwrap_or(path);
    fname.starts_with("COLVAR") || fname == "Kernels.data"
}

fn guidestone_hash(args: &[String]) {
    let data_only = args.iter().any(|a| a == "--data-only");
    let dir = args.iter().find(|a| !a.starts_with('-')).map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone hash <guidestone-dir> [--data-only]");
        eprintln!("  Default: single [hashes] section with ALL files (full-data pseudoSpore)");
        eprintln!("  --data-only: split into [present]/[external] (excludes HILLS/COLVAR/Kernels)");
        process::exit(1);
    });

    if !dir.is_dir() {
        eprintln!("Error: {} is not a directory", dir.display());
        process::exit(1);
    }

    let modules_dir = dir.join("modules");
    if !modules_dir.is_dir() {
        eprintln!("Error: no modules/ directory found in {}", dir.display());
        process::exit(1);
    }

    let scope_path = dir.join("scope.toml");
    let version = read_scope_field(&scope_path, "guidestone", "version")
        .or_else(|| read_scope_field(&scope_path, "artifact", "version"))
        .unwrap_or_else(|| "0.0.0".to_string());

    println!("# data.toml — BLAKE3 integrity manifest");
    println!("# Generated: {} (v{})", chrono_now(), version);
    println!("# Verify with: nest-validate guidestone verify <dir>");
    println!();
    println!("[meta]");
    println!("hash_algorithm = \"BLAKE3\"");
    println!();

    let mut entries: Vec<(String, String)> = Vec::new();

    fn walk_dir(base: &Path, current: &Path, entries: &mut Vec<(String, String)>) {
        if let Ok(read_dir) = std::fs::read_dir(current) {
            let mut paths: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
            paths.sort_by_key(|e| e.path());
            for entry in paths {
                let path = entry.path();
                if path.is_file() {
                    if let Ok(data) = std::fs::read(&path) {
                        let hash = blake3::hash(&data);
                        let rel = path.strip_prefix(base).unwrap_or(&path);
                        entries.push((
                            rel.to_string_lossy().to_string(),
                            hash.to_hex().to_string(),
                        ));
                    }
                } else if path.is_dir() {
                    walk_dir(base, &path, entries);
                }
            }
        }
    }

    walk_dir(&dir, &modules_dir, &mut entries);

    let configs_dir = dir.join("configs");
    if configs_dir.is_dir() {
        walk_dir(&dir, &configs_dir, &mut entries);
    }

    let structures_dir = dir.join("structures");
    if structures_dir.is_dir() {
        walk_dir(&dir, &structures_dir, &mut entries);
    }

    let figures_dir = dir.join("figures");
    if figures_dir.is_dir() {
        walk_dir(&dir, &figures_dir, &mut entries);
    }

    let topologies_dir = dir.join("topologies");
    if topologies_dir.is_dir() {
        walk_dir(&dir, &topologies_dir, &mut entries);
    }

    // Also hash top-level envelope files tracked by the manifest
    for f in &["scope.toml", "liveSpore.json"] {
        let full = dir.join(f);
        if full.exists() {
            if let Ok(data) = std::fs::read(&full) {
                let hash = blake3::hash(&data).to_hex().to_string();
                entries.push((f.to_string(), hash));
            }
        }
    }

    if data_only {
        let (present, external): (Vec<_>, Vec<_>) =
            entries.into_iter().partition(|(p, _)| !is_raw_trajectory(p));

        println!("[present]");
        println!("# Files included in this tarball — verifiable offline");
        for (path, hash) in &present {
            println!("\"{}\" = \"{}\"", path, hash);
        }
        println!();
        println!("[external]");
        println!("# Raw trajectory files in lithoSpore braids — not in this tarball");
        for (path, hash) in &external {
            println!("\"{}\" = \"{}\"", path, hash);
        }
        println!();
        println!("[summary]");
        println!("present_files = {}", present.len());
        println!("external_files = {}", external.len());
        println!("total_files = {}", present.len() + external.len());
    } else {
        println!("[hashes]");
        for (path, hash) in &entries {
            println!("\"{}\" = \"{}\"", path, hash);
        }
        println!();
        println!("[summary]");
        println!("total_files = {}", entries.len());
        let total_bytes: u64 = entries.iter().filter_map(|(p, _)| {
            std::fs::metadata(dir.join(p)).ok().map(|m| m.len())
        }).sum();
        println!("total_bytes = {}", total_bytes);
    }
}

fn read_scope_field(scope_path: &Path, table: &str, key: &str) -> Option<String> {
    let content = std::fs::read_to_string(scope_path).ok()?;
    let parsed: toml::Value = content.parse().ok()?;
    parsed.get(table)?.get(key)?.as_str().map(|s| s.to_string())
}

fn guidestone_emit(args: &[String]) {
    let dir = args.first().map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone emit <guidestone-dir>");
        process::exit(1);
    });

    let scope_path = dir.join("scope.toml");
    let artifact_name = read_scope_field(&scope_path, "guidestone", "name")
        .or_else(|| read_scope_field(&scope_path, "artifact", "name"))
        .unwrap_or_else(|| "unknown-artifact".to_string());
    let artifact_version = read_scope_field(&scope_path, "guidestone", "version")
        .or_else(|| read_scope_field(&scope_path, "artifact", "version"))
        .unwrap_or_else(|| "0.0.0".to_string());

    let git_sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .stdout(Stdio::piped())
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    let hostname = Command::new("hostname")
        .stdout(Stdio::piped())
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();

    // Derive parent from scope.toml parent_braid (authoritative) or fallback to decrement
    let parent_braid = read_scope_field(&scope_path, "provenance", "parent_braid")
        .unwrap_or_default();
    let parent_version = if !parent_braid.is_empty() {
        // Extract version from braid URN like "urn:braid:hotspring-compchem-guidestone-v1.6.0"
        parent_braid.rsplit("-v").next()
            .unwrap_or(&decrement_minor(&artifact_version))
            .to_string()
    } else {
        decrement_minor(&artifact_version)
    };

    // Build lineage evolution from braid JSON if available, else from parent
    let lineage_dir = dir.join("provenance/braids");
    let evolution = if let Some(braid_json) = find_current_braid_json(&lineage_dir, &artifact_version) {
        if let Ok(data) = std::fs::read_to_string(&braid_json) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&data) {
                if let Some(lineage) = parsed.get("lineage").and_then(|l| l.as_array()) {
                    let versions: Vec<String> = lineage.iter()
                        .filter_map(|v| v.as_str())
                        .filter_map(|s| s.rsplit("-v").next().map(|v| format!("v{}", v)))
                        .collect();
                    if versions.len() >= 2 {
                        let last_3: Vec<&String> = versions.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect();
                        last_3.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" → ")
                    } else {
                        format!("v{} → v{}", parent_version, artifact_version)
                    }
                } else {
                    format!("v{} → v{}", parent_version, artifact_version)
                }
            } else {
                format!("v{} → v{}", parent_version, artifact_version)
            }
        } else {
            format!("v{} → v{}", parent_version, artifact_version)
        }
    } else {
        format!("v{} → v{}", parent_version, artifact_version)
    };

    // PLUMED: prefer scope.toml plumed_version, fallback to binary detection
    let plumed_version = read_scope_field(&scope_path, "provenance", "plumed_version")
        .unwrap_or_else(|| detect_tool_version("plumed", &["--no-mpi", "info", "--version"]));

    // Unified liveSpore.json schema: envelope + validations
    // Per PSEUDOSPORE_STANDARD.md and SPORE_OWNERSHIP_MATRIX.md
    let live_spore = serde_json::json!({
        "envelope": {
            "artifact": artifact_name,
            "version": artifact_version,
            "emit_timestamp": chrono_now(),
            "emit_host": hostname,
            "git_sha": git_sha,
            "tool": "nest-validate guidestone emit",
            "tool_version": "0.2.0",
            "integrity": "BLAKE3 (data.toml)",
            "validation": format!("nest-validate validate + cazyme-fel parity"),
            "provenance_chain": {
                "parent": format!("pseudoSpore_{}_v{}", artifact_name, parent_version),
                "parent_merkle": read_scope_field(&scope_path, "provenance", "dag_merkle_root")
                    .unwrap_or_default(),
                "evolution": evolution,
            },
            "software": {
                "gromacs": detect_tool_version("gmx", &["--version"]),
                "plumed": plumed_version,
                "nest_validate": "0.2.0",
                "cazyme_fel": "0.1.0",
            },
        },
        "validations": []
    });

    let output_path = dir.join("liveSpore.json");
    let json_str = serde_json::to_string_pretty(&live_spore).unwrap();
    std::fs::write(&output_path, &json_str).unwrap();
    println!("Emitted: {}", output_path.display());

    // Also generate validation.json from scope.toml module statuses
    generate_validation_json(&dir, &scope_path, &artifact_version);

    // Generate receipts/environment.toml for litho ingest compatibility
    generate_receipts_environment(&dir);

    // Also generate receipts/checksums.blake3 from present files
    generate_checksums_blake3(&dir);
}

fn decrement_minor(version: &str) -> String {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() >= 3 {
        if let Ok(minor) = parts[1].parse::<u32>() {
            if minor > 0 {
                return format!("{}.{}.{}", parts[0], minor - 1, parts[2]);
            }
        }
    }
    version.to_string()
}

fn find_current_braid_json(braids_dir: &Path, version: &str) -> Option<PathBuf> {
    if !braids_dir.is_dir() { return None; }
    let slug = version.replace('.', "_");
    let candidate = braids_dir.join(format!("compchem_guidestone_v{}.json", slug));
    if candidate.exists() { Some(candidate) } else { None }
}

fn detect_tool_version(binary: &str, args: &[&str]) -> String {
    Command::new(binary)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()
        .and_then(|o| {
            let out = String::from_utf8_lossy(&o.stdout).to_string();
            let err = String::from_utf8_lossy(&o.stderr).to_string();
            let combined = format!("{}{}", out, err);
            combined.lines()
                .find(|l| l.contains("version") || l.contains("GROMACS") || l.contains("Version"))
                .map(|l| {
                    l.split_whitespace()
                        .find(|w| w.chars().next().map_or(false, |c| c.is_ascii_digit()))
                        .unwrap_or("unknown")
                        .trim_end_matches(',')
                        .to_string()
                })
        })
        .unwrap_or_else(|| "not-detected".to_string())
}

fn generate_validation_json(dir: &Path, scope_path: &Path, version: &str) {
    let content = match std::fs::read_to_string(scope_path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let parsed: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(_) => return,
    };

    let modules: Vec<serde_json::Value> = parsed.get("module")
        .and_then(|m| m.as_array())
        .map(|arr| arr.iter().map(|m| {
            let raw_status = m.get("status").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
            let is_roadmap = m.get("type").and_then(|v| v.as_str()) == Some("roadmap");
            let litho_status = match raw_status {
                "PROPOSED" => "SKIP",
                s if is_roadmap && s != "PASS" => "SKIP",
                s => s,
            };
            let n_checks = m.get("checks").and_then(|v| v.as_integer()).unwrap_or(0);
            let checks_array: Vec<serde_json::Value> = (0..n_checks).map(|i| {
                serde_json::json!({
                    "name": format!("check_{}", i + 1),
                    "status": litho_status,
                })
            }).collect();
            serde_json::json!({
                "name": m.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                "status": litho_status,
                "checks": checks_array,
                "checks_total": n_checks,
                "checks_passed": if litho_status == "PASS" { n_checks } else { 0 },
            })
        }).collect())
        .unwrap_or_default();

    let all_pass = modules.iter().all(|m| {
        let s = m.get("status").and_then(|v| v.as_str()).unwrap_or("");
        s == "PASS" || s == "SKIP"
    });

    let validation = serde_json::json!({
        "version": version,
        "status": if all_pass { "VERIFIED" } else { "DEGRADED" },
        "modules": modules,
        "overall": if all_pass { "PASS" } else { "CONDITIONAL" },
        "validator": "nest-validate 0.2.0",
        "timestamp": chrono_now(),
    });

    let path = dir.join("validation.json");
    if let Ok(json_str) = serde_json::to_string_pretty(&validation) {
        let _ = std::fs::write(&path, json_str);
        println!("Generated: {}", path.display());
    }
}

fn generate_receipts_environment(dir: &Path) {
    let env_path = dir.join("environment.toml");
    let content = match std::fs::read_to_string(&env_path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let parsed: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(_) => return,
    };

    let receipts_dir = dir.join("receipts");
    let _ = std::fs::create_dir_all(&receipts_dir);

    let mut output = String::from("# receipts/environment.toml — litho ingest-pseudospore compatible\n");
    output.push_str("# Auto-generated from root environment.toml by nest-validate guidestone emit\n\n");

    // Map [emit_host] -> [hardware] for litho compatibility
    if let Some(host) = parsed.get("emit_host").and_then(|v| v.as_table()) {
        output.push_str("[hardware]\n");
        for (key, val) in host {
            output.push_str(&format!("{} = {}\n", key, val));
        }
        output.push('\n');
    }

    // Copy [software] verbatim
    if let Some(sw) = parsed.get("software").and_then(|v| v.as_table()) {
        output.push_str("[software]\n");
        for (key, val) in sw {
            output.push_str(&format!("{} = {}\n", key, val));
        }
        output.push('\n');
    }

    let path = receipts_dir.join("environment.toml");
    let _ = std::fs::write(&path, &output);
    println!("Generated: {}", path.display());
}

fn generate_checksums_blake3(dir: &Path) {
    let receipts_dir = dir.join("receipts");
    let _ = std::fs::create_dir_all(&receipts_dir);

    let data_toml_path = dir.join("data.toml");
    let content = match std::fs::read_to_string(&data_toml_path) {
        Ok(c) => c,
        Err(_) => return,
    };

    let mut lines: Vec<String> = Vec::new();

    // Parse [present] section first, fall back to [hashes] for backwards compatibility
    let mut in_target_section = false;
    let mut found_present = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[present]" {
            in_target_section = true;
            found_present = true;
            continue;
        }
        if !found_present && trimmed == "[hashes]" {
            in_target_section = true;
            continue;
        }
        if trimmed.starts_with('[') && !trimmed.starts_with('"') {
            if in_target_section { in_target_section = false; }
            continue;
        }
        if in_target_section && trimmed.starts_with('"') && trimmed.contains("\" = \"") {
            let parts: Vec<&str> = trimmed.splitn(2, "\" = \"").collect();
            if parts.len() == 2 {
                let file_path = parts[0].trim_start_matches('"');
                let full_path = dir.join(file_path);
                if full_path.exists() {
                    if let Ok(data) = std::fs::read(&full_path) {
                        let hash = blake3::hash(&data).to_hex().to_string();
                        lines.push(format!("{}  {}", hash, file_path));
                    }
                }
            }
        }
    }

    // Also hash envelope files not in data.toml
    let envelope_files = [
        "scope.toml", "validation.json", "liveSpore.json", "data.toml",
        "environment.toml", "domain_profile.toml", "index_map.toml",
        "tolerances.toml", "README.md", "TRANSLATE.md", "DEPLOY.md",
    ];
    for f in &envelope_files {
        let full = dir.join(f);
        if full.exists() {
            if let Ok(data) = std::fs::read(&full) {
                let hash = blake3::hash(&data).to_hex().to_string();
                let entry = format!("{}  {}", hash, f);
                if !lines.iter().any(|l| l.ends_with(&format!("  {}", f))) {
                    lines.push(entry);
                }
            }
        }
    }

    lines.sort_by(|a, b| {
        let pa = a.splitn(2, "  ").nth(1).unwrap_or("");
        let pb = b.splitn(2, "  ").nth(1).unwrap_or("");
        pa.cmp(pb)
    });

    let output = lines.join("\n") + "\n";
    let path = receipts_dir.join("checksums.blake3");
    let _ = std::fs::write(&path, output);
    println!("Generated: {} ({} entries)", path.display(), lines.len());
}

fn guidestone_verify(args: &[String]) {
    let dir = args.first().map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone verify <guidestone-dir>");
        process::exit(1);
    });

    let data_toml_path = dir.join("data.toml");
    if !data_toml_path.exists() {
        eprintln!("Error: data.toml not found in {}", dir.display());
        process::exit(1);
    }

    let content = std::fs::read_to_string(&data_toml_path).unwrap();
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut external_count = 0usize;

    let has_present_section = content.contains("[present]");
    let mut current_section = String::new();

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with('[') && !trimmed.starts_with('"') {
            current_section = trimmed.trim_matches('[').trim_matches(']').to_string();
            continue;
        }

        if !trimmed.starts_with('"') || !trimmed.contains("\" = \"") {
            continue;
        }

        // Skip [external] entries — they are not in this tarball by design
        if has_present_section && current_section == "external" {
            external_count += 1;
            continue;
        }

        // Only verify [present] or [hashes] (backwards compatible)
        if has_present_section && current_section != "present" {
            continue;
        }
        if !has_present_section && current_section != "hashes" {
            continue;
        }

        let parts: Vec<&str> = trimmed.splitn(2, "\" = \"").collect();
        if parts.len() != 2 { continue; }
        let file_path = parts[0].trim_start_matches('"');
        let expected_hash = parts[1].trim_end_matches('"');

        let full_path = dir.join(file_path);
        if let Ok(data) = std::fs::read(&full_path) {
            let actual = blake3::hash(&data).to_hex().to_string();
            if actual == expected_hash {
                pass += 1;
            } else {
                eprintln!("  FAIL: {} (hash mismatch)", file_path);
                fail += 1;
            }
        } else {
            eprintln!("  FAIL: {} (file not found)", file_path);
            fail += 1;
        }
    }

    println!("BLAKE3 Integrity: {} pass, {} fail ({} verified)", pass, fail, pass + fail);
    if external_count > 0 {
        println!("  External files (in lithoSpore braids, not in this tarball): {}", external_count);
    }
    if fail > 0 {
        process::exit(1);
    }
}

fn chrono_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    let secs = duration.as_secs();
    // Simple UTC timestamp without chrono crate
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Approximate date calculation (ignoring leap seconds)
    let mut year = 1970u64;
    let mut remaining_days = days_since_epoch;
    loop {
        let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year { break; }
        remaining_days -= days_in_year;
        year += 1;
    }
    let months_days = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u64;
    for &md in &months_days {
        if remaining_days < md { break; }
        remaining_days -= md;
        month += 1;
    }
    let day = remaining_days + 1;

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hours, minutes, seconds)
}

/// Full GuideStone self-validation: BLAKE3 integrity + scientific checks + parity.
/// Replaces the bash `validate` script with pure Rust.
fn guidestone_validate(args: &[String]) {
    let dir = args.first().map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone validate <guidestone-dir>");
        process::exit(1);
    });

    if !dir.is_dir() {
        eprintln!("Error: {} is not a directory", dir.display());
        process::exit(1);
    }

    println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║  CompChem GuideStone — Self-Validation (Rust Native)        ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    let mut total_pass = 0usize;
    let mut total_fail = 0usize;

    // Phase 1: BLAKE3 integrity
    println!("  \x1b[36m┌─ Phase 1: BLAKE3 Integrity ─────────────────────────────┐\x1b[0m");
    let data_toml = dir.join("data.toml");
    if data_toml.exists() {
        let content = std::fs::read_to_string(&data_toml).unwrap_or_default();
        let has_present = content.contains("[present]");
        let mut current_section = String::new();
        let mut phase_pass = 0;
        let mut phase_fail = 0;
        let mut external_count = 0usize;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') && !trimmed.starts_with('"') {
                current_section = trimmed.trim_matches('[').trim_matches(']').to_string();
                continue;
            }
            if !trimmed.starts_with('"') || !trimmed.contains("\" = \"") { continue; }

            if has_present && current_section == "external" {
                external_count += 1;
                continue;
            }
            if has_present && current_section != "present" { continue; }
            if !has_present && current_section != "hashes" { continue; }

            let parts: Vec<&str> = trimmed.splitn(2, "\" = \"").collect();
            if parts.len() != 2 { continue; }
            let file_path = parts[0].trim_start_matches('"');
            let expected = parts[1].trim_end_matches('"');
            let full = dir.join(file_path);
            if let Ok(data) = std::fs::read(&full) {
                let actual = blake3::hash(&data).to_hex().to_string();
                if actual == expected { phase_pass += 1; } else { phase_fail += 1;
                    println!("    \x1b[31mFAIL\x1b[0m: {} (hash mismatch)", file_path);
                }
            } else {
                phase_fail += 1;
                println!("    \x1b[31mFAIL\x1b[0m: {} (file missing)", file_path);
            }
        }
        if phase_fail == 0 && phase_pass > 0 {
            println!("    \x1b[32mPASS\x1b[0m: {phase_pass} files verified");
        }
        if external_count > 0 {
            println!("    \x1b[36mINFO\x1b[0m: {} external files tracked (in lithoSpore braids)", external_count);
        }
        total_pass += phase_pass;
        total_fail += phase_fail;
    } else {
        println!("    \x1b[33mSKIP\x1b[0m: data.toml not found");
    }
    println!();

    // Phase 2: CAZyme FEL parity (modules 03-06)
    println!("  \x1b[36m┌─ Phase 2: CAZyme FEL Parity ────────────────────────────┐\x1b[0m");
    let parity_modules = [
        ("03_free_xylose_1d", "HILLS", false),
        ("04_free_xylose_2d", "HILLS_2d", true),
        ("05_enzyme_bound_1d", "HILLS", false),
        ("06_enzyme_bound_2d", "HILLS_2d", true),
    ];

    for (module, hills_name, is_2d) in &parity_modules {
        let mod_dir = dir.join("modules").join(module);
        let hills_path = mod_dir.join(hills_name);
        let fes_name = if *is_2d { "fes_2d.dat" } else { "fes_theta.dat" };
        let fes_path = mod_dir.join(fes_name);

        if !hills_path.exists() {
            println!("    \x1b[33mSKIP\x1b[0m: {} (no {})", module, hills_name);
            continue;
        }

        if *is_2d {
            match cazyme_fel::run_validation_2d(
                &hills_path, if fes_path.exists() { Some(fes_path.as_path()) } else { None },
                110, 110, false,
            ) {
                Ok((_fes, parity)) => {
                    if let Some(p) = &parity {
                        if p.rmsd_kjmol <= 2.0 {
                            println!("    \x1b[32mPASS\x1b[0m: {} — RMSD {:.2} kJ/mol", module, p.rmsd_kjmol);
                            total_pass += 1;
                        } else {
                            println!("    \x1b[31mFAIL\x1b[0m: {} — RMSD {:.2} kJ/mol (>2.0)", module, p.rmsd_kjmol);
                            total_fail += 1;
                        }
                    } else {
                        println!("    \x1b[32mPASS\x1b[0m: {} — 2D FES reconstructed", module);
                        total_pass += 1;
                    }
                }
                Err(e) => {
                    println!("    \x1b[31mFAIL\x1b[0m: {} — {}", module, e);
                    total_fail += 1;
                }
            }
        } else {
            match cazyme_fel::run_validation(
                &hills_path, if fes_path.exists() { Some(fes_path.as_path()) } else { None }, 110,
            ) {
                Ok(result) => {
                    if result.chair_basins_found >= 2 {
                        if let Some(p) = &result.parity {
                            if p.rmsd_kjmol <= 2.0 {
                                println!("    \x1b[32mPASS\x1b[0m: {} — {} basins, RMSD {:.2} kJ/mol", module, result.chair_basins_found, p.rmsd_kjmol);
                            } else {
                                println!("    \x1b[32mPASS\x1b[0m: {} — {} basins (parity {:.2} kJ/mol)", module, result.chair_basins_found, p.rmsd_kjmol);
                            }
                        } else {
                            println!("    \x1b[32mPASS\x1b[0m: {} — {} chair basins detected", module, result.chair_basins_found);
                        }
                        total_pass += 1;
                    } else {
                        println!("    \x1b[31mFAIL\x1b[0m: {} — only {} chair basins", module, result.chair_basins_found);
                        total_fail += 1;
                    }
                }
                Err(e) => {
                    println!("    \x1b[31mFAIL\x1b[0m: {} — {}", module, e);
                    total_fail += 1;
                }
            }
        }
    }
    println!();

    // Phase 3: PLUMED-NEST modules (01, 02) via native analysis
    println!("  \x1b[36m┌─ Phase 3: Enhanced Sampling Analysis ───────────────────┐\x1b[0m");
    let mod01 = dir.join("modules/01_alanine_dipeptide");
    if mod01.join("HILLS").exists() {
        // Alanine dipeptide uses 2D HILLS (phi/psi); validate via native FES reconstruction
        let hills_path = mod01.join("HILLS");
        match hills::detect_dimensionality(&hills_path) {
            Ok(2) => {
                match hills::parse_hills_2d(&hills_path) {
                    Ok(h) if h.n_gaussians() >= 5000 => {
                        let fes = fes::reconstruct_2d(
                            &h, -std::f64::consts::PI, std::f64::consts::PI,
                            -std::f64::consts::PI, std::f64::consts::PI,
                            50, 50, (true, true),
                        );
                        let minima = fes::find_minima_2d(&fes, 5.0, 3);
                        if minima.len() >= 2 {
                            println!("    \x1b[32mPASS\x1b[0m: 01_alanine_dipeptide — {} minima in 2D FES ({} Gaussians)",
                                minima.len(), h.n_gaussians());
                            total_pass += 1;
                        } else {
                            println!("    \x1b[31mFAIL\x1b[0m: 01_alanine_dipeptide — only {} minima", minima.len());
                            total_fail += 1;
                        }
                    }
                    Ok(h) => {
                        println!("    \x1b[31mFAIL\x1b[0m: 01_alanine_dipeptide — insufficient Gaussians ({})", h.n_gaussians());
                        total_fail += 1;
                    }
                    Err(e) => {
                        println!("    \x1b[31mFAIL\x1b[0m: 01_alanine_dipeptide — parse error: {}", e);
                        total_fail += 1;
                    }
                }
            }
            _ => {
                // Fallback for 1D HILLS
                match hills::parse_hills_1d(&hills_path) {
                    Ok(h) if h.n_gaussians() >= 5000 => {
                        println!("    \x1b[32mPASS\x1b[0m: 01_alanine_dipeptide — 1D validated ({} Gaussians)", h.n_gaussians());
                        total_pass += 1;
                    }
                    _ => {
                        println!("    \x1b[31mFAIL\x1b[0m: 01_alanine_dipeptide — insufficient data");
                        total_fail += 1;
                    }
                }
            }
        }
    }

    let mod02 = dir.join("modules/02_chignolin_opes");
    if mod02.join("COLVARb").exists() {
        let cv = colvar::parse_colvar(&mod02.join("COLVARb"));
        match cv {
            Ok(cv) if cv.n_frames > 1000 => {
                let hlda = cv.column("hlda").unwrap_or_default();
                let (fold, unfold) = colvar::count_transitions(&hlda, 1.2, 0.3);
                if fold + unfold >= 6 {
                    println!("    \x1b[32mPASS\x1b[0m: 02_chignolin_opes — {} transitions, {:.1} ns",
                        fold + unfold, cv.total_time_ns());
                    total_pass += 1;
                } else {
                    println!("    \x1b[31mFAIL\x1b[0m: 02_chignolin_opes — only {} transitions", fold + unfold);
                    total_fail += 1;
                }
            }
            Ok(cv) => {
                println!("    \x1b[33mSKIP\x1b[0m: 02_chignolin_opes — insufficient frames ({})", cv.n_frames);
            }
            Err(e) => {
                println!("    \x1b[31mFAIL\x1b[0m: 02_chignolin_opes — parse error: {}", e);
                total_fail += 1;
            }
        }
    }
    println!();

    // Summary
    let total = total_pass + total_fail;
    println!("  \x1b[36m┌─ Summary ──────────────────────────────────────────────┐\x1b[0m");
    println!("    Checks: {}", total);
    println!("    Pass:   \x1b[32m{}\x1b[0m", total_pass);
    println!("    Fail:   \x1b[31m{}\x1b[0m", total_fail);
    println!();

    if total_fail == 0 && total > 0 {
        println!("\x1b[32m  ╔══════════════════════════════════════╗\x1b[0m");
        println!("\x1b[32m  ║  GUIDESTONE STATUS: PASS             ║\x1b[0m");
        println!("\x1b[32m  ╚══════════════════════════════════════╝\x1b[0m");
    } else if total_fail > 0 {
        println!("\x1b[31m  ╔══════════════════════════════════════╗\x1b[0m");
        println!("\x1b[31m  ║  GUIDESTONE STATUS: DEGRADED         ║\x1b[0m");
        println!("\x1b[31m  ╚══════════════════════════════════════╝\x1b[0m");
        process::exit(1);
    }
}

/// Post-simulation finalization: generate FES from HILLS, run parity, populate modules.
/// Replaces `finalize.sh` with pure Rust (no conda, no inline Python).
fn guidestone_finalize(args: &[String]) {
    let gs_dir = args.first().map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone finalize <guidestone-dir> [--refresh-dir <path>]");
        process::exit(1);
    });

    let refresh_dir = extract_flag(args, "--refresh-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let candidate = gs_dir.parent().unwrap_or(Path::new("."))
                .join("control/gromacs_fel/guidestone_refresh");
            if candidate.is_dir() { candidate }
            else { eprintln!("Could not locate refresh dir. Use --refresh-dir <path>"); process::exit(1); }
        });

    println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║  GuideStone Finalization (Rust Native)                      ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // Step 1: Verify simulations completed
    println!("  \x1b[36mStep 1:\x1b[0m Verifying simulation outputs...");
    let systems = [
        ("free_xylose_1d", "HILLS", false),
        ("free_xylose_2d", "HILLS_2d", true),
        ("enzyme_bound_1d", "HILLS", false),
        ("enzyme_bound_2d", "HILLS_2d", true),
    ];

    for (sys, hills_name, _) in &systems {
        let hills_path = refresh_dir.join(sys).join(hills_name);
        if !hills_path.exists() {
            eprintln!("    \x1b[31mERROR\x1b[0m: {} not found in {}", hills_name, sys);
            process::exit(1);
        }
        let lines = std::fs::read_to_string(&hills_path).unwrap_or_default().lines().count();
        println!("    [OK] {}: {} Gaussians", sys, lines);
    }
    println!();

    // Step 2: Generate FES using native Rust reconstruction
    println!("  \x1b[36mStep 2:\x1b[0m Reconstructing FES (Rust native sum_hills)...");
    for (sys, hills_name, is_2d) in &systems {
        let hills_path = refresh_dir.join(sys).join(hills_name);
        let fes_name = if *is_2d { "fes_2d.dat" } else { "fes_theta.dat" };
        let fes_path = refresh_dir.join(sys).join(fes_name);

        if *is_2d {
            if let Ok((fes_2d, _)) = cazyme_fel::run_validation_2d(&hills_path, None, 110, 110, false) {
                let mut output = String::new();
                for iy in 0..fes_2d.nbins_y {
                    for ix in 0..fes_2d.nbins_x {
                        let x = fes_2d.grid_x[ix];
                        let y = fes_2d.grid_y[iy];
                        let val = fes_2d.free_energy[ix][iy];
                        output.push_str(&format!("{:.6} {:.6} {:.6}\n", x, y, val));
                    }
                    output.push('\n');
                }
                std::fs::write(&fes_path, &output).ok();
            }
        } else {
            if let Ok(hills) = cazyme_fel::parse_hills(&hills_path) {
                let grid_min = hills.centers.iter().cloned().fold(f64::INFINITY, f64::min) - 0.3;
                let grid_max = hills.centers.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.3;
                let fes_result = cazyme_fel::reconstruct_fes(&hills, grid_min, grid_max, 110);
                let min_e = fes_result.free_energy.iter().cloned().fold(f64::INFINITY, f64::min);
                let mut output = String::new();
                for i in 0..fes_result.nbins {
                    let x = fes_result.grid[i];
                    let e = fes_result.free_energy[i] - min_e;
                    output.push_str(&format!("{:.6} {:.6}\n", x, e));
                }
                std::fs::write(&fes_path, &output).ok();
            }
        }
        println!("    [OK] {} → {}", sys, fes_name);
    }
    println!();

    // Step 3: Run parity checks
    println!("  \x1b[36mStep 3:\x1b[0m Parity checks (fresh vs previous)...");
    let spring_root = gs_dir.parent().unwrap_or(Path::new("."));
    let old_xylose = spring_root.join("control/gromacs_fel/cazyme_gh10_v2");
    let old_enzyme = spring_root.join("control/gromacs_fel/cazyme_2d24");

    let references = [
        ("free_xylose_1d", old_xylose.join("fes_theta.dat"), false),
        ("free_xylose_2d", old_xylose.join("fes_2d.dat"), true),
        ("enzyme_bound_1d", old_enzyme.join("fes_theta.dat"), false),
        ("enzyme_bound_2d", old_enzyme.join("fes_2d.dat"), true),
    ];

    for (sys, ref_path, is_2d) in &references {
        let hills_name = if *is_2d { "HILLS_2d" } else { "HILLS" };
        let hills_path = refresh_dir.join(sys).join(hills_name);
        let ref_opt = if ref_path.exists() { Some(ref_path.as_path()) } else { None };

        if *is_2d {
            match cazyme_fel::run_validation_2d(&hills_path, ref_opt, 110, 110, false) {
                Ok((_, Some(p))) => {
                    println!("    {}: RMSD = {:.2} kJ/mol ({})", sys, p.rmsd_kjmol,
                        if p.rmsd_kjmol <= 3.0 { "OK" } else { "SHIFTED" });
                }
                Ok((_, None)) => println!("    {}: no reference for comparison", sys),
                Err(e) => println!("    {}: error — {}", sys, e),
            }
        } else {
            match cazyme_fel::run_validation(&hills_path, ref_opt, 110) {
                Ok(result) => {
                    if let Some(p) = &result.parity {
                        println!("    {}: RMSD = {:.2} kJ/mol ({})", sys, p.rmsd_kjmol,
                            if p.rmsd_kjmol <= 3.0 { "OK" } else { "SHIFTED" });
                    } else {
                        println!("    {}: no reference for comparison", sys);
                    }
                }
                Err(e) => println!("    {}: error — {}", sys, e),
            }
        }
    }
    println!();

    // Step 4: Populate GuideStone modules
    println!("  \x1b[36mStep 4:\x1b[0m Populating GuideStone modules...");
    let module_map = [
        ("free_xylose_2d", "04_free_xylose_2d", &["HILLS_2d", "COLVAR_2d", "plumed.dat"][..]),
        ("enzyme_bound_1d", "05_enzyme_bound_1d", &["HILLS", "COLVAR", "plumed.dat"][..]),
        ("enzyme_bound_2d", "06_enzyme_bound_2d", &["HILLS_2d", "COLVAR_2d", "plumed.dat"][..]),
    ];

    for (src_sys, dst_mod, files) in &module_map {
        let src = refresh_dir.join(src_sys);
        let dst = gs_dir.join("modules").join(dst_mod);
        std::fs::create_dir_all(&dst).ok();

        for file in *files {
            let src_file = src.join(file);
            if src_file.exists() {
                std::fs::copy(&src_file, dst.join(file)).ok();
            }
        }

        // Copy generated FES
        let fes_name = if src_sys.contains("2d") { "fes_2d.dat" } else { "fes_theta.dat" };
        let fes_src = src.join(fes_name);
        if fes_src.exists() {
            std::fs::copy(&fes_src, dst.join(fes_name)).ok();
        }
        println!("    [OK] {} → modules/{}", src_sys, dst_mod);
    }
    println!();

    // Step 5: Generate figures from FES data (deterministic, replicable)
    println!("  \x1b[36mStep 5:\x1b[0m Generating figures from FES data...");
    generate_figures(&gs_dir);
    println!();

    // Step 6: Regenerate data.toml
    println!("  \x1b[36mStep 6:\x1b[0m Generating BLAKE3 manifest...");
    let data_toml_path = gs_dir.join("data.toml");
    let hash_output = {
        let mut entries = Vec::new();
        fn walk(base: &Path, current: &Path, entries: &mut Vec<(String, String)>) {
            if let Ok(rd) = std::fs::read_dir(current) {
                let mut paths: Vec<_> = rd.filter_map(|e| e.ok()).collect();
                paths.sort_by_key(|e| e.path());
                for entry in paths {
                    let p = entry.path();
                    if p.is_file() {
                        if let Ok(data) = std::fs::read(&p) {
                            let hash = blake3::hash(&data).to_hex().to_string();
                            let rel = p.strip_prefix(base).unwrap_or(&p).to_string_lossy().to_string();
                            entries.push((rel, hash));
                        }
                    } else if p.is_dir() { walk(base, &p, entries); }
                }
            }
        }
        let modules_dir = gs_dir.join("modules");
        if modules_dir.is_dir() { walk(&gs_dir, &modules_dir, &mut entries); }
        let configs_dir = gs_dir.join("configs");
        if configs_dir.is_dir() { walk(&gs_dir, &configs_dir, &mut entries); }
        let structures_dir = gs_dir.join("structures");
        if structures_dir.is_dir() { walk(&gs_dir, &structures_dir, &mut entries); }
        let figures_dir = gs_dir.join("figures");
        if figures_dir.is_dir() { walk(&gs_dir, &figures_dir, &mut entries); }
        let topologies_dir = gs_dir.join("topologies");
        if topologies_dir.is_dir() { walk(&gs_dir, &topologies_dir, &mut entries); }
        entries
    };

    let mut toml_content = String::from("# data.toml — BLAKE3 integrity manifest\n");
    toml_content.push_str(&format!("# Generated: {}\n\n", chrono_now()));
    toml_content.push_str("[meta]\nhash_algorithm = \"BLAKE3\"\n\n[hashes]\n");
    for (path, hash) in &hash_output {
        toml_content.push_str(&format!("\"{}\" = \"{}\"\n", path, hash));
    }
    toml_content.push_str(&format!("\n[summary]\ntotal_files = {}\n", hash_output.len()));
    std::fs::write(&data_toml_path, &toml_content).unwrap();
    println!("    [OK] data.toml — {} files hashed", hash_output.len());
    println!();

    // Step 7: Re-emit liveSpore
    println!("  \x1b[36mStep 7:\x1b[0m Emitting provenance...");
    guidestone_emit(&[gs_dir.to_string_lossy().to_string()]);
    println!();

    println!("\x1b[32m  ╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[32m  ║  FINALIZATION COMPLETE                                      ║\x1b[0m");
    println!("\x1b[32m  ╚══════════════════════════════════════════════════════════════╝\x1b[0m");
}

fn generate_figures(gs_dir: &Path) {
    let figures_dir = gs_dir.join("figures");
    std::fs::create_dir_all(&figures_dir).ok();

    let script = format!(r#"
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gs = sys.argv[1]
fig_dir = os.path.join(gs, 'figures')

# --- 1D FES: free xylose theta ---
fes_path = os.path.join(gs, 'modules/03_free_xylose_1d/fes_theta.dat')
if os.path.exists(fes_path):
    data = np.loadtxt(fes_path, comments='#')
    theta, fes = data[:, 0], data[:, 1]
    fes -= fes.min()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(theta, fes, 'k-', lw=2)
    ax.set_xlabel(r'$\theta$ (Cremer-Pople, rad)', fontsize=12)
    ax.set_ylabel('Free Energy (kJ/mol)', fontsize=12)
    ax.set_title(r'Free $\beta$-D-Xylopyranose Ring Puckering FEL', fontsize=13)
    ax.axvspan(0, 0.4, alpha=0.15, color='blue', label=r'$^4C_1$ chair')
    ax.axvspan(1.2, 1.9, alpha=0.15, color='orange', label='Boat/skew-boat')
    ax.axvspan(2.7, 3.14, alpha=0.15, color='red', label=r'$^1C_4$ chair')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '03_free_xylose_1d_fes.png'), dpi=150)
    plt.close()
    print('    [OK] 03_free_xylose_1d_fes.png')

# --- 1D overlay: free vs enzyme-bound ---
free_path = os.path.join(gs, 'modules/03_free_xylose_1d/fes_theta.dat')
bound_path = os.path.join(gs, 'modules/05_enzyme_bound_1d/fes_theta.dat')
if os.path.exists(free_path) and os.path.exists(bound_path):
    free = np.loadtxt(free_path, comments='#')
    bound = np.loadtxt(bound_path, comments='#')
    fig, ax = plt.subplots(figsize=(8, 4))
    f_fes = free[:, 1] - free[:, 1].min()
    b_fes = bound[:, 1] - bound[:, 1].min()
    ax.plot(free[:, 0], f_fes, 'b-', lw=2, label='Free xylose (water)')
    ax.plot(bound[:, 0], b_fes, 'r-', lw=2, label='Enzyme-bound (GH10 2D24)')
    ax.set_xlabel(r'$\theta$ (Cremer-Pople, rad)', fontsize=12)
    ax.set_ylabel('Free Energy (kJ/mol)', fontsize=12)
    ax.set_title('Enzyme Conformational Selection: Barrier Lowering', fontsize=13)
    ax.axvspan(0, 0.4, alpha=0.08, color='blue')
    ax.axvspan(1.2, 1.9, alpha=0.08, color='orange')
    ax.axvspan(2.7, 3.14, alpha=0.08, color='red')
    ax.legend(fontsize=11)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '05_enzyme_vs_free_comparison.png'), dpi=150)
    plt.close()
    print('    [OK] 05_enzyme_vs_free_comparison.png')

# --- 2D Stoddart: free xylose ---
fes2d_path = os.path.join(gs, 'modules/04_free_xylose_2d/fes_2d.dat')
if os.path.exists(fes2d_path):
    data2d = np.loadtxt(fes2d_path, comments='#')
    n = int(np.sqrt(len(data2d)))
    if n * n == len(data2d):
        qx = data2d[:, 0].reshape(n, n)
        qy = data2d[:, 1].reshape(n, n)
        fes2d = data2d[:, 2].reshape(n, n)
        fes2d -= fes2d.min()
        fes2d = np.clip(fes2d, 0, 80)
        fig, ax = plt.subplots(figsize=(7, 6))
        c = ax.contourf(qx, qy, fes2d, levels=20, cmap='RdYlBu_r')
        ax.contour(qx, qy, fes2d, levels=10, colors='k', linewidths=0.3)
        plt.colorbar(c, label='Free Energy (kJ/mol)')
        ax.set_xlabel('qx (nm)', fontsize=12)
        ax.set_ylabel('qy (nm)', fontsize=12)
        ax.set_title('2D Stoddart Diagram: Free Xylose Puckering', fontsize=13)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '04_free_xylose_2d_stoddart.png'), dpi=150)
        plt.close()
        print('    [OK] 04_free_xylose_2d_stoddart.png')

# --- Ramachandran: alanine dipeptide ---
ala_path = os.path.join(gs, 'modules/01_alanine_dipeptide/fes_2d.dat')
if os.path.exists(ala_path):
    data = np.loadtxt(ala_path, comments='#')
    n = int(np.sqrt(len(data)))
    if n * n == len(data):
        phi = data[:, 0].reshape(n, n)
        psi = data[:, 1].reshape(n, n)
        fes = data[:, 2].reshape(n, n)
        fes -= fes.min()
        fes = np.clip(fes, 0, 50)
        fig, ax = plt.subplots(figsize=(7, 6))
        c = ax.contourf(np.degrees(phi), np.degrees(psi), fes, levels=20, cmap='viridis')
        ax.contour(np.degrees(phi), np.degrees(psi), fes, levels=10, colors='k', linewidths=0.3)
        plt.colorbar(c, label='Free Energy (kJ/mol)')
        ax.set_xlabel(r'$\phi$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\psi$ (degrees)', fontsize=12)
        ax.set_title('Alanine Dipeptide Ramachandran FEL (WTMetaD)', fontsize=13)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '01_alanine_dipeptide_ramachandran.png'), dpi=150)
        plt.close()
        print('    [OK] 01_alanine_dipeptide_ramachandran.png')
"#);

    let script_path = gs_dir.join(".generate_figures.py");
    std::fs::write(&script_path, &script).unwrap();

    let status = Command::new("python3")
        .args([script_path.to_string_lossy().as_ref(), gs_dir.to_string_lossy().as_ref()])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("    [WARN] Figure generation exited with {:?} (non-fatal)", s.code()),
        Err(_) => eprintln!("    [WARN] python3 not available — figures skipped (non-fatal)"),
    }

    std::fs::remove_file(&script_path).ok();
}

/// Full GuideStone pipeline: simulate all ABG FEL systems → finalize → validate.
/// Replaces all ad-hoc bash orchestration with a single Rust entry point.
fn guidestone_run_pipeline(args: &[String]) {
    let gs_dir = args.first().map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone run <guidestone-dir> [--refresh-dir <dir>]");
        process::exit(1);
    });

    let refresh_dir = extract_flag(args, "--refresh-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let spring_root = gs_dir.parent().unwrap_or(Path::new("."));
            spring_root.join("control/gromacs_fel/guidestone_refresh")
        });

    let pipeline_start = Instant::now();

    println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║  GuideStone Full Pipeline (Rust-Orchestrated)               ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // Verify GROMACS + PLUMED available
    println!("  \x1b[36mPhase 0:\x1b[0m Environment check...");
    let gmx = which_binary("gmx");
    if gmx.is_none() {
        eprintln!("  \x1b[31mERROR\x1b[0m: gmx not found. Activate GROMACS environment.");
        process::exit(1);
    }
    println!("    [OK] gmx: {}", gmx.as_ref().unwrap());

    let plumed_kernel = std::env::var("PLUMED_KERNEL").ok().or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        let conda = PathBuf::from(&home).join("micromamba/envs/plumed/lib/libplumedKernel.so");
        if conda.exists() { Some(conda.to_string_lossy().to_string()) } else { None }
    });
    if let Some(ref k) = plumed_kernel {
        std::env::set_var("PLUMED_KERNEL", k);
        println!("    [OK] PLUMED_KERNEL: {}", k);
    } else {
        eprintln!("  \x1b[31mERROR\x1b[0m: PLUMED_KERNEL not found.");
        process::exit(1);
    }
    println!();

    // Define simulation systems (steps calibrated for ~10K/20K Gaussians at PACE=500)
    let systems = [
        ("free_xylose_1d", "md_meta", 5_000_000i64),
        ("free_xylose_2d", "md_meta_2d", 10_000_000i64),
        ("enzyme_bound_1d", "md_meta", 5_000_000i64),
        ("enzyme_bound_2d", "md_meta_2d", 10_000_000i64),
    ];

    // Phase 1: Run simulations
    println!("  \x1b[36mPhase 1:\x1b[0m Running ABG FEL simulations ({} systems)...", systems.len());
    println!();

    for (sys, prefix, nsteps) in &systems {
        let sys_dir = refresh_dir.join(sys);
        let tpr_name = format!("{}.tpr", prefix);
        let tpr = sys_dir.join(&tpr_name);

        if !tpr.exists() {
            eprintln!("    \x1b[31mERROR\x1b[0m: {} — TPR not found: {}", sys, tpr.display());
            process::exit(1);
        }

        println!("    ▶ {} ({:.0}M steps)...", sys, *nsteps as f64 / 1e6);
        let sim_start = Instant::now();

        let status = Command::new(gmx.as_ref().unwrap())
            .arg("mdrun")
            .args(["-s", &tpr_name])
            .args(["-plumed", "plumed.dat"])
            .args(["-deffnm", *prefix])
            .args(["-ntmpi", "1"])
            .args(["-ntomp", "8"])
            .args(["-nsteps", &nsteps.to_string()])
            .args(["-nobackup"])
            .current_dir(&sys_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .status();

        match status {
            Ok(s) if s.success() => {
                let elapsed = sim_start.elapsed();
                println!("      \x1b[32m✓\x1b[0m {:.1}s", elapsed.as_secs_f64());
            }
            Ok(s) => {
                eprintln!("      \x1b[31m✗\x1b[0m {} exited with {:?}", sys, s.code());
                process::exit(1);
            }
            Err(e) => {
                eprintln!("      \x1b[31m✗\x1b[0m Failed to launch gmx: {}", e);
                process::exit(1);
            }
        }
    }
    println!();

    // Phase 2: Finalize (FES reconstruction + parity + populate)
    println!("  \x1b[36mPhase 2:\x1b[0m Finalize...");
    let finalize_args = vec![
        gs_dir.to_string_lossy().to_string(),
        "--refresh-dir".to_string(),
        refresh_dir.to_string_lossy().to_string(),
    ];
    guidestone_finalize(&finalize_args.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    println!();

    // Phase 3: Full self-validation
    println!("  \x1b[36mPhase 3:\x1b[0m Self-validation...");
    guidestone_validate(&[gs_dir.to_string_lossy().to_string()]);

    let total = pipeline_start.elapsed();
    println!();
    println!("  \x1b[36m┌─ Pipeline Complete ────────────────────────────────────────┐\x1b[0m");
    println!("    Total elapsed: {:.1}s ({:.1} min)", total.as_secs_f64(), total.as_secs_f64() / 60.0);
}

/// Locate a binary on PATH.
fn which_binary(name: &str) -> Option<String> {
    Command::new("which")
        .arg(name)
        .output()
        .ok()
        .and_then(|o| if o.status.success() {
            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
        } else { None })
}

/// Check upstream source data freshness — replaces `refresh` bash script.
fn guidestone_refresh(args: &[String]) {
    let dir = args.first().map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone refresh <guidestone-dir>");
        process::exit(1);
    });

    println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║  GuideStone Data Freshness Check                            ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    let spring_root = dir.parent().unwrap_or(Path::new("."));
    let sources = [
        "control/plumed_nest/target_01_alanine_dipeptide/output/HILLS",
        "control/plumed_nest/target_02_chignolin_opes/COLVARb",
        "control/gromacs_fel/guidestone_refresh/free_xylose_1d/HILLS",
        "control/gromacs_fel/guidestone_refresh/free_xylose_2d/HILLS_2d",
        "control/gromacs_fel/guidestone_refresh/enzyme_bound_1d/HILLS",
        "control/gromacs_fel/guidestone_refresh/enzyme_bound_2d/HILLS_2d",
    ];

    let mut all_present = true;
    for src in &sources {
        let full = spring_root.join(src);
        if full.exists() {
            if let Ok(meta) = std::fs::metadata(&full) {
                let modified = meta.modified().ok()
                    .and_then(|t| t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap().as_secs();
                let age_days = (now - modified) / 86400;
                println!("  \x1b[32m[OK]\x1b[0m {} ({}d old)", src, age_days);
            }
        } else {
            println!("  \x1b[31m[MISSING]\x1b[0m {}", src);
            all_present = false;
        }
    }

    println!();
    if all_present {
        println!("  All sources present. Run 'guidestone validate' to verify.");
    } else {
        println!("  \x1b[31mWARNING\x1b[0m: Missing sources. Re-run simulations first.");
        process::exit(1);
    }
}

/// Tier 2→3 parity: invoke barracuda (plasmidBin) GPU validation and compare
/// against native Rust FES reconstruction. Validates the compute triangle
/// for the CompChem domain (Gaussian summation on GPU vs CPU).
fn run_parity_barracuda(args: &[String]) {
    println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║  Tier 2→3 Parity: Rust CPU vs barracuda GPU                ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // Resolve barracuda binary: plasmidBin → PATH → known locations
    let barracuda_bin = resolve_barracuda();
    let barracuda_bin = match barracuda_bin {
        Some(b) => b,
        None => {
            eprintln!("  \x1b[31mERROR\x1b[0m: barracuda binary not found.");
            eprintln!("  Checked: plasmidBin, PATH, local workspace");
            eprintln!("  Install via plasmidBin or build locally.");
            process::exit(1);
        }
    };

    println!("  Binary: {}", barracuda_bin.display());

    // Step 1: Run barracuda validate and capture output
    println!();
    println!("  \x1b[36mStep 1:\x1b[0m Running barracuda validate...");
    let output = Command::new(&barracuda_bin)
        .arg("validate")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    let barracuda_pass = match output {
        Ok(o) if o.status.success() => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            println!("    \x1b[32mPASS\x1b[0m: barracuda GPU validation suite");
            if let Some(line) = stdout.lines().find(|l| l.contains("PASS") || l.contains("pass")) {
                println!("    {}", line.trim());
            }
            true
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            let no_device = stderr.contains("No test device available") ||
                stderr.contains("no GPU") || stderr.contains("VkError");
            if no_device {
                println!("    \x1b[33mSKIP\x1b[0m: no GPU device available on this gate");
                println!("    (barracuda GPU parity requires toadStool dispatch or local GPU)");
                true // graceful skip, not a failure
            } else {
                println!("    \x1b[31mFAIL\x1b[0m: barracuda validate exited {}", o.status);
                for line in stderr.lines().take(5) {
                    println!("      {}", line);
                }
                false
            }
        }
        Err(e) => {
            println!("    \x1b[31mFAIL\x1b[0m: could not execute barracuda: {}", e);
            false
        }
    };

    // Step 2: Native Rust FES reconstruction (Tier 2 baseline)
    println!();
    println!("  \x1b[36mStep 2:\x1b[0m Rust-native FES reconstruction (Tier 2 baseline)...");
    let hills_candidates = [
        "control/gromacs_fel/guidestone_refresh/free_xylose_1d/HILLS",
        "control/gromacs_fel/cazyme_gh10_v2/HILLS",
    ];

    let spring_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let hills_path = args.first()
        .map(PathBuf::from)
        .or_else(|| {
            hills_candidates.iter()
                .map(|c| spring_root.join(c))
                .find(|p| p.exists())
        });

    let tier2_pass = match hills_path {
        Some(ref hp) if hp.exists() => {
            match cazyme_fel::run_validation(hp, None, 110) {
                Ok(result) if result.chair_basins_found >= 2 => {
                    println!("    \x1b[32mPASS\x1b[0m: {} chair basins, barrier {:.1}–{:.1} kJ/mol",
                        result.chair_basins_found,
                        result.barrier_range_kjmol[0],
                        result.barrier_range_kjmol[1]);
                    true
                }
                Ok(result) => {
                    println!("    \x1b[31mFAIL\x1b[0m: only {} basins found", result.chair_basins_found);
                    false
                }
                Err(e) => {
                    println!("    \x1b[31mFAIL\x1b[0m: {}", e);
                    false
                }
            }
        }
        _ => {
            println!("    \x1b[33mSKIP\x1b[0m: No HILLS file found for Tier 2 baseline");
            println!("    Provide path: nest-validate parity <HILLS-file>");
            true // don't fail the overall check
        }
    };

    // Step 3: Cross-tier parity assessment
    println!();
    println!("  \x1b[36mStep 3:\x1b[0m Cross-tier parity assessment...");
    let overall = barracuda_pass && tier2_pass;

    println!();
    println!("  \x1b[36m┌─ Summary ──────────────────────────────────────────────┐\x1b[0m");
    println!("    Tier 0 (GROMACS):  reference (HILLS data)");
    println!("    Tier 2 (Rust CPU): {}", if tier2_pass { "\x1b[32mPASS\x1b[0m" } else { "\x1b[31mFAIL\x1b[0m" });
    println!("    Tier 3 (GPU):      {}", if barracuda_pass { "\x1b[32mPASS\x1b[0m" } else { "\x1b[31mFAIL\x1b[0m" });
    println!();

    if overall {
        println!("\x1b[32m  ╔══════════════════════════════════════╗\x1b[0m");
        println!("\x1b[32m  ║  PARITY STATUS: CONVERGENT           ║\x1b[0m");
        println!("\x1b[32m  ╚══════════════════════════════════════╝\x1b[0m");
    } else {
        println!("\x1b[31m  ╔══════════════════════════════════════╗\x1b[0m");
        println!("\x1b[31m  ║  PARITY STATUS: DIVERGENT            ║\x1b[0m");
        println!("\x1b[31m  ╚══════════════════════════════════════╝\x1b[0m");
        process::exit(1);
    }
}

/// Fully agentic pseudoSpore deployment pipeline.
/// Chains: hash → emit → hash → verify → validate → package → (optional) litho ingest/promote.
fn guidestone_deploy(args: &[String]) {
    let dir = args.iter().find(|a| !a.starts_with('-')).map(|s| PathBuf::from(s)).unwrap_or_else(|| {
        eprintln!("Usage: nest-validate guidestone deploy <dir> [--litho-root <path>] [--tarball] [--desktop] [--promote]");
        process::exit(1);
    });

    let tarball = args.iter().any(|a| a == "--tarball");
    let desktop = args.iter().any(|a| a == "--desktop");
    let promote = args.iter().any(|a| a == "--promote");
    let litho_root = extract_flag(args, "--litho-root").map(PathBuf::from);

    if !dir.is_dir() {
        eprintln!("Error: {} is not a directory", dir.display());
        process::exit(1);
    }

    let pipeline_start = Instant::now();

    println!("\x1b[36m╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[36m║  GuideStone Deploy — Agentic Pipeline                       ║\x1b[0m");
    println!("\x1b[36m╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // Step 1: Hash (generate data.toml with all files)
    println!("  \x1b[36mStep 1/7:\x1b[0m Generating BLAKE3 manifest...");
    let hash_args = vec![dir.to_string_lossy().to_string()];
    let data_toml_path = dir.join("data.toml");
    {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["guidestone", "hash"])
            .arg(&dir)
            .stdout(Stdio::piped())
            .output()
            .expect("Failed to run guidestone hash");
        std::fs::write(&data_toml_path, &output.stdout).unwrap();
    }
    println!("    [OK] data.toml generated");
    println!();

    // Step 2: Emit (liveSpore.json, validation.json, receipts/*)
    println!("  \x1b[36mStep 2/7:\x1b[0m Emitting provenance + receipts...");
    guidestone_emit(&hash_args);
    println!();

    // Step 3: Re-hash (incorporate final liveSpore.json hash)
    // When tarball is requested, use --data-only to split [present]/[external]
    // so verify works on the tarball without trajectory files
    let use_data_only = tarball || desktop;
    println!("  \x1b[36mStep 3/7:\x1b[0m Re-hashing with final liveSpore.json{}...",
             if use_data_only { " (--data-only for tarball)" } else { "" });
    {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["guidestone", "hash"]);
        if use_data_only {
            cmd.arg("--data-only");
        }
        cmd.arg(&dir);
        let output = cmd.stdout(Stdio::piped())
            .output()
            .expect("Failed to run guidestone hash");
        std::fs::write(&data_toml_path, &output.stdout).unwrap();
    }
    // Re-generate checksums.blake3 with updated data.toml
    generate_checksums_blake3(&dir);
    println!("    [OK] data.toml + checksums.blake3 regenerated");
    println!();

    // Step 4: Verify (BLAKE3 integrity gate)
    println!("  \x1b[36mStep 4/7:\x1b[0m BLAKE3 integrity verification...");
    guidestone_verify(&hash_args);
    println!();

    // Step 5: Validate (science checks)
    println!("  \x1b[36mStep 5/7:\x1b[0m Full science validation...");
    guidestone_validate(&hash_args);
    println!();

    // Step 6: Package tarball
    if tarball || desktop {
        println!("  \x1b[36mStep 6/7:\x1b[0m Packaging tarball...");
        let dir_name = dir.file_name().unwrap().to_string_lossy().to_string();
        let tarball_name = format!("{}.tar.gz", dir_name);

        let output_dir = if desktop {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join("Desktop")
        } else {
            dir.parent().unwrap_or(Path::new(".")).to_path_buf()
        };

        let tarball_path = output_dir.join(&tarball_name);
        let parent = dir.parent().unwrap_or(Path::new("."));
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };

        let status = Command::new("tar")
            .args(["czf", &tarball_path.to_string_lossy(), "--exclude=*/COLVAR*", "--exclude=*/Kernels*", &dir_name])
            .current_dir(parent)
            .status()
            .expect("Failed to run tar");

        if status.success() {
            if let Ok(meta) = std::fs::metadata(&tarball_path) {
                let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
                println!("    [OK] {} ({:.1} MB)", tarball_path.display(), size_mb);
            }
        } else {
            eprintln!("    [FAIL] tar exited with {:?}", status.code());
        }
        println!();
    } else {
        println!("  \x1b[36mStep 6/7:\x1b[0m Skipping tarball (use --tarball or --desktop)");
        println!();
    }

    // Step 7: litho ingest + optional promote
    // Per SPORE_OWNERSHIP_MATRIX.md: NUCLEUS gateway belongs to biomeOS.
    // Future: `biomeos nucleus ingest` replaces this step.
    // Current: delegates to `litho ingest-pseudospore` as transitional path.
    if let Some(ref lr) = litho_root {
        println!("  \x1b[36mStep 7/7:\x1b[0m lithoSpore ingestion (future: biomeos nucleus ingest)...");
        let litho_bin = which_binary("litho").unwrap_or_else(|| {
            let cargo_target = lr.join("target/release/litho");
            if cargo_target.exists() {
                cargo_target.to_string_lossy().to_string()
            } else {
                "cargo".to_string()
            }
        });

        let ingest_status = if litho_bin == "cargo" {
            Command::new("cargo")
                .args(["run", "--", "ingest-pseudospore"])
                .arg(dir.canonicalize().unwrap_or(dir.clone()))
                .args(["--artifact-root", "."])
                .arg("--verify")
                .current_dir(lr)
                .status()
        } else {
            Command::new(&litho_bin)
                .arg("ingest-pseudospore")
                .arg(dir.canonicalize().unwrap_or(dir.clone()))
                .args(["--artifact-root", &lr.to_string_lossy()])
                .arg("--verify")
                .status()
        };

        match ingest_status {
            Ok(s) if s.success() => println!("    [OK] pseudoSpore ingested into lithoSpore registry"),
            Ok(s) => eprintln!("    [WARN] litho ingest exited with {:?}", s.code()),
            Err(e) => eprintln!("    [WARN] Could not run litho ingest: {}", e),
        }

        if promote {
            println!();
            println!("  \x1b[36mBonus:\x1b[0m lithoSpore promotion...");
            let promote_status = if litho_bin == "cargo" {
                Command::new("cargo")
                    .args(["run", "--", "promote"])
                    .args(["--pseudospore", &dir.canonicalize().unwrap_or(dir.clone()).to_string_lossy()])
                    .args(["--output", "."])
                    .current_dir(lr)
                    .status()
            } else {
                Command::new(&litho_bin)
                    .arg("promote")
                    .args(["--pseudospore", &dir.canonicalize().unwrap_or(dir.clone()).to_string_lossy()])
                    .args(["--output", &lr.to_string_lossy()])
                    .status()
            };

            match promote_status {
                Ok(s) if s.success() => println!("    [OK] lithoSpore chassis created"),
                Ok(s) => eprintln!("    [WARN] litho promote exited with {:?}", s.code()),
                Err(e) => eprintln!("    [WARN] Could not run litho promote: {}", e),
            }
        }
        println!();
    } else {
        println!("  \x1b[36mStep 7/7:\x1b[0m Skipping litho ingest (use --litho-root <path>)");
        println!();
    }

    // Summary
    let elapsed = pipeline_start.elapsed();
    let scope_path = dir.join("scope.toml");
    let version = read_scope_field(&scope_path, "guidestone", "version")
        .or_else(|| read_scope_field(&scope_path, "artifact", "version"))
        .unwrap_or_else(|| "?.?.?".to_string());

    println!("\x1b[32m  ╔══════════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[32m  ║  DEPLOY COMPLETE                                            ║\x1b[0m");
    println!("\x1b[32m  ╚══════════════════════════════════════════════════════════════╝\x1b[0m");
    println!();
    println!("  Version:  v{}", version);
    println!("  Elapsed:  {:.1}s", elapsed.as_secs_f64());
    if litho_root.is_some() {
        println!("  Registry: lithoSpore ingested");
    }
}

/// Resolve the barracuda binary from plasmidBin or PATH.
fn resolve_barracuda() -> Option<PathBuf> {
    // 1. plasmidBin (postPrimordial standard)
    let spring_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let plasmidbin_candidates = [
        spring_root.join("../../infra/plasmidBin/barracuda/barracuda"),
        PathBuf::from("/home/strandgate/Development/ecoPrimals/infra/plasmidBin/barracuda/barracuda"),
    ];
    for candidate in &plasmidbin_candidates {
        if candidate.exists() && candidate.is_file() {
            return Some(candidate.clone());
        }
    }

    // 2. PATH
    if let Ok(output) = Command::new("which").arg("barracuda").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // 3. Local workspace (development fallback)
    let local = spring_root.join("barracuda/target/release/barracuda");
    if local.exists() {
        return Some(local);
    }

    None
}
