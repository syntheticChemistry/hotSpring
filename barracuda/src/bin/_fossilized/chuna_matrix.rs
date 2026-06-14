// SPDX-License-Identifier: AGPL-3.0-or-later

//! Chuna Engine: Matrix — task orchestration for production data generation.
//!
//! Manages a persistent task matrix of (ensemble_params × observable_set) cells,
//! supporting both continuous sweeps and on-demand targeted requests. Designed
//! for Chuna's collaboration with Bazavov at LANL.
//!
//! # Subcommands
//!
//! - `init`   — Create a new sweep matrix or empty matrix
//! - `add`    — Add an on-demand target task
//! - `status` — Show matrix with cost estimates and progress
//! - `run`    — Execute next tasks within a time budget
//! - `export` — Generate QCDml package from completed tasks
//!
//! # Usage
//!
//! ```bash
//! # Initialize a quenched β-sweep:
//! cargo run --release --bin chuna_matrix -- init \
//!   --betas=5.8,5.9,6.0 --lattices=8,12,16 \
//!   --observables=flow,topology,wilson --configs=50 --file=matrix.json
//!
//! # Add on-demand measurement:
//! cargo run --release --bin chuna_matrix -- add \
//!   --beta=6.0 --lattice=16 --observable=pbp --mass=0.1 --file=matrix.json
//!
//! # Check progress:
//! cargo run --release --bin chuna_matrix -- status --file=matrix.json
//!
//! # Run tasks within time budget:
//! cargo run --release --bin chuna_matrix -- run \
//!   --max-hours=2 --hardware=cpu --file=matrix.json
//!
//! # Export completed work as QCDml:
//! cargo run --release --bin chuna_matrix -- export \
//!   --format=qcdml --outdir=export/ --file=matrix.json
//! ```

use hotspring_barracuda::lattice::gradient_flow::{
    FlowIntegrator, find_t0, find_w0, run_flow, topological_charge,
};
use hotspring_barracuda::lattice::hmc::{HmcConfig, IntegratorType, hmc_trajectory};
use hotspring_barracuda::lattice::ildg::{IldgMetadata, ildg_crc, write_gauge_config_file};
use hotspring_barracuda::lattice::measurement::{
    ConfigEntry, ConfigMeasurement, EnsembleManifest, FlowPoint, FlowResults, ImplementationInfo,
    TopologyResults, WilsonLoopEntry,
};
use hotspring_barracuda::lattice::process_catalog::{
    CostModel, HardwareTier, PhysicsProcess, ProcessParams,
};
use hotspring_barracuda::lattice::qcdml::{QcdmlEnsembleInfo, generate_ensemble_xml};
use hotspring_barracuda::lattice::task_matrix::{SweepParams, TaskMatrix};
use hotspring_barracuda::lattice::wilson::Lattice;

use std::time::{Duration, Instant};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    if argv.is_empty() {
        print_usage();
        std::process::exit(1);
    }

    let subcmd = argv[0].as_str();
    let rest = &argv[1..];

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Chuna Engine: Task Matrix Orchestrator                    ║");
    println!("║  hotSpring-barracuda — production data generation          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    match subcmd {
        "init" => cmd_init(rest),
        "add" => cmd_add(rest),
        "status" => cmd_status(rest),
        "run" => cmd_run(rest),
        "export" => cmd_export(rest),
        _ => {
            eprintln!("Unknown subcommand: {subcmd}");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("chuna_matrix — Task matrix orchestrator for production data generation");
    eprintln!();
    eprintln!("Subcommands:");
    eprintln!("  init     Create a new task matrix (sweep or empty)");
    eprintln!("  add      Add an on-demand target task");
    eprintln!("  status   Show matrix progress and cost estimates");
    eprintln!("  run      Execute next ready tasks within time budget");
    eprintln!("  export   Generate QCDml package from completed data");
    eprintln!();
    eprintln!("Examples:");
    eprintln!(
        "  chuna_matrix init --betas=5.8,6.0 --lattices=8,12 --observables=flow,wilson --file=matrix.json"
    );
    eprintln!("  chuna_matrix add --beta=6.0 --lattice=16 --observable=pbp --file=matrix.json");
    eprintln!("  chuna_matrix status --file=matrix.json");
    eprintln!("  chuna_matrix run --max-hours=4 --file=matrix.json");
    eprintln!("  chuna_matrix export --format=qcdml --outdir=export/ --file=matrix.json");
}

fn parse_kv<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    let prefix = format!("--{key}=");
    args.iter()
        .find(|a| a.starts_with(&prefix))
        .map(|a| &a[prefix.len()..])
}

fn parse_hardware(args: &[String]) -> HardwareTier {
    match parse_kv(args, "hardware") {
        Some("gpu" | "consumer-gpu") => HardwareTier::ConsumerGpu,
        Some("hpc" | "hpc-gpu") => HardwareTier::HpcGpu,
        Some("cluster") => HardwareTier::Cluster,
        _ => HardwareTier::Cpu,
    }
}

fn matrix_file(args: &[String]) -> String {
    parse_kv(args, "file")
        .unwrap_or("task_matrix.json")
        .to_string()
}

fn parse_observable(name: &str) -> Option<PhysicsProcess> {
    match name.to_lowercase().as_str() {
        "flow" | "gradientflow" => Some(PhysicsProcess::GradientFlow),
        "wilson" | "wilsonloops" => Some(PhysicsProcess::WilsonLoops),
        "topology" | "topologicalcharge" => Some(PhysicsProcess::TopologicalCharge),
        "pbp" | "chiralcondensate" => Some(PhysicsProcess::ChiralCondensate),
        "correlators" => Some(PhysicsProcess::Correlators),
        "scale" | "scalesetting" => Some(PhysicsProcess::ScaleSetting),
        "generate" => Some(PhysicsProcess::Generate),
        _ => None,
    }
}

fn cmd_init(args: &[String]) {
    let betas: Vec<f64> = parse_kv(args, "betas")
        .unwrap_or("5.8,6.0,6.2")
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    let lattices: Vec<usize> = parse_kv(args, "lattices")
        .unwrap_or("8,12,16")
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    let observables: Vec<PhysicsProcess> = parse_kv(args, "observables")
        .unwrap_or("flow,topology,wilson")
        .split(',')
        .filter_map(parse_observable)
        .collect();

    let n_configs: usize = parse_kv(args, "configs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mass: f64 = parse_kv(args, "mass")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let nf: usize = parse_kv(args, "nf")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let hardware = parse_hardware(args);
    let file = matrix_file(args);

    let matrix_id = parse_kv(args, "id").unwrap_or("chuna_sweep").to_string();

    let nt_override: Option<Vec<usize>> =
        parse_kv(args, "nt").map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect());

    let sweep = SweepParams {
        betas,
        lattice_sizes: lattices,
        nt_override,
        observables,
        n_configs,
        mass,
        nf,
        default_params: ProcessParams {
            n_therm: Some(200),
            meas_interval: Some(10),
            t_max: Some(4.0),
            epsilon: Some(0.01),
            r_max: Some(4),
            n_sources: Some(10),
            cg_tol: Some(1e-8),
            seed: Some(42),
        },
    };

    let matrix = TaskMatrix::from_sweep(&matrix_id, hardware, &sweep);

    println!("\n  Created task matrix: {}", matrix.matrix_id);
    println!("{}", matrix.summary_table());

    matrix.save(&file).expect("save matrix");
    println!("  Saved to: {file}");
}

fn cmd_add(args: &[String]) {
    let file = matrix_file(args);
    let mut matrix = TaskMatrix::load(&file).unwrap_or_else(|e| {
        eprintln!("Cannot load matrix from {file}: {e}");
        std::process::exit(1);
    });

    let beta: f64 = parse_kv(args, "beta")
        .and_then(|s| s.parse().ok())
        .unwrap_or(6.0);

    let lattice: usize = parse_kv(args, "lattice")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let observable = parse_kv(args, "observable")
        .and_then(parse_observable)
        .unwrap_or(PhysicsProcess::GradientFlow);

    let n_configs: usize = parse_kv(args, "configs")
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mass: f64 = parse_kv(args, "mass")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let nt: usize = parse_kv(args, "nt")
        .and_then(|s| s.parse().ok())
        .unwrap_or(lattice);
    let dims = [lattice, lattice, lattice, nt];
    let dims_tag = if lattice == nt {
        format!("L{lattice}")
    } else {
        format!("L{lattice}x{nt}")
    };
    let task_id = format!(
        "ondemand_{}_b{beta:.2}_{dims_tag}",
        observable.name().to_lowercase()
    );

    use hotspring_barracuda::lattice::process_catalog::ProcessSpec;
    let spec = ProcessSpec {
        process: observable,
        dims,
        beta,
        mass,
        nf: 0,
        n_configs,
        params: ProcessParams::default(),
    };

    matrix.add_target(&task_id, spec);
    matrix.save(&file).expect("save matrix");

    println!("\n  Added on-demand task: {task_id}");
    println!("{}", matrix.summary_table());
}

fn cmd_status(args: &[String]) {
    let file = matrix_file(args);
    let matrix = TaskMatrix::load(&file).unwrap_or_else(|e| {
        eprintln!("Cannot load matrix from {file}: {e}");
        std::process::exit(1);
    });

    println!();
    println!("{}", matrix.summary_table());

    let remaining = matrix.estimated_remaining_seconds();
    let counts = matrix.status_counts();
    let pct = if counts.completed + counts.pending + counts.running > 0 {
        100.0 * counts.completed as f64
            / (counts.completed + counts.pending + counts.running) as f64
    } else {
        0.0
    };
    println!(
        "  Progress: {:.1}% complete  |  Remaining: {}",
        pct,
        CostModel::format_duration(remaining)
    );
}

fn cmd_run(args: &[String]) {
    let file = matrix_file(args);
    let mut matrix = TaskMatrix::load(&file).unwrap_or_else(|e| {
        eprintln!("Cannot load matrix from {file}: {e}");
        std::process::exit(1);
    });

    let max_hours: f64 = parse_kv(args, "max-hours")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let max_seconds = max_hours * 3600.0;

    let outdir = parse_kv(args, "outdir")
        .unwrap_or("chuna_matrix_output")
        .to_string();
    std::fs::create_dir_all(&outdir).expect("create output dir");
    let meas_dir = format!("{outdir}/measurements");
    std::fs::create_dir_all(&meas_dir).expect("create measurements dir");

    println!("\n  Time budget: {max_hours:.1}h ({max_seconds:.0}s)");
    println!("  Output: {outdir}");

    let run_start = Instant::now();
    let mut tasks_run = 0usize;
    let budget = Duration::from_secs_f64(max_seconds);

    while run_start.elapsed() < budget {
        let Some(idx) = matrix.next_ready() else {
            println!("  No more ready tasks.");
            break;
        };

        let task = &matrix.tasks[idx];
        let budget_left = budget.saturating_sub(run_start.elapsed()).as_secs_f64();
        if task.estimated_seconds > budget_left * 2.0 && tasks_run > 0 {
            println!(
                "  Next task '{}' estimated at {} — exceeds remaining budget, stopping.",
                task.id,
                CostModel::format_duration(task.estimated_seconds)
            );
            break;
        }

        println!(
            "\n  ─── Running: {} ({}) ───",
            task.id,
            task.spec.process.name()
        );

        matrix.mark_running(idx);
        let task_start = Instant::now();

        let result = execute_task(&matrix.tasks[idx], &outdir, &meas_dir);

        let wall = task_start.elapsed().as_secs_f64();
        match result {
            Ok(output_path) => {
                println!("  ✓ Completed in {wall:.1}s");
                matrix.mark_completed(idx, wall, output_path);
            }
            Err(e) => {
                println!("  ✗ Failed: {e}");
                matrix.mark_failed(idx, &e);
            }
        }

        matrix.save(&file).expect("save matrix");
        tasks_run += 1;
    }

    println!(
        "\n═══ Session complete: {} tasks in {:.1}s ═══",
        tasks_run,
        run_start.elapsed().as_secs_f64()
    );
    println!("{}", matrix.summary_table());
}

fn execute_task(
    task: &hotspring_barracuda::lattice::task_matrix::TaskCell,
    outdir: &str,
    meas_dir: &str,
) -> Result<Option<String>, String> {
    let spec = &task.spec;
    let [ns, _, _, nt] = spec.dims;
    let dims = spec.dims;

    match spec.process {
        PhysicsProcess::Generate => {
            let n_therm = spec.params.n_therm.unwrap_or(200);
            let meas_interval = spec.params.meas_interval.unwrap_or(10);
            let seed = spec.params.seed.unwrap_or(42);
            let ensemble_id = format!("hotspring_b{:.2}_L{ns}x{nt}", spec.beta);
            let ens_dir = format!("{outdir}/{ensemble_id}");
            std::fs::create_dir_all(&ens_dir).map_err(|e| e.to_string())?;
            let ens_meas_dir = format!("{ens_dir}/measurements");
            std::fs::create_dir_all(&ens_meas_dir).map_err(|e| e.to_string())?;

            let mut lattice = Lattice::hot_start(dims, spec.beta, seed);
            let mut hmc_cfg = HmcConfig {
                n_md_steps: 20,
                dt: 0.05,
                seed,
                integrator: IntegratorType::Omelyan,
            };

            // Thermalize
            for _ in 0..n_therm {
                hmc_trajectory(&mut lattice, &mut hmc_cfg);
            }

            // Generate configs
            let mut manifest = EnsembleManifest::new(&ensemble_id, dims, spec.beta);
            manifest.algorithm.n_therm = n_therm;
            manifest.algorithm.meas_interval = meas_interval;
            manifest.algorithm.seed = seed;

            let mut traj = n_therm;
            for _ in 0..spec.n_configs {
                for _ in 0..meas_interval {
                    hmc_trajectory(&mut lattice, &mut hmc_cfg);
                    traj += 1;
                }

                let meta = IldgMetadata::for_lattice(&lattice, traj);
                let conf_file = format!("conf_{traj:06}.lime");
                let conf_path = format!("{ens_dir}/{conf_file}");
                write_gauge_config_file(&conf_path, &lattice, &meta).map_err(|e| e.to_string())?;

                let file_bytes = std::fs::read(&conf_path).map_err(|e| e.to_string())?;
                let crc = ildg_crc(&file_bytes);

                manifest.configs.push(ConfigEntry {
                    trajectory: traj,
                    filename: conf_file,
                    ildg_lfn: meta.lfn.clone(),
                    checksum_crc32: None,
                    checksum_ildg_crc: Some(crc),
                    plaquette: lattice.average_plaquette(),
                });

                println!("    traj={traj:>5}  ⟨P⟩={:.6}", lattice.average_plaquette());
            }

            std::fs::write(format!("{ens_dir}/ensemble.json"), manifest.to_json())
                .map_err(|e| e.to_string())?;

            Ok(Some(ens_dir))
        }
        PhysicsProcess::GradientFlow | PhysicsProcess::TopologicalCharge => {
            let ensemble_id = format!("hotspring_b{:.2}_L{ns}x{nt}", spec.beta);
            let ens_dir = format!("{outdir}/{ensemble_id}");

            let configs = collect_lime_files(&ens_dir);
            if configs.is_empty() {
                return Err(format!("No ILDG configs in {ens_dir}"));
            }

            let t_max = spec.params.t_max.unwrap_or(4.0);
            let eps = spec.params.epsilon.unwrap_or(0.01);

            for path in &configs {
                let (lattice, meta) =
                    hotspring_barracuda::lattice::ildg::read_gauge_config_file(path)
                        .map_err(|e| e.to_string())?;

                let mut flow_lat = lattice.clone();
                let measure_interval = (0.05 / eps).max(1.0) as usize;
                let results = run_flow(
                    &mut flow_lat,
                    FlowIntegrator::Lscfrk3w7,
                    eps,
                    t_max,
                    measure_interval,
                );
                let t0 = find_t0(&results);
                let w0 = find_w0(&results);
                let q = topological_charge(&flow_lat);

                let mut meas =
                    ConfigMeasurement::new(&meta.ensemble_id, meta.trajectory, &meta.lfn);
                meas.gauge.plaquette = lattice.average_plaquette();
                meas.implementation = Some(ImplementationInfo::auto_detect_cpu_only());
                meas.flow = Some(FlowResults {
                    integrator: "Lscfrk3w7".to_string(),
                    epsilon: eps,
                    t_max,
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
                    flow_time: t_max,
                });

                let meas_path = format!("{meas_dir}/flow_{:06}.json", meta.trajectory);
                std::fs::write(&meas_path, meas.to_json()).map_err(|e| e.to_string())?;

                let t0s = t0.map_or_else(|| "N/A".to_string(), |v| format!("{v:.4}"));
                println!("    traj={}  t₀={}  Q={q:.2}", meta.trajectory, t0s);
            }

            Ok(Some(meas_dir.to_string()))
        }
        PhysicsProcess::WilsonLoops => {
            let ensemble_id = format!("hotspring_b{:.2}_L{ns}x{nt}", spec.beta);
            let ens_dir = format!("{outdir}/{ensemble_id}");
            let configs = collect_lime_files(&ens_dir);
            if configs.is_empty() {
                return Err(format!("No ILDG configs in {ens_dir}"));
            }

            let r_max = spec.params.r_max.unwrap_or(4).min(ns / 2);

            for path in &configs {
                let (lattice, meta) =
                    hotspring_barracuda::lattice::ildg::read_gauge_config_file(path)
                        .map_err(|e| e.to_string())?;

                let mut meas =
                    ConfigMeasurement::new(&meta.ensemble_id, meta.trajectory, &meta.lfn);
                meas.gauge.plaquette = lattice.average_plaquette();
                meas.implementation = Some(ImplementationInfo::auto_detect_cpu_only());

                let mut wl_entries = Vec::new();
                for r in 1..=r_max {
                    for t in 1..=r_max {
                        let val = lattice.spatial_temporal_wilson_loop(r, t);
                        wl_entries.push(WilsonLoopEntry { r, t, value: val });
                    }
                }
                meas.wilson_loops = Some(wl_entries);

                let meas_path = format!("{meas_dir}/wilson_{:06}.json", meta.trajectory);
                std::fs::write(&meas_path, meas.to_json()).map_err(|e| e.to_string())?;

                println!(
                    "    traj={}  W(1,1)={:.6}",
                    meta.trajectory,
                    lattice.spatial_temporal_wilson_loop(1, 1)
                );
            }

            Ok(Some(meas_dir.to_string()))
        }
        PhysicsProcess::ChiralCondensate => {
            let ensemble_id = format!("hotspring_b{:.2}_L{ns}x{nt}", spec.beta);
            let ens_dir = format!("{outdir}/{ensemble_id}");
            let configs = collect_lime_files(&ens_dir);
            if configs.is_empty() {
                return Err(format!("No ILDG configs in {ens_dir}"));
            }

            let mass = if spec.mass > 0.0 { spec.mass } else { 0.1 };
            let n_sources = spec.params.n_sources.unwrap_or(10);
            let cg_tol = spec.params.cg_tol.unwrap_or(1e-8);
            let seed = spec.params.seed.unwrap_or(42);

            for path in &configs {
                let (lattice, meta) =
                    hotspring_barracuda::lattice::ildg::read_gauge_config_file(path)
                        .map_err(|e| e.to_string())?;

                let result = hotspring_barracuda::lattice::correlator::chiral_condensate_stochastic(
                    &lattice, mass, cg_tol, 5000, n_sources, seed,
                );

                let mut meas =
                    ConfigMeasurement::new(&meta.ensemble_id, meta.trajectory, &meta.lfn);
                meas.gauge.plaquette = lattice.average_plaquette();
                meas.implementation = Some(ImplementationInfo::auto_detect_cpu_only());
                meas.fermion = Some(
                    hotspring_barracuda::lattice::measurement::FermionObservables {
                        chiral_condensate: result.condensate,
                        chiral_condensate_error: result.error,
                        n_sources: result.n_sources,
                        mass,
                    },
                );

                let meas_path = format!("{meas_dir}/pbp_{:06}.json", meta.trajectory);
                std::fs::write(&meas_path, meas.to_json()).map_err(|e| e.to_string())?;

                println!(
                    "    traj={}  ⟨ψ̄ψ⟩={:.6} ± {:.6}",
                    meta.trajectory, result.condensate, result.error
                );
            }

            Ok(Some(meas_dir.to_string()))
        }
        PhysicsProcess::Correlators => {
            eprintln!(
                "    [SKIP] Correlator measurement requires point-to-point propagator \
                       computation (Dirac inversion on each source). Blocked on GPU CG \
                       solver integration for full-lattice inversions."
            );
            Ok(None)
        }
        PhysicsProcess::ScaleSetting => {
            eprintln!(
                "    [SKIP] Scale setting (Wilson flow t₀/w₀) requires gradient flow \
                       on production-size lattices with statistical averaging. Use \
                       validate_gradient_flow for single-config flow validation."
            );
            Ok(None)
        }
    }
}

fn collect_lime_files(dir: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<String> = entries
        .filter_map(std::result::Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("lime"))
        })
        .map(|e| e.path().display().to_string())
        .collect();
    files.sort();
    files
}

fn cmd_export(args: &[String]) {
    let file = matrix_file(args);
    let matrix = TaskMatrix::load(&file).unwrap_or_else(|e| {
        eprintln!("Cannot load matrix from {file}: {e}");
        std::process::exit(1);
    });

    let outdir = parse_kv(args, "outdir").unwrap_or("export").to_string();
    std::fs::create_dir_all(&outdir).expect("create export dir");

    let format = parse_kv(args, "format").unwrap_or("qcdml");

    println!(
        "\n  Exporting completed tasks from matrix: {}",
        matrix.matrix_id
    );
    println!("  Format: {format}");
    println!("  Output: {outdir}");

    let completed: Vec<_> = matrix
        .tasks
        .iter()
        .filter(|t| {
            t.status == hotspring_barracuda::lattice::process_catalog::ProcessStatus::Completed
        })
        .collect();

    if completed.is_empty() {
        println!("  No completed tasks to export.");
        return;
    }

    // Collect unique ensemble points from completed Generate tasks
    let gen_tasks: Vec<_> = completed
        .iter()
        .filter(|t| t.spec.process == PhysicsProcess::Generate)
        .collect();

    for task in &gen_tasks {
        let [ns, _, _, nt] = task.spec.dims;
        let ensemble_id = format!("hotspring_b{:.2}_L{ns}x{nt}", task.spec.beta);

        let manifest = EnsembleManifest::new(&ensemble_id, task.spec.dims, task.spec.beta);
        let ens_info = QcdmlEnsembleInfo::for_manifest(&manifest);
        let ens_xml = generate_ensemble_xml(&manifest, &ens_info);

        let xml_path = format!("{outdir}/{ensemble_id}_ensemble.xml");
        std::fs::write(&xml_path, &ens_xml).expect("write ensemble XML");
        println!("  → {xml_path}");
    }

    // Write a summary manifest
    let summary = serde_json::json!({
        "matrix_id": matrix.matrix_id,
        "exported_at": hotspring_barracuda::lattice::measurement::iso8601_now(),
        "format": format,
        "completed_tasks": completed.len(),
        "ensembles": gen_tasks.iter().map(|t| {
            let [ns, _, _, nt] = t.spec.dims;
            serde_json::json!({
                "ensemble_id": format!("hotspring_b{:.2}_L{ns}x{nt}", t.spec.beta),
                "dims": t.spec.dims,
                "beta": t.spec.beta,
                "n_configs": t.spec.n_configs,
            })
        }).collect::<Vec<_>>(),
    });

    let summary_path = format!("{outdir}/export_manifest.json");
    std::fs::write(
        &summary_path,
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write summary");
    println!("  → {summary_path}");

    println!(
        "\n  Exported {} completed tasks ({} ensemble points).",
        completed.len(),
        gen_tasks.len()
    );
}
