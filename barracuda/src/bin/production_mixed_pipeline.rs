// SPDX-License-Identifier: AGPL-3.0-only

//! Production Mixed-Pipeline β-Scan: RTX 3090 (DF64) + NPU Offloading + Titan V Oracle
//!
//! Exp 022: Full NPU offload of screening/classification workloads, freeing the
//! GPU to focus exclusively on HMC physics. Builds on:
//! - Exp 015 (paused mixed pipeline, 3-substrate architecture)
//! - Exp 020 (NPU characterization — therm detector 87.5%, reject predictor 96.2%)
//! - Exp 021 (cross-substrate ESN — NPU at 2.8µs/step)
//!
//! # NPU Offload Architecture
//!
//! ```text
//! GPU (RTX 3090)             NPU Worker Thread
//! ──────────────             ─────────────────
//! HMC trajectory (7.6s) ──►  A: Thermalization detect (0.3ms)
//!   plaquette readback   ──►  B: Rejection predict (0.3ms)
//!                        ──►  C: Phase classify (0.3ms)
//!                        ◄──  D: Adaptive β steering
//! ```
//!
//! Total NPU overhead: ~1.2ms per trajectory (0.016% of 7.6s).
//!
//! # Usage
//!
//! ```bash
//! HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_mixed_pipeline -- \
//!   --lattice=32 --seed=42 --output=results/mixed_pipeline_022.json \
//!   --trajectory-log=results/mixed_pipeline_022_trajectories.jsonl
//! ```

use hotspring_barracuda::error::HotSpringError;
use hotspring_barracuda::gpu::{discover_primary_and_secondary_adapters, GpuF64};
use hotspring_barracuda::lattice::gpu_hmc::GpuHmcStreamingPipelines;
use hotspring_barracuda::production::beta_scan::{
    run_beta_points_npu, spawn_quenched_npu_worker, QuenchedNpuRequest, QuenchedNpuResponse,
    QuenchedNpuStats,
};
use hotspring_barracuda::production::mixed_summary::{print_mixed_summary, write_mixed_json};
use hotspring_barracuda::production::titan_validation::run_titan_validation;
use hotspring_barracuda::production::BetaResult;

use std::io::Write;
use std::time::Instant;

use hotspring_barracuda::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;

// ═══════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════

struct CliArgs {
    lattice: usize,
    n_therm: usize,
    seed: u64,
    output: Option<String>,
    trajectory_log: Option<String>,
    /// Path to a previous trajectory log (JSONL) to bootstrap ESN before Phase 1.
    bootstrap_from: Option<String>,
    /// Path to save final ESN weights (JSON) for the next run's bootstrap.
    save_weights: Option<String>,
}

fn parse_args() -> CliArgs {
    let mut lattice = 32;
    let mut n_therm = 200;
    let mut seed = 42u64;
    let mut output = None;
    let mut trajectory_log = None;
    let mut bootstrap_from = None;
    let mut save_weights = None;

    for arg in std::env::args().skip(1) {
        if let Some(val) = arg.strip_prefix("--lattice=") {
            lattice = val.parse().expect("--lattice=N");
        } else if let Some(val) = arg.strip_prefix("--therm=") {
            n_therm = val.parse().expect("--therm=N");
        } else if let Some(val) = arg.strip_prefix("--seed=") {
            seed = val.parse().expect("--seed=N");
        } else if let Some(val) = arg.strip_prefix("--output=") {
            output = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--trajectory-log=") {
            trajectory_log = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--bootstrap-from=") {
            bootstrap_from = Some(val.to_string());
        } else if let Some(val) = arg.strip_prefix("--save-weights=") {
            save_weights = Some(val.to_string());
        }
    }

    CliArgs {
        lattice,
        n_therm,
        seed,
        output,
        trajectory_log,
        bootstrap_from,
        save_weights,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let args = parse_args();
    let dims = [args.lattice, args.lattice, args.lattice, args.lattice];
    let vol: usize = dims.iter().product();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Mixed Pipeline β-Scan: 3090 DF64 + NPU Offload + Titan V     ║");
    println!("║  Experiment 022: NPU Screening for Maximum GPU Throughput      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Lattice:  {}⁴ ({} sites)", args.lattice, vol);
    println!(
        "  VRAM est: {:.1} GB (quenched)",
        vol as f64 * 4.0 * 18.0 * 8.0 * 3.0 / 1e9
    );
    println!(
        "  Therm:    {} max per β point (NPU early-exit enabled)",
        args.n_therm
    );
    println!("  Strategy: NPU offload (therm detect + reject predict + adaptive steer)");
    println!("  Seed:     {}", args.seed);
    if args.trajectory_log.is_some() {
        println!("  Trajectory log: ENABLED (JSONL per-trajectory)");
    }
    println!();

    // ═══ Substrate Discovery ═══
    println!("═══ Substrate Discovery ═══");
    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

    let (primary_id, secondary_id) = discover_primary_and_secondary_adapters();
    let primary_id = std::env::var("HOTSPRING_GPU_ADAPTER")
        .ok()
        .or(primary_id)
        .expect("no primary GPU with SHADER_F64 found");

    #[allow(deprecated)]
    std::env::set_var("HOTSPRING_GPU_ADAPTER", &primary_id);
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => {
            println!("  Primary GPU: {}", g.adapter_name);
            g
        }
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };

    let gpu_titan = {
        let result = if let Some(ref sid) = secondary_id.filter(|s| s != &primary_id) {
            let prev = std::env::var("HOTSPRING_GPU_ADAPTER").ok();
            #[allow(deprecated)]
            std::env::set_var("HOTSPRING_GPU_ADAPTER", sid);
            let r = rt.block_on(GpuF64::new());
            #[allow(deprecated)]
            match prev {
                Some(v) => std::env::set_var("HOTSPRING_GPU_ADAPTER", v),
                None => std::env::remove_var("HOTSPRING_GPU_ADAPTER"),
            }
            r
        } else {
            Err(HotSpringError::NoAdapter)
        };
        match result {
            Ok(g) if g.adapter_name != gpu.adapter_name => {
                println!("  Validation:  {} (f64 oracle)", g.adapter_name);
                Some(g)
            }
            _ => {
                println!("  Validation:  not available — CPU f64 fallback");
                None
            }
        }
    };

    let npu_available = hotspring_barracuda::discovery::probe_npu_available();
    println!(
        "  NPU:        {} + worker thread",
        if npu_available {
            "AKD1000 (hardware)"
        } else {
            "NpuSimulator (ESN)"
        }
    );

    // Spawn NPU worker
    let (npu_tx, npu_rx) = spawn_quenched_npu_worker();
    println!("  NPU worker: spawned");

    // ═══ Placement E: Bootstrap from previous run ═══
    if let Some(ref path) = args.bootstrap_from {
        let p = std::path::Path::new(path.as_str());
        let is_weights_json = p
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        if is_weights_json {
            println!("  Bootstrap: loading ESN weights from {path}");
            npu_tx
                .send(QuenchedNpuRequest::BootstrapFromWeights { path: path.clone() })
                .ok();
        } else {
            println!("  Bootstrap: training ESN from trajectory log {path}");
            npu_tx
                .send(QuenchedNpuRequest::BootstrapFromLog { path: path.clone() })
                .ok();
        }
        match npu_rx.recv() {
            Ok(QuenchedNpuResponse::Bootstrapped { n_points, beta_c }) => {
                println!("  Bootstrap: loaded {n_points} data points, β_c estimate = {beta_c:.4}");
            }
            _ => {
                println!("  Bootstrap: failed, starting cold");
            }
        }
    }
    println!();

    let vol_f = vol as f64;
    let ref_vol = 4096.0_f64;
    let scale = (ref_vol / vol_f).powf(0.25);
    let dt = (0.05 * scale).max(0.002);
    let n_md = ((0.5 / dt).round() as usize).max(10);
    println!(
        "  HMC:        dt={dt:.4}, n_md={n_md}, traj_len={:.3}",
        dt * n_md as f64
    );
    println!();

    let pipelines = GpuHmcStreamingPipelines::new(&gpu);
    let total_start = Instant::now();
    let mut results: Vec<BetaResult> = Vec::new();
    let mut total_trajectories = 0usize;
    let mut npu_stats = QuenchedNpuStats::new();

    let mut traj_writer: Option<std::io::BufWriter<std::fs::File>> =
        args.trajectory_log.as_ref().map(|path| {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).ok();
            }
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| panic!("Cannot create trajectory log {path}: {e}"));
            std::io::BufWriter::new(f)
        });

    // ═══ Phase 1: Seed Scan (3 strategic β values) ═══
    let seed_betas = vec![5.0, 5.69, 6.5];
    let seed_meas = 500;

    println!(
        "═══ Phase 1: Seed Scan ({} points × {} meas) ═══",
        seed_betas.len(),
        seed_meas
    );
    let seed_data = run_beta_points_npu(
        &gpu,
        &pipelines,
        &seed_betas,
        dims,
        args.n_therm,
        seed_meas,
        n_md,
        dt,
        args.seed,
        &npu_tx,
        &npu_rx,
        &mut npu_stats,
        &mut traj_writer,
    );
    for r in &seed_data {
        total_trajectories += r.n_traj + r.therm_used;
        let therm_info = if r.npu_therm_early_exit {
            format!("therm={}/{} (NPU early-exit)", r.therm_used, r.therm_budget)
        } else {
            format!("therm={}", r.therm_used)
        };
        println!(
            "  β={:.4}: ⟨P⟩={:.6}±{:.6}  |L|={:.4}  χ={:.2}  acc={:.0}%  {therm_info}  ({:.1}s)",
            r.beta,
            r.mean_plaq,
            r.std_plaq,
            r.polyakov,
            r.susceptibility,
            r.acceptance * 100.0,
            r.wall_s,
        );
    }
    results.extend(seed_data.iter().cloned());
    println!();

    // ═══ Phase 2: Train ESN Phase Classifier via NPU Worker ═══
    println!("═══ Phase 2: Train ESN from Seed Data (NPU Worker) ═══");
    let esn_start = Instant::now();
    npu_tx
        .send(QuenchedNpuRequest::Retrain {
            results: results.clone(),
        })
        .ok();
    npu_stats.total_npu_calls += 1;

    let esn_beta_c = match npu_rx.recv() {
        Ok(QuenchedNpuResponse::Retrained { beta_c }) => beta_c,
        _ => KNOWN_BETA_C,
    };
    println!(
        "  ESN trained in {:.1}ms via NPU worker",
        esn_start.elapsed().as_secs_f64() * 1000.0,
    );
    println!("  ESN β_c estimate: {esn_beta_c:.4} (known: {KNOWN_BETA_C:.4})");
    println!();

    // ═══ Phase 3: Adaptive Steering (NPU picks next β) ═══
    let max_adaptive_points = 6;
    let adaptive_meas = 600;

    println!("═══ Phase 3: NPU Adaptive Steering (up to {max_adaptive_points} extra points) ═══");
    let mut measured_betas: Vec<f64> = seed_betas;
    let mut adaptive_count = 0;

    for round in 0..max_adaptive_points {
        // Placement D: NPU worker picks next β
        npu_tx
            .send(QuenchedNpuRequest::SteerNextBeta {
                measured_betas: measured_betas.clone(),
                beta_min: 4.0,
                beta_max: 7.0,
            })
            .ok();
        npu_stats.steer_queries += 1;
        npu_stats.total_npu_calls += 1;

        let next_beta = match npu_rx.recv() {
            Ok(QuenchedNpuResponse::NextBeta(b)) => b,
            _ => f64::NAN,
        };

        if next_beta.is_nan() || measured_betas.iter().any(|&b| (b - next_beta).abs() < 0.03) {
            println!(
                "  Round {}: no new β with sufficient uncertainty, stopping",
                round + 1
            );
            break;
        }

        println!(
            "  Round {}: NPU selects β={:.4} (max uncertainty)",
            round + 1,
            next_beta
        );
        let point_data = run_beta_points_npu(
            &gpu,
            &pipelines,
            &[next_beta],
            dims,
            args.n_therm,
            adaptive_meas,
            n_md,
            dt,
            args.seed + 100 + round as u64,
            &npu_tx,
            &npu_rx,
            &mut npu_stats,
            &mut traj_writer,
        );
        for r in &point_data {
            total_trajectories += r.n_traj + r.therm_used;
            let therm_info = if r.npu_therm_early_exit {
                format!("therm={}/{} (NPU early-exit)", r.therm_used, r.therm_budget)
            } else {
                format!("therm={}", r.therm_used)
            };
            println!(
                "    ⟨P⟩={:.6}±{:.6}  |L|={:.4}  χ={:.2}  acc={:.0}%  {therm_info}  ({:.1}s)",
                r.mean_plaq,
                r.std_plaq,
                r.polyakov,
                r.susceptibility,
                r.acceptance * 100.0,
                r.wall_s,
            );
        }
        results.extend(point_data.iter().cloned());
        measured_betas.push(next_beta);
        adaptive_count += 1;

        // Retrain ESN with all data so far
        npu_tx
            .send(QuenchedNpuRequest::Retrain {
                results: results.clone(),
            })
            .ok();
        npu_stats.total_npu_calls += 1;

        if let Ok(QuenchedNpuResponse::Retrained { beta_c }) = npu_rx.recv() {
            println!("    ESN β_c → {beta_c:.4}");
        }
    }
    println!("  Adaptive rounds completed: {adaptive_count}");
    println!();

    // ═══ Phase 4: Refinement Near β_c ═══
    // Ask NPU for current best β_c
    npu_tx
        .send(QuenchedNpuRequest::SteerNextBeta {
            measured_betas: measured_betas.clone(),
            beta_min: KNOWN_BETA_C - 0.2,
            beta_max: KNOWN_BETA_C + 0.2,
        })
        .ok();
    npu_stats.total_npu_calls += 1;
    let _ = npu_rx.recv(); // consume response

    // Use ESN β_c estimate for refinement
    let final_beta_c = esn_beta_c;
    let refine_betas: Vec<f64> = vec![final_beta_c - 0.05, final_beta_c, final_beta_c + 0.05]
        .into_iter()
        .filter(|b| !measured_betas.iter().any(|mb| (mb - b).abs() < 0.03))
        .collect();

    if refine_betas.is_empty() {
        println!("═══ Phase 4: Refinement — β_c region already covered ═══");
    } else {
        println!(
            "═══ Phase 4: Refinement Near β_c={:.4} ({} extra points) ═══",
            final_beta_c,
            refine_betas.len()
        );
        let refine_data = run_beta_points_npu(
            &gpu,
            &pipelines,
            &refine_betas,
            dims,
            args.n_therm,
            800,
            n_md,
            dt,
            args.seed + 500,
            &npu_tx,
            &npu_rx,
            &mut npu_stats,
            &mut traj_writer,
        );
        for r in &refine_data {
            total_trajectories += r.n_traj + r.therm_used;
            let therm_info = if r.npu_therm_early_exit {
                format!("therm={}/{} (NPU early-exit)", r.therm_used, r.therm_budget)
            } else {
                format!("therm={}", r.therm_used)
            };
            println!(
                "  β={:.4}: ⟨P⟩={:.6}±{:.6}  |L|={:.4}  χ={:.2}  acc={:.0}%  {therm_info}  ({:.1}s)",
                r.beta,
                r.mean_plaq,
                r.std_plaq,
                r.polyakov,
                r.susceptibility,
                r.acceptance * 100.0,
                r.wall_s,
            );
        }
        results.extend(refine_data);
    }
    println!();

    // ═══ Phase 5: Titan V Validation Oracle ═══
    println!("═══ Phase 5: Titan V Validation Oracle ═══");
    run_titan_validation(gpu_titan.as_ref(), &results, dims, n_md, dt);
    println!();

    // Flush trajectory log
    if let Some(ref mut w) = traj_writer {
        w.flush().ok();
    }

    // ═══ Save ESN weights for next run's bootstrap ═══
    if let Some(ref path) = args.save_weights {
        npu_tx
            .send(QuenchedNpuRequest::ExportWeights { path: path.clone() })
            .ok();
        match npu_rx.recv() {
            Ok(QuenchedNpuResponse::WeightsSaved { path: saved_path })
                if !saved_path.is_empty() =>
            {
                println!("  ESN weights saved to: {saved_path}");
                println!("  Next run: --bootstrap-from={saved_path}");
            }
            _ => {
                eprintln!("  Warning: failed to save ESN weights");
            }
        }
    }

    // Shut down NPU worker
    npu_tx.send(QuenchedNpuRequest::Shutdown).ok();

    // ═══ Summary & Comparison ═══
    results.sort_by(|a, b| a.beta.total_cmp(&b.beta));
    let total_wall = total_start.elapsed().as_secs_f64();
    let total_meas: usize = results.iter().map(|r| r.n_traj).sum();

    print_mixed_summary(
        &results,
        &npu_stats,
        args.lattice,
        total_trajectories,
        total_wall,
        adaptive_count,
        final_beta_c,
        &gpu.adapter_name,
        gpu_titan.as_ref().map(|g| g.adapter_name.as_str()),
        args.trajectory_log.as_deref(),
    );

    if let Some(path) = args.output {
        write_mixed_json(
            &path,
            &results,
            args.lattice,
            dims,
            vol,
            &gpu.adapter_name,
            gpu_titan.as_ref().map(|g| g.adapter_name.as_str()),
            if npu_available {
                "AKD1000"
            } else {
                "NpuSimulator"
            },
            args.n_therm,
            args.seed,
            total_wall,
            total_trajectories,
            total_meas,
            adaptive_count,
            final_beta_c,
            &npu_stats,
        );
    }
}
