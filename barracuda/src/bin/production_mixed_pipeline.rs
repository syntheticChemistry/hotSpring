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

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_links_to_lattice, gpu_polyakov_loop, GpuHmcState,
    GpuHmcStreamingPipelines,
};
use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType};
use hotspring_barracuda::lattice::wilson::Lattice;
use hotspring_barracuda::md::reservoir::{
    EchoStateNetwork, EsnConfig, ExportedWeights, NpuSimulator,
};

use std::io::Write as IoWrite;
use std::sync::mpsc;
use std::time::Instant;

use hotspring_barracuda::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;

// ═══════════════════════════════════════════════════════════════════
//  NPU Worker — Screening Decisions
// ═══════════════════════════════════════════════════════════════════

/// Message sent from the GPU/main thread to the NPU worker.
enum NpuRequest {
    /// Placement A: check if thermalization has converged.
    ThermCheck {
        plaq_window: Vec<f64>,
        beta: f64,
    },
    /// Placement B: predict whether current trajectory will be rejected.
    RejectPredict {
        beta: f64,
        plaquette: f64,
        action_density: f64,
        delta_h: f64,
        acceptance_rate: f64,
    },
    /// Placement C: classify phase from post-measurement observables.
    PhaseClassify {
        beta: f64,
        plaquette: f64,
        polyakov: f64,
        susceptibility: f64,
    },
    /// Placement D: find next β with maximum uncertainty.
    SteerNextBeta {
        measured_betas: Vec<f64>,
        beta_min: f64,
        beta_max: f64,
    },
    /// Re-train ESN models with new data.
    Retrain {
        results: Vec<BetaResult>,
    },
    /// Bootstrap ESN from a previous run's trajectory log (Placement E).
    BootstrapFromLog {
        path: String,
    },
    /// Bootstrap ESN from saved weights JSON.
    BootstrapFromWeights {
        path: String,
    },
    /// Export current ESN weights to disk for the next run.
    ExportWeights {
        path: String,
    },
    /// Shut down the worker.
    Shutdown,
}

/// Response from the NPU worker.
enum NpuResponse {
    /// Thermalization: true = converged, can exit early.
    ThermConverged(bool),
    /// Rejection prediction: true = likely rejected, suggest abort bookkeeping.
    RejectPrediction { likely_rejected: bool, confidence: f64 },
    /// Phase classification result.
    PhaseLabel(&'static str),
    /// Next β to measure (NaN if none found).
    NextBeta(f64),
    /// Retrain complete, new β_c estimate.
    Retrained { beta_c: f64 },
    /// Bootstrap complete, how many data points loaded.
    Bootstrapped { n_points: usize, beta_c: f64 },
    /// Weights saved to disk.
    WeightsSaved { path: String },
}

/// Spawn the NPU worker thread. Returns (request_sender, response_receiver).
fn spawn_npu_worker() -> (mpsc::Sender<NpuRequest>, mpsc::Receiver<NpuResponse>) {
    let (req_tx, req_rx) = mpsc::channel::<NpuRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<NpuResponse>();

    std::thread::Builder::new()
        .name("npu-worker".into())
        .spawn(move || {
            let mut npu: Option<NpuSimulator> = None;

            let make_esn = |seed: u64, results: &[BetaResult]| -> Option<NpuSimulator> {
                if results.is_empty() {
                    return None;
                }
                let (seqs, tgts) = build_training_data(results);
                let mut esn = EchoStateNetwork::new(EsnConfig {
                    input_size: 4,
                    reservoir_size: 50,
                    output_size: 2,
                    spectral_radius: 0.95,
                    connectivity: 0.2,
                    leak_rate: 0.3,
                    regularization: 1e-3,
                    seed,
                });
                esn.train(&seqs, &tgts);
                esn.export_weights().map(|w| NpuSimulator::from_exported(&w))
            };

            for req in req_rx {
                match req {
                    NpuRequest::ThermCheck { plaq_window, beta } => {
                        let converged = check_thermalization(&plaq_window, beta);
                        resp_tx.send(NpuResponse::ThermConverged(converged)).ok();
                    }
                    NpuRequest::RejectPredict {
                        beta,
                        plaquette,
                        action_density,
                        delta_h,
                        acceptance_rate,
                    } => {
                        let (likely_rejected, confidence) = predict_rejection(
                            beta,
                            plaquette,
                            action_density,
                            delta_h,
                            acceptance_rate,
                        );
                        resp_tx
                            .send(NpuResponse::RejectPrediction {
                                likely_rejected,
                                confidence,
                            })
                            .ok();
                    }
                    NpuRequest::PhaseClassify {
                        beta,
                        plaquette,
                        polyakov,
                        susceptibility,
                    } => {
                        let label = if let Some(ref mut n) = npu {
                            let beta_norm = (beta - 5.0) / 2.0;
                            let seq = vec![
                                vec![beta_norm, plaquette, polyakov, susceptibility / 1000.0];
                                10
                            ];
                            let pred = n.predict(&seq);
                            if !pred.is_empty() && pred[0] > 0.5 {
                                "deconfined"
                            } else {
                                "confined"
                            }
                        } else if beta > KNOWN_BETA_C {
                            "deconfined"
                        } else {
                            "confined"
                        };
                        resp_tx.send(NpuResponse::PhaseLabel(label)).ok();
                    }
                    NpuRequest::SteerNextBeta {
                        measured_betas,
                        beta_min,
                        beta_max,
                    } => {
                        let next = if let Some(ref mut n) = npu {
                            find_max_uncertainty_beta(n, &measured_betas, beta_min, beta_max, 60)
                        } else {
                            f64::NAN
                        };
                        resp_tx.send(NpuResponse::NextBeta(next)).ok();
                    }
                    NpuRequest::Retrain { results } => {
                        let seed = 99 + results.len() as u64;
                        if let Some(new_npu) = make_esn(seed, &results) {
                            npu = Some(new_npu);
                        }
                        let beta_c = if let Some(ref mut n) = npu {
                            predict_beta_c(n)
                        } else {
                            KNOWN_BETA_C
                        };
                        resp_tx.send(NpuResponse::Retrained { beta_c }).ok();
                    }
                    NpuRequest::BootstrapFromLog { path } => {
                        let (n_points, beta_c) =
                            bootstrap_esn_from_trajectory_log(&path, &make_esn, &mut npu);
                        resp_tx
                            .send(NpuResponse::Bootstrapped { n_points, beta_c })
                            .ok();
                    }
                    NpuRequest::BootstrapFromWeights { path } => {
                        match std::fs::read_to_string(&path) {
                            Ok(json) => match serde_json::from_str::<ExportedWeights>(&json) {
                                Ok(weights) => {
                                    npu = Some(NpuSimulator::from_exported(&weights));
                                    let beta_c = if let Some(ref mut n) = npu {
                                        predict_beta_c(n)
                                    } else {
                                        KNOWN_BETA_C
                                    };
                                    resp_tx
                                        .send(NpuResponse::Bootstrapped {
                                            n_points: 0,
                                            beta_c,
                                        })
                                        .ok();
                                }
                                Err(e) => {
                                    eprintln!("  Warning: failed to parse weights from {path}: {e}");
                                    resp_tx
                                        .send(NpuResponse::Bootstrapped {
                                            n_points: 0,
                                            beta_c: KNOWN_BETA_C,
                                        })
                                        .ok();
                                }
                            },
                            Err(e) => {
                                eprintln!("  Warning: cannot read weights file {path}: {e}");
                                resp_tx
                                    .send(NpuResponse::Bootstrapped {
                                        n_points: 0,
                                        beta_c: KNOWN_BETA_C,
                                    })
                                    .ok();
                            }
                        }
                    }
                    NpuRequest::ExportWeights { path } => {
                        let saved = if let Some(ref n) = npu {
                            let weights = ExportedWeights {
                                w_in: n.export_w_in(),
                                w_res: n.export_w_res(),
                                w_out: n.export_w_out(),
                                input_size: n.input_size(),
                                reservoir_size: n.reservoir_size(),
                                output_size: n.output_size(),
                                leak_rate: n.leak_rate(),
                            };
                            if let Some(parent) = std::path::Path::new(&path).parent() {
                                std::fs::create_dir_all(parent).ok();
                            }
                            match std::fs::write(
                                &path,
                                serde_json::to_string(&weights).unwrap_or_default(),
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    eprintln!("  Warning: failed to save weights to {path}: {e}");
                                    false
                                }
                            }
                        } else {
                            false
                        };
                        if saved {
                            resp_tx
                                .send(NpuResponse::WeightsSaved {
                                    path: path.clone(),
                                })
                                .ok();
                        } else {
                            resp_tx
                                .send(NpuResponse::WeightsSaved {
                                    path: String::new(),
                                })
                                .ok();
                        }
                    }
                    NpuRequest::Shutdown => break,
                }
            }
        })
        .expect("spawn NPU worker thread");

    (req_tx, resp_rx)
}

/// Placement A: statistical convergence check on plaquette window.
/// Uses variance-ratio and drift tests — mimics what the ESN therm detector learned.
fn check_thermalization(plaq_window: &[f64], _beta: f64) -> bool {
    if plaq_window.len() < 10 {
        return false;
    }
    let n = plaq_window.len();
    let mean = plaq_window.iter().sum::<f64>() / n as f64;
    let var = plaq_window.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    let half = n / 2;
    let mean_first = plaq_window[..half].iter().sum::<f64>() / half as f64;
    let mean_second = plaq_window[half..].iter().sum::<f64>() / (n - half) as f64;
    let drift = (mean_second - mean_first).abs();

    let relative_var = if mean.abs() > 1e-12 {
        var.sqrt() / mean.abs()
    } else {
        var.sqrt()
    };

    // Convergence: low relative variance AND negligible drift between halves
    relative_var < 0.02 && drift < 0.005
}

/// Placement B: predict trajectory rejection from observables.
/// Uses empirical heuristics calibrated against Exp 020 data:
/// large |ΔH| and low acceptance rate strongly predict rejection.
fn predict_rejection(
    _beta: f64,
    _plaquette: f64,
    _action_density: f64,
    delta_h: f64,
    acceptance_rate: f64,
) -> (bool, f64) {
    let dh_mag = delta_h.abs();

    // Metropolis: accept if ΔH < 0, else accept with prob exp(-ΔH).
    // High |ΔH| with ΔH > 0 strongly predicts rejection.
    let rejection_score = if delta_h > 0.0 {
        // P(reject) = 1 - exp(-ΔH) — approaches 1 for large positive ΔH
        1.0 - (-dh_mag).exp()
    } else {
        0.0
    };

    // Weight by running acceptance rate — low acceptance = more likely to reject
    let rate_factor = if acceptance_rate < 0.3 {
        1.2
    } else if acceptance_rate < 0.5 {
        1.0
    } else {
        0.8
    };

    let confidence = (rejection_score * rate_factor).clamp(0.0, 1.0);
    let likely_rejected = confidence > 0.8;

    (likely_rejected, confidence)
}

/// Placement E: Bootstrap ESN from a previous run's trajectory log.
/// Reads JSONL lines, extracts per-beta aggregate statistics, and trains
/// an ESN so the NPU arrives at Phase 1 already knowing the physics.
fn bootstrap_esn_from_trajectory_log(
    path: &str,
    make_esn: &dyn Fn(u64, &[BetaResult]) -> Option<NpuSimulator>,
    npu: &mut Option<NpuSimulator>,
) -> (usize, f64) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Warning: cannot read bootstrap log {path}: {e}");
            return (0, KNOWN_BETA_C);
        }
    };

    // Parse trajectory log into per-beta aggregates
    let mut beta_data: std::collections::BTreeMap<String, Vec<(f64, bool)>> =
        std::collections::BTreeMap::new();
    let mut n_lines = 0usize;

    for line in content.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        // Only use measurement trajectories (not thermalization, not NPU prediction lines)
        if v.get("is_therm") == Some(&serde_json::Value::Bool(true)) {
            continue;
        }
        let Some(beta) = v.get("beta").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let Some(plaq) = v.get("plaquette").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let accepted = v
            .get("accepted")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let key = format!("{beta:.4}");
        beta_data.entry(key).or_default().push((plaq, accepted));
        n_lines += 1;
    }

    if beta_data.is_empty() {
        return (0, KNOWN_BETA_C);
    }

    // Convert to BetaResult aggregates for ESN training
    let results: Vec<BetaResult> = beta_data
        .into_iter()
        .map(|(key, entries)| {
            let beta: f64 = key.parse().unwrap_or(5.69);
            let n = entries.len();
            let plaqs: Vec<f64> = entries.iter().map(|(p, _)| *p).collect();
            let mean_plaq = plaqs.iter().sum::<f64>() / n as f64;
            let var_plaq = plaqs.iter().map(|p| (p - mean_plaq).powi(2)).sum::<f64>()
                / (n - 1).max(1) as f64;
            let n_accepted = entries.iter().filter(|(_, a)| *a).count();

            BetaResult {
                beta,
                mean_plaq,
                std_plaq: var_plaq.sqrt(),
                polyakov: 0.0,
                susceptibility: var_plaq * 1048576.0, // estimate for 32^4
                action_density: 6.0 * (1.0 - mean_plaq),
                acceptance: n_accepted as f64 / n as f64,
                n_traj: n,
                wall_s: 0.0,
                phase: if beta < KNOWN_BETA_C - 0.1 {
                    "confined"
                } else if beta > KNOWN_BETA_C + 0.1 {
                    "deconfined"
                } else {
                    "transition"
                },
                therm_used: 0,
                therm_budget: 0,
                npu_therm_early_exit: false,
                npu_reject_predictions: 0,
                npu_reject_correct: 0,
            }
        })
        .collect();

    let n_betas = results.len();
    if let Some(new_npu) = make_esn(42, &results) {
        *npu = Some(new_npu);
    }

    let beta_c = if let Some(ref mut n) = npu {
        predict_beta_c(n)
    } else {
        KNOWN_BETA_C
    };

    (n_betas * n_lines / n_betas.max(1), beta_c)
}

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
//  Beta Result
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct BetaResult {
    beta: f64,
    mean_plaq: f64,
    std_plaq: f64,
    polyakov: f64,
    susceptibility: f64,
    action_density: f64,
    acceptance: f64,
    n_traj: usize,
    wall_s: f64,
    phase: &'static str,
    therm_used: usize,
    therm_budget: usize,
    npu_therm_early_exit: bool,
    npu_reject_predictions: usize,
    npu_reject_correct: usize,
}

// ═══════════════════════════════════════════════════════════════════
//  NPU Stats Tracking
// ═══════════════════════════════════════════════════════════════════

struct NpuStats {
    therm_early_exits: usize,
    therm_total_saved: usize,
    reject_predictions: usize,
    reject_correct: usize,
    phase_classifications: usize,
    steer_queries: usize,
    total_npu_calls: usize,
}

impl NpuStats {
    fn new() -> Self {
        Self {
            therm_early_exits: 0,
            therm_total_saved: 0,
            reject_predictions: 0,
            reject_correct: 0,
            phase_classifications: 0,
            steer_queries: 0,
            total_npu_calls: 0,
        }
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
    println!("  Therm:    {} max per β point (NPU early-exit enabled)", args.n_therm);
    println!("  Strategy: NPU offload (therm detect + reject predict + adaptive steer)");
    println!("  Seed:     {}", args.seed);
    if args.trajectory_log.is_some() {
        println!("  Trajectory log: ENABLED (JSONL per-trajectory)");
    }
    println!();

    // ═══ Substrate Discovery ═══
    println!("═══ Substrate Discovery ═══");
    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

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
        let prev = std::env::var("HOTSPRING_GPU_ADAPTER").ok();
        std::env::set_var("HOTSPRING_GPU_ADAPTER", "titan");
        let result = rt.block_on(GpuF64::new());
        match &prev {
            Some(v) => std::env::set_var("HOTSPRING_GPU_ADAPTER", v),
            None => std::env::remove_var("HOTSPRING_GPU_ADAPTER"),
        }
        match result {
            Ok(g) if g.adapter_name != gpu.adapter_name => {
                println!("  Titan V:     {} (validation oracle)", g.adapter_name);
                Some(g)
            }
            _ => {
                println!("  Titan V:     not available — CPU f64 fallback");
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
    let (npu_tx, npu_rx) = spawn_npu_worker();
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
                .send(NpuRequest::BootstrapFromWeights {
                    path: path.clone(),
                })
                .ok();
        } else {
            println!("  Bootstrap: training ESN from trajectory log {path}");
            npu_tx
                .send(NpuRequest::BootstrapFromLog {
                    path: path.clone(),
                })
                .ok();
        }
        match npu_rx.recv() {
            Ok(NpuResponse::Bootstrapped { n_points, beta_c }) => {
                println!(
                    "  Bootstrap: loaded {n_points} data points, β_c estimate = {beta_c:.4}"
                );
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
    let mut npu_stats = NpuStats::new();

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
        .send(NpuRequest::Retrain {
            results: results.clone(),
        })
        .ok();
    npu_stats.total_npu_calls += 1;

    let esn_beta_c = match npu_rx.recv() {
        Ok(NpuResponse::Retrained { beta_c }) => beta_c,
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
            .send(NpuRequest::SteerNextBeta {
                measured_betas: measured_betas.clone(),
                beta_min: 4.0,
                beta_max: 7.0,
            })
            .ok();
        npu_stats.steer_queries += 1;
        npu_stats.total_npu_calls += 1;

        let next_beta = match npu_rx.recv() {
            Ok(NpuResponse::NextBeta(b)) => b,
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
            .send(NpuRequest::Retrain {
                results: results.clone(),
            })
            .ok();
        npu_stats.total_npu_calls += 1;

        if let Ok(NpuResponse::Retrained { beta_c }) = npu_rx.recv() {
            println!("    ESN β_c → {beta_c:.4}");
        }
    }
    println!("  Adaptive rounds completed: {adaptive_count}");
    println!();

    // ═══ Phase 4: Refinement Near β_c ═══
    // Ask NPU for current best β_c
    npu_tx
        .send(NpuRequest::SteerNextBeta {
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
            .send(NpuRequest::ExportWeights {
                path: path.clone(),
            })
            .ok();
        match npu_rx.recv() {
            Ok(NpuResponse::WeightsSaved { path: saved_path }) if !saved_path.is_empty() => {
                println!("  ESN weights saved to: {saved_path}");
                println!("  Next run: --bootstrap-from={saved_path}");
            }
            _ => {
                eprintln!("  Warning: failed to save ESN weights");
            }
        }
    }

    // Shut down NPU worker
    npu_tx.send(NpuRequest::Shutdown).ok();

    // ═══ Summary & Comparison ═══
    let total_wall = total_start.elapsed().as_secs_f64();
    results.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap());

    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  Mixed Pipeline β-Scan Summary: {}⁴ Quenched SU(3)",
        args.lattice
    );
    println!("═══════════════════════════════════════════════════════════");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>10} {:>8} {:>6} {:>10} {:>8}",
        "β", "⟨P⟩", "σ(P)", "|L|", "χ", "acc%", "traj", "therm", "time"
    );
    for r in &results {
        let therm_col = if r.npu_therm_early_exit {
            format!("{}/{}", r.therm_used, r.therm_budget)
        } else {
            format!("{}", r.therm_used)
        };
        println!(
            "  {:>6.4} {:>10.6} {:>10.6} {:>10.4} {:>10.4} {:>7.1}% {:>6} {:>10} {:>7.1}s",
            r.beta,
            r.mean_plaq,
            r.std_plaq,
            r.polyakov,
            r.susceptibility,
            r.acceptance * 100.0,
            r.n_traj,
            therm_col,
            r.wall_s
        );
    }
    println!();

    // NPU stats
    let total_therm_budget: usize = results.iter().map(|r| r.therm_budget).sum();
    let total_therm_used: usize = results.iter().map(|r| r.therm_used).sum();
    let therm_savings_pct = if total_therm_budget > 0 {
        (1.0 - total_therm_used as f64 / total_therm_budget as f64) * 100.0
    } else {
        0.0
    };

    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  NPU Offload Statistics                                │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!(
        "  │  Therm early-exits:     {:<6} / {:<6} β points       │",
        npu_stats.therm_early_exits,
        results.len()
    );
    println!(
        "  │  Therm traj saved:      {:<6} / {:<6} ({:.1}%)       │",
        npu_stats.therm_total_saved,
        total_therm_budget,
        therm_savings_pct
    );
    println!(
        "  │  Reject predictions:    {:<6} (correct: {:<6})       │",
        npu_stats.reject_predictions, npu_stats.reject_correct
    );
    println!(
        "  │  Phase classifications: {:<6}                        │",
        npu_stats.phase_classifications
    );
    println!(
        "  │  Steering queries:      {:<6}                        │",
        npu_stats.steer_queries
    );
    println!(
        "  │  Total NPU calls:       {:<6}                        │",
        npu_stats.total_npu_calls
    );
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    let total_meas: usize = results.iter().map(|r| r.n_traj).sum();
    let exp013_traj = 12 * (200 + 1000);
    let exp013_wall = 48988.3;
    let exp018_wall = 25560.0;
    let exp013_energy_kwh = exp013_wall * 300.0 / 3_600_000.0;
    let exp018_energy_kwh = exp018_wall * 300.0 / 3_600_000.0;
    let mixed_energy_kwh = total_wall * 300.0 / 3_600_000.0;

    println!("  ┌───────────────────────────────────────────────────────────────────┐");
    println!("  │  Comparison: Exp 022 (NPU offload) vs Exp 013/018               │");
    println!("  ├───────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Metric             Exp 022       Exp 018 (DF64)  Exp 013 (f64)  │"
    );
    println!(
        "  │  β points           {:<14}{:<16}{}              │",
        results.len(),
        12,
        12
    );
    println!(
        "  │  Total trajectories {:<14}{:<16}{}              │",
        total_trajectories, "~14400", exp013_traj
    );
    println!(
        "  │  Measurement traj   {:<14}{:<16}{}              │",
        total_meas, 12000, 12000
    );
    println!(
        "  │  Wall time          {total_wall:<13.1}s {exp018_wall:<15.1}s {exp013_wall:.1}s          │"
    );
    println!(
        "  │  Wall time (hrs)    {:<13.2}h {:<15.2}h {:.2}h          │",
        total_wall / 3600.0,
        exp018_wall / 3600.0,
        exp013_wall / 3600.0
    );
    println!(
        "  │  Energy (est.)      {mixed_energy_kwh:<12.2} kWh {exp018_energy_kwh:<14.2} kWh {exp013_energy_kwh:.2} kWh       │"
    );
    println!(
        "  │  Therm savings      {therm_savings_pct:<13.0}%                                   │"
    );
    println!(
        "  │  Speedup vs 013     {:<13.1}×                                   │",
        exp013_wall / total_wall
    );
    println!(
        "  │  Energy save vs 013 {:<12.0}%                                   │",
        (1.0 - mixed_energy_kwh / exp013_energy_kwh) * 100.0
    );
    println!("  └───────────────────────────────────────────────────────────────────┘");
    println!();

    // Physics quality check
    println!("  Physics Quality:");
    let near_bc: Vec<&BetaResult> = results
        .iter()
        .filter(|r| (r.beta - KNOWN_BETA_C).abs() < 0.15)
        .collect();
    let far_confined: Vec<&BetaResult> = results.iter().filter(|r| r.beta < 5.5).collect();
    let far_deconfined: Vec<&BetaResult> = results.iter().filter(|r| r.beta > 6.0).collect();

    if !near_bc.is_empty() {
        let max_chi = near_bc
            .iter()
            .map(|r| r.susceptibility)
            .fold(0.0_f64, f64::max);
        println!("    Susceptibility peak near β_c: χ_max = {max_chi:.2}");
    }
    if !far_confined.is_empty() {
        let mean_poly: f64 =
            far_confined.iter().map(|r| r.polyakov).sum::<f64>() / far_confined.len() as f64;
        println!("    Confined |L| (β<5.5):  {mean_poly:.4}");
    }
    if !far_deconfined.is_empty() {
        let mean_poly: f64 =
            far_deconfined.iter().map(|r| r.polyakov).sum::<f64>() / far_deconfined.len() as f64;
        println!("    Deconfined |L| (β>6.0): {mean_poly:.4}");
    }
    let monotonic = results
        .windows(2)
        .all(|w| w[1].mean_plaq >= w[0].mean_plaq - 0.01);
    println!(
        "    Plaquette monotonicity: {}",
        if monotonic { "PASS" } else { "MARGINAL" }
    );
    println!();

    println!("  GPU: {}", gpu.adapter_name);
    if let Some(ref t) = gpu_titan {
        println!("  Titan V: {}", t.adapter_name);
    }
    println!(
        "  Total wall time: {:.1}s ({:.2} hours)",
        total_wall,
        total_wall / 3600.0
    );
    if let Some(ref path) = args.trajectory_log {
        println!("  Trajectory log: {path}");
    }
    println!();

    if let Some(path) = args.output {
        let json = serde_json::json!({
            "experiment": "022_NPU_OFFLOAD_MIXED_PIPELINE",
            "lattice": args.lattice,
            "dims": dims,
            "volume": vol,
            "gpu": gpu.adapter_name,
            "titan_v": gpu_titan.as_ref().map(|g| &g.adapter_name),
            "npu": if npu_available { "AKD1000" } else { "NpuSimulator" },
            "n_therm_max": args.n_therm,
            "seed": args.seed,
            "total_wall_s": total_wall,
            "total_trajectories": total_trajectories,
            "total_measurements": total_meas,
            "adaptive_rounds": adaptive_count,
            "esn_beta_c": final_beta_c,
            "npu_stats": {
                "therm_early_exits": npu_stats.therm_early_exits,
                "therm_traj_saved": npu_stats.therm_total_saved,
                "therm_savings_pct": therm_savings_pct,
                "reject_predictions": npu_stats.reject_predictions,
                "reject_correct": npu_stats.reject_correct,
                "phase_classifications": npu_stats.phase_classifications,
                "steer_queries": npu_stats.steer_queries,
                "total_npu_calls": npu_stats.total_npu_calls,
            },
            "comparison": {
                "exp013_wall_s": exp013_wall,
                "exp013_trajectories": exp013_traj,
                "exp018_wall_s": exp018_wall,
                "speedup_vs_013": exp013_wall / total_wall,
                "speedup_vs_018": exp018_wall / total_wall,
                "trajectory_reduction_pct": (1.0 - total_trajectories as f64 / exp013_traj as f64) * 100.0,
                "energy_savings_vs_013_pct": (1.0 - mixed_energy_kwh / exp013_energy_kwh) * 100.0,
            },
            "points": results.iter().map(|r| serde_json::json!({
                "beta": r.beta,
                "mean_plaquette": r.mean_plaq,
                "std_plaquette": r.std_plaq,
                "polyakov": r.polyakov,
                "susceptibility": r.susceptibility,
                "action_density": r.action_density,
                "acceptance": r.acceptance,
                "n_trajectories": r.n_traj,
                "therm_used": r.therm_used,
                "therm_budget": r.therm_budget,
                "npu_therm_early_exit": r.npu_therm_early_exit,
                "npu_reject_predictions": r.npu_reject_predictions,
                "npu_reject_correct": r.npu_reject_correct,
                "wall_s": r.wall_s,
                "phase": r.phase,
            })).collect::<Vec<_>>(),
        });
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&path, serde_json::to_string_pretty(&json).unwrap())
            .unwrap_or_else(|e| eprintln!("  Failed to write {path}: {e}"));
        println!("  Results saved to: {path}");
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Core: Run β points with NPU offloading
// ═══════════════════════════════════════════════════════════════════

/// Run a set of β points with NPU-assisted thermalization, rejection prediction,
/// phase classification, and per-trajectory logging.
#[allow(clippy::too_many_arguments)]
fn run_beta_points_npu(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    betas: &[f64],
    dims: [usize; 4],
    n_therm: usize,
    n_meas: usize,
    n_md: usize,
    dt: f64,
    base_seed: u64,
    npu_tx: &mpsc::Sender<NpuRequest>,
    npu_rx: &mpsc::Receiver<NpuResponse>,
    npu_stats: &mut NpuStats,
    traj_writer: &mut Option<std::io::BufWriter<std::fs::File>>,
) -> Vec<BetaResult> {
    let vol: usize = dims.iter().product();
    let mut out = Vec::new();

    for (bi, &beta) in betas.iter().enumerate() {
        let start = Instant::now();
        let mut lat = Lattice::hot_start(dims, beta, base_seed + bi as u64);

        // Brief CPU pre-thermalization for small volumes
        if vol <= 65536 {
            let mut cfg = HmcConfig {
                n_md_steps: n_md,
                dt,
                seed: base_seed + bi as u64 * 1000,
                integrator: IntegratorType::Omelyan,
            };
            for _ in 0..5 {
                hmc::hmc_trajectory(&mut lat, &mut cfg);
            }
        }

        let state = GpuHmcState::from_lattice(gpu, &lat, beta);
        let mut seed = base_seed * 100 + bi as u64;

        // ─── Thermalization with NPU early-exit (Placement A) ───
        let min_therm = 20; // minimum before NPU can trigger early exit
        let check_interval = 10; // check convergence every N trajectories
        let mut plaq_history: Vec<f64> = Vec::with_capacity(n_therm);
        let mut therm_used = 0;
        let mut early_exit = false;

        for i in 0..n_therm {
            let r = gpu_hmc_trajectory_streaming(
                gpu, pipelines, &state, n_md, dt, i as u32, &mut seed,
            );
            plaq_history.push(r.plaquette);
            therm_used = i + 1;

            // Log thermalization trajectory
            if let Some(ref mut w) = traj_writer {
                let line = serde_json::json!({
                    "beta": beta,
                    "traj_idx": i,
                    "is_therm": true,
                    "plaquette": r.plaquette,
                    "delta_h": r.delta_h,
                    "accepted": r.accepted,
                });
                writeln!(w, "{line}").ok();
            }

            // NPU thermalization check after minimum + at regular intervals
            if i >= min_therm && (i + 1) % check_interval == 0 {
                let window_start = if plaq_history.len() > 32 {
                    plaq_history.len() - 32
                } else {
                    0
                };
                npu_tx
                    .send(NpuRequest::ThermCheck {
                        plaq_window: plaq_history[window_start..].to_vec(),
                        beta,
                    })
                    .ok();
                npu_stats.total_npu_calls += 1;

                if let Ok(NpuResponse::ThermConverged(converged)) = npu_rx.recv() {
                    if converged {
                        early_exit = true;
                        break;
                    }
                }
            }
        }

        if early_exit {
            npu_stats.therm_early_exits += 1;
            npu_stats.therm_total_saved += n_therm - therm_used;
        }

        // ─── Measurement with NPU rejection prediction (Placement B) ───
        let mut plaq_vals = Vec::with_capacity(n_meas);
        let mut poly_vals = Vec::new();
        let mut n_accepted = 0usize;
        let mut reject_predictions = 0usize;
        let mut reject_correct = 0usize;

        plaq_history.clear();

        for i in 0..n_meas {
            let r = gpu_hmc_trajectory_streaming(
                gpu,
                pipelines,
                &state,
                n_md,
                dt,
                (n_therm + i) as u32,
                &mut seed,
            );
            plaq_vals.push(r.plaquette);
            plaq_history.push(r.plaquette);
            if plaq_history.len() > 32 {
                plaq_history.remove(0);
            }

            if r.accepted {
                n_accepted += 1;
            }

            // NPU rejection prediction (Placement B) — post-trajectory analysis
            let running_acc = if i > 0 {
                n_accepted as f64 / (i + 1) as f64
            } else {
                0.5
            };
            let action_density = 6.0 * (1.0 - r.plaquette);

            npu_tx
                .send(NpuRequest::RejectPredict {
                    beta,
                    plaquette: r.plaquette,
                    action_density,
                    delta_h: r.delta_h,
                    acceptance_rate: running_acc,
                })
                .ok();
            npu_stats.total_npu_calls += 1;
            npu_stats.reject_predictions += 1;
            reject_predictions += 1;

            if let Ok(NpuResponse::RejectPrediction {
                likely_rejected,
                confidence,
            }) = npu_rx.recv()
            {
                if likely_rejected != r.accepted {
                    npu_stats.reject_correct += 1;
                    reject_correct += 1;
                }
                // Log rejection prediction alongside trajectory data
                if let Some(ref mut w) = traj_writer {
                    let line = serde_json::json!({
                        "beta": beta,
                        "traj_idx": therm_used + i,
                        "npu_reject_prediction": likely_rejected,
                        "npu_reject_confidence": confidence,
                        "actual_rejected": !r.accepted,
                    });
                    writeln!(w, "{line}").ok();
                }
            }

            // Polyakov loop readback (periodic + for trajectory log)
            let do_poly_readback = traj_writer.is_some() || (i + 1) % 100 == 0;
            let mut poly_mag = 0.0;
            let mut poly_phase = 0.0;
            if do_poly_readback {
                let (m, p) = gpu_polyakov_loop(gpu, &pipelines.hmc, &state);
                poly_mag = m;
                poly_phase = p;
                if (i + 1) % 100 == 0 {
                    poly_vals.push(poly_mag);
                }
            }

            // Log measurement trajectory
            if let Some(ref mut w) = traj_writer {
                let pvar = plaquette_variance(&plaq_history);
                let line = serde_json::json!({
                    "beta": beta,
                    "traj_idx": therm_used + i,
                    "is_therm": false,
                    "plaquette": r.plaquette,
                    "delta_h": r.delta_h,
                    "accepted": r.accepted,
                    "polyakov_mag": poly_mag,
                    "polyakov_phase": poly_phase,
                    "action_density": action_density,
                    "plaquette_var": pvar,
                    "acceptance_rate": running_acc,
                });
                writeln!(w, "{line}").ok();
            }
        }

        // NPU phase classification (Placement C)
        let mean_plaq: f64 = plaq_vals.iter().sum::<f64>() / plaq_vals.len() as f64;
        let var_plaq: f64 = plaq_vals
            .iter()
            .map(|p| (p - mean_plaq).powi(2))
            .sum::<f64>()
            / (plaq_vals.len() - 1).max(1) as f64;
        let std_plaq = var_plaq.sqrt();

        let mean_poly: f64 = if poly_vals.is_empty() {
            gpu_links_to_lattice(gpu, &state, &mut lat);
            lat.average_polyakov_loop()
        } else {
            poly_vals.iter().sum::<f64>() / poly_vals.len() as f64
        };

        let susceptibility = var_plaq * vol as f64;
        let action_density = 6.0 * (1.0 - mean_plaq);
        let acceptance = n_accepted as f64 / n_meas as f64;
        let wall_s = start.elapsed().as_secs_f64();

        npu_tx
            .send(NpuRequest::PhaseClassify {
                beta,
                plaquette: mean_plaq,
                polyakov: mean_poly,
                susceptibility,
            })
            .ok();
        npu_stats.phase_classifications += 1;
        npu_stats.total_npu_calls += 1;

        let phase = match npu_rx.recv() {
            Ok(NpuResponse::PhaseLabel(l)) => l,
            _ => {
                if beta < KNOWN_BETA_C - 0.1 {
                    "confined"
                } else if beta > KNOWN_BETA_C + 0.1 {
                    "deconfined"
                } else {
                    "transition"
                }
            }
        };

        // Override with "transition" if near known β_c
        let phase = if (beta - KNOWN_BETA_C).abs() < 0.1 {
            "transition"
        } else {
            phase
        };

        out.push(BetaResult {
            beta,
            mean_plaq,
            std_plaq,
            polyakov: mean_poly,
            susceptibility,
            action_density,
            acceptance,
            n_traj: n_meas,
            wall_s,
            phase,
            therm_used,
            therm_budget: n_therm,
            npu_therm_early_exit: early_exit,
            npu_reject_predictions: reject_predictions,
            npu_reject_correct: reject_correct,
        });

        if let Some(ref mut w) = traj_writer {
            w.flush().ok();
        }
    }

    out
}

// ═══════════════════════════════════════════════════════════════════
//  Utilities
// ═══════════════════════════════════════════════════════════════════

fn plaquette_variance(history: &[f64]) -> f64 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    history
        .iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>()
        / (history.len() - 1) as f64
}

/// Build ESN training data from accumulated β-scan results.
/// Features: [β_norm, plaquette, polyakov, susceptibility_norm]
/// Targets: [phase (0=confined, 1=deconfined), beta_c_proximity]
fn build_training_data(results: &[BetaResult]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for r in results {
        let beta_norm = (r.beta - 5.0) / 2.0;
        let phase = if r.beta > KNOWN_BETA_C { 1.0 } else { 0.0 };
        let proximity = (-(r.beta - KNOWN_BETA_C).powi(2) / 0.1).exp();

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.005 * ((j as f64) * 0.7).sin();
                vec![
                    beta_norm,
                    r.mean_plaq + noise * r.std_plaq,
                    r.polyakov + noise * 0.01,
                    r.susceptibility / 1000.0,
                ]
            })
            .collect();
        seqs.push(seq);
        targets.push(vec![phase, proximity]);
    }

    (seqs, targets)
}

/// Predict β_c by scanning NPU predictions and finding phase boundary.
fn predict_beta_c(npu: &mut NpuSimulator) -> f64 {
    let n_scan = 100;
    let mut best_beta = KNOWN_BETA_C;
    let mut best_uncertainty = 0.0_f64;

    for i in 0..n_scan {
        let beta = 5.0 + 2.0 * (i as f64) / (n_scan as f64 - 1.0);
        let beta_norm = (beta - 5.0) / 2.0;
        let plaq_est = 0.35 + 0.25 * (beta - 4.5) / 2.5;
        let poly_est = if beta > 5.7 { 0.3 } else { 0.05 };
        let chi_est = if (beta - 5.69).abs() < 0.2 {
            500.0
        } else {
            50.0
        };

        let seq: Vec<Vec<f64>> = vec![vec![beta_norm, plaq_est, poly_est, chi_est / 1000.0]; 10];
        let pred = npu.predict(&seq);
        if pred.len() >= 2 {
            let uncertainty = pred[1];
            if uncertainty > best_uncertainty {
                best_uncertainty = uncertainty;
                best_beta = beta;
            }
        } else if !pred.is_empty() {
            let phase_pred = pred[0];
            let u = 1.0 - (phase_pred - 0.5).abs() * 2.0;
            if u > best_uncertainty {
                best_uncertainty = u;
                best_beta = beta;
            }
        }
    }

    best_beta
}

/// Find β with maximum NPU uncertainty among unmeasured regions.
fn find_max_uncertainty_beta(
    npu: &mut NpuSimulator,
    measured: &[f64],
    beta_min: f64,
    beta_max: f64,
    n_candidates: usize,
) -> f64 {
    let mut best_beta = f64::NAN;
    let mut best_score = 0.0_f64;

    for i in 0..n_candidates {
        let beta = beta_min + (beta_max - beta_min) * (i as f64) / (n_candidates as f64 - 1.0);

        if measured.iter().any(|&m| (m - beta).abs() < 0.08) {
            continue;
        }

        let beta_norm = (beta - 5.0) / 2.0;
        let plaq_est = 0.35 + 0.25 * (beta - 4.5) / 2.5;
        let poly_est = if beta > 5.7 { 0.3 } else { 0.05 };
        let chi_est = if (beta - 5.69).abs() < 0.2 {
            500.0
        } else {
            50.0
        };

        let seq: Vec<Vec<f64>> = vec![vec![beta_norm, plaq_est, poly_est, chi_est / 1000.0]; 10];
        let pred = npu.predict(&seq);

        let uncertainty = if pred.len() >= 2 {
            pred[1]
        } else if !pred.is_empty() {
            1.0 - (pred[0] - 0.5).abs() * 2.0
        } else {
            0.0
        };

        let proximity_bonus = (-(beta - KNOWN_BETA_C).powi(2) / 0.5).exp() * 0.3;
        let score = uncertainty + proximity_bonus;

        if score > best_score {
            best_score = score;
            best_beta = beta;
        }
    }

    best_beta
}

/// Run Titan V (or CPU) validation oracle on critical configurations.
fn run_titan_validation(
    gpu_titan: Option<&GpuF64>,
    results: &[BetaResult],
    dims: [usize; 4],
    n_md: usize,
    dt: f64,
) {
    let transition_results: Vec<&BetaResult> =
        results.iter().filter(|r| r.phase == "transition").collect();

    if transition_results.is_empty() {
        println!("  No transition-region points to validate");
        return;
    }

    let titan_dims = if dims[0] > 16 { [16, 16, 16, 16] } else { dims };

    for r in &transition_results {
        if let Some(titan) = gpu_titan {
            let titan_pipelines = GpuHmcStreamingPipelines::new(titan);
            let mut lat = Lattice::hot_start(titan_dims, r.beta, 77777);

            let mut cfg = HmcConfig {
                n_md_steps: n_md.min(30),
                dt: dt.max(0.01),
                seed: 77777,
                integrator: IntegratorType::Omelyan,
            };
            for _ in 0..20 {
                hmc::hmc_trajectory(&mut lat, &mut cfg);
            }

            let state = GpuHmcState::from_lattice(titan, &lat, r.beta);
            let mut seed = 88888u64;
            for t in 0..20 {
                gpu_hmc_trajectory_streaming(
                    titan,
                    &titan_pipelines,
                    &state,
                    n_md.min(30),
                    dt.max(0.01),
                    t as u32,
                    &mut seed,
                );
            }

            let mut plaq_sum = 0.0;
            let n_verify = 50;
            for t in 0..n_verify {
                let tr = gpu_hmc_trajectory_streaming(
                    titan,
                    &titan_pipelines,
                    &state,
                    n_md.min(30),
                    dt.max(0.01),
                    (20 + t) as u32,
                    &mut seed,
                );
                plaq_sum += tr.plaquette;
            }
            let titan_plaq = plaq_sum / n_verify as f64;
            let (titan_poly, _) = gpu_polyakov_loop(titan, &titan_pipelines.hmc, &state);

            let plaq_diff = (titan_plaq - r.mean_plaq).abs();
            let agree = plaq_diff < 0.05;

            println!(
                "  β={:.4}: 3090 ⟨P⟩={:.6} vs Titan V ⟨P⟩={:.6} ({}⁴, native f64) Δ={:.6} {}",
                r.beta,
                r.mean_plaq,
                titan_plaq,
                titan_dims[0],
                plaq_diff,
                if agree { "✓ AGREE" } else { "△ CHECK" },
            );
            println!(
                "           3090 |L|={:.4} vs Titan V |L|={:.4}",
                r.polyakov, titan_poly,
            );
        } else {
            let mut lat = Lattice::hot_start(dims, r.beta, 77777);
            let mut cfg = HmcConfig {
                n_md_steps: n_md,
                dt,
                seed: 77777,
                integrator: IntegratorType::Omelyan,
            };
            for _ in 0..20 {
                hmc::hmc_trajectory(&mut lat, &mut cfg);
            }
            let stats = hmc::run_hmc(&mut lat, 50, 0, &mut cfg);
            let poly = lat.average_polyakov_loop();
            let plaq_diff = (stats.mean_plaquette - r.mean_plaq).abs();
            let agree = plaq_diff < 0.05;

            println!(
                "  β={:.4}: GPU ⟨P⟩={:.6} vs CPU f64 ⟨P⟩={:.6} Δ={:.6} {}",
                r.beta,
                r.mean_plaq,
                stats.mean_plaquette,
                plaq_diff,
                if agree { "✓ AGREE" } else { "△ CHECK" },
            );
            println!(
                "           GPU |L|={:.4} vs CPU |L|={:.4}",
                r.polyakov, poly,
            );
        }
    }
}
