// SPDX-License-Identifier: AGPL-3.0-only

//! Quenched β-scan with NPU offloading.
//!
//! Extracted from production_mixed_pipeline to reduce binary size.
//! Provides the quenched NPU worker (therm detect, reject predict, phase classify,
//! adaptive steer) and `run_beta_points_npu` for running β points with NPU assistance.

use crate::gpu::GpuF64;
use crate::lattice::gpu_hmc::{
    gpu_hmc_trajectory_streaming, gpu_links_to_lattice, gpu_polyakov_loop, GpuHmcState,
    GpuHmcStreamingPipelines,
};
use crate::lattice::hmc::{self, HmcConfig, IntegratorType};
use crate::lattice::wilson::Lattice;
use crate::md::reservoir::{EchoStateNetwork, EsnConfig, ExportedWeights, NpuSimulator};
use crate::production::{
    bootstrap_esn_from_trajectory_log, build_training_data, check_thermalization,
    find_max_uncertainty_beta, plaquette_variance, predict_beta_c, predict_rejection, BetaResult,
};
use crate::provenance::KNOWN_BETA_C_SU3_NT4 as KNOWN_BETA_C;
use std::io::Write as IoWrite;
use std::sync::mpsc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
//  Quenched NPU Types
// ═══════════════════════════════════════════════════════════════════

/// Message sent from the GPU/main thread to the quenched NPU worker.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum QuenchedNpuRequest {
    /// Placement A: check if thermalization has converged.
    ThermCheck { plaq_window: Vec<f64>, beta: f64 },
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
    Retrain { results: Vec<BetaResult> },
    /// Bootstrap ESN from a previous run's trajectory log (Placement E).
    BootstrapFromLog { path: String },
    /// Bootstrap ESN from saved weights JSON.
    BootstrapFromWeights { path: String },
    /// Export current ESN weights to disk for the next run.
    ExportWeights { path: String },
    /// Shut down the worker.
    Shutdown,
}

/// Response from the quenched NPU worker.
#[derive(Debug)]
#[allow(missing_docs)]
pub enum QuenchedNpuResponse {
    /// Thermalization: true = converged, can exit early.
    ThermConverged(bool),
    /// Rejection prediction: true = likely rejected, suggest abort bookkeeping.
    RejectPrediction {
        likely_rejected: bool,
        confidence: f64,
    },
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

/// NPU statistics for quenched mixed pipeline.
#[derive(Clone, Debug, Default)]
pub struct QuenchedNpuStats {
    /// Number of thermalization early-exits triggered.
    pub therm_early_exits: usize,
    /// Total thermalization trajectories saved by early-exit.
    pub therm_total_saved: usize,
    /// Number of rejection predictions.
    pub reject_predictions: usize,
    /// Number of rejection predictions that were correct.
    pub reject_correct: usize,
    /// Number of phase classifications.
    pub phase_classifications: usize,
    /// Number of steering queries.
    pub steer_queries: usize,
    /// Total NPU calls.
    pub total_npu_calls: usize,
}

impl QuenchedNpuStats {
    /// Create a new stats tracker with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Spawn the quenched NPU worker thread.
///
/// Returns (request_sender, response_receiver) for communication.
#[allow(clippy::expect_used)] // thread spawn failure is fatal
pub fn spawn_quenched_npu_worker() -> (
    mpsc::Sender<QuenchedNpuRequest>,
    mpsc::Receiver<QuenchedNpuResponse>,
) {
    let (req_tx, req_rx) = mpsc::channel::<QuenchedNpuRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<QuenchedNpuResponse>();

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
                esn.export_weights()
                    .map(|w| NpuSimulator::from_exported(&w))
            };

            for req in req_rx {
                match req {
                    QuenchedNpuRequest::ThermCheck { plaq_window, beta } => {
                        let converged = check_thermalization(&plaq_window, beta);
                        resp_tx
                            .send(QuenchedNpuResponse::ThermConverged(converged))
                            .ok();
                    }
                    QuenchedNpuRequest::RejectPredict {
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
                            .send(QuenchedNpuResponse::RejectPrediction {
                                likely_rejected,
                                confidence,
                            })
                            .ok();
                    }
                    QuenchedNpuRequest::PhaseClassify {
                        beta,
                        plaquette,
                        polyakov,
                        susceptibility,
                    } => {
                        let label = if let Some(ref mut n) = npu {
                            let beta_norm = (beta - 5.0) / 2.0;
                            let seq =
                                vec![
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
                        resp_tx.send(QuenchedNpuResponse::PhaseLabel(label)).ok();
                    }
                    QuenchedNpuRequest::SteerNextBeta {
                        measured_betas,
                        beta_min,
                        beta_max,
                    } => {
                        let next = if let Some(ref mut n) = npu {
                            find_max_uncertainty_beta(n, &measured_betas, beta_min, beta_max, 60)
                        } else {
                            f64::NAN
                        };
                        resp_tx.send(QuenchedNpuResponse::NextBeta(next)).ok();
                    }
                    QuenchedNpuRequest::Retrain { results } => {
                        let seed = 99 + results.len() as u64;
                        if let Some(new_npu) = make_esn(seed, &results) {
                            npu = Some(new_npu);
                        }
                        let beta_c = if let Some(ref mut n) = npu {
                            predict_beta_c(n)
                        } else {
                            KNOWN_BETA_C
                        };
                        resp_tx.send(QuenchedNpuResponse::Retrained { beta_c }).ok();
                    }
                    QuenchedNpuRequest::BootstrapFromLog { path } => {
                        let (n_points, beta_c) =
                            bootstrap_esn_from_trajectory_log(&path, &make_esn, &mut npu)
                                .unwrap_or((0, KNOWN_BETA_C));
                        resp_tx
                            .send(QuenchedNpuResponse::Bootstrapped { n_points, beta_c })
                            .ok();
                    }
                    QuenchedNpuRequest::BootstrapFromWeights { path } => {
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
                                        .send(QuenchedNpuResponse::Bootstrapped {
                                            n_points: 0,
                                            beta_c,
                                        })
                                        .ok();
                                }
                                Err(e) => {
                                    eprintln!(
                                        "  Warning: failed to parse weights from {path}: {e}"
                                    );
                                    resp_tx
                                        .send(QuenchedNpuResponse::Bootstrapped {
                                            n_points: 0,
                                            beta_c: KNOWN_BETA_C,
                                        })
                                        .ok();
                                }
                            },
                            Err(e) => {
                                eprintln!("  Warning: cannot read weights file {path}: {e}");
                                resp_tx
                                    .send(QuenchedNpuResponse::Bootstrapped {
                                        n_points: 0,
                                        beta_c: KNOWN_BETA_C,
                                    })
                                    .ok();
                            }
                        }
                    }
                    QuenchedNpuRequest::ExportWeights { path } => {
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
                                .send(QuenchedNpuResponse::WeightsSaved { path: path.clone() })
                                .ok();
                        } else {
                            resp_tx
                                .send(QuenchedNpuResponse::WeightsSaved {
                                    path: String::new(),
                                })
                                .ok();
                        }
                    }
                    QuenchedNpuRequest::Shutdown => break,
                }
            }
        })
        .expect("spawn NPU worker thread");

    (req_tx, resp_rx)
}

/// Run a set of β points with NPU-assisted thermalization, rejection prediction,
/// phase classification, and per-trajectory logging.
#[allow(clippy::too_many_arguments)]
pub fn run_beta_points_npu(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    betas: &[f64],
    dims: [usize; 4],
    n_therm: usize,
    n_meas: usize,
    n_md: usize,
    dt: f64,
    base_seed: u64,
    npu_tx: &mpsc::Sender<QuenchedNpuRequest>,
    npu_rx: &mpsc::Receiver<QuenchedNpuResponse>,
    npu_stats: &mut QuenchedNpuStats,
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
            let r =
                gpu_hmc_trajectory_streaming(gpu, pipelines, &state, n_md, dt, i as u32, &mut seed);
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
                    .send(QuenchedNpuRequest::ThermCheck {
                        plaq_window: plaq_history[window_start..].to_vec(),
                        beta,
                    })
                    .ok();
                npu_stats.total_npu_calls += 1;

                if let Ok(QuenchedNpuResponse::ThermConverged(converged)) = npu_rx.recv() {
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
                .send(QuenchedNpuRequest::RejectPredict {
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

            if let Ok(QuenchedNpuResponse::RejectPrediction {
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
            .send(QuenchedNpuRequest::PhaseClassify {
                beta,
                plaquette: mean_plaq,
                polyakov: mean_poly,
                susceptibility,
            })
            .ok();
        npu_stats.phase_classifications += 1;
        npu_stats.total_npu_calls += 1;

        let phase = match npu_rx.recv() {
            Ok(QuenchedNpuResponse::PhaseLabel(l)) => l,
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
            ..Default::default()
        });

        if let Some(ref mut w) = traj_writer {
            w.flush().ok();
        }
    }

    out
}
