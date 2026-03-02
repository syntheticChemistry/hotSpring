// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU worker thread for the 11-head dynamical mixed pipeline.
//!
//! Handles pre/during/post computation screening, Brain Layer 1 (CG residual
//! monitoring), and Brain Layer 3 (proxy features).

use crate::lattice::gpu_hmc::{BrainInterrupt, CgResidualUpdate};
use crate::md::reservoir::{
    heads, EchoStateNetwork, EsnConfig, ExportedWeights, HeadGroupDisagreement, MultiHeadNpu,
};
use crate::production::{
    check_thermalization, plaquette_variance, predict_rejection, AttentionState, BetaResult,
    MetaRow,
};
use crate::provenance::KNOWN_BETA_C_SU3_NT4;
use crate::proxy::ProxyFeatures;
use std::sync::mpsc;

/// Message sent from the GPU/main thread to the NPU worker.
#[derive(Debug)]
pub enum NpuRequest {
    // ─── Pre-computation ───
    PreScreenBeta {
        candidates: Vec<f64>,
        meta_context: Vec<MetaRow>,
    },
    SuggestParameters {
        lattice: usize,
        beta: f64,
        mass: f64,
    },
    PredictCgIters {
        beta: f64,
        mass: f64,
        lattice: usize,
    },
    PredictQuenchedLength {
        beta: f64,
        mass: f64,
        lattice: usize,
        meta_context: Vec<MetaRow>,
    },

    // ─── During computation (quenched phase) ───
    QuenchedThermCheck {
        plaq_window: Vec<f64>,
        beta: f64,
    },

    // ─── During computation (dynamical phase) ───
    ThermCheck {
        plaq_window: Vec<f64>,
        beta: f64,
    },
    RejectPredict {
        beta: f64,
        plaquette: f64,
        delta_h: f64,
        acceptance_rate: f64,
    },
    PhaseClassify {
        beta: f64,
        plaquette: f64,
        polyakov: f64,
        susceptibility: f64,
    },

    // ─── Post-computation ───
    QualityScore {
        result: BetaResult,
    },
    AnomalyCheck {
        plaq: f64,
        delta_h: f64,
        cg_iters: usize,
        acceptance: f64,
    },
    SteerAdaptive {
        measured_betas: Vec<f64>,
        queued_betas: Vec<f64>,
        beta_min: f64,
        beta_max: f64,
        n_candidates: usize,
    },
    RecommendNextRun {
        all_results: Vec<BetaResult>,
        meta_table: Vec<MetaRow>,
    },
    // ─── Brain Layer 1: CG residual monitoring ───
    CgResidual(CgResidualUpdate),
    // ─── Brain Layer 3: Proxy features from CPU cortex ───
    ProxyFeatures {
        beta: f64,
        level_spacing_ratio: f64,
        lambda_min: f64,
        ipr: f64,
        tier: u8,
    },

    // ─── Lifecycle ───
    Retrain {
        results: Vec<BetaResult>,
    },
    BootstrapFromMeta {
        rows: Vec<MetaRow>,
    },
    BootstrapFromWeights {
        path: String,
    },
    ExportWeights {
        path: String,
    },
    DisagreementQuery {
        beta: f64,
        plaq: f64,
        mass: f64,
        chi: f64,
        acceptance: f64,
    },
    Shutdown,
}

/// Response from the NPU worker.
#[derive(Debug)]
pub enum NpuResponse {
    BetaPriorities(Vec<(f64, f64)>),
    ParameterSuggestion {
        dt: f64,
        n_md: usize,
    },
    CgEstimate(usize),
    QuenchedLengthEstimate(usize),
    QuenchedThermConverged(bool),
    ThermConverged(bool),
    RejectPrediction {
        likely_rejected: bool,
        _confidence: f64,
    },
    PhaseLabel(&'static str),
    Quality(f64),
    AnomalyFlag {
        is_anomaly: bool,
        _score: f64,
    },
    AdaptiveSteered(Option<f64>),
    NextRunPlan {
        betas: Vec<f64>,
        mass: f64,
        lattice: usize,
    },
    Retrained {
        beta_c: f64,
    },
    Bootstrapped {
        n_points: usize,
    },
    WeightsSaved {
        path: String,
    },
    ResidualAck,
    ProxyFeaturesAck,
    DisagreementSnapshot {
        delta_cg: f64,
        delta_phase: f64,
        delta_anomaly: f64,
        delta_priority: f64,
        urgency: f64,
    },
}

/// Spawn result includes the interrupt channel for brain-mode CG monitoring.
pub struct NpuWorkerHandles {
    pub npu_tx: mpsc::Sender<NpuRequest>,
    pub npu_rx: mpsc::Receiver<NpuResponse>,
    pub interrupt_rx: mpsc::Receiver<BrainInterrupt>,
}

/// Spawn the NPU worker thread. Returns handles for request/response and brain interrupt.
#[allow(clippy::expect_used)]
pub fn spawn_npu_worker() -> NpuWorkerHandles {
    let (req_tx, req_rx) = mpsc::channel::<NpuRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<NpuResponse>();
    let (interrupt_tx_out, interrupt_rx_out) = mpsc::channel::<BrainInterrupt>();

    std::thread::Builder::new()
        .name("npu-cerebellum".into())
        .spawn(move || {
            let mut multi_npu: Option<MultiHeadNpu> = None;
            let interrupt_tx: Option<mpsc::Sender<BrainInterrupt>> = Some(interrupt_tx_out);

            // Brain Layer 1 + 4 state
            let mut residual_history: Vec<(usize, f64)> = Vec::new();
            let mut attention_state = AttentionState::Green;
            let mut green_count: usize = 0;
            let mut yellow_count: usize = 0;

            #[allow(unused_assignments)]
            let mut latest_proxy: Option<ProxyFeatures> = None;

            let make_multi_esn = |seed: u64, results: &[BetaResult]| -> Option<MultiHeadNpu> {
                if results.is_empty() {
                    return None;
                }
                let (seqs, tgts) = build_training_data(results);
                let mut esn = EchoStateNetwork::new(EsnConfig {
                    input_size: 5,
                    output_size: heads::NUM_HEADS,
                    regularization: 1e-3,
                    seed,
                    ..EsnConfig::default()
                });
                esn.train(&seqs, &tgts);
                esn.export_weights()
                    .map(|w| MultiHeadNpu::from_exported(&w))
            };

            for req in req_rx {
                match req {
                    NpuRequest::PreScreenBeta {
                        candidates,
                        meta_context,
                    } => {
                        let priorities: Vec<(f64, f64)> = candidates
                            .iter()
                            .map(|&beta| {
                                let score = if let Some(ref mut npu) = multi_npu {
                                    let meta_plaq = meta_context
                                        .iter()
                                        .find(|r| (r.beta - beta).abs() < 0.01)
                                        .map_or(0.5, |r| r.mean_plaq);
                                    let meta_chi = meta_context
                                        .iter()
                                        .find(|r| (r.beta - beta).abs() < 0.01)
                                        .map_or(10.0, |r| r.chi);
                                    let input = vec![
                                        (beta - 5.0) / 2.0,
                                        meta_plaq,
                                        0.1,
                                        meta_chi / 1000.0,
                                        0.5,
                                    ];
                                    let seq = vec![input; 10];
                                    npu.predict_head(&seq, heads::BETA_PRIORITY)
                                } else {
                                    (-(beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.3).exp()
                                };
                                (beta, score)
                            })
                            .collect();
                        resp_tx.send(NpuResponse::BetaPriorities(priorities)).ok();
                    }

                    NpuRequest::SuggestParameters {
                        lattice,
                        beta,
                        mass,
                    } => {
                        let (dt, n_md) = if let Some(ref mut npu) = multi_npu {
                            let input =
                                vec![(beta - 5.0) / 2.0, lattice as f64 / 32.0, mass, 0.5, 0.0];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::PARAM_SUGGEST);
                            let dt_suggest = 0.001 + raw.abs() * 0.05;
                            let n_md_suggest = ((0.5 / dt_suggest).round() as usize).max(10);
                            (dt_suggest, n_md_suggest)
                        } else {
                            let vol = (lattice as f64).powi(4);
                            let scale = (4096.0 / vol).powf(0.25);
                            let dt = (0.05 * scale).max(0.001);
                            let n_md = ((0.5 / dt).round() as usize).max(20);
                            (dt, n_md)
                        };
                        resp_tx
                            .send(NpuResponse::ParameterSuggestion { dt, n_md })
                            .ok();
                    }

                    NpuRequest::PredictCgIters {
                        beta,
                        mass,
                        lattice,
                    } => {
                        let est = if let Some(ref mut npu) = multi_npu {
                            let input =
                                vec![(beta - 5.0) / 2.0, mass, lattice as f64 / 32.0, 0.0, 0.0];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::CG_ESTIMATE);
                            (raw.abs() * 500.0).round() as usize
                        } else {
                            let vol = (lattice as f64).powi(4);
                            (100.0 + vol.sqrt() / mass.max(0.01)).round() as usize
                        };
                        resp_tx.send(NpuResponse::CgEstimate(est)).ok();
                    }

                    NpuRequest::PredictQuenchedLength {
                        beta,
                        mass,
                        lattice,
                        meta_context,
                    } => {
                        let steps = if let Some(ref mut npu) = multi_npu {
                            let meta_plaq = meta_context
                                .iter()
                                .find(|r| (r.beta - beta).abs() < 0.01)
                                .map_or(0.5, |r| r.mean_plaq);
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                meta_plaq,
                                mass,
                                lattice as f64 / 32.0,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUENCHED_LENGTH);
                            (raw.abs() * 200.0).round().clamp(5.0, 200.0) as usize
                        } else {
                            let proximity = (-(beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.3).exp();
                            let base = 20 + (80.0 * proximity) as usize;
                            base.min(lattice * 10)
                        };
                        resp_tx
                            .send(NpuResponse::QuenchedLengthEstimate(steps))
                            .ok();
                    }

                    NpuRequest::QuenchedThermCheck { plaq_window, beta } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean =
                                plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                mean,
                                var.sqrt(),
                                plaq_window.len() as f64 / 100.0,
                                1.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUENCHED_THERM);
                            raw > 0.5
                        } else {
                            check_thermalization(&plaq_window, beta)
                        };
                        resp_tx
                            .send(NpuResponse::QuenchedThermConverged(converged))
                            .ok();
                    }

                    NpuRequest::ThermCheck { plaq_window, beta } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean =
                                plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                mean,
                                var.sqrt(),
                                plaq_window.len() as f64 / 200.0,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::THERM_DETECT);
                            raw > 0.5
                        } else {
                            check_thermalization(&plaq_window, beta)
                        };
                        resp_tx.send(NpuResponse::ThermConverged(converged)).ok();
                    }

                    NpuRequest::RejectPredict {
                        beta,
                        plaquette,
                        delta_h,
                        acceptance_rate,
                    } => {
                        let (likely_rejected, confidence) = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                plaquette,
                                delta_h / 10.0,
                                acceptance_rate,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::REJECT_PREDICT);
                            (raw > 0.5, raw.clamp(0.0, 1.0))
                        } else {
                            predict_rejection(0.0, 0.0, 0.0, delta_h, acceptance_rate)
                        };
                        resp_tx
                            .send(NpuResponse::RejectPrediction {
                                likely_rejected,
                                _confidence: confidence,
                            })
                            .ok();
                    }

                    NpuRequest::PhaseClassify {
                        beta,
                        plaquette,
                        polyakov,
                        susceptibility,
                    } => {
                        let label = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                (beta - 5.0) / 2.0,
                                plaquette,
                                polyakov,
                                susceptibility / 1000.0,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::PHASE_CLASSIFY);
                            match () {
                                () if raw > 0.6 => "deconfined",
                                () if raw > 0.3 => "transition",
                                () => "confined",
                            }
                        } else {
                            match () {
                                () if beta > 5.79 => "deconfined",
                                () if beta > 5.59 => "transition",
                                () => "confined",
                            }
                        };
                        resp_tx.send(NpuResponse::PhaseLabel(label)).ok();
                    }

                    NpuRequest::QualityScore { result } => {
                        let score = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                result.n_traj as f64 / 1000.0,
                                result.std_plaq / result.mean_plaq.abs().max(1e-10),
                                result.acceptance,
                                result.susceptibility / 1000.0,
                                result.mean_cg_iters / 500.0,
                            ];
                            let seq = vec![input; 10];
                            npu.predict_head(&seq, heads::QUALITY_SCORE).clamp(0.0, 1.0)
                        } else {
                            let acc_ok = if result.acceptance > 0.3 { 0.4 } else { 0.1 };
                            let stats_ok = if result.n_traj >= 100 { 0.3 } else { 0.1 };
                            let cg_ok = if result.mean_cg_iters < 1000.0 {
                                0.3
                            } else {
                                0.1
                            };
                            acc_ok + stats_ok + cg_ok
                        };
                        resp_tx.send(NpuResponse::Quality(score)).ok();
                    }

                    NpuRequest::AnomalyCheck {
                        plaq,
                        delta_h,
                        cg_iters,
                        acceptance,
                    } => {
                        let (is_anomaly, score) = if let Some(ref mut npu) = multi_npu {
                            let input = vec![
                                plaq,
                                delta_h / 10.0,
                                cg_iters as f64 / 500.0,
                                acceptance,
                                0.0,
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::ANOMALY_DETECT);
                            (raw > 0.7, raw.clamp(0.0, 1.0))
                        } else {
                            let anomaly = delta_h.abs() > 50.0
                                || cg_iters > 4000
                                || !(0.0..=1.0).contains(&plaq);
                            let s = if anomaly { 0.9 } else { 0.1 };
                            (anomaly, s)
                        };
                        resp_tx
                            .send(NpuResponse::AnomalyFlag {
                                is_anomaly,
                                _score: score,
                            })
                            .ok();
                    }

                    NpuRequest::SteerAdaptive {
                        measured_betas,
                        queued_betas,
                        beta_min,
                        beta_max,
                        n_candidates,
                    } => {
                        let suggestion = if let Some(ref mut npu) = multi_npu {
                            let mut best_beta = None;
                            let mut best_score = f64::NEG_INFINITY;
                            let step = (beta_max - beta_min) / (n_candidates as f64 + 1.0);
                            let exclusion = step * 0.3_f64.max(0.025);
                            for ci in 1..=n_candidates {
                                let candidate = beta_min + ci as f64 * step;
                                let too_close_measured = measured_betas
                                    .iter()
                                    .any(|&m| (m - candidate).abs() < exclusion);
                                let too_close_queued = queued_betas
                                    .iter()
                                    .any(|&q| (q - candidate).abs() < exclusion);
                                if too_close_measured || too_close_queued {
                                    continue;
                                }
                                let input = vec![
                                    (candidate - 5.0) / 2.0,
                                    0.5,
                                    0.0,
                                    measured_betas.len() as f64 / 20.0,
                                    0.0,
                                ];
                                let seq = vec![input; 10];
                                let all = npu.predict_all_heads(&seq);
                                let priority = all[heads::BETA_PRIORITY];
                                let uncertainty = (all[heads::QUALITY_SCORE] - 0.5).abs();
                                let score = priority + uncertainty * 0.5;
                                if score > best_score {
                                    best_score = score;
                                    best_beta = Some(candidate);
                                }
                            }
                            best_beta
                        } else {
                            let all_known: Vec<f64> = measured_betas
                                .iter()
                                .chain(queued_betas.iter())
                                .copied()
                                .collect();
                            let gaps = find_largest_gaps(&all_known, beta_min, beta_max, 1);
                            gaps.into_iter().next()
                        };
                        resp_tx.send(NpuResponse::AdaptiveSteered(suggestion)).ok();
                    }

                    NpuRequest::RecommendNextRun {
                        all_results,
                        meta_table,
                    } => {
                        let plan = if let Some(ref mut npu) = multi_npu {
                            let last = all_results.last();
                            let input = vec![
                                all_results.len() as f64 / 10.0,
                                last.map_or(0.5, |r| r.mean_plaq),
                                last.map_or(0.5, |r| r.acceptance),
                                meta_table.len() as f64 / 100.0,
                                last.map_or(0.1, |r| r.mass),
                            ];
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::NEXT_RUN_RECOMMEND);
                            let suggested_beta = 5.0 + raw.abs() * 2.0;
                            let mass = last.map_or(0.1, |r| r.mass);
                            let lattice = 8;
                            (vec![suggested_beta], mass, lattice)
                        } else {
                            let measured: Vec<f64> = all_results.iter().map(|r| r.beta).collect();
                            let gaps = find_largest_gaps(&measured, 5.0, 7.0, 3);
                            let mass = all_results.first().map_or(0.1, |r| r.mass);
                            (gaps, mass, 8)
                        };
                        resp_tx
                            .send(NpuResponse::NextRunPlan {
                                betas: plan.0,
                                mass: plan.1,
                                lattice: plan.2,
                            })
                            .ok();
                    }

                    NpuRequest::Retrain { results } => {
                        let seed = 99 + results.len() as u64;
                        if let Some(new_npu) = make_multi_esn(seed, &results) {
                            multi_npu = Some(new_npu);
                        }
                        let beta_c = estimate_beta_c(&results);
                        resp_tx.send(NpuResponse::Retrained { beta_c }).ok();
                    }

                    NpuRequest::BootstrapFromMeta { rows } => {
                        let n = rows.len();
                        let results: Vec<BetaResult> = rows
                            .into_iter()
                            .map(|r| BetaResult {
                                beta: r.beta,
                                mass: r.mass.unwrap_or(0.0),
                                mean_plaq: r.mean_plaq,
                                std_plaq: 0.0,
                                polyakov: 0.0,
                                susceptibility: r.chi,
                                action_density: 6.0 * (1.0 - r.mean_plaq),
                                acceptance: r.acceptance,
                                mean_cg_iters: r.mean_cg_iters,
                                n_traj: r.n_meas,
                                wall_s: 0.0,
                                phase: if r.beta < 5.6 {
                                    "confined"
                                } else if r.beta > 5.8 {
                                    "deconfined"
                                } else {
                                    "transition"
                                },
                                therm_used: 0,
                                therm_budget: 0,
                                dt_used: 0.01,
                                n_md_used: 100,
                                npu_therm_early_exit: false,
                                npu_quenched_budget: 0,
                                npu_quenched_used: 0,
                                npu_quenched_early_exit: false,
                                npu_reject_predictions: 0,
                                npu_reject_correct: 0,
                                npu_anomalies: 0,
                                npu_cg_check_interval: 10,
                            })
                            .collect();
                        if let Some(new_npu) = make_multi_esn(42, &results) {
                            multi_npu = Some(new_npu);
                        }
                        resp_tx.send(NpuResponse::Bootstrapped { n_points: n }).ok();
                    }

                    NpuRequest::BootstrapFromWeights { path } => {
                        match std::fs::read_to_string(&path) {
                            Ok(json) => match serde_json::from_str::<ExportedWeights>(&json) {
                                Ok(weights) => {
                                    multi_npu = Some(MultiHeadNpu::from_exported(&weights));
                                    resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 }).ok();
                                }
                                Err(e) => {
                                    eprintln!("  Warning: failed to parse weights {path}: {e}");
                                    resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 }).ok();
                                }
                            },
                            Err(e) => {
                                eprintln!("  Warning: cannot read {path}: {e}");
                                resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 }).ok();
                            }
                        }
                    }

                    NpuRequest::ExportWeights { path } => {
                        let saved = if let Some(ref mut npu) = multi_npu {
                            let base = npu.base_mut();
                            let weights = ExportedWeights {
                                w_in: base.export_w_in(),
                                w_res: base.export_w_res(),
                                w_out: base.export_w_out(),
                                input_size: base.input_size(),
                                reservoir_size: base.reservoir_size(),
                                output_size: base.output_size(),
                                leak_rate: base.leak_rate(),
                            };
                            if let Some(parent) = std::path::Path::new(&path).parent() {
                                std::fs::create_dir_all(parent).ok();
                            }
                            std::fs::write(
                                &path,
                                serde_json::to_string(&weights).unwrap_or_default(),
                            )
                            .is_ok()
                        } else {
                            false
                        };
                        resp_tx
                            .send(NpuResponse::WeightsSaved {
                                path: if saved { path } else { String::new() },
                            })
                            .ok();
                    }

                    NpuRequest::CgResidual(update) => {
                        residual_history.push((update.iteration, update.rz_new));
                        if residual_history.len() > 50 {
                            residual_history.drain(..residual_history.len() - 50);
                        }

                        let anomaly_score = if let Some(ref mut npu) = multi_npu {
                            let window: Vec<f64> = residual_history
                                .iter()
                                .rev()
                                .take(10)
                                .map(|(_, rz)| rz.log10().max(-20.0) / 20.0)
                                .collect();
                            let input: Vec<f64> = window
                                .iter()
                                .copied()
                                .chain(std::iter::repeat(0.0))
                                .take(5)
                                .collect();
                            let _ = npu.base_mut().predict_return_state(&[input]);
                            npu.base_mut().readout_head(heads::CG_RESIDUAL_MONITOR)
                        } else if residual_history.len() >= 3 {
                            let recent: Vec<f64> = residual_history
                                .iter()
                                .rev()
                                .take(3)
                                .map(|(_, rz)| *rz)
                                .collect();
                            if recent.windows(2).all(|w| w[0] >= w[1]) {
                                0.8
                            } else {
                                0.1
                            }
                        } else {
                            0.0
                        };

                        match attention_state {
                            AttentionState::Green => {
                                if anomaly_score > 0.7 {
                                    attention_state = AttentionState::Red;
                                    yellow_count = 0;
                                    green_count = 0;
                                    if residual_history.len() >= 3 {
                                        let recent: Vec<f64> = residual_history
                                            .iter()
                                            .rev()
                                            .take(3)
                                            .map(|(_, rz)| *rz)
                                            .collect();
                                        if recent.windows(2).all(|w| w[0] >= w[1]) {
                                            if let Some(ref itx) = interrupt_tx {
                                                let _ = itx.send(BrainInterrupt::KillCg);
                                            }
                                        }
                                    }
                                } else if anomaly_score > 0.3 {
                                    attention_state = AttentionState::Yellow;
                                    green_count = 0;
                                    if let Some(ref itx) = interrupt_tx {
                                        let _ = itx.send(BrainInterrupt::AdjustCheckInterval(20));
                                    }
                                }
                            }
                            AttentionState::Yellow => {
                                if anomaly_score > 0.7 {
                                    yellow_count += 1;
                                    if yellow_count >= 2 {
                                        attention_state = AttentionState::Red;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ =
                                                itx.send(BrainInterrupt::AdjustCheckInterval(5));
                                        }
                                    }
                                } else if anomaly_score < 0.3 {
                                    green_count += 1;
                                    if green_count >= 3 {
                                        attention_state = AttentionState::Green;
                                        green_count = 0;
                                        yellow_count = 0;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ =
                                                itx.send(BrainInterrupt::AdjustCheckInterval(100));
                                        }
                                    }
                                } else {
                                    green_count = 0;
                                }
                            }
                            AttentionState::Red => {
                                if residual_history.len() >= 3 {
                                    let recent: Vec<f64> = residual_history
                                        .iter()
                                        .rev()
                                        .take(3)
                                        .map(|(_, rz)| *rz)
                                        .collect();
                                    let diverging = recent.windows(2).all(|w| w[0] >= w[1]);
                                    if diverging {
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ = itx.send(BrainInterrupt::KillCg);
                                        }
                                    }
                                }
                                if anomaly_score < 0.3 {
                                    green_count += 1;
                                    if green_count >= 3 {
                                        attention_state = AttentionState::Yellow;
                                        green_count = 0;
                                        yellow_count = 0;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ =
                                                itx.send(BrainInterrupt::AdjustCheckInterval(20));
                                        }
                                    }
                                } else {
                                    green_count = 0;
                                }
                            }
                        }
                        resp_tx.send(NpuResponse::ResidualAck).ok();
                    }

                    NpuRequest::ProxyFeatures {
                        beta,
                        level_spacing_ratio,
                        lambda_min,
                        ipr,
                        tier,
                    } => {
                        latest_proxy = Some(ProxyFeatures {
                            beta,
                            level_spacing_ratio,
                            lambda_min,
                            ipr,
                            bandwidth: 0.0,
                            phase: String::new(),
                            tier,
                            wall_ms: 0.0,
                        });
                        if let Some(ref proxy) = latest_proxy {
                            println!(
                                "  [Brain L3] Proxy stored: β={:.4}, tier={}",
                                proxy.beta, proxy.tier
                            );
                        }
                        resp_tx.send(NpuResponse::ProxyFeaturesAck).ok();
                    }

                    NpuRequest::DisagreementQuery {
                        beta,
                        plaq,
                        mass,
                        chi,
                        acceptance,
                    } => {
                        let dis = if let Some(ref mut npu) = multi_npu {
                            let beta_norm = (beta - 5.0) / 2.0;
                            let input: Vec<Vec<f64>> = (0..10)
                                .map(|_| vec![beta_norm, plaq, mass, chi / 1000.0, acceptance])
                                .collect();
                            let (outputs, d) = npu.predict_with_disagreement(&input);
                            let _ = outputs;
                            d
                        } else {
                            HeadGroupDisagreement::default()
                        };
                        resp_tx
                            .send(NpuResponse::DisagreementSnapshot {
                                delta_cg: dis.delta_cg,
                                delta_phase: dis.delta_phase,
                                delta_anomaly: dis.delta_anomaly,
                                delta_priority: dis.delta_priority,
                                urgency: dis.urgency(),
                            })
                            .ok();
                    }

                    NpuRequest::Shutdown => break,
                }
            }
        })
        .expect("NPU worker thread spawn");

    NpuWorkerHandles {
        npu_tx: req_tx,
        npu_rx: resp_rx,
        interrupt_rx: interrupt_rx_out,
    }
}

fn estimate_beta_c(results: &[BetaResult]) -> f64 {
    if results.is_empty() {
        return KNOWN_BETA_C_SU3_NT4;
    }
    let max_chi_r = results
        .iter()
        .max_by(|a, b| a.susceptibility.total_cmp(&b.susceptibility));
    max_chi_r.map_or(KNOWN_BETA_C_SU3_NT4, |r| r.beta)
}

fn find_largest_gaps(measured: &[f64], min: f64, max: f64, n: usize) -> Vec<f64> {
    let mut sorted = measured.to_vec();
    sorted.push(min);
    sorted.push(max);
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();

    let mut gaps: Vec<(f64, f64)> = sorted
        .windows(2)
        .map(|w| (w[1] - w[0], f64::midpoint(w[0], w[1])))
        .collect();
    gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
    gaps.iter().take(n).map(|g| g.1).collect()
}

fn build_training_data(results: &[BetaResult]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for r in results {
        let beta_norm = (r.beta - 5.0) / 2.0;
        let phase_val = match r.phase {
            "deconfined" => 1.0,
            "transition" => 0.5,
            _ => 0.0,
        };
        let proximity = (-(r.beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.1).exp();
        let anomaly_val = f64::from(r.acceptance < 0.05 || r.mean_cg_iters > 3000.0);
        let quality = (r.acceptance * 0.4
            + (1.0 - r.std_plaq / r.mean_plaq.abs().max(1e-10)).clamp(0.0, 1.0) * 0.3
            + (r.n_traj as f64 / 1000.0).min(1.0) * 0.3)
            .clamp(0.0, 1.0);
        let cg_norm = r.mean_cg_iters / 500.0;
        let quenched_therm_target = f64::from(r.npu_quenched_early_exit);

        let anderson_phase = if r.beta > 5.5 {
            1.0
        } else if r.beta < 5.0 {
            0.0
        } else {
            0.5
        };
        let potts_phase = if r.beta > 5.8 {
            1.0
        } else if r.beta < 5.2 {
            0.0
        } else {
            0.5
        };
        let target_acc = 0.70;
        let acc_error = r.acceptance - target_acc;
        let optimal_dt = (r.dt_used * (1.0 - 0.5 * acc_error)).clamp(0.002, 0.02);
        let optimal_nmd = ((1.0 / optimal_dt).round() / 200.0).clamp(0.0, 1.0);

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.005 * ((j as f64) * 0.7).sin();
                vec![
                    beta_norm,
                    r.mean_plaq + noise * r.std_plaq,
                    r.mass,
                    r.susceptibility / 1000.0,
                    r.acceptance,
                ]
            })
            .collect();
        seqs.push(seq);

        let mut t = vec![0.0; heads::NUM_HEADS];

        t[heads::A0_ANDERSON_CG_COST] = cg_norm;
        t[heads::A1_ANDERSON_PHASE] = anderson_phase;
        t[heads::A2_ANDERSON_LAMBDA_MIN] = (1.0 / (cg_norm + 0.01)).clamp(0.0, 1.0);
        t[heads::A3_ANDERSON_ANOMALY] = anomaly_val;
        t[heads::A4_ANDERSON_THERM] = quenched_therm_target;
        t[heads::A5_ANDERSON_PRIORITY] = proximity;

        t[heads::B0_QCD_CG_COST] = cg_norm;
        t[heads::B1_QCD_PHASE] = phase_val;
        t[heads::B2_QCD_ACCEPTANCE] = 1.0 - r.acceptance;
        t[heads::B3_QCD_ANOMALY] = anomaly_val;
        t[heads::B4_QCD_THERM] = if r.acceptance > 0.3 { 1.0 } else { 0.0 };
        t[heads::B5_QCD_PRIORITY] = proximity;

        t[heads::C0_POTTS_CG_COST] = cg_norm;
        t[heads::C1_POTTS_PHASE] = potts_phase;
        t[heads::C2_POTTS_BETA_C] = (r.beta - KNOWN_BETA_C_SU3_NT4).abs().min(1.0);
        t[heads::C3_POTTS_ANOMALY] = anomaly_val;
        t[heads::C4_POTTS_ORDER] = r.polyakov.abs().clamp(0.0, 1.0);
        t[heads::C5_POTTS_PRIORITY] = proximity;

        t[heads::D0_NEXT_BETA] = proximity;
        t[heads::D1_OPTIMAL_DT] = optimal_dt;
        t[heads::D2_OPTIMAL_NMD] = optimal_nmd;
        t[heads::D3_CHECK_INTERVAL] = if cg_norm > 0.5 { 0.2 } else { 0.8 };
        t[heads::D4_KILL_DECISION] = if r.mean_cg_iters > 400.0 { 0.8 } else { 0.1 };
        t[heads::D5_SKIP_DECISION] = if quality < 0.2 { 0.8 } else { 0.1 };

        t[heads::E0_RESIDUAL_ETA] = cg_norm;
        t[heads::E1_RESIDUAL_ANOMALY] = anomaly_val;
        t[heads::E2_CONVERGENCE_RATE] = (1.0 - cg_norm).clamp(0.0, 1.0);
        t[heads::E3_STALL_DETECTOR] = if r.mean_cg_iters > 300.0 { 0.5 } else { 0.0 };
        t[heads::E4_DIVERGENCE_DETECTOR] = if anomaly_val > 0.5 { 0.8 } else { 0.0 };
        t[heads::E5_QUALITY_FORECAST] = quality;

        t[heads::M0_CG_CONSENSUS] = cg_norm;
        t[heads::M1_PHASE_CONSENSUS] = phase_val;
        t[heads::M2_CG_UNCERTAINTY] = (anderson_phase - phase_val).abs();
        t[heads::M3_PHASE_UNCERTAINTY] = (potts_phase - phase_val).abs();
        let proxy_agrees = (anderson_phase - phase_val).abs() < 0.3;
        t[heads::M4_PROXY_TRUST] = if proxy_agrees { 0.8 } else { 0.3 };
        t[heads::M5_ATTENTION_LEVEL] = match () {
            () if anomaly_val > 0.5 => 0.8,
            () if cg_norm > 0.5 => 0.4,
            () => 0.1,
        };

        targets.push(t);
    }

    (seqs, targets)
}
