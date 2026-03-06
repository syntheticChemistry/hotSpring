// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU worker thread for the 11-head dynamical mixed pipeline.
//!
//! Handles pre/during/post computation screening, Brain Layer 1 (CG residual
//! monitoring), and Brain Layer 3 (proxy features).

mod head_confidence;
mod messages;
mod training;

use crate::lattice::gpu_hmc::BrainInterrupt;
use crate::md::reservoir::{
    heads, Activation, EchoStateNetwork, EsnConfig, HeadGroupDisagreement, MultiHeadNpu,
};
use crate::production::checkpoint::{
    load_esn_weights, load_nautilus_shell, nautilus_shell_path_from_weights, save_esn_weights,
    save_nautilus_shell,
};
use crate::production::sub_models::SubModelRegistry;
use crate::production::trajectory_input::{canonical_input, canonical_seq, heuristic_phase};
use crate::production::{
    check_thermalization, plaquette_variance, predict_rejection, AttentionState, BetaResult,
};
use crate::provenance::KNOWN_BETA_C_SU3_NT4;
use crate::proxy::ProxyFeatures;
use barracuda::nautilus::{BetaObservation, NautilusBrain, NautilusBrainConfig};
use head_confidence::HeadConfidence;
use std::sync::mpsc;
use training::{build_training_data, estimate_beta_c, find_largest_gaps};

pub use crate::production::trajectory_input::{
    trajectory_input, trajectory_input_with_proxy, TRAJECTORY_INPUT_DIM,
};
pub use messages::{NpuRequest, NpuResponse, NpuWorkerHandles};

/// Spawn the NPU worker thread. Returns handles for request/response and brain interrupt.
///
/// # Errors
/// Returns `Err` if the thread fails to spawn (OOM, resource exhaustion).
pub fn spawn_npu_worker(lattice: usize) -> Result<NpuWorkerHandles, std::io::Error> {
    let (req_tx, req_rx) = mpsc::channel::<NpuRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<NpuResponse>();
    let (interrupt_tx_out, interrupt_rx_out) = mpsc::channel::<BrainInterrupt>();

    std::thread::Builder::new()
        .name("npu-cerebellum".into())
        .spawn(move || {
            let mut multi_npu: Option<MultiHeadNpu> = None;
            let interrupt_tx: Option<mpsc::Sender<BrainInterrupt>> = Some(interrupt_tx_out);

            let mut residual_history: Vec<(usize, f64)> = Vec::new();
            let mut attention_state = AttentionState::Green;
            let mut green_count: usize = 0;
            let mut yellow_count: usize = 0;

            #[allow(unused_assignments)]
            let mut latest_proxy: Option<ProxyFeatures> = None;

            let mut last_plaq: f64 = 0.5;
            let mut last_chi: f64 = 0.0;
            let mut last_acc: f64 = 0.5;

            let mut head_confidence = HeadConfidence::new(heads::NUM_HEADS);

            let mut nautilus_brain = NautilusBrain::new(
                NautilusBrainConfig::default(),
                &format!("hotspring-{lattice}"),
            );

            let mut sub_models = SubModelRegistry::default_models();
            let mut traj_batch: Vec<crate::production::TrajectoryEvent> = Vec::with_capacity(8);

            let make_multi_esn = |seed: u64, results: &[BetaResult]| -> Option<MultiHeadNpu> {
                if results.is_empty() {
                    return None;
                }
                let (seqs, tgts) = build_training_data(results, lattice);
                let mut esn = EchoStateNetwork::new(EsnConfig {
                    input_size: 6,
                    output_size: heads::NUM_HEADS,
                    regularization: 1e-3,
                    seed,
                    activation: Activation::ReluTanhApprox,
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
                                    let meta_acc = meta_context
                                        .iter()
                                        .find(|r| (r.beta - beta).abs() < 0.01)
                                        .map_or(0.5, |r| r.acceptance);
                                    let meta_mass = meta_context
                                        .iter()
                                        .find(|r| (r.beta - beta).abs() < 0.01)
                                        .map_or(0.1, |r| r.mass.unwrap_or(0.1));
                                    let seq = canonical_seq(
                                        beta, meta_plaq, meta_mass, meta_chi, meta_acc, lattice,
                                    );
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
                        let heuristic_params = |lat: usize| -> (f64, usize) {
                            let vol = (lat as f64).powi(4);
                            let scale = (4096.0 / vol).powf(0.25);
                            let dt = (0.05 * scale).max(0.001);
                            let n_md = ((0.5 / dt).round() as usize).max(20);
                            (dt, n_md)
                        };
                        let (dt, n_md) = if let Some(ref mut npu) = multi_npu {
                            let seq =
                                canonical_seq(beta, last_plaq, mass, last_chi, last_acc, lattice);
                            let raw = npu.predict_head(&seq, heads::PARAM_SUGGEST);
                            head_confidence
                                .record_prediction(heads::PARAM_SUGGEST, raw.clamp(0.0, 0.05));
                            if head_confidence.is_trusted(heads::PARAM_SUGGEST) {
                                let dt_suggest = raw.abs().mul_add(0.05, 0.001);
                                let n_md_suggest = ((0.5 / dt_suggest).round() as usize).max(10);
                                (dt_suggest, n_md_suggest)
                            } else {
                                heuristic_params(lattice)
                            }
                        } else {
                            heuristic_params(lattice)
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
                        let heuristic_cg = |lat: usize, m: f64| -> usize {
                            let vol = (lat as f64).powi(4);
                            (100.0 + vol.sqrt() / m.max(0.01)).round() as usize
                        };
                        let est = if let Some(ref mut npu) = multi_npu {
                            let seq =
                                canonical_seq(beta, last_plaq, mass, last_chi, last_acc, lattice);
                            let raw = npu.predict_head(&seq, heads::CG_ESTIMATE);
                            head_confidence
                                .record_prediction(heads::CG_ESTIMATE, raw.clamp(0.0, 5.0));
                            if head_confidence.is_trusted(heads::CG_ESTIMATE) {
                                (raw.abs() * 500.0).round() as usize
                            } else {
                                heuristic_cg(lattice, mass)
                            }
                        } else {
                            heuristic_cg(lattice, mass)
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
                            let meta_acc = meta_context
                                .iter()
                                .find(|r| (r.beta - beta).abs() < 0.01)
                                .map_or(0.5, |r| r.acceptance);
                            let meta_chi = meta_context
                                .iter()
                                .find(|r| (r.beta - beta).abs() < 0.01)
                                .map_or(0.0, |r| r.chi);
                            let input =
                                canonical_input(beta, meta_plaq, mass, meta_chi, meta_acc, lattice);
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

                    NpuRequest::QuenchedThermCheck {
                        plaq_window,
                        beta,
                        mass,
                    } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean =
                                plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
                            let chi = var * (plaq_window.len() as f64);
                            let input = canonical_input(beta, mean, mass, chi, last_acc, lattice);
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUENCHED_THERM);
                            raw > 0.7
                        } else {
                            check_thermalization(&plaq_window, beta)
                        };
                        resp_tx
                            .send(NpuResponse::QuenchedThermConverged(converged))
                            .ok();
                    }

                    NpuRequest::ThermCheck {
                        plaq_window,
                        beta,
                        mass,
                    } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean =
                                plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
                            let chi = var * (plaq_window.len() as f64);
                            let input = canonical_input(beta, mean, mass, chi, last_acc, lattice);
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::THERM_DETECT);
                            raw > 0.7
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
                        mass,
                    } => {
                        let (likely_rejected, confidence) = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_input(
                                beta,
                                plaquette,
                                mass,
                                last_chi,
                                acceptance_rate,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::REJECT_PREDICT);
                            let predicted_delta_h = raw * 10.0;
                            head_confidence
                                .record_prediction(heads::REJECT_PREDICT, predicted_delta_h);
                            if head_confidence.is_trusted(heads::REJECT_PREDICT) {
                                let rejected = predicted_delta_h > 0.0;
                                let conf = 1.0 / (1.0 + (-predicted_delta_h.abs()).exp());
                                (rejected, conf)
                            } else {
                                let (lr, conf) =
                                    predict_rejection(0.0, 0.0, 0.0, delta_h, acceptance_rate);
                                (lr, conf)
                            }
                        } else {
                            predict_rejection(0.0, 0.0, 0.0, delta_h, acceptance_rate)
                        };
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
                        polyakov: _,
                        susceptibility,
                        mass,
                        acceptance,
                    } => {
                        let label = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_input(
                                beta,
                                plaquette,
                                mass,
                                susceptibility,
                                acceptance,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::PHASE_CLASSIFY);
                            head_confidence
                                .record_prediction(heads::PHASE_CLASSIFY, raw.clamp(0.0, 1.0));
                            if head_confidence.is_trusted(heads::PHASE_CLASSIFY) {
                                match () {
                                    () if raw > 0.6 => "deconfined",
                                    () if raw > 0.3 => "transition",
                                    () => "confined",
                                }
                            } else {
                                heuristic_phase(beta)
                            }
                        } else {
                            heuristic_phase(beta)
                        };
                        resp_tx.send(NpuResponse::PhaseLabel(label)).ok();
                    }

                    NpuRequest::QualityScore { result } => {
                        last_plaq = result.mean_plaq;
                        last_chi = result.susceptibility;
                        last_acc = result.acceptance;
                        let actual_delta_h = -(result.acceptance - 0.5) * 4.0;
                        head_confidence.record_actual(heads::REJECT_PREDICT, actual_delta_h);
                        let actual_phase = result.polyakov.abs().clamp(0.0, 1.0);
                        head_confidence.record_actual(heads::PHASE_CLASSIFY, actual_phase);
                        head_confidence
                            .record_actual(heads::CG_ESTIMATE, result.mean_cg_iters / 100_000.0);
                        let actual_quality = if result.mean_plaq > 1e-9 {
                            1.0 - (result.std_plaq / result.mean_plaq).min(1.0)
                        } else {
                            0.0
                        };
                        head_confidence.record_actual(heads::QUALITY_SCORE, actual_quality);

                        let score = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_input(
                                result.beta,
                                result.mean_plaq,
                                result.mass,
                                result.susceptibility,
                                result.acceptance,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUALITY_SCORE).clamp(0.0, 1.0);
                            head_confidence.record_prediction(heads::QUALITY_SCORE, raw);
                            if head_confidence.is_trusted(heads::QUALITY_SCORE) {
                                raw
                            } else {
                                let acc_ok = if result.acceptance > 0.3 { 0.4 } else { 0.1 };
                                let stats_ok = if result.n_traj >= 100 { 0.3 } else { 0.1 };
                                let cg_ok = if result.mean_cg_iters < 1000.0 {
                                    0.3
                                } else {
                                    0.1
                                };
                                acc_ok + stats_ok + cg_ok
                            }
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
                        nautilus_brain.observe(BetaObservation {
                            beta: result.beta,
                            plaquette: result.mean_plaq,
                            cg_iters: result.mean_cg_iters,
                            acceptance: result.acceptance,
                            delta_h_abs: (1.0 - result.acceptance).abs(),
                            quenched_plaq: None,
                            quenched_plaq_var: None,
                            anderson_r: None,
                            anderson_lambda_min: None,
                        });

                        resp_tx.send(NpuResponse::Quality(score)).ok();
                    }

                    NpuRequest::AnomalyCheck {
                        beta,
                        plaq,
                        delta_h,
                        cg_iters,
                        acceptance,
                        mass,
                    } => {
                        let (is_anomaly, score) = if let Some(ref mut npu) = multi_npu {
                            let seq =
                                canonical_seq(beta, plaq, mass, last_chi, acceptance, lattice);
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
                        let all_known: Vec<f64> = measured_betas
                            .iter()
                            .chain(queued_betas.iter())
                            .copied()
                            .collect();
                        let n_measured = measured_betas.len();

                        let (suggestion, best_score) = if let Some(ref mut npu) = multi_npu {
                            let mut best_beta = None;
                            let mut top_score = f64::NEG_INFINITY;
                            let step = (beta_max - beta_min) / (n_candidates as f64 + 1.0);
                            let exclusion = step * 0.3_f64.max(0.025);
                            for ci in 1..=n_candidates {
                                let candidate = (ci as f64).mul_add(step, beta_min);
                                let too_close_measured = measured_betas
                                    .iter()
                                    .any(|&m| (m - candidate).abs() < exclusion);
                                let too_close_queued = queued_betas
                                    .iter()
                                    .any(|&q| (q - candidate).abs() < exclusion);
                                if too_close_measured || too_close_queued {
                                    continue;
                                }
                                let seq = canonical_seq(
                                    candidate, last_plaq, 0.1, last_chi, last_acc, lattice,
                                );
                                let all = npu.predict_all_heads(&seq);
                                let priority = all[heads::BETA_PRIORITY];
                                let uncertainty = (all[heads::QUALITY_SCORE] - 0.5).abs();
                                let score = priority + uncertainty * 0.5;
                                if score > top_score {
                                    top_score = score;
                                    best_beta = Some(candidate);
                                }
                            }
                            (best_beta, top_score)
                        } else if nautilus_brain.trained {
                            let edges = nautilus_brain.detect_concept_edges();
                            if let Some((edge_beta, edge_score)) = edges
                                .iter()
                                .filter(|(b, _)| *b >= beta_min && *b <= beta_max)
                                .max_by(|a, b| a.1.total_cmp(&b.1))
                            {
                                let too_close =
                                    all_known.iter().any(|&m| (m - edge_beta).abs() < 0.05);
                                if too_close {
                                    (
                                        find_largest_gaps(&all_known, beta_min, beta_max, 1)
                                            .into_iter()
                                            .next(),
                                        0.0,
                                    )
                                } else {
                                    (Some(*edge_beta), *edge_score)
                                }
                            } else {
                                (
                                    find_largest_gaps(&all_known, beta_min, beta_max, 1)
                                        .into_iter()
                                        .next(),
                                    0.0,
                                )
                            }
                        } else {
                            let gaps = find_largest_gaps(&all_known, beta_min, beta_max, 1);
                            (gaps.into_iter().next(), 0.0)
                        };

                        let range = beta_max - beta_min;
                        let largest_gap = {
                            let mut sorted = all_known.clone();
                            sorted.push(beta_min);
                            sorted.push(beta_max);
                            sorted.sort_by(f64::total_cmp);
                            sorted.dedup();
                            sorted
                                .windows(2)
                                .map(|w| w[1] - w[0])
                                .fold(0.0f64, f64::max)
                        };
                        let dense_enough = largest_gap < range * 0.05;
                        let low_novelty = best_score < 0.15 && best_score > f64::NEG_INFINITY;
                        let enough_points = n_measured >= 5;
                        let saturated = enough_points && (dense_enough || low_novelty);

                        if saturated {
                            eprintln!(
                                "  [NPU] Saturation detected: gap={largest_gap:.3}/{range:.1} \
                                score={best_score:.3} n={n_measured} — recommend moving on"
                            );
                        }

                        resp_tx
                            .send(NpuResponse::AdaptiveSteered {
                                suggestion,
                                saturated,
                            })
                            .ok();
                    }

                    NpuRequest::RecommendNextRun {
                        all_results,
                        meta_table: _,
                    } => {
                        let plan = if let Some(ref mut npu) = multi_npu {
                            let last = all_results.last();
                            let last_beta = last.map_or(5.69, |r| r.beta);
                            let seq = canonical_seq(
                                last_beta,
                                last.map_or(0.5, |r| r.mean_plaq),
                                last.map_or(0.1, |r| r.mass),
                                last.map_or(0.0, |r| r.susceptibility),
                                last.map_or(0.5, |r| r.acceptance),
                                lattice,
                            );
                            let raw = npu.predict_head(&seq, heads::NEXT_RUN_RECOMMEND);
                            let suggested_beta = raw.abs().mul_add(2.0, 5.0);
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
                        for r in &results {
                            let actual_delta_h = -(r.acceptance - 0.5) * 4.0;
                            head_confidence.record_actual(heads::REJECT_PREDICT, actual_delta_h);
                            head_confidence
                                .record_actual(heads::CG_ESTIMATE, r.mean_cg_iters / 100_000.0);
                            let actual_phase = r.polyakov.abs().clamp(0.0, 1.0);
                            head_confidence.record_actual(heads::PHASE_CLASSIFY, actual_phase);
                            let quality = if r.mean_plaq > 1e-9 {
                                1.0 - (r.std_plaq / r.mean_plaq).min(1.0)
                            } else {
                                0.0
                            };
                            head_confidence.record_actual(heads::QUALITY_SCORE, quality);
                        }
                        eprintln!("  [NPU] Head confidence: {}", head_confidence.status_line());

                        let seed = 99 + results.len() as u64;
                        if let Some(new_npu) = make_multi_esn(seed, &results) {
                            multi_npu = Some(new_npu);
                        }
                        if let Some(mse) = nautilus_brain.train() {
                            let n_obs = nautilus_brain.observations.len();
                            let drifting = nautilus_brain.is_drifting();
                            eprintln!(
                                "  [Nautilus] Trained: {n_obs} obs, MSE={mse:.6}, drift={drifting}"
                            );
                        }
                        eprintln!("  [Sub-models] {}", sub_models.status_line());

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
                        if let Some(weights) = load_esn_weights(&path) {
                            multi_npu = Some(MultiHeadNpu::from_exported(&weights));
                        } else {
                            eprintln!("  Warning: failed to load weights from {path}");
                        }
                        resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 }).ok();
                        let shell_path = nautilus_shell_path_from_weights(&path);
                        if let Some(brain) = load_nautilus_shell(&shell_path) {
                            let n_obs = brain.observations.len();
                            let n_gen = brain.shell.history.len();
                            nautilus_brain = brain;
                            eprintln!(
                                "  [Nautilus] Auto-loaded shell: {n_obs} obs, {n_gen} generations"
                            );
                        }
                    }

                    NpuRequest::ExportWeights { path } => {
                        let saved = if let Some(ref mut npu) = multi_npu {
                            save_esn_weights(npu, &path)
                        } else {
                            false
                        };
                        let shell_path = nautilus_shell_path_from_weights(&path);
                        save_nautilus_shell(&nautilus_brain, &shell_path);
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

                        let transition_proximity =
                            ((update.beta - KNOWN_BETA_C_SU3_NT4).abs() / 0.2).clamp(0.0, 1.0);
                        let regime_scale = 0.5f64.mul_add(1.0 - transition_proximity, 1.0);

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

                        let yellow_thresh = 0.3 * regime_scale;
                        let red_thresh = 0.7 * regime_scale;

                        match attention_state {
                            AttentionState::Green => {
                                if anomaly_score > red_thresh {
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
                                } else if anomaly_score > yellow_thresh {
                                    attention_state = AttentionState::Yellow;
                                    green_count = 0;
                                    if let Some(ref itx) = interrupt_tx {
                                        let _ = itx.send(BrainInterrupt::AdjustCheckInterval(20));
                                    }
                                }
                            }
                            AttentionState::Yellow => {
                                if anomaly_score > red_thresh {
                                    yellow_count += 1;
                                    if yellow_count >= 2 {
                                        attention_state = AttentionState::Red;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ =
                                                itx.send(BrainInterrupt::AdjustCheckInterval(5));
                                        }
                                    }
                                } else if anomaly_score < yellow_thresh {
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
                        bandwidth,
                        condition_number,
                        phase,
                        tier,
                        potts_magnetization,
                        potts_susceptibility,
                        potts_phase,
                    } => {
                        latest_proxy = Some(ProxyFeatures {
                            beta,
                            level_spacing_ratio,
                            lambda_min,
                            ipr,
                            bandwidth,
                            condition_number,
                            phase,
                            tier,
                            potts_magnetization,
                            potts_susceptibility,
                            potts_phase,
                            wall_ms: 0.0,
                        });

                        let spectral_obs = BetaObservation {
                            beta,
                            plaquette: bandwidth,
                            cg_iters: condition_number,
                            acceptance: 1.0,
                            delta_h_abs: 0.0,
                            quenched_plaq: None,
                            quenched_plaq_var: None,
                            anderson_r: Some(level_spacing_ratio),
                            anderson_lambda_min: Some(lambda_min),
                        };
                        nautilus_brain.observe(spectral_obs);
                    }

                    NpuRequest::DisagreementQuery {
                        beta,
                        plaq,
                        mass,
                        chi,
                        acceptance,
                    } => {
                        let dis = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_seq(beta, plaq, mass, chi, acceptance, lattice);
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

                    NpuRequest::ExportNautilusShell { path } => {
                        let saved = save_nautilus_shell(&nautilus_brain, &path);
                        resp_tx
                            .send(NpuResponse::NautilusShellSaved {
                                path: if saved { path } else { String::new() },
                            })
                            .ok();
                    }

                    NpuRequest::BootstrapNautilusShell { path } => {
                        let (n_obs, n_gen) = if let Some(brain) = load_nautilus_shell(&path) {
                            let n_obs = brain.observations.len();
                            let n_gen = brain.shell.history.len();
                            nautilus_brain = brain;
                            eprintln!(
                                "  [Nautilus] Loaded shell: {n_obs} obs, {n_gen} generations"
                            );
                            (n_obs, n_gen)
                        } else {
                            eprintln!("  [Nautilus] Failed to load shell from {path}");
                            (0, 0)
                        };
                        resp_tx
                            .send(NpuResponse::NautilusShellLoaded {
                                n_observations: n_obs,
                                n_generations: n_gen,
                            })
                            .ok();
                    }

                    NpuRequest::TrajectoryEvent(evt) => {
                        traj_batch.push(evt);
                        if traj_batch.len() >= 8 {
                            let n = traj_batch.len();
                            for evt in std::mem::take(&mut traj_batch) {
                                sub_models.observe_event(&evt, latest_proxy.as_ref());
                            }
                            resp_tx
                                .send(NpuResponse::TrajectoryBatchProcessed { n_events: n })
                                .ok();
                        }
                    }

                    NpuRequest::FlushTrajectoryBatch => {
                        if traj_batch.is_empty() {
                            resp_tx
                                .send(NpuResponse::TrajectoryBatchProcessed { n_events: 0 })
                                .ok();
                        } else {
                            let n = traj_batch.len();
                            for evt in std::mem::take(&mut traj_batch) {
                                sub_models.observe_event(&evt, latest_proxy.as_ref());
                            }
                            eprintln!(
                                "  [NPU] Sub-models: flushed {n} events — {}",
                                sub_models.status_line()
                            );
                            resp_tx
                                .send(NpuResponse::TrajectoryBatchProcessed { n_events: n })
                                .ok();
                        }
                    }

                    NpuRequest::SubModelMetrics => {
                        resp_tx
                            .send(NpuResponse::SubModelMetricsSnapshot(
                                sub_models.metrics_json(),
                            ))
                            .ok();
                    }

                    NpuRequest::SubModelPredict(evt) => {
                        let predictions = sub_models.predict_all(&evt, latest_proxy.as_ref());
                        let mut cg_cost = None;
                        let mut steering = None;
                        let mut phase = None;
                        let mut trajectory = None;
                        for (name, pred) in predictions {
                            match name {
                                "cg_cost_predictor" => cg_cost = pred,
                                "steering_brain" => steering = pred,
                                "phase_oracle" => phase = pred,
                                "trajectory_predictor" => trajectory = pred,
                                _ => {}
                            }
                        }
                        resp_tx
                            .send(NpuResponse::SubModelPredictions {
                                cg_cost,
                                steering,
                                phase,
                                trajectory,
                            })
                            .ok();
                    }

                    NpuRequest::Shutdown => break,
                }
            }
        })?;

    Ok(NpuWorkerHandles {
        npu_tx: req_tx,
        npu_rx: resp_rx,
        interrupt_rx: interrupt_rx_out,
    })
}
