// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU request handlers — pre/during/post computation, lifecycle, and sub-models.

use crate::md::reservoir::heads;
use crate::production::trajectory_input::{canonical_input, canonical_seq, heuristic_phase};
use crate::production::{check_thermalization, plaquette_variance, predict_rejection, BetaResult};
use crate::production::{MetaRow, TrajectoryEvent};
use barracuda::nautilus::BetaObservation;
use std::sync::mpsc;

use super::messages::{NpuRequest, NpuResponse};
use super::training::find_largest_gaps;
use super::worker_state::WorkerState;
use crate::provenance::KNOWN_BETA_C_SU3_NT4;

/// Dispatch a request to the appropriate handler.
pub(super) fn handle(
    req: NpuRequest,
    state: &mut WorkerState,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    match req {
        NpuRequest::PreScreenBeta {
            candidates,
            meta_context,
        } => handle_prescreen_beta(state, &candidates, &meta_context, resp_tx),
        NpuRequest::SuggestParameters {
            lattice: lat,
            beta,
            mass,
        } => {
            handle_suggest_params(state, lat, beta, mass, resp_tx);
        }
        NpuRequest::PredictCgIters {
            lattice: lat,
            beta,
            mass,
        } => {
            handle_predict_cg(state, lat, beta, mass, resp_tx);
        }
        NpuRequest::PredictQuenchedLength {
            beta,
            mass,
            lattice: lat,
            meta_context,
        } => handle_predict_quenched_length(state, lat, beta, mass, &meta_context, resp_tx),
        NpuRequest::QuenchedThermCheck {
            plaq_window,
            beta,
            mass,
        } => handle_quenched_therm(state, &plaq_window, beta, mass, resp_tx),
        NpuRequest::ThermCheck {
            plaq_window,
            beta,
            mass,
        } => handle_therm(state, &plaq_window, beta, mass, resp_tx),
        NpuRequest::RejectPredict {
            beta,
            plaquette,
            delta_h,
            acceptance_rate,
            mass,
        } => handle_reject_predict(
            state,
            beta,
            plaquette,
            delta_h,
            acceptance_rate,
            mass,
            resp_tx,
        ),
        NpuRequest::PhaseClassify {
            beta,
            plaquette,
            polyakov: _,
            susceptibility,
            mass,
            acceptance,
        } => handle_phase_classify(
            state,
            beta,
            plaquette,
            susceptibility,
            mass,
            acceptance,
            resp_tx,
        ),
        NpuRequest::QualityScore { result } => handle_quality_score(state, result, resp_tx),
        NpuRequest::AnomalyCheck {
            beta,
            plaq,
            delta_h,
            cg_iters,
            acceptance,
            mass,
        } => handle_anomaly_check(
            state, beta, plaq, delta_h, cg_iters, acceptance, mass, resp_tx,
        ),
        NpuRequest::SteerAdaptive {
            measured_betas,
            queued_betas,
            beta_min,
            beta_max,
            n_candidates,
        } => handle_steer_adaptive(
            state,
            &measured_betas,
            &queued_betas,
            beta_min,
            beta_max,
            n_candidates,
            resp_tx,
        ),
        NpuRequest::RecommendNextRun {
            all_results,
            meta_table: _,
        } => handle_recommend_next_run(state, &all_results, resp_tx),
        NpuRequest::Retrain { results } => {
            super::handlers_lifecycle::handle_retrain(state, &results, resp_tx);
        }
        NpuRequest::BootstrapFromMeta { rows } => {
            super::handlers_lifecycle::handle_bootstrap_meta(state, rows, resp_tx);
        }
        NpuRequest::BootstrapFromWeights { path } => {
            super::handlers_lifecycle::handle_bootstrap_weights(state, &path, resp_tx);
        }
        NpuRequest::ExportWeights { path } => {
            super::handlers_lifecycle::handle_export_weights(state, &path, resp_tx);
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
        } => handle_proxy_features(
            state,
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
            resp_tx,
        ),
        NpuRequest::DisagreementQuery {
            beta,
            plaq,
            mass,
            chi,
            acceptance,
        } => handle_disagreement(state, beta, plaq, mass, chi, acceptance, resp_tx),
        NpuRequest::ExportNautilusShell { path } => {
            super::handlers_lifecycle::handle_export_nautilus_shell(state, &path, resp_tx);
        }
        NpuRequest::BootstrapNautilusShell { path } => {
            super::handlers_lifecycle::handle_bootstrap_nautilus_shell(state, &path, resp_tx);
        }
        NpuRequest::TrajectoryEvent(evt) => handle_trajectory_event(state, evt, resp_tx),
        NpuRequest::FlushTrajectoryBatch => handle_flush_trajectory_batch(state, resp_tx),
        NpuRequest::SubModelMetrics => handle_sub_model_metrics(state, resp_tx),
        NpuRequest::SubModelPredict(evt) => handle_sub_model_predict(state, &evt, resp_tx),
        NpuRequest::Shutdown | NpuRequest::CgResidual(_) => {
            // Handled in mod.rs (Shutdown breaks loop; CgResidual dispatched to cg_residual)
        }
    }
}

fn handle_prescreen_beta(
    state: &mut WorkerState,
    candidates: &[f64],
    meta_context: &[MetaRow],
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let lattice = state.lattice;
    let priorities: Vec<(f64, f64)> = candidates
        .iter()
        .map(|&beta| {
            let score = if let Some(ref mut npu) = state.multi_npu {
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
                let seq = canonical_seq(beta, meta_plaq, meta_mass, meta_chi, meta_acc, lattice);
                npu.predict_head(&seq, heads::BETA_PRIORITY)
            } else {
                (-(beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.3).exp()
            };
            (beta, score)
        })
        .collect();
    let _ = resp_tx.send(NpuResponse::BetaPriorities(priorities));
}

fn handle_suggest_params(
    state: &mut WorkerState,
    _lat: usize,
    beta: f64,
    mass: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let heuristic_params = |l: usize| -> (f64, usize) {
        let vol = (l as f64).powi(4);
        let scale = (4096.0 / vol).powf(0.25);
        let dt = (0.05 * scale).max(0.001);
        let n_md = ((0.5 / dt).round() as usize).max(20);
        (dt, n_md)
    };
    let (dt, n_md) = if let Some(ref mut npu) = state.multi_npu {
        let seq = canonical_seq(
            beta,
            state.last_plaq,
            mass,
            state.last_chi,
            state.last_acc,
            state.lattice,
        );
        let raw = npu.predict_head(&seq, heads::PARAM_SUGGEST);
        state
            .head_confidence
            .record_prediction(heads::PARAM_SUGGEST, raw.clamp(0.0, 0.05));
        if state.head_confidence.is_trusted(heads::PARAM_SUGGEST) {
            let dt_suggest = raw.abs().mul_add(0.05, 0.001);
            let n_md_suggest = ((0.5 / dt_suggest).round() as usize).max(10);
            (dt_suggest, n_md_suggest)
        } else {
            heuristic_params(state.lattice)
        }
    } else {
        heuristic_params(state.lattice)
    };
    let _ = resp_tx.send(NpuResponse::ParameterSuggestion { dt, n_md });
}

fn handle_predict_cg(
    state: &mut WorkerState,
    lat: usize,
    beta: f64,
    mass: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let heuristic_cg = |l: usize, m: f64| -> usize {
        let vol = (l as f64).powi(4);
        (100.0 + vol.sqrt() / m.max(0.01)).round() as usize
    };
    let est = if let Some(ref mut npu) = state.multi_npu {
        let seq = canonical_seq(
            beta,
            state.last_plaq,
            mass,
            state.last_chi,
            state.last_acc,
            state.lattice,
        );
        let raw = npu.predict_head(&seq, heads::CG_ESTIMATE);
        state
            .head_confidence
            .record_prediction(heads::CG_ESTIMATE, raw.clamp(0.0, 5.0));
        if state.head_confidence.is_trusted(heads::CG_ESTIMATE) {
            (raw.abs() * 500.0).round() as usize
        } else {
            heuristic_cg(lat, mass)
        }
    } else {
        heuristic_cg(lat, mass)
    };
    let _ = resp_tx.send(NpuResponse::CgEstimate(est));
}

fn handle_predict_quenched_length(
    state: &mut WorkerState,
    lat: usize,
    beta: f64,
    mass: f64,
    meta_context: &[MetaRow],
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let steps = if let Some(ref mut npu) = state.multi_npu {
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
        let input = canonical_input(beta, meta_plaq, mass, meta_chi, meta_acc, lat);
        let seq = vec![input; 10];
        let raw = npu.predict_head(&seq, heads::QUENCHED_LENGTH);
        (raw.abs() * 200.0).round().clamp(5.0, 200.0) as usize
    } else {
        let proximity = (-(beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.3).exp();
        let base = 20 + (80.0 * proximity) as usize;
        base.min(lat * 10)
    };
    let _ = resp_tx.send(NpuResponse::QuenchedLengthEstimate(steps));
}

fn handle_quenched_therm(
    state: &mut WorkerState,
    plaq_window: &[f64],
    beta: f64,
    mass: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let converged = if let Some(ref mut npu) = state.multi_npu {
        let var = plaquette_variance(plaq_window);
        let mean = plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
        let chi = var * (plaq_window.len() as f64);
        let input = canonical_input(beta, mean, mass, chi, state.last_acc, state.lattice);
        let seq = vec![input; 10];
        let raw = npu.predict_head(&seq, heads::QUENCHED_THERM);
        raw > 0.7
    } else {
        check_thermalization(plaq_window, beta)
    };
    let _ = resp_tx.send(NpuResponse::QuenchedThermConverged(converged));
}

fn handle_therm(
    state: &mut WorkerState,
    plaq_window: &[f64],
    beta: f64,
    mass: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let converged = if let Some(ref mut npu) = state.multi_npu {
        let var = plaquette_variance(plaq_window);
        let mean = plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
        let chi = var * (plaq_window.len() as f64);
        let input = canonical_input(beta, mean, mass, chi, state.last_acc, state.lattice);
        let seq = vec![input; 10];
        let raw = npu.predict_head(&seq, heads::THERM_DETECT);
        raw > 0.7
    } else {
        check_thermalization(plaq_window, beta)
    };
    let _ = resp_tx.send(NpuResponse::ThermConverged(converged));
}

fn handle_reject_predict(
    state: &mut WorkerState,
    beta: f64,
    plaquette: f64,
    delta_h: f64,
    acceptance_rate: f64,
    mass: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let (likely_rejected, confidence) = if let Some(ref mut npu) = state.multi_npu {
        let input = canonical_input(
            beta,
            plaquette,
            mass,
            state.last_chi,
            acceptance_rate,
            state.lattice,
        );
        let seq = vec![input; 10];
        let raw = npu.predict_head(&seq, heads::REJECT_PREDICT);
        let predicted_delta_h = raw * 10.0;
        state
            .head_confidence
            .record_prediction(heads::REJECT_PREDICT, predicted_delta_h);
        if state.head_confidence.is_trusted(heads::REJECT_PREDICT) {
            let rejected = predicted_delta_h > 0.0;
            let conf = 1.0 / (1.0 + (-predicted_delta_h.abs()).exp());
            (rejected, conf)
        } else {
            predict_rejection(0.0, 0.0, 0.0, delta_h, acceptance_rate)
        }
    } else {
        predict_rejection(0.0, 0.0, 0.0, delta_h, acceptance_rate)
    };
    let _ = resp_tx.send(NpuResponse::RejectPrediction {
        likely_rejected,
        confidence,
    });
}

fn handle_phase_classify(
    state: &mut WorkerState,
    beta: f64,
    plaquette: f64,
    susceptibility: f64,
    mass: f64,
    acceptance: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let label = if let Some(ref mut npu) = state.multi_npu {
        let input = canonical_input(
            beta,
            plaquette,
            mass,
            susceptibility,
            acceptance,
            state.lattice,
        );
        let seq = vec![input; 10];
        let raw = npu.predict_head(&seq, heads::PHASE_CLASSIFY);
        state
            .head_confidence
            .record_prediction(heads::PHASE_CLASSIFY, raw.clamp(0.0, 1.0));
        if state.head_confidence.is_trusted(heads::PHASE_CLASSIFY) {
            if raw > 0.6 {
                "deconfined"
            } else if raw > 0.3 {
                "transition"
            } else {
                "confined"
            }
        } else {
            heuristic_phase(beta)
        }
    } else {
        heuristic_phase(beta)
    };
    let _ = resp_tx.send(NpuResponse::PhaseLabel(label));
}

fn handle_quality_score(
    state: &mut WorkerState,
    result: BetaResult,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    state.last_plaq = result.mean_plaq;
    state.last_chi = result.susceptibility;
    state.last_acc = result.acceptance;
    let actual_delta_h = -(result.acceptance - 0.5) * 4.0;
    state
        .head_confidence
        .record_actual(heads::REJECT_PREDICT, actual_delta_h);
    let actual_phase = result.polyakov.abs().clamp(0.0, 1.0);
    state
        .head_confidence
        .record_actual(heads::PHASE_CLASSIFY, actual_phase);
    state
        .head_confidence
        .record_actual(heads::CG_ESTIMATE, result.mean_cg_iters / 100_000.0);
    let actual_quality = if result.mean_plaq > 1e-9 {
        1.0 - (result.std_plaq / result.mean_plaq).min(1.0)
    } else {
        0.0
    };
    state
        .head_confidence
        .record_actual(heads::QUALITY_SCORE, actual_quality);

    let score = if let Some(ref mut npu) = state.multi_npu {
        let input = canonical_input(
            result.beta,
            result.mean_plaq,
            result.mass,
            result.susceptibility,
            result.acceptance,
            state.lattice,
        );
        let seq = vec![input; 10];
        let raw = npu.predict_head(&seq, heads::QUALITY_SCORE).clamp(0.0, 1.0);
        state
            .head_confidence
            .record_prediction(heads::QUALITY_SCORE, raw);
        if state.head_confidence.is_trusted(heads::QUALITY_SCORE) {
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
    state.nautilus_brain.observe(BetaObservation {
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

    let _ = resp_tx.send(NpuResponse::Quality(score));
}

fn handle_anomaly_check(
    state: &mut WorkerState,
    beta: f64,
    plaq: f64,
    delta_h: f64,
    cg_iters: usize,
    acceptance: f64,
    mass: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let (is_anomaly, score) = if let Some(ref mut npu) = state.multi_npu {
        let seq = canonical_seq(beta, plaq, mass, state.last_chi, acceptance, state.lattice);
        let raw = npu.predict_head(&seq, heads::ANOMALY_DETECT);
        (raw > 0.7, raw.clamp(0.0, 1.0))
    } else {
        let anomaly = delta_h.abs() > 50.0 || cg_iters > 4000 || !(0.0..=1.0).contains(&plaq);
        let s = if anomaly { 0.9 } else { 0.1 };
        (anomaly, s)
    };
    let _ = resp_tx.send(NpuResponse::AnomalyFlag {
        is_anomaly,
        _score: score,
    });
}

fn handle_steer_adaptive(
    state: &mut WorkerState,
    measured_betas: &[f64],
    queued_betas: &[f64],
    beta_min: f64,
    beta_max: f64,
    n_candidates: usize,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let all_known: Vec<f64> = measured_betas
        .iter()
        .chain(queued_betas.iter())
        .copied()
        .collect();
    let n_measured = measured_betas.len();

    let (suggestion, best_score) = if let Some(ref mut npu) = state.multi_npu {
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
                candidate,
                state.last_plaq,
                0.1,
                state.last_chi,
                state.last_acc,
                state.lattice,
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
    } else if state.nautilus_brain.trained {
        let edges = state.nautilus_brain.detect_concept_edges();
        if let Some((edge_beta, edge_score)) = edges
            .iter()
            .filter(|(b, _)| *b >= beta_min && *b <= beta_max)
            .max_by(|a, b| a.1.total_cmp(&b.1))
        {
            let too_close = all_known.iter().any(|&m| (m - edge_beta).abs() < 0.05);
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
        let mut sorted: Vec<f64> = all_known
            .iter()
            .copied()
            .chain([beta_min, beta_max])
            .collect();
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

    let _ = resp_tx.send(NpuResponse::AdaptiveSteered {
        suggestion,
        saturated,
    });
}

fn handle_recommend_next_run(
    state: &mut WorkerState,
    all_results: &[BetaResult],
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let plan = if let Some(ref mut npu) = state.multi_npu {
        let last = all_results.last();
        let last_beta = last.map_or(5.69, |r| r.beta);
        let seq = canonical_seq(
            last_beta,
            last.map_or(0.5, |r| r.mean_plaq),
            last.map_or(0.1, |r| r.mass),
            last.map_or(0.0, |r| r.susceptibility),
            last.map_or(0.5, |r| r.acceptance),
            state.lattice,
        );
        let raw = npu.predict_head(&seq, heads::NEXT_RUN_RECOMMEND);
        let suggested_beta = raw.abs().mul_add(2.0, 5.0);
        let mass = last.map_or(0.1, |r| r.mass);
        (vec![suggested_beta], mass, 8)
    } else {
        let measured: Vec<f64> = all_results.iter().map(|r| r.beta).collect();
        let gaps = find_largest_gaps(&measured, 5.0, 7.0, 3);
        let mass = all_results.first().map_or(0.1, |r| r.mass);
        (gaps, mass, 8)
    };
    let _ = resp_tx.send(NpuResponse::NextRunPlan {
        betas: plan.0,
        mass: plan.1,
        lattice: plan.2,
    });
}

#[allow(clippy::too_many_arguments)] // mirrors the ProxyFeatures struct fields directly
fn handle_proxy_features(
    state: &mut WorkerState,
    beta: f64,
    level_spacing_ratio: f64,
    lambda_min: f64,
    ipr: f64,
    bandwidth: f64,
    condition_number: f64,
    phase: String,
    tier: u8,
    potts_magnetization: f64,
    potts_susceptibility: f64,
    potts_phase: String,
    _resp_tx: &mpsc::Sender<NpuResponse>,
) {
    state.latest_proxy = Some(crate::proxy::ProxyFeatures {
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
    state.nautilus_brain.observe(spectral_obs);
}

fn handle_disagreement(
    state: &mut WorkerState,
    beta: f64,
    plaq: f64,
    mass: f64,
    chi: f64,
    acceptance: f64,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    use crate::md::reservoir::HeadGroupDisagreement;

    let dis = if let Some(ref mut npu) = state.multi_npu {
        let input = canonical_seq(beta, plaq, mass, chi, acceptance, state.lattice);
        let (outputs, d) = npu.predict_with_disagreement(&input);
        let _ = outputs;
        d
    } else {
        HeadGroupDisagreement::default()
    };
    let _ = resp_tx.send(NpuResponse::DisagreementSnapshot {
        delta_cg: dis.delta_cg,
        delta_phase: dis.delta_phase,
        delta_anomaly: dis.delta_anomaly,
        delta_priority: dis.delta_priority,
        urgency: dis.urgency(),
    });
}

fn handle_trajectory_event(
    state: &mut WorkerState,
    evt: TrajectoryEvent,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    state.traj_batch.push(evt);
    if state.traj_batch.len() >= 8 {
        let n = state.traj_batch.len();
        for evt in std::mem::take(&mut state.traj_batch) {
            state
                .sub_models
                .observe_event(&evt, state.latest_proxy.as_ref());
        }
        let _ = resp_tx.send(NpuResponse::TrajectoryBatchProcessed { n_events: n });
    }
}

fn handle_flush_trajectory_batch(state: &mut WorkerState, resp_tx: &mpsc::Sender<NpuResponse>) {
    if state.traj_batch.is_empty() {
        let _ = resp_tx.send(NpuResponse::TrajectoryBatchProcessed { n_events: 0 });
    } else {
        let n = state.traj_batch.len();
        for evt in std::mem::take(&mut state.traj_batch) {
            state
                .sub_models
                .observe_event(&evt, state.latest_proxy.as_ref());
        }
        eprintln!(
            "  [NPU] Sub-models: flushed {n} events — {}",
            state.sub_models.status_line()
        );
        let _ = resp_tx.send(NpuResponse::TrajectoryBatchProcessed { n_events: n });
    }
}

fn handle_sub_model_metrics(state: &WorkerState, resp_tx: &mpsc::Sender<NpuResponse>) {
    let _ = resp_tx.send(NpuResponse::SubModelMetricsSnapshot(
        state.sub_models.metrics_json(),
    ));
}

fn handle_sub_model_predict(
    state: &mut WorkerState,
    evt: &TrajectoryEvent,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let predictions = state
        .sub_models
        .predict_all(evt, state.latest_proxy.as_ref());
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
    let _ = resp_tx.send(NpuResponse::SubModelPredictions {
        cg_cost,
        steering,
        phase,
        trajectory,
    });
}
