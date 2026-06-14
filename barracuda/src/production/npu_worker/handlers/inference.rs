// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::md::reservoir::heads;
use crate::production::trajectory_input::{canonical_input, canonical_seq, heuristic_phase};
use crate::production::{BetaResult, predict_rejection};
use barracuda::nautilus::BetaObservation;
use std::sync::mpsc;

use super::super::messages::NpuResponse;
use super::super::training::find_largest_gaps;
use super::super::worker_state::WorkerState;

pub(super) fn handle_reject_predict(
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

pub(super) fn handle_phase_classify(
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

pub(super) fn handle_quality_score(
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

pub(super) fn handle_anomaly_check(
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

pub(super) fn handle_steer_adaptive(
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

pub(super) fn handle_recommend_next_run(
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
