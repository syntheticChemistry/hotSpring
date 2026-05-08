// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::production::trajectory_input::canonical_seq;
use crate::production::TrajectoryEvent;
use barracuda::nautilus::BetaObservation;
use std::sync::mpsc;

use super::super::messages::NpuResponse;
use super::super::worker_state::WorkerState;

pub(super) fn handle_proxy_features(
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

pub(super) fn handle_disagreement(
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

pub(super) fn handle_trajectory_event(
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

pub(super) fn handle_flush_trajectory_batch(state: &mut WorkerState, resp_tx: &mpsc::Sender<NpuResponse>) {
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

pub(super) fn handle_sub_model_metrics(state: &WorkerState, resp_tx: &mpsc::Sender<NpuResponse>) {
    let _ = resp_tx.send(NpuResponse::SubModelMetricsSnapshot(
        state.sub_models.metrics_json(),
    ));
}

pub(super) fn handle_sub_model_predict(
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
