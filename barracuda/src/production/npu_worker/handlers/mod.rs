// SPDX-License-Identifier: AGPL-3.0-or-later

//! NPU request handlers — pre/during/post computation, lifecycle, and sub-models.

mod inference;
mod precompute;
mod proxy;
mod thermalization;

use inference::{
    handle_anomaly_check, handle_phase_classify, handle_quality_score, handle_recommend_next_run,
    handle_reject_predict, handle_steer_adaptive,
};
use precompute::{
    handle_predict_cg, handle_predict_quenched_length, handle_prescreen_beta, handle_suggest_params,
};
use proxy::{
    handle_disagreement, handle_flush_trajectory_batch, handle_proxy_features,
    handle_sub_model_metrics, handle_sub_model_predict, handle_trajectory_event,
};
use thermalization::{handle_quenched_therm, handle_therm};

use std::sync::mpsc;

use super::messages::{NpuRequest, NpuResponse};
use super::worker_state::WorkerState;

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
