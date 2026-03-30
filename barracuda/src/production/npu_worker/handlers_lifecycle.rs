// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU lifecycle handlers: retrain, bootstrap, export.

use crate::md::reservoir::{MultiHeadNpu, heads};
use crate::production::checkpoint::{
    load_esn_weights, load_nautilus_shell, nautilus_shell_path_from_weights, save_esn_weights,
    save_nautilus_shell,
};
use crate::production::{BetaResult, MetaRow};
use std::sync::mpsc;

use super::messages::NpuResponse;
use super::training::estimate_beta_c;
use super::worker_state::WorkerState;

/// Retrain ESN and Nautilus from accumulated results.
pub(super) fn handle_retrain(
    state: &mut WorkerState,
    results: &[BetaResult],
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    for r in results {
        let actual_delta_h = -(r.acceptance - 0.5) * 4.0;
        state
            .head_confidence
            .record_actual(heads::REJECT_PREDICT, actual_delta_h);
        state
            .head_confidence
            .record_actual(heads::CG_ESTIMATE, r.mean_cg_iters / 100_000.0);
        let actual_phase = r.polyakov.abs().clamp(0.0, 1.0);
        state
            .head_confidence
            .record_actual(heads::PHASE_CLASSIFY, actual_phase);
        let quality = if r.mean_plaq > 1e-9 {
            1.0 - (r.std_plaq / r.mean_plaq).min(1.0)
        } else {
            0.0
        };
        state
            .head_confidence
            .record_actual(heads::QUALITY_SCORE, quality);
    }
    eprintln!(
        "  [NPU] Head confidence: {}",
        state.head_confidence.status_line()
    );

    let seed = 99 + results.len() as u64;
    if let Some(new_npu) = state.make_multi_esn(seed, results) {
        state.multi_npu = Some(new_npu);
    }
    if let Some(mse) = state.nautilus_brain.train() {
        let n_obs = state.nautilus_brain.observations.len();
        let drifting = state.nautilus_brain.is_drifting();
        eprintln!("  [Nautilus] Trained: {n_obs} obs, MSE={mse:.6}, drift={drifting}");
    }
    eprintln!("  [Sub-models] {}", state.sub_models.status_line());

    let beta_c = estimate_beta_c(results);
    let _ = resp_tx.send(NpuResponse::Retrained { beta_c });
}

/// Bootstrap ESN from meta table rows.
pub(super) fn handle_bootstrap_meta(
    state: &mut WorkerState,
    rows: Vec<MetaRow>,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
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
    if let Some(new_npu) = state.make_multi_esn(42, &results) {
        state.multi_npu = Some(new_npu);
    }
    let _ = resp_tx.send(NpuResponse::Bootstrapped { n_points: n });
}

/// Bootstrap ESN from saved weights; auto-load Nautilus shell if present.
pub(super) fn handle_bootstrap_weights(
    state: &mut WorkerState,
    path: &str,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    if let Some(weights) = load_esn_weights(path) {
        state.multi_npu = Some(MultiHeadNpu::from_exported(&weights));
    } else {
        eprintln!("  Warning: failed to load weights from {path}");
    }
    let _ = resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 });
    let shell_path = nautilus_shell_path_from_weights(path);
    if let Some(brain) = load_nautilus_shell(&shell_path) {
        let n_obs = brain.observations.len();
        let n_gen = brain.shell.history.len();
        state.nautilus_brain = brain;
        eprintln!("  [Nautilus] Auto-loaded shell: {n_obs} obs, {n_gen} generations");
    }
}

/// Export ESN weights and Nautilus shell.
pub(super) fn handle_export_weights(
    state: &mut WorkerState,
    path: &str,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let saved = if let Some(ref mut npu) = state.multi_npu {
        save_esn_weights(npu, path)
    } else {
        false
    };
    let shell_path = nautilus_shell_path_from_weights(path);
    save_nautilus_shell(&state.nautilus_brain, &shell_path);
    let _ = resp_tx.send(NpuResponse::WeightsSaved {
        path: if saved {
            path.to_string()
        } else {
            String::new()
        },
    });
}

/// Export Nautilus shell to path.
pub(super) fn handle_export_nautilus_shell(
    state: &WorkerState,
    path: &str,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let saved = save_nautilus_shell(&state.nautilus_brain, path);
    let _ = resp_tx.send(NpuResponse::NautilusShellSaved {
        path: if saved {
            path.to_string()
        } else {
            String::new()
        },
    });
}

/// Bootstrap Nautilus shell from path.
pub(super) fn handle_bootstrap_nautilus_shell(
    state: &mut WorkerState,
    path: &str,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    let (n_obs, n_gen) = if let Some(brain) = load_nautilus_shell(path) {
        let n_obs = brain.observations.len();
        let n_gen = brain.shell.history.len();
        state.nautilus_brain = brain;
        eprintln!("  [Nautilus] Loaded shell: {n_obs} obs, {n_gen} generations");
        (n_obs, n_gen)
    } else {
        eprintln!("  [Nautilus] Failed to load shell from {path}");
        (0, 0)
    };
    let _ = resp_tx.send(NpuResponse::NautilusShellLoaded {
        n_observations: n_obs,
        n_generations: n_gen,
    });
}
