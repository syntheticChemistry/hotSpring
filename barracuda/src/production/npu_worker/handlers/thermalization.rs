// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::md::reservoir::heads;
use crate::production::trajectory_input::canonical_input;
use crate::production::{check_thermalization, plaquette_variance};
use std::sync::mpsc;

use super::super::messages::NpuResponse;
use super::super::worker_state::WorkerState;

pub(super) fn handle_quenched_therm(
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

pub(super) fn handle_therm(
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
