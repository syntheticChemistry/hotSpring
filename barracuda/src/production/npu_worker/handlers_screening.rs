// SPDX-License-Identifier: AGPL-3.0-or-later

//! NPU handlers for screening and parameter prediction.
//!
//! Covers: beta pre-screening, parameter suggestion, CG estimation,
//! quenched thermalization length, and thermalization convergence.

use std::sync::mpsc;

use crate::md::reservoir::heads;
use crate::production::trajectory_input::{canonical_input, canonical_seq};
use crate::production::{MetaRow, check_thermalization, plaquette_variance};
use crate::provenance::KNOWN_BETA_C_SU3_NT4;

use super::messages::NpuResponse;
use super::worker_state::WorkerState;

pub(super) fn handle_prescreen_beta(
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

pub(super) fn handle_suggest_params(
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

pub(super) fn handle_predict_cg(
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

pub(super) fn handle_predict_quenched_length(
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
