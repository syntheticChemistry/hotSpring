// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! Brain Layer 1: CG residual monitoring and attention state machine.
//!
//! Monitors CG convergence in real time, transitions Green→Yellow→Red based on
//! anomaly scores, and sends BrainInterrupt signals (KillCg, AdjustCheckInterval).

use super::worker_state::WorkerState;
use crate::lattice::gpu_hmc::{BrainInterrupt, CgResidualUpdate};
use crate::md::reservoir::heads;
use crate::production::AttentionState;
use crate::provenance::KNOWN_BETA_C_SU3_NT4;
use std::sync::mpsc;

use super::messages::NpuResponse;

/// Process a CG residual update: update attention state and optionally send interrupts.
pub(super) fn handle_cg_residual(
    update: CgResidualUpdate,
    state: &mut WorkerState,
    resp_tx: &mpsc::Sender<NpuResponse>,
) {
    state
        .residual_history
        .push((update.iteration, update.rz_new));
    if state.residual_history.len() > 50 {
        state
            .residual_history
            .drain(..state.residual_history.len() - 50);
    }

    let transition_proximity = ((update.beta - KNOWN_BETA_C_SU3_NT4).abs() / 0.2).clamp(0.0, 1.0);
    let regime_scale = 0.5f64.mul_add(1.0 - transition_proximity, 1.0);

    let anomaly_score = if let Some(ref mut npu) = state.multi_npu {
        let window: Vec<f64> = state
            .residual_history
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
    } else if state.residual_history.len() >= 3 {
        let recent: Vec<f64> = state
            .residual_history
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

    match state.attention_state {
        AttentionState::Green => {
            if anomaly_score > red_thresh {
                state.attention_state = AttentionState::Red;
                state.yellow_count = 0;
                state.green_count = 0;
                if state.residual_history.len() >= 3 {
                    let recent: Vec<f64> = state
                        .residual_history
                        .iter()
                        .rev()
                        .take(3)
                        .map(|(_, rz)| *rz)
                        .collect();
                    if recent.windows(2).all(|w| w[0] >= w[1])
                        && let Some(ref itx) = state.interrupt_tx
                    {
                        let _ = itx.send(BrainInterrupt::KillCg);
                    }
                }
            } else if anomaly_score > yellow_thresh {
                state.attention_state = AttentionState::Yellow;
                state.green_count = 0;
                if let Some(ref itx) = state.interrupt_tx {
                    let _ = itx.send(BrainInterrupt::AdjustCheckInterval(20));
                }
            }
        }
        AttentionState::Yellow => {
            if anomaly_score > red_thresh {
                state.yellow_count += 1;
                if state.yellow_count >= 2 {
                    state.attention_state = AttentionState::Red;
                    if let Some(ref itx) = state.interrupt_tx {
                        let _ = itx.send(BrainInterrupt::AdjustCheckInterval(5));
                    }
                }
            } else if anomaly_score < yellow_thresh {
                state.green_count += 1;
                if state.green_count >= 3 {
                    state.attention_state = AttentionState::Green;
                    state.green_count = 0;
                    state.yellow_count = 0;
                    if let Some(ref itx) = state.interrupt_tx {
                        let _ = itx.send(BrainInterrupt::AdjustCheckInterval(100));
                    }
                }
            } else {
                state.green_count = 0;
            }
        }
        AttentionState::Red => {
            if state.residual_history.len() >= 3 {
                let recent: Vec<f64> = state
                    .residual_history
                    .iter()
                    .rev()
                    .take(3)
                    .map(|(_, rz)| *rz)
                    .collect();
                let diverging = recent.windows(2).all(|w| w[0] >= w[1]);
                if diverging && let Some(ref itx) = state.interrupt_tx {
                    let _ = itx.send(BrainInterrupt::KillCg);
                }
            }
            if anomaly_score < 0.3 {
                state.green_count += 1;
                if state.green_count >= 3 {
                    state.attention_state = AttentionState::Yellow;
                    state.green_count = 0;
                    state.yellow_count = 0;
                    if let Some(ref itx) = state.interrupt_tx {
                        let _ = itx.send(BrainInterrupt::AdjustCheckInterval(20));
                    }
                }
            } else {
                state.green_count = 0;
            }
        }
    }
    let _ = resp_tx.send(NpuResponse::ResidualAck);
}
