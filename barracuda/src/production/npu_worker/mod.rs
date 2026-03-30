// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU worker thread for the 11-head dynamical mixed pipeline.
//!
//! Handles pre/during/post computation screening, Brain Layer 1 (CG residual
//! monitoring), and Brain Layer 3 (proxy features).

mod cg_residual;
mod handlers;
mod handlers_lifecycle;
mod head_confidence;
mod messages;
mod training;
mod worker_state;

use crate::lattice::gpu_hmc::BrainInterrupt;
use std::sync::mpsc;

pub use crate::production::trajectory_input::{
    TRAJECTORY_INPUT_DIM, trajectory_input, trajectory_input_with_proxy,
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
            let mut state = worker_state::WorkerState::new(lattice, interrupt_tx_out);

            for req in req_rx {
                match req {
                    NpuRequest::Shutdown => break,
                    NpuRequest::CgResidual(update) => {
                        cg_residual::handle_cg_residual(update, &mut state, &resp_tx);
                    }
                    other => {
                        handlers::handle(other, &mut state, &resp_tx);
                    }
                }
            }
        })?;

    Ok(NpuWorkerHandles {
        npu_tx: req_tx,
        npu_rx: resp_rx,
        interrupt_rx: interrupt_rx_out,
    })
}
