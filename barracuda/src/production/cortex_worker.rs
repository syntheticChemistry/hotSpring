// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU cortex worker for Anderson 3D proxy evaluation.
//!
//! Runs the physics proxy (Anderson 3D level statistics) in a background
//! thread and sends `ProxyFeatures` to the main thread for NPU training.

use crate::proxy::{self, CortexRequest, ProxyFeatures};
use std::sync::mpsc;

/// Handles for communicating with the CPU cortex worker thread.
pub struct CortexWorkerHandles {
    /// Send requests to the worker.
    pub cortex_tx: mpsc::Sender<CortexRequest>,
    /// Receive proxy features from the worker.
    pub proxy_rx: mpsc::Receiver<ProxyFeatures>,
}

/// Spawn the CPU cortex worker.
///
/// The worker evaluates the Anderson 3D proxy for each request and
/// sends `ProxyFeatures` back for NPU proxy-head training.
///
/// # Errors
/// Returns `Err` if the thread fails to spawn (OOM, resource exhaustion).
pub fn spawn_cortex_worker() -> Result<CortexWorkerHandles, std::io::Error> {
    let (req_tx, req_rx) = mpsc::channel::<CortexRequest>();
    let (feat_tx, feat_rx) = mpsc::channel::<ProxyFeatures>();

    std::thread::Builder::new()
        .name("cpu-cortex".into())
        .spawn(move || {
            let mut seed_counter = 42u64;
            for req in req_rx {
                seed_counter += 1;
                let features = proxy::combined_proxy(&req, seed_counter);
                eprintln!(
                    "  [Cortex] β={:.4}: ⟨r⟩={:.3} |λ|_min={:.3} [{}] Potts={:.3} [{}] ({:.0}ms)",
                    features.beta,
                    features.level_spacing_ratio,
                    features.lambda_min,
                    features.phase,
                    features.potts_magnetization,
                    features.potts_phase,
                    features.wall_ms,
                );
                feat_tx.send(features).ok();
            }
        })?;

    Ok(CortexWorkerHandles {
        cortex_tx: req_tx,
        proxy_rx: feat_rx,
    })
}
