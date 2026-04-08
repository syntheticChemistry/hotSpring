// SPDX-License-Identifier: AGPL-3.0-or-later

#![allow(missing_docs)]

//! NPU worker mutable state and ESN factory.

use crate::md::reservoir::{Activation, EchoStateNetwork, EsnConfig, MultiHeadNpu, heads};
use crate::production::{AttentionState, BetaResult};
use barracuda::nautilus::{NautilusBrain, NautilusBrainConfig};
use std::sync::mpsc;

use super::head_confidence::HeadConfidence;
use super::training::build_training_data;
use crate::lattice::gpu_hmc::BrainInterrupt;
use crate::production::TrajectoryEvent;
use crate::production::sub_models::SubModelRegistry;
use crate::proxy::ProxyFeatures;

/// Mutable state for the NPU worker thread.
pub(super) struct WorkerState {
    pub multi_npu: Option<MultiHeadNpu>,
    pub residual_history: Vec<(usize, f64)>,
    pub attention_state: AttentionState,
    pub green_count: usize,
    pub yellow_count: usize,
    pub latest_proxy: Option<ProxyFeatures>,
    pub last_plaq: f64,
    pub last_chi: f64,
    pub last_acc: f64,
    pub head_confidence: HeadConfidence,
    pub nautilus_brain: NautilusBrain,
    pub sub_models: SubModelRegistry,
    pub traj_batch: Vec<TrajectoryEvent>,
    pub lattice: usize,
    pub interrupt_tx: Option<mpsc::Sender<BrainInterrupt>>,
}

impl WorkerState {
    /// Create fresh worker state for the given lattice size.
    #[must_use]
    pub fn new(lattice: usize, interrupt_tx: mpsc::Sender<BrainInterrupt>) -> Self {
        Self {
            multi_npu: None,
            residual_history: Vec::new(),
            attention_state: AttentionState::Green,
            green_count: 0,
            yellow_count: 0,
            latest_proxy: None,
            last_plaq: 0.5,
            last_chi: 0.0,
            last_acc: 0.5,
            head_confidence: HeadConfidence::new(heads::NUM_HEADS),
            nautilus_brain: NautilusBrain::new(
                NautilusBrainConfig::default(),
                &format!("hotspring-{lattice}"),
            ),
            sub_models: SubModelRegistry::default_models(),
            traj_batch: Vec::with_capacity(8),
            lattice,
            interrupt_tx: Some(interrupt_tx),
        }
    }

    /// Build a new MultiHeadNpu from accumulated BetaResults, or None if insufficient data.
    #[must_use]
    pub fn make_multi_esn(&self, seed: u64, results: &[BetaResult]) -> Option<MultiHeadNpu> {
        if results.is_empty() {
            return None;
        }
        let (seqs, tgts) = build_training_data(results, self.lattice);
        let mut esn = EchoStateNetwork::new(EsnConfig {
            input_size: 6,
            output_size: heads::NUM_HEADS,
            regularization: 1e-3,
            seed,
            activation: Activation::ReluTanhApprox,
            ..EsnConfig::default()
        });
        esn.train(&seqs, &tgts);
        esn.export_weights()
            .map(|w| MultiHeadNpu::from_exported(&w))
    }
}
