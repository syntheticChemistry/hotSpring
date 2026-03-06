// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU worker request/response types and handles.

use crate::lattice::gpu_hmc::{BrainInterrupt, CgResidualUpdate};
use crate::production::{BetaResult, MetaRow};
use std::sync::mpsc;

/// Message sent from the GPU/main thread to the NPU worker.
#[derive(Debug)]
pub enum NpuRequest {
    // ─── Pre-computation ───
    PreScreenBeta {
        candidates: Vec<f64>,
        meta_context: Vec<MetaRow>,
    },
    SuggestParameters {
        lattice: usize,
        beta: f64,
        mass: f64,
    },
    PredictCgIters {
        beta: f64,
        mass: f64,
        lattice: usize,
    },
    PredictQuenchedLength {
        beta: f64,
        mass: f64,
        lattice: usize,
        meta_context: Vec<MetaRow>,
    },

    // ─── During computation (quenched phase) ───
    QuenchedThermCheck {
        plaq_window: Vec<f64>,
        beta: f64,
        mass: f64,
    },

    // ─── During computation (dynamical phase) ───
    ThermCheck {
        plaq_window: Vec<f64>,
        beta: f64,
        mass: f64,
    },
    RejectPredict {
        beta: f64,
        plaquette: f64,
        delta_h: f64,
        acceptance_rate: f64,
        mass: f64,
    },
    PhaseClassify {
        beta: f64,
        plaquette: f64,
        polyakov: f64,
        susceptibility: f64,
        mass: f64,
        acceptance: f64,
    },

    // ─── Post-computation ───
    QualityScore {
        result: BetaResult,
    },
    AnomalyCheck {
        beta: f64,
        plaq: f64,
        delta_h: f64,
        cg_iters: usize,
        acceptance: f64,
        mass: f64,
    },
    SteerAdaptive {
        measured_betas: Vec<f64>,
        queued_betas: Vec<f64>,
        beta_min: f64,
        beta_max: f64,
        n_candidates: usize,
    },
    RecommendNextRun {
        all_results: Vec<BetaResult>,
        meta_table: Vec<MetaRow>,
    },
    // ─── Brain Layer 1: CG residual monitoring ───
    CgResidual(CgResidualUpdate),
    // ─── Brain Layer 3: Proxy features from CPU cortex ───
    ProxyFeatures {
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
    },

    // ─── Lifecycle ───
    Retrain {
        results: Vec<BetaResult>,
    },
    BootstrapFromMeta {
        rows: Vec<MetaRow>,
    },
    BootstrapFromWeights {
        path: String,
    },
    ExportWeights {
        path: String,
    },
    /// Save the Nautilus shell alongside ESN weights.
    ExportNautilusShell {
        path: String,
    },
    /// Load a Nautilus shell from disk.
    BootstrapNautilusShell {
        path: String,
    },
    DisagreementQuery {
        beta: f64,
        plaq: f64,
        mass: f64,
        chi: f64,
        acceptance: f64,
    },

    // ─── Full-stream trajectory events ───
    TrajectoryEvent(crate::production::TrajectoryEvent),
    /// Flush any buffered trajectory events (e.g. at end of beta scan).
    FlushTrajectoryBatch,
    /// Query structured metrics from all sub-models for logging.
    SubModelMetrics,
    /// Request predictions from sub-models for a given event (used for steering).
    SubModelPredict(crate::production::TrajectoryEvent),

    Shutdown,
}

/// Response from the NPU worker.
#[derive(Debug)]
pub enum NpuResponse {
    BetaPriorities(Vec<(f64, f64)>),
    ParameterSuggestion {
        dt: f64,
        n_md: usize,
    },
    CgEstimate(usize),
    QuenchedLengthEstimate(usize),
    QuenchedThermConverged(bool),
    ThermConverged(bool),
    RejectPrediction {
        likely_rejected: bool,
        confidence: f64,
    },
    PhaseLabel(&'static str),
    Quality(f64),
    AnomalyFlag {
        is_anomaly: bool,
        _score: f64,
    },
    AdaptiveSteered {
        suggestion: Option<f64>,
        saturated: bool,
    },
    NextRunPlan {
        betas: Vec<f64>,
        mass: f64,
        lattice: usize,
    },
    Retrained {
        beta_c: f64,
    },
    Bootstrapped {
        n_points: usize,
    },
    WeightsSaved {
        path: String,
    },
    NautilusShellSaved {
        path: String,
    },
    NautilusShellLoaded {
        n_observations: usize,
        n_generations: usize,
    },
    ResidualAck,
    ProxyFeaturesAck,
    DisagreementSnapshot {
        delta_cg: f64,
        delta_phase: f64,
        delta_anomaly: f64,
        delta_priority: f64,
        urgency: f64,
    },
    /// Ack for trajectory events (fire-and-forget, not recv'd by main thread).
    TrajectoryBatchProcessed {
        n_events: usize,
    },
    /// Structured sub-model metrics for experimentation logging.
    SubModelMetricsSnapshot(serde_json::Value),
    /// Sub-model predictions for steering decisions.
    SubModelPredictions {
        /// CG cost: (`predicted_cg_iters_norm`, `stall_probability`)
        cg_cost: Option<Vec<f64>>,
        /// Steering brain: (`next_beta_priority`, `optimal_dt`, `optimal_n_md`, saturation, skip)
        steering: Option<Vec<f64>>,
        /// Phase oracle: (`phase_continuous`, `polyakov_mag`, susceptibility)
        phase: Option<Vec<f64>>,
        /// Trajectory predictor: (`next_delta_h`, `next_plaq`, `therm_progress`)
        trajectory: Option<Vec<f64>>,
    },
}

/// Spawn result includes the interrupt channel for brain-mode CG monitoring.
pub struct NpuWorkerHandles {
    pub npu_tx: mpsc::Sender<NpuRequest>,
    pub npu_rx: mpsc::Receiver<NpuResponse>,
    pub interrupt_rx: mpsc::Receiver<BrainInterrupt>,
}
