// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! NPU worker thread for the 11-head dynamical mixed pipeline.
//!
//! Handles pre/during/post computation screening, Brain Layer 1 (CG residual
//! monitoring), and Brain Layer 3 (proxy features).

use crate::lattice::gpu_hmc::{BrainInterrupt, CgResidualUpdate};
use crate::md::reservoir::{
    heads, Activation, EchoStateNetwork, EsnConfig, ExportedWeights, HeadGroupDisagreement,
    MultiHeadNpu,
};
use crate::production::{
    check_thermalization, plaquette_variance, predict_rejection, AttentionState, BetaResult,
    MetaRow, TrajectoryEvent, TrajectoryPhase,
};
use crate::production::sub_models::SubModelRegistry;
use crate::provenance::KNOWN_BETA_C_SU3_NT4;
use crate::proxy::ProxyFeatures;
use barracuda::nautilus::{BetaObservation, NautilusBrain, NautilusBrainConfig};
use std::sync::mpsc;

/// Gen 1 canonical 6D input vector (kept for backward compat with existing ESN).
fn canonical_input(beta: f64, plaq: f64, mass: f64, chi: f64, acceptance: f64, lattice: usize) -> Vec<f64> {
    vec![
        (beta - 5.0) / 2.0,
        plaq,
        mass,
        chi / 1000.0,
        acceptance,
        lattice as f64 / 8.0,
    ]
}

fn canonical_seq(beta: f64, plaq: f64, mass: f64, chi: f64, acceptance: f64, lattice: usize) -> Vec<Vec<f64>> {
    vec![canonical_input(beta, plaq, mass, chi, acceptance, lattice); 10]
}

/// Gen 2 full-stream 21D input vector from a per-trajectory event + proxy context.
///
/// Dimensions 0-14: trajectory data (always available).
/// Dimensions 15-20: proxy features (0.0 when no proxy has been evaluated yet).
pub fn trajectory_input(evt: &TrajectoryEvent) -> Vec<f64> {
    trajectory_input_with_proxy(evt, None)
}

/// Build the full input vector with optional proxy features appended.
pub fn trajectory_input_with_proxy(evt: &TrajectoryEvent, proxy: Option<&ProxyFeatures>) -> Vec<f64> {
    vec![
        // 0-14: trajectory data
        (evt.beta - 5.0) / 2.0,
        evt.plaquette,
        evt.mass,
        evt.delta_h.clamp(-5.0, 5.0) / 5.0,
        evt.cg_iterations as f64 / 100_000.0,
        if evt.accepted { 1.0 } else { 0.0 },
        evt.polyakov_re.clamp(-1.0, 1.0),
        evt.polyakov_phase / std::f64::consts::PI,
        evt.action_density / 6.0,
        evt.plaquette_var.clamp(0.0, 0.1) * 10.0,
        evt.running_acceptance,
        evt.lattice as f64 / 12.0,
        evt.traj_idx as f64 / 60.0,
        evt.wall_us as f64 / 1_000_000.0,
        match evt.phase_tag {
            TrajectoryPhase::Quenched => 0.0,
            TrajectoryPhase::Therm => 0.5,
            TrajectoryPhase::Measurement => 1.0,
        },
        // 15-20: proxy features (Anderson + Potts)
        proxy.map_or(0.0, |p| p.level_spacing_ratio),
        proxy.map_or(0.0, |p| p.lambda_min.clamp(0.0, 1.0)),
        proxy.map_or(0.0, |p| p.ipr),
        proxy.map_or(0.0, |p| p.condition_number.clamp(0.0, 100.0) / 100.0),
        proxy.map_or(0.0, |p| p.potts_magnetization),
        proxy.map_or(0.0, |p| p.potts_susceptibility.clamp(0.0, 100.0) / 100.0),
    ]
}

/// Number of dimensions in the Gen 2 trajectory input vector (15 traj + 6 proxy).
pub const TRAJECTORY_INPUT_DIM: usize = 21;

fn heuristic_phase(beta: f64) -> &'static str {
    if beta > 5.79 { "deconfined" }
    else if beta > 5.59 { "transition" }
    else { "confined" }
}

// ═══════════════════════════════════════════════════════════════════
//  Head Confidence Tracker
// ═══════════════════════════════════════════════════════════════════

const HEAD_CONFIDENCE_WINDOW: usize = 20;
const REGRESSION_TRUST_THRESHOLD: f64 = 0.3;
const CLASSIFICATION_TRUST_THRESHOLD: f64 = 0.6;

/// Whether a head predicts a continuous value or a discrete class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadMetric {
    /// R² for continuous targets (CG cost, delta_H, quality, etc.).
    Regression,
    /// Accuracy for binary/categorical targets (therm, anomaly, kill).
    Classification,
}

/// Rolling per-head confidence tracker with per-head metric type.
///
/// Tracks prediction-vs-outcome for each of the 36 heads. A head is "trusted"
/// for steering when its rolling metric exceeds the type-specific threshold:
/// - Regression heads: R² >= 0.3
/// - Classification heads: accuracy >= 0.6
///
/// Untrusted heads fall back to heuristics but continue receiving training data,
/// so they can graduate to trusted status as the ESN learns.
struct HeadConfidence {
    predictions: Vec<Vec<f64>>,
    actuals: Vec<Vec<f64>>,
    trusted: Vec<bool>,
    r2: Vec<f64>,
    metrics: Vec<HeadMetric>,
}

impl HeadConfidence {
    fn new(n_heads: usize) -> Self {
        let mut metrics = vec![HeadMetric::Regression; n_heads];
        // Classification heads: therm, anomaly, kill/skip, quenched therm
        for &h in &[
            heads::A3_ANDERSON_ANOMALY, heads::A4_ANDERSON_THERM,
            heads::B3_QCD_ANOMALY, heads::B4_QCD_THERM,
            heads::C3_POTTS_ANOMALY,
            heads::D4_KILL_DECISION, heads::D5_SKIP_DECISION,
        ] {
            if h < n_heads {
                metrics[h] = HeadMetric::Classification;
            }
        }
        Self {
            predictions: vec![Vec::with_capacity(HEAD_CONFIDENCE_WINDOW); n_heads],
            actuals: vec![Vec::with_capacity(HEAD_CONFIDENCE_WINDOW); n_heads],
            trusted: vec![false; n_heads],
            r2: vec![0.0; n_heads],
            metrics,
        }
    }

    fn record_prediction(&mut self, head: usize, predicted: f64) {
        if head < self.predictions.len() {
            let buf = &mut self.predictions[head];
            if buf.len() >= HEAD_CONFIDENCE_WINDOW {
                buf.remove(0);
            }
            buf.push(predicted);
        }
    }

    fn record_actual(&mut self, head: usize, actual: f64) {
        if head < self.actuals.len() {
            let buf = &mut self.actuals[head];
            if buf.len() >= HEAD_CONFIDENCE_WINDOW {
                buf.remove(0);
            }
            buf.push(actual);
            self.recompute(head);
        }
    }

    fn recompute(&mut self, head: usize) {
        let pred = &self.predictions[head];
        let actual = &self.actuals[head];
        let n = pred.len().min(actual.len());
        if n < 3 {
            self.trusted[head] = false;
            self.r2[head] = 0.0;
            return;
        }
        let offset = pred.len().saturating_sub(n);
        let a_offset = actual.len().saturating_sub(n);

        match self.metrics[head] {
            HeadMetric::Regression => {
                let mean_a: f64 = actual[a_offset..].iter().sum::<f64>() / n as f64;
                let ss_tot: f64 = actual[a_offset..].iter().map(|a| (a - mean_a).powi(2)).sum();
                let ss_res: f64 = pred[offset..].iter().zip(&actual[a_offset..])
                    .map(|(p, a)| (a - p).powi(2)).sum();
                let r2 = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
                self.r2[head] = r2;
                self.trusted[head] = r2 >= REGRESSION_TRUST_THRESHOLD;
            }
            HeadMetric::Classification => {
                let correct = pred[offset..].iter().zip(&actual[a_offset..])
                    .filter(|(&p, &a)| (p > 0.5) == (a > 0.5))
                    .count();
                let accuracy = correct as f64 / n as f64;
                self.r2[head] = accuracy;
                self.trusted[head] = accuracy >= CLASSIFICATION_TRUST_THRESHOLD;
            }
        }
    }

    fn is_trusted(&self, head: usize) -> bool {
        head < self.trusted.len() && self.trusted[head]
    }

    fn status_line(&self) -> String {
        let trusted_count = self.trusted.iter().filter(|&&t| t).count();
        let total = self.trusted.len();
        let best: Vec<String> = self.r2.iter().enumerate()
            .filter(|(_, &r)| r > 0.1)
            .map(|(i, r)| {
                let tag = match self.metrics.get(i) {
                    Some(HeadMetric::Classification) => "acc",
                    _ => "R²",
                };
                format!("H{i}={r:.2}{tag}")
            })
            .collect();
        format!("{trusted_count}/{total} trusted [{}]",
            if best.is_empty() { "none above 0.1".into() } else { best.join(", ") })
    }
}

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
        _confidence: f64,
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
        /// CG cost: (predicted_cg_iters_norm, stall_probability)
        cg_cost: Option<Vec<f64>>,
        /// Steering brain: (next_beta_priority, optimal_dt, optimal_n_md, saturation, skip)
        steering: Option<Vec<f64>>,
        /// Phase oracle: (phase_continuous, polyakov_mag, susceptibility)
        phase: Option<Vec<f64>>,
        /// Trajectory predictor: (next_delta_h, next_plaq, therm_progress)
        trajectory: Option<Vec<f64>>,
    },
}

/// Spawn result includes the interrupt channel for brain-mode CG monitoring.
pub struct NpuWorkerHandles {
    pub npu_tx: mpsc::Sender<NpuRequest>,
    pub npu_rx: mpsc::Receiver<NpuResponse>,
    pub interrupt_rx: mpsc::Receiver<BrainInterrupt>,
}

/// Spawn the NPU worker thread. Returns handles for request/response and brain interrupt.
#[allow(clippy::expect_used)]
pub fn spawn_npu_worker(lattice: usize) -> NpuWorkerHandles {
    let (req_tx, req_rx) = mpsc::channel::<NpuRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<NpuResponse>();
    let (interrupt_tx_out, interrupt_rx_out) = mpsc::channel::<BrainInterrupt>();

    std::thread::Builder::new()
        .name("npu-cerebellum".into())
        .spawn(move || {
            let mut multi_npu: Option<MultiHeadNpu> = None;
            let interrupt_tx: Option<mpsc::Sender<BrainInterrupt>> = Some(interrupt_tx_out);

            // Brain Layer 1 + 4 state
            let mut residual_history: Vec<(usize, f64)> = Vec::new();
            let mut attention_state = AttentionState::Green;
            let mut green_count: usize = 0;
            let mut yellow_count: usize = 0;

            #[allow(unused_assignments)]
            let mut latest_proxy: Option<ProxyFeatures> = None;

            // Last-known summary stats for input alignment (updated on Retrain/Quality)
            let mut last_plaq: f64 = 0.5;
            let mut last_chi: f64 = 0.0;
            let mut last_acc: f64 = 0.5;

            let mut head_confidence = HeadConfidence::new(heads::NUM_HEADS);

            // Gen 2: Nautilus evolutionary reservoir (cross-run structural learning)
            let mut nautilus_brain = NautilusBrain::new(
                NautilusBrainConfig::default(),
                &format!("hotspring-{lattice}"),
            );

            // Gen 2: Sub-model registry (specialized ESNs on full trajectory stream)
            let mut sub_models = SubModelRegistry::default_models();
            let mut traj_batch: Vec<crate::production::TrajectoryEvent> = Vec::with_capacity(8);

            let make_multi_esn = |seed: u64, results: &[BetaResult]| -> Option<MultiHeadNpu> {
                if results.is_empty() {
                    return None;
                }
                let (seqs, tgts) = build_training_data(results, lattice);
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
            };

            for req in req_rx {
                match req {
                    NpuRequest::PreScreenBeta {
                        candidates,
                        meta_context,
                    } => {
                        let priorities: Vec<(f64, f64)> = candidates
                            .iter()
                            .map(|&beta| {
                                let score = if let Some(ref mut npu) = multi_npu {
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
                        resp_tx.send(NpuResponse::BetaPriorities(priorities)).ok();
                    }

                    NpuRequest::SuggestParameters {
                        lattice,
                        beta,
                        mass,
                    } => {
                        let heuristic_params = |lat: usize| -> (f64, usize) {
                            let vol = (lat as f64).powi(4);
                            let scale = (4096.0 / vol).powf(0.25);
                            let dt = (0.05 * scale).max(0.001);
                            let n_md = ((0.5 / dt).round() as usize).max(20);
                            (dt, n_md)
                        };
                        let (dt, n_md) = if let Some(ref mut npu) = multi_npu {
                            let seq = canonical_seq(beta, last_plaq, mass, last_chi, last_acc, lattice);
                            let raw = npu.predict_head(&seq, heads::PARAM_SUGGEST);
                            head_confidence.record_prediction(heads::PARAM_SUGGEST, raw.clamp(0.0, 0.05));
                            if head_confidence.is_trusted(heads::PARAM_SUGGEST) {
                                let dt_suggest = 0.001 + raw.abs() * 0.05;
                                let n_md_suggest = ((0.5 / dt_suggest).round() as usize).max(10);
                                (dt_suggest, n_md_suggest)
                            } else {
                                heuristic_params(lattice)
                            }
                        } else {
                            heuristic_params(lattice)
                        };
                        resp_tx
                            .send(NpuResponse::ParameterSuggestion { dt, n_md })
                            .ok();
                    }

                    NpuRequest::PredictCgIters {
                        beta,
                        mass,
                        lattice,
                    } => {
                        let heuristic_cg = |lat: usize, m: f64| -> usize {
                            let vol = (lat as f64).powi(4);
                            (100.0 + vol.sqrt() / m.max(0.01)).round() as usize
                        };
                        let est = if let Some(ref mut npu) = multi_npu {
                            let seq = canonical_seq(beta, last_plaq, mass, last_chi, last_acc, lattice);
                            let raw = npu.predict_head(&seq, heads::CG_ESTIMATE);
                            head_confidence.record_prediction(heads::CG_ESTIMATE, raw.clamp(0.0, 5.0));
                            if head_confidence.is_trusted(heads::CG_ESTIMATE) {
                                (raw.abs() * 500.0).round() as usize
                            } else {
                                heuristic_cg(lattice, mass)
                            }
                        } else {
                            heuristic_cg(lattice, mass)
                        };
                        resp_tx.send(NpuResponse::CgEstimate(est)).ok();
                    }

                    NpuRequest::PredictQuenchedLength {
                        beta,
                        mass,
                        lattice,
                        meta_context,
                    } => {
                        let steps = if let Some(ref mut npu) = multi_npu {
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
                            let input = canonical_input(
                                beta,
                                meta_plaq,
                                mass,
                                meta_chi,
                                meta_acc,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUENCHED_LENGTH);
                            (raw.abs() * 200.0).round().clamp(5.0, 200.0) as usize
                        } else {
                            let proximity = (-(beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.3).exp();
                            let base = 20 + (80.0 * proximity) as usize;
                            base.min(lattice * 10)
                        };
                        resp_tx
                            .send(NpuResponse::QuenchedLengthEstimate(steps))
                            .ok();
                    }

                    NpuRequest::QuenchedThermCheck { plaq_window, beta, mass } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean =
                                plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
                            let chi = var * (plaq_window.len() as f64);
                            let input = canonical_input(
                                beta,
                                mean,
                                mass,
                                chi,
                                last_acc,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUENCHED_THERM);
                            raw > 0.5
                        } else {
                            check_thermalization(&plaq_window, beta)
                        };
                        resp_tx
                            .send(NpuResponse::QuenchedThermConverged(converged))
                            .ok();
                    }

                    NpuRequest::ThermCheck { plaq_window, beta, mass } => {
                        let converged = if let Some(ref mut npu) = multi_npu {
                            let var = plaquette_variance(&plaq_window);
                            let mean =
                                plaq_window.iter().sum::<f64>() / plaq_window.len().max(1) as f64;
                            let chi = var * (plaq_window.len() as f64);
                            let input = canonical_input(
                                beta,
                                mean,
                                mass,
                                chi,
                                last_acc,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::THERM_DETECT);
                            raw > 0.5
                        } else {
                            check_thermalization(&plaq_window, beta)
                        };
                        resp_tx.send(NpuResponse::ThermConverged(converged)).ok();
                    }

                    NpuRequest::RejectPredict {
                        beta,
                        plaquette,
                        delta_h,
                        acceptance_rate,
                        mass,
                    } => {
                        let (likely_rejected, confidence) = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_input(
                                beta,
                                plaquette,
                                mass,
                                last_chi,
                                acceptance_rate,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            // Predict continuous delta_H; decision boundary at 0
                            let raw = npu.predict_head(&seq, heads::REJECT_PREDICT);
                            let predicted_delta_h = raw * 10.0;
                            head_confidence.record_prediction(heads::REJECT_PREDICT, predicted_delta_h);
                            if head_confidence.is_trusted(heads::REJECT_PREDICT) {
                                let rejected = predicted_delta_h > 0.0;
                                let conf = 1.0 / (1.0 + (-predicted_delta_h.abs()).exp());
                                (rejected, conf)
                            } else {
                                let (lr, conf) = predict_rejection(0.0, 0.0, 0.0, delta_h, acceptance_rate);
                                (lr, conf)
                            }
                        } else {
                            predict_rejection(0.0, 0.0, 0.0, delta_h, acceptance_rate)
                        };
                        resp_tx
                            .send(NpuResponse::RejectPrediction {
                                likely_rejected,
                                _confidence: confidence,
                            })
                            .ok();
                    }

                    NpuRequest::PhaseClassify {
                        beta,
                        plaquette,
                        polyakov: _,
                        susceptibility,
                        mass,
                        acceptance,
                    } => {
                        let label = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_input(
                                beta, plaquette, mass, susceptibility, acceptance, lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::PHASE_CLASSIFY);
                            head_confidence.record_prediction(heads::PHASE_CLASSIFY, raw.clamp(0.0, 1.0));
                            if head_confidence.is_trusted(heads::PHASE_CLASSIFY) {
                                match () {
                                    () if raw > 0.6 => "deconfined",
                                    () if raw > 0.3 => "transition",
                                    () => "confined",
                                }
                            } else {
                                heuristic_phase(beta)
                            }
                        } else {
                            heuristic_phase(beta)
                        };
                        resp_tx.send(NpuResponse::PhaseLabel(label)).ok();
                    }

                    NpuRequest::QualityScore { result } => {
                        last_plaq = result.mean_plaq;
                        last_chi = result.susceptibility;
                        last_acc = result.acceptance;
                        // Ground truth: continuous delta_H proxy from acceptance rate
                        // High acceptance → negative delta_H (accepted), low → positive (rejected)
                        let actual_delta_h = -(result.acceptance - 0.5) * 4.0;
                        head_confidence.record_actual(heads::REJECT_PREDICT, actual_delta_h);
                        // Continuous phase: use Polyakov loop magnitude as order parameter
                        let actual_phase = result.polyakov.abs().clamp(0.0, 1.0);
                        head_confidence.record_actual(heads::PHASE_CLASSIFY, actual_phase);
                        head_confidence.record_actual(heads::CG_ESTIMATE, result.mean_cg_iters / 100_000.0);
                        let actual_quality = if result.mean_plaq > 1e-9 {
                            1.0 - (result.std_plaq / result.mean_plaq).min(1.0)
                        } else { 0.0 };
                        head_confidence.record_actual(heads::QUALITY_SCORE, actual_quality);

                        let score = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_input(
                                result.beta,
                                result.mean_plaq,
                                result.mass,
                                result.susceptibility,
                                result.acceptance,
                                lattice,
                            );
                            let seq = vec![input; 10];
                            let raw = npu.predict_head(&seq, heads::QUALITY_SCORE).clamp(0.0, 1.0);
                            head_confidence.record_prediction(heads::QUALITY_SCORE, raw);
                            if head_confidence.is_trusted(heads::QUALITY_SCORE) {
                                raw
                            } else {
                                let acc_ok = if result.acceptance > 0.3 { 0.4 } else { 0.1 };
                                let stats_ok = if result.n_traj >= 100 { 0.3 } else { 0.1 };
                                let cg_ok = if result.mean_cg_iters < 1000.0 { 0.3 } else { 0.1 };
                                acc_ok + stats_ok + cg_ok
                            }
                        } else {
                            let acc_ok = if result.acceptance > 0.3 { 0.4 } else { 0.1 };
                            let stats_ok = if result.n_traj >= 100 { 0.3 } else { 0.1 };
                            let cg_ok = if result.mean_cg_iters < 1000.0 { 0.3 } else { 0.1 };
                            acc_ok + stats_ok + cg_ok
                        };
                        // Feed observation to Nautilus brain for cross-run learning
                        nautilus_brain.observe(BetaObservation {
                            beta: result.beta,
                            plaquette: result.mean_plaq,
                            cg_iters: result.mean_cg_iters,
                            acceptance: result.acceptance,
                            delta_h_abs: (1.0 - result.acceptance).abs(),
                            quenched_plaq: None,
                            quenched_plaq_var: None,
                            anderson_r: None,
                            anderson_lambda_min: None,
                        });

                        resp_tx.send(NpuResponse::Quality(score)).ok();
                    }

                    NpuRequest::AnomalyCheck {
                        beta,
                        plaq,
                        delta_h,
                        cg_iters,
                        acceptance,
                        mass,
                    } => {
                        let (is_anomaly, score) = if let Some(ref mut npu) = multi_npu {
                            let seq = canonical_seq(beta, plaq, mass, last_chi, acceptance, lattice);
                            let raw = npu.predict_head(&seq, heads::ANOMALY_DETECT);
                            (raw > 0.7, raw.clamp(0.0, 1.0))
                        } else {
                            let anomaly = delta_h.abs() > 50.0
                                || cg_iters > 4000
                                || !(0.0..=1.0).contains(&plaq);
                            let s = if anomaly { 0.9 } else { 0.1 };
                            (anomaly, s)
                        };
                        resp_tx
                            .send(NpuResponse::AnomalyFlag {
                                is_anomaly,
                                _score: score,
                            })
                            .ok();
                    }

                    NpuRequest::SteerAdaptive {
                        measured_betas,
                        queued_betas,
                        beta_min,
                        beta_max,
                        n_candidates,
                    } => {
                        let all_known: Vec<f64> = measured_betas
                            .iter()
                            .chain(queued_betas.iter())
                            .copied()
                            .collect();
                        let n_measured = measured_betas.len();

                        let (suggestion, best_score) = if let Some(ref mut npu) = multi_npu {
                            let mut best_beta = None;
                            let mut top_score = f64::NEG_INFINITY;
                            let step = (beta_max - beta_min) / (n_candidates as f64 + 1.0);
                            let exclusion = step * 0.3_f64.max(0.025);
                            for ci in 1..=n_candidates {
                                let candidate = beta_min + ci as f64 * step;
                                let too_close_measured = measured_betas
                                    .iter()
                                    .any(|&m| (m - candidate).abs() < exclusion);
                                let too_close_queued = queued_betas
                                    .iter()
                                    .any(|&q| (q - candidate).abs() < exclusion);
                                if too_close_measured || too_close_queued {
                                    continue;
                                }
                                let seq = canonical_seq(candidate, last_plaq, 0.1, last_chi, last_acc, lattice);
                                let all = npu.predict_all_heads(&seq);
                                let priority = all[heads::BETA_PRIORITY];
                                let uncertainty = (all[heads::QUALITY_SCORE] - 0.5).abs();
                                let score = priority + uncertainty * 0.5;
                                if score > top_score {
                                    top_score = score;
                                    best_beta = Some(candidate);
                                }
                            }
                            (best_beta, top_score)
                        } else if nautilus_brain.trained {
                            let edges = nautilus_brain.detect_concept_edges();
                            if let Some((edge_beta, edge_score)) = edges.iter()
                                .filter(|(b, _)| *b >= beta_min && *b <= beta_max)
                                .max_by(|a, b| a.1.total_cmp(&b.1))
                            {
                                let too_close = all_known.iter()
                                    .any(|&m| (m - edge_beta).abs() < 0.05);
                                if !too_close {
                                    (Some(*edge_beta), *edge_score)
                                } else {
                                    (find_largest_gaps(&all_known, beta_min, beta_max, 1).into_iter().next(), 0.0)
                                }
                            } else {
                                (find_largest_gaps(&all_known, beta_min, beta_max, 1).into_iter().next(), 0.0)
                            }
                        } else {
                            let gaps = find_largest_gaps(&all_known, beta_min, beta_max, 1);
                            (gaps.into_iter().next(), 0.0)
                        };

                        // Saturation detection: the NPU believes this parameter
                        // set has diminishing returns when:
                        //  1. Dense coverage: largest remaining gap < 5% of range
                        //  2. Low novelty: best candidate score below threshold
                        //  3. Sufficient data: at least 5 measured points
                        let range = beta_max - beta_min;
                        let largest_gap = {
                            let mut sorted = all_known.clone();
                            sorted.push(beta_min);
                            sorted.push(beta_max);
                            sorted.sort_by(f64::total_cmp);
                            sorted.dedup();
                            sorted.windows(2)
                                .map(|w| w[1] - w[0])
                                .fold(0.0f64, f64::max)
                        };
                        let dense_enough = largest_gap < range * 0.05;
                        let low_novelty = best_score < 0.15 && best_score > f64::NEG_INFINITY;
                        let enough_points = n_measured >= 5;
                        let saturated = enough_points && (dense_enough || low_novelty);

                        if saturated {
                            eprintln!("  [NPU] Saturation detected: gap={largest_gap:.3}/{range:.1} \
                                score={best_score:.3} n={n_measured} — recommend moving on");
                        }

                        resp_tx.send(NpuResponse::AdaptiveSteered { suggestion, saturated }).ok();
                    }

                    NpuRequest::RecommendNextRun {
                        all_results,
                        meta_table: _,
                    } => {
                        let plan = if let Some(ref mut npu) = multi_npu {
                            let last = all_results.last();
                            let last_beta = last.map_or(5.69, |r| r.beta);
                            let seq = canonical_seq(
                                last_beta,
                                last.map_or(0.5, |r| r.mean_plaq),
                                last.map_or(0.1, |r| r.mass),
                                last.map_or(0.0, |r| r.susceptibility),
                                last.map_or(0.5, |r| r.acceptance),
                                lattice,
                            );
                            let raw = npu.predict_head(&seq, heads::NEXT_RUN_RECOMMEND);
                            let suggested_beta = 5.0 + raw.abs() * 2.0;
                            let mass = last.map_or(0.1, |r| r.mass);
                            let lattice = 8;
                            (vec![suggested_beta], mass, lattice)
                        } else {
                            let measured: Vec<f64> = all_results.iter().map(|r| r.beta).collect();
                            let gaps = find_largest_gaps(&measured, 5.0, 7.0, 3);
                            let mass = all_results.first().map_or(0.1, |r| r.mass);
                            (gaps, mass, 8)
                        };
                        resp_tx
                            .send(NpuResponse::NextRunPlan {
                                betas: plan.0,
                                mass: plan.1,
                                lattice: plan.2,
                            })
                            .ok();
                    }

                    NpuRequest::Retrain { results } => {
                        for r in &results {
                            let actual_delta_h = -(r.acceptance - 0.5) * 4.0;
                            head_confidence.record_actual(heads::REJECT_PREDICT, actual_delta_h);
                            head_confidence.record_actual(heads::CG_ESTIMATE, r.mean_cg_iters / 100_000.0);
                            let actual_phase = r.polyakov.abs().clamp(0.0, 1.0);
                            head_confidence.record_actual(heads::PHASE_CLASSIFY, actual_phase);
                            let quality = if r.mean_plaq > 1e-9 {
                                1.0 - (r.std_plaq / r.mean_plaq).min(1.0)
                            } else { 0.0 };
                            head_confidence.record_actual(heads::QUALITY_SCORE, quality);
                        }
                        eprintln!("  [NPU] Head confidence: {}", head_confidence.status_line());

                        let seed = 99 + results.len() as u64;
                        if let Some(new_npu) = make_multi_esn(seed, &results) {
                            multi_npu = Some(new_npu);
                        }
                        // Evolve Nautilus shell on accumulated observations
                        if let Some(mse) = nautilus_brain.train() {
                            let n_obs = nautilus_brain.observations.len();
                            let drifting = nautilus_brain.is_drifting();
                            eprintln!("  [Nautilus] Trained: {n_obs} obs, MSE={mse:.6}, drift={drifting}");
                        }
                        // Gen 2: Sub-model status
                        eprintln!("  [Sub-models] {}", sub_models.status_line());

                        let beta_c = estimate_beta_c(&results);
                        resp_tx.send(NpuResponse::Retrained { beta_c }).ok();
                    }

                    NpuRequest::BootstrapFromMeta { rows } => {
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
                        if let Some(new_npu) = make_multi_esn(42, &results) {
                            multi_npu = Some(new_npu);
                        }
                        resp_tx.send(NpuResponse::Bootstrapped { n_points: n }).ok();
                    }

                    NpuRequest::BootstrapFromWeights { path } => {
                        match std::fs::read_to_string(&path) {
                            Ok(json) => match serde_json::from_str::<ExportedWeights>(&json) {
                                Ok(weights) => {
                                    multi_npu = Some(MultiHeadNpu::from_exported(&weights));
                                    resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 }).ok();
                                }
                                Err(e) => {
                                    eprintln!("  Warning: failed to parse weights {path}: {e}");
                                    resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 }).ok();
                                }
                            },
                            Err(e) => {
                                eprintln!("  Warning: cannot read {path}: {e}");
                                resp_tx.send(NpuResponse::Bootstrapped { n_points: 0 }).ok();
                            }
                        }
                        // Auto-load sibling Nautilus shell if it exists
                        let shell_path = path.replace(".bin", ".nautilus.json")
                            .replace(".json", ".nautilus.json");
                        if let Ok(json) = std::fs::read_to_string(&shell_path) {
                            if let Ok(brain) = NautilusBrain::from_json(&json) {
                                let n_obs = brain.observations.len();
                                let n_gen = brain.shell.history.len();
                                nautilus_brain = brain;
                                eprintln!("  [Nautilus] Auto-loaded shell: {n_obs} obs, {n_gen} generations");
                            }
                        }
                    }

                    NpuRequest::ExportWeights { path } => {
                        let saved = if let Some(ref mut npu) = multi_npu {
                            let base = npu.base_mut();
                            let weights = ExportedWeights {
                                w_in: base.export_w_in(),
                                w_res: base.export_w_res(),
                                w_out: base.export_w_out(),
                                input_size: base.input_size(),
                                reservoir_size: base.reservoir_size(),
                                output_size: base.output_size(),
                                leak_rate: base.leak_rate(),
                                activation: Default::default(),
                            };
                            if let Some(parent) = std::path::Path::new(&path).parent() {
                                std::fs::create_dir_all(parent).ok();
                            }
                            std::fs::write(
                                &path,
                                serde_json::to_string(&weights).unwrap_or_default(),
                            )
                            .is_ok()
                        } else {
                            false
                        };
                        // Auto-save Nautilus shell alongside ESN weights
                        let shell_path = path.replace(".bin", ".nautilus.json")
                            .replace(".json", ".nautilus.json");
                        if let Ok(json) = nautilus_brain.to_json() {
                            let _ = std::fs::write(&shell_path, json);
                        }
                        resp_tx
                            .send(NpuResponse::WeightsSaved {
                                path: if saved { path } else { String::new() },
                            })
                            .ok();
                    }

                    NpuRequest::CgResidual(update) => {
                        residual_history.push((update.iteration, update.rz_new));
                        if residual_history.len() > 50 {
                            residual_history.drain(..residual_history.len() - 50);
                        }

                        // Regime-aware thresholds: near phase transition, CG is expected
                        // to be harder so raise the anomaly bar.
                        let transition_proximity = ((update.beta - KNOWN_BETA_C_SU3_NT4).abs() / 0.2).clamp(0.0, 1.0);
                        let regime_scale = 1.0 + 0.5 * (1.0 - transition_proximity);

                        let anomaly_score = if let Some(ref mut npu) = multi_npu {
                            let window: Vec<f64> = residual_history
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
                        } else if residual_history.len() >= 3 {
                            let recent: Vec<f64> = residual_history
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

                        // regime_scale raises thresholds near the phase transition
                        // where CG difficulty is physically expected
                        let yellow_thresh = 0.3 * regime_scale;
                        let red_thresh = 0.7 * regime_scale;

                        match attention_state {
                            AttentionState::Green => {
                                if anomaly_score > red_thresh {
                                    attention_state = AttentionState::Red;
                                    yellow_count = 0;
                                    green_count = 0;
                                    if residual_history.len() >= 3 {
                                        let recent: Vec<f64> = residual_history
                                            .iter()
                                            .rev()
                                            .take(3)
                                            .map(|(_, rz)| *rz)
                                            .collect();
                                        if recent.windows(2).all(|w| w[0] >= w[1]) {
                                            if let Some(ref itx) = interrupt_tx {
                                                let _ = itx.send(BrainInterrupt::KillCg);
                                            }
                                        }
                                    }
                                } else if anomaly_score > yellow_thresh {
                                    attention_state = AttentionState::Yellow;
                                    green_count = 0;
                                    if let Some(ref itx) = interrupt_tx {
                                        let _ = itx.send(BrainInterrupt::AdjustCheckInterval(20));
                                    }
                                }
                            }
                            AttentionState::Yellow => {
                                if anomaly_score > red_thresh {
                                    yellow_count += 1;
                                    if yellow_count >= 2 {
                                        attention_state = AttentionState::Red;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ =
                                                itx.send(BrainInterrupt::AdjustCheckInterval(5));
                                        }
                                    }
                                } else if anomaly_score < yellow_thresh {
                                    green_count += 1;
                                    if green_count >= 3 {
                                        attention_state = AttentionState::Green;
                                        green_count = 0;
                                        yellow_count = 0;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ =
                                                itx.send(BrainInterrupt::AdjustCheckInterval(100));
                                        }
                                    }
                                } else {
                                    green_count = 0;
                                }
                            }
                            AttentionState::Red => {
                                if residual_history.len() >= 3 {
                                    let recent: Vec<f64> = residual_history
                                        .iter()
                                        .rev()
                                        .take(3)
                                        .map(|(_, rz)| *rz)
                                        .collect();
                                    let diverging = recent.windows(2).all(|w| w[0] >= w[1]);
                                    if diverging {
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ = itx.send(BrainInterrupt::KillCg);
                                        }
                                    }
                                }
                                if anomaly_score < 0.3 {
                                    green_count += 1;
                                    if green_count >= 3 {
                                        attention_state = AttentionState::Yellow;
                                        green_count = 0;
                                        yellow_count = 0;
                                        if let Some(ref itx) = interrupt_tx {
                                            let _ =
                                                itx.send(BrainInterrupt::AdjustCheckInterval(20));
                                        }
                                    }
                                } else {
                                    green_count = 0;
                                }
                            }
                        }
                        resp_tx.send(NpuResponse::ResidualAck).ok();
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
                    } => {
                        latest_proxy = Some(ProxyFeatures {
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

                        // Feed spectral + Potts features to Nautilus
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
                        nautilus_brain.observe(spectral_obs);
                    }

                    NpuRequest::DisagreementQuery {
                        beta,
                        plaq,
                        mass,
                        chi,
                        acceptance,
                    } => {
                        let dis = if let Some(ref mut npu) = multi_npu {
                            let input = canonical_seq(beta, plaq, mass, chi, acceptance, lattice);
                            let (outputs, d) = npu.predict_with_disagreement(&input);
                            let _ = outputs;
                            d
                        } else {
                            HeadGroupDisagreement::default()
                        };
                        resp_tx
                            .send(NpuResponse::DisagreementSnapshot {
                                delta_cg: dis.delta_cg,
                                delta_phase: dis.delta_phase,
                                delta_anomaly: dis.delta_anomaly,
                                delta_priority: dis.delta_priority,
                                urgency: dis.urgency(),
                            })
                            .ok();
                    }

                    NpuRequest::ExportNautilusShell { path } => {
                        let saved = match nautilus_brain.to_json() {
                            Ok(json) => {
                                if let Some(parent) = std::path::Path::new(&path).parent() {
                                    std::fs::create_dir_all(parent).ok();
                                }
                                std::fs::write(&path, json).is_ok()
                            }
                            Err(e) => {
                                eprintln!("  [Nautilus] Failed to serialize shell: {e}");
                                false
                            }
                        };
                        resp_tx.send(NpuResponse::NautilusShellSaved {
                            path: if saved { path } else { String::new() },
                        }).ok();
                    }

                    NpuRequest::BootstrapNautilusShell { path } => {
                        match std::fs::read_to_string(&path) {
                            Ok(json) => match NautilusBrain::from_json(&json) {
                                Ok(brain) => {
                                    let n_obs = brain.observations.len();
                                    let n_gen = brain.shell.history.len();
                                    nautilus_brain = brain;
                                    eprintln!("  [Nautilus] Loaded shell: {n_obs} obs, {n_gen} generations");
                                    resp_tx.send(NpuResponse::NautilusShellLoaded {
                                        n_observations: n_obs,
                                        n_generations: n_gen,
                                    }).ok();
                                }
                                Err(e) => {
                                    eprintln!("  [Nautilus] Failed to parse shell {path}: {e}");
                                    resp_tx.send(NpuResponse::NautilusShellLoaded {
                                        n_observations: 0,
                                        n_generations: 0,
                                    }).ok();
                                }
                            },
                            Err(e) => {
                                eprintln!("  [Nautilus] Cannot read {path}: {e}");
                                resp_tx.send(NpuResponse::NautilusShellLoaded {
                                    n_observations: 0,
                                    n_generations: 0,
                                }).ok();
                            }
                        }
                    }

                    // ─── Full-stream trajectory events ───
                    NpuRequest::TrajectoryEvent(evt) => {
                        traj_batch.push(evt);
                        if traj_batch.len() >= 8 {
                            let n = traj_batch.len();
                            for evt in traj_batch.drain(..) {
                                sub_models.observe_event(&evt, latest_proxy.as_ref());
                            }
                            resp_tx
                                .send(NpuResponse::TrajectoryBatchProcessed { n_events: n })
                                .ok();
                        }
                    }

                    NpuRequest::FlushTrajectoryBatch => {
                        if !traj_batch.is_empty() {
                            let n = traj_batch.len();
                            for evt in traj_batch.drain(..) {
                                sub_models.observe_event(&evt, latest_proxy.as_ref());
                            }
                            eprintln!(
                                "  [NPU] Sub-models: flushed {n} events — {}",
                                sub_models.status_line()
                            );
                            resp_tx
                                .send(NpuResponse::TrajectoryBatchProcessed { n_events: n })
                                .ok();
                        } else {
                            resp_tx
                                .send(NpuResponse::TrajectoryBatchProcessed { n_events: 0 })
                                .ok();
                        }
                    }

                    NpuRequest::SubModelMetrics => {
                        resp_tx
                            .send(NpuResponse::SubModelMetricsSnapshot(
                                sub_models.metrics_json(),
                            ))
                            .ok();
                    }

                    NpuRequest::SubModelPredict(evt) => {
                        let predictions = sub_models.predict_all(&evt, latest_proxy.as_ref());
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
                        resp_tx
                            .send(NpuResponse::SubModelPredictions {
                                cg_cost,
                                steering,
                                phase,
                                trajectory,
                            })
                            .ok();
                    }

                    NpuRequest::Shutdown => break,
                }
            }
        })
        .expect("NPU worker thread spawn");

    NpuWorkerHandles {
        npu_tx: req_tx,
        npu_rx: resp_rx,
        interrupt_rx: interrupt_rx_out,
    }
}

fn estimate_beta_c(results: &[BetaResult]) -> f64 {
    if results.is_empty() {
        return KNOWN_BETA_C_SU3_NT4;
    }
    let max_chi_r = results
        .iter()
        .max_by(|a, b| a.susceptibility.total_cmp(&b.susceptibility));
    max_chi_r.map_or(KNOWN_BETA_C_SU3_NT4, |r| r.beta)
}

fn find_largest_gaps(measured: &[f64], min: f64, max: f64, n: usize) -> Vec<f64> {
    let mut sorted = measured.to_vec();
    sorted.push(min);
    sorted.push(max);
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();

    let mut gaps: Vec<(f64, f64)> = sorted
        .windows(2)
        .map(|w| (w[1] - w[0], f64::midpoint(w[0], w[1])))
        .collect();
    gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
    gaps.iter().take(n).map(|g| g.1).collect()
}

fn build_training_data(results: &[BetaResult], lattice: usize) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let mut seqs = Vec::new();
    let mut targets = Vec::new();
    let seq_len = 10;

    for r in results {
        // Continuous phase: use Polyakov loop magnitude as order parameter
        let phase_continuous = r.polyakov.abs().clamp(0.0, 1.0);
        // Continuous delta_H proxy: negative = accepted, positive = rejected
        let delta_h_continuous = (-(r.acceptance - 0.5) * 4.0) / 10.0;

        let proximity = (-(r.beta - KNOWN_BETA_C_SU3_NT4).powi(2) / 0.1).exp();
        let anomaly_val = f64::from(r.acceptance < 0.05 || r.mean_cg_iters > 3000.0);
        let quality = (r.acceptance * 0.4
            + (1.0 - r.std_plaq / r.mean_plaq.abs().max(1e-10)).clamp(0.0, 1.0) * 0.3
            + (r.n_traj as f64 / 1000.0).min(1.0) * 0.3)
            .clamp(0.0, 1.0);
        let cg_norm = r.mean_cg_iters / 500.0;
        let quenched_therm_target = f64::from(r.npu_quenched_early_exit);

        // Anderson/Potts phases remain categorical for their respective proxy heads
        let anderson_phase = if r.beta > 5.5 {
            1.0
        } else if r.beta < 5.0 {
            0.0
        } else {
            0.5
        };
        let potts_phase = if r.beta > 5.8 {
            1.0
        } else if r.beta < 5.2 {
            0.0
        } else {
            0.5
        };
        let target_acc = 0.70;
        let acc_error = r.acceptance - target_acc;
        let optimal_dt = (r.dt_used * (1.0 - 0.5 * acc_error)).clamp(0.002, 0.02);
        let optimal_nmd = ((1.0 / optimal_dt).round() / 200.0).clamp(0.0, 1.0);

        let seq: Vec<Vec<f64>> = (0..seq_len)
            .map(|j| {
                let noise = 0.005 * ((j as f64) * 0.7).sin();
                canonical_input(
                    r.beta,
                    r.mean_plaq + noise * r.std_plaq,
                    r.mass,
                    r.susceptibility,
                    r.acceptance,
                    lattice,
                )
            })
            .collect();
        seqs.push(seq);

        let mut t = vec![0.0; heads::NUM_HEADS];

        t[heads::A0_ANDERSON_CG_COST] = cg_norm;
        t[heads::A1_ANDERSON_PHASE] = anderson_phase;
        t[heads::A2_ANDERSON_LAMBDA_MIN] = (1.0 / (cg_norm + 0.01)).clamp(0.0, 1.0);
        t[heads::A3_ANDERSON_ANOMALY] = anomaly_val;
        t[heads::A4_ANDERSON_THERM] = quenched_therm_target;
        t[heads::A5_ANDERSON_PRIORITY] = proximity;

        t[heads::B0_QCD_CG_COST] = cg_norm;
        t[heads::B1_QCD_PHASE] = phase_continuous;
        t[heads::B2_QCD_ACCEPTANCE] = delta_h_continuous;
        t[heads::B3_QCD_ANOMALY] = anomaly_val;
        t[heads::B4_QCD_THERM] = if r.acceptance > 0.3 { 1.0 } else { 0.0 };
        t[heads::B5_QCD_PRIORITY] = proximity;

        t[heads::C0_POTTS_CG_COST] = cg_norm;
        t[heads::C1_POTTS_PHASE] = potts_phase;
        t[heads::C2_POTTS_BETA_C] = (r.beta - KNOWN_BETA_C_SU3_NT4).abs().min(1.0);
        t[heads::C3_POTTS_ANOMALY] = anomaly_val;
        t[heads::C4_POTTS_ORDER] = r.polyakov.abs().clamp(0.0, 1.0);
        t[heads::C5_POTTS_PRIORITY] = proximity;

        t[heads::D0_NEXT_BETA] = proximity;
        t[heads::D1_OPTIMAL_DT] = optimal_dt;
        t[heads::D2_OPTIMAL_NMD] = optimal_nmd;
        t[heads::D3_CHECK_INTERVAL] = if cg_norm > 0.5 { 0.2 } else { 0.8 };
        t[heads::D4_KILL_DECISION] = if r.mean_cg_iters > 400.0 { 0.8 } else { 0.1 };
        t[heads::D5_SKIP_DECISION] = if quality < 0.2 { 0.8 } else { 0.1 };

        t[heads::E0_RESIDUAL_ETA] = cg_norm;
        t[heads::E1_RESIDUAL_ANOMALY] = anomaly_val;
        t[heads::E2_CONVERGENCE_RATE] = (1.0 - cg_norm).clamp(0.0, 1.0);
        t[heads::E3_STALL_DETECTOR] = if r.mean_cg_iters > 300.0 { 0.5 } else { 0.0 };
        t[heads::E4_DIVERGENCE_DETECTOR] = if anomaly_val > 0.5 { 0.8 } else { 0.0 };
        t[heads::E5_QUALITY_FORECAST] = quality;

        t[heads::M0_CG_CONSENSUS] = cg_norm;
        t[heads::M1_PHASE_CONSENSUS] = phase_continuous;
        t[heads::M2_CG_UNCERTAINTY] = (anderson_phase - phase_continuous).abs();
        t[heads::M3_PHASE_UNCERTAINTY] = (potts_phase - phase_continuous).abs();
        let proxy_agrees = (anderson_phase - phase_continuous).abs() < 0.3;
        t[heads::M4_PROXY_TRUST] = if proxy_agrees { 0.8 } else { 0.3 };
        t[heads::M5_ATTENTION_LEVEL] = match () {
            () if anomaly_val > 0.5 => 0.8,
            () if cg_norm > 0.5 => 0.4,
            () => 0.1,
        };

        targets.push(t);
    }

    (seqs, targets)
}
