// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sub-model registry for specialized NPU learning.
//!
//! Instead of one 36-head ESN seeing summary data, multiple specialized
//! ESN sub-models each see the full trajectory event stream, filtered to
//! their domain. Each model maintains its own reservoir, rolling training
//! buffer, and per-head confidence tracker.

use crate::md::reservoir::{Activation, EchoStateNetwork, EsnConfig, IncrementalBuffer};
use crate::production::npu_worker::trajectory_input_with_proxy;
use crate::production::{TrajectoryEvent, TrajectoryPhase};
use crate::proxy::ProxyFeatures;

/// Per-head metric type for confidence tracking.
///
/// Evolution note (v0.6.17): non-binary (Regression) heads learn better than
/// Classification heads. Prefer Regression for all heads; use soft thresholds
/// on the steering side for decision outputs.
#[derive(Clone, Copy, Debug)]
pub enum HeadMetric {
    /// Continuous-valued output — trust threshold based on R² (preferred for all heads).
    Regression,
    /// Binary output — deprecated, prefer Regression with soft threshold.
    Classification,
}

/// Rolling confidence tracker for sub-model outputs.
pub struct SubModelConfidence {
    predictions: Vec<Vec<f64>>,
    actuals: Vec<Vec<f64>>,
    metrics: Vec<HeadMetric>,
    r2: Vec<f64>,
    trusted: Vec<bool>,
    window: usize,
}

const DEFAULT_WINDOW: usize = 20;
const REGRESSION_TRUST: f64 = 0.3;
const CLASSIFICATION_TRUST: f64 = 0.6;

impl SubModelConfidence {
    /// Create a confidence tracker for `n_heads` output heads.
    #[must_use]
    pub fn new(n_heads: usize, metrics: Vec<HeadMetric>) -> Self {
        Self {
            predictions: vec![Vec::with_capacity(DEFAULT_WINDOW); n_heads],
            actuals: vec![Vec::with_capacity(DEFAULT_WINDOW); n_heads],
            metrics,
            r2: vec![0.0; n_heads],
            trusted: vec![false; n_heads],
            window: DEFAULT_WINDOW,
        }
    }

    /// Record a prediction/actual pair for rolling confidence computation.
    pub fn record(&mut self, head: usize, predicted: f64, actual: f64) {
        if head >= self.predictions.len() {
            return;
        }
        let pred_buf = &mut self.predictions[head];
        if pred_buf.len() >= self.window {
            pred_buf.remove(0);
        }
        pred_buf.push(predicted);

        let act_buf = &mut self.actuals[head];
        if act_buf.len() >= self.window {
            act_buf.remove(0);
        }
        act_buf.push(actual);

        self.recompute(head);
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
                let ss_tot: f64 = actual[a_offset..]
                    .iter()
                    .map(|a| (a - mean_a).powi(2))
                    .sum();
                let ss_res: f64 = pred[offset..]
                    .iter()
                    .zip(&actual[a_offset..])
                    .map(|(p, a)| (a - p).powi(2))
                    .sum();
                let r2 = if ss_tot > 1e-15 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                };
                self.r2[head] = r2;
                self.trusted[head] = r2 >= REGRESSION_TRUST;
            }
            HeadMetric::Classification => {
                let correct = pred[offset..]
                    .iter()
                    .zip(&actual[a_offset..])
                    .filter(|&(&p, &a)| (p > 0.5) == (a > 0.5))
                    .count();
                let accuracy = correct as f64 / n as f64;
                self.r2[head] = accuracy;
                self.trusted[head] = accuracy >= CLASSIFICATION_TRUST;
            }
        }
    }

    /// Whether the given head has crossed its trust threshold.
    #[must_use]
    pub fn is_trusted(&self, head: usize) -> bool {
        head < self.trusted.len() && self.trusted[head]
    }

    /// Whether any head has crossed its trust threshold.
    #[must_use]
    pub fn any_trusted(&self) -> bool {
        self.trusted.iter().any(|&t| t)
    }

    /// Human-readable trust status (e.g. "2/5 [0=0.85R², 3=0.72acc]").
    #[must_use]
    pub fn status_line(&self) -> String {
        let trusted_count = self.trusted.iter().filter(|&&t| t).count();
        let total = self.trusted.len();
        let details: Vec<String> = self
            .r2
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r > 0.1)
            .map(|(i, r)| {
                let tag = match self.metrics.get(i) {
                    Some(HeadMetric::Classification) => "acc",
                    _ => "R²",
                };
                format!("{i}={r:.2}{tag}")
            })
            .collect();
        format!(
            "{trusted_count}/{total} [{}]",
            if details.is_empty() {
                "none>0.1".into()
            } else {
                details.join(", ")
            }
        )
    }
}

/// Configuration for a single sub-model.
pub struct SubModelConfig {
    /// Human-readable model name (e.g. "`acceptance_predictor`").
    pub name: &'static str,
    /// ESN reservoir dimensionality.
    pub reservoir_size: usize,
    /// Reservoir connectivity fraction (0.0–1.0).
    pub connectivity: f64,
    /// Names of the output heads (parallel to `head_metrics`).
    pub output_names: Vec<&'static str>,
    /// Metric type for each output head.
    pub head_metrics: Vec<HeadMetric>,
    /// Maximum number of samples in the rolling training buffer.
    pub buffer_capacity: usize,
    /// Minimum samples before first training pass.
    pub min_samples_to_train: usize,
    /// Phase filter — returns true if this model should ingest events from the given phase.
    pub accepts_phase: fn(TrajectoryPhase) -> bool,
}

/// A single sub-model: ESN + training buffer + confidence tracker.
pub struct SubModel {
    /// Immutable configuration for this sub-model.
    pub config: SubModelConfig,
    /// Echo state network reservoir.
    pub esn: EchoStateNetwork,
    /// Rolling training data buffer.
    pub buffer: IncrementalBuffer,
    /// Per-head confidence tracker (R² / accuracy).
    pub confidence: SubModelConfidence,
    /// Number of training passes completed.
    pub n_trained: usize,
    /// MSE from the most recent training pass.
    pub last_mse: f64,
}

impl SubModel {
    /// Create a sub-model from config with a deterministic PRNG seed.
    #[must_use]
    pub fn new(config: SubModelConfig, seed: u64) -> Self {
        let n_outputs = config.output_names.len();
        let esn = EchoStateNetwork::new(EsnConfig {
            input_size: super::npu_worker::TRAJECTORY_INPUT_DIM,
            reservoir_size: config.reservoir_size,
            output_size: n_outputs,
            connectivity: config.connectivity,
            regularization: 1e-3,
            seed,
            activation: Activation::ReluTanhApprox,
            ..EsnConfig::default()
        });
        let buffer = IncrementalBuffer::new(config.buffer_capacity);
        let confidence = SubModelConfidence::new(n_outputs, config.head_metrics.clone());
        Self {
            config,
            esn,
            buffer,
            confidence,
            n_trained: 0,
            last_mse: f64::NAN,
        }
    }

    /// Whether this model accepts events from the given trajectory phase.
    #[must_use]
    pub fn accepts(&self, phase: TrajectoryPhase) -> bool {
        (self.config.accepts_phase)(phase)
    }

    /// Add a training sample and retrain if buffer has enough data.
    pub fn observe(&mut self, input: Vec<f64>, target: Vec<f64>) {
        let seq = vec![input; 5];
        self.buffer.push(seq, target);

        if self.buffer.len() >= self.config.min_samples_to_train {
            let (seqs, tgts) = self.buffer.as_training_pair();
            self.esn.train(&seqs, &tgts);
            self.n_trained = self.buffer.len();
        }
    }

    /// Predict from a trajectory input vector.
    pub fn predict(&mut self, input: &[f64]) -> Option<Vec<f64>> {
        if self.n_trained == 0 {
            return None;
        }
        let seq = vec![input.to_vec(); 5];
        self.esn.predict(&seq).ok()
    }
}

/// Registry managing all sub-models.
pub struct SubModelRegistry {
    /// All registered sub-models.
    pub models: Vec<SubModel>,
}

impl SubModelRegistry {
    /// Create the default set of 4 sub-models.
    #[must_use]
    pub fn default_models() -> Self {
        let models = vec![
            // Model A: Trajectory Predictor — fires every trajectory
            SubModel::new(
                SubModelConfig {
                    name: "trajectory_predictor",
                    reservoir_size: 200,
                    connectivity: 0.3,
                    output_names: vec!["next_delta_h", "next_plaq", "therm_progress"],
                    head_metrics: vec![
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                    ],
                    buffer_capacity: 2000,
                    min_samples_to_train: 30,
                    accepts_phase: |_| true,
                },
                100,
            ),
            // Model B: Phase Oracle — fires on measurement trajectories
            SubModel::new(
                SubModelConfig {
                    name: "phase_oracle",
                    reservoir_size: 200,
                    connectivity: 0.3,
                    output_names: vec!["phase_continuous", "polyakov_mag", "susceptibility"],
                    head_metrics: vec![
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                    ],
                    buffer_capacity: 1000,
                    min_samples_to_train: 15,
                    accepts_phase: |p| p == TrajectoryPhase::Measurement,
                },
                200,
            ),
            // Model C: CG Cost Predictor — fires on dynamical trajectories
            SubModel::new(
                SubModelConfig {
                    name: "cg_cost_predictor",
                    reservoir_size: 150,
                    connectivity: 0.25,
                    output_names: vec!["cg_iterations", "stall_probability"],
                    head_metrics: vec![HeadMetric::Regression, HeadMetric::Regression],
                    buffer_capacity: 1500,
                    min_samples_to_train: 20,
                    accepts_phase: |p| {
                        p == TrajectoryPhase::Therm || p == TrajectoryPhase::Measurement
                    },
                },
                300,
            ),
            // Model D: Steering Brain — fires on measurement trajectories,
            // uses predictions from A/B/C as additional features
            SubModel::new(
                SubModelConfig {
                    name: "steering_brain",
                    reservoir_size: 200,
                    connectivity: 0.3,
                    output_names: vec![
                        "next_beta_priority",
                        "optimal_dt",
                        "optimal_n_md",
                        "saturation_score",
                        "skip_decision",
                    ],
                    head_metrics: vec![
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                        HeadMetric::Regression,
                    ],
                    buffer_capacity: 500,
                    min_samples_to_train: 10,
                    accepts_phase: |p| p == TrajectoryPhase::Measurement,
                },
                400,
            ),
        ];
        Self { models }
    }

    /// Process a trajectory event through all applicable sub-models.
    /// Proxy features are appended to the input vector when available.
    /// Returns the number of models that accepted the event.
    pub fn observe_event(&mut self, evt: &TrajectoryEvent, proxy: Option<&ProxyFeatures>) -> usize {
        let input = trajectory_input_with_proxy(evt, proxy);
        let mut n_accepted = 0;

        for model in &mut self.models {
            if !model.accepts(evt.phase_tag) {
                continue;
            }

            let target = build_target(model.config.name, evt);
            model.observe(input.clone(), target);
            n_accepted += 1;
        }
        n_accepted
    }

    /// Predict from all sub-models for a given event.
    pub fn predict_all(
        &mut self,
        evt: &TrajectoryEvent,
        proxy: Option<&ProxyFeatures>,
    ) -> Vec<(&'static str, Option<Vec<f64>>)> {
        let input = trajectory_input_with_proxy(evt, proxy);
        self.models
            .iter_mut()
            .map(|m| (m.config.name, m.predict(&input)))
            .collect()
    }

    /// Status line for all sub-models.
    #[must_use]
    pub fn status_line(&self) -> String {
        self.models
            .iter()
            .map(|m| {
                format!(
                    "{}(n={},{})",
                    m.config.name.chars().next().unwrap_or('?').to_uppercase(),
                    m.n_trained,
                    m.confidence.status_line()
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Structured per-sub-model metrics for logging/experimentation.
    /// Returns a JSON-serializable map of `model_name` → { `n_trained`, `last_mse`, heads: [...] }.
    #[must_use]
    pub fn metrics_json(&self) -> serde_json::Value {
        let models: serde_json::Map<String, serde_json::Value> = self
            .models
            .iter()
            .map(|m| {
                let heads: Vec<serde_json::Value> = m
                    .config
                    .output_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let metric_type = match m.confidence.metrics.get(i) {
                            Some(HeadMetric::Classification) => "accuracy",
                            _ => "r2",
                        };
                        serde_json::json!({
                            "name": name,
                            "metric_type": metric_type,
                            "score": m.confidence.r2.get(i).copied().unwrap_or(0.0),
                            "trusted": m.confidence.is_trusted(i),
                        })
                    })
                    .collect();
                (
                    m.config.name.to_string(),
                    serde_json::json!({
                        "n_trained": m.n_trained,
                        "last_mse": m.last_mse,
                        "heads": heads,
                    }),
                )
            })
            .collect();
        serde_json::Value::Object(models)
    }
}

/// Build target vector for a sub-model from a trajectory event.
fn build_target(model_name: &str, evt: &TrajectoryEvent) -> Vec<f64> {
    match model_name {
        "trajectory_predictor" => vec![
            evt.delta_h.clamp(-5.0, 5.0) / 5.0,
            evt.plaquette,
            if evt.phase_tag == TrajectoryPhase::Measurement {
                1.0
            } else {
                evt.traj_idx as f64 / 60.0
            },
        ],
        "phase_oracle" => vec![
            if evt.plaquette > 0.57 {
                1.0
            } else if evt.plaquette > 0.50 {
                0.5
            } else {
                0.0
            },
            evt.polyakov_re.abs().clamp(0.0, 1.0),
            evt.plaquette_var.clamp(0.0, 0.1) * 10.0,
        ],
        "cg_cost_predictor" => vec![
            evt.cg_iterations as f64 / 100_000.0,
            if evt.cg_iterations > 50_000 { 1.0 } else { 0.0 },
        ],
        "steering_brain" => vec![
            (evt.beta - 5.0) / 2.0,
            0.02,
            50.0 / 150.0,
            0.5,
            if evt.accepted { 0.0 } else { 1.0 },
        ],
        _ => vec![0.0],
    }
}
