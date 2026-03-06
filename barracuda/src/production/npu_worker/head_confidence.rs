// SPDX-License-Identifier: AGPL-3.0-only

#![allow(missing_docs)]

//! Rolling per-head confidence tracker for NPU steering decisions.

const HEAD_CONFIDENCE_WINDOW: usize = 20;
const REGRESSION_TRUST_THRESHOLD: f64 = 0.3;
const CLASSIFICATION_TRUST_THRESHOLD: f64 = 0.6;

/// Whether a head predicts a continuous value or a discrete class.
///
/// Evolution note (v0.6.17): non-binary (Regression) heads learn better than
/// Classification heads on the ESN. The ESN output is continuous — forcing it
/// through a 0.5 threshold loses gradient signal and makes confidence opaque.
/// Prefer `Regression` for all heads and use soft thresholds (e.g. > 0.7 for
/// high-confidence decisions) on the steering side. Classification is kept for
/// backward compat but new heads should use Regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadMetric {
    /// R² for continuous targets — preferred for all heads (captures gradient).
    Regression,
    /// Accuracy for binary targets — kept for backward compatibility with saved weights.
    #[allow(dead_code)]
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
pub(crate) struct HeadConfidence {
    predictions: Vec<Vec<f64>>,
    actuals: Vec<Vec<f64>>,
    trusted: Vec<bool>,
    r2: Vec<f64>,
    metrics: Vec<HeadMetric>,
}

impl HeadConfidence {
    pub(crate) fn new(n_heads: usize) -> Self {
        let metrics = vec![HeadMetric::Regression; n_heads];
        Self {
            predictions: vec![Vec::with_capacity(HEAD_CONFIDENCE_WINDOW); n_heads],
            actuals: vec![Vec::with_capacity(HEAD_CONFIDENCE_WINDOW); n_heads],
            trusted: vec![false; n_heads],
            r2: vec![0.0; n_heads],
            metrics,
        }
    }

    pub(crate) fn record_prediction(&mut self, head: usize, predicted: f64) {
        if head < self.predictions.len() {
            let buf = &mut self.predictions[head];
            if buf.len() >= HEAD_CONFIDENCE_WINDOW {
                buf.remove(0);
            }
            buf.push(predicted);
        }
    }

    pub(crate) fn record_actual(&mut self, head: usize, actual: f64) {
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
                self.trusted[head] = r2 >= REGRESSION_TRUST_THRESHOLD;
            }
            HeadMetric::Classification => {
                let correct = pred[offset..]
                    .iter()
                    .zip(&actual[a_offset..])
                    .filter(|(&p, &a)| (p > 0.5) == (a > 0.5))
                    .count();
                let accuracy = correct as f64 / n as f64;
                self.r2[head] = accuracy;
                self.trusted[head] = accuracy >= CLASSIFICATION_TRUST_THRESHOLD;
            }
        }
    }

    pub(crate) fn is_trusted(&self, head: usize) -> bool {
        head < self.trusted.len() && self.trusted[head]
    }

    pub(crate) fn status_line(&self) -> String {
        let trusted_count = self.trusted.iter().filter(|&&t| t).count();
        let total = self.trusted.len();
        let best: Vec<String> = self
            .r2
            .iter()
            .enumerate()
            .filter(|(_, &r)| r > 0.1)
            .map(|(i, r)| {
                let tag = match self.metrics.get(i) {
                    Some(HeadMetric::Classification) => "acc",
                    _ => "R²",
                };
                format!("H{i}={r:.2}{tag}")
            })
            .collect();
        format!(
            "{trusted_count}/{total} trusted [{}]",
            if best.is_empty() {
                "none above 0.1".into()
            } else {
                best.join(", ")
            }
        )
    }
}
