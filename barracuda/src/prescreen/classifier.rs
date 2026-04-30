// SPDX-License-Identifier: AGPL-3.0-or-later

/// Simple logistic regression classifier for parameter pre-screening.
///
/// Trained on accumulated L1/L2 data to predict whether a parameter set
/// will produce a reasonable L2 χ². Designed for NPU deployment
/// (small, fixed-size model), with CPU fallback.
///
/// Architecture: 10 inputs → normalize → linear(10→1) → sigmoid → P(promising)
#[derive(Debug, Clone)]
#[must_use]
pub struct PreScreenClassifier {
    /// Weights \[10\] — one per Skyrme parameter.
    pub weights: Vec<f64>,
    /// Bias term.
    pub bias: f64,
    /// Normalization: (mean, std) per feature.
    pub norm: Vec<(f64, f64)>,
    /// Decision threshold for P(promising).
    pub threshold: f64,
    /// Number of training examples.
    pub n_train: usize,
    /// Number of positive (promising) training examples.
    pub n_positive: usize,
}

impl PreScreenClassifier {
    /// Train classifier on accumulated evaluation data.
    ///
    /// Training data: (params, `f_value`) pairs from previous runs.
    /// Labels: `f_value` &lt; `label_threshold` → positive (promising)
    pub fn train(xs: &[Vec<f64>], ys: &[f64], label_threshold: f64) -> Self {
        let n = xs.len();
        let dim = if n > 0 { xs[0].len() } else { 10 };

        let labels: Vec<f64> = ys
            .iter()
            .map(|&y| if y < label_threshold { 1.0 } else { 0.0 })
            .collect();

        let n_positive = labels.iter().filter(|&&l| l > 0.5).count();

        let mut means = vec![0.0; dim];
        let mut stds = vec![1.0; dim];

        if n > 0 {
            for i in 0..dim {
                let vals: Vec<f64> = xs.iter().map(|x| x[i]).collect();
                means[i] = vals.iter().sum::<f64>() / n as f64;
                let var = vals.iter().map(|v| (v - means[i]).powi(2)).sum::<f64>() / n as f64;
                stds[i] = var.sqrt().max(crate::tolerances::CLASSIFIER_VARIANCE_GUARD);
            }
        }

        let norm: Vec<(f64, f64)> = means
            .iter()
            .zip(stds.iter())
            .map(|(&m, &s)| (m, s))
            .collect();

        let mut weights = vec![0.0; dim];
        let mut bias = 0.0;
        let lr = crate::tolerances::CLASSIFIER_LEARNING_RATE;
        let epochs = crate::tolerances::CLASSIFIER_EPOCHS;

        for _epoch in 0..epochs {
            let mut dw = vec![0.0; dim];
            let mut db = 0.0;

            for (x, &label) in xs.iter().zip(labels.iter()) {
                let x_norm: Vec<f64> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (v - norm[i].0) / norm[i].1)
                    .collect();

                let z: f64 = x_norm
                    .iter()
                    .zip(weights.iter())
                    .map(|(&xi, &wi)| xi * wi)
                    .sum::<f64>()
                    + bias;
                let p = 1.0 / (1.0 + (-z).exp());

                let err = p - label;
                for i in 0..dim {
                    dw[i] += err * x_norm[i];
                }
                db += err;
            }

            let scale = lr / n.max(1) as f64;
            for i in 0..dim {
                weights[i] -= scale * dw[i];
            }
            bias -= scale * db;
        }

        Self {
            weights,
            bias,
            norm,
            threshold: 0.5,
            n_train: n,
            n_positive,
        }
    }

    /// Predict probability of being "promising"
    #[must_use]
    pub fn predict_prob(&self, params: &[f64]) -> f64 {
        let x_norm: Vec<f64> = params
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.norm[i].0) / self.norm[i].1)
            .collect();

        let z: f64 = x_norm
            .iter()
            .zip(self.weights.iter())
            .map(|(&xi, &wi)| xi * wi)
            .sum::<f64>()
            + self.bias;

        1.0 / (1.0 + (-z).exp())
    }

    /// Binary prediction: is this parameter set promising?
    #[must_use]
    pub fn predict(&self, params: &[f64]) -> bool {
        self.predict_prob(params) > self.threshold
    }

    /// Get model weights for NPU deployment
    #[must_use]
    pub fn export_weights(&self) -> (Vec<f64>, f64, Vec<(f64, f64)>) {
        (self.weights.clone(), self.bias, self.norm.clone())
    }
}
