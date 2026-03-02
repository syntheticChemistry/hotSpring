// SPDX-License-Identifier: AGPL-3.0-only

//! NPU simulator and weight export for cross-substrate deployment.

use super::heads::HeadGroupDisagreement;
use super::EchoStateNetwork;

/// Exported ESN weights for cross-substrate deployment (NPU, GPU).
///
/// All weights are f32 to match the Akida `load_reservoir` API and
/// `ToadStool`'s f32 tensor convention.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExportedWeights {
    /// Input weights, flattened row-major: (`reservoir_size` × `input_size`)
    pub w_in: Vec<f32>,
    /// Reservoir weights, flattened row-major: (`reservoir_size` × `reservoir_size`)
    pub w_res: Vec<f32>,
    /// Readout weights, flattened row-major: (`output_size` × `reservoir_size`)
    pub w_out: Vec<f32>,
    /// Input dimensionality.
    pub input_size: usize,
    /// Reservoir dimensionality.
    pub reservoir_size: usize,
    /// Output dimensionality.
    pub output_size: usize,
    /// Leak rate for reservoir state update.
    pub leak_rate: f32,
}

impl EchoStateNetwork {
    /// Export trained weights as flat f32 arrays for NPU/GPU deployment.
    ///
    /// The Akida driver expects `load_reservoir(w_in: &[f32], w_res: &[f32])`
    /// with flattened row-major layout. The readout `W_out` is applied on the
    /// host CPU after NPU inference returns the reservoir state.
    #[must_use]
    pub fn export_weights(&self) -> Option<ExportedWeights> {
        let w_out_2d = self.w_out.as_ref()?;
        let rs = self.config.reservoir_size;
        let is = self.config.input_size;
        let os = self.config.output_size;

        let w_in_flat: Vec<f32> = self
            .w_in
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();

        let w_res_flat: Vec<f32> = self
            .w_res
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();

        let w_out_flat: Vec<f32> = w_out_2d
            .iter()
            .flat_map(|row| row.iter().map(|&v| v as f32))
            .collect();

        Some(ExportedWeights {
            w_in: w_in_flat,
            w_res: w_res_flat,
            w_out: w_out_flat,
            input_size: is,
            reservoir_size: rs,
            output_size: os,
            leak_rate: self.config.leak_rate as f32,
        })
    }
}

/// Simulated NPU inference using exported f32 weights.
///
/// Mirrors what the Akida hardware does: process input through the reservoir
/// using f32 arithmetic. This allows validating NPU parity before hardware
/// is available, and serves as the software reference for the NPU path.
pub struct NpuSimulator {
    w_in: Vec<Vec<f32>>,
    w_res: Vec<Vec<f32>>,
    w_out: Vec<Vec<f32>>,
    state: Vec<f32>,
    reservoir_size: usize,
    leak_rate: f32,
}

impl NpuSimulator {
    /// Create NPU simulator from exported weights.
    #[must_use]
    pub fn from_exported(weights: &ExportedWeights) -> Self {
        let rs = weights.reservoir_size;
        let is = weights.input_size;
        let os = weights.output_size;

        let w_in: Vec<Vec<f32>> = (0..rs)
            .map(|i| weights.w_in[i * is..(i + 1) * is].to_vec())
            .collect();
        let w_res: Vec<Vec<f32>> = (0..rs)
            .map(|i| weights.w_res[i * rs..(i + 1) * rs].to_vec())
            .collect();
        let w_out: Vec<Vec<f32>> = (0..os)
            .map(|i| weights.w_out[i * rs..(i + 1) * rs].to_vec())
            .collect();

        Self {
            w_in,
            w_res,
            w_out,
            state: vec![0.0; rs],
            reservoir_size: rs,
            leak_rate: weights.leak_rate,
        }
    }

    /// Process an input sequence and return prediction (f32 arithmetic).
    pub fn predict(&mut self, input_sequence: &[Vec<f64>]) -> Vec<f64> {
        self.state.fill(0.0);

        for input in input_sequence {
            let mut pre = vec![0.0f32; self.reservoir_size];
            for (i, pre_i) in pre.iter_mut().enumerate() {
                let mut val = 0.0f32;
                for (j, &u) in input.iter().enumerate() {
                    val += self.w_in[i][j] * u as f32;
                }
                for j in 0..self.reservoir_size {
                    val += self.w_res[i][j] * self.state[j];
                }
                *pre_i = val;
            }
            for (i, s) in self.state.iter_mut().enumerate() {
                *s = (1.0 - self.leak_rate).mul_add(*s, self.leak_rate * pre[i].tanh());
            }
        }

        self.w_out
            .iter()
            .map(|row| {
                let sum: f32 = row.iter().zip(self.state.iter()).map(|(w, s)| w * s).sum();
                f64::from(sum)
            })
            .collect()
    }

    /// Flat row-major input weights for serialization.
    #[must_use]
    pub fn export_w_in(&self) -> Vec<f32> {
        self.w_in
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Flat row-major reservoir weights for serialization.
    #[must_use]
    pub fn export_w_res(&self) -> Vec<f32> {
        self.w_res
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Flat row-major readout weights for serialization.
    #[must_use]
    pub fn export_w_out(&self) -> Vec<f32> {
        self.w_out
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Input dimensionality.
    #[must_use]
    pub fn input_size(&self) -> usize {
        if self.w_in.is_empty() {
            0
        } else {
            self.w_in[0].len()
        }
    }

    /// Reservoir dimensionality.
    #[must_use]
    pub fn reservoir_size(&self) -> usize {
        self.reservoir_size
    }

    /// Output dimensionality.
    #[must_use]
    pub fn output_size(&self) -> usize {
        self.w_out.len()
    }

    /// Leak rate.
    #[must_use]
    pub fn leak_rate(&self) -> f32 {
        self.leak_rate
    }

    /// Process input and return the raw reservoir state (before readout).
    pub fn predict_return_state(&mut self, input_sequence: &[Vec<f64>]) -> Vec<f32> {
        self.state.fill(0.0);
        for input in input_sequence {
            let mut pre = vec![0.0f32; self.reservoir_size];
            for (i, pre_i) in pre.iter_mut().enumerate() {
                let mut val = 0.0f32;
                for (j, &u) in input.iter().enumerate() {
                    val += self.w_in[i][j] * u as f32;
                }
                for j in 0..self.reservoir_size {
                    val += self.w_res[i][j] * self.state[j];
                }
                *pre_i = val;
            }
            for (i, s) in self.state.iter_mut().enumerate() {
                *s = (1.0 - self.leak_rate).mul_add(*s, self.leak_rate * pre[i].tanh());
            }
        }
        self.state.clone()
    }

    /// Swap the readout weight matrix at runtime (~14 ms on Akida via `set_variable()`).
    ///
    /// The reservoir weights stay fixed; only the readout layer changes. This enables
    /// regime-specific readout heads (confined/transition/deconfined) while sharing
    /// the same reservoir dynamics.
    pub fn set_readout_weights(&mut self, w_out_flat: &[f32]) {
        let os = w_out_flat.len() / self.reservoir_size;
        self.w_out = (0..os)
            .map(|i| w_out_flat[i * self.reservoir_size..(i + 1) * self.reservoir_size].to_vec())
            .collect();
    }

    /// Apply a specific readout head to the current reservoir state.
    ///
    /// Returns the scalar output for a single head (row index into w_out).
    /// Useful for multi-head ESN where different heads serve different purposes.
    #[must_use]
    pub fn readout_head(&self, head_idx: usize) -> f64 {
        if head_idx >= self.w_out.len() {
            return 0.0;
        }
        let sum: f32 = self.w_out[head_idx]
            .iter()
            .zip(self.state.iter())
            .map(|(w, s)| w * s)
            .sum();
        f64::from(sum)
    }
}

/// Multi-head NPU simulator wrapping a single reservoir with N readout heads.
///
/// The reservoir state is computed once per input sequence; each head applies
/// its own readout weights to produce independent outputs. Weight swapping
/// via `set_regime_weights()` takes ~14 ms on Akida hardware.
///
/// Gen 2: supports 36 overlapping heads with cross-group disagreement.
/// Backward-compatible with Gen 1 (15-head) saved weights — missing
/// heads are zero-padded on load.
pub struct MultiHeadNpu {
    base: NpuSimulator,
    regime_weight_sets: Vec<ExportedWeights>,
    current_regime: usize,
}

impl MultiHeadNpu {
    /// Create from exported weights with `output_size == NUM_HEADS`.
    #[must_use]
    pub fn from_exported(weights: &ExportedWeights) -> Self {
        Self {
            base: NpuSimulator::from_exported(weights),
            regime_weight_sets: Vec::new(),
            current_regime: 0,
        }
    }

    /// Register a set of readout weights for a specific physics regime.
    ///
    /// Regime 0 = confined, 1 = transition, 2 = deconfined. Each set must
    /// have the same reservoir_size but may have different output_size.
    pub fn register_regime_weights(&mut self, weights: ExportedWeights) {
        self.regime_weight_sets.push(weights);
    }

    /// Swap readout weights to a specific regime (e.g., when phase changes).
    /// Returns `true` if the swap was performed.
    pub fn set_regime(&mut self, regime_idx: usize) -> bool {
        if regime_idx >= self.regime_weight_sets.len() || regime_idx == self.current_regime {
            return false;
        }
        let w = &self.regime_weight_sets[regime_idx];
        self.base.set_readout_weights(&w.w_out);
        self.current_regime = regime_idx;
        true
    }

    /// Run the reservoir on an input sequence, then return all head outputs.
    ///
    /// One reservoir pass, N readout values. Gen 2: 36 heads. Gen 1: 15 heads.
    pub fn predict_all_heads(&mut self, input_sequence: &[Vec<f64>]) -> Vec<f64> {
        self.base.predict(input_sequence)
    }

    /// Run all heads and compute cross-group disagreement (Gen 2).
    ///
    /// Returns `(outputs, disagreement)`. The disagreement signal measures
    /// epistemic uncertainty across physics-informed head groups.
    /// For Gen 1 models (< 36 heads), disagreement is zero.
    pub fn predict_with_disagreement(
        &mut self,
        input_sequence: &[Vec<f64>],
    ) -> (Vec<f64>, HeadGroupDisagreement) {
        let outputs = self.base.predict(input_sequence);
        let disagreement = HeadGroupDisagreement::from_outputs(&outputs);
        (outputs, disagreement)
    }

    /// Run reservoir and return a single head's output.
    pub fn predict_head(&mut self, input_sequence: &[Vec<f64>], head_idx: usize) -> f64 {
        let _ = self.base.predict_return_state(input_sequence);
        self.base.readout_head(head_idx)
    }

    /// Access the underlying single-output predictor.
    pub fn base_mut(&mut self) -> &mut NpuSimulator {
        &mut self.base
    }

    /// Current regime index.
    #[must_use]
    pub fn current_regime(&self) -> usize {
        self.current_regime
    }
}
