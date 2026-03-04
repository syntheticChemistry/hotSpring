// SPDX-License-Identifier: AGPL-3.0-only

//! Echo State Network for MD Transport Prediction
//!
//! Predicts transport coefficients (D*) from short velocity trajectory
//! segments using reservoir computing. Matches the Python reference
//! implementation in `control/reservoir_transport/scripts/reservoir_vacf.py`.
//!
//! # Architecture
//!
//! ```text
//! velocity snapshots → feature extraction → ESN reservoir → readout → D*
//! ```
//!
//! The reservoir captures temporal correlations in the velocity stream
//! (exponential VACF decay, caging oscillations) and the readout layer
//! maps final reservoir state to transport coefficients.
//!
//! # References
//!
//! - Jaeger (2001) "The echo state approach to recurrent neural networks"
//! - Stanton & Murillo, PRE 93, 043203 (2016) — transport coefficients

/// Head indices for the multi-head ESN (Gen 2 "Developed Organism" layout).
pub mod heads;
/// NPU simulator and weight export for cross-substrate deployment.
pub mod npu;

#[cfg(test)]
mod tests;

pub use heads::HeadGroupDisagreement;
pub use npu::{ExportedWeights, MultiHeadNpu, NpuSimulator};

/// Reservoir activation function.
///
/// `Tanh` is the classic ESN activation — smooth, bounded [-1, 1].
/// `ReluTanhApprox` is a 5-segment piecewise-linear approximation of tanh
/// built entirely from ReLU operations. On the AKD1000 this compiles to a
/// 2-layer FC chain (10 hidden ReLU neurons + linear output) that merges
/// into the reservoir update pass at ~3 µs extra latency (Discovery 2).
/// Same system, same weights — hardware maps the math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    /// Classic smooth activation, bounded [-1, 1].
    Tanh,
    /// Piecewise-linear ReLU approximation of tanh — deployable on AKD1000.
    ReluTanhApprox,
}

impl Default for Activation {
    fn default() -> Self {
        Self::Tanh
    }
}

/// 5-segment piecewise-linear approximation of tanh using only ReLU-
/// representable operations. Breakpoints match tanh to <0.5% max error.
///
/// On the AKD1000, this is a 10-neuron bounded-ReLU hidden layer + linear
/// output: the FC chain merges into one hardware pass.
#[inline]
pub fn relu_tanh_approx_f64(x: f64) -> f64 {
    let ax = x.abs();
    let y = if ax < 0.5 {
        0.924_24 * ax
    } else if ax < 1.0 {
        0.462_12_f64.mul_add(1.0, 0.598_94 * (ax - 0.5))
    } else if ax < 1.5 {
        0.761_59_f64.mul_add(1.0, 0.287_12 * (ax - 1.0))
    } else if ax < 2.0 {
        0.905_15_f64.mul_add(1.0, 0.117_76 * (ax - 1.5))
    } else if ax < 3.0 {
        0.964_03_f64.mul_add(1.0, 0.031_02 * (ax - 2.0))
    } else {
        0.995_05
    };
    y.copysign(x)
}

/// f32 variant for the NPU simulator path.
#[inline]
pub fn relu_tanh_approx_f32(x: f32) -> f32 {
    let ax = x.abs();
    let y = if ax < 0.5 {
        0.924_24_f32 * ax
    } else if ax < 1.0 {
        0.462_12_f32.mul_add(1.0, 0.598_94 * (ax - 0.5))
    } else if ax < 1.5 {
        0.761_59_f32.mul_add(1.0, 0.287_12 * (ax - 1.0))
    } else if ax < 2.0 {
        0.905_15_f32.mul_add(1.0, 0.117_76 * (ax - 1.5))
    } else if ax < 3.0 {
        0.964_03_f32.mul_add(1.0, 0.031_02 * (ax - 2.0))
    } else {
        0.995_05
    };
    y.copysign(x)
}

/// ESN configuration matching `ToadStool`'s `barracuda::esn_v2::ESNConfig`.
#[derive(Debug, Clone)]
pub struct EsnConfig {
    /// Input feature dimensionality.
    pub input_size: usize,
    /// Reservoir state dimensionality.
    pub reservoir_size: usize,
    /// Output (readout) dimensionality.
    pub output_size: usize,
    /// Target spectral radius for reservoir matrix.
    pub spectral_radius: f64,
    /// Sparse connectivity (fraction of non-zero weights).
    pub connectivity: f64,
    /// Leak rate (state update blending).
    pub leak_rate: f64,
    /// Ridge regression regularization.
    pub regularization: f64,
    /// PRNG seed for reproducible init.
    pub seed: u64,
    /// Reservoir activation function.
    pub activation: Activation,
}

impl Default for EsnConfig {
    fn default() -> Self {
        Self {
            input_size: 8,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: crate::tolerances::ESN_REGULARIZATION,
            seed: 42,
            activation: Activation::default(),
        }
    }
}

/// Pure-CPU Echo State Network (f64 precision).
///
/// Matches the Python `NumPy` ESN implementation exactly for cross-validation.
/// Uses the same PRNG seeding strategy for reproducible weight initialization.
pub struct EchoStateNetwork {
    config: EsnConfig,
    w_in: Vec<Vec<f64>>,
    w_res: Vec<Vec<f64>>,
    w_out: Option<Vec<Vec<f64>>>,
    state: Vec<f64>,
}

impl EchoStateNetwork {
    /// Create a new ESN with random weights initialized from seed.
    #[must_use]
    pub fn new(config: EsnConfig) -> Self {
        let rs = config.reservoir_size;
        let is = config.input_size;

        let mut rng = Xoshiro256pp::new(config.seed);

        // W_in: uniform [-0.5, 0.5]
        let w_in: Vec<Vec<f64>> = (0..rs)
            .map(|_| (0..is).map(|_| rng.uniform() - 0.5).collect())
            .collect();

        // W_res: sparse random, scaled to spectral_radius
        let mut w_res: Vec<Vec<f64>> = (0..rs)
            .map(|_| {
                (0..rs)
                    .map(|_| {
                        if rng.uniform() < config.connectivity {
                            rng.standard_normal()
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        let sr = spectral_radius_estimate(&w_res);
        if sr > crate::tolerances::ESN_SPECTRAL_RADIUS_NEGLIGIBLE {
            let scale = config.spectral_radius / sr;
            for row in &mut w_res {
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
        }

        Self {
            config,
            w_in,
            w_res,
            w_out: None,
            state: vec![0.0; rs],
        }
    }

    fn reset_state(&mut self) {
        self.state.fill(0.0);
    }

    fn update(&mut self, input: &[f64]) {
        let rs = self.config.reservoir_size;
        let alpha = self.config.leak_rate;
        let activation = self.config.activation;
        let mut pre = vec![0.0; rs];

        for (i, pre_i) in pre.iter_mut().enumerate().take(rs) {
            let mut val = 0.0;
            for (j, &u) in input.iter().enumerate() {
                val += self.w_in[i][j] * u;
            }
            for j in 0..rs {
                val += self.w_res[i][j] * self.state[j];
            }
            *pre_i = val;
        }

        let activate: fn(f64) -> f64 = match activation {
            Activation::Tanh => f64::tanh,
            Activation::ReluTanhApprox => relu_tanh_approx_f64,
        };
        for (i, s) in self.state.iter_mut().enumerate() {
            *s = (1.0 - alpha).mul_add(*s, alpha * activate(pre[i]));
        }
    }

    fn collect_states(&mut self, input_sequence: &[Vec<f64>]) -> Vec<Vec<f64>> {
        input_sequence
            .iter()
            .map(|input| {
                self.update(input);
                self.state.clone()
            })
            .collect()
    }

    /// Train readout via ridge regression.
    ///
    /// Each input sequence maps to one target vector (the final reservoir
    /// state is used as features).
    pub fn train(&mut self, input_sequences: &[Vec<Vec<f64>>], targets: &[Vec<f64>]) {
        let rs = self.config.reservoir_size;
        let os = self.config.output_size;
        let n = input_sequences.len();

        let mut x_mat = vec![vec![0.0; rs]; n];
        for (i, seq) in input_sequences.iter().enumerate() {
            self.reset_state();
            let states = self.collect_states(seq);
            let state = states
                .last()
                .map_or_else(|| self.state.as_slice(), Vec::as_slice);
            x_mat[i].clone_from_slice(state);
        }

        // Ridge regression: W_out = Y^T X (X^T X + lambda I)^{-1}
        // Compute X^T X + lambda I
        let mut xtx = vec![vec![0.0; rs]; rs];
        for i in 0..rs {
            for j in 0..rs {
                let sum: f64 = x_mat.iter().take(n).map(|row| row[i] * row[j]).sum();
                xtx[i][j] = sum;
            }
            xtx[i][i] += self.config.regularization;
        }

        // Compute X^T Y
        let mut xty = vec![vec![0.0; os]; rs];
        for i in 0..rs {
            for j in 0..os {
                xty[i][j] = x_mat
                    .iter()
                    .zip(targets)
                    .take(n)
                    .map(|(row, t)| row[i] * t[j])
                    .sum();
            }
        }

        // Solve via Cholesky (or LU fallback)
        let w_out_t = solve_linear_system(&xtx, &xty);
        let mut w_out = vec![vec![0.0; rs]; os];
        for i in 0..os {
            for j in 0..rs {
                w_out[i][j] = w_out_t[j][i];
            }
        }
        self.w_out = Some(w_out);
    }

    /// Incrementally retrain the readout by appending new data to a rolling buffer.
    ///
    /// Re-solves the ridge regression on the combined (old + new) dataset.
    /// The buffer caps at `max_buffer` samples; oldest are evicted first.
    pub fn train_incremental(
        &mut self,
        new_sequences: &[Vec<Vec<f64>>],
        new_targets: &[Vec<f64>],
        buffer: &mut IncrementalBuffer,
    ) {
        for (seq, tgt) in new_sequences.iter().zip(new_targets) {
            buffer.push(seq.clone(), tgt.clone());
        }
        let (all_seqs, all_tgts) = buffer.as_training_pair();
        if !all_seqs.is_empty() {
            self.train(&all_seqs, &all_tgts);
        }
    }

    /// Predict transport coefficients from a velocity feature sequence.
    ///
    /// # Errors
    /// Returns `Err` if the ESN has not been trained (no `w_out`).
    pub fn predict(
        &mut self,
        input_sequence: &[Vec<f64>],
    ) -> Result<Vec<f64>, crate::error::HotSpringError> {
        self.reset_state();
        let states = self.collect_states(input_sequence);
        let final_state = states
            .last()
            .map_or_else(|| self.state.as_slice(), Vec::as_slice);
        let w_out = self.w_out.as_ref().ok_or_else(|| {
            crate::error::HotSpringError::InvalidOperation("ESN not trained".into())
        })?;
        Ok(w_out
            .iter()
            .map(|row| row.iter().zip(final_state.iter()).map(|(w, s)| w * s).sum())
            .collect())
    }
}

/// Rolling buffer for incremental ESN training.
///
/// Stores sequence/target pairs up to `max_size`, evicting the oldest when full.
/// Shared across retrain cycles so the ESN sees accumulated experience.
pub struct IncrementalBuffer {
    sequences: Vec<Vec<Vec<f64>>>,
    targets: Vec<Vec<f64>>,
    max_size: usize,
}

impl IncrementalBuffer {
    /// Create a new buffer with the given capacity.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            sequences: Vec::with_capacity(max_size),
            targets: Vec::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a sample, evicting the oldest if at capacity.
    pub fn push(&mut self, seq: Vec<Vec<f64>>, target: Vec<f64>) {
        if self.sequences.len() >= self.max_size {
            self.sequences.remove(0);
            self.targets.remove(0);
        }
        self.sequences.push(seq);
        self.targets.push(target);
    }

    /// Current number of buffered samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Borrow the full dataset for training.
    #[must_use]
    pub fn as_training_pair(&self) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        (self.sequences.clone(), self.targets.clone())
    }
}

/// Extract per-frame features from velocity trajectory.
///
/// From each frame of (N*3) flat velocities, computes:
/// `[mean_vx, mean_vy, mean_vz, mean_speed, ke_per_particle, v_rms, kappa_scaled, gamma_scaled]`
///
/// Physics parameters (κ, Γ) are included as constant features to allow
/// the ESN to generalize across the phase diagram.
#[must_use]
pub fn velocity_features(
    vel_snapshots: &[Vec<f64>],
    n_particles: usize,
    kappa: f64,
    gamma: f64,
) -> Vec<Vec<f64>> {
    let kappa_scaled = kappa / 3.0;
    let gamma_scaled = gamma.log10() / 3.0;

    vel_snapshots
        .iter()
        .map(|frame| {
            let n = n_particles;
            let mut sum_vx = 0.0;
            let mut sum_vy = 0.0;
            let mut sum_vz = 0.0;
            let mut sum_speed = 0.0;
            let mut sum_v2 = 0.0;
            let mut sum_ke = 0.0;

            for i in 0..n {
                let vx = frame[3 * i];
                let vy = frame[3 * i + 1];
                let vz = frame[3 * i + 2];
                let v2 = vz.mul_add(vz, vx.mul_add(vx, vy * vy));
                let speed = v2.sqrt();
                sum_vx += vx;
                sum_vy += vy;
                sum_vz += vz;
                sum_speed += speed;
                sum_v2 += v2;
                sum_ke += 0.5 * v2;
            }

            let nf = n as f64;
            vec![
                sum_vx / nf,
                sum_vy / nf,
                sum_vz / nf,
                sum_speed / nf,
                sum_ke / nf,
                (sum_v2 / nf).sqrt(),
                kappa_scaled,
                gamma_scaled,
            ]
        })
        .collect()
}

/// Solve AX = B for multiple right-hand sides via LU decomposition (partial pivoting, f64).
///
/// Delegates to `barracuda::ops::linalg::lu_solve` — the shared primitive for dense
/// linear solves. ESN ridge regression produces small systems (reservoir_size × reservoir_size,
/// typically 50–200); each column of B is solved independently.
fn solve_linear_system(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();

    let a_flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();

    let mut x = vec![vec![0.0; m]; n];
    for col in 0..m {
        let b_col: Vec<f64> = (0..n).map(|row| b[row][col]).collect();
        if let Ok(sol) = barracuda::ops::linalg::lu_solve(&a_flat, n, &b_col) {
            for (row, val) in sol.iter().enumerate() {
                x[row][col] = *val;
            }
        }
    }
    x
}

pub(crate) fn spectral_radius_estimate(w: &[Vec<f64>]) -> f64 {
    let n = w.len();
    if n == 0 {
        return 0.0;
    }

    // Power iteration for dominant eigenvalue magnitude
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut lambda = 0.0;

    for _ in 0..100 {
        let mut w_v = vec![0.0; n];
        for (i, w_v_i) in w_v.iter_mut().enumerate() {
            *w_v_i = w[i].iter().zip(v.iter()).map(|(wij, vj)| wij * vj).sum();
        }
        let norm: f64 = w_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < crate::tolerances::DIVISION_GUARD {
            return 0.0;
        }
        lambda = norm;
        for (vi, w_vi) in v.iter_mut().zip(w_v.iter()) {
            *vi = w_vi / norm;
        }
    }
    lambda
}

// ═══════════════════════════════════════════════════════════════════
//  PRNG (xoshiro256++ for reproducible weight init)
// ═══════════════════════════════════════════════════════════════

pub(crate) struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    pub(crate) fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for slot in &mut s {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    const fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(crate::tolerances::DIVISION_GUARD);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
