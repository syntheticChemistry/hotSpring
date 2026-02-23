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

/// ESN configuration matching `ToadStool`'s `barracuda::esn_v2::ESNConfig`.
#[derive(Debug, Clone)]
pub struct EsnConfig {
    pub input_size: usize,
    pub reservoir_size: usize,
    pub output_size: usize,
    pub spectral_radius: f64,
    pub connectivity: f64,
    pub leak_rate: f64,
    pub regularization: f64,
    pub seed: u64,
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

        for (i, s) in self.state.iter_mut().enumerate() {
            *s = (1.0 - alpha).mul_add(*s, alpha * pre[i].tanh());
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

// ═══════════════════════════════════════════════════════════════════
//  Weight export (for NPU/GPU substrate transfer)
// ═══════════════════════════════════════════════════════════════════

/// Exported ESN weights for cross-substrate deployment (NPU, GPU).
///
/// All weights are f32 to match the Akida `load_reservoir` API and
/// `ToadStool`'s f32 tensor convention.
#[derive(Debug, Clone)]
pub struct ExportedWeights {
    /// Input weights, flattened row-major: (`reservoir_size` × `input_size`)
    pub w_in: Vec<f32>,
    /// Reservoir weights, flattened row-major: (`reservoir_size` × `reservoir_size`)
    pub w_res: Vec<f32>,
    /// Readout weights, flattened row-major: (`output_size` × `reservoir_size`)
    pub w_out: Vec<f32>,
    pub input_size: usize,
    pub reservoir_size: usize,
    pub output_size: usize,
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
}

/// Solve AX = B for multiple right-hand sides via Gauss-Jordan (partial pivoting, f64).
///
/// CPU-only: ESN ridge regression solves small systems (reservoir_size × reservoir_size,
/// typically 50-200). Each column of B is solved independently.
fn solve_linear_system(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();

    let a_flat: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();

    let mut x = vec![vec![0.0; m]; n];
    for col in 0..m {
        let b_col: Vec<f64> = (0..n).map(|row| b[row][col]).collect();
        if let Some(sol) = gauss_jordan_solve(&a_flat, &b_col, n) {
            for (row, &val) in sol.iter().enumerate() {
                x[row][col] = val;
            }
        }
    }
    x
}

/// Gauss-Jordan elimination with partial pivoting for Ax = b.
///
/// Returns None if the system is singular (pivot below epsilon).
fn gauss_jordan_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for k in 0..n {
        let mut max_row = k;
        let mut max_val = aug[k * (n + 1) + k].abs();
        for i in (k + 1)..n {
            let v = aug[i * (n + 1) + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        if max_row != k {
            for j in 0..=(n) {
                aug.swap(k * (n + 1) + j, max_row * (n + 1) + j);
            }
        }
        let pivot = aug[k * (n + 1) + k];
        for j in k..=(n) {
            aug[k * (n + 1) + j] /= pivot;
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = aug[i * (n + 1) + k];
            for j in k..=(n) {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }

    Some((0..n).map(|i| aug[i * (n + 1) + n]).collect())
}

fn spectral_radius_estimate(w: &[Vec<f64>]) -> f64 {
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
// ═══════════════════════════════════════════════════════════════════

struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    fn new(seed: u64) -> Self {
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

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(crate::tolerances::DIVISION_GUARD);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn esn_trains_and_predicts() {
        let config = EsnConfig {
            input_size: 2,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.9,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
        };
        let mut esn = EchoStateNetwork::new(config);

        let seq1: Vec<Vec<f64>> = (0..100)
            .map(|t| {
                let x = f64::from(t) * 0.1;
                vec![x.sin(), x.cos()]
            })
            .collect();
        let seq2: Vec<Vec<f64>> = (0..100)
            .map(|t| {
                let x = f64::from(t) * 0.2;
                vec![x.sin(), x.cos()]
            })
            .collect();

        esn.train(&[seq1.clone(), seq2.clone()], &[vec![1.0], vec![2.0]]);

        let p1 = esn.predict(&seq1).expect("ESN trained");
        let p2 = esn.predict(&seq2).expect("ESN trained");

        assert!((p1[0] - 1.0).abs() < 0.5, "p1={}", p1[0]);
        assert!((p2[0] - 2.0).abs() < 0.5, "p2={}", p2[0]);
    }

    #[test]
    fn esn_default_config() {
        let config = EsnConfig::default();
        assert_eq!(config.reservoir_size, 50);
        assert_eq!(config.input_size, 8);
        assert!((config.spectral_radius - 0.95).abs() < 1e-10);
    }

    #[test]
    fn velocity_features_correct_shape() {
        let n = 10;
        let frame = vec![0.1; n * 3];
        let features = velocity_features(&[frame], n, 2.0, 100.0);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].len(), 8);
    }

    #[test]
    fn export_and_npu_parity() {
        let config = EsnConfig {
            input_size: 2,
            reservoir_size: 30,
            output_size: 1,
            spectral_radius: 0.9,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-4,
            seed: 42,
        };
        let mut esn = EchoStateNetwork::new(config);

        let seq1: Vec<Vec<f64>> = (0..50)
            .map(|t| {
                let x = f64::from(t) * 0.1;
                vec![x.sin(), x.cos()]
            })
            .collect();
        let seq2: Vec<Vec<f64>> = (0..50)
            .map(|t| {
                let x = f64::from(t) * 0.2;
                vec![x.sin(), x.cos()]
            })
            .collect();

        let sequences = vec![seq1, seq2];
        esn.train(&sequences, &[vec![1.0], vec![2.0]]);

        let exported = esn.export_weights().expect("ESN export_weights");
        assert_eq!(exported.w_in.len(), 30 * 2);
        assert_eq!(exported.w_res.len(), 30 * 30);
        assert_eq!(exported.w_out.len(), 30);

        let mut npu = NpuSimulator::from_exported(&exported);
        let cpu_pred = esn.predict(&sequences[0]).expect("ESN trained")[0];
        let npu_pred = npu.predict(&sequences[0])[0];
        let diff = (cpu_pred - npu_pred).abs() / cpu_pred.abs().max(1e-10);
        assert!(diff < 0.01, "CPU/NPU diff {diff} should be < 1%");
    }

    #[test]
    fn esn_benchmark_vs_python() {
        use std::time::Instant;

        let n_features = 8;
        let n_frames = 500;
        let n_cases = 6;

        let config = EsnConfig {
            input_size: n_features,
            reservoir_size: 50,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-2,
            seed: 42,
        };

        let mut rng = Xoshiro256pp::new(99);
        let sequences: Vec<Vec<Vec<f64>>> = (0..n_cases)
            .map(|_| {
                (0..n_frames)
                    .map(|_| (0..n_features).map(|_| rng.standard_normal()).collect())
                    .collect()
            })
            .collect();
        let targets: Vec<Vec<f64>> = (0..n_cases).map(|_| vec![rng.standard_normal()]).collect();

        let n_reps = if cfg!(debug_assertions) { 3 } else { 100 };
        let t0 = Instant::now();
        for _ in 0..n_reps {
            let mut esn = EchoStateNetwork::new(config.clone());
            esn.train(&sequences[..4], &targets[..4]);
            for seq in &sequences {
                let _ = esn.predict(seq).expect("ESN trained");
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_iter_ms = elapsed * 1000.0 / f64::from(n_reps);

        println!("Rust ESN: {n_reps} reps in {elapsed:.3}s");
        println!("  Per iteration (train+6 predict): {per_iter_ms:.1} ms");
        if !cfg!(debug_assertions) {
            assert!(
                per_iter_ms < 50.0,
                "ESN should complete in <50ms per iteration (release)"
            );
        }
    }

    #[test]
    fn spectral_radius_identity() {
        let n = 5;
        let w: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let sr = spectral_radius_estimate(&w);
        assert!((sr - 1.0).abs() < 0.01, "sr={sr}");
    }

    #[test]
    fn velocity_features_known_values() {
        let vels = vec![vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]; // 2 particles
        let feats = velocity_features(&vels, 2, 1.5, 100.0);
        assert_eq!(feats.len(), 1);
        let f = &feats[0];
        assert_eq!(f.len(), 8);
        assert!((f[0] - 0.5).abs() < 1e-10, "mean_vx: {}", f[0]); // (1+0)/2
        assert!((f[1] - 0.5).abs() < 1e-10, "mean_vy: {}", f[1]); // (0+1)/2
        assert!((f[2] - 0.0).abs() < 1e-10, "mean_vz: {}", f[2]); // (0+0)/2
        assert!((f[3] - 1.0).abs() < 1e-10, "mean_speed: {}", f[3]); // (1+1)/2
        assert!((f[4] - 0.5).abs() < 1e-10, "ke_per_particle: {}", f[4]); // (0.5+0.5)/2
        let v_rms_expected = 1.0; // sqrt((1+1)/2) = 1.0
        assert!((f[5] - v_rms_expected).abs() < 1e-10, "v_rms: {}", f[5]);
        let kappa_scaled = 1.5 / 3.0;
        assert!(
            (f[6] - kappa_scaled).abs() < 1e-10,
            "kappa_scaled: {}",
            f[6]
        );
        let gamma_scaled = 100.0_f64.log10() / 3.0;
        assert!(
            (f[7] - gamma_scaled).abs() < 1e-10,
            "gamma_scaled: {}",
            f[7]
        );
    }

    #[test]
    fn velocity_features_multiple_frames() {
        let frame1 = vec![3.0, 0.0, 0.0]; // 1 particle, speed=3
        let frame2 = vec![0.0, 4.0, 0.0]; // 1 particle, speed=4
        let feats = velocity_features(&[frame1, frame2], 1, 1.0, 1.0);
        assert_eq!(feats.len(), 2);
        assert!((feats[0][3] - 3.0).abs() < 1e-10, "frame1 speed");
        assert!((feats[1][3] - 4.0).abs() < 1e-10, "frame2 speed");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn npu_predict_return_state_consistent() {
        let config = EsnConfig {
            input_size: 3,
            reservoir_size: 15,
            output_size: 1,
            spectral_radius: 0.9,
            connectivity: 0.3,
            leak_rate: 0.3,
            regularization: 1e-2,
            seed: 77,
        };
        let seq: Vec<Vec<f64>> = (0..5)
            .map(|i| vec![f64::from(i) * 0.1, 0.5, -0.2])
            .collect();
        let mut esn = EchoStateNetwork::new(config);
        esn.train(std::slice::from_ref(&seq), &[vec![1.0]]);
        let exported = esn.export_weights().expect("export");

        let mut npu1 = NpuSimulator::from_exported(&exported);
        let mut npu2 = NpuSimulator::from_exported(&exported);

        let pred = npu1.predict(&seq);
        let state = npu2.predict_return_state(&seq);

        assert_eq!(state.len(), exported.reservoir_size);
        let readout_from_state: f64 = exported
            .w_out
            .chunks(exported.reservoir_size)
            .next()
            .expect("w_out row")
            .iter()
            .zip(state.iter())
            .map(|(&w, &s)| f64::from(w) * f64::from(s))
            .sum();
        assert!(
            (pred[0] - readout_from_state).abs() < 1e-4,
            "predict vs state readout: {} vs {}",
            pred[0],
            readout_from_state
        );
    }

    #[test]
    fn esn_predict_determinism() {
        let config = EsnConfig {
            input_size: 3,
            reservoir_size: 20,
            output_size: 1,
            spectral_radius: 0.95,
            connectivity: 0.2,
            leak_rate: 0.3,
            regularization: 1e-2,
            seed: 42,
        };
        let seqs: Vec<Vec<Vec<f64>>> = vec![
            (0..10)
                .map(|i| vec![f64::from(i) * 0.1, 0.5, -0.3])
                .collect(),
            (0..10)
                .map(|i| vec![0.0, f64::from(i) * 0.05, 0.2])
                .collect(),
        ];
        let targets = vec![vec![1.0], vec![0.0]];

        let results: Vec<Vec<f64>> = (0..2)
            .map(|_| {
                let mut esn = EchoStateNetwork::new(config.clone());
                esn.train(&seqs, &targets);
                esn.predict(&seqs[0]).expect("ESN trained")
            })
            .collect();
        assert!(
            (results[0][0] - results[1][0]).abs() < f64::EPSILON,
            "ESN predictions must be identical: {} vs {}",
            results[0][0],
            results[1][0]
        );
    }
}
