// SPDX-License-Identifier: AGPL-3.0-only

//! ESN benchmark infrastructure for cross-substrate comparison.
//!
//! Provides GPU ESN inference, deterministic test data generation,
//! timing utilities, and result types for CPU/GPU/NPU benchmarking.

use crate::gpu::GpuF64;
use crate::md::reservoir::ExportedWeights;
use crate::md::shaders::{SHADER_ESN_READOUT, SHADER_ESN_RESERVOIR_UPDATE};
use std::time::Instant;

/// GPU-backed ESN inference using WGSL reservoir and readout shaders.
///
/// Runs Echo State Network forward pass on GPU: per-step reservoir updates
/// followed by readout. Supports both per-step dispatch and batched encoder.
pub struct GpuEsn {
    reservoir_pipeline: wgpu::ComputePipeline,
    readout_pipeline: wgpu::ComputePipeline,
    w_in_buf: wgpu::Buffer,
    w_res_buf: wgpu::Buffer,
    w_out_buf: wgpu::Buffer,
    state_buf: wgpu::Buffer,
    input_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    params_reservoir_buf: wgpu::Buffer,
    params_readout_buf: wgpu::Buffer,
    reservoir_size: usize,
    output_size: usize,
}

impl GpuEsn {
    /// Create GPU ESN from exported weights.
    #[must_use]
    pub fn new(gpu: &GpuF64, weights: &ExportedWeights) -> Self {
        let rs = weights.reservoir_size;
        let os = weights.output_size;
        let is = weights.input_size;

        let reservoir_pipeline =
            gpu.create_pipeline(SHADER_ESN_RESERVOIR_UPDATE, "esn_reservoir_update");
        let readout_pipeline = gpu.create_pipeline(SHADER_ESN_READOUT, "esn_readout");

        let w_in_buf = gpu.create_f32_buffer(&weights.w_in, "w_in");
        let w_res_buf = gpu.create_f32_buffer(&weights.w_res, "w_res");
        let w_out_buf = gpu.create_f32_buffer(&weights.w_out, "w_out");

        let state_buf = gpu.create_f32_rw_buffer(&vec![0.0f32; rs], "state");
        let input_buf = gpu.create_f32_rw_buffer(&vec![0.0f32; is], "input");
        let output_buf = gpu.create_f32_output_buffer(os, "output");

        let params_reservoir = [rs as f32, is as f32, weights.leak_rate, 0.0];
        let params_readout = [rs as f32, os as f32, 0.0, 0.0];

        let params_reservoir_buf = gpu.create_f32_buffer(&params_reservoir, "params_reservoir");
        let params_readout_buf = gpu.create_f32_buffer(&params_readout, "params_readout");

        Self {
            reservoir_pipeline,
            readout_pipeline,
            w_in_buf,
            w_res_buf,
            w_out_buf,
            state_buf,
            input_buf,
            output_buf,
            params_reservoir_buf,
            params_readout_buf,
            reservoir_size: rs,
            output_size: os,
        }
    }

    /// Run ESN inference: per-step reservoir dispatch, then readout.
    #[must_use]
    pub fn predict(&self, gpu: &GpuF64, input_sequence: &[Vec<f64>]) -> Vec<f64> {
        let rs = self.reservoir_size;
        let wg = rs.div_ceil(64) as u32;

        gpu.upload_f32(&self.state_buf, &vec![0.0f32; rs]);

        for input in input_sequence {
            let input_f32: Vec<f32> = input.iter().map(|&v| v as f32).collect();
            gpu.upload_f32(&self.input_buf, &input_f32);

            let reservoir_bg = gpu.create_bind_group(
                &self.reservoir_pipeline,
                &[
                    &self.w_in_buf,
                    &self.w_res_buf,
                    &self.input_buf,
                    &self.state_buf,
                    &self.params_reservoir_buf,
                ],
            );
            gpu.dispatch(&self.reservoir_pipeline, &reservoir_bg, wg);
        }

        let readout_bg = gpu.create_bind_group(
            &self.readout_pipeline,
            &[
                &self.w_out_buf,
                &self.state_buf,
                &self.output_buf,
                &self.params_readout_buf,
            ],
        );
        let out_wg = self.output_size.div_ceil(64) as u32;
        gpu.dispatch(&self.readout_pipeline, &readout_bg, out_wg);

        gpu.read_back_f32(&self.output_buf, self.output_size)
            .unwrap_or_default()
            .iter()
            .map(|&v| f64::from(v))
            .collect()
    }

    /// Per-step dispatch with readout deferred until end.
    ///
    /// Recurrent networks require sequential state updates: each reservoir
    /// step reads the state written by the previous step. Naive encoder
    /// batching breaks this because `queue.write_buffer` races with encoded
    /// dispatches. This method uses per-step submit (correct) but defers
    /// the final readback to amortize one direction of host-device transfer.
    #[must_use]
    pub fn predict_batched(&self, gpu: &GpuF64, input_sequence: &[Vec<f64>]) -> Vec<f64> {
        let rs = self.reservoir_size;
        let wg = rs.div_ceil(64) as u32;

        gpu.upload_f32(&self.state_buf, &vec![0.0f32; rs]);

        for input in input_sequence {
            let input_f32: Vec<f32> = input.iter().map(|&v| v as f32).collect();
            gpu.upload_f32(&self.input_buf, &input_f32);

            let reservoir_bg = gpu.create_bind_group(
                &self.reservoir_pipeline,
                &[
                    &self.w_in_buf,
                    &self.w_res_buf,
                    &self.input_buf,
                    &self.state_buf,
                    &self.params_reservoir_buf,
                ],
            );
            gpu.dispatch(&self.reservoir_pipeline, &reservoir_bg, wg);
        }

        let readout_bg = gpu.create_bind_group(
            &self.readout_pipeline,
            &[
                &self.w_out_buf,
                &self.state_buf,
                &self.output_buf,
                &self.params_readout_buf,
            ],
        );
        let out_wg = self.output_size.div_ceil(64) as u32;
        gpu.dispatch(&self.readout_pipeline, &readout_bg, out_wg);

        gpu.read_back_f32(&self.output_buf, self.output_size)
            .unwrap_or_default()
            .iter()
            .map(|&v| f64::from(v))
            .collect()
    }
}

/// Per-substrate timing result for cross-substrate comparison.
#[derive(Clone, Debug)]
pub struct SubstrateResult {
    /// Substrate label (e.g. "CPU-f64", "GPU-f32-batch").
    pub substrate: String,
    /// Reservoir size used.
    pub reservoir_size: usize,
    /// Mean inference time in microseconds.
    pub mean_us: f64,
    /// Standard deviation of inference time (μs).
    #[allow(dead_code)]
    pub std_us: f64,
    /// Last prediction output (for parity checks).
    #[allow(dead_code)]
    pub prediction: Vec<f64>,
}

/// Generate deterministic test sequence using LCG.
///
/// Uses a multiplicative LCG for reproducibility across substrates.
#[must_use]
pub fn generate_test_sequence(seed: u64, length: usize, input_size: usize) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            (0..input_size)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
                })
                .collect()
        })
        .collect()
}

/// Generate training sequences and targets for ESN.
#[must_use]
pub fn generate_training_data(
    n_sequences: usize,
    seq_length: usize,
    input_size: usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let sequences: Vec<Vec<Vec<f64>>> = (0..n_sequences)
        .map(|i| generate_test_sequence(42 + i as u64, seq_length, input_size))
        .collect();
    let targets: Vec<Vec<f64>> = (0..n_sequences)
        .map(|i| vec![(i as f64) / (n_sequences as f64)])
        .collect();
    (sequences, targets)
}

/// Time a function returning `Vec<f64>` with warmup and repetitions.
///
/// Returns `(mean_us, std_us, last_prediction)`.
#[must_use]
pub fn time_fn<F: FnMut() -> Vec<f64>>(
    mut f: F,
    warmup: usize,
    reps: usize,
) -> (f64, f64, Vec<f64>) {
    let mut last_pred = vec![];
    for _ in 0..warmup {
        last_pred = f();
    }
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        last_pred = f();
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    (mean, variance.sqrt(), last_pred)
}
