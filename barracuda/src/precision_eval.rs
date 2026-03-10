// SPDX-License-Identifier: AGPL-3.0-only

//! Per-shader precision/throughput profiler across all 3 tiers.
//!
//! Compiles and benchmarks a WGSL shader at F32, F64, DF64, and F64Precise
//! on a single `GpuF64`, measuring compile time, dispatch throughput,
//! readback latency, and numerical accuracy (ULP error vs F64 reference).

use crate::gpu::GpuF64;
use crate::precision_routing::PrecisionTier;
use std::time::Instant;

const WARMUP: usize = 5;
const MEASURE: usize = 20;

/// Result of evaluating a single precision tier.
#[derive(Debug, Clone)]
pub struct TierResult {
    /// Which precision tier was evaluated.
    pub tier: PrecisionTier,
    /// Whether the shader compiled at this tier.
    pub compiled: bool,
    /// Shader compilation time in microseconds.
    pub compile_us: f64,
    /// Mean dispatch time in microseconds (median over N runs).
    pub dispatch_us: f64,
    /// Readback time in microseconds.
    pub readback_us: f64,
    /// Output values from the final dispatch.
    pub output: Vec<f64>,
    /// Max ULP error vs F64 native reference.
    pub max_ulp_error: f64,
}

/// Result of evaluating a shader across all applicable tiers.
#[derive(Debug, Clone)]
pub struct ShaderEvalResult {
    /// Shader name label.
    pub shader_name: String,
    /// Per-tier results.
    pub tiers: Vec<TierResult>,
}

/// Per-shader precision/throughput profiler.
pub struct PrecisionEval<'a> {
    gpu: &'a GpuF64,
}

impl<'a> PrecisionEval<'a> {
    /// Create a precision evaluator for the given GPU.
    #[must_use]
    pub fn new(gpu: &'a GpuF64) -> Self {
        Self { gpu }
    }

    /// Evaluate a shader at all applicable tiers using a simple element-wise
    /// test pattern: input buffer of N f64 values, output buffer of N f64 values.
    ///
    /// The shader must have bindings: @group(0) @binding(0) input, @binding(1) output.
    /// Evaluate a shader at all applicable tiers using a simple element-wise
    /// test pattern: input buffer of N f64 values, output buffer of N f64 values.
    ///
    /// The shader must have bindings: @group(0) @binding(0) input, @binding(1) output.
    pub fn eval_shader(
        &self,
        name: &str,
        f64_source: &str,
        input: &[f64],
        n_elements: usize,
        workgroups: u32,
    ) -> ShaderEvalResult {
        self.eval_shader_tiers(
            name,
            f64_source,
            input,
            n_elements,
            workgroups,
            &[
                PrecisionTier::F64,
                PrecisionTier::F64Precise,
                PrecisionTier::DF64,
                PrecisionTier::F32,
            ],
        )
    }

    /// Evaluate a shader at only the specified tiers (skipping unsafe ones).
    pub fn eval_shader_tiers(
        &self,
        name: &str,
        f64_source: &str,
        input: &[f64],
        n_elements: usize,
        workgroups: u32,
        safe_tiers: &[PrecisionTier],
    ) -> ShaderEvalResult {
        let tiers_to_test = [
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
            PrecisionTier::DF64,
            PrecisionTier::F32,
        ];

        let mut results = Vec::new();
        let mut reference: Option<Vec<f64>> = None;

        for &tier in &tiers_to_test {
            if !safe_tiers.contains(&tier) {
                results.push(TierResult {
                    tier,
                    compiled: false,
                    compile_us: 0.0,
                    dispatch_us: 0.0,
                    readback_us: 0.0,
                    output: vec![],
                    max_ulp_error: f64::NAN,
                });
                continue;
            }
            let ref_slice = reference.as_deref();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.eval_tier(name, f64_source, input, n_elements, workgroups, tier, ref_slice)
            }))
            .unwrap_or(TierResult {
                tier,
                compiled: false,
                compile_us: 0.0,
                dispatch_us: 0.0,
                readback_us: 0.0,
                output: vec![],
                max_ulp_error: f64::NAN,
            });
            if tier == PrecisionTier::F64 && result.compiled {
                reference = Some(result.output.clone());
            }
            results.push(result);
        }

        ShaderEvalResult {
            shader_name: name.to_string(),
            tiers: results,
        }
    }

    fn eval_tier(
        &self,
        name: &str,
        f64_source: &str,
        input: &[f64],
        n_elements: usize,
        workgroups: u32,
        tier: PrecisionTier,
        reference: Option<&[f64]>,
    ) -> TierResult {
        let label = format!("{name}_{tier:?}");

        let t_compile = Instant::now();
        let pipeline = self.compile_at_tier(f64_source, &label, tier);
        let compile_us = t_compile.elapsed().as_secs_f64() * 1e6;

        let input_buf = self.gpu.create_f64_buffer(input, &format!("{label}_in"));
        let output_buf = self
            .gpu
            .create_f64_output_buffer(n_elements, &format!("{label}_out"));
        let bind_group = self
            .gpu
            .create_bind_group(&pipeline, &[&input_buf, &output_buf]);

        for _ in 0..WARMUP {
            self.gpu.dispatch(&pipeline, &bind_group, workgroups);
        }
        let _ = self.gpu.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let t_dispatch = Instant::now();
        for _ in 0..MEASURE {
            self.gpu.dispatch(&pipeline, &bind_group, workgroups);
        }
        let _ = self.gpu.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let dispatch_us = t_dispatch.elapsed().as_secs_f64() * 1e6 / MEASURE as f64;

        let t_readback = Instant::now();
        let output = self
            .gpu
            .read_back_f64(&output_buf, n_elements)
            .unwrap_or_default();
        let readback_us = t_readback.elapsed().as_secs_f64() * 1e6;

        let max_ulp_error = reference.map_or(0.0, |ref_vals| max_ulp(ref_vals, &output));

        TierResult {
            tier,
            compiled: true,
            compile_us,
            dispatch_us,
            readback_us,
            output,
            max_ulp_error,
        }
    }

    fn compile_at_tier(
        &self,
        source: &str,
        label: &str,
        tier: PrecisionTier,
    ) -> wgpu::ComputePipeline {
        match tier {
            PrecisionTier::F32 => self.gpu.create_pipeline(source, label),
            PrecisionTier::F64 => self.gpu.create_pipeline_f64(source, label),
            PrecisionTier::DF64 => self.gpu.compile_full_df64_pipeline(source, label),
            PrecisionTier::F64Precise => self.gpu.create_pipeline_f64_precise(source, label),
        }
    }
}

/// Compute max ULP error between two f64 slices.
fn max_ulp(reference: &[f64], actual: &[f64]) -> f64 {
    reference
        .iter()
        .zip(actual.iter())
        .map(|(&r, &a)| ulp_distance(r, a))
        .fold(0.0_f64, f64::max)
}

#[allow(clippy::float_cmp)]
fn ulp_distance(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return 0.0;
    }
    if a.is_infinite() || b.is_infinite() {
        return f64::INFINITY;
    }
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;
    (a_bits - b_bits).unsigned_abs() as f64
}

/// Simple element-wise test shader: output[i] = input[i] * input[i] + 1.0.
///
/// Useful for quick precision evaluation — the FMA pattern exercises
/// precision differences between tiers.
pub const EVAL_SHADER_SQUARE_PLUS_ONE: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    let x = input[i];
    output[i] = x * x + 1.0;
}
";

/// Stress test shader: Kahan summation loop to amplify precision differences.
pub const EVAL_SHADER_KAHAN_SUM: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    var sum: f64 = 0.0;
    var c: f64 = 0.0;
    let x = input[i];
    for (var k: u32 = 0u; k < 100u; k = k + 1u) {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    output[i] = sum;
}
";

/// Transcendental stress test: exp(log(x)) should return x.
pub const EVAL_SHADER_EXP_LOG: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&input) { return; }
    let x = input[i];
    output[i] = exp(log(x));
}
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_distance_identical() {
        assert!((ulp_distance(1.0, 1.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ulp_distance_one_ulp() {
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert!((ulp_distance(a, b) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ulp_distance_nan() {
        assert!(ulp_distance(f64::NAN, 1.0).is_nan());
        assert!(ulp_distance(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn ulp_distance_infinity() {
        assert!(ulp_distance(f64::INFINITY, 1.0).is_infinite());
    }

    #[test]
    fn max_ulp_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((max_ulp(&a, &a) - 0.0).abs() < f64::EPSILON);
    }
}
