// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-accelerated BGK dielectric functions (Chuna & Murillo 2024).
//!
//! Promotes the CPU `dielectric` module to GPU by computing the Mermin
//! dielectric function ε(k,ω) for all ω values in a single dispatch.
//! Each GPU thread evaluates one ω point through the full chain:
//!
//! ```text
//! Z(z) → W(z) → χ₀(k,ω) → ε_Vlasov → ε_Mermin → Im[1/ε] → S(k,ω)
//! ```
//!
//! Uses `complex_f64.wgsl` preamble from the lattice QCD shader library.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::similar_names)]
#![allow(clippy::suboptimal_flops)]

use super::dielectric::{self, f_sum_rule_integral, PlasmaParams};
use crate::gpu::GpuF64;

use barracuda::shaders::precision::ShaderTemplate;

const WGSL_COMPLEX: &str = include_str!("../lattice/shaders/complex_f64.wgsl");
const WGSL_MERMIN: &str = include_str!("shaders/dielectric_mermin_f64.wgsl");

/// Uniform parameter buffer layout (must match WGSL `DielectricParams`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DielectricParams {
    n_points: u32,
    pad0: u32,
    k: f64,
    nu: f64,
    k_debye: f64,
    v_th: f64,
    omega_p: f64,
    temperature: f64,
    density: f64,
}

/// GPU dielectric pipeline.
pub struct GpuDielectricPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl GpuDielectricPipeline {
    /// Compile the batched Mermin shader.
    ///
    /// Injects barracuda's `math_f64` polyfills for `exp_f64`, `sin_f64`,
    /// `cos_f64` — native WGSL `exp`/`sin`/`cos` are f32-only on open-source
    /// drivers (NVK/NAK, RADV/ACO). See `math_f64.wgsl` for details.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        let complex_no_exp = WGSL_COMPLEX
            .lines()
            .take_while(|l| !l.contains("fn c64_exp"))
            .collect::<Vec<_>>()
            .join("\n");
        let full_complex = format!("{complex_no_exp}\n{}", c64_exp_f64_source());
        let user_shader = format!("{full_complex}\n{WGSL_MERMIN}");
        let source = ShaderTemplate::with_math_f64_auto(&user_shader);
        Self {
            pipeline: gpu.create_pipeline_f64(&source, "dielectric_mermin"),
        }
    }
}

const fn c64_exp_f64_source() -> &'static str {
    "fn c64_exp(a: Complex64) -> Complex64 {\n    let r = exp_f64(a.re);\n    return Complex64(r * cos_f64(a.im), r * sin_f64(a.im));\n}"
}

/// Result of a GPU dielectric evaluation.
pub struct GpuDielectricResult {
    /// Im[1/ε(k,ω)] for each ω (loss function).
    pub loss: Vec<f64>,
    /// S(k,ω) for each ω (dynamic structure factor).
    pub dsf: Vec<f64>,
    /// Wall time in seconds.
    pub wall_seconds: f64,
}

/// Evaluate Mermin dielectric function for all ω values on GPU.
///
/// # Panics
///
/// Panics if `omegas` is empty.
#[must_use]
pub fn gpu_dielectric_batch(
    gpu: &GpuF64,
    pipeline: &GpuDielectricPipeline,
    k: f64,
    nu: f64,
    params: &PlasmaParams,
    omegas: &[f64],
) -> GpuDielectricResult {
    assert!(!omegas.is_empty(), "omegas must not be empty");

    let start = std::time::Instant::now();
    let n = omegas.len();

    let uniform = DielectricParams {
        n_points: n as u32,
        pad0: 0,
        k,
        nu,
        k_debye: params.k_debye,
        v_th: params.v_th,
        omega_p: params.omega_p,
        temperature: params.temperature,
        density: params.n,
    };

    let param_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&uniform), "diel_params");
    let omega_buf = gpu.create_f64_buffer(omegas, "diel_omegas");
    let loss_buf = gpu.create_f64_output_buffer(n, "diel_loss");
    let dsf_buf = gpu.create_f64_output_buffer(n, "diel_dsf");

    let bg = gpu.create_bind_group(
        &pipeline.pipeline,
        &[&param_buf, &omega_buf, &loss_buf, &dsf_buf],
    );

    let wg = (n as u32).div_ceil(64);
    gpu.dispatch(&pipeline.pipeline, &bg, wg);

    let loss = gpu
        .read_back_f64(&loss_buf, n)
        .unwrap_or_else(|_| vec![0.0; n]);
    let dsf = gpu
        .read_back_f64(&dsf_buf, n)
        .unwrap_or_else(|_| vec![0.0; n]);

    GpuDielectricResult {
        loss,
        dsf,
        wall_seconds: start.elapsed().as_secs_f64(),
    }
}

/// GPU-accelerated f-sum rule integral via batched Mermin + host reduction.
#[must_use]
pub fn gpu_f_sum_integral(
    gpu: &GpuF64,
    pipeline: &GpuDielectricPipeline,
    k: f64,
    nu: f64,
    params: &PlasmaParams,
    omega_max: f64,
    n_points: usize,
) -> f64 {
    let d_omega = omega_max / n_points as f64;
    let omegas: Vec<f64> = (1..n_points)
        .map(|i| f64::from(i as i32) * d_omega)
        .collect();

    let result = gpu_dielectric_batch(gpu, pipeline, k, nu, params, &omegas);

    let mut sum = 0.0;
    for (i, &loss_im) in result.loss.iter().enumerate() {
        let omega = omegas[i];
        sum += omega * loss_im;
    }
    sum * d_omega
}

/// Run full GPU validation matching CPU `validate_dielectric`.
pub struct GpuDielectricValidation {
    /// f-sum integral (GPU).
    pub f_sum_gpu: f64,
    /// f-sum integral (CPU).
    pub f_sum_cpu: f64,
    /// DSF positivity fraction (GPU).
    pub dsf_pos_fraction_gpu: f64,
    /// DSF positivity fraction (CPU).
    pub dsf_pos_fraction_cpu: f64,
    /// Max relative error in loss function (only where |loss| > 0.1% of peak).
    pub max_loss_rel_error: f64,
    /// L² relative error over all ω points.
    pub l2_loss_rel_error: f64,
    /// GPU wall time for batched Mermin.
    pub gpu_wall_seconds: f64,
    /// CPU wall time for batched Mermin.
    pub cpu_wall_seconds: f64,
}

/// Compare GPU vs CPU dielectric for given coupling parameters.
#[must_use]
pub fn validate_gpu_dielectric(
    gpu: &GpuF64,
    pipeline: &GpuDielectricPipeline,
    gamma: f64,
    kappa: f64,
) -> GpuDielectricValidation {
    let params = PlasmaParams::from_coupling(gamma, kappa);
    let nu = 0.1 * params.omega_p;
    let k = 1.0;

    let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + f64::from(i) * 0.02).collect();

    let gpu_start = std::time::Instant::now();
    let gpu_result = gpu_dielectric_batch(gpu, pipeline, k, nu, &params, &omegas);
    let gpu_wall = gpu_start.elapsed().as_secs_f64();

    let cpu_start = std::time::Instant::now();
    let cpu_dsf = dielectric::dynamic_structure_factor(k, &omegas, nu, &params);
    let cpu_wall = cpu_start.elapsed().as_secs_f64();

    let cpu_loss: Vec<f64> = omegas
        .iter()
        .map(|&omega| {
            let eps = dielectric::epsilon_mermin(k, omega, nu, &params);
            eps.inv().im
        })
        .collect();

    let peak_loss = cpu_loss
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()));
    let significance_threshold = 0.05 * peak_loss;

    let mut max_rel_err = 0.0_f64;
    let mut l2_num = 0.0_f64;
    let mut l2_den = 0.0_f64;
    for (&gpu_l, &cpu_l) in gpu_result.loss.iter().zip(cpu_loss.iter()) {
        let diff = (gpu_l - cpu_l).abs();
        l2_num += diff * diff;
        l2_den += cpu_l * cpu_l;
        if cpu_l.abs() > significance_threshold {
            let rel = diff / cpu_l.abs();
            if rel > max_rel_err {
                max_rel_err = rel;
            }
        }
    }
    let l2_rel = (l2_num / l2_den.max(1e-30)).sqrt();

    let dsf_max_gpu = gpu_result.dsf.iter().copied().fold(0.0_f64, f64::max);
    let n_pos_gpu = gpu_result
        .dsf
        .iter()
        .filter(|&&s| s >= -1e-6 * dsf_max_gpu.max(1e-10))
        .count();
    let dsf_pos_gpu = n_pos_gpu as f64 / gpu_result.dsf.len() as f64;

    let dsf_max_cpu = cpu_dsf.iter().copied().fold(0.0_f64, f64::max);
    let n_pos_cpu = cpu_dsf
        .iter()
        .filter(|&&s| s >= -1e-6 * dsf_max_cpu.max(1e-10))
        .count();
    let dsf_pos_cpu = n_pos_cpu as f64 / cpu_dsf.len() as f64;

    let f_sum_gpu = gpu_f_sum_integral(gpu, pipeline, k, nu, &params, 200.0, 50_000);
    let f_sum_cpu = f_sum_rule_integral(k, nu, &params, 200.0);

    GpuDielectricValidation {
        f_sum_gpu,
        f_sum_cpu,
        dsf_pos_fraction_gpu: dsf_pos_gpu,
        dsf_pos_fraction_cpu: dsf_pos_cpu,
        max_loss_rel_error: max_rel_err,
        l2_loss_rel_error: l2_rel,
        gpu_wall_seconds: gpu_wall,
        cpu_wall_seconds: cpu_wall,
    }
}
