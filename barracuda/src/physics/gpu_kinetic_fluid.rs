// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-accelerated BGK relaxation for multi-species kinetic plasma (Paper 45).
//!
//! Promotes the velocity-space operations from `kinetic_fluid.rs` to GPU:
//! - Moment computation (parallel per velocity point, CPU reduction)
//! - Maxwellian evaluation + BGK relaxation (parallel per velocity point)
//!
//! The Euler fluid solver remains on CPU since it operates on the spatial
//! mesh (small N_x) rather than velocity space (large N_v × N_species).

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::suboptimal_flops)]

use super::kinetic_fluid::{
    BgkRelaxationResult, BgkSpecies, bgk_target_params, compute_moments, maxwellian_1d,
};
use crate::gpu::GpuF64;

use barracuda::shaders::precision::ShaderTemplate;
use wgpu::util::DeviceExt;

const WGSL_BGK: &str = include_str!("shaders/bgk_relaxation_f64.wgsl");

/// Uniform parameter buffer (must match WGSL `BgkParams`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BgkParams {
    nv: u32,
    n_species: u32,
    dt: f64,
    v_min: f64,
    dv: f64,
}

/// GPU BGK pipeline with two entry points.
pub struct GpuBgkPipeline {
    moments_pipeline: wgpu::ComputePipeline,
    relax_pipeline: wgpu::ComputePipeline,
}

impl GpuBgkPipeline {
    /// Compile BGK shaders.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        let source = ShaderTemplate::with_math_f64_auto(WGSL_BGK);
        Self {
            moments_pipeline: gpu.create_pipeline_f64_entry(
                &source,
                "compute_moments",
                "compute_moments",
            ),
            relax_pipeline: gpu.create_pipeline_f64_entry(&source, "bgk_relax", "bgk_relax"),
        }
    }
}

/// Result of GPU BGK relaxation.
pub struct GpuBgkResult {
    /// Same as `BgkRelaxationResult` from CPU module.
    pub result: BgkRelaxationResult,
    /// GPU wall time for BGK relaxation steps (seconds).
    pub gpu_wall_seconds: f64,
    /// CPU wall time for equivalent computation (seconds).
    pub cpu_wall_seconds: f64,
}

/// Run two-species BGK relaxation on GPU, matching the CPU `run_bgk_relaxation`.
///
/// The velocity-space operations (moment computation + BGK relaxation step)
/// run on GPU. Target parameter computation (small reduction) runs on CPU.
#[must_use]
pub fn gpu_bgk_relaxation(
    gpu: &GpuF64,
    pipeline: &GpuBgkPipeline,
    n_steps: usize,
    dt: f64,
) -> GpuBgkResult {
    let nv: usize = 201;
    let v_max = 8.0;
    let nv_m1 = (nv - 1) as f64;
    let v: Vec<f64> = (0..nv)
        .map(|i| -v_max + (i as f64) * 2.0 * v_max / nv_m1)
        .collect();
    let dv = v[1] - v[0];

    let (m1, m2) = (1.0, 4.0);
    let (n1, n2) = (1.0, 1.0);
    let (u1, u2) = (0.0, 0.0);
    let (t1, t2) = (2.0, 0.5);
    let (nu1, nu2) = (1.0, 1.0);

    let f1_init = maxwellian_1d(&v, n1, u1, t1, m1);
    let f2_init = maxwellian_1d(&v, n2, u2, t2, m2);

    let (n1_0, u1_0, _, e1_0) = compute_moments(&f1_init, &v, dv, m1);
    let (n2_0, u2_0, _, e2_0) = compute_moments(&f2_init, &v, dv, m2);
    let e_total_0 = e1_0 + e2_0;
    let mom_total_0 = m1 * n1_0 * u1_0 + m2 * n2_0 * u2_0;

    let n_species: usize = 2;
    let total_points = n_species * nv;

    // Flat f data: [species0_v0, species0_v1, ..., species1_v0, ...]
    let mut f_flat: Vec<f64> = Vec::with_capacity(total_points);
    f_flat.extend_from_slice(&f1_init);
    f_flat.extend_from_slice(&f2_init);

    let params = BgkParams {
        nv: nv as u32,
        n_species: n_species as u32,
        dt,
        v_min: -v_max,
        dv,
    };

    let gpu_start = std::time::Instant::now();

    // GPU buffers
    let param_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "bgk_params");

    let f_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bgk_f_data"),
            contents: bytemuck::cast_slice(&f_flat),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

    // species_params: [m, nu, n_star, u_star, t_star] per species
    let sp_data: Vec<f64> = vec![
        m1, nu1, n1, u1, t1, // initial (overwritten each step)
        m2, nu2, n2, u2, t2,
    ];
    let sp_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bgk_species_params"),
            contents: bytemuck::cast_slice(&sp_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    let moment_size = n_species * nv * 3;
    let moment_buf = gpu.create_f64_output_buffer(moment_size, "bgk_moments");

    let mut entropy_decreasing = false;

    // Compute initial entropy
    let h_fn = |f: &[f64]| -> f64 {
        f.iter()
            .map(|&fi| {
                let fp = fi.max(1e-300);
                fp * fp.ln()
            })
            .sum::<f64>()
            * dv
    };
    let mut h_prev = h_fn(&f_flat[..nv]) + h_fn(&f_flat[nv..]);

    for _ in 0..n_steps {
        // 1. Compute moments on GPU (per-point contributions)
        let bg_moments = gpu.create_bind_group(
            &pipeline.moments_pipeline,
            &[&param_buf, &f_buf, &sp_buf, &moment_buf],
        );
        let wg = (total_points as u32).div_ceil(64);
        gpu.dispatch(&pipeline.moments_pipeline, &bg_moments, wg);

        // 2. Read back moment contributions and reduce on CPU
        let contributions = gpu
            .read_back_f64(&moment_buf, moment_size)
            .unwrap_or_else(|_| vec![0.0; moment_size]);

        let mut species_cpu: Vec<BgkSpecies> = Vec::new();
        for sp in 0..n_species {
            let mut n_sum = 0.0;
            let mut nu_sum = 0.0;
            let mut nv2_sum = 0.0;
            for j in 0..nv {
                let base = sp * nv * 3 + j * 3;
                n_sum += contributions[base];
                nu_sum += contributions[base + 1];
                nv2_sum += contributions[base + 2];
            }
            let mass = if sp == 0 { m1 } else { m2 };
            let nu_sp = if sp == 0 { nu1 } else { nu2 };
            let u_sp = if n_sum.abs() > 1e-30 {
                nu_sum / n_sum
            } else {
                0.0
            };
            let e_sp = 0.5 * mass * nv2_sum;
            let _t_sp = if n_sum.abs() > 1e-30 {
                (2.0 * e_sp / n_sum - mass * u_sp * u_sp).max(1e-12)
            } else {
                1e-12
            };

            // Build a BgkSpecies with f data for bgk_target_params
            let f_zero = vec![0.0; nv];
            species_cpu.push(BgkSpecies {
                m: mass,
                nu: nu_sp,
                f: f_zero,
            });
            // Overwrite the f data from the reduced moments for target_params
            species_cpu[sp].f = {
                let start = sp * nv;
                let end = start + nv;
                // Read f back from GPU for target computation
                let f_read = gpu
                    .read_back_f64(&f_buf, total_points)
                    .unwrap_or_else(|_| vec![0.0; total_points]);
                f_read[start..end].to_vec()
            };
        }

        let targets = bgk_target_params(&species_cpu, &v, dv);

        // 3. Upload target params to GPU and run relaxation
        let mut sp_update: Vec<f64> = Vec::with_capacity(n_species * 5);
        for (sp, &(n_star, u_star, t_star)) in targets.iter().enumerate() {
            let mass = if sp == 0 { m1 } else { m2 };
            let nu_sp = if sp == 0 { nu1 } else { nu2 };
            sp_update.extend_from_slice(&[mass, nu_sp, n_star, u_star, t_star]);
        }

        gpu.queue()
            .write_buffer(&sp_buf, 0, bytemuck::cast_slice(&sp_update));

        let bg_relax = gpu.create_bind_group(
            &pipeline.relax_pipeline,
            &[&param_buf, &f_buf, &sp_buf, &moment_buf],
        );
        gpu.dispatch(&pipeline.relax_pipeline, &bg_relax, wg);

        {
            let f_current = gpu
                .read_back_f64(&f_buf, total_points)
                .unwrap_or_else(|_| vec![0.0; total_points]);
            let h_curr = h_fn(&f_current[..nv]) + h_fn(&f_current[nv..]);
            if h_curr > h_prev + 1e-10 {
                entropy_decreasing = true;
            }
            h_prev = h_curr;
            f_flat = f_current;
        }
    }

    let gpu_wall = gpu_start.elapsed().as_secs_f64();

    // CPU timing for comparison
    let cpu_start = std::time::Instant::now();
    let _cpu_result = super::kinetic_fluid::run_bgk_relaxation(n_steps, dt);
    let cpu_wall = cpu_start.elapsed().as_secs_f64();

    // Final moments from GPU
    let f1_final = &f_flat[..nv];
    let f2_final = &f_flat[nv..];
    let (n1_f, u1_f, t1_f, e1_f) = compute_moments(f1_final, &v, dv, m1);
    let (n2_f, u2_f, t2_f, e2_f) = compute_moments(f2_final, &v, dv, m2);
    let e_total_f = e1_f + e2_f;
    let mom_total_f = m1 * n1_f * u1_f + m2 * n2_f * u2_f;

    let temp_relaxed = if t1_f.max(t2_f) > 0.0 {
        (t1_f - t2_f).abs() / t1_f.max(t2_f)
    } else {
        0.0
    };

    GpuBgkResult {
        result: BgkRelaxationResult {
            mass_err_1: (n1_f - n1_0).abs(),
            mass_err_2: (n2_f - n2_0).abs(),
            momentum_err: (mom_total_f - mom_total_0).abs(),
            energy_err: (e_total_f - e_total_0).abs() / e_total_0.abs().max(1e-30),
            entropy_monotonic: !entropy_decreasing,
            t1_final: t1_f,
            t2_final: t2_f,
            temp_relaxed,
        },
        gpu_wall_seconds: gpu_wall,
        cpu_wall_seconds: cpu_wall,
    }
}

/// Validate GPU BGK against CPU: both should conserve mass, momentum, energy.
#[must_use]
pub fn validate_gpu_bgk(
    gpu: &GpuF64,
    pipeline: &GpuBgkPipeline,
) -> (GpuBgkResult, BgkRelaxationResult) {
    let gpu_result = gpu_bgk_relaxation(gpu, pipeline, 3000, 0.005);
    let cpu_result = super::kinetic_fluid::run_bgk_relaxation(3000, 0.005);
    (gpu_result, cpu_result)
}
