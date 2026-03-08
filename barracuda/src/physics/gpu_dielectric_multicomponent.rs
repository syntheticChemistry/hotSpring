// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-accelerated multi-component Mermin dielectric function (Paper 44 extension).
//!
//! Promotes `dielectric_multicomponent.rs` to GPU, dispatching batched ε(k,ω)
//! evaluation across frequency points in parallel.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

use super::dielectric_multicomponent::{
    epsilon_multicomponent_mermin, MultiComponentPlasma, SpeciesParams,
};
use crate::gpu::GpuF64;

use barracuda::shaders::precision::ShaderTemplate;
use wgpu::util::DeviceExt;

const WGSL_MULTICOMP: &str = include_str!("shaders/dielectric_multicomponent_f64.wgsl");

/// Uniform parameter buffer (must match WGSL `MulticompParams`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MulticompParams {
    n_points: u32,
    n_species: u32,
    use_completed: u32,
    _pad: u32,
    k: f64,
}

/// GPU multi-component dielectric pipeline.
pub struct GpuMulticompPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl GpuMulticompPipeline {
    /// Compile multi-component dielectric shader.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        let source = ShaderTemplate::with_math_f64_auto(WGSL_MULTICOMP);
        Self {
            pipeline: gpu.create_pipeline_f64_entry_precise(
                &source,
                "compute_multicomp",
                "compute_multicomp",
            ),
        }
    }
}

/// Result of GPU multi-component dielectric batch.
pub struct GpuMulticompResult {
    /// Loss function: -Im[1/ε(k,ω)] at each frequency point.
    pub loss: Vec<f64>,
    /// Dynamic structure factor S(k,ω) at each frequency point.
    pub dsf: Vec<f64>,
    /// GPU wall time (seconds).
    pub wall_seconds: f64,
}

/// Flatten species parameters into GPU buffer layout.
/// Per species: [mass, charge, density, temperature, nu, v_th, k_debye]
fn flatten_species(plasma: &MultiComponentPlasma) -> Vec<f64> {
    let mut buf = Vec::with_capacity(plasma.species.len() * 7);
    for s in &plasma.species {
        buf.extend_from_slice(&[
            s.mass,
            s.charge,
            s.density,
            s.temperature,
            s.nu,
            s.v_th(),
            s.k_debye(),
        ]);
    }
    buf
}

/// Batch-evaluate multi-component Mermin on GPU.
#[must_use]
pub fn gpu_multicomponent_dielectric(
    gpu: &GpuF64,
    pipeline: &GpuMulticompPipeline,
    k: f64,
    plasma: &MultiComponentPlasma,
    omegas: &[f64],
    completed: bool,
) -> GpuMulticompResult {
    assert!(!omegas.is_empty(), "omegas must not be empty");

    let start = std::time::Instant::now();
    let n = omegas.len();

    let params = MulticompParams {
        n_points: n as u32,
        n_species: plasma.species.len() as u32,
        use_completed: u32::from(completed),
        _pad: 0,
        k,
    };

    let param_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "mc_params");
    let omega_buf = gpu.create_f64_buffer(omegas, "mc_omegas");

    let species_data = flatten_species(plasma);
    let sp_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc_species"),
            contents: bytemuck::cast_slice(&species_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let loss_buf = gpu.create_f64_output_buffer(n, "mc_loss");
    let dsf_buf = gpu.create_f64_output_buffer(n, "mc_dsf");

    let bg = gpu.create_bind_group(
        &pipeline.pipeline,
        &[&param_buf, &omega_buf, &sp_buf, &loss_buf, &dsf_buf],
    );

    let wg = (n as u32).div_ceil(64);
    gpu.dispatch(&pipeline.pipeline, &bg, wg);

    let loss = gpu
        .read_back_f64(&loss_buf, n)
        .unwrap_or_else(|_| vec![0.0; n]);
    let dsf = gpu
        .read_back_f64(&dsf_buf, n)
        .unwrap_or_else(|_| vec![0.0; n]);

    GpuMulticompResult {
        loss,
        dsf,
        wall_seconds: start.elapsed().as_secs_f64(),
    }
}

/// Validate GPU multi-component against CPU.
#[must_use]
pub fn validate_gpu_multicomponent(
    gpu: &GpuF64,
    pipeline: &GpuMulticompPipeline,
) -> (Vec<f64>, Vec<f64>) {
    let plasma = MultiComponentPlasma {
        species: vec![
            SpeciesParams {
                mass: 1.0 / 1836.0,
                charge: 1.0,
                density: 1.0,
                temperature: 1.0,
                nu: 0.1,
            },
            SpeciesParams {
                mass: 1.0,
                charge: 1.0,
                density: 1.0,
                temperature: 1.0,
                nu: 0.01,
            },
        ],
    };

    let k = 1.0;
    let omegas: Vec<f64> = (1..50).map(|i| 0.2 * i as f64).collect();

    let gpu_result = gpu_multicomponent_dielectric(gpu, pipeline, k, &plasma, &omegas, true);

    let cpu_loss: Vec<f64> = omegas
        .iter()
        .map(|&omega| {
            let eps = epsilon_multicomponent_mermin(k, omega, &plasma, true);
            -eps.inv().im
        })
        .collect();

    (gpu_result.loss, cpu_loss)
}
