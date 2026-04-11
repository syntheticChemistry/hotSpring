// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident scalar observables — O(1) readback for plaquette, KE, and Wilson action.
//!
//! Replaces the deprecated O(V)-readback pattern (`gpu_wilson_action`, `gpu_plaquette`,
//! `gpu_kinetic_energy`) with GPU reduce chains that produce single f64 scalars on-device.
//! CPU only reads back 8–24 bytes instead of O(V) per-site/per-link arrays.
//!
//! Used by `streaming`, `resident_cg`, `hasenbusch`, and `unidirectional_rhmc` paths.
//! The `unidirectional_rhmc` module extends this with GPU-resident Hamiltonian assembly
//! (B2) and Metropolis (B3) for zero-readback trajectories.

use super::resident_cg_buffers::{ReduceChain, build_reduce_chain_pub, encode_reduce_chain};
use super::{GpuF64, GpuHmcPipelines, GpuHmcState, make_u32x4_params};

/// Persistent GPU buffers for O(1)-readback scalar observables.
///
/// Allocate once per GPU, reuse across trajectories. The reduce chains
/// tree-sum per-site plaquette and per-link KE arrays into single f64
/// scalars, avoiding all O(V) readbacks.
pub struct ResidentObservableBuffers {
    /// Reduced plaquette sum (1 f64, GPU-resident).
    pub plaq_sum_buf: wgpu::Buffer,
    /// Reduced kinetic energy (1 f64, GPU-resident).
    pub ke_buf: wgpu::Buffer,
    /// Reduction scratch (ping).
    pub scratch_a: wgpu::Buffer,
    /// Reduction scratch (pong).
    pub scratch_b: wgpu::Buffer,
    /// Staging for scalar readbacks (MAP_READ, 64 bytes — room for 8 f64s).
    pub staging: wgpu::Buffer,
    /// Plaquette reduce chain: plaq_out_buf (vol entries) → plaq_sum_buf (1 entry).
    pub reduce_plaq: ReduceChain,
    /// KE reduce chain: ke_out_buf (n_links entries) → ke_buf (1 entry).
    pub reduce_ke: ReduceChain,
}

impl ResidentObservableBuffers {
    /// Allocate observable buffers sized for a given gauge state.
    #[must_use]
    pub fn new(gpu: &GpuF64, reduce_pl: &wgpu::ComputePipeline, gauge: &GpuHmcState) -> Self {
        let vol = gauge.volume;
        let n_links = gauge.n_links;

        let max_wg = vol.max(n_links).div_ceil(256);
        let scratch_a = gpu.create_f64_output_buffer(max_wg.max(1), "obs_scratch_a");
        let scratch_b = gpu.create_f64_output_buffer(max_wg.max(1), "obs_scratch_b");

        let plaq_sum_buf = gpu.create_f64_output_buffer(1, "obs_plaq_sum");
        let ke_buf = gpu.create_f64_output_buffer(1, "obs_ke");
        let staging = gpu.create_staging_buffer(64, "obs_staging");

        let reduce_plaq = build_reduce_chain_pub(
            gpu,
            reduce_pl,
            &gauge.plaq_out_buf,
            &scratch_a,
            &scratch_b,
            &plaq_sum_buf,
            vol,
        );
        let reduce_ke = build_reduce_chain_pub(
            gpu,
            reduce_pl,
            &gauge.ke_out_buf,
            &scratch_a,
            &scratch_b,
            &ke_buf,
            n_links,
        );

        Self {
            plaq_sum_buf,
            ke_buf,
            scratch_a,
            scratch_b,
            staging,
            reduce_plaq,
            reduce_ke,
        }
    }
}

/// Encode plaquette dispatch + GPU reduce → `obs.plaq_sum_buf` (no readback).
pub fn encode_plaquette_reduce(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    obs: &ResidentObservableBuffers,
) {
    let params = make_u32x4_params(gauge.volume as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "obs_plaq_p");
    let bg = gpu.create_bind_group(
        &pipelines.plaquette_pipeline,
        &[&pbuf, &gauge.link_buf, &gauge.nbr_buf, &gauge.plaq_out_buf],
    );
    GpuF64::encode_pass(enc, &pipelines.plaquette_pipeline, &bg, gauge.wg_vol);
    encode_reduce_chain(enc, reduce_pl, &obs.reduce_plaq);
}

/// Encode KE dispatch + GPU reduce → `obs.ke_buf` (no readback).
pub fn encode_ke_reduce(
    enc: &mut wgpu::CommandEncoder,
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    obs: &ResidentObservableBuffers,
) {
    let params = make_u32x4_params(gauge.n_links as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "obs_ke_p");
    let bg = gpu.create_bind_group(
        &pipelines.kinetic_pipeline,
        &[&pbuf, &gauge.mom_buf, &gauge.ke_out_buf],
    );
    GpuF64::encode_pass(enc, &pipelines.kinetic_pipeline, &bg, gauge.wg_links);
    encode_reduce_chain(enc, reduce_pl, &obs.reduce_ke);
}

/// Compute gauge + KE on GPU and read back 2 scalars (16 bytes total).
///
/// Returns `(plaq_sum, kinetic_energy)`. Caller computes
/// `S_gauge = beta * (6V - plaq_sum)` and `H = S_gauge + KE + S_ferm`.
/// # Errors
///
/// Returns `GpuCompute` if the staging buffer readback fails (device lost, timeout).
pub fn gauge_ke_resident(
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    obs: &ResidentObservableBuffers,
) -> Result<(f64, f64), crate::error::HotSpringError> {
    let mut enc = gpu.begin_encoder("obs_gauge_ke");
    encode_plaquette_reduce(&mut enc, gpu, pipelines, gauge, reduce_pl, obs);
    encode_ke_reduce(&mut enc, gpu, pipelines, gauge, reduce_pl, obs);
    enc.copy_buffer_to_buffer(&obs.plaq_sum_buf, 0, &obs.staging, 0, 8);
    enc.copy_buffer_to_buffer(&obs.ke_buf, 0, &obs.staging, 8, 8);
    gpu.submit_encoder(enc);

    let data = gpu.read_staging_f64_n(&obs.staging, 2).map_err(|e| {
        crate::error::HotSpringError::GpuCompute(format!(
            "gauge/KE readback failed (GPU lost?): {e}"
        ))
    })?;
    Ok((data[0], data[1]))
}

/// GPU-resident Wilson action from 16-byte readback.
///
/// S_gauge = beta * (6V - plaq_sum).
///
/// # Errors
///
/// Returns `GpuCompute` if the staging buffer readback fails.
pub fn wilson_action_resident(
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    obs: &ResidentObservableBuffers,
) -> Result<f64, crate::error::HotSpringError> {
    let (plaq_sum, _) = gauge_ke_resident(gpu, pipelines, gauge, reduce_pl, obs)?;
    Ok(gauge.beta * 6.0f64.mul_add(gauge.volume as f64, -plaq_sum))
}

/// GPU-resident average plaquette from 8-byte readback.
///
/// # Errors
///
/// Returns `GpuCompute` if the staging buffer readback fails.
pub fn plaquette_resident(
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    gauge: &GpuHmcState,
    reduce_pl: &wgpu::ComputePipeline,
    obs: &ResidentObservableBuffers,
) -> Result<f64, crate::error::HotSpringError> {
    let mut enc = gpu.begin_encoder("obs_plaq");
    encode_plaquette_reduce(&mut enc, gpu, pipelines, gauge, reduce_pl, obs);
    enc.copy_buffer_to_buffer(&obs.plaq_sum_buf, 0, &obs.staging, 0, 8);
    gpu.submit_encoder(enc);

    let data = gpu.read_staging_f64_n(&obs.staging, 1).map_err(|e| {
        crate::error::HotSpringError::GpuCompute(format!(
            "plaquette readback failed (GPU lost?): {e}"
        ))
    })?;
    Ok(data[0] / (6.0 * gauge.volume as f64))
}
