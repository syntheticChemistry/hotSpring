// SPDX-License-Identifier: AGPL-3.0-only

//! ROP-accelerated fermion force accumulation (Tier 3 silicon routing).
//!
//! Uses `atomicAdd(i32)` in fixed-point to accumulate weighted force
//! contributions from multiple RHMC poles simultaneously, eliminating
//! N_poles sequential momentum-update dispatches.
//!
//! The fixed-point scale (2^20) provides ~6 significant digits — sufficient
//! for force accumulation where the Omelyan integrator error is O(dt^2).
//!
//! ## Flow
//!
//! 1. Zero the i32 atomic accumulation buffer
//! 2. For each pole: dispatch fused force+atomicAdd shader (independent, no barriers)
//! 3. Single conversion dispatch: momentum += f64(accum) / scale

use super::GpuF64;

const WGSL_FERM_FORCE_ACCUM_ROP: &str =
    include_str!("../shaders/su3_fermion_force_accumulate_rop_f64.wgsl");
const WGSL_ATOMIC_TO_MOMENTUM: &str =
    include_str!("../shaders/su3_force_atomic_to_momentum_f64.wgsl");

const FIXED_POINT_SCALE: f64 = 1_048_576.0; // 2^20

/// ROP-accelerated force accumulation pipelines and buffers.
pub struct RopForceAccumulator {
    /// Fused fermion force + atomic accumulation pipeline.
    pub force_accum_pipeline: wgpu::ComputePipeline,
    /// Conversion: i32 fixed-point → f64 momentum addition.
    pub convert_pipeline: wgpu::ComputePipeline,
    /// Atomic accumulation buffer (i32 fixed-point, 18 * n_links entries).
    pub accum_buf: wgpu::Buffer,
    /// Number of i32 entries (= 18 * volume * 4).
    pub n_values: u32,
    /// Workgroup count for force shader (volume / 64, ceil).
    pub wg_force: u32,
    /// Workgroup count for conversion shader (n_values / 256, ceil).
    pub wg_convert: u32,
}

impl RopForceAccumulator {
    /// Create the ROP force accumulator for a given lattice volume.
    #[must_use]
    pub fn new(gpu: &GpuF64, volume: usize) -> Self {
        let n_links = volume * 4;
        let n_values = (n_links * 18) as u32;

        let force_accum_pipeline =
            gpu.create_pipeline_f64(WGSL_FERM_FORCE_ACCUM_ROP, "rop_ferm_force_accum");
        let convert_pipeline =
            gpu.create_pipeline_f64(WGSL_ATOMIC_TO_MOMENTUM, "rop_atomic_to_mom");

        let accum_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("rop_force_accum"),
            size: u64::from(n_values) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wg_force = (volume as u32).div_ceil(64);
        let wg_convert = n_values.div_ceil(256);

        eprintln!(
            "[ROP] Force accumulator: {} links, {} atomic i32 entries, scale=2^20",
            n_links, n_values
        );

        Self {
            force_accum_pipeline,
            convert_pipeline,
            accum_buf,
            n_values,
            wg_force,
            wg_convert,
        }
    }

    /// Zero the accumulation buffer (call before starting pole dispatches).
    pub fn zero_accum(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.accum_buf, 0, None);
    }

    /// Build the uniform params for one pole's force+accumulate dispatch.
    #[must_use]
    pub fn make_pole_params(volume: u32, alpha_dt: f64) -> Vec<u8> {
        let mut v = Vec::with_capacity(24);
        v.extend_from_slice(&volume.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        let bits = alpha_dt.to_bits();
        let hi = (bits >> 32) as u32;
        let lo = bits as u32;
        v.extend_from_slice(&hi.to_le_bytes());
        v.extend_from_slice(&lo.to_le_bytes());
        v.extend_from_slice(&FIXED_POINT_SCALE.to_le_bytes());
        v
    }

    /// Build the uniform params for the final conversion dispatch.
    #[must_use]
    pub fn make_convert_params(&self) -> Vec<u8> {
        let inv_scale = 1.0 / FIXED_POINT_SCALE;
        let mut v = Vec::with_capacity(16);
        v.extend_from_slice(&self.n_values.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&inv_scale.to_le_bytes());
        v
    }

    /// Encode one pole's fused force+atomicAdd dispatch into a command encoder.
    pub fn encode_pole_dispatch(
        &self,
        gpu: &GpuF64,
        encoder: &mut wgpu::CommandEncoder,
        links_buf: &wgpu::Buffer,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
        nbr_buf: &wgpu::Buffer,
        phases_buf: &wgpu::Buffer,
        volume: u32,
        alpha_s_dt: f64,
    ) {
        let params = Self::make_pole_params(volume, alpha_s_dt);
        let params_buf = gpu.create_uniform_buffer(&params, "rop_pole_params");
        let bg = gpu.create_bind_group(
            &self.force_accum_pipeline,
            &[
                &params_buf,
                links_buf,
                x_buf,
                y_buf,
                nbr_buf,
                phases_buf,
                &self.accum_buf,
            ],
        );
        GpuF64::encode_pass(encoder, &self.force_accum_pipeline, &bg, self.wg_force);
    }

    /// Encode the final conversion: momentum += f64(accum) / scale.
    pub fn encode_convert_to_momentum(
        &self,
        gpu: &GpuF64,
        encoder: &mut wgpu::CommandEncoder,
        momentum_buf: &wgpu::Buffer,
    ) {
        let params = self.make_convert_params();
        let params_buf = gpu.create_uniform_buffer(&params, "rop_convert_params");
        let bg = gpu.create_bind_group(
            &self.convert_pipeline,
            &[&params_buf, &self.accum_buf, momentum_buf],
        );
        GpuF64::encode_pass(encoder, &self.convert_pipeline, &bg, self.wg_convert);
    }
}
