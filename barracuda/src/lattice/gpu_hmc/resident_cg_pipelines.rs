// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-resident CG pipeline compilation.
//!
//! ## Reduce pipeline vs upstream ReduceScalarPipeline
//!
//! The upstream barracuda crate provides `barracuda::pipeline::ReduceScalarPipeline`
//! (sum_reduce_f64.wgsl). We use a local multi-pass reduce because:
//! - **GPU-resident:** CG needs encode-only, no readback; result feeds compute_alpha/compute_beta.
//! - **Arbitrary N:** Upstream does 2 passes only (N ≤ 65,536); we need 3+ passes for 32⁴ lattices.
//!
//! See `docs/REDUCE_PIPELINE_ANALYSIS.md` for full analysis.

use super::GpuF64;

/// Shader constants for GPU-resident CG.
pub const WGSL_SUM_REDUCE: &str = super::super::cg::WGSL_SUM_REDUCE_F64;
/// CG α = ⟨r,r⟩ / ⟨p,Ap⟩ shader.
pub const WGSL_CG_COMPUTE_ALPHA: &str = super::super::cg::WGSL_CG_COMPUTE_ALPHA_F64;
/// CG β = ⟨r_new,r_new⟩ / ⟨r_old,r_old⟩ shader.
pub const WGSL_CG_COMPUTE_BETA: &str = super::super::cg::WGSL_CG_COMPUTE_BETA_F64;
/// CG x += α·p and r -= α·Ap update shader.
pub const WGSL_CG_UPDATE_XR: &str = super::super::cg::WGSL_CG_UPDATE_XR_F64;
/// CG p = r + β·p search-direction update shader.
pub const WGSL_CG_UPDATE_P: &str = super::super::cg::WGSL_CG_UPDATE_P_F64;

/// Compiled pipelines for GPU-resident CG.
pub struct GpuResidentCgPipelines {
    /// Parallel sum-reduce for dot-product accumulation.
    pub reduce_pipeline: wgpu::ComputePipeline,
    /// Step-length α = ⟨r,r⟩ / ⟨p,Ap⟩ computation.
    pub compute_alpha_pipeline: wgpu::ComputePipeline,
    /// Search-direction scaling β = ⟨r',r'⟩ / ⟨r,r⟩ computation.
    pub compute_beta_pipeline: wgpu::ComputePipeline,
    /// Solution x and residual r update in one dispatch.
    pub update_xr_pipeline: wgpu::ComputePipeline,
    /// Conjugate search-direction p update.
    pub update_p_pipeline: wgpu::ComputePipeline,
}

impl GpuResidentCgPipelines {
    /// Compile all GPU-resident CG pipelines (reduce, alpha, beta, update_xr, update_p).
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            reduce_pipeline: gpu.create_pipeline_f64(WGSL_SUM_REDUCE, "cg_reduce"),
            compute_alpha_pipeline: gpu.create_pipeline_f64(WGSL_CG_COMPUTE_ALPHA, "cg_alpha"),
            compute_beta_pipeline: gpu.create_pipeline_f64(WGSL_CG_COMPUTE_BETA, "cg_beta"),
            update_xr_pipeline: gpu.create_pipeline_f64(WGSL_CG_UPDATE_XR, "cg_update_xr"),
            update_p_pipeline: gpu.create_pipeline_f64(WGSL_CG_UPDATE_P, "cg_update_p"),
        }
    }
}
