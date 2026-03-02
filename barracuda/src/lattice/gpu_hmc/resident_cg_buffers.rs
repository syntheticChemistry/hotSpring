// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-resident CG buffers and reduction chain.
//!
//! The reduce chain uses a multi-pass tree reduction (sum_reduce_f64.wgsl) to
//! accumulate dot products into rz, pap, rz_new. Unlike upstream
//! `ReduceScalarPipeline`, this is encode-only (no readback) and supports
//! arbitrary N via multiple passes. See `docs/REDUCE_PIPELINE_ANALYSIS.md`.

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
use super::resident_cg_pipelines::GpuResidentCgPipelines;
use super::{make_u32x4_params, GpuF64};

pub(crate) struct ReducePass {
    pub(crate) bg: wgpu::BindGroup,
    pub(crate) num_wg: u32,
}

pub(crate) struct ReduceChain {
    pub(crate) passes: Vec<ReducePass>,
}

/// GPU-resident CG buffers: scalar buffers + reduction scratch + staging.
pub struct GpuResidentCgBuffers {
    /// Scratch for multi-pass reduction (ping buffer).
    pub scratch_a: wgpu::Buffer,
    /// Scratch for multi-pass reduction (pong buffer).
    pub scratch_b: wgpu::Buffer,
    /// rz = <r|r> (1 f64, GPU-resident).
    pub rz_buf: wgpu::Buffer,
    /// rz_new = <r_new|r_new> (1 f64, GPU-resident).
    pub rz_new_buf: wgpu::Buffer,
    /// pAp = <p|Ap> (1 f64, GPU-resident).
    pub pap_buf: wgpu::Buffer,
    /// CG alpha scalar (1 f64, GPU-resident).
    pub alpha_buf: wgpu::Buffer,
    /// CG beta scalar (1 f64, GPU-resident).
    pub beta_buf: wgpu::Buffer,
    /// Staging buffer for convergence readback (8 bytes, MAP_READ).
    pub convergence_staging_a: wgpu::Buffer,
    /// Double-buffered staging for async readback (Level 4).
    pub convergence_staging_b: wgpu::Buffer,
    pub(crate) dirac_d_bg: wgpu::BindGroup,
    pub(crate) dirac_ddag_bg: wgpu::BindGroup,
    pub(crate) dot_pap_bg: wgpu::BindGroup,
    pub(crate) dot_rr_bg: wgpu::BindGroup,
    pub(crate) reduce_to_pap: ReduceChain,
    pub(crate) reduce_to_rz: ReduceChain,
    pub(crate) reduce_to_rz_new: ReduceChain,
    pub(crate) compute_alpha_bg: wgpu::BindGroup,
    pub(crate) compute_beta_bg: wgpu::BindGroup,
    pub(crate) update_xr_bg: wgpu::BindGroup,
    pub(crate) update_p_bg: wgpu::BindGroup,
    pub(crate) wg_dirac: u32,
    pub(crate) wg_dot: u32,
    pub(crate) wg_vec: u32,
    _n_pairs: usize,
}

impl GpuResidentCgBuffers {
    /// Allocate all GPU-resident CG buffers and pre-build bind groups.
    #[must_use]
    pub fn new(
        gpu: &GpuF64,
        dyn_pipelines: &GpuDynHmcPipelines,
        resident_pipelines: &GpuResidentCgPipelines,
        state: &GpuDynHmcState,
    ) -> Self {
        let vol = state.gauge.volume;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let max_wg = n_pairs.div_ceil(256);
        let scratch_a = gpu.create_f64_output_buffer(max_wg.max(1), "cg_scratch_a");
        let scratch_b = gpu.create_f64_output_buffer(max_wg.max(1), "cg_scratch_b");

        let rz_buf = gpu.create_f64_output_buffer(1, "cg_rz");
        let rz_new_buf = gpu.create_f64_output_buffer(1, "cg_rz_new");
        let pap_buf = gpu.create_f64_output_buffer(1, "cg_pap");
        let alpha_buf = gpu.create_f64_output_buffer(1, "cg_alpha");
        let beta_buf = gpu.create_f64_output_buffer(1, "cg_beta");

        let convergence_staging_a = gpu.create_staging_buffer(8, "cg_conv_staging_a");
        let convergence_staging_b = gpu.create_staging_buffer(8, "cg_conv_staging_b");

        let dirac_d_bg = make_dirac_bg(
            gpu,
            &dyn_pipelines.dirac_pipeline,
            state,
            &state.p_buf,
            &state.temp_buf,
            1.0,
        );
        let dirac_ddag_bg = make_dirac_bg(
            gpu,
            &dyn_pipelines.dirac_pipeline,
            state,
            &state.temp_buf,
            &state.ap_buf,
            -1.0,
        );

        let dot_params = make_u32x4_params(n_pairs as u32);
        let dot_pap_pbuf = gpu.create_uniform_buffer(&dot_params, "cg_dot_pap_p");
        let dot_pap_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_pap_pbuf, &state.p_buf, &state.ap_buf, &state.dot_buf],
        );
        let dot_rr_pbuf = gpu.create_uniform_buffer(&dot_params, "cg_dot_rr_p");
        let dot_rr_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_rr_pbuf, &state.r_buf, &state.r_buf, &state.dot_buf],
        );

        let reduce_to_pap = build_reduce_chain(
            gpu,
            &resident_pipelines.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &pap_buf,
            n_pairs,
        );
        let reduce_to_rz = build_reduce_chain(
            gpu,
            &resident_pipelines.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &rz_buf,
            n_pairs,
        );
        let reduce_to_rz_new = build_reduce_chain(
            gpu,
            &resident_pipelines.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &rz_new_buf,
            n_pairs,
        );

        let compute_alpha_bg = gpu.create_bind_group(
            &resident_pipelines.compute_alpha_pipeline,
            &[&rz_buf, &pap_buf, &alpha_buf],
        );
        let compute_beta_bg = gpu.create_bind_group(
            &resident_pipelines.compute_beta_pipeline,
            &[&rz_new_buf, &rz_buf, &beta_buf],
        );

        let xr_params = make_u32x4_params(n_flat as u32);
        let xr_pbuf = gpu.create_uniform_buffer(&xr_params, "cg_xr_p");
        let update_xr_bg = gpu.create_bind_group(
            &resident_pipelines.update_xr_pipeline,
            &[
                &xr_pbuf,
                &state.x_buf,
                &state.r_buf,
                &state.p_buf,
                &state.ap_buf,
                &alpha_buf,
            ],
        );
        let p_params = make_u32x4_params(n_flat as u32);
        let p_pbuf = gpu.create_uniform_buffer(&p_params, "cg_p_p");
        let update_p_bg = gpu.create_bind_group(
            &resident_pipelines.update_p_pipeline,
            &[&p_pbuf, &state.p_buf, &state.r_buf, &beta_buf],
        );

        let wg_dirac = (vol as u32).div_ceil(64);
        let wg_dot = (n_pairs as u32).div_ceil(64);
        let wg_vec = (n_flat as u32).div_ceil(64);

        Self {
            scratch_a,
            scratch_b,
            rz_buf,
            rz_new_buf,
            pap_buf,
            alpha_buf,
            beta_buf,
            convergence_staging_a,
            convergence_staging_b,
            dirac_d_bg,
            dirac_ddag_bg,
            dot_pap_bg,
            dot_rr_bg,
            reduce_to_pap,
            reduce_to_rz,
            reduce_to_rz_new,
            compute_alpha_bg,
            compute_beta_bg,
            update_xr_bg,
            update_p_bg,
            wg_dirac,
            wg_dot,
            wg_vec,
            _n_pairs: n_pairs,
        }
    }
}

fn make_dirac_bg(
    gpu: &GpuF64,
    dirac_pl: &wgpu::ComputePipeline,
    state: &GpuDynHmcState,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    hop_sign: f64,
) -> wgpu::BindGroup {
    let vol = state.gauge.volume;
    let mut params = Vec::with_capacity(24);
    params.extend_from_slice(&(vol as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&state.mass.to_le_bytes());
    params.extend_from_slice(&hop_sign.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "cg_dirac_p");
    gpu.create_bind_group(
        dirac_pl,
        &[
            &pbuf,
            &state.gauge.link_buf,
            input,
            output,
            &state.gauge.nbr_buf,
            &state.phases_buf,
        ],
    )
}

fn build_reduce_chain(
    gpu: &GpuF64,
    reduce_pl: &wgpu::ComputePipeline,
    input: &wgpu::Buffer,
    scratch_a: &wgpu::Buffer,
    scratch_b: &wgpu::Buffer,
    target: &wgpu::Buffer,
    n: usize,
) -> ReduceChain {
    let mut passes = Vec::new();
    let mut current_n = n;
    let mut src = input;
    let mut use_a = true;

    while current_n > 1 {
        let num_wg = current_n.div_ceil(256);
        let dst = if num_wg == 1 {
            target
        } else if use_a {
            scratch_a
        } else {
            scratch_b
        };

        let params = make_reduce_params(current_n as u32);
        let pbuf = gpu.create_uniform_buffer(&params, "cg_reduce_p");
        let bg = gpu.create_bind_group(reduce_pl, &[src, dst, &pbuf]);
        passes.push(ReducePass {
            bg,
            num_wg: num_wg as u32,
        });

        current_n = num_wg;
        src = dst;
        if num_wg > 1 {
            use_a = !use_a;
        }
    }

    ReduceChain { passes }
}

fn make_reduce_params(size: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&size.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

/// Encode a reduction chain into a command encoder.
pub(crate) fn encode_reduce_chain(
    enc: &mut wgpu::CommandEncoder,
    reduce_pl: &wgpu::ComputePipeline,
    chain: &ReduceChain,
) {
    for pass in &chain.passes {
        GpuF64::encode_pass(enc, reduce_pl, &pass.bg, pass.num_wg);
    }
}

/// Encode one batch of CG iterations into a command encoder.
pub(crate) fn encode_cg_batch(
    enc: &mut wgpu::CommandEncoder,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    cg_bufs: &GpuResidentCgBuffers,
    batch: usize,
) {
    for _ in 0..batch {
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dirac_pipeline,
            &cg_bufs.dirac_d_bg,
            cg_bufs.wg_dirac,
        );
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dirac_pipeline,
            &cg_bufs.dirac_ddag_bg,
            cg_bufs.wg_dirac,
        );
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dot_pipeline,
            &cg_bufs.dot_pap_bg,
            cg_bufs.wg_dot,
        );
        encode_reduce_chain(
            enc,
            &resident_pipelines.reduce_pipeline,
            &cg_bufs.reduce_to_pap,
        );
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.compute_alpha_pipeline,
            &cg_bufs.compute_alpha_bg,
            1,
        );
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.update_xr_pipeline,
            &cg_bufs.update_xr_bg,
            cg_bufs.wg_vec,
        );
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dot_pipeline,
            &cg_bufs.dot_rr_bg,
            cg_bufs.wg_dot,
        );
        encode_reduce_chain(
            enc,
            &resident_pipelines.reduce_pipeline,
            &cg_bufs.reduce_to_rz_new,
        );
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.compute_beta_pipeline,
            &cg_bufs.compute_beta_bg,
            1,
        );
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.update_p_pipeline,
            &cg_bufs.update_p_bg,
            cg_bufs.wg_vec,
        );
    }
}
