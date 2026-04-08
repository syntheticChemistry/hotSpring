// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident shifted CG for RHMC multi-shift solves.
//!
//! Extends the resident CG pattern (zero per-iteration readback) to handle
//! shifted systems `(D†D + σ) x = b`. Each shift runs its own resident CG
//! with a `sigma` buffer on GPU. The only readback is 8 bytes every
//! `check_interval` iterations for convergence testing.
//!
//! This eliminates ~19,000 CPU-GPU syncs per RHMC trajectory (the dominant
//! bottleneck on small lattices where sync cost exceeds compute cost).

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
use super::resident_cg_buffers::{ReduceChain, build_reduce_chain_pub, encode_reduce_chain};
use super::resident_cg_pipelines::GpuResidentCgPipelines;
use super::{GpuF64, make_u32x4_params};

/// Compiled pipelines for GPU-resident shifted CG.
pub struct GpuResidentShiftedCgPipelines {
    /// Base resident CG pipelines (reduce, beta, update_p).
    pub base: GpuResidentCgPipelines,
    /// Shifted alpha: alpha = rz / (pAp + sigma * pp).
    pub compute_alpha_shifted_pipeline: wgpu::ComputePipeline,
    /// Shifted xr update: x += alpha*p, r -= alpha*(ap + sigma*p).
    pub update_xr_shifted_pipeline: wgpu::ComputePipeline,
}

impl GpuResidentShiftedCgPipelines {
    /// Compile all shifted CG pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            base: GpuResidentCgPipelines::new(gpu),
            compute_alpha_shifted_pipeline: gpu.create_pipeline_f64(
                super::super::cg::WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64,
                "cg_alpha_shifted",
            ),
            update_xr_shifted_pipeline: gpu.create_pipeline_f64(
                super::super::cg::WGSL_CG_UPDATE_XR_SHIFTED_F64,
                "cg_xr_shifted",
            ),
        }
    }
}

/// GPU buffers for shifted CG workspace (shared across all shifts in a multi-shift solve).
///
/// The x_buf-dependent bind group is NOT stored here — it's created per-shift
/// via `make_xr_bg()` since each shift writes to a different output buffer.
pub struct GpuResidentShiftedCgBuffers {
    /// rz = ⟨r|r⟩ (1 f64, GPU-resident).
    pub rz_buf: wgpu::Buffer,
    /// rz_new = ⟨r_new|r_new⟩ (1 f64, GPU-resident).
    pub rz_new_buf: wgpu::Buffer,
    /// pAp = ⟨p|D†D·p⟩ (1 f64, GPU-resident).
    pub pap_buf: wgpu::Buffer,
    /// pp = ⟨p|p⟩ (1 f64, GPU-resident).
    pub pp_buf: wgpu::Buffer,
    /// CG alpha scalar (1 f64, GPU-resident).
    pub alpha_buf: wgpu::Buffer,
    /// CG beta scalar (1 f64, GPU-resident).
    pub beta_buf: wgpu::Buffer,
    /// Shift value σ (1 f64, GPU-resident).
    pub sigma_buf: wgpu::Buffer,
    /// Reduction scratch (ping).
    pub scratch_a: wgpu::Buffer,
    /// Reduction scratch (pong).
    pub scratch_b: wgpu::Buffer,
    /// Staging for convergence readback (8 bytes, MAP_READ).
    pub convergence_staging: wgpu::Buffer,
    /// Uniform params for xr update (n_flat).
    pub xr_params_buf: wgpu::Buffer,
    pub(crate) dirac_d_bg: wgpu::BindGroup,
    pub(crate) dirac_ddag_bg: wgpu::BindGroup,
    pub(crate) dot_pap_bg: wgpu::BindGroup,
    pub(crate) dot_pp_bg: wgpu::BindGroup,
    pub(crate) dot_rr_bg: wgpu::BindGroup,
    pub(crate) reduce_to_pap: ReduceChain,
    pub(crate) reduce_to_pp: ReduceChain,
    pub(crate) reduce_to_rz: ReduceChain,
    pub(crate) reduce_to_rz_new: ReduceChain,
    pub(crate) compute_alpha_shifted_bg: wgpu::BindGroup,
    pub(crate) compute_beta_bg: wgpu::BindGroup,
    pub(crate) update_p_bg: wgpu::BindGroup,
    pub(crate) wg_dirac: u32,
    pub(crate) wg_dot: u32,
    pub(crate) wg_vec: u32,
}

impl GpuResidentShiftedCgBuffers {
    /// Allocate shared workspace for shifted CG solves.
    #[must_use]
    pub fn new(
        gpu: &GpuF64,
        dyn_pipelines: &GpuDynHmcPipelines,
        shifted_pipelines: &GpuResidentShiftedCgPipelines,
        state: &GpuDynHmcState,
    ) -> Self {
        let vol = state.gauge.volume;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let max_wg = n_pairs.div_ceil(256);
        let scratch_a = gpu.create_f64_output_buffer(max_wg.max(1), "scg_scratch_a");
        let scratch_b = gpu.create_f64_output_buffer(max_wg.max(1), "scg_scratch_b");

        let rz_buf = gpu.create_f64_output_buffer(1, "scg_rz");
        let rz_new_buf = gpu.create_f64_output_buffer(1, "scg_rz_new");
        let pap_buf = gpu.create_f64_output_buffer(1, "scg_pap");
        let pp_buf = gpu.create_f64_output_buffer(1, "scg_pp");
        let alpha_buf = gpu.create_f64_output_buffer(1, "scg_alpha");
        let beta_buf = gpu.create_f64_output_buffer(1, "scg_beta");
        let sigma_buf = gpu.create_f64_output_buffer(1, "scg_sigma");
        let convergence_staging = gpu.create_staging_buffer(8, "scg_conv_staging");

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
        let dot_pap_pbuf = gpu.create_uniform_buffer(&dot_params, "scg_dot_pap_p");
        let dot_pap_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_pap_pbuf, &state.p_buf, &state.ap_buf, &state.dot_buf],
        );
        let dot_pp_pbuf = gpu.create_uniform_buffer(&dot_params, "scg_dot_pp_p");
        let dot_pp_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_pp_pbuf, &state.p_buf, &state.p_buf, &state.dot_buf],
        );
        let dot_rr_pbuf = gpu.create_uniform_buffer(&dot_params, "scg_dot_rr_p");
        let dot_rr_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_rr_pbuf, &state.r_buf, &state.r_buf, &state.dot_buf],
        );

        let reduce_to_pap = build_reduce_chain_pub(
            gpu,
            &shifted_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &pap_buf,
            n_pairs,
        );
        let reduce_to_pp = build_reduce_chain_pub(
            gpu,
            &shifted_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &pp_buf,
            n_pairs,
        );
        let reduce_to_rz = build_reduce_chain_pub(
            gpu,
            &shifted_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &rz_buf,
            n_pairs,
        );
        let reduce_to_rz_new = build_reduce_chain_pub(
            gpu,
            &shifted_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &rz_new_buf,
            n_pairs,
        );

        let compute_alpha_shifted_bg = gpu.create_bind_group(
            &shifted_pipelines.compute_alpha_shifted_pipeline,
            &[&rz_buf, &pap_buf, &pp_buf, &sigma_buf, &alpha_buf],
        );
        let compute_beta_bg = gpu.create_bind_group(
            &shifted_pipelines.base.compute_beta_pipeline,
            &[&rz_new_buf, &rz_buf, &beta_buf],
        );

        let xr_params = make_u32x4_params(n_flat as u32);
        let xr_params_buf = gpu.create_uniform_buffer(&xr_params, "scg_xr_p");

        let p_params = make_u32x4_params(n_flat as u32);
        let p_pbuf = gpu.create_uniform_buffer(&p_params, "scg_p_p");
        let update_p_bg = gpu.create_bind_group(
            &shifted_pipelines.base.update_p_pipeline,
            &[&p_pbuf, &state.p_buf, &state.r_buf, &beta_buf],
        );

        let wg_dirac = (vol as u32).div_ceil(64);
        let wg_dot = (n_pairs as u32).div_ceil(64);
        let wg_vec = (n_flat as u32).div_ceil(64);

        Self {
            rz_buf,
            rz_new_buf,
            pap_buf,
            pp_buf,
            alpha_buf,
            beta_buf,
            sigma_buf,
            scratch_a,
            scratch_b,
            convergence_staging,
            xr_params_buf,
            dirac_d_bg,
            dirac_ddag_bg,
            dot_pap_bg,
            dot_pp_bg,
            dot_rr_bg,
            reduce_to_pap,
            reduce_to_pp,
            reduce_to_rz,
            reduce_to_rz_new,
            compute_alpha_shifted_bg,
            compute_beta_bg,
            update_p_bg,
            wg_dirac,
            wg_dot,
            wg_vec,
        }
    }

    /// Upload the shift value σ to GPU.
    pub fn set_sigma(&self, gpu: &GpuF64, sigma: f64) {
        gpu.upload_f64(&self.sigma_buf, &[sigma]);
    }

    /// Create the xr update bind group for a specific output buffer x_buf.
    ///
    /// Called once per shift at the start of each CG solve.
    #[must_use]
    pub fn make_xr_bg(
        &self,
        gpu: &GpuF64,
        pipeline: &wgpu::ComputePipeline,
        state: &GpuDynHmcState,
        x_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.create_bind_group(
            pipeline,
            &[
                &self.xr_params_buf,
                x_buf,
                &state.r_buf,
                &state.p_buf,
                &state.ap_buf,
                &self.alpha_buf,
                &self.sigma_buf,
            ],
        )
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
    let pbuf = gpu.create_uniform_buffer(&params, "scg_dirac_p");
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

/// Encode one batch of shifted CG iterations into a command encoder.
///
/// Each iteration:
///   1. D†D·p → ap  (2 Dirac dispatches)
///   2. ⟨p|ap⟩ → pap  (dot + reduce)
///   3. ⟨p|p⟩ → pp    (dot + reduce, for shift correction)
///   4. alpha = rz / (pap + sigma*pp)  (shifted alpha shader)
///   5. x += alpha*p, r -= alpha*(ap + sigma*p)  (shifted xr update)
///   6. ⟨r|r⟩ → rz_new  (dot + reduce)
///   7. beta = rz_new / rz, rz ← rz_new  (beta shader)
///   8. p = r + beta*p  (p update)
pub fn encode_shifted_cg_batch(
    enc: &mut wgpu::CommandEncoder,
    dyn_pipelines: &GpuDynHmcPipelines,
    shifted_pipelines: &GpuResidentShiftedCgPipelines,
    bufs: &GpuResidentShiftedCgBuffers,
    xr_bg: &wgpu::BindGroup,
    batch: usize,
) {
    for _ in 0..batch {
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dirac_pipeline,
            &bufs.dirac_d_bg,
            bufs.wg_dirac,
        );
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dirac_pipeline,
            &bufs.dirac_ddag_bg,
            bufs.wg_dirac,
        );
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dot_pipeline,
            &bufs.dot_pap_bg,
            bufs.wg_dot,
        );
        encode_reduce_chain(
            enc,
            &shifted_pipelines.base.reduce_pipeline,
            &bufs.reduce_to_pap,
        );
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dot_pipeline,
            &bufs.dot_pp_bg,
            bufs.wg_dot,
        );
        encode_reduce_chain(
            enc,
            &shifted_pipelines.base.reduce_pipeline,
            &bufs.reduce_to_pp,
        );
        GpuF64::encode_pass(
            enc,
            &shifted_pipelines.compute_alpha_shifted_pipeline,
            &bufs.compute_alpha_shifted_bg,
            1,
        );
        GpuF64::encode_pass(
            enc,
            &shifted_pipelines.update_xr_shifted_pipeline,
            xr_bg,
            bufs.wg_vec,
        );
        GpuF64::encode_pass(
            enc,
            &dyn_pipelines.dot_pipeline,
            &bufs.dot_rr_bg,
            bufs.wg_dot,
        );
        encode_reduce_chain(
            enc,
            &shifted_pipelines.base.reduce_pipeline,
            &bufs.reduce_to_rz_new,
        );
        GpuF64::encode_pass(
            enc,
            &shifted_pipelines.base.compute_beta_pipeline,
            &bufs.compute_beta_bg,
            1,
        );
        GpuF64::encode_pass(
            enc,
            &shifted_pipelines.base.update_p_pipeline,
            &bufs.update_p_bg,
            bufs.wg_vec,
        );
    }
}

/// Maximum batch size for exponential back-off convergence checking.
/// Limits wasted iterations past actual convergence while keeping GPU busy.
const CG_BACKOFF_CAP: usize = 2000;

/// GPU-resident shifted CG solver: (D†D + σ) x = b with minimal readback.
///
/// Uses exponential back-off: the first convergence check happens after
/// `check_interval` iterations, then the interval doubles each time until
/// convergence or `max_iter`. This reduces GPU→CPU sync points from
/// O(I/C) to O(log(I/C)) while keeping the GPU pipeline saturated.
///
/// Returns total CG iterations.
#[must_use]
pub fn gpu_shifted_cg_solve_resident(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    shifted_pipelines: &GpuResidentShiftedCgPipelines,
    state: &GpuDynHmcState,
    bufs: &GpuResidentShiftedCgBuffers,
    x_buf: &wgpu::Buffer,
    b_buf: &wgpu::Buffer,
    sigma: f64,
    tol: f64,
    max_iter: usize,
    check_interval: usize,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;

    bufs.set_sigma(gpu, sigma);

    let xr_bg = bufs.make_xr_bg(
        gpu,
        &shifted_pipelines.update_xr_shifted_pipeline,
        state,
        x_buf,
    );

    // x = 0, r = b, p = b, compute initial ||b||² → rz
    gpu.zero_buffer(x_buf, (n_flat * 8) as u64);
    {
        let mut enc = gpu.begin_encoder("scg_init");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        enc.copy_buffer_to_buffer(b_buf, 0, &state.p_buf, 0, (n_flat * 8) as u64);
        GpuF64::encode_pass(
            &mut enc,
            &dyn_pipelines.dot_pipeline,
            &bufs.dot_rr_bg,
            bufs.wg_dot,
        );
        encode_reduce_chain(
            &mut enc,
            &shifted_pipelines.base.reduce_pipeline,
            &bufs.reduce_to_rz,
        );
        enc.copy_buffer_to_buffer(&bufs.rz_buf, 0, &bufs.convergence_staging, 0, 8);
        gpu.submit_encoder(enc);
    }

    let b_norm_sq = match gpu.read_staging_f64(&bufs.convergence_staging) {
        Ok(v) => v.first().copied().unwrap_or(0.0),
        Err(_) => return 0,
    };
    if b_norm_sq < 1e-30 {
        return 0;
    }
    let tol_sq = tol * tol * b_norm_sq;
    let mut current_interval = check_interval.max(1);
    let mut total_iters = 0;

    loop {
        let batch = current_interval.min(max_iter - total_iters);
        if batch == 0 {
            break;
        }

        let mut enc = gpu.begin_encoder("scg_batch");
        encode_shifted_cg_batch(
            &mut enc,
            dyn_pipelines,
            shifted_pipelines,
            bufs,
            &xr_bg,
            batch,
        );
        enc.copy_buffer_to_buffer(&bufs.rz_new_buf, 0, &bufs.convergence_staging, 0, 8);
        gpu.submit_encoder(enc);
        total_iters += batch;

        let rz_new = match gpu.read_staging_f64(&bufs.convergence_staging) {
            Ok(v) => v.first().copied().unwrap_or(f64::MAX),
            Err(_) => break,
        };
        if rz_new < tol_sq || total_iters >= max_iter {
            break;
        }

        // Exponential back-off: double interval, keep GPU pipeline saturated
        current_interval = (current_interval * 2).min(CG_BACKOFF_CAP);
    }

    total_iters
}

/// GPU-resident multi-shift CG: solve (D†D + σ_s) x_s = b for all shifts.
///
/// Runs shifted resident CG sequentially per shift (shared workspace).
/// Returns total CG iterations across all shifts.
#[must_use]
pub fn gpu_multi_shift_cg_solve_resident(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    shifted_pipelines: &GpuResidentShiftedCgPipelines,
    state: &GpuDynHmcState,
    bufs: &GpuResidentShiftedCgBuffers,
    x_bufs: &[wgpu::Buffer],
    b_buf: &wgpu::Buffer,
    shifts: &[f64],
    tol: f64,
    max_iter: usize,
    check_interval: usize,
) -> usize {
    let mut total = 0;
    for (s, &sigma) in shifts.iter().enumerate() {
        total += gpu_shifted_cg_solve_resident(
            gpu,
            dyn_pipelines,
            shifted_pipelines,
            state,
            bufs,
            &x_bufs[s],
            b_buf,
            sigma,
            tol,
            max_iter,
            check_interval,
        );
    }
    total
}
