// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU buffer allocation for true multi-shift CG.

use super::TrueMultiShiftPipelines;
use crate::lattice::gpu_hmc::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
use crate::lattice::gpu_hmc::resident_cg_buffers::{ReduceChain, build_reduce_chain_pub};
use crate::lattice::gpu_hmc::{GpuF64, make_u32x4_params};

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
    let pbuf = gpu.create_uniform_buffer(&params, "ms_dirac_p");
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

/// GPU buffers for true multi-shift CG.
pub struct CgBuffers {
    pub p_shift_bufs: Vec<wgpu::Buffer>,
    pub zeta_curr_buf: wgpu::Buffer,
    pub zeta_prev_buf: wgpu::Buffer,
    pub alpha_s_buf: wgpu::Buffer,
    pub beta_ratio_buf: wgpu::Buffer,
    pub sigma_buf: wgpu::Buffer,
    pub zeta_params_buf: wgpu::Buffer,
    pub rz_buf: wgpu::Buffer,
    pub rz_new_buf: wgpu::Buffer,
    pub pap_buf: wgpu::Buffer,
    pub pp_buf: wgpu::Buffer,
    pub sigma_min_buf: wgpu::Buffer,
    pub alpha_buf: wgpu::Buffer,
    pub beta_buf: wgpu::Buffer,
    pub alpha_prev_buf: wgpu::Buffer,
    pub beta_prev_buf: wgpu::Buffer,
    pub scratch_a: wgpu::Buffer,
    pub scratch_b: wgpu::Buffer,
    pub reduce_to_pap: ReduceChain,
    pub reduce_to_pp: ReduceChain,
    pub reduce_to_rz: ReduceChain,
    pub reduce_to_rz_new: ReduceChain,
    pub convergence_staging: wgpu::Buffer,
    pub dirac_d_bg: wgpu::BindGroup,
    pub dirac_ddag_bg: wgpu::BindGroup,
    pub dot_pap_bg: wgpu::BindGroup,
    pub dot_pp_bg: wgpu::BindGroup,
    pub dot_rr_bg: wgpu::BindGroup,
    pub compute_alpha_bg: wgpu::BindGroup,
    pub compute_beta_bg: wgpu::BindGroup,
    pub update_xr_bg: wgpu::BindGroup,
    pub update_p_bg: wgpu::BindGroup,
    pub zeta_bg: wgpu::BindGroup,
    pub ms_p_bgs: Vec<wgpu::BindGroup>,
    pub xr_params_buf: wgpu::Buffer,
    pub p_params_buf: wgpu::Buffer,
    pub wg_dirac: u32,
    pub wg_dot: u32,
    pub wg_vec: u32,
    pub n_shifts: usize,
    pub n_flat: usize,
}

pub type TrueMultiShiftBuffers = CgBuffers;

impl CgBuffers {
    /// Allocate multi-shift CG buffers for `n_shifts` shifts.
    #[must_use]
    pub fn new(
        gpu: &GpuF64,
        dyn_pipelines: &GpuDynHmcPipelines,
        ms_pipelines: &TrueMultiShiftPipelines,
        state: &GpuDynHmcState,
        n_shifts: usize,
    ) -> Self {
        let vol = state.gauge.volume;
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let max_wg = n_pairs.div_ceil(256);
        let scratch_a = gpu.create_f64_output_buffer(max_wg.max(1), "ms_scratch_a");
        let scratch_b = gpu.create_f64_output_buffer(max_wg.max(1), "ms_scratch_b");

        let p_shift_bufs: Vec<_> = (0..n_shifts)
            .map(|s| gpu.create_f64_output_buffer(n_flat, &format!("ms_p_{s}")))
            .collect();

        let zeta_curr_buf = gpu.create_f64_output_buffer(n_shifts, "ms_zeta_curr");
        let zeta_prev_buf = gpu.create_f64_output_buffer(n_shifts, "ms_zeta_prev");
        let alpha_s_buf = gpu.create_f64_output_buffer(n_shifts, "ms_alpha_s");
        let beta_ratio_buf = gpu.create_f64_output_buffer(n_shifts, "ms_beta_ratio");
        let sigma_buf = gpu.create_f64_output_buffer(n_shifts, "ms_sigma");
        let zeta_params = make_u32x4_params(n_shifts as u32);
        let zeta_params_buf = gpu.create_uniform_buffer(&zeta_params, "ms_zeta_params");

        let rz_buf = gpu.create_f64_output_buffer(1, "ms_rz");
        let rz_new_buf = gpu.create_f64_output_buffer(1, "ms_rz_new");
        let pap_buf = gpu.create_f64_output_buffer(1, "ms_pap");
        let pp_buf = gpu.create_f64_output_buffer(1, "ms_pp");
        let sigma_min_buf = gpu.create_f64_output_buffer(1, "ms_sigma_min");
        let alpha_buf = gpu.create_f64_output_buffer(1, "ms_alpha");
        let beta_buf = gpu.create_f64_output_buffer(1, "ms_beta");
        let alpha_prev_buf = gpu.create_f64_output_buffer(1, "ms_alpha_prev");
        let beta_prev_buf = gpu.create_f64_output_buffer(1, "ms_beta_prev");

        let convergence_staging = gpu.create_staging_buffer(8, "ms_conv_staging");

        // Reduce chains for dot products → scalar accumulators
        let reduce_to_pap = build_reduce_chain_pub(
            gpu,
            &ms_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &pap_buf,
            n_pairs,
        );
        let reduce_to_pp = build_reduce_chain_pub(
            gpu,
            &ms_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &pp_buf,
            n_pairs,
        );
        let reduce_to_rz = build_reduce_chain_pub(
            gpu,
            &ms_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &rz_buf,
            n_pairs,
        );
        let reduce_to_rz_new = build_reduce_chain_pub(
            gpu,
            &ms_pipelines.base.reduce_pipeline,
            &state.dot_buf,
            &scratch_a,
            &scratch_b,
            &rz_new_buf,
            n_pairs,
        );

        // Dirac: D·p → temp, D†·temp → ap
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

        // Dot products: <p|ap>, <p|p>, <r|r>
        let dot_params = make_u32x4_params(n_pairs as u32);
        let dot_pap_pbuf = gpu.create_uniform_buffer(&dot_params, "ms_dot_pap_p");
        let dot_pap_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_pap_pbuf, &state.p_buf, &state.ap_buf, &state.dot_buf],
        );
        let dot_pp_pbuf = gpu.create_uniform_buffer(&dot_params, "ms_dot_pp_p");
        let dot_pp_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_pp_pbuf, &state.p_buf, &state.p_buf, &state.dot_buf],
        );
        let dot_rr_pbuf = gpu.create_uniform_buffer(&dot_params, "ms_dot_rr_p");
        let dot_rr_bg = gpu.create_bind_group(
            &dyn_pipelines.dot_pipeline,
            &[&dot_rr_pbuf, &state.r_buf, &state.r_buf, &state.dot_buf],
        );

        // Shifted alpha: α = rz / (pAp + σ_min · pp)
        let compute_alpha_bg = gpu.create_bind_group(
            &ms_pipelines.shifted_alpha_pipeline,
            &[&rz_buf, &pap_buf, &pp_buf, &sigma_min_buf, &alpha_buf],
        );
        let compute_beta_bg = gpu.create_bind_group(
            &ms_pipelines.base.compute_beta_pipeline,
            &[&rz_new_buf, &rz_buf, &beta_buf],
        );

        // Shifted xr update: r -= α · (ap + σ_min · p)
        let xr_params = make_u32x4_params(n_flat as u32);
        let xr_params_buf = gpu.create_uniform_buffer(&xr_params, "ms_xr_p");
        let update_xr_bg = gpu.create_bind_group(
            &ms_pipelines.shifted_xr_pipeline,
            &[
                &xr_params_buf,
                &state.x_buf,
                &state.r_buf,
                &state.p_buf,
                &state.ap_buf,
                &alpha_buf,
                &sigma_min_buf,
            ],
        );
        let p_params = make_u32x4_params(n_flat as u32);
        let p_params_buf = gpu.create_uniform_buffer(&p_params, "ms_p_p");
        let update_p_bg = gpu.create_bind_group(
            &ms_pipelines.base.update_p_pipeline,
            &[&p_params_buf, &state.p_buf, &state.r_buf, &beta_buf],
        );

        let wg_dirac = (vol as u32).div_ceil(64);
        let wg_dot = (n_pairs as u32).div_ceil(64);
        let wg_vec = (n_flat as u32).div_ceil(64);

        // Pre-create bind groups for per-shift operations (avoid allocation in hot loop)
        let zeta_bg = gpu.create_bind_group(
            &ms_pipelines.zeta_pipeline,
            &[
                &zeta_params_buf,
                &sigma_buf,
                &zeta_curr_buf,
                &zeta_prev_buf,
                &alpha_s_buf,
                &beta_ratio_buf,
                &alpha_buf,
                &beta_prev_buf,
                &alpha_prev_buf,
            ],
        );

        let ms_p_bgs: Vec<_> = (0..n_shifts)
            .map(|s| {
                let mut params = [0u32; 4];
                params[0] = n_flat as u32;
                params[1] = s as u32;
                let pbuf =
                    gpu.create_uniform_buffer(bytemuck::cast_slice(&params), &format!("ms_pu_{s}"));
                gpu.create_bind_group(
                    &ms_pipelines.ms_p_pipeline,
                    &[
                        &pbuf,
                        &p_shift_bufs[s],
                        &state.r_buf,
                        &zeta_curr_buf,
                        &beta_ratio_buf,
                        &beta_buf,
                    ],
                )
            })
            .collect();

        Self {
            p_shift_bufs,
            zeta_curr_buf,
            zeta_prev_buf,
            alpha_s_buf,
            beta_ratio_buf,
            sigma_buf,
            zeta_params_buf,
            rz_buf,
            rz_new_buf,
            pap_buf,
            pp_buf,
            sigma_min_buf,
            alpha_buf,
            beta_buf,
            alpha_prev_buf,
            beta_prev_buf,
            scratch_a,
            scratch_b,
            reduce_to_pap,
            reduce_to_pp,
            reduce_to_rz,
            reduce_to_rz_new,
            convergence_staging,
            dirac_d_bg,
            dirac_ddag_bg,
            dot_pap_bg,
            dot_pp_bg,
            dot_rr_bg,
            compute_alpha_bg,
            compute_beta_bg,
            update_xr_bg,
            update_p_bg,
            zeta_bg,
            ms_p_bgs,
            xr_params_buf,
            p_params_buf,
            wg_dirac,
            wg_dot,
            wg_vec,
            n_shifts,
            n_flat,
        }
    }
}
