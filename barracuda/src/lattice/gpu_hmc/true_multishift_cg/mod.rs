// SPDX-License-Identifier: AGPL-3.0-or-later

//! True multi-shift CG: shared Krylov basis across all shifts.

mod ms_cg_buffers;
mod ms_cg_convergence;
mod ms_cg_iteration;

pub use ms_cg_buffers::{CgBuffers, TrueMultiShiftBuffers};
pub use ms_cg_convergence::{
    ms_cg_batch_size, ms_cg_next_check_interval, ms_cg_should_stop, ms_cg_tolerance_sq,
};

use barracuda::ops::lattice::absorbed_shaders::{
    WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64, WGSL_CG_UPDATE_XR_SHIFTED_F64, WGSL_MS_P_UPDATE_F64,
    WGSL_MS_X_UPDATE_F64, WGSL_MS_ZETA_UPDATE_F64,
};

use crate::lattice::gpu_hmc::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
use crate::lattice::gpu_hmc::resident_cg_buffers::encode_reduce_chain;
use crate::lattice::gpu_hmc::resident_cg_pipelines::GpuResidentCgPipelines;
use crate::lattice::gpu_hmc::GpuF64;
#[cfg(test)]
use crate::lattice::gpu_hmc::make_u32x4_params;

const WGSL_MS_ZETA: &str = WGSL_MS_ZETA_UPDATE_F64;
const WGSL_MS_X_UPDATE: &str = WGSL_MS_X_UPDATE_F64;
const WGSL_MS_P_UPDATE: &str = WGSL_MS_P_UPDATE_F64;
const WGSL_CG_ALPHA_SHIFTED: &str = WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64;
const WGSL_CG_XR_SHIFTED: &str = WGSL_CG_UPDATE_XR_SHIFTED_F64;

/// Pipelines for true multi-shift CG.
pub struct TrueMultiShiftPipelines {
    pub base: GpuResidentCgPipelines,
    pub shifted_alpha_pipeline: wgpu::ComputePipeline,
    pub shifted_xr_pipeline: wgpu::ComputePipeline,
    pub zeta_pipeline: wgpu::ComputePipeline,
    pub ms_x_pipeline: wgpu::ComputePipeline,
    pub ms_p_pipeline: wgpu::ComputePipeline,
}

impl TrueMultiShiftPipelines {
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            base: GpuResidentCgPipelines::new(gpu),
            shifted_alpha_pipeline: gpu
                .create_pipeline_f64(WGSL_CG_ALPHA_SHIFTED, "ms_alpha_shifted"),
            shifted_xr_pipeline: gpu.create_pipeline_f64(WGSL_CG_XR_SHIFTED, "ms_xr_shifted"),
            zeta_pipeline: gpu.create_pipeline_f64(WGSL_MS_ZETA, "ms_zeta"),
            ms_x_pipeline: gpu.create_pipeline_f64(WGSL_MS_X_UPDATE, "ms_x_update"),
            ms_p_pipeline: gpu.create_pipeline_f64(WGSL_MS_P_UPDATE, "ms_p_update"),
        }
    }
}

/// True multi-shift CG: solve (D†D + σ_s) x_s = b for all shifts simultaneously.
#[must_use]
#[inline(never)]
pub fn gpu_true_multi_shift_cg_solve(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    ms_pipelines: &TrueMultiShiftPipelines,
    state: &GpuDynHmcState,
    bufs: &CgBuffers,
    x_bufs: &[wgpu::Buffer],
    b_buf: &wgpu::Buffer,
    shifts: &[f64],
    tol: f64,
    max_iter: usize,
    check_interval: usize,
) -> usize {
    let n_flat = bufs.n_flat;
    let n_shifts = shifts.len();
    assert_eq!(n_shifts, bufs.n_shifts);

    let sigma_min = shifts.iter().copied().fold(f64::INFINITY, f64::min);
    #[expect(
        clippy::float_cmp,
        reason = "exact equality required for determinism check"
    )]
    let i_min = shifts.iter().position(|&s| s == sigma_min).unwrap_or(0);
    let effective_shifts: Vec<f64> = shifts.iter().map(|&s| s - sigma_min).collect();

    gpu.upload_f64(&bufs.sigma_buf, &effective_shifts);
    gpu.upload_f64(&bufs.sigma_min_buf, &[sigma_min]);

    for xb in x_bufs.iter().take(n_shifts) {
        gpu.zero_buffer(xb, (n_flat * 8) as u64);
    }
    gpu.zero_buffer(&state.x_buf, (n_flat * 8) as u64);

    {
        let mut enc = gpu.begin_encoder("ms_init");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        enc.copy_buffer_to_buffer(b_buf, 0, &state.p_buf, 0, (n_flat * 8) as u64);
        for s in 0..n_shifts {
            enc.copy_buffer_to_buffer(b_buf, 0, &bufs.p_shift_bufs[s], 0, (n_flat * 8) as u64);
        }

        GpuF64::encode_pass(
            &mut enc,
            &dyn_pipelines.dot_pipeline,
            &bufs.dot_rr_bg,
            bufs.wg_dot,
        );
        encode_reduce_chain(
            &mut enc,
            &ms_pipelines.base.reduce_pipeline,
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
    let tol_sq = ms_cg_tolerance_sq(tol, b_norm_sq);

    let ones = vec![1.0_f64; n_shifts];
    gpu.upload_f64(&bufs.zeta_curr_buf, &ones);
    gpu.upload_f64(&bufs.zeta_prev_buf, &ones);
    gpu.upload_f64(&bufs.alpha_prev_buf, &[1.0_f64]);
    gpu.upload_f64(&bufs.beta_prev_buf, &[0.0_f64]);

    let ms_x_bgs: Vec<_> = (0..n_shifts)
        .map(|s| {
            let mut params = [0u32; 4];
            params[0] = n_flat as u32;
            params[1] = s as u32;
            let pbuf =
                gpu.create_uniform_buffer(bytemuck::cast_slice(&params), &format!("ms_xu_{s}"));
            gpu.create_bind_group(
                &ms_pipelines.ms_x_pipeline,
                &[&pbuf, &x_bufs[s], &bufs.p_shift_bufs[s], &bufs.alpha_s_buf],
            )
        })
        .collect();

    let mut current_interval = check_interval.max(1);
    let mut total_iters = 0;

    loop {
        let batch = ms_cg_batch_size(current_interval, total_iters, max_iter);
        if batch == 0 {
            break;
        }

        let mut enc = gpu.begin_encoder("ms_batch");
        bufs.encode_iteration_batch(
            &mut enc,
            dyn_pipelines,
            ms_pipelines,
            &ms_x_bgs,
            batch,
        );
        enc.copy_buffer_to_buffer(&bufs.rz_new_buf, 0, &bufs.convergence_staging, 0, 8);
        gpu.submit_encoder(enc);
        total_iters += batch;

        let rz_new = match gpu.read_staging_f64(&bufs.convergence_staging) {
            Ok(v) => v.first().copied().unwrap_or(f64::MAX),
            Err(_) => break,
        };
        if ms_cg_should_stop(rz_new, tol_sq, total_iters, max_iter) {
            break;
        }

        current_interval = ms_cg_next_check_interval(current_interval);
    }

    {
        let mut enc = gpu.begin_encoder("ms_base_copy");
        enc.copy_buffer_to_buffer(&state.x_buf, 0, &x_bufs[i_min], 0, (n_flat * 8) as u64);
        gpu.submit_encoder(enc);
    }

    total_iters
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, clippy::expect_used, reason = "test assertions")]

    use super::*;

    fn make_gpu() -> GpuF64 {
        let rt = tokio::runtime::Runtime::new().expect("tokio");
        rt.block_on(GpuF64::new()).expect("GPU required for test")
    }

    #[test]
    fn zeta_recurrence_first_iteration() {
        let gpu = make_gpu();
        let n_shifts = 3usize;
        let sigma = [0.1, 0.5, 1.0];

        let pl = gpu.create_pipeline_f64(WGSL_MS_ZETA, "test_zeta");

        let params = make_u32x4_params(n_shifts as u32);
        let params_buf = gpu.create_uniform_buffer(&params, "tz_params");
        let sigma_buf = gpu.create_f64_output_buffer(n_shifts, "tz_sigma");
        let zeta_curr_buf = gpu.create_f64_output_buffer(n_shifts, "tz_zc");
        let zeta_prev_buf = gpu.create_f64_output_buffer(n_shifts, "tz_zp");
        let alpha_s_buf = gpu.create_f64_output_buffer(n_shifts, "tz_as");
        let beta_ratio_buf = gpu.create_f64_output_buffer(n_shifts, "tz_br");
        let alpha_buf = gpu.create_f64_output_buffer(1, "tz_alpha");
        let beta_prev_buf = gpu.create_f64_output_buffer(1, "tz_bprev");
        let alpha_prev_buf = gpu.create_f64_output_buffer(1, "tz_aprev");

        gpu.upload_f64(&sigma_buf, &sigma);
        gpu.upload_f64(&zeta_curr_buf, &[1.0, 1.0, 1.0]);
        gpu.upload_f64(&zeta_prev_buf, &[1.0, 1.0, 1.0]);
        gpu.upload_f64(&alpha_buf, &[0.3]);
        gpu.upload_f64(&beta_prev_buf, &[0.0]);
        gpu.upload_f64(&alpha_prev_buf, &[1.0]);

        let bg = gpu.create_bind_group(
            &pl,
            &[
                &params_buf,
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

        let mut enc = gpu.begin_encoder("test_zeta");
        GpuF64::encode_pass(&mut enc, &pl, &bg, 1);
        gpu.submit_encoder(enc);

        let zeta = gpu.read_back_f64(&zeta_curr_buf, n_shifts).unwrap();
        let alpha_s = gpu.read_back_f64(&alpha_s_buf, n_shifts).unwrap();

        for (s, &sig) in sigma.iter().enumerate() {
            let expected_z = 1.0 / (1.0 + sig * 0.3);
            let expected_a = 0.3 * expected_z;
            assert!(
                (zeta[s] - expected_z).abs() < 1e-10,
                "shift {s}: ζ expected {expected_z}, got {}",
                zeta[s]
            );
            assert!(
                (alpha_s[s] - expected_a).abs() < 1e-10,
                "shift {s}: α_s expected {expected_a}, got {}",
                alpha_s[s]
            );
        }
        eprintln!("  zeta recurrence passed: ζ={zeta:?}");
    }

    #[test]
    fn ms_x_update_kernel() {
        let gpu = make_gpu();
        let n = 64usize;
        let shift_idx = 1u32;
        let n_shifts = 3usize;

        let pl = gpu.create_pipeline_f64(WGSL_MS_X_UPDATE, "test_mx");

        let mut params = [0u32; 4];
        params[0] = n as u32;
        params[1] = shift_idx;
        let pbuf = gpu.create_uniform_buffer(bytemuck::cast_slice(&params), "tmx_p");
        let x_buf = gpu.create_f64_output_buffer(n, "tmx_x");
        let p_buf = gpu.create_f64_output_buffer(n, "tmx_p2");
        let alpha_s_buf = gpu.create_f64_output_buffer(n_shifts, "tmx_as");

        let x_init: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let p_init: Vec<f64> = (0..n).map(|_| 2.0).collect();
        gpu.upload_f64(&x_buf, &x_init);
        gpu.upload_f64(&p_buf, &p_init);
        gpu.upload_f64(&alpha_s_buf, &[0.0, 0.5, 0.0]);

        let bg = gpu.create_bind_group(&pl, &[&pbuf, &x_buf, &p_buf, &alpha_s_buf]);
        let wg = (n as u32).div_ceil(64);
        let mut enc = gpu.begin_encoder("test_mx");
        GpuF64::encode_pass(&mut enc, &pl, &bg, wg);
        gpu.submit_encoder(enc);

        let x_out = gpu.read_back_f64(&x_buf, n).unwrap();
        for (i, &val) in x_out.iter().enumerate() {
            let expected = i as f64 + 0.5 * 2.0;
            assert!(
                (val - expected).abs() < 1e-10,
                "x[{i}] = {val}, expected {expected}"
            );
        }
        eprintln!("  ms_x_update passed");
    }

    #[test]
    fn ms_p_update_kernel() {
        let gpu = make_gpu();
        let n = 64usize;
        let shift_idx = 0u32;
        let n_shifts = 2usize;

        let pl = gpu.create_pipeline_f64(WGSL_MS_P_UPDATE, "test_mp");

        let mut params = [0u32; 4];
        params[0] = n as u32;
        params[1] = shift_idx;
        let pbuf = gpu.create_uniform_buffer(bytemuck::cast_slice(&params), "tmp_p");
        let p_buf = gpu.create_f64_output_buffer(n, "tmp_p2");
        let r_buf = gpu.create_f64_output_buffer(n, "tmp_r");
        let zeta_buf = gpu.create_f64_output_buffer(n_shifts, "tmp_z");
        let beta_ratio_buf = gpu.create_f64_output_buffer(n_shifts, "tmp_br");
        let beta_base_buf = gpu.create_f64_output_buffer(1, "tmp_bb");

        let p_init: Vec<f64> = (0..n).map(|_| 3.0).collect();
        let r_init: Vec<f64> = (0..n).map(|_| 1.0).collect();
        gpu.upload_f64(&p_buf, &p_init);
        gpu.upload_f64(&r_buf, &r_init);
        gpu.upload_f64(&zeta_buf, &[0.8, 0.6]);
        gpu.upload_f64(&beta_ratio_buf, &[0.9, 0.7]);
        gpu.upload_f64(&beta_base_buf, &[0.4]);

        let bg = gpu.create_bind_group(
            &pl,
            &[
                &pbuf,
                &p_buf,
                &r_buf,
                &zeta_buf,
                &beta_ratio_buf,
                &beta_base_buf,
            ],
        );
        let wg = (n as u32).div_ceil(64);
        let mut enc = gpu.begin_encoder("test_mp");
        GpuF64::encode_pass(&mut enc, &pl, &bg, wg);
        gpu.submit_encoder(enc);

        let p_out = gpu.read_back_f64(&p_buf, n).unwrap();
        let beta_s = 0.9 * 0.9 * 0.4;
        let expected = 0.8 * 1.0 + beta_s * 3.0;
        for (i, &val) in p_out.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-10,
                "p[{i}] = {val}, expected {expected}"
            );
        }
        eprintln!("  ms_p_update passed: p[0] = {}", p_out[0]);
    }
}
