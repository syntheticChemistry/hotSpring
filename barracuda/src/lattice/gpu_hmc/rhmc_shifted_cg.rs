// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(deprecated, reason = "transitional — migration to new API pending")]

//! Legacy multi-shift CG with per-iteration CPU readback of dot products.
//!
//! This module holds the deprecated path used only by [`super::gpu_rhmc`]'s
//! legacy trajectory and parity tests. Production code should use
//! [`super::resident_shifted_cg::gpu_multi_shift_cg_solve_resident`] instead.

use super::super::dynamical::{GpuDynHmcPipelines, GpuDynHmcState, gpu_axpy};
use super::super::{GpuF64, gpu_dot_re};

use super::{GpuRhmcPipelines, GpuRhmcSectorBuffers, dirac_dispatch};

/// GPU multi-shift CG: solve (D†D + σ_s) x_s = b for all shifts.
///
/// # Deprecated — use `gpu_multi_shift_cg_solve_resident` instead
///
/// Wrapper around `gpu_shifted_cg_solve` which reads back dot products to
/// CPU every CG iteration. The resident version in `resident_shifted_cg.rs`
/// keeps scalars on GPU and checks convergence every ~50 iterations.
///
/// Returns total CG iterations across all shifts.
#[deprecated(
    note = "use gpu_multi_shift_cg_solve_resident from resident_shifted_cg.rs (~50x fewer sync points)"
)]
pub fn gpu_multi_shift_cg_solve(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    _rhmc_pipelines: &GpuRhmcPipelines,
    state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    b_buf: &wgpu::Buffer,
    shifts: &[f64],
    mass: f64,
    tol: f64,
    max_iter: usize,
) -> usize {
    let mut total_iters = 0;

    for (s, &sigma) in shifts.iter().enumerate() {
        #[expect(deprecated, reason = "transitional — migration to new API pending")]
        let iters = gpu_shifted_cg_solve(
            gpu,
            dyn_pipelines,
            state,
            &sector.x_bufs[s],
            b_buf,
            mass,
            sigma,
            tol,
            max_iter,
        );
        total_iters += iters;
    }

    total_iters
}

/// Single shifted CG: solve (D†D + σ) x = b.
///
/// # Deprecated — use `gpu_shifted_cg_solve_resident` instead
///
/// **Legacy path** — reads back dot products to CPU every iteration
/// via `gpu_dot_re`. For 7200 CG iters this means ~20,000 GPU→CPU
/// sync points. Use `gpu_shifted_cg_solve_resident` from
/// `resident_shifted_cg.rs` which keeps all scalars on GPU.
#[deprecated(note = "use gpu_shifted_cg_solve_resident from resident_shifted_cg.rs")]
#[expect(deprecated, reason = "transitional — migration to new API pending")]
fn gpu_shifted_cg_solve(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    x_buf: &wgpu::Buffer,
    b_buf: &wgpu::Buffer,
    mass: f64,
    sigma: f64,
    tol: f64,
    max_iter: usize,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let n_pairs = vol * 3;
    let gauge = &state.gauge;
    let phases = &state.phases_buf;

    // x = 0, r = b, p = b
    gpu.zero_buffer(x_buf, (n_flat * 8) as u64);
    {
        let mut enc = gpu.begin_encoder("scg_init");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        enc.copy_buffer_to_buffer(b_buf, 0, &state.p_buf, 0, (n_flat * 8) as u64);
        gpu.submit_encoder(enc);
    }

    let b_norm_sq = gpu_dot_re(
        gpu,
        &pipelines.dot_pipeline,
        &state.dot_buf,
        &state.r_buf,
        &state.r_buf,
        n_pairs,
    );
    if b_norm_sq < 1e-30 {
        return 0;
    }

    let tol_sq = tol * tol * b_norm_sq;
    let mut rz = b_norm_sq;
    let mut iterations = 0;

    for _iter in 0..max_iter {
        iterations += 1;

        // ap = D†D·p
        dirac_dispatch(
            gpu,
            pipelines,
            gauge,
            phases,
            &state.p_buf,
            &state.temp_buf,
            mass,
            1.0,
        );
        dirac_dispatch(
            gpu,
            pipelines,
            gauge,
            phases,
            &state.temp_buf,
            &state.ap_buf,
            mass,
            -1.0,
        );

        // pAp = ⟨p|D†D·p⟩ + σ·⟨p|p⟩
        let mut p_ap = gpu_dot_re(
            gpu,
            &pipelines.dot_pipeline,
            &state.dot_buf,
            &state.p_buf,
            &state.ap_buf,
            n_pairs,
        );
        if sigma.abs() > 1e-30 {
            let p_p = gpu_dot_re(
                gpu,
                &pipelines.dot_pipeline,
                &state.dot_buf,
                &state.p_buf,
                &state.p_buf,
                n_pairs,
            );
            p_ap += sigma * p_p;
        }

        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / p_ap;

        // x += α·p
        gpu_axpy(
            gpu,
            &pipelines.axpy_pipeline,
            alpha,
            &state.p_buf,
            x_buf,
            n_flat,
        );

        // r -= α·(D†D·p + σ·p)
        gpu_axpy(
            gpu,
            &pipelines.axpy_pipeline,
            -alpha,
            &state.ap_buf,
            &state.r_buf,
            n_flat,
        );
        if sigma.abs() > 1e-30 {
            gpu_axpy(
                gpu,
                &pipelines.axpy_pipeline,
                -alpha * sigma,
                &state.p_buf,
                &state.r_buf,
                n_flat,
            );
        }

        let rz_new = gpu_dot_re(
            gpu,
            &pipelines.dot_pipeline,
            &state.dot_buf,
            &state.r_buf,
            &state.r_buf,
            n_pairs,
        );

        if rz_new < tol_sq {
            break;
        }

        let beta_cg = rz_new / rz;
        rz = rz_new;

        super::super::dynamical::gpu_xpay(
            gpu,
            &pipelines.xpay_pipeline,
            &state.r_buf,
            beta_cg,
            &state.p_buf,
            n_flat,
        );
    }

    iterations
}
