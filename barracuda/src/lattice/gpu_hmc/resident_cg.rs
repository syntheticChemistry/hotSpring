// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident conjugate gradient — zero per-iteration scalar readback.
//!
//! All CG scalars (alpha, beta, rz, pAp) live on GPU. The CPU only
//! reads back 8 bytes (one f64) every `check_interval` iterations
//! for convergence testing. This eliminates 245,000× of readback
//! volume compared to the per-iteration approach.

pub use super::resident_cg_async::{AsyncCgReadback, gpu_cg_solve_resident_async};
pub use super::resident_cg_brain::{
    BrainInterrupt, CgResidualUpdate, gpu_cg_solve_brain, gpu_dynamical_hmc_trajectory_brain,
};
pub use super::resident_cg_buffers::GpuResidentCgBuffers;
pub use super::resident_cg_pipelines::{
    GpuResidentCgPipelines, WGSL_CG_COMPUTE_ALPHA, WGSL_CG_COMPUTE_BETA, WGSL_CG_UPDATE_P,
    WGSL_CG_UPDATE_XR, WGSL_SUM_REDUCE,
};

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcResult, GpuDynHmcState};
use super::resident_cg_buffers::{encode_cg_batch, encode_reduce_chain};
use super::streaming::{GpuDynHmcStreamingPipelines, make_ferm_prng_params};
#[expect(deprecated, reason = "transitional — migration to new API pending")]
use super::{
    GpuF64, gpu_dirac_dispatch, gpu_dot_re, gpu_fermion_force_dispatch, gpu_force_dispatch,
    gpu_kinetic_energy, gpu_link_update_dispatch, gpu_mom_update_dispatch, gpu_plaquette,
    gpu_wilson_action, make_link_mom_params, make_prng_params,
};

/// Maximum batch size for exponential back-off convergence checking.
const CG_BACKOFF_CAP: usize = 2000;

/// GPU-resident CG solver: (D†D)x = b with minimal readback.
///
/// Uses exponential back-off: the first convergence check happens after
/// `check_interval` iterations, then the interval doubles each time until
/// convergence or `cg_max_iter`. This reduces GPU→CPU sync points from
/// O(I/C) to O(log(I/C)) while keeping the GPU pipeline saturated.
///
/// Returns total CG iterations.
#[must_use]
pub fn gpu_cg_solve_resident(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    b_buf: &wgpu::Buffer,
    check_interval: usize,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;

    gpu.zero_buffer(&state.x_buf, (n_flat * 8) as u64);
    {
        let mut enc = gpu.begin_encoder("rcg_init");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        enc.copy_buffer_to_buffer(b_buf, 0, &state.p_buf, 0, (n_flat * 8) as u64);
        GpuF64::encode_pass(
            &mut enc,
            &dyn_pipelines.dot_pipeline,
            &cg_bufs.dot_rr_bg,
            cg_bufs.wg_dot,
        );
        encode_reduce_chain(
            &mut enc,
            &resident_pipelines.reduce_pipeline,
            &cg_bufs.reduce_to_rz,
        );
        enc.copy_buffer_to_buffer(&cg_bufs.rz_buf, 0, &cg_bufs.convergence_staging_a, 0, 8);
        gpu.submit_encoder(enc);
    }
    let b_norm_sq = match gpu.read_staging_f64(&cg_bufs.convergence_staging_a) {
        Ok(v) => v.first().copied().unwrap_or(0.0),
        Err(_) => return 0,
    };
    if b_norm_sq < 1e-30 {
        return 0;
    }
    let tol_sq = state.cg_tol * state.cg_tol * b_norm_sq;
    let mut current_interval = check_interval.max(1);
    let mut total_iters = 0usize;

    loop {
        let batch = current_interval.min(state.cg_max_iter - total_iters);
        if batch == 0 {
            break;
        }

        let mut enc = gpu.begin_encoder("rcg_batch");
        encode_cg_batch(&mut enc, dyn_pipelines, resident_pipelines, cg_bufs, batch);
        enc.copy_buffer_to_buffer(&cg_bufs.rz_new_buf, 0, &cg_bufs.convergence_staging_a, 0, 8);
        gpu.submit_encoder(enc);
        total_iters += batch;

        let rz_new = match gpu.read_staging_f64(&cg_bufs.convergence_staging_a) {
            Ok(v) => v.first().copied().unwrap_or(f64::MAX),
            Err(_) => break,
        };
        if rz_new < tol_sq || total_iters >= state.cg_max_iter {
            break;
        }

        // Exponential back-off: double interval, keep GPU pipeline saturated
        current_interval = (current_interval * 2).min(CG_BACKOFF_CAP);
    }

    total_iters
}

/// Single `gpu_dot_re` for final S_f = φ†x — intentional, not per-iteration.
#[expect(deprecated, reason = "transitional — migration to new API pending")]
fn gpu_fermion_action_resident_single(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    phi_buf: &wgpu::Buffer,
    check_interval: usize,
) -> (f64, usize) {
    let iters = gpu_cg_solve_resident(
        gpu,
        dyn_pipelines,
        resident_pipelines,
        state,
        cg_bufs,
        phi_buf,
        check_interval,
    );

    let vol = state.gauge.volume;
    let n_pairs = vol * 3;
    let dot_val = gpu_dot_re(
        gpu,
        &dyn_pipelines.dot_pipeline,
        &state.dot_buf,
        phi_buf,
        &state.x_buf,
        n_pairs,
    );

    (dot_val, iters)
}

fn gpu_fermion_action_resident_all(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    check_interval: usize,
) -> (f64, usize) {
    let mut total_action = 0.0;
    let mut total_iters = 0;
    for phi_buf in &state.phi_bufs {
        let (sf, iters) = gpu_fermion_action_resident_single(
            gpu,
            dyn_pipelines,
            resident_pipelines,
            state,
            cg_bufs,
            phi_buf,
            check_interval,
        );
        total_action += sf;
        total_iters += iters;
    }
    (total_action, total_iters)
}

fn gpu_total_force_dispatch_resident(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    dt: f64,
    check_interval: usize,
) -> usize {
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;

    gpu_force_dispatch(gpu, &dyn_pipelines.gauge, gs);
    gpu_mom_update_dispatch(gpu, &dyn_pipelines.gauge, gs, dt);

    let mut total_cg = 0;
    for phi_buf in &state.phi_bufs {
        let cg_iters = gpu_cg_solve_resident(
            gpu,
            dyn_pipelines,
            resident_pipelines,
            state,
            cg_bufs,
            phi_buf,
            check_interval,
        );
        total_cg += cg_iters;

        gpu_dirac_dispatch(gpu, dyn_pipelines, state, &state.x_buf, &state.y_buf, 1.0);
        gpu_fermion_force_dispatch(gpu, dyn_pipelines, state);

        let ferm_mom_params = make_link_mom_params(n_links, -2.0 * dt, gpu.full_df64_mode);
        let ferm_mom_pbuf = gpu.create_uniform_buffer(&ferm_mom_params, "rcg_fmom_p");
        let ferm_mom_bg = gpu.create_bind_group(
            &dyn_pipelines.gauge.momentum_pipeline,
            &[&ferm_mom_pbuf, &state.ferm_force_buf, &gs.mom_buf],
        );
        gpu.dispatch(
            &dyn_pipelines.gauge.momentum_pipeline,
            &ferm_mom_bg,
            gs.wg_links,
        );
    }

    total_cg
}

/// Full dynamical fermion HMC trajectory with GPU-resident CG.
///
/// Transfer budget per trajectory:
///   CPU→GPU: 0 bytes (GPU PRNG for momenta + pseudofermion)
///   GPU→CPU: ~480 bytes for CG convergence + 24 bytes for ΔH
pub fn gpu_dynamical_hmc_trajectory_resident(
    gpu: &GpuF64,
    streaming_pipelines: &GpuDynHmcStreamingPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    n_md_steps: usize,
    dt: f64,
    traj_id: u32,
    seed: &mut u64,
    check_interval: usize,
) -> GpuDynHmcResult {
    let vol = state.gauge.volume;
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;
    let dp = &streaming_pipelines.dyn_hmc;

    {
        let mut enc = gpu.begin_encoder("rcg_prng");
        let mom_prng_params = make_prng_params(n_links as u32, traj_id, seed);
        let mom_prng_pbuf = gpu.create_uniform_buffer(&mom_prng_params, "rcg_mom_p");
        let mom_prng_bg = gpu.create_bind_group(
            &streaming_pipelines.momenta_prng_pipeline,
            &[&mom_prng_pbuf, &gs.mom_buf],
        );
        GpuF64::encode_pass(
            &mut enc,
            &streaming_pipelines.momenta_prng_pipeline,
            &mom_prng_bg,
            gs.wg_links,
        );

        gpu.submit_encoder(enc);
    }

    for (fi, phi_buf) in state.phi_bufs.iter().enumerate() {
        let mut enc = gpu.begin_encoder(&format!("rcg_ferm_prng_{fi}"));
        let ferm_prng_params = make_ferm_prng_params(vol as u32, traj_id + fi as u32 * 1000, seed);
        let ferm_prng_pbuf =
            gpu.create_uniform_buffer(&ferm_prng_params, &format!("rcg_ferm_p_{fi}"));
        let ferm_prng_bg = gpu.create_bind_group(
            &streaming_pipelines.fermion_prng_pipeline,
            &[&ferm_prng_pbuf, &state.temp_buf],
        );
        let wg_vol = (vol as u32).div_ceil(64);
        GpuF64::encode_pass(
            &mut enc,
            &streaming_pipelines.fermion_prng_pipeline,
            &ferm_prng_bg,
            wg_vol,
        );
        gpu.submit_encoder(enc);

        gpu_dirac_dispatch(
            gpu,
            &streaming_pipelines.dyn_hmc,
            state,
            &state.temp_buf,
            phi_buf,
            -1.0,
        );
    }

    {
        let mut enc = gpu.begin_encoder("rcg_backup");
        enc.copy_buffer_to_buffer(
            &gs.link_buf,
            0,
            &gs.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    // EVOLUTION(B2): GPU-resident Hamiltonian assembly — blocked on fused
    // gauge-action + fermion-force pipeline in barraCuda TensorSession.
    #[expect(deprecated, reason = "transitional — migration to new API pending")]
    let s_gauge_old = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_old = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action_resident_all(
        gpu,
        dp,
        resident_pipelines,
        state,
        cg_bufs,
        check_interval,
    );
    let h_old = s_gauge_old + t_old + s_ferm_old;
    let mut total_cg = cg_iters_old;

    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg1 = gpu_total_force_dispatch_resident(
            gpu,
            dp,
            resident_pipelines,
            state,
            cg_bufs,
            lam * dt,
            check_interval,
        );
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg2 = gpu_total_force_dispatch_resident(
            gpu,
            dp,
            resident_pipelines,
            state,
            cg_bufs,
            2.0f64.mul_add(-lam, 1.0) * dt,
            check_interval,
        );
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg3 = gpu_total_force_dispatch_resident(
            gpu,
            dp,
            resident_pipelines,
            state,
            cg_bufs,
            lam * dt,
            check_interval,
        );
        total_cg += cg1 + cg2 + cg3;
    }

    #[expect(deprecated, reason = "transitional — migration to new API pending")]
    let s_gauge_new = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_new = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action_resident_all(
        gpu,
        dp,
        resident_pipelines,
        state,
        cg_bufs,
        check_interval,
    );
    let h_new = s_gauge_new + t_new + s_ferm_new;
    total_cg += cg_iters_new;

    let delta_h = h_new - h_old;

    let r: f64 = super::super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("rcg_restore");
        enc.copy_buffer_to_buffer(
            &gs.link_backup,
            0,
            &gs.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    #[expect(deprecated, reason = "transitional — migration to new API pending")]
    let plaquette = gpu_plaquette(gpu, &dp.gauge, gs);

    GpuDynHmcResult {
        accepted,
        delta_h,
        plaquette,
        cg_iterations: total_cg,
    }
}
