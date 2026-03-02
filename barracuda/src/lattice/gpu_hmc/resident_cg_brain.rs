// SPDX-License-Identifier: AGPL-3.0-only

//! Brain-mode CG: NPU residual streaming and interrupt support.

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcResult, GpuDynHmcState};
use super::resident_cg_buffers::{encode_cg_batch, encode_reduce_chain, GpuResidentCgBuffers};
use super::resident_cg_pipelines::GpuResidentCgPipelines;
use super::streaming::{make_ferm_prng_params, GpuDynHmcStreamingPipelines};
use super::{
    gpu_dirac_dispatch, gpu_dot_re, gpu_fermion_force_dispatch, gpu_force_dispatch,
    gpu_kinetic_energy, gpu_link_update_dispatch, gpu_mom_update_dispatch, gpu_plaquette,
    gpu_wilson_action, make_link_mom_params, make_prng_params, GpuF64,
};

/// CG residual update sent to the NPU cerebellum during a brain-mode solve.
#[derive(Debug, Clone)]
pub struct CgResidualUpdate {
    /// CG iteration count at this readback.
    pub iteration: usize,
    /// Current residual norm squared (un-normalized).
    pub rz_new: f64,
    /// Wall time of this batch in microseconds.
    pub batch_wall_us: u64,
}

/// Interrupt from the NPU cerebellum requesting corrective action.
#[derive(Debug, Clone)]
pub enum BrainInterrupt {
    /// Abort the current CG solve immediately.
    KillCg,
    /// Change the check interval for subsequent batches.
    AdjustCheckInterval(usize),
    /// Informational only — no action required.
    Info(String),
}

/// Brain-mode CG solver with NPU residual streaming and interrupt support.
///
/// Functionally identical to `gpu_cg_solve_resident` but streams residual
/// values to the NPU at every batch boundary and checks for interrupt
/// signals. This enables the NPU to monitor CG convergence in real time
/// and abort or adjust the solve if anomalies are detected.
pub fn gpu_cg_solve_brain(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    b_buf: &wgpu::Buffer,
    check_interval: usize,
    residual_tx: &std::sync::mpsc::Sender<CgResidualUpdate>,
    interrupt_rx: &std::sync::mpsc::Receiver<BrainInterrupt>,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;

    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(&state.x_buf, &zeros);
    {
        let mut enc = gpu.begin_encoder("rcg_brain_init");
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
    let mut current_check_interval = check_interval.max(1);
    let mut total_iters = 0usize;

    loop {
        let batch = current_check_interval.min(state.cg_max_iter - total_iters);
        if batch == 0 {
            break;
        }

        let batch_start = std::time::Instant::now();

        let mut enc = gpu.begin_encoder("rcg_brain_batch");
        encode_cg_batch(&mut enc, dyn_pipelines, resident_pipelines, cg_bufs, batch);
        enc.copy_buffer_to_buffer(&cg_bufs.rz_new_buf, 0, &cg_bufs.convergence_staging_a, 0, 8);
        gpu.submit_encoder(enc);
        total_iters += batch;

        let rz_new = match gpu.read_staging_f64(&cg_bufs.convergence_staging_a) {
            Ok(v) => v.first().copied().unwrap_or(f64::MAX),
            Err(_) => break,
        };

        let batch_wall_us = batch_start.elapsed().as_micros() as u64;

        let _ = residual_tx.send(CgResidualUpdate {
            iteration: total_iters,
            rz_new,
            batch_wall_us,
        });

        match interrupt_rx.try_recv() {
            Ok(BrainInterrupt::KillCg) | Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
            Ok(BrainInterrupt::AdjustCheckInterval(n)) => {
                current_check_interval = n.max(1);
            }
            Ok(BrainInterrupt::Info(_)) | Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }

        if rz_new < tol_sq || total_iters >= state.cg_max_iter {
            break;
        }
    }

    total_iters
}

fn gpu_fermion_action_brain(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    check_interval: usize,
    residual_tx: &std::sync::mpsc::Sender<CgResidualUpdate>,
    interrupt_rx: &std::sync::mpsc::Receiver<BrainInterrupt>,
) -> (f64, usize) {
    let iters = gpu_cg_solve_brain(
        gpu,
        dyn_pipelines,
        resident_pipelines,
        state,
        cg_bufs,
        &state.phi_buf,
        check_interval,
        residual_tx,
        interrupt_rx,
    );

    let vol = state.gauge.volume;
    let n_pairs = vol * 3;
    let dot_val = gpu_dot_re(
        gpu,
        &dyn_pipelines.dot_pipeline,
        &state.dot_buf,
        &state.phi_buf,
        &state.x_buf,
        n_pairs,
    );

    (dot_val, iters)
}

fn gpu_total_force_dispatch_brain(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    dt: f64,
    check_interval: usize,
    residual_tx: &std::sync::mpsc::Sender<CgResidualUpdate>,
    interrupt_rx: &std::sync::mpsc::Receiver<BrainInterrupt>,
) -> usize {
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;

    gpu_force_dispatch(gpu, &dyn_pipelines.gauge, gs);
    gpu_mom_update_dispatch(gpu, &dyn_pipelines.gauge, gs, dt);

    let cg_iters = gpu_cg_solve_brain(
        gpu,
        dyn_pipelines,
        resident_pipelines,
        state,
        cg_bufs,
        &state.phi_buf,
        check_interval,
        residual_tx,
        interrupt_rx,
    );

    gpu_dirac_dispatch(gpu, dyn_pipelines, state, &state.x_buf, &state.y_buf, 1.0);
    gpu_fermion_force_dispatch(gpu, dyn_pipelines, state);

    let ferm_mom_params = make_link_mom_params(n_links, -2.0 * dt);
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

    cg_iters
}

/// Full dynamical fermion HMC trajectory with brain-mode CG.
///
/// Identical to `gpu_dynamical_hmc_trajectory_resident` but streams CG
/// residuals to the NPU cerebellum and checks for interrupt signals at
/// every batch boundary. Enables real-time monitoring and adaptive
/// intervention during long CG solves.
pub fn gpu_dynamical_hmc_trajectory_brain(
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
    residual_tx: &std::sync::mpsc::Sender<CgResidualUpdate>,
    interrupt_rx: &std::sync::mpsc::Receiver<BrainInterrupt>,
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

        let ferm_prng_params = make_ferm_prng_params(vol as u32, traj_id, seed);
        let ferm_prng_pbuf = gpu.create_uniform_buffer(&ferm_prng_params, "rcg_ferm_p");
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
    }

    gpu_dirac_dispatch(
        gpu,
        &streaming_pipelines.dyn_hmc,
        state,
        &state.temp_buf,
        &state.phi_buf,
        -1.0,
    );

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

    let s_gauge_old = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_old = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action_brain(
        gpu,
        dp,
        resident_pipelines,
        state,
        cg_bufs,
        check_interval,
        residual_tx,
        interrupt_rx,
    );
    let h_old = s_gauge_old + t_old + s_ferm_old;
    let mut total_cg = cg_iters_old;

    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg1 = gpu_total_force_dispatch_brain(
            gpu,
            dp,
            resident_pipelines,
            state,
            cg_bufs,
            lam * dt,
            check_interval,
            residual_tx,
            interrupt_rx,
        );
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg2 = gpu_total_force_dispatch_brain(
            gpu,
            dp,
            resident_pipelines,
            state,
            cg_bufs,
            (1.0 - 2.0 * lam) * dt,
            check_interval,
            residual_tx,
            interrupt_rx,
        );
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg3 = gpu_total_force_dispatch_brain(
            gpu,
            dp,
            resident_pipelines,
            state,
            cg_bufs,
            lam * dt,
            check_interval,
            residual_tx,
            interrupt_rx,
        );
        total_cg += cg1 + cg2 + cg3;
    }

    let s_gauge_new = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_new = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action_brain(
        gpu,
        dp,
        resident_pipelines,
        state,
        cg_bufs,
        check_interval,
        residual_tx,
        interrupt_rx,
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

    let plaquette = gpu_plaquette(gpu, &dp.gauge, gs);

    GpuDynHmcResult {
        accepted,
        delta_h,
        plaquette,
        cg_iterations: total_cg,
    }
}
