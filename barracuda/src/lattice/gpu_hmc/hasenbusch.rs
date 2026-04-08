// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU Hasenbusch mass preconditioning for dynamical fermion HMC.
//!
//! Two-level split: `det(D†D(m_l))` = `det(D†D(m_h))` × `det(D†D(m_l)/D†D(m_h))`
//!
//! Heavy sector is cheap (few CG iters), ratio sector has smaller condition
//! number than the full light operator. Multi-scale leapfrog integrator
//! puts the cheap heavy force on the outer timescale and the expensive
//! ratio force on the inner timescale.
//!
//! Port of CPU implementation in `lattice::pseudofermion::hasenbusch_hmc_trajectory`.

use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcResult, GpuDynHmcState, gen_random_fermion};
#[allow(deprecated)]
use super::{
    GpuF64, gpu_dot_re, gpu_force_dispatch, gpu_kinetic_energy, gpu_plaquette, gpu_wilson_action,
    make_link_mom_params,
};

/// Extra GPU buffers for Hasenbusch mass preconditioning.
///
/// These supplement the standard `GpuDynHmcState` which provides the
/// CG scratch space, gauge buffers, and the first pseudofermion field.
pub struct GpuHasenbuschBuffers {
    /// Second CG solution buffer (standard state has `x_buf` for the first).
    pub x2_buf: wgpu::Buffer,
    /// Buffer for `D†D(m_heavy)·phi_ratio` intermediate.
    pub ddh_buf: wgpu::Buffer,
    /// Extra scratch for Dirac intermediates in bilinear force.
    pub scratch_a: wgpu::Buffer,
    /// Extra scratch for Dirac intermediates in bilinear force.
    pub scratch_b: wgpu::Buffer,
}

impl GpuHasenbuschBuffers {
    /// Allocate Hasenbusch scratch buffers for a lattice with `vol` sites.
    #[must_use]
    pub fn new(gpu: &GpuF64, vol: usize) -> Self {
        let n_flat = vol * 6;
        Self {
            x2_buf: gpu.create_f64_output_buffer(n_flat, "has_x2"),
            ddh_buf: gpu.create_f64_output_buffer(n_flat, "has_ddh"),
            scratch_a: gpu.create_f64_output_buffer(n_flat, "has_scratch_a"),
            scratch_b: gpu.create_f64_output_buffer(n_flat, "has_scratch_b"),
        }
    }
}

/// Hasenbusch mass preconditioning configuration.
#[derive(Clone, Debug)]
pub struct GpuHasenbuschConfig {
    /// Heavy (intermediate) mass, typically 0.3-0.5.
    pub heavy_mass: f64,
    /// Light (physical) mass, typically 0.01-0.1.
    pub light_mass: f64,
    /// Outer MD steps (heavy sector, cheap).
    pub n_md_heavy: usize,
    /// Inner MD steps per outer step (ratio sector, expensive).
    pub n_md_light: usize,
}

impl Default for GpuHasenbuschConfig {
    fn default() -> Self {
        Self {
            heavy_mass: 0.4,
            light_mass: 0.1,
            n_md_heavy: 4,
            n_md_light: 16,
        }
    }
}

/// Apply D†D(mass) to a vector: output = D(-mass) · D(mass) · input.
/// Uses `state.temp_buf` as scratch for the intermediate D·input result.
fn gpu_apply_dirac_sq(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    scratch: &wgpu::Buffer,
    mass: f64,
) {
    gpu_dirac_dispatch_mass(gpu, pipelines, state, input, scratch, mass, 1.0);
    gpu_dirac_dispatch_mass(gpu, pipelines, state, scratch, output, mass, -1.0);
}

/// Dirac dispatch with explicit mass override (the standard dispatch uses state.mass).
fn gpu_dirac_dispatch_mass(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    mass: f64,
    hop_sign: f64,
) {
    let vol = state.gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let mut params = Vec::with_capacity(24);
    params.extend_from_slice(&(vol as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&mass.to_le_bytes());
    params.extend_from_slice(&hop_sign.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "has_dirac_p");
    let bg = gpu.create_bind_group(
        &pipelines.dirac_pipeline,
        &[
            &pbuf,
            &state.gauge.link_buf,
            input,
            output,
            &state.gauge.nbr_buf,
            &state.phases_buf,
        ],
    );
    gpu.dispatch(&pipelines.dirac_pipeline, &bg, wg);
}

/// Dispatch fermion force with explicit `x_field` and `y_field` buffers.
fn gpu_fermion_force_dispatch_xy(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    x_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
) {
    let vol = state.gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let ff_params = super::make_u32x4_params(vol as u32);
    let ff_pbuf = gpu.create_uniform_buffer(&ff_params, "has_ff_p");
    let ff_bg = gpu.create_bind_group(
        &pipelines.fermion_force_pipeline,
        &[
            &ff_pbuf,
            &state.gauge.link_buf,
            x_buf,
            y_buf,
            &state.gauge.nbr_buf,
            &state.phases_buf,
            &state.ferm_force_buf,
        ],
    );
    gpu.dispatch(&pipelines.fermion_force_pipeline, &ff_bg, wg);
}

/// Apply momentum kick: momenta += sign * dt * force.
fn gpu_mom_kick(gpu: &GpuF64, pipelines: &GpuDynHmcPipelines, state: &GpuDynHmcState, dt: f64) {
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;
    let params = make_link_mom_params(n_links, dt, gpu.full_df64_mode);
    let pbuf = gpu.create_uniform_buffer(&params, "has_mom_p");
    let bg = gpu.create_bind_group(
        &pipelines.gauge.momentum_pipeline,
        &[&pbuf, &state.ferm_force_buf, &gs.mom_buf],
    );
    gpu.dispatch(&pipelines.gauge.momentum_pipeline, &bg, gs.wg_links);
}

/// CG solve with explicit mass (overrides state.mass temporarily via params).
///
/// **Legacy** — per-iteration `gpu_dot_re` readback. No resident equivalent yet.
#[allow(deprecated)]
fn gpu_cg_solve_mass(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    b_buf: &wgpu::Buffer,
    x_out: &wgpu::Buffer,
    mass: f64,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let n_pairs = vol * 3;

    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(x_out, &zeros);
    {
        let mut enc = gpu.begin_encoder("has_cg_init");
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

    let mut r_norm_sq = b_norm_sq;
    let tol_sq = state.cg_tol * state.cg_tol * b_norm_sq;
    let mut iterations = 0;

    for iter in 0..state.cg_max_iter {
        iterations = iter + 1;

        gpu_dirac_dispatch_mass(
            gpu,
            pipelines,
            state,
            &state.p_buf,
            &state.temp_buf,
            mass,
            1.0,
        );
        gpu_dirac_dispatch_mass(
            gpu,
            pipelines,
            state,
            &state.temp_buf,
            &state.ap_buf,
            mass,
            -1.0,
        );

        let p_ap = gpu_dot_re(
            gpu,
            &pipelines.dot_pipeline,
            &state.dot_buf,
            &state.p_buf,
            &state.ap_buf,
            n_pairs,
        );
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = r_norm_sq / p_ap;

        super::dynamical::gpu_axpy(
            gpu,
            &pipelines.axpy_pipeline,
            alpha,
            &state.p_buf,
            x_out,
            n_flat,
        );
        super::dynamical::gpu_axpy(
            gpu,
            &pipelines.axpy_pipeline,
            -alpha,
            &state.ap_buf,
            &state.r_buf,
            n_flat,
        );

        let r_norm_sq_new = gpu_dot_re(
            gpu,
            &pipelines.dot_pipeline,
            &state.dot_buf,
            &state.r_buf,
            &state.r_buf,
            n_pairs,
        );
        if r_norm_sq_new < tol_sq {
            break;
        }

        let beta_cg = r_norm_sq_new / r_norm_sq;
        r_norm_sq = r_norm_sq_new;

        super::dynamical::gpu_xpay(
            gpu,
            &pipelines.xpay_pipeline,
            &state.r_buf,
            beta_cg,
            &state.p_buf,
            n_flat,
        );
    }

    // Copy solution to the target buffer if it's not x_buf (which gpu_axpy wrote to directly)
    if !std::ptr::eq(x_out, &raw const state.x_buf) {
        // x_out is a different buffer — gpu_axpy already wrote to it via the binding
    }

    iterations
}

/// Compute fermion action for the heavy sector: `S_h` = `phi_h`† (D†D(m_h))^{-1} `phi_h`.
#[allow(deprecated)]
fn gpu_heavy_action(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    phi_heavy: &wgpu::Buffer,
    mass: f64,
) -> (f64, usize) {
    let iters = gpu_cg_solve_mass(gpu, pipelines, state, phi_heavy, &state.x_buf, mass);
    let vol = state.gauge.volume;
    let n_pairs = vol * 3;
    let dot = gpu_dot_re(
        gpu,
        &pipelines.dot_pipeline,
        &state.dot_buf,
        phi_heavy,
        &state.x_buf,
        n_pairs,
    );
    (dot, iters)
}

/// Compute ratio action: `S_r` = `phi_r`† `D†D(m_h)` (D†D(m_l))^{-1} `phi_r`.
///
/// The ratio determinant `det(D†D(m_l)/D†D(m_h))` is represented by the action
/// `S_r` = `phi_r`† [`D†D(m_h)` · (D†D(m_l))^{-1}] `phi_r`.
/// Steps: 1) CG solve (`D†D(m_l)`) x = `phi_r`, 2) compute `D†D(m_h)` x, 3) dot `phi_r` with result.
#[allow(deprecated)]
fn gpu_ratio_action(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    hbufs: &GpuHasenbuschBuffers,
    phi_ratio: &wgpu::Buffer,
    config: &GpuHasenbuschConfig,
) -> (f64, usize) {
    // x = (D†D(m_l))^{-1} phi_r
    let iters = gpu_cg_solve_mass(
        gpu,
        pipelines,
        state,
        phi_ratio,
        &state.x_buf,
        config.light_mass,
    );

    // D†D(m_h) x → ddh_buf
    gpu_apply_dirac_sq(
        gpu,
        pipelines,
        state,
        &state.x_buf,
        &hbufs.ddh_buf,
        &hbufs.scratch_a,
        config.heavy_mass,
    );

    // S_r = phi_r† · ddh_buf = phi_r† D†D(m_h) (D†D(m_l))^{-1} phi_r
    let vol = state.gauge.volume;
    let n_pairs = vol * 3;
    let action = gpu_dot_re(
        gpu,
        &pipelines.dot_pipeline,
        &state.dot_buf,
        phi_ratio,
        &hbufs.ddh_buf,
        n_pairs,
    );

    (action, iters)
}

/// Dispatch heavy sector force: gauge force + heavy pseudofermion force.
fn gpu_heavy_force_kick(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    phi_heavy: &wgpu::Buffer,
    config: &GpuHasenbuschConfig,
    dt: f64,
) -> usize {
    let gs = &state.gauge;
    gpu_force_dispatch(gpu, &pipelines.gauge, gs);
    super::gpu_mom_update_dispatch(gpu, &pipelines.gauge, gs, dt);

    let cg_iters = gpu_cg_solve_mass(
        gpu,
        pipelines,
        state,
        phi_heavy,
        &state.x_buf,
        config.heavy_mass,
    );

    gpu_dirac_dispatch_mass(
        gpu,
        pipelines,
        state,
        &state.x_buf,
        &state.y_buf,
        config.heavy_mass,
        1.0,
    );
    gpu_fermion_force_dispatch_xy(gpu, pipelines, state, &state.x_buf, &state.y_buf);
    gpu_mom_kick(gpu, pipelines, state, dt);

    cg_iters
}

/// Dispatch ratio sector force using bilinear decomposition.
///
/// The ratio force = `force_bilinear(y`, x, `m_l`) - `force_bilinear(x`, `phi_r`, `m_h`),
/// where x = (D†D(m_l))^{-1} `phi_r` and y = (D†D(m_l))^{-1} `D†D(m_h)` `phi_r`.
///
/// `force_bilinear(a`, b, m) = `force_shader(a`, Db) + `force_shader(b`, Da),
/// so we reuse the standard fermion force shader twice per bilinear term.
fn gpu_ratio_force_kick(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    hbufs: &GpuHasenbuschBuffers,
    phi_ratio: &wgpu::Buffer,
    config: &GpuHasenbuschConfig,
    dt: f64,
) -> usize {
    // x = (D†D(m_l))^{-1} phi_r
    let cg1 = gpu_cg_solve_mass(
        gpu,
        pipelines,
        state,
        phi_ratio,
        &state.x_buf,
        config.light_mass,
    );

    // D†D(m_h) phi_r → ddh_buf
    gpu_apply_dirac_sq(
        gpu,
        pipelines,
        state,
        phi_ratio,
        &hbufs.ddh_buf,
        &hbufs.scratch_a,
        config.heavy_mass,
    );

    // y = (D†D(m_l))^{-1} ddh_buf → x2_buf
    let cg2 = gpu_cg_solve_mass(
        gpu,
        pipelines,
        state,
        &hbufs.ddh_buf,
        &hbufs.x2_buf,
        config.light_mass,
    );

    // --- Term 1: force_bilinear(y, x, m_l) → kick momenta by +dt ---
    // Da = D(m_l)·y → scratch_a
    gpu_dirac_dispatch_mass(
        gpu,
        pipelines,
        state,
        &hbufs.x2_buf,
        &hbufs.scratch_a,
        config.light_mass,
        1.0,
    );
    // Db = D(m_l)·x → scratch_b
    gpu_dirac_dispatch_mass(
        gpu,
        pipelines,
        state,
        &state.x_buf,
        &hbufs.scratch_b,
        config.light_mass,
        1.0,
    );
    // force(y, Db) → ferm_force_buf, kick +dt
    gpu_fermion_force_dispatch_xy(gpu, pipelines, state, &hbufs.x2_buf, &hbufs.scratch_b);
    gpu_mom_kick(gpu, pipelines, state, dt);
    // force(x, Da) → ferm_force_buf, kick +dt
    gpu_fermion_force_dispatch_xy(gpu, pipelines, state, &state.x_buf, &hbufs.scratch_a);
    gpu_mom_kick(gpu, pipelines, state, dt);

    // --- Term 2: force_bilinear(x, phi_r, m_h) → kick momenta by -dt ---
    // Da = D(m_h)·x → scratch_a
    gpu_dirac_dispatch_mass(
        gpu,
        pipelines,
        state,
        &state.x_buf,
        &hbufs.scratch_a,
        config.heavy_mass,
        1.0,
    );
    // Db = D(m_h)·phi_r → scratch_b
    gpu_dirac_dispatch_mass(
        gpu,
        pipelines,
        state,
        phi_ratio,
        &hbufs.scratch_b,
        config.heavy_mass,
        1.0,
    );
    // force(x, Db) → ferm_force_buf, kick -dt
    gpu_fermion_force_dispatch_xy(gpu, pipelines, state, &state.x_buf, &hbufs.scratch_b);
    gpu_mom_kick(gpu, pipelines, state, -dt);
    // force(phi_r, Da) → ferm_force_buf, kick -dt
    gpu_fermion_force_dispatch_xy(gpu, pipelines, state, phi_ratio, &hbufs.scratch_a);
    gpu_mom_kick(gpu, pipelines, state, -dt);

    cg1 + cg2
}

/// Link update (reuses `encode_link_update` from brain module).
fn link_update(gpu: &GpuF64, pipelines: &GpuDynHmcPipelines, state: &GpuDynHmcState, dt: f64) {
    let gs = &state.gauge;
    let params = make_link_mom_params(gs.n_links, dt, gpu.full_df64_mode);
    let pbuf = gpu.create_uniform_buffer(&params, "has_link_p");
    let bg = gpu.create_bind_group(
        &pipelines.gauge.link_pipeline,
        &[&pbuf, &gs.mom_buf, &gs.link_buf],
    );
    gpu.dispatch(&pipelines.gauge.link_pipeline, &bg, gs.wg_links);
}

/// Full Hasenbusch HMC trajectory with multi-scale leapfrog.
///
/// Outer loop: heavy (gauge + heavy fermion) force kicks.
/// Inner loop: ratio force kicks + link updates.
pub fn gpu_hasenbusch_hmc_trajectory(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    hbufs: &GpuHasenbuschBuffers,
    config: &GpuHasenbuschConfig,
    dt: f64,
    seed: &mut u64,
) -> GpuDynHmcResult {
    let vol = state.gauge.volume;
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;

    assert!(
        state.phi_bufs.len() >= 2,
        "Hasenbusch requires at least 2 pseudofermion fields"
    );
    let phi_heavy = &state.phi_bufs[0];
    let phi_ratio = &state.phi_bufs[1];

    let momenta: Vec<super::super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = super::flatten_momenta(&momenta);
    gpu.upload_f64(&gs.mom_buf, &mom_flat);

    let phi_h_flat = gen_random_fermion(vol, seed);
    gpu.upload_f64(phi_heavy, &phi_h_flat);
    let phi_r_flat = gen_random_fermion(vol, seed);
    gpu.upload_f64(phi_ratio, &phi_r_flat);

    {
        let mut enc = gpu.begin_encoder("has_backup");
        enc.copy_buffer_to_buffer(
            &gs.link_buf,
            0,
            &gs.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    // TODO(B2): replace with GPU-resident Hamiltonian assembly
    #[allow(deprecated)]
    let s_gauge_old = gpu_wilson_action(gpu, &pipelines.gauge, gs);
    let t_old = gpu_kinetic_energy(gpu, &pipelines.gauge, gs);
    let (s_heavy_old, cg_h0) =
        gpu_heavy_action(gpu, pipelines, state, phi_heavy, config.heavy_mass);
    let (s_ratio_old, cg_r0) = gpu_ratio_action(gpu, pipelines, state, hbufs, phi_ratio, config);
    let h_old = s_gauge_old + t_old + s_heavy_old + s_ratio_old;
    let mut total_cg = cg_h0 + cg_r0;

    let dt_heavy = dt / config.n_md_heavy as f64;
    let dt_light = dt_heavy / config.n_md_light as f64;

    for _ in 0..config.n_md_heavy {
        total_cg += gpu_heavy_force_kick(gpu, pipelines, state, phi_heavy, config, 0.5 * dt_heavy);

        for _ in 0..config.n_md_light {
            total_cg += gpu_ratio_force_kick(
                gpu,
                pipelines,
                state,
                hbufs,
                phi_ratio,
                config,
                0.5 * dt_light,
            );
            link_update(gpu, pipelines, state, dt_light);
            total_cg += gpu_ratio_force_kick(
                gpu,
                pipelines,
                state,
                hbufs,
                phi_ratio,
                config,
                0.5 * dt_light,
            );
        }

        total_cg += gpu_heavy_force_kick(gpu, pipelines, state, phi_heavy, config, 0.5 * dt_heavy);
    }

    #[allow(deprecated)]
    let s_gauge_new = gpu_wilson_action(gpu, &pipelines.gauge, gs);
    let t_new = gpu_kinetic_energy(gpu, &pipelines.gauge, gs);
    let (s_heavy_new, cg_h1) =
        gpu_heavy_action(gpu, pipelines, state, phi_heavy, config.heavy_mass);
    let (s_ratio_new, cg_r1) = gpu_ratio_action(gpu, pipelines, state, hbufs, phi_ratio, config);
    let h_new = s_gauge_new + t_new + s_heavy_new + s_ratio_new;
    total_cg += cg_h1 + cg_r1;

    let delta_h = h_new - h_old;
    let r: f64 = super::super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("has_restore");
        enc.copy_buffer_to_buffer(
            &gs.link_backup,
            0,
            &gs.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    #[allow(deprecated)]
    let plaquette = gpu_plaquette(gpu, &pipelines.gauge, gs);

    GpuDynHmcResult {
        accepted,
        delta_h,
        plaquette,
        cg_iterations: total_cg,
    }
}
