// SPDX-License-Identifier: AGPL-3.0-only

//! Dynamical fermion GPU HMC — full QCD with staggered quarks.

use super::{
    flatten_momenta, gpu_dirac_dispatch, gpu_dot_re, gpu_fermion_force_dispatch,
    gpu_force_dispatch, gpu_kinetic_energy, gpu_link_update_dispatch, gpu_mom_update_dispatch,
    gpu_plaquette, gpu_wilson_action, make_link_mom_params, GpuF64, GpuHmcPipelines, GpuHmcState,
};

/// WGSL shader: staggered Dirac operator D·ψ.
pub const WGSL_DIRAC_STAGGERED: &str = include_str!("../shaders/dirac_staggered_f64.wgsl");

/// WGSL shader: staggered fermion force TA[U·M].
pub const WGSL_FERMION_FORCE: &str = include_str!("../shaders/staggered_fermion_force_f64.wgsl");

/// WGSL shader: complex dot product (Re part) for CG.
pub const WGSL_COMPLEX_DOT_RE: &str = super::super::cg::WGSL_COMPLEX_DOT_RE_F64;

/// WGSL shader: axpy y += α·x for CG.
pub const WGSL_AXPY: &str = super::super::cg::WGSL_AXPY_F64;

/// WGSL shader: xpay p = x + β·p for CG.
pub const WGSL_XPAY: &str = super::super::cg::WGSL_XPAY_F64;

/// WGSL shared PRNG core (PCG hash → uniform f64).
const WGSL_PRNG_CORE: &str = include_str!("../shaders/prng_pcg_f64.wgsl");
/// WGSL shader: GPU-resident PRNG for SU(3) algebra momenta.
pub static WGSL_RANDOM_MOMENTA: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let body = include_str!("../shaders/su3_random_momenta_f64.wgsl");
    format!("{WGSL_PRNG_CORE}\n{body}")
});

/// Pipelines for GPU dynamical fermion HMC.
pub struct GpuDynHmcPipelines {
    /// Quenched HMC pipelines (gauge force, link/mom updates, plaquette, KE)
    pub gauge: GpuHmcPipelines,
    /// Staggered Dirac operator
    pub dirac_pipeline: wgpu::ComputePipeline,
    /// Staggered fermion force
    pub fermion_force_pipeline: wgpu::ComputePipeline,
    /// CG helper: complex dot product
    pub dot_pipeline: wgpu::ComputePipeline,
    /// CG helper: axpy
    pub axpy_pipeline: wgpu::ComputePipeline,
    /// CG helper: xpay
    pub xpay_pipeline: wgpu::ComputePipeline,
}

impl GpuDynHmcPipelines {
    /// Compile all dynamical HMC shader pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            gauge: GpuHmcPipelines::new(gpu),
            dirac_pipeline: gpu.create_pipeline_f64(WGSL_DIRAC_STAGGERED, "dyn_dirac"),
            fermion_force_pipeline: gpu.create_pipeline_f64(WGSL_FERMION_FORCE, "dyn_ferm_force"),
            dot_pipeline: gpu.create_pipeline_f64(WGSL_COMPLEX_DOT_RE, "dyn_dot"),
            axpy_pipeline: gpu.create_pipeline_f64(WGSL_AXPY, "dyn_axpy"),
            xpay_pipeline: gpu.create_pipeline_f64(WGSL_XPAY, "dyn_xpay"),
        }
    }
}

/// GPU-resident state for dynamical fermion HMC.
pub struct GpuDynHmcState {
    /// Quenched HMC buffers (links, momenta, force, etc.)
    pub gauge: GpuHmcState,
    /// CG solution vector x: (D†D)x = b.
    pub x_buf: wgpu::Buffer,
    /// CG residual r = b − Ax.
    pub r_buf: wgpu::Buffer,
    /// CG search direction p.
    pub p_buf: wgpu::Buffer,
    /// CG Ap = (D†D)p.
    pub ap_buf: wgpu::Buffer,
    /// Scratch buffer for D·p intermediate.
    pub temp_buf: wgpu::Buffer,
    /// Scalar dot-product output (one f64).
    pub dot_buf: wgpu::Buffer,
    /// y = D·x buffer for fermion force
    pub y_buf: wgpu::Buffer,
    /// Separate fermion force output buffer
    pub ferm_force_buf: wgpu::Buffer,
    /// Pseudofermion field φ buffer
    pub phi_buf: wgpu::Buffer,
    /// Phase table for staggered fermions
    pub phases_buf: wgpu::Buffer,
    /// Fermion mass
    pub mass: f64,
    /// CG tolerance
    pub cg_tol: f64,
    /// CG max iterations
    pub cg_max_iter: usize,
}

impl GpuDynHmcState {
    /// Upload lattice and fermion configuration to GPU.
    #[must_use]
    pub fn from_lattice(
        gpu: &GpuF64,
        lattice: &super::super::wilson::Lattice,
        beta: f64,
        mass: f64,
        cg_tol: f64,
        cg_max_iter: usize,
    ) -> Self {
        let gauge = GpuHmcState::from_lattice(gpu, lattice, beta);
        let vol = lattice.volume();
        let n_flat = vol * 6;
        let n_pairs = vol * 3;

        let x_buf = gpu.create_f64_output_buffer(n_flat, "dyn_x");
        let r_buf = gpu.create_f64_output_buffer(n_flat, "dyn_r");
        let p_buf = gpu.create_f64_output_buffer(n_flat, "dyn_p");
        let ap_buf = gpu.create_f64_output_buffer(n_flat, "dyn_ap");
        let temp_buf = gpu.create_f64_output_buffer(n_flat, "dyn_temp");
        let dot_buf = gpu.create_f64_output_buffer(n_pairs, "dyn_dot");
        let y_buf = gpu.create_f64_output_buffer(n_flat, "dyn_y");
        let ferm_force_buf = gpu.create_f64_output_buffer(vol * 4 * 18, "dyn_ferm_force");
        let phi_buf = gpu.create_f64_output_buffer(n_flat, "dyn_phi");

        let mut phases = vec![0.0_f64; vol * 4];
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let sum: usize = x.iter().take(mu).sum();
                phases[idx * 4 + mu] = if sum.is_multiple_of(2) { 1.0 } else { -1.0 };
            }
        }
        let phases_buf = gpu.create_f64_buffer(&phases, "dyn_phases");

        Self {
            gauge,
            x_buf,
            r_buf,
            p_buf,
            ap_buf,
            temp_buf,
            dot_buf,
            y_buf,
            ferm_force_buf,
            phi_buf,
            phases_buf,
            mass,
            cg_tol,
            cg_max_iter,
        }
    }
}

/// Result of a dynamical fermion GPU HMC trajectory.
pub struct GpuDynHmcResult {
    /// Whether Metropolis accepted.
    pub accepted: bool,
    /// ΔH = H_new - H_old
    pub delta_h: f64,
    /// Average plaquette after trajectory.
    pub plaquette: f64,
    /// Total CG iterations across all solves in this trajectory.
    pub cg_iterations: usize,
}

/// Run one GPU dynamical fermion Omelyan HMC trajectory.
///
/// Gauge + fermion force computed on GPU. CG solver runs entirely on GPU.
/// CPU only: random momenta/pseudofermion generation, Metropolis decision.
pub fn gpu_dynamical_hmc_trajectory(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    n_md_steps: usize,
    dt: f64,
    seed: &mut u64,
) -> GpuDynHmcResult {
    let vol = state.gauge.volume;
    let n_links = state.gauge.n_links;

    let momenta: Vec<super::super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = flatten_momenta(&momenta);
    gpu.upload_f64(&state.gauge.mom_buf, &mom_flat);

    let phi_flat = gen_random_fermion(vol, seed);
    gpu.upload_f64(&state.phi_buf, &phi_flat);

    {
        let mut enc = gpu.begin_encoder("dyn_backup_links");
        enc.copy_buffer_to_buffer(
            &state.gauge.link_buf,
            0,
            &state.gauge.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let s_gauge_old = gpu_wilson_action(gpu, &pipelines.gauge, &state.gauge);
    let t_old = gpu_kinetic_energy(gpu, &pipelines.gauge, &state.gauge);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action(gpu, pipelines, state);
    let h_old = s_gauge_old + t_old + s_ferm_old;

    let mut total_cg = cg_iters_old;

    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for step in 0..n_md_steps {
        gpu_total_force_dispatch(gpu, pipelines, state, lam * dt);
        gpu_link_update_dispatch(gpu, &pipelines.gauge, &state.gauge, 0.5 * dt);
        gpu_total_force_dispatch(gpu, pipelines, state, (1.0 - 2.0 * lam) * dt);
        gpu_link_update_dispatch(gpu, &pipelines.gauge, &state.gauge, 0.5 * dt);
        let cg_step = gpu_total_force_dispatch(gpu, pipelines, state, lam * dt);
        total_cg += cg_step;

        if step == 0 {
            // Already counted in step 5 of previous iteration (but no previous)
        }
    }

    let s_gauge_new = gpu_wilson_action(gpu, &pipelines.gauge, &state.gauge);
    let t_new = gpu_kinetic_energy(gpu, &pipelines.gauge, &state.gauge);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action(gpu, pipelines, state);
    let h_new = s_gauge_new + t_new + s_ferm_new;
    total_cg += cg_iters_new;

    let delta_h = h_new - h_old;

    let r: f64 = super::super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("dyn_restore_links");
        enc.copy_buffer_to_buffer(
            &state.gauge.link_backup,
            0,
            &state.gauge.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette = gpu_plaquette(gpu, &pipelines.gauge, &state.gauge);

    GpuDynHmcResult {
        accepted,
        delta_h,
        plaquette,
        cg_iterations: total_cg,
    }
}

fn gen_random_fermion(vol: usize, seed: &mut u64) -> Vec<f64> {
    let n = vol * 6;
    let mut flat = vec![0.0_f64; n];
    for v in &mut flat {
        let u1 = super::super::constants::lcg_uniform_f64(seed).max(1e-30);
        let u2 = super::super::constants::lcg_uniform_f64(seed);
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }
    flat
}

/// Compute S_f = φ†(D†D)⁻¹φ on GPU via CG solve.
/// Returns (S_f, cg_iterations).
pub(super) fn gpu_fermion_action(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
) -> (f64, usize) {
    let iters = gpu_cg_solve_internal(gpu, pipelines, state, &state.phi_buf);

    let vol = state.gauge.volume;
    let n_pairs = vol * 3;
    let dot_val = gpu_dot_re(
        gpu,
        &pipelines.dot_pipeline,
        &state.dot_buf,
        &state.phi_buf,
        &state.x_buf,
        n_pairs,
    );

    (dot_val, iters)
}

/// Dispatch gauge force + fermion force + combined momentum update.
/// Returns CG iterations from the fermion force computation.
pub(super) fn gpu_total_force_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    dt: f64,
) -> usize {
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;

    gpu_force_dispatch(gpu, &pipelines.gauge, gs);
    gpu_mom_update_dispatch(gpu, &pipelines.gauge, gs, dt);

    let cg_iters = gpu_cg_solve_internal(gpu, pipelines, state, &state.phi_buf);

    gpu_dirac_dispatch(gpu, pipelines, state, &state.x_buf, &state.y_buf, 1.0);
    gpu_fermion_force_dispatch(gpu, pipelines, state);

    let ferm_mom_params = make_link_mom_params(n_links, dt);
    let ferm_mom_pbuf = gpu.create_uniform_buffer(&ferm_mom_params, "fmom_p");
    let ferm_mom_bg = gpu.create_bind_group(
        &pipelines.gauge.momentum_pipeline,
        &[&ferm_mom_pbuf, &state.ferm_force_buf, &gs.mom_buf],
    );
    gpu.dispatch(
        &pipelines.gauge.momentum_pipeline,
        &ferm_mom_bg,
        gs.wg_links,
    );

    cg_iters
}

/// GPU CG solver: (D†D)x = b, solution in state.x_buf.
fn gpu_cg_solve_internal(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    b_buf: &wgpu::Buffer,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let n_pairs = vol * 3;

    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(&state.x_buf, &zeros);
    {
        let mut enc = gpu.begin_encoder("cg_init_r");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        gpu.submit_encoder(enc);
    }
    {
        let mut enc = gpu.begin_encoder("cg_init_p");
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

        gpu_dirac_dispatch(gpu, pipelines, state, &state.p_buf, &state.temp_buf, 1.0);
        gpu_dirac_dispatch(gpu, pipelines, state, &state.temp_buf, &state.ap_buf, -1.0);

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

        gpu_axpy(
            gpu,
            &pipelines.axpy_pipeline,
            alpha,
            &state.p_buf,
            &state.x_buf,
            n_flat,
        );
        gpu_axpy(
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

        gpu_xpay(
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

fn gpu_axpy(
    gpu: &GpuF64,
    axpy_pl: &wgpu::ComputePipeline,
    alpha: f64,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
) {
    let wg = (n as u32).div_ceil(64);
    let mut params = Vec::with_capacity(16);
    params.extend_from_slice(&(n as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&alpha.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "axpy_p");
    let bg = gpu.create_bind_group(axpy_pl, &[&pbuf, x, y]);
    gpu.dispatch(axpy_pl, &bg, wg);
}

fn gpu_xpay(
    gpu: &GpuF64,
    xpay_pl: &wgpu::ComputePipeline,
    x: &wgpu::Buffer,
    beta: f64,
    p: &wgpu::Buffer,
    n: usize,
) {
    let wg = (n as u32).div_ceil(64);
    let mut params = Vec::with_capacity(16);
    params.extend_from_slice(&(n as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&beta.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "xpay_p");
    let bg = gpu.create_bind_group(xpay_pl, &[&pbuf, x, p]);
    gpu.dispatch(xpay_pl, &bg, wg);
}
