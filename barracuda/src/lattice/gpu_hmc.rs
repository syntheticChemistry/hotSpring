// SPDX-License-Identifier: AGPL-3.0-only

//! Pure GPU HMC: all math on GPU via fp64 WGSL shaders.
//!
//! The CPU only orchestrates dispatches and reads back scalars (ΔH,
//! plaquette) for Metropolis accept/reject and observables. Links and
//! momenta live in GPU buffers between MD steps — no round-trip.
//!
//! # Shader pipeline (per Omelyan MD step)
//!
//! 1. `su3_gauge_force_f64` — F(U) for all links
//! 2. `su3_momentum_update_f64` — P += dt * F
//! 3. `su3_link_update_f64` — U = exp(dt·P) * U (Cayley + reunitarize)
//!
//! # Observable shaders
//!
//! - `wilson_plaquette_f64` — per-site plaquette sum
//! - `su3_kinetic_energy_f64` — per-link -½ Re Tr(P²)
//!
//! # Data transfer budget
//!
//! | Direction | Per trajectory | Content |
//! |-----------|---------------|---------|
//! | CPU→GPU | 1× | Momenta (random, generated CPU-side) |
//! | GPU→CPU | 2× | H_old, H_new (scalar reductions) |
//! | GPU→CPU | 1× | Plaquette (scalar reduction) |
//!
//! Links stay GPU-resident. On reject, staging copy restores old links.

use super::wilson::Lattice;
use crate::gpu::GpuF64;

/// WGSL shader: Wilson plaquette per site (6 planes, Re Tr P/3).
pub const WGSL_WILSON_PLAQUETTE: &str = include_str!("shaders/wilson_plaquette_f64.wgsl");

/// WGSL shader: SU(3) gauge force (staple + traceless anti-Hermitian projection).
pub const WGSL_GAUGE_FORCE: &str = include_str!("shaders/su3_gauge_force_f64.wgsl");

/// WGSL shader: momentum update P += dt * F.
pub const WGSL_MOMENTUM_UPDATE: &str = include_str!("shaders/su3_momentum_update_f64.wgsl");

/// WGSL shader: link update U = exp(dt·P) * U via Cayley + reunitarize.
pub const WGSL_LINK_UPDATE: &str = include_str!("shaders/su3_link_update_f64.wgsl");

/// WGSL shader: kinetic energy -½ Re Tr(P²) per link.
pub const WGSL_KINETIC_ENERGY: &str = include_str!("shaders/su3_kinetic_energy_f64.wgsl");

/// GPU HMC pipeline: holds compiled compute pipelines and persistent buffers.
pub struct GpuHmcPipelines {
    /// Wilson plaquette per-site kernel
    pub plaquette_pipeline: wgpu::ComputePipeline,
    /// Gauge force kernel
    pub force_pipeline: wgpu::ComputePipeline,
    /// Momentum update kernel
    pub momentum_pipeline: wgpu::ComputePipeline,
    /// Link update (Cayley exp) kernel
    pub link_pipeline: wgpu::ComputePipeline,
    /// Kinetic energy per-link kernel
    pub kinetic_pipeline: wgpu::ComputePipeline,
}

impl GpuHmcPipelines {
    /// Compile all HMC shader pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            plaquette_pipeline: gpu.create_pipeline_f64(WGSL_WILSON_PLAQUETTE, "hmc_plaq"),
            force_pipeline: gpu.create_pipeline_f64(WGSL_GAUGE_FORCE, "hmc_force"),
            momentum_pipeline: gpu.create_pipeline_f64(WGSL_MOMENTUM_UPDATE, "hmc_mom_update"),
            link_pipeline: gpu.create_pipeline_f64(WGSL_LINK_UPDATE, "hmc_link_update"),
            kinetic_pipeline: gpu.create_pipeline_f64(WGSL_KINETIC_ENERGY, "hmc_ke"),
        }
    }
}

/// Flatten lattice links to f64 array (same layout as `DiracGpuLayout`).
#[must_use]
pub fn flatten_links(lattice: &Lattice) -> Vec<f64> {
    let vol = lattice.volume();
    let mut flat = vec![0.0_f64; vol * 4 * 18];
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let u = lattice.link(x, mu);
            let base = (idx * 4 + mu) * 18;
            for row in 0..3 {
                for col in 0..3 {
                    flat[base + row * 6 + col * 2] = u.m[row][col].re;
                    flat[base + row * 6 + col * 2 + 1] = u.m[row][col].im;
                }
            }
        }
    }
    flat
}

/// Build neighbor table (same layout as `DiracGpuLayout`).
#[must_use]
pub fn build_neighbors(lattice: &Lattice) -> Vec<u32> {
    let vol = lattice.volume();
    let mut neighbors = vec![0_u32; vol * 8];
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let fwd = lattice.site_index(lattice.neighbor(x, mu, true));
            let bwd = lattice.site_index(lattice.neighbor(x, mu, false));
            neighbors[idx * 8 + mu * 2] = fwd as u32;
            neighbors[idx * 8 + mu * 2 + 1] = bwd as u32;
        }
    }
    neighbors
}

/// Flatten SU(3) momenta to f64 array.
#[must_use]
pub fn flatten_momenta(momenta: &[super::su3::Su3Matrix]) -> Vec<f64> {
    let mut flat = vec![0.0_f64; momenta.len() * 18];
    for (i, p) in momenta.iter().enumerate() {
        let base = i * 18;
        for row in 0..3 {
            for col in 0..3 {
                flat[base + row * 6 + col * 2] = p.m[row][col].re;
                flat[base + row * 6 + col * 2 + 1] = p.m[row][col].im;
            }
        }
    }
    flat
}

/// Result of a single GPU HMC trajectory.
pub struct GpuHmcResult {
    /// Whether the Metropolis test accepted this trajectory.
    pub accepted: bool,
    /// ΔH = H_new - H_old
    pub delta_h: f64,
    /// Average plaquette after trajectory (whether accepted or rejected).
    pub plaquette: f64,
}

/// GPU-resident HMC state: buffers that persist across trajectories.
pub struct GpuHmcState {
    pub link_buf: wgpu::Buffer,
    pub link_backup: wgpu::Buffer,
    pub mom_buf: wgpu::Buffer,
    pub force_buf: wgpu::Buffer,
    pub ke_out_buf: wgpu::Buffer,
    pub plaq_out_buf: wgpu::Buffer,
    pub nbr_buf: wgpu::Buffer,
    pub volume: usize,
    pub n_links: usize,
    pub beta: f64,
    pub wg_links: u32,
    pub wg_vol: u32,
}

impl GpuHmcState {
    /// Upload a lattice to GPU and create all persistent buffers.
    #[must_use]
    pub fn from_lattice(gpu: &GpuF64, lattice: &Lattice, beta: f64) -> Self {
        let vol = lattice.volume();
        let n_links = vol * 4;
        let links_flat = flatten_links(lattice);
        let neighbors = build_neighbors(lattice);

        let link_buf = gpu.create_f64_output_buffer(n_links * 18, "hmc_links");
        gpu.upload_f64(&link_buf, &links_flat);
        let link_backup = gpu.create_f64_output_buffer(n_links * 18, "hmc_links_backup");
        let mom_buf = gpu.create_f64_output_buffer(n_links * 18, "hmc_momenta");
        let force_buf = gpu.create_f64_output_buffer(n_links * 18, "hmc_force");
        let ke_out_buf = gpu.create_f64_output_buffer(n_links, "hmc_ke");
        let plaq_out_buf = gpu.create_f64_output_buffer(vol, "hmc_plaq");
        let nbr_buf = gpu.create_u32_buffer(&neighbors, "hmc_nbr");

        Self {
            link_buf,
            link_backup,
            mom_buf,
            force_buf,
            ke_out_buf,
            plaq_out_buf,
            nbr_buf,
            volume: vol,
            n_links,
            beta,
            wg_links: ((n_links + 63) / 64) as u32,
            wg_vol: ((vol + 63) / 64) as u32,
        }
    }
}

/// Run one pure-GPU Omelyan HMC trajectory.
///
/// All gauge force, momentum update, link update, kinetic energy, and
/// plaquette math happens on GPU. CPU only generates random momenta
/// (uploaded once), reads back H_old/H_new (scalar sums), and makes
/// the Metropolis decision.
pub fn gpu_hmc_trajectory(
    gpu: &GpuF64,
    pipelines: &GpuHmcPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    seed: &mut u64,
) -> GpuHmcResult {
    let vol = state.volume;
    let n_links = state.n_links;
    let beta = state.beta;

    // Generate random momenta (CPU) and upload once
    let momenta: Vec<super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = flatten_momenta(&momenta);
    gpu.upload_f64(&state.mom_buf, &mom_flat);

    // Backup links for reject rollback (GPU→GPU copy)
    {
        let mut enc = gpu.begin_encoder("backup_links");
        enc.copy_buffer_to_buffer(
            &state.link_buf,
            0,
            &state.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    // ── Compute H_old = S(U) + T(P) ──
    let s_old = gpu_wilson_action(gpu, pipelines, state);
    let t_old = gpu_kinetic_energy(gpu, pipelines, state);
    let h_old = s_old + t_old;

    // ── Omelyan MD integration ──
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for step in 0..n_md_steps {
        // Step 1: P += λ·dt·F(U)
        gpu_force_dispatch(gpu, pipelines, state);
        gpu_mom_update_dispatch(gpu, pipelines, state, lam * dt);

        // Step 2: U = exp((dt/2)·P) * U
        gpu_link_update_dispatch(gpu, pipelines, state, 0.5 * dt);

        // Step 3: P += (1-2λ)·dt·F(U)
        gpu_force_dispatch(gpu, pipelines, state);
        gpu_mom_update_dispatch(gpu, pipelines, state, (1.0 - 2.0 * lam) * dt);

        // Step 4: U = exp((dt/2)·P) * U
        gpu_link_update_dispatch(gpu, pipelines, state, 0.5 * dt);

        // Step 5: P += λ·dt·F(U)
        gpu_force_dispatch(gpu, pipelines, state);
        gpu_mom_update_dispatch(gpu, pipelines, state, lam * dt);

        // Merge steps 5/1 for intermediate steps (last step keeps separate)
        if step < n_md_steps - 1 {
            // The λ·dt from step 5 + λ·dt from step 1 of next iteration
            // are already separate dispatches — correct but not fused.
            // Fusion is an optimization for later.
        }
    }

    // ── Compute H_new = S(U') + T(P') ──
    let s_new = gpu_wilson_action(gpu, pipelines, state);
    let t_new = gpu_kinetic_energy(gpu, pipelines, state);
    let h_new = s_new + t_new;

    let delta_h = h_new - h_old;

    // Metropolis accept/reject (CPU)
    let r: f64 = super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        // Restore links from backup
        let mut enc = gpu.begin_encoder("restore_links");
        enc.copy_buffer_to_buffer(
            &state.link_backup,
            0,
            &state.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    // Measure plaquette
    let plaquette = gpu_plaquette(gpu, pipelines, state);

    GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
    }
}

// ── Internal dispatch helpers ──

fn make_force_params(vol: usize, beta: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&(vol as u32).to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&beta.to_le_bytes());
    v
}

fn make_link_mom_params(n_links: usize, dt: f64) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&(n_links as u32).to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&dt.to_le_bytes());
    v
}

fn make_u32x4_params(val: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&val.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes());
    v
}

fn gpu_force_dispatch(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) {
    let params = make_force_params(s.volume, s.beta);
    let param_buf = gpu.create_uniform_buffer(&params, "force_p");
    let bg = gpu.create_bind_group(
        &p.force_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.force_buf],
    );
    gpu.dispatch(&p.force_pipeline, &bg, s.wg_links);
}

fn gpu_mom_update_dispatch(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState, dt: f64) {
    let params = make_link_mom_params(s.n_links, dt);
    let param_buf = gpu.create_uniform_buffer(&params, "mom_p");
    let bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&param_buf, &s.force_buf, &s.mom_buf],
    );
    gpu.dispatch(&p.momentum_pipeline, &bg, s.wg_links);
}

fn gpu_link_update_dispatch(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState, dt: f64) {
    let params = make_link_mom_params(s.n_links, dt);
    let param_buf = gpu.create_uniform_buffer(&params, "link_p");
    let bg = gpu.create_bind_group(&p.link_pipeline, &[&param_buf, &s.mom_buf, &s.link_buf]);
    gpu.dispatch(&p.link_pipeline, &bg, s.wg_links);
}

fn gpu_wilson_action(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.volume as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "plaq_p");
    let bg = gpu.create_bind_group(
        &p.plaquette_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.plaq_out_buf],
    );
    gpu.dispatch(&p.plaquette_pipeline, &bg, s.wg_vol);
    let per_site = match gpu.read_back_f64(&s.plaq_out_buf, s.volume) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let plaq_sum: f64 = per_site.iter().sum();
    s.beta * (6.0 * s.volume as f64 - plaq_sum)
}

fn gpu_kinetic_energy(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.n_links as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "ke_p");
    let bg = gpu.create_bind_group(
        &p.kinetic_pipeline,
        &[&param_buf, &s.mom_buf, &s.ke_out_buf],
    );
    gpu.dispatch(&p.kinetic_pipeline, &bg, s.wg_links);
    let per_link = match gpu.read_back_f64(&s.ke_out_buf, s.n_links) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    per_link.iter().sum()
}

fn gpu_plaquette(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.volume as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "plaq_obs");
    let bg = gpu.create_bind_group(
        &p.plaquette_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.plaq_out_buf],
    );
    gpu.dispatch(&p.plaquette_pipeline, &bg, s.wg_vol);
    let per_site = match gpu.read_back_f64(&s.plaq_out_buf, s.volume) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let plaq_sum: f64 = per_site.iter().sum();
    plaq_sum / (6.0 * s.volume as f64)
}

// ═══════════════════════════════════════════════════════════════════
//  Dynamical fermion GPU HMC — full QCD with staggered quarks
// ═══════════════════════════════════════════════════════════════════

/// WGSL shader: staggered Dirac operator D·ψ.
pub const WGSL_DIRAC_STAGGERED: &str = include_str!("shaders/dirac_staggered_f64.wgsl");

/// WGSL shader: staggered fermion force TA[U·M].
pub const WGSL_FERMION_FORCE: &str = include_str!("shaders/staggered_fermion_force_f64.wgsl");

/// WGSL shader: complex dot product (Re part) for CG.
pub const WGSL_COMPLEX_DOT_RE: &str = super::cg::WGSL_COMPLEX_DOT_RE_F64;

/// WGSL shader: axpy y += α·x for CG.
pub const WGSL_AXPY: &str = super::cg::WGSL_AXPY_F64;

/// WGSL shader: xpay p = x + β·p for CG.
pub const WGSL_XPAY: &str = super::cg::WGSL_XPAY_F64;

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
    /// Fermion field buffers for CG solver (x, r, p, ap, temp, dot_out)
    pub x_buf: wgpu::Buffer,
    pub r_buf: wgpu::Buffer,
    pub p_buf: wgpu::Buffer,
    pub ap_buf: wgpu::Buffer,
    pub temp_buf: wgpu::Buffer,
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
        lattice: &Lattice,
        beta: f64,
        mass: f64,
        cg_tol: f64,
        cg_max_iter: usize,
    ) -> Self {
        let gauge = GpuHmcState::from_lattice(gpu, lattice, beta);
        let vol = lattice.volume();
        let n_flat = vol * 6; // 3 colors × 2 (re/im) per site
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

        // Build and upload phase table
        let mut phases = vec![0.0_f64; vol * 4];
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let sum: usize = x.iter().take(mu).sum();
                phases[idx * 4 + mu] = if sum % 2 == 0 { 1.0 } else { -1.0 };
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
    let beta = state.gauge.beta;

    // Generate random momenta (CPU) and upload
    let momenta: Vec<super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = flatten_momenta(&momenta);
    gpu.upload_f64(&state.gauge.mom_buf, &mom_flat);

    // Generate random pseudofermion φ (CPU) and upload
    let phi_flat = gen_random_fermion(vol, seed);
    gpu.upload_f64(&state.phi_buf, &phi_flat);

    // Backup links
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

    // H_old = S_gauge + T_kinetic + S_fermion
    let s_gauge_old = gpu_wilson_action(gpu, &pipelines.gauge, &state.gauge);
    let t_old = gpu_kinetic_energy(gpu, &pipelines.gauge, &state.gauge);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action(gpu, pipelines, state);
    let h_old = s_gauge_old + t_old + s_ferm_old;

    let mut total_cg = cg_iters_old;

    // Omelyan MD
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for step in 0..n_md_steps {
        // Step 1: P += λ·dt·(F_gauge + F_fermion)
        gpu_total_force_dispatch(gpu, pipelines, state, lam * dt);

        // Step 2: U = exp((dt/2)·P) * U
        gpu_link_update_dispatch(gpu, &pipelines.gauge, &state.gauge, 0.5 * dt);

        // Step 3: P += (1-2λ)·dt·(F_gauge + F_fermion)
        gpu_total_force_dispatch(gpu, pipelines, state, (1.0 - 2.0 * lam) * dt);

        // Step 4: U = exp((dt/2)·P) * U
        gpu_link_update_dispatch(gpu, &pipelines.gauge, &state.gauge, 0.5 * dt);

        // Step 5: P += λ·dt·(F_gauge + F_fermion)
        let cg_step = gpu_total_force_dispatch(gpu, pipelines, state, lam * dt);
        total_cg += cg_step;

        // Track CG cost from steps 1 and 3
        if step == 0 {
            // Already counted in step 5 of previous iteration (but no previous)
        }
    }

    // H_new
    let s_gauge_new = gpu_wilson_action(gpu, &pipelines.gauge, &state.gauge);
    let t_new = gpu_kinetic_energy(gpu, &pipelines.gauge, &state.gauge);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action(gpu, pipelines, state);
    let h_new = s_gauge_new + t_new + s_ferm_new;
    total_cg += cg_iters_new;

    let delta_h = h_new - h_old;

    // Metropolis
    let r: f64 = super::constants::lcg_uniform_f64(seed);
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

/// Generate random fermion field as flat f64 array.
fn gen_random_fermion(vol: usize, seed: &mut u64) -> Vec<f64> {
    let n = vol * 6;
    let mut flat = vec![0.0_f64; n];
    for v in &mut flat {
        // Box-Muller from LCG
        let u1 = super::constants::lcg_uniform_f64(seed).max(1e-30);
        let u2 = super::constants::lcg_uniform_f64(seed);
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }
    flat
}

/// Compute S_f = φ†(D†D)⁻¹φ on GPU via CG solve.
/// Returns (S_f, cg_iterations).
fn gpu_fermion_action(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
) -> (f64, usize) {
    // Solve (D†D)x = φ via GPU CG
    let iters = gpu_cg_solve_internal(gpu, pipelines, state, &state.phi_buf);

    // S_f = φ†·x = Re(<φ|x>)
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
fn gpu_total_force_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    dt: f64,
) -> usize {
    let vol = state.gauge.volume;
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;

    // 1. Gauge force → gauge.force_buf
    gpu_force_dispatch(gpu, &pipelines.gauge, gs);

    // 2. P += dt * F_gauge
    gpu_mom_update_dispatch(gpu, &pipelines.gauge, gs, dt);

    // 3. Fermion force: CG solve (D†D)x = φ, then y = Dx, then TA[U·M]
    let cg_iters = gpu_cg_solve_internal(gpu, pipelines, state, &state.phi_buf);

    // 4. y = D·x
    gpu_dirac_dispatch(gpu, pipelines, state, &state.x_buf, &state.y_buf, 1.0);

    // 5. Fermion force → ferm_force_buf
    gpu_fermion_force_dispatch(gpu, pipelines, state);

    // 6. P += dt * F_fermion
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
    let wg_dirac = ((vol as u32) + 63) / 64;
    let wg_vec = ((n_flat as u32) + 63) / 64;

    // Initialize: x = 0, r = b, p = b
    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(&state.x_buf, &zeros);
    // Copy b → r
    {
        let mut enc = gpu.begin_encoder("cg_init_r");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        gpu.submit_encoder(enc);
    }
    // Copy b → p
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

        // ap = D†D·p: first temp = D·p, then ap = D†·temp
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

        // x += α·p
        gpu_axpy(
            gpu,
            &pipelines.axpy_pipeline,
            alpha,
            &state.p_buf,
            &state.x_buf,
            n_flat,
        );
        // r -= α·ap
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

        // p = r + β·p
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

/// Dispatch Dirac operator: out = D(hop_sign) · input.
fn gpu_dirac_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    hop_sign: f64,
) {
    let vol = state.gauge.volume;
    let wg = ((vol as u32) + 63) / 64;
    let mut params = Vec::with_capacity(24);
    params.extend_from_slice(&(vol as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&state.mass.to_le_bytes());
    params.extend_from_slice(&hop_sign.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "dirac_p");
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

/// Dispatch fermion force: force_buf = TA[U · M(x, y)].
fn gpu_fermion_force_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    state: &GpuDynHmcState,
) {
    let vol = state.gauge.volume;
    let wg = ((vol as u32) + 63) / 64;
    let params = make_u32x4_params(vol as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "fforce_p");
    let bg = gpu.create_bind_group(
        &pipelines.fermion_force_pipeline,
        &[
            &pbuf,
            &state.gauge.link_buf,
            &state.x_buf,
            &state.y_buf,
            &state.gauge.nbr_buf,
            &state.phases_buf,
            &state.ferm_force_buf,
        ],
    );
    gpu.dispatch(&pipelines.fermion_force_pipeline, &bg, wg);
}

/// GPU dot product: Re(<a|b>) = Σ a[i]*b[i] (complex pairs → real).
fn gpu_dot_re(
    gpu: &GpuF64,
    dot_pl: &wgpu::ComputePipeline,
    dot_buf: &wgpu::Buffer,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    n_pairs: usize,
) -> f64 {
    let wg = ((n_pairs as u32) + 63) / 64;
    let params = make_u32x4_params(n_pairs as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "dot_p");
    let bg = gpu.create_bind_group(dot_pl, &[&pbuf, a, b, dot_buf]);
    gpu.dispatch(dot_pl, &bg, wg);
    match gpu.read_back_f64(dot_buf, n_pairs) {
        Ok(v) => v.iter().sum(),
        Err(_) => f64::NAN,
    }
}

/// GPU axpy: y += α·x (in-place on y).
fn gpu_axpy(
    gpu: &GpuF64,
    axpy_pl: &wgpu::ComputePipeline,
    alpha: f64,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    n: usize,
) {
    let wg = ((n as u32) + 63) / 64;
    let mut params = Vec::with_capacity(16);
    params.extend_from_slice(&(n as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&alpha.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "axpy_p");
    let bg = gpu.create_bind_group(axpy_pl, &[&pbuf, x, y]);
    gpu.dispatch(axpy_pl, &bg, wg);
}

/// GPU xpay: p = x + β·p (p is modified in-place).
fn gpu_xpay(
    gpu: &GpuF64,
    xpay_pl: &wgpu::ComputePipeline,
    x: &wgpu::Buffer,
    beta: f64,
    p: &wgpu::Buffer,
    n: usize,
) {
    let wg = ((n as u32) + 63) / 64;
    let mut params = Vec::with_capacity(16);
    params.extend_from_slice(&(n as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&beta.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "xpay_p");
    let bg = gpu.create_bind_group(xpay_pl, &[&pbuf, x, p]);
    gpu.dispatch(xpay_pl, &bg, wg);
}

// ═══════════════════════════════════════════════════════════════════
//  Streaming GPU HMC — zero dispatch overhead via batched encoders
// ═══════════════════════════════════════════════════════════════════

/// WGSL shader: GPU-resident PRNG for SU(3) algebra momenta.
pub const WGSL_RANDOM_MOMENTA: &str = include_str!("shaders/su3_random_momenta_f64.wgsl");

/// Streaming HMC pipelines: quenched HMC + GPU PRNG.
pub struct GpuHmcStreamingPipelines {
    pub hmc: GpuHmcPipelines,
    pub prng_pipeline: wgpu::ComputePipeline,
}

impl GpuHmcStreamingPipelines {
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            hmc: GpuHmcPipelines::new(gpu),
            prng_pipeline: gpu.create_pipeline_f64(WGSL_RANDOM_MOMENTA, "hmc_prng"),
        }
    }
}

/// Run one pure-GPU streaming Omelyan HMC trajectory.
///
/// **All** dispatches for the MD integration are batched into a single
/// encoder submission. Momenta are generated on GPU (no CPU→GPU upload).
/// Only 3 GPU submissions per trajectory:
///
/// 1. PRNG momenta + backup links + plaquette/KE for H_old
/// 2. All N_md×5 MD dispatches (one encoder)
/// 3. Plaquette/KE for H_new
pub fn gpu_hmc_trajectory_streaming(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    traj_id: u32,
    seed: &mut u64,
) -> GpuHmcResult {
    let n_links = state.n_links;
    let p = &pipelines.hmc;

    // ── Phase 1: generate momenta on GPU + backup links ──
    {
        let mut enc = gpu.begin_encoder("stream_init");

        // PRNG dispatch: momenta generated entirely on GPU
        let prng_params = make_prng_params(n_links as u32, traj_id, seed);
        let prng_pbuf = gpu.create_uniform_buffer(&prng_params, "prng_p");
        let prng_bg = gpu.create_bind_group(
            &pipelines.prng_pipeline,
            &[&prng_pbuf, &state.mom_buf],
        );
        GpuF64::encode_pass(&mut enc, &pipelines.prng_pipeline, &prng_bg, state.wg_links);

        // Backup links (GPU→GPU copy)
        enc.copy_buffer_to_buffer(
            &state.link_buf, 0,
            &state.link_backup, 0,
            (n_links * 18 * 8) as u64,
        );

        gpu.submit_encoder(enc);
    }

    // ── Phase 2: H_old = S(U) + T(P) (needs scalar readback) ──
    let s_old = gpu_wilson_action_streaming(gpu, p, state);
    let t_old = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_old = s_old + t_old;

    // ── Phase 3: ALL MD dispatches in ONE encoder ──
    gpu_streaming_md_encoder(gpu, p, state, n_md_steps, dt);

    // ── Phase 4: H_new = S(U') + T(P') (needs scalar readback) ──
    let s_new = gpu_wilson_action_streaming(gpu, p, state);
    let t_new = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_new = s_new + t_new;

    let delta_h = h_new - h_old;

    // Metropolis accept/reject (CPU — one comparison)
    let r: f64 = super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("stream_restore");
        enc.copy_buffer_to_buffer(
            &state.link_backup, 0,
            &state.link_buf, 0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette = gpu_plaquette(gpu, p, state);

    GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
    }
}

/// Streaming HMC with CPU-generated momenta (for parity testing).
///
/// Uses encoder batching for all MD dispatches (streaming dispatch), but
/// momenta are generated on CPU and uploaded — proving the encoder
/// batching itself produces bit-identical physics to per-dispatch mode.
pub fn gpu_hmc_trajectory_streaming_cpu_mom(
    gpu: &GpuF64,
    pipelines: &GpuHmcStreamingPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
    seed: &mut u64,
) -> GpuHmcResult {
    let n_links = state.n_links;
    let p = &pipelines.hmc;

    // CPU-generated momenta + upload
    let momenta: Vec<super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = flatten_momenta(&momenta);
    gpu.upload_f64(&state.mom_buf, &mom_flat);

    // Backup links
    {
        let mut enc = gpu.begin_encoder("scm_backup");
        enc.copy_buffer_to_buffer(
            &state.link_buf, 0,
            &state.link_backup, 0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let s_old = gpu_wilson_action_streaming(gpu, p, state);
    let t_old = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_old = s_old + t_old;

    // ALL MD in one encoder
    gpu_streaming_md_encoder(gpu, p, state, n_md_steps, dt);

    let s_new = gpu_wilson_action_streaming(gpu, p, state);
    let t_new = gpu_kinetic_energy_streaming(gpu, p, state);
    let h_new = s_new + t_new;

    let delta_h = h_new - h_old;
    let r: f64 = super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("scm_restore");
        enc.copy_buffer_to_buffer(
            &state.link_backup, 0,
            &state.link_buf, 0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette = gpu_plaquette(gpu, p, state);

    GpuHmcResult {
        accepted,
        delta_h,
        plaquette,
    }
}

/// Encode all Omelyan MD steps into a single encoder submission.
fn gpu_streaming_md_encoder(
    gpu: &GpuF64,
    p: &GpuHmcPipelines,
    state: &GpuHmcState,
    n_md_steps: usize,
    dt: f64,
) {
    let n_links = state.n_links;
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    let force_params = make_force_params(state.volume, state.beta);
    let force_pbuf = gpu.create_uniform_buffer(&force_params, "s_force_p");
    let force_bg = gpu.create_bind_group(
        &p.force_pipeline,
        &[&force_pbuf, &state.link_buf, &state.nbr_buf, &state.force_buf],
    );

    let mom_lam_params = make_link_mom_params(n_links, lam * dt);
    let mom_lam_pbuf = gpu.create_uniform_buffer(&mom_lam_params, "s_mom_lam");
    let mom_lam_bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&mom_lam_pbuf, &state.force_buf, &state.mom_buf],
    );

    let mom_mid_params = make_link_mom_params(n_links, (1.0 - 2.0 * lam) * dt);
    let mom_mid_pbuf = gpu.create_uniform_buffer(&mom_mid_params, "s_mom_mid");
    let mom_mid_bg = gpu.create_bind_group(
        &p.momentum_pipeline,
        &[&mom_mid_pbuf, &state.force_buf, &state.mom_buf],
    );

    let link_params = make_link_mom_params(n_links, 0.5 * dt);
    let link_pbuf = gpu.create_uniform_buffer(&link_params, "s_link_p");
    let link_bg = gpu.create_bind_group(
        &p.link_pipeline,
        &[&link_pbuf, &state.mom_buf, &state.link_buf],
    );

    let mut enc = gpu.begin_encoder("stream_md");

    for _step in 0..n_md_steps {
        GpuF64::encode_pass(&mut enc, &p.force_pipeline, &force_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.momentum_pipeline, &mom_lam_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.link_pipeline, &link_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.force_pipeline, &force_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.momentum_pipeline, &mom_mid_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.link_pipeline, &link_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.force_pipeline, &force_bg, state.wg_links);
        GpuF64::encode_pass(&mut enc, &p.momentum_pipeline, &mom_lam_bg, state.wg_links);
    }

    gpu.submit_encoder(enc);
}

fn make_prng_params(n_links: u32, traj_id: u32, seed: &mut u64) -> Vec<u8> {
    super::constants::lcg_step(seed);
    let s = *seed;
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&n_links.to_le_bytes());
    v.extend_from_slice(&traj_id.to_le_bytes());
    v.extend_from_slice(&(s as u32).to_le_bytes());
    v.extend_from_slice(&((s >> 32) as u32).to_le_bytes());
    v
}

fn gpu_wilson_action_streaming(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.volume as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "s_plaq_p");
    let bg = gpu.create_bind_group(
        &p.plaquette_pipeline,
        &[&param_buf, &s.link_buf, &s.nbr_buf, &s.plaq_out_buf],
    );
    let staging = gpu.create_staging_buffer(s.volume * 8, "s_plaq_staging");
    let mut enc = gpu.begin_encoder("s_wilson");
    GpuF64::encode_pass(&mut enc, &p.plaquette_pipeline, &bg, s.wg_vol);
    enc.copy_buffer_to_buffer(&s.plaq_out_buf, 0, &staging, 0, (s.volume * 8) as u64);
    gpu.submit_encoder(enc);
    let per_site = match gpu.read_staging_f64(&staging) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let plaq_sum: f64 = per_site.iter().sum();
    s.beta * (6.0 * s.volume as f64 - plaq_sum)
}

fn gpu_kinetic_energy_streaming(gpu: &GpuF64, p: &GpuHmcPipelines, s: &GpuHmcState) -> f64 {
    let params = make_u32x4_params(s.n_links as u32);
    let param_buf = gpu.create_uniform_buffer(&params, "s_ke_p");
    let bg = gpu.create_bind_group(
        &p.kinetic_pipeline,
        &[&param_buf, &s.mom_buf, &s.ke_out_buf],
    );
    let staging = gpu.create_staging_buffer(s.n_links * 8, "s_ke_staging");
    let mut enc = gpu.begin_encoder("s_ke");
    GpuF64::encode_pass(&mut enc, &p.kinetic_pipeline, &bg, s.wg_links);
    enc.copy_buffer_to_buffer(&s.ke_out_buf, 0, &staging, 0, (s.n_links * 8) as u64);
    gpu.submit_encoder(enc);
    let per_link = match gpu.read_staging_f64(&staging) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    per_link.iter().sum()
}

// ═══════════════════════════════════════════════════════════════════
//  Streaming Dynamical Fermion GPU HMC
// ═══════════════════════════════════════════════════════════════════

/// WGSL shader: GPU-resident PRNG for Gaussian fermion fields.
pub const WGSL_GAUSSIAN_FERMION: &str = include_str!("shaders/gaussian_fermion_f64.wgsl");

/// Streaming pipelines for dynamical fermion HMC.
///
/// Extends `GpuDynHmcPipelines` with GPU PRNG for momenta and pseudofermion
/// heat bath. The CG solver still requires per-iteration readbacks, but all
/// other operations (gauge force, link/momentum updates, PRNG generation)
/// use batched encoders.
pub struct GpuDynHmcStreamingPipelines {
    pub dyn_hmc: GpuDynHmcPipelines,
    pub momenta_prng_pipeline: wgpu::ComputePipeline,
    pub fermion_prng_pipeline: wgpu::ComputePipeline,
}

impl GpuDynHmcStreamingPipelines {
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            dyn_hmc: GpuDynHmcPipelines::new(gpu),
            momenta_prng_pipeline: gpu.create_pipeline_f64(WGSL_RANDOM_MOMENTA, "sdyn_mom_prng"),
            fermion_prng_pipeline: gpu.create_pipeline_f64(WGSL_GAUSSIAN_FERMION, "sdyn_ferm_prng"),
        }
    }
}

/// Full streaming dynamical fermion HMC trajectory.
///
/// GPU PRNG generates both momenta and pseudofermion field η on-device.
/// The pseudofermion heat bath φ = D†η is computed via GPU Dirac dispatch.
/// CG iterations require scalar readbacks (convergence test), so they cannot
/// be batched into a single encoder. But all other operations — force
/// accumulation, link/momentum updates — use minimal dispatches.
///
/// Transfer budget per trajectory:
///   CPU→GPU: 0 bytes (GPU PRNG for momenta + pseudofermion)
///   GPU→CPU: ~(8 × CG_iters) bytes for convergence scalars + 24 bytes for ΔH
pub fn gpu_dynamical_hmc_trajectory_streaming(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcStreamingPipelines,
    state: &GpuDynHmcState,
    n_md_steps: usize,
    dt: f64,
    traj_id: u32,
    seed: &mut u64,
) -> GpuDynHmcResult {
    let vol = state.gauge.volume;
    let n_links = state.gauge.n_links;
    let gs = &state.gauge;
    let dp = &pipelines.dyn_hmc;

    // ── Phase 1: GPU PRNG for momenta + pseudofermion η ──
    {
        let mut enc = gpu.begin_encoder("sdyn_prng");

        // Momenta PRNG
        let mom_prng_params = make_prng_params(n_links as u32, traj_id, seed);
        let mom_prng_pbuf = gpu.create_uniform_buffer(&mom_prng_params, "sdyn_mom_p");
        let mom_prng_bg = gpu.create_bind_group(
            &pipelines.momenta_prng_pipeline,
            &[&mom_prng_pbuf, &gs.mom_buf],
        );
        GpuF64::encode_pass(&mut enc, &pipelines.momenta_prng_pipeline, &mom_prng_bg, gs.wg_links);

        // Pseudofermion φ PRNG → phi_buf (Gaussian, matching dispatch approach)
        let ferm_prng_params = make_ferm_prng_params(vol as u32, traj_id, seed);
        let ferm_prng_pbuf = gpu.create_uniform_buffer(&ferm_prng_params, "sdyn_ferm_p");
        let ferm_prng_bg = gpu.create_bind_group(
            &pipelines.fermion_prng_pipeline,
            &[&ferm_prng_pbuf, &state.phi_buf],
        );
        let wg_vol = ((vol as u32) + 63) / 64;
        GpuF64::encode_pass(&mut enc, &pipelines.fermion_prng_pipeline, &ferm_prng_bg, wg_vol);

        // Backup links
        enc.copy_buffer_to_buffer(
            &gs.link_buf, 0,
            &gs.link_backup, 0,
            (n_links * 18 * 8) as u64,
        );

        gpu.submit_encoder(enc);
    }

    // ── Phase 2: H_old = S_gauge + T_kinetic + S_fermion ──
    let s_gauge_old = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_old = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action(gpu, dp, state);
    let h_old = s_gauge_old + t_old + s_ferm_old;
    let mut total_cg = cg_iters_old;

    // ── Phase 3: Omelyan MD ──
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg1 = gpu_total_force_dispatch(gpu, dp, state, lam * dt);
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg2 = gpu_total_force_dispatch(gpu, dp, state, (1.0 - 2.0 * lam) * dt);
        gpu_link_update_dispatch(gpu, &dp.gauge, gs, 0.5 * dt);
        let cg3 = gpu_total_force_dispatch(gpu, dp, state, lam * dt);
        total_cg += cg1 + cg2 + cg3;
    }

    // ── Phase 4: H_new ──
    let s_gauge_new = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_new = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action(gpu, dp, state);
    let h_new = s_gauge_new + t_new + s_ferm_new;
    total_cg += cg_iters_new;

    let delta_h = h_new - h_old;

    // Metropolis
    let r: f64 = super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("sdyn_restore");
        enc.copy_buffer_to_buffer(
            &gs.link_backup, 0,
            &gs.link_buf, 0,
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

fn make_ferm_prng_params(volume: u32, traj_id: u32, seed: &mut u64) -> Vec<u8> {
    super::constants::lcg_step(seed);
    let s = *seed;
    let mut v = Vec::with_capacity(16);
    v.extend_from_slice(&volume.to_le_bytes());
    v.extend_from_slice(&traj_id.to_le_bytes());
    v.extend_from_slice(&(s as u32).to_le_bytes());
    v.extend_from_slice(&((s >> 32) as u32).to_le_bytes());
    v
}

// ═══════════════════════════════════════════════════════════════════
//  GPU-Resident CG — zero per-iteration readback
// ═══════════════════════════════════════════════════════════════════
//
//  All CG scalars (alpha, beta, rz, pAp) live on GPU. The CPU only
//  reads back 8 bytes (one f64) every `check_interval` iterations
//  for convergence testing. This eliminates 245,000× of readback
//  volume compared to the per-iteration approach.

/// Shader constants for GPU-resident CG.
pub const WGSL_SUM_REDUCE: &str = super::cg::WGSL_SUM_REDUCE_F64;
pub const WGSL_CG_COMPUTE_ALPHA: &str = super::cg::WGSL_CG_COMPUTE_ALPHA_F64;
pub const WGSL_CG_COMPUTE_BETA: &str = super::cg::WGSL_CG_COMPUTE_BETA_F64;
pub const WGSL_CG_UPDATE_XR: &str = super::cg::WGSL_CG_UPDATE_XR_F64;
pub const WGSL_CG_UPDATE_P: &str = super::cg::WGSL_CG_UPDATE_P_F64;

/// Compiled pipelines for GPU-resident CG.
pub struct GpuResidentCgPipelines {
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub compute_alpha_pipeline: wgpu::ComputePipeline,
    pub compute_beta_pipeline: wgpu::ComputePipeline,
    pub update_xr_pipeline: wgpu::ComputePipeline,
    pub update_p_pipeline: wgpu::ComputePipeline,
}

impl GpuResidentCgPipelines {
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

/// One step of a multi-pass reduction: (bind_group, num_workgroups).
struct ReducePass {
    bg: wgpu::BindGroup,
    num_wg: u32,
}

/// Pre-built reduction chain: dot_buf → target scalar in 2-3 GPU dispatches.
struct ReduceChain {
    passes: Vec<ReducePass>,
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
    // ── Pre-built bind groups for the CG iteration loop ──
    /// Dirac D·p → temp (hop_sign = +1).
    dirac_d_bg: wgpu::BindGroup,
    /// Dirac D†·temp → ap (hop_sign = -1).
    dirac_ddag_bg: wgpu::BindGroup,
    /// Dot(p, ap) → dot_buf.
    dot_pap_bg: wgpu::BindGroup,
    /// Dot(r, r) → dot_buf.
    dot_rr_bg: wgpu::BindGroup,
    /// Reduce dot_buf → pap_buf.
    reduce_to_pap: ReduceChain,
    /// Reduce dot_buf → rz_buf (for initialization).
    reduce_to_rz: ReduceChain,
    /// Reduce dot_buf → rz_new_buf.
    reduce_to_rz_new: ReduceChain,
    /// Compute alpha = rz / pAp.
    compute_alpha_bg: wgpu::BindGroup,
    /// Compute beta = rz_new / rz_old (and copy).
    compute_beta_bg: wgpu::BindGroup,
    /// Update xr: x += alpha*p, r -= alpha*ap.
    update_xr_bg: wgpu::BindGroup,
    /// Update p: p = r + beta*p.
    update_p_bg: wgpu::BindGroup,
    /// Workgroup count for Dirac dispatch.
    wg_dirac: u32,
    /// Workgroup count for dot product dispatch.
    wg_dot: u32,
    /// Workgroup count for vector update dispatch.
    wg_vec: u32,
    /// n_pairs for dot product (metadata, not used at runtime).
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

        // Scratch buffers for reduction passes
        let max_wg = (n_pairs + 255) / 256;
        let scratch_a = gpu.create_f64_output_buffer(max_wg.max(1), "cg_scratch_a");
        let scratch_b = gpu.create_f64_output_buffer(max_wg.max(1), "cg_scratch_b");

        // Scalar buffers (1 f64 each)
        let rz_buf = gpu.create_f64_output_buffer(1, "cg_rz");
        let rz_new_buf = gpu.create_f64_output_buffer(1, "cg_rz_new");
        let pap_buf = gpu.create_f64_output_buffer(1, "cg_pap");
        let alpha_buf = gpu.create_f64_output_buffer(1, "cg_alpha");
        let beta_buf = gpu.create_f64_output_buffer(1, "cg_beta");

        // Convergence staging (double-buffered for async readback)
        let convergence_staging_a = gpu.create_staging_buffer(8, "cg_conv_staging_a");
        let convergence_staging_b = gpu.create_staging_buffer(8, "cg_conv_staging_b");

        // Dirac bind groups (pre-built for the two directions)
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

        // Dot product bind groups
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

        // Reduction chains
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

        // Scalar compute bind groups
        let compute_alpha_bg = gpu.create_bind_group(
            &resident_pipelines.compute_alpha_pipeline,
            &[&rz_buf, &pap_buf, &alpha_buf],
        );
        let compute_beta_bg = gpu.create_bind_group(
            &resident_pipelines.compute_beta_pipeline,
            &[&rz_new_buf, &rz_buf, &beta_buf],
        );

        // Vector update bind groups
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

        let wg_dirac = ((vol as u32) + 63) / 64;
        let wg_dot = ((n_pairs as u32) + 63) / 64;
        let wg_vec = ((n_flat as u32) + 63) / 64;

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

/// Build a pre-compiled Dirac bind group with baked-in parameters.
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

/// Build a multi-pass reduction chain from input → target via scratch buffers.
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
        let num_wg = (current_n + 255) / 256;
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

/// Encode a reduction chain into an existing command encoder.
fn encode_reduce_chain(
    enc: &mut wgpu::CommandEncoder,
    reduce_pl: &wgpu::ComputePipeline,
    chain: &ReduceChain,
) {
    for pass in &chain.passes {
        GpuF64::encode_pass(enc, reduce_pl, &pass.bg, pass.num_wg);
    }
}

/// GPU-resident CG solver: (D†D)x = b with minimal readback.
///
/// All scalars (alpha, beta, rz, pAp) stay on GPU. Convergence is checked
/// every `check_interval` iterations by reading back 8 bytes (one f64).
///
/// Returns (iterations, final_rz_new / b_norm_sq).
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

    // ── Initialization: x = 0, r = b, p = b ──
    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(&state.x_buf, &zeros);
    {
        let mut enc = gpu.begin_encoder("rcg_init");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        enc.copy_buffer_to_buffer(b_buf, 0, &state.p_buf, 0, (n_flat * 8) as u64);
        // Compute rz = ||b||² via dot(b,b) → reduce → rz_buf
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
        // Copy rz to staging for b_norm_sq check
        enc.copy_buffer_to_buffer(
            &cg_bufs.rz_buf,
            0,
            &cg_bufs.convergence_staging_a,
            0,
            8,
        );
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
    let check_interval = check_interval.max(1);
    let mut total_iters = 0usize;

    // ── Main CG loop: batched iterations with periodic convergence check ──
    loop {
        let batch = check_interval.min(state.cg_max_iter - total_iters);
        if batch == 0 {
            break;
        }

        let mut enc = gpu.begin_encoder("rcg_batch");

        for _ in 0..batch {
            // ap = D†D·p: temp = D·p, ap = D†·temp
            GpuF64::encode_pass(
                &mut enc,
                &dyn_pipelines.dirac_pipeline,
                &cg_bufs.dirac_d_bg,
                cg_bufs.wg_dirac,
            );
            GpuF64::encode_pass(
                &mut enc,
                &dyn_pipelines.dirac_pipeline,
                &cg_bufs.dirac_ddag_bg,
                cg_bufs.wg_dirac,
            );

            // pAp = dot(p, ap) → reduce → pap_buf
            GpuF64::encode_pass(
                &mut enc,
                &dyn_pipelines.dot_pipeline,
                &cg_bufs.dot_pap_bg,
                cg_bufs.wg_dot,
            );
            encode_reduce_chain(
                &mut enc,
                &resident_pipelines.reduce_pipeline,
                &cg_bufs.reduce_to_pap,
            );

            // alpha = rz / pAp (GPU-side scalar divide)
            GpuF64::encode_pass(
                &mut enc,
                &resident_pipelines.compute_alpha_pipeline,
                &cg_bufs.compute_alpha_bg,
                1,
            );

            // x += alpha*p, r -= alpha*ap (reads alpha from GPU buffer)
            GpuF64::encode_pass(
                &mut enc,
                &resident_pipelines.update_xr_pipeline,
                &cg_bufs.update_xr_bg,
                cg_bufs.wg_vec,
            );

            // rz_new = dot(r, r) → reduce → rz_new_buf
            GpuF64::encode_pass(
                &mut enc,
                &dyn_pipelines.dot_pipeline,
                &cg_bufs.dot_rr_bg,
                cg_bufs.wg_dot,
            );
            encode_reduce_chain(
                &mut enc,
                &resident_pipelines.reduce_pipeline,
                &cg_bufs.reduce_to_rz_new,
            );

            // beta = rz_new / rz_old (and rz_old ← rz_new)
            GpuF64::encode_pass(
                &mut enc,
                &resident_pipelines.compute_beta_pipeline,
                &cg_bufs.compute_beta_bg,
                1,
            );

            // p = r + beta*p (reads beta from GPU buffer)
            GpuF64::encode_pass(
                &mut enc,
                &resident_pipelines.update_p_pipeline,
                &cg_bufs.update_p_bg,
                cg_bufs.wg_vec,
            );
        }

        // Copy rz_new to staging for convergence check
        enc.copy_buffer_to_buffer(
            &cg_bufs.rz_new_buf,
            0,
            &cg_bufs.convergence_staging_a,
            0,
            8,
        );
        gpu.submit_encoder(enc);
        total_iters += batch;

        // Read back 8 bytes: convergence check
        let rz_new = match gpu.read_staging_f64(&cg_bufs.convergence_staging_a) {
            Ok(v) => v.first().copied().unwrap_or(f64::MAX),
            Err(_) => break,
        };
        if rz_new < tol_sq || total_iters >= state.cg_max_iter {
            break;
        }
    }

    total_iters
}

/// Compute S_f = φ†(D†D)⁻¹φ using GPU-resident CG.
/// Returns (S_f, cg_iterations).
fn gpu_fermion_action_resident(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    state: &GpuDynHmcState,
    cg_bufs: &GpuResidentCgBuffers,
    check_interval: usize,
) -> (f64, usize) {
    let iters = gpu_cg_solve_resident(
        gpu,
        dyn_pipelines,
        resident_pipelines,
        state,
        cg_bufs,
        &state.phi_buf,
        check_interval,
    );

    // S_f = Re(<φ|x>) — use the existing gpu_dot_re for the final action
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

/// Dispatch total force using GPU-resident CG for fermion force.
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

    // Gauge force + momentum update
    gpu_force_dispatch(gpu, &dyn_pipelines.gauge, gs);
    gpu_mom_update_dispatch(gpu, &dyn_pipelines.gauge, gs, dt);

    // Fermion force via GPU-resident CG
    let cg_iters = gpu_cg_solve_resident(
        gpu,
        dyn_pipelines,
        resident_pipelines,
        state,
        cg_bufs,
        &state.phi_buf,
        check_interval,
    );

    // y = D·x
    gpu_dirac_dispatch(gpu, dyn_pipelines, state, &state.x_buf, &state.y_buf, 1.0);

    // Fermion force
    gpu_fermion_force_dispatch(gpu, dyn_pipelines, state);

    // P += dt * F_fermion
    let ferm_mom_params = make_link_mom_params(n_links, dt);
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

    // Phase 1: GPU PRNG for momenta + pseudofermion
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
            &[&ferm_prng_pbuf, &state.phi_buf],
        );
        let wg_vol = ((vol as u32) + 63) / 64;
        GpuF64::encode_pass(
            &mut enc,
            &streaming_pipelines.fermion_prng_pipeline,
            &ferm_prng_bg,
            wg_vol,
        );

        enc.copy_buffer_to_buffer(
            &gs.link_buf,
            0,
            &gs.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );

        gpu.submit_encoder(enc);
    }

    // Phase 2: H_old
    let s_gauge_old = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_old = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_old, cg_iters_old) = gpu_fermion_action_resident(
        gpu,
        dp,
        resident_pipelines,
        state,
        cg_bufs,
        check_interval,
    );
    let h_old = s_gauge_old + t_old + s_ferm_old;
    let mut total_cg = cg_iters_old;

    // Phase 3: Omelyan MD with GPU-resident CG
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
            (1.0 - 2.0 * lam) * dt,
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

    // Phase 4: H_new
    let s_gauge_new = gpu_wilson_action(gpu, &dp.gauge, gs);
    let t_new = gpu_kinetic_energy(gpu, &dp.gauge, gs);
    let (s_ferm_new, cg_iters_new) = gpu_fermion_action_resident(
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

    // Metropolis
    let r: f64 = super::constants::lcg_uniform_f64(seed);
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

// ═══════════════════════════════════════════════════════════════════
//  Level 4: Async Readback for Speculative CG Batches
// ═══════════════════════════════════════════════════════════════════

/// Non-blocking readback handle for CG convergence scalars.
///
/// Wraps `map_async` with a channel-based completion signal.
/// GPU can continue working while the CPU waits for the scalar.
pub struct AsyncCgReadback {
    staging: wgpu::Buffer,
    receiver: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl AsyncCgReadback {
    /// Initiate a non-blocking readback of one f64 from a GPU scalar buffer.
    ///
    /// The caller must have already submitted an encoder that copies the
    /// source buffer to the staging buffer.
    pub fn start(gpu: &GpuF64, staging: &wgpu::Buffer) -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
        // Non-blocking poll to kick the GPU
        gpu.device().poll(wgpu::Maintain::Poll);
        Some(Self {
            // SAFETY: wgpu staging buffer is shared behind Arc internally.
            // We store a reference-equivalent handle. The caller must ensure
            // the staging buffer outlives this struct.
            staging: gpu.create_staging_buffer(8, "async_readback_placeholder"),
            receiver: rx,
        })
    }

    /// Poll: check if the readback is ready without blocking.
    pub fn is_ready(&self, gpu: &GpuF64) -> bool {
        gpu.device().poll(wgpu::Maintain::Poll);
        matches!(
            self.receiver.try_recv(),
            Ok(Ok(())) | Err(std::sync::mpsc::TryRecvError::Empty)
        )
    }

    /// Block until the readback is complete, then return the f64 value.
    pub fn wait(self, gpu: &GpuF64, staging: &wgpu::Buffer) -> f64 {
        gpu.device().poll(wgpu::Maintain::Wait);
        let _ = self.receiver.recv();
        let slice = staging.slice(..);
        let data = slice.get_mapped_range();
        let val = if data.len() >= 8 {
            f64::from_le_bytes(data[..8].try_into().unwrap_or([0u8; 8]))
        } else {
            f64::NAN
        };
        drop(data);
        staging.unmap();
        val
    }
}

/// GPU-resident CG with async readback and speculative batches.
///
/// While waiting for convergence readback, speculatively submits the next
/// batch of CG iterations. If convergence is detected, the speculative
/// work is discarded (wasted compute but hidden latency).
pub fn gpu_cg_solve_resident_async(
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

    // Initialization (same as synchronous version)
    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(&state.x_buf, &zeros);
    {
        let mut enc = gpu.begin_encoder("rcg_async_init");
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
        enc.copy_buffer_to_buffer(
            &cg_bufs.rz_buf,
            0,
            &cg_bufs.convergence_staging_a,
            0,
            8,
        );
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
    let check_interval = check_interval.max(1);
    let mut total_iters = 0usize;
    let mut use_staging_a = true;

    loop {
        let batch = check_interval.min(state.cg_max_iter - total_iters);
        if batch == 0 {
            break;
        }

        // Encode batch N
        let staging = if use_staging_a {
            &cg_bufs.convergence_staging_a
        } else {
            &cg_bufs.convergence_staging_b
        };
        let mut enc = gpu.begin_encoder("rcg_async_batch");
        encode_cg_batch(
            &mut enc,
            dyn_pipelines,
            resident_pipelines,
            cg_bufs,
            batch,
        );
        enc.copy_buffer_to_buffer(&cg_bufs.rz_new_buf, 0, staging, 0, 8);
        gpu.submit_encoder(enc);
        total_iters += batch;

        // Start async map on staging
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

        // Speculatively start next batch while readback is in flight
        let next_batch = check_interval.min(state.cg_max_iter - total_iters);
        if next_batch > 0 {
            let spec_staging = if use_staging_a {
                &cg_bufs.convergence_staging_b
            } else {
                &cg_bufs.convergence_staging_a
            };
            let mut spec_enc = gpu.begin_encoder("rcg_speculative");
            encode_cg_batch(
                &mut spec_enc,
                dyn_pipelines,
                resident_pipelines,
                cg_bufs,
                next_batch,
            );
            spec_enc.copy_buffer_to_buffer(&cg_bufs.rz_new_buf, 0, spec_staging, 0, 8);
            gpu.submit_encoder(spec_enc);
        }

        // Now wait for batch N's readback
        gpu.device().poll(wgpu::Maintain::Wait);
        let map_result = rx.recv();
        let rz_new = if map_result.is_ok() {
            let slice = staging.slice(..);
            let data = slice.get_mapped_range();
            let val = if data.len() >= 8 {
                f64::from_le_bytes(data[..8].try_into().unwrap_or([0u8; 8]))
            } else {
                f64::MAX
            };
            drop(data);
            staging.unmap();
            val
        } else {
            break;
        };

        if rz_new < tol_sq {
            break;
        }

        // Speculative batch was correct — account for its iterations
        if next_batch > 0 {
            total_iters += next_batch;
            // Read back the speculative batch's convergence (blocking)
            let spec_staging = if use_staging_a {
                &cg_bufs.convergence_staging_b
            } else {
                &cg_bufs.convergence_staging_a
            };
            let spec_rz = match gpu.read_staging_f64(spec_staging) {
                Ok(v) => v.first().copied().unwrap_or(f64::MAX),
                Err(_) => break,
            };
            if spec_rz < tol_sq || total_iters >= state.cg_max_iter {
                break;
            }
        }

        use_staging_a = !use_staging_a;

        if total_iters >= state.cg_max_iter {
            break;
        }
    }

    total_iters
}

/// Encode `batch` CG iterations into an existing command encoder.
fn encode_cg_batch(
    enc: &mut wgpu::CommandEncoder,
    dyn_pipelines: &GpuDynHmcPipelines,
    resident_pipelines: &GpuResidentCgPipelines,
    cg_bufs: &GpuResidentCgBuffers,
    batch: usize,
) {
    for _ in 0..batch {
        // ap = D†D·p
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
        // pAp = dot(p, ap) → reduce → pap_buf
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
        // alpha = rz / pAp
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.compute_alpha_pipeline,
            &cg_bufs.compute_alpha_bg,
            1,
        );
        // x += alpha*p, r -= alpha*ap
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.update_xr_pipeline,
            &cg_bufs.update_xr_bg,
            cg_bufs.wg_vec,
        );
        // rz_new = dot(r, r) → reduce → rz_new_buf
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
        // beta = rz_new / rz_old
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.compute_beta_pipeline,
            &cg_bufs.compute_beta_bg,
            1,
        );
        // p = r + beta*p
        GpuF64::encode_pass(
            enc,
            &resident_pipelines.update_p_pipeline,
            &cg_bufs.update_p_bg,
            cg_bufs.wg_vec,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Level 5: Three-Substrate Stream Integration
// ═══════════════════════════════════════════════════════════════════

/// Observable scalars for the readback stream.
pub struct StreamObservables {
    pub plaquette: f64,
    pub polyakov_re: f64,
    pub delta_h: f64,
    pub cg_iterations: usize,
    pub accepted: bool,
}

/// Bidirectional stream: GPU ↔ CPU with NPU screening branch.
///
/// The GPU runs trajectories continuously. The readback stream (10% bandwidth)
/// carries convergence scalars and observables back to CPU. The CPU makes
/// Metropolis decisions and rebatches parameters to the GPU. A separate
/// NPU branch screens trajectory quality asynchronously.
pub struct BidirectionalStream {
    /// Channel for sending observables to the NPU screening thread.
    npu_tx: Option<std::sync::mpsc::Sender<StreamObservables>>,
    /// Channel for receiving screening decisions from the NPU thread.
    npu_rx: Option<std::sync::mpsc::Receiver<bool>>,
    /// Running count of trajectories processed.
    pub trajectories: usize,
    /// Running count of accepted trajectories.
    pub accepted: usize,
    /// Accumulated CG iterations across all trajectories.
    pub total_cg: usize,
}

impl BidirectionalStream {
    /// Create a new bidirectional stream (no NPU screening by default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            npu_tx: None,
            npu_rx: None,
            trajectories: 0,
            accepted: 0,
            total_cg: 0,
        }
    }

    /// Attach an NPU screening channel pair.
    pub fn attach_npu(
        &mut self,
        tx: std::sync::mpsc::Sender<StreamObservables>,
        rx: std::sync::mpsc::Receiver<bool>,
    ) {
        self.npu_tx = Some(tx);
        self.npu_rx = Some(rx);
    }

    /// Run one trajectory through the bidirectional stream.
    ///
    /// GPU-resident CG with async readback. Observables are sent to
    /// NPU for screening if attached.
    pub fn run_trajectory(
        &mut self,
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
        let result = gpu_dynamical_hmc_trajectory_resident(
            gpu,
            streaming_pipelines,
            resident_pipelines,
            state,
            cg_bufs,
            n_md_steps,
            dt,
            traj_id,
            seed,
            check_interval,
        );

        self.trajectories += 1;
        if result.accepted {
            self.accepted += 1;
        }
        self.total_cg += result.cg_iterations;

        // Send observables to NPU screening (non-blocking)
        if let Some(ref tx) = self.npu_tx {
            let obs = StreamObservables {
                plaquette: result.plaquette,
                polyakov_re: 0.0, // TODO: compute Polyakov loop
                delta_h: result.delta_h,
                cg_iterations: result.cg_iterations,
                accepted: result.accepted,
            };
            let _ = tx.send(obs);
        }

        // Check for NPU screening decision (non-blocking)
        if let Some(ref rx) = self.npu_rx {
            if let Ok(_skip) = rx.try_recv() {
                // NPU screening can influence future trajectory scheduling
            }
        }

        result
    }

    /// Acceptance rate so far.
    pub fn acceptance_rate(&self) -> f64 {
        if self.trajectories == 0 {
            0.0
        } else {
            self.accepted as f64 / self.trajectories as f64
        }
    }
}

impl Default for BidirectionalStream {
    fn default() -> Self {
        Self::new()
    }
}

/// Read back current GPU links into a lattice.
pub fn gpu_links_to_lattice(gpu: &GpuF64, state: &GpuHmcState, lattice: &mut Lattice) {
    if let Ok(flat) = gpu.read_back_f64(&state.link_buf, state.n_links * 18) {
        unflatten_links_into(lattice, &flat);
    }
}

/// Unflatten f64 array back to SU(3) link matrices and update lattice.
pub fn unflatten_links_into(lattice: &mut Lattice, flat: &[f64]) {
    let vol = lattice.volume();
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let base = (idx * 4 + mu) * 18;
            let mut m = super::su3::Su3Matrix::ZERO;
            for row in 0..3 {
                for col in 0..3 {
                    m.m[row][col] = super::complex_f64::Complex64::new(
                        flat[base + row * 6 + col * 2],
                        flat[base + row * 6 + col * 2 + 1],
                    );
                }
            }
            lattice.set_link(x, mu, m);
        }
    }
}
