// SPDX-License-Identifier: AGPL-3.0-only

//! GPU Rational HMC — multi-shift CG for fractional determinant powers.
//!
//! Enables GPU-accelerated simulation of Nf=2, 2+1 via the rooting trick:
//! `det(D†D)^{Nf/8}` ≈ product of rational functions.
//!
//! The multi-shift CG shares a single Krylov space across all shifts — only
//! ONE `D†D·p` per iteration regardless of pole count. This makes GPU RHMC
//! efficient: the expensive Dirac dispatch happens once, then cheap BLAS-1
//! updates run per shift.
//!
//! ## Shader pipeline
//!
//! - `multi_shift_zeta_f64.wgsl` — scalar zeta recurrence (1 workgroup)
//! - `shift_update_p_f64.wgsl` — fused `p = ζ·r + β·p` per shift
//! - Reuses existing `axpy_f64.wgsl` for shifted x updates
//! - Reuses `dirac_staggered_f64.wgsl` for the shared matrix-vector product
//! - Reuses `complex_dot_re_f64.wgsl` + `sum_reduce_f64.wgsl` for scalar products
//!
//! ## References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — multi-shift CG
//! - Jegerlehner, hep-lat/9612014 — shifted Krylov methods

use super::super::rhmc::{RhmcConfig, RhmcFermionConfig};
use super::dynamical::{gen_random_fermion, gpu_axpy, GpuDynHmcPipelines, GpuDynHmcState};
use super::{
    flatten_momenta, gpu_dot_re, gpu_force_dispatch, gpu_kinetic_energy, gpu_link_update_dispatch,
    gpu_mom_update_dispatch, gpu_plaquette, gpu_wilson_action, make_link_mom_params,
    make_u32x4_params, GpuF64, GpuHmcState,
};

/// WGSL source for the multi-shift zeta recurrence kernel.
pub const WGSL_MULTI_SHIFT_ZETA: &str =
    include_str!("../shaders/multi_shift_zeta_f64.wgsl");

/// WGSL source for the shifted search direction update: p = ζ·r + β·p.
pub const WGSL_SHIFT_UPDATE_P: &str =
    include_str!("../shaders/shift_update_p_f64.wgsl");

/// Maximum number of rational approximation poles supported.
///
/// Typical RHMC uses 8-16 poles. Buffers are allocated for this maximum.
pub const MAX_POLES: usize = 32;

/// GPU buffers for one RHMC fermion sector (one flavor group).
///
/// Each sector has its own set of shifted CG vectors and scalars.
pub struct GpuRhmcSectorBuffers {
    /// Number of active shifts (poles) for this sector.
    pub n_shifts: usize,
    /// Shift values σ_i (GPU buffer, `n_shifts` × f64).
    pub shifts_buf: wgpu::Buffer,
    /// Solution vectors x_s per shift (`n_shifts` buffers, each `vol × 6 × f64`).
    pub x_bufs: Vec<wgpu::Buffer>,
    /// Search direction vectors p_s per shift.
    pub p_bufs: Vec<wgpu::Buffer>,
    /// Pseudofermion field φ for this sector.
    pub phi_buf: wgpu::Buffer,
    /// Zeta recurrence state (current, per shift).
    pub zeta_curr_buf: wgpu::Buffer,
    /// Zeta recurrence state (previous, per shift).
    pub zeta_prev_buf: wgpu::Buffer,
    /// Beta per shift (previous iteration).
    pub beta_prev_buf: wgpu::Buffer,
    /// Output: alpha per shift (computed by zeta kernel).
    pub alpha_shift_buf: wgpu::Buffer,
    /// Output: beta per shift.
    pub beta_shift_buf: wgpu::Buffer,
    /// Active flags per shift (u32, 0 or 1).
    pub active_buf: wgpu::Buffer,
}

impl GpuRhmcSectorBuffers {
    /// Allocate GPU buffers for one RHMC sector.
    pub fn new(gpu: &GpuF64, config: &RhmcFermionConfig, volume: usize) -> Self {
        let n_shifts = config.action_approx.sigma.len();
        let n_flat = volume * 6;
        let vec_bytes = (n_flat * 8) as u64;
        let shift_bytes = (n_shifts * 8) as u64;

        let device = gpu.device();

        let shifts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rhmc_shifts"),
            size: shift_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vec_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let x_bufs: Vec<wgpu::Buffer> = (0..n_shifts)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rhmc_x_{i}")),
                    size: vec_bytes,
                    usage: vec_usage,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let p_bufs: Vec<wgpu::Buffer> = (0..n_shifts)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rhmc_p_{i}")),
                    size: vec_bytes,
                    usage: vec_usage,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let phi_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rhmc_phi"),
            size: vec_bytes,
            usage: vec_usage,
            mapped_at_creation: false,
        });

        let scalar_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: shift_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let active_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rhmc_active"),
            size: (n_shifts * 4) as u64, // u32 per shift
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            n_shifts,
            shifts_buf,
            x_bufs,
            p_bufs,
            phi_buf,
            zeta_curr_buf: scalar_buf("rhmc_zeta_curr"),
            zeta_prev_buf: scalar_buf("rhmc_zeta_prev"),
            beta_prev_buf: scalar_buf("rhmc_beta_prev"),
            alpha_shift_buf: scalar_buf("rhmc_alpha_s"),
            beta_shift_buf: scalar_buf("rhmc_beta_s"),
            active_buf,
        }
    }

    /// Initialize scalar state for a new solve: zeta=1, beta_prev=0, active=1.
    pub fn init_solve(&self, gpu: &GpuF64, config: &RhmcFermionConfig) {
        let queue = gpu.queue();

        let sigma_bytes: Vec<u8> = config
            .action_approx
            .sigma
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        queue.write_buffer(&self.shifts_buf, 0, &sigma_bytes);

        let ones: Vec<u8> = (0..self.n_shifts)
            .flat_map(|_| 1.0_f64.to_le_bytes())
            .collect();
        queue.write_buffer(&self.zeta_curr_buf, 0, &ones);
        queue.write_buffer(&self.zeta_prev_buf, 0, &ones);

        let zeros_f64: Vec<u8> = (0..self.n_shifts)
            .flat_map(|_| 0.0_f64.to_le_bytes())
            .collect();
        queue.write_buffer(&self.beta_prev_buf, 0, &zeros_f64);

        let active: Vec<u8> = (0..self.n_shifts)
            .flat_map(|_| 1u32.to_le_bytes())
            .collect();
        queue.write_buffer(&self.active_buf, 0, &active);
    }
}

/// GPU RHMC pipelines for multi-shift CG and RHMC trajectory.
pub struct GpuRhmcPipelines {
    /// Zeta recurrence kernel (scalar, 1 workgroup per shift).
    pub zeta_pipeline: wgpu::ComputePipeline,
    /// Shifted search direction update: p = ζ·r + β·p.
    pub shift_update_p_pipeline: wgpu::ComputePipeline,
}

impl GpuRhmcPipelines {
    /// Compile RHMC-specific pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            zeta_pipeline: gpu.create_pipeline_f64(WGSL_MULTI_SHIFT_ZETA, "rhmc_zeta"),
            shift_update_p_pipeline: gpu.create_pipeline_f64_precise(
                WGSL_SHIFT_UPDATE_P,
                "rhmc_shift_p",
            ),
        }
    }
}

/// Full GPU RHMC state: gauge + per-sector fermion buffers.
pub struct GpuRhmcState {
    /// Gauge field state (links, momenta, neighbor table) + shared CG workspace.
    pub gauge: GpuDynHmcState,
    /// Per-sector RHMC buffers (one per flavor group in `RhmcConfig::sectors`).
    pub sectors: Vec<GpuRhmcSectorBuffers>,
}

impl GpuRhmcState {
    /// Allocate GPU state for an RHMC simulation.
    pub fn new(gpu: &GpuF64, config: &RhmcConfig, gauge: GpuDynHmcState) -> Self {
        let volume = gauge.gauge.volume;
        let sectors = config
            .sectors
            .iter()
            .map(|s| GpuRhmcSectorBuffers::new(gpu, s, volume))
            .collect();
        Self { gauge, sectors }
    }
}

/// Result of a GPU RHMC trajectory.
#[derive(Debug, Clone)]
pub struct GpuRhmcResult {
    /// Whether the Metropolis test accepted this trajectory.
    pub accepted: bool,
    /// Hamiltonian change ΔH.
    pub delta_h: f64,
    /// Mean plaquette after trajectory.
    pub plaquette: f64,
    /// Total CG iterations across all sectors and force evaluations.
    pub total_cg_iterations: usize,
}

// ═══════════════════════════════════════════════════════════════════
//  Decoupled dispatch helpers (mass and buffers taken explicitly)
// ═══════════════════════════════════════════════════════════════════

fn dirac_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    gauge: &GpuHmcState,
    phases_buf: &wgpu::Buffer,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    mass: f64,
    hop_sign: f64,
) {
    let vol = gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let mut params = Vec::with_capacity(24);
    params.extend_from_slice(&(vol as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&mass.to_le_bytes());
    params.extend_from_slice(&hop_sign.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "dirac_p");
    let bg = gpu.create_bind_group(
        &pipelines.dirac_pipeline,
        &[&pbuf, &gauge.link_buf, input, output, &gauge.nbr_buf, phases_buf],
    );
    gpu.dispatch(&pipelines.dirac_pipeline, &bg, wg);
}

fn fermion_force_dispatch(
    gpu: &GpuF64,
    pipelines: &GpuDynHmcPipelines,
    gauge: &GpuHmcState,
    phases_buf: &wgpu::Buffer,
    x_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
    ferm_force_buf: &wgpu::Buffer,
) {
    let vol = gauge.volume;
    let wg = (vol as u32).div_ceil(64);
    let params = make_u32x4_params(vol as u32);
    let pbuf = gpu.create_uniform_buffer(&params, "fforce_p");
    let bg = gpu.create_bind_group(
        &pipelines.fermion_force_pipeline,
        &[&pbuf, &gauge.link_buf, x_buf, y_buf, &gauge.nbr_buf, phases_buf, ferm_force_buf],
    );
    gpu.dispatch(&pipelines.fermion_force_pipeline, &bg, wg);
}

fn gpu_shift_update_p(
    gpu: &GpuF64,
    pipeline: &wgpu::ComputePipeline,
    r: &wgpu::Buffer,
    p: &wgpu::Buffer,
    zeta: f64,
    beta: f64,
    n: usize,
) {
    let wg = (n as u32).div_ceil(64);
    let mut params = Vec::with_capacity(24);
    params.extend_from_slice(&(n as u32).to_le_bytes());
    params.extend_from_slice(&0u32.to_le_bytes());
    params.extend_from_slice(&zeta.to_le_bytes());
    params.extend_from_slice(&beta.to_le_bytes());
    let pbuf = gpu.create_uniform_buffer(&params, "sup_p");
    let bg = gpu.create_bind_group(pipeline, &[&pbuf, r, p]);
    gpu.dispatch(pipeline, &bg, wg);
}

// ═══════════════════════════════════════════════════════════════════
//  GPU Multi-Shift CG Solver
// ═══════════════════════════════════════════════════════════════════

/// GPU multi-shift CG: solve (D†D + σ_s) x_s = b for all shifts simultaneously.
///
/// All shifted systems share the same Krylov space. Only one D†D·p per
/// iteration regardless of shift count. The CPU handles the lightweight
/// zeta recurrence (~20 FLOPs per shift per iteration) while the GPU
/// does all vector operations.
///
/// Uses `state.r_buf`, `state.p_buf`, `state.ap_buf`, `state.temp_buf` as
/// shared workspace. Per-shift solutions go into `sector.x_bufs[s]` and
/// per-shift search directions into `sector.p_bufs[s]`.
///
/// Returns total CG iterations.
pub fn gpu_multi_shift_cg_solve(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    rhmc_pipelines: &GpuRhmcPipelines,
    state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    b_buf: &wgpu::Buffer,
    shifts: &[f64],
    mass: f64,
    tol: f64,
    max_iter: usize,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let n_pairs = vol * 3;
    let n_shifts = shifts.len();

    // Initialize: all x_s = 0
    let zeros = vec![0.0_f64; n_flat];
    for x_buf in &sector.x_bufs {
        gpu.upload_f64(x_buf, &zeros);
    }

    // r = b, p_0 = b, all p_s = b
    {
        let mut enc = gpu.begin_encoder("mcg_init");
        enc.copy_buffer_to_buffer(b_buf, 0, &state.r_buf, 0, (n_flat * 8) as u64);
        enc.copy_buffer_to_buffer(b_buf, 0, &state.p_buf, 0, (n_flat * 8) as u64);
        for p_buf in &sector.p_bufs {
            enc.copy_buffer_to_buffer(b_buf, 0, p_buf, 0, (n_flat * 8) as u64);
        }
        gpu.submit_encoder(enc);
    }

    let b_norm_sq = gpu_dot_re(
        gpu,
        &dyn_pipelines.dot_pipeline,
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

    // CPU-side zeta recurrence state (cheap: ~20 FLOPs per shift per iteration)
    let mut zeta_prev = vec![1.0_f64; n_shifts];
    let mut zeta_curr = vec![1.0_f64; n_shifts];
    let mut beta_shift_prev = vec![0.0_f64; n_shifts];
    let mut alpha_prev = 0.0_f64;
    let mut active = vec![true; n_shifts];

    let mut iterations = 0;
    let gauge = &state.gauge;
    let phases = &state.phases_buf;

    for _iter in 0..max_iter {
        iterations += 1;

        // D†D·p_0 → ap_buf (the expensive part, shared across all shifts)
        dirac_dispatch(gpu, dyn_pipelines, gauge, phases, &state.p_buf, &state.temp_buf, mass, 1.0);
        dirac_dispatch(gpu, dyn_pipelines, gauge, phases, &state.temp_buf, &state.ap_buf, mass, -1.0);

        // pAp = ⟨p_0 | D†D·p_0 + σ_0·p_0⟩ = ⟨p_0|ap⟩ + σ_0·⟨p_0|p_0⟩
        let mut p_ap = gpu_dot_re(
            gpu,
            &dyn_pipelines.dot_pipeline,
            &state.dot_buf,
            &state.p_buf,
            &state.ap_buf,
            n_pairs,
        );
        if shifts[0].abs() > 1e-30 {
            let p_p = gpu_dot_re(
                gpu,
                &dyn_pipelines.dot_pipeline,
                &state.dot_buf,
                &state.p_buf,
                &state.p_buf,
                n_pairs,
            );
            p_ap += shifts[0] * p_p;
        }

        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / p_ap;

        // Reference shift (s=0): x_0 += α·p_0
        gpu_axpy(
            gpu,
            &dyn_pipelines.axpy_pipeline,
            alpha,
            &state.p_buf,
            &sector.x_bufs[0],
            n_flat,
        );

        // r -= α·(D†D·p_0 + σ_0·p_0)
        gpu_axpy(
            gpu,
            &dyn_pipelines.axpy_pipeline,
            -alpha,
            &state.ap_buf,
            &state.r_buf,
            n_flat,
        );
        if shifts[0].abs() > 1e-30 {
            gpu_axpy(
                gpu,
                &dyn_pipelines.axpy_pipeline,
                -alpha * shifts[0],
                &state.p_buf,
                &state.r_buf,
                n_flat,
            );
        }

        let rz_new = gpu_dot_re(
            gpu,
            &dyn_pipelines.dot_pipeline,
            &state.dot_buf,
            &state.r_buf,
            &state.r_buf,
            n_pairs,
        );

        // CPU-side zeta recurrence + GPU per-shift vector updates
        for s in 1..n_shifts {
            if !active[s] {
                continue;
            }

            let ds = shifts[s] - shifts[0];
            let bp = beta_shift_prev[s];
            let mut denom = 1.0 + alpha * ds;
            if bp.abs() > 1e-30 {
                denom += alpha * alpha_prev * (1.0 - zeta_prev[s] / zeta_curr[s]) / bp;
            }

            if denom.abs() < 1e-30 {
                active[s] = false;
                continue;
            }

            let zeta_next = zeta_curr[s] / denom;
            let alpha_s = alpha * zeta_next / zeta_curr[s];

            // x_s += α_s · p_s
            gpu_axpy(
                gpu,
                &dyn_pipelines.axpy_pipeline,
                alpha_s,
                &sector.p_bufs[s],
                &sector.x_bufs[s],
                n_flat,
            );

            let beta_s = if rz.abs() > 1e-30 {
                (zeta_next / zeta_curr[s]).powi(2) * (rz_new / rz)
            } else {
                0.0
            };

            // p_s = ζ_next · r + β_s · p_s
            gpu_shift_update_p(
                gpu,
                &rhmc_pipelines.shift_update_p_pipeline,
                &state.r_buf,
                &sector.p_bufs[s],
                zeta_next,
                beta_s,
                n_flat,
            );

            zeta_prev[s] = zeta_curr[s];
            zeta_curr[s] = zeta_next;
            beta_shift_prev[s] = beta_s;
        }

        // Reference search direction update: p_0 = r + β·p_0
        let beta_cg = if rz.abs() > 1e-30 { rz_new / rz } else { 0.0 };
        super::dynamical::gpu_xpay(
            gpu,
            &dyn_pipelines.xpay_pipeline,
            &state.r_buf,
            beta_cg,
            &state.p_buf,
            n_flat,
        );

        alpha_prev = alpha;
        rz = rz_new;

        if rz < tol_sq {
            break;
        }
    }

    iterations
}

// ═══════════════════════════════════════════════════════════════════
//  RHMC Heatbath, Action, Force
// ═══════════════════════════════════════════════════════════════════

/// RHMC heatbath for one sector: generate φ = r_hb(D†D) η.
///
/// r_hb(x) = α₀ + Σ αₛ/(x + σₛ) approximates x^{-det_power/2}.
/// η is Gaussian noise. The multi-shift CG solves (D†D + σₛ) x_s = η,
/// then φ = α₀·η + Σ αₛ·x_s.
///
/// Returns CG iteration count.
fn gpu_rhmc_heatbath_sector(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    rhmc_pipelines: &GpuRhmcPipelines,
    state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: &mut u64,
) -> usize {
    let vol = state.gauge.volume;
    let n_flat = vol * 6;
    let approx = &config.heatbath_approx;

    // Generate Gaussian noise η and upload to phi_buf
    let eta = gen_random_fermion(vol, seed);
    gpu.upload_f64(&sector.phi_buf, &eta);

    // Multi-shift CG: (D†D + σₛ) x_s = η (b_buf = phi_buf = η)
    let cg_iters = gpu_multi_shift_cg_solve(
        gpu,
        dyn_pipelines,
        rhmc_pipelines,
        state,
        sector,
        &sector.phi_buf,
        &approx.sigma,
        config.mass,
        cg_tol,
        cg_max_iter,
    );

    // Accumulate φ = α₀·η + Σ αₛ·x_s using state.x_buf as temporary
    let zeros = vec![0.0_f64; n_flat];
    gpu.upload_f64(&state.x_buf, &zeros);

    // state.x_buf += α₀ · phi_buf (phi_buf still holds η)
    gpu_axpy(
        gpu,
        &dyn_pipelines.axpy_pipeline,
        approx.alpha_0,
        &sector.phi_buf,
        &state.x_buf,
        n_flat,
    );

    // state.x_buf += αₛ · x_bufs[s] for each shift
    for (s, a_s) in approx.alpha.iter().enumerate() {
        gpu_axpy(
            gpu,
            &dyn_pipelines.axpy_pipeline,
            *a_s,
            &sector.x_bufs[s],
            &state.x_buf,
            n_flat,
        );
    }

    // Copy result to phi_buf: φ = accumulated sum
    {
        let mut enc = gpu.begin_encoder("rhmc_phi_copy");
        enc.copy_buffer_to_buffer(&state.x_buf, 0, &sector.phi_buf, 0, (n_flat * 8) as u64);
        gpu.submit_encoder(enc);
    }

    cg_iters
}

/// RHMC fermion action for one sector: S_f = φ† r(D†D) φ.
///
/// r(x) = α₀ + Σ αₛ/(x + σₛ) approximates x^{det_power}.
/// After multi-shift CG: S_f = α₀·⟨φ|φ⟩ + Σ αₛ·⟨φ|x_s⟩.
///
/// Returns (action, cg_iterations).
fn gpu_rhmc_fermion_action_sector(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    rhmc_pipelines: &GpuRhmcPipelines,
    state: &GpuDynHmcState,
    sector: &GpuRhmcSectorBuffers,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (f64, usize) {
    let vol = state.gauge.volume;
    let n_pairs = vol * 3;
    let approx = &config.action_approx;

    let cg_iters = gpu_multi_shift_cg_solve(
        gpu,
        dyn_pipelines,
        rhmc_pipelines,
        state,
        sector,
        &sector.phi_buf,
        &approx.sigma,
        config.mass,
        cg_tol,
        cg_max_iter,
    );

    // S_f = α₀·⟨φ|φ⟩ + Σ αₛ·⟨φ|x_s⟩
    let phi_phi = gpu_dot_re(
        gpu,
        &dyn_pipelines.dot_pipeline,
        &state.dot_buf,
        &sector.phi_buf,
        &sector.phi_buf,
        n_pairs,
    );
    let mut action = approx.alpha_0 * phi_phi;

    for (s, a_s) in approx.alpha.iter().enumerate() {
        let phi_xs = gpu_dot_re(
            gpu,
            &dyn_pipelines.dot_pipeline,
            &state.dot_buf,
            &sector.phi_buf,
            &sector.x_bufs[s],
            n_pairs,
        );
        action += a_s * phi_xs;
    }

    (action, cg_iters)
}

/// RHMC total force: gauge + Σ_sectors Σ_poles αₛ · fermion_force(x_s).
///
/// For each sector, runs multi-shift CG, then for each pole computes the
/// fermion force from the shifted solution and accumulates into momenta.
///
/// Returns total CG iterations.
fn gpu_rhmc_total_force_dispatch(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    rhmc_pipelines: &GpuRhmcPipelines,
    state: &GpuDynHmcState,
    sectors: &[GpuRhmcSectorBuffers],
    configs: &[RhmcFermionConfig],
    dt: f64,
    cg_tol: f64,
    cg_max_iter: usize,
) -> usize {
    let gauge = &state.gauge;
    let n_links = gauge.n_links;

    // Gauge force → P += dt · F_gauge
    gpu_force_dispatch(gpu, &dyn_pipelines.gauge, gauge);
    gpu_mom_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, dt);

    let mut total_cg = 0;

    for (sector, config) in sectors.iter().zip(configs.iter()) {
        let approx = &config.force_approx;

        // Multi-shift CG: (D†D + σₛ) x_s = φ
        let cg_iters = gpu_multi_shift_cg_solve(
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            state,
            sector,
            &sector.phi_buf,
            &approx.sigma,
            config.mass,
            cg_tol,
            cg_max_iter,
        );
        total_cg += cg_iters;

        // For each pole: compute fermion force from x_s and accumulate
        for (s, a_s) in approx.alpha.iter().enumerate() {
            // Copy x_s → state.x_buf for fermion force shader
            {
                let n_flat = gauge.volume * 6;
                let mut enc = gpu.begin_encoder("rhmc_xcopy");
                enc.copy_buffer_to_buffer(
                    &sector.x_bufs[s],
                    0,
                    &state.x_buf,
                    0,
                    (n_flat * 8) as u64,
                );
                gpu.submit_encoder(enc);
            }

            // y = D · x_s
            dirac_dispatch(
                gpu,
                dyn_pipelines,
                gauge,
                &state.phases_buf,
                &state.x_buf,
                &state.y_buf,
                config.mass,
                1.0,
            );

            // Fermion force using x_s and y_s
            fermion_force_dispatch(
                gpu,
                dyn_pipelines,
                gauge,
                &state.phases_buf,
                &state.x_buf,
                &state.y_buf,
                &state.ferm_force_buf,
            );

            // P += αₛ · dt · F_ferm
            let ferm_mom_params =
                make_link_mom_params(n_links, *a_s * dt, gpu.full_df64_mode);
            let ferm_mom_pbuf = gpu.create_uniform_buffer(&ferm_mom_params, "fmom_p");
            let ferm_mom_bg = gpu.create_bind_group(
                &dyn_pipelines.gauge.momentum_pipeline,
                &[&ferm_mom_pbuf, &state.ferm_force_buf, &gauge.mom_buf],
            );
            gpu.dispatch(
                &dyn_pipelines.gauge.momentum_pipeline,
                &ferm_mom_bg,
                gauge.wg_links,
            );
        }
    }

    total_cg
}

// ═══════════════════════════════════════════════════════════════════
//  Full GPU RHMC Trajectory
// ═══════════════════════════════════════════════════════════════════

/// Run one GPU RHMC trajectory with Omelyan integrator.
///
/// Supports Nf=2 (one sector, det^{1/2}) and Nf=2+1 (two sectors,
/// det^{1/2} + det^{1/4}) via rational approximation of fractional
/// determinant powers. All heavy compute (Dirac, CG, force) runs on GPU;
/// CPU handles momenta generation, zeta recurrence, and Metropolis decision.
pub fn gpu_rhmc_trajectory(
    gpu: &GpuF64,
    dyn_pipelines: &GpuDynHmcPipelines,
    rhmc_pipelines: &GpuRhmcPipelines,
    state: &GpuRhmcState,
    config: &RhmcConfig,
    seed: &mut u64,
) -> GpuRhmcResult {
    let gauge = &state.gauge.gauge;
    let n_links = gauge.n_links;
    let n_md_steps = config.n_md_steps;
    let dt = config.dt;

    // 1. Generate random momenta (CPU → GPU)
    let momenta: Vec<super::super::su3::Su3Matrix> = (0..n_links)
        .map(|_| super::super::su3::Su3Matrix::random_algebra(seed))
        .collect();
    let mom_flat = flatten_momenta(&momenta);
    gpu.upload_f64(&gauge.mom_buf, &mom_flat);

    // Backup gauge links
    {
        let mut enc = gpu.begin_encoder("rhmc_backup_links");
        enc.copy_buffer_to_buffer(
            &gauge.link_buf,
            0,
            &gauge.link_backup,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    // 2. RHMC heatbath: generate φ for each sector
    let mut total_cg: usize = 0;
    for (sector, fconfig) in state.sectors.iter().zip(config.sectors.iter()) {
        let cg = gpu_rhmc_heatbath_sector(
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            &state.gauge,
            sector,
            fconfig,
            config.cg_tol,
            config.cg_max_iter,
            seed,
        );
        total_cg += cg;
    }

    // 3. H_old = S_gauge + T + Σ S_f
    let s_gauge_old = gpu_wilson_action(gpu, &dyn_pipelines.gauge, gauge);
    let t_old = gpu_kinetic_energy(gpu, &dyn_pipelines.gauge, gauge);
    let mut s_ferm_old = 0.0;
    for (sector, fconfig) in state.sectors.iter().zip(config.sectors.iter()) {
        let (sf, cg) = gpu_rhmc_fermion_action_sector(
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            &state.gauge,
            sector,
            fconfig,
            config.cg_tol,
            config.cg_max_iter,
        );
        s_ferm_old += sf;
        total_cg += cg;
    }
    let h_old = s_gauge_old + t_old + s_ferm_old;

    // 4. Omelyan MD integration
    let lam = crate::tolerances::OMELYAN_LAMBDA;

    for _step in 0..n_md_steps {
        let cg = gpu_rhmc_total_force_dispatch(
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            &state.gauge,
            &state.sectors,
            &config.sectors,
            lam * dt,
            config.cg_tol,
            config.cg_max_iter,
        );
        total_cg += cg;

        gpu_link_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, 0.5 * dt);

        let cg = gpu_rhmc_total_force_dispatch(
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            &state.gauge,
            &state.sectors,
            &config.sectors,
            2.0f64.mul_add(-lam, 1.0) * dt,
            config.cg_tol,
            config.cg_max_iter,
        );
        total_cg += cg;

        gpu_link_update_dispatch(gpu, &dyn_pipelines.gauge, gauge, 0.5 * dt);

        let cg = gpu_rhmc_total_force_dispatch(
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            &state.gauge,
            &state.sectors,
            &config.sectors,
            lam * dt,
            config.cg_tol,
            config.cg_max_iter,
        );
        total_cg += cg;
    }

    // 5. H_new = S_gauge + T + Σ S_f
    let s_gauge_new = gpu_wilson_action(gpu, &dyn_pipelines.gauge, gauge);
    let t_new = gpu_kinetic_energy(gpu, &dyn_pipelines.gauge, gauge);
    let mut s_ferm_new = 0.0;
    for (sector, fconfig) in state.sectors.iter().zip(config.sectors.iter()) {
        let (sf, cg) = gpu_rhmc_fermion_action_sector(
            gpu,
            dyn_pipelines,
            rhmc_pipelines,
            &state.gauge,
            sector,
            fconfig,
            config.cg_tol,
            config.cg_max_iter,
        );
        s_ferm_new += sf;
        total_cg += cg;
    }
    let h_new = s_gauge_new + t_new + s_ferm_new;

    // 6. Metropolis accept/reject
    let delta_h = h_new - h_old;
    let r: f64 = super::super::constants::lcg_uniform_f64(seed);
    let accepted = delta_h <= 0.0 || r < (-delta_h).exp();

    if !accepted {
        let mut enc = gpu.begin_encoder("rhmc_restore_links");
        enc.copy_buffer_to_buffer(
            &gauge.link_backup,
            0,
            &gauge.link_buf,
            0,
            (n_links * 18 * 8) as u64,
        );
        gpu.submit_encoder(enc);
    }

    let plaquette = gpu_plaquette(gpu, &dyn_pipelines.gauge, gauge);

    GpuRhmcResult {
        accepted,
        delta_h,
        plaquette,
        total_cg_iterations: total_cg,
    }
}
