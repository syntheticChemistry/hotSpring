// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU Rational HMC — multi-shift CG for fractional determinant powers.
//!
//! Enables GPU-accelerated simulation of Nf=2, 2+1 via the rooting trick:
//! `det(D†D)^{Nf/8}` via rational approximation of `(D†D)^{-α}`.
//!
//! Pseudofermion setup (Clark & Kennedy, NPB 552):
//!   - Action: `S_f = φ† (D†D)^{-α} φ` where α = Nf/8
//!   - Heatbath: `φ = (D†D)^{α/2} η` (consistency: `r_hb² · r_act = 1`)
//!
//! Uses independent per-shift CG solves for numerical stability.
//!
//! ## Shader pipeline
//!
//! - `multi_shift_zeta_f64.wgsl` — scalar zeta recurrence (1 workgroup)
//! - Reuses existing `axpy_f64.wgsl`, `xpay_f64.wgsl` for CG vector updates
//! - Reuses `dirac_staggered_f64.wgsl` for the matrix-vector product
//! - Reuses `complex_dot_re_f64.wgsl` + `sum_reduce_f64.wgsl` for scalar products
//!
//! ## References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — multi-shift CG
//! - Jegerlehner, hep-lat/9612014 — shifted Krylov methods

use super::super::rhmc::{RhmcConfig, RhmcFermionConfig};
use super::dynamical::{GpuDynHmcPipelines, GpuDynHmcState};
use super::{GpuF64, GpuHmcState, make_u32x4_params};

/// WGSL source for the multi-shift zeta recurrence kernel.
pub const WGSL_MULTI_SHIFT_ZETA: &str = include_str!("../shaders/multi_shift_zeta_f64.wgsl");

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
}

impl GpuRhmcPipelines {
    /// Compile RHMC-specific pipelines.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        Self {
            zeta_pipeline: gpu.create_pipeline_f64(WGSL_MULTI_SHIFT_ZETA, "rhmc_zeta"),
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
    /// Gauge action at start of trajectory.
    pub s_gauge_old: f64,
    /// Gauge action at end of trajectory.
    pub s_gauge_new: f64,
    /// Kinetic energy at start.
    pub t_old: f64,
    /// Kinetic energy at end.
    pub t_new: f64,
    /// Fermion action at start (sum over sectors).
    pub s_ferm_old: f64,
    /// Fermion action at end (sum over sectors).
    pub s_ferm_new: f64,
}

// ═══════════════════════════════════════════════════════════════════
//  Decoupled dispatch helpers (mass and buffers taken explicitly)
// ═══════════════════════════════════════════════════════════════════

pub(super) fn dirac_dispatch(
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
        &[
            &pbuf,
            &gauge.link_buf,
            input,
            output,
            &gauge.nbr_buf,
            phases_buf,
        ],
    );
    gpu.dispatch(&pipelines.dirac_pipeline, &bg, wg);
}

pub(super) fn fermion_force_dispatch(
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
        &[
            &pbuf,
            &gauge.link_buf,
            x_buf,
            y_buf,
            &gauge.nbr_buf,
            phases_buf,
            ferm_force_buf,
        ],
    );
    gpu.dispatch(&pipelines.fermion_force_pipeline, &bg, wg);
}

#[path = "rhmc_shifted_cg.rs"]
mod rhmc_shifted_cg;

// ═══════════════════════════════════════════════════════════════════
//  RHMC Heatbath, Action, Force (legacy helpers removed — use
//  unidirectional_rhmc::gpu_rhmc_trajectory_unidirectional)
// ═══════════════════════════════════════════════════════════════════

// Legacy RHMC trajectory helpers (gpu_rhmc_heatbath_sector,
// gpu_rhmc_fermion_action_sector, gpu_rhmc_total_force_dispatch,
// gpu_rhmc_trajectory) have been excised. Use
// `gpu_rhmc_trajectory_unidirectional` from `unidirectional_rhmc.rs`
// which achieves ~50x fewer GPU-CPU sync points via resident CG.
