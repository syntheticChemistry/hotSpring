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
//! - Reuses existing `axpy_f64.wgsl`, `xpay_f64.wgsl` for shifted vector updates
//! - Reuses `dirac_staggered_f64.wgsl` for the shared matrix-vector product
//! - Reuses `complex_dot_re_f64.wgsl` + `sum_reduce_f64.wgsl` for scalar products
//!
//! ## References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — multi-shift CG
//! - Jegerlehner, hep-lat/9612014 — shifted Krylov methods

use super::super::rhmc::{RhmcConfig, RhmcFermionConfig};
use super::{GpuDynHmcState, GpuF64};

/// WGSL source for the multi-shift zeta recurrence kernel.
pub const WGSL_MULTI_SHIFT_ZETA: &str =
    include_str!("../shaders/multi_shift_zeta_f64.wgsl");

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

        let x_bufs: Vec<wgpu::Buffer> = (0..n_shifts)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rhmc_x_{i}")),
                    size: vec_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let p_bufs: Vec<wgpu::Buffer> = (0..n_shifts)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rhmc_p_{i}")),
                    size: vec_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let phi_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rhmc_phi"),
            size: vec_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        // Upload shift values
        let sigma_bytes: Vec<u8> = config
            .action_approx
            .sigma
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        queue.write_buffer(&self.shifts_buf, 0, &sigma_bytes);

        // zeta_curr = 1.0, zeta_prev = 1.0
        let ones: Vec<u8> = (0..self.n_shifts)
            .flat_map(|_| 1.0_f64.to_le_bytes())
            .collect();
        queue.write_buffer(&self.zeta_curr_buf, 0, &ones);
        queue.write_buffer(&self.zeta_prev_buf, 0, &ones);

        // beta_prev = 0.0
        let zeros_f64: Vec<u8> = (0..self.n_shifts)
            .flat_map(|_| 0.0_f64.to_le_bytes())
            .collect();
        queue.write_buffer(&self.beta_prev_buf, 0, &zeros_f64);

        // active = 1
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
    /// Gauge field state (links, momenta, neighbor table).
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
