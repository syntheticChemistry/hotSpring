// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-resident transport observables — unidirectional streaming.
//!
//! Computes VACF and stress ACF entirely on GPU, without position/velocity
//! readback. Only scalar transport coefficients come back to CPU.
//!
//! **Architecture**: During MD production, velocity snapshots are stored in
//! a single flat GPU buffer (ring). After production, a batched VACF shader
//! computes C(lag) for each lag in ONE dispatch (iterating over all time
//! origins inside the shader), then reduces to a scalar. Total GPU round
//! trips: `n_lag` × 2 (one dispatch + one reduction per lag).
//!
//! **Data flow**: GPU MD → GPU ring (copy) → batched VACF shader → reduce → scalar
//!
//! This eliminates both:
//!   - `O(n_frames` × N × 3 × 8) bytes of velocity readback
//!   - `O(n_frames` × `n_lag`) individual GPU dispatches (old per-origin approach)

use crate::gpu::GpuF64;
use crate::tolerances::DIVISION_GUARD;

use barracuda::pipeline::ReduceScalarPipeline;

/// Compute shader workgroup size (must match WGSL `@workgroup_size(N)`).
const WORKGROUP_SIZE: u32 = 64;

/// VACF plateau detection: D* considered converged after this many
/// seconds of non-increasing integral (expressed as time / `dt_dump`).
const PLATEAU_DETECTION_TIME: f64 = 20.0;

/// WGSL shader: batched VACF — one dispatch computes C(lag) across all origins.
pub const WGSL_VACF_BATCH_F64: &str = include_str!("../shaders/vacf_batch_f64.wgsl");

/// WGSL shader: per-particle v(t0) · v(t) for VACF (single-origin, kept for tests).
pub const WGSL_VACF_DOT_F64: &str = include_str!("../shaders/vacf_dot_f64.wgsl");

/// WGSL shader: per-particle `σ_xy` for Green-Kubo viscosity.
pub const WGSL_STRESS_VIRIAL_F64: &str = include_str!("../shaders/stress_virial_f64.wgsl");

/// GPU-resident velocity ring buffer backed by a single flat buffer.
///
/// Layout: `vel_flat[snapshot_idx * stride + particle * 3 + component]`
/// where `stride = n_particles * 3`.
///
/// GPU→GPU copy during production: `copy_buffer_to_buffer` from the live
/// velocity buffer into the correct offset of this flat buffer.
pub struct GpuVelocityRing {
    /// Single flat buffer: [`n_slots` × N × 3] f64 values.
    pub flat_buf: wgpu::Buffer,
    /// Individual slot buffers (for backward compat with single-origin API).
    pub slots: Vec<wgpu::Buffer>,
    /// Ring buffer capacity (number of velocity snapshots).
    pub n_slots: usize,
    /// Current write position (wraps modulo n_slots).
    pub write_idx: usize,
    /// Total snapshots stored (may exceed n_slots for overwrite mode).
    pub total_stored: usize,
    /// Number of particles.
    pub n_particles: usize,
    /// Stride per snapshot in f64 elements: N × 3.
    pub stride: usize,
}

impl GpuVelocityRing {
    /// Create a ring buffer for `n_slots` velocity snapshots of `n_particles`.
    #[must_use]
    pub fn new(gpu: &GpuF64, n_particles: usize, n_slots: usize) -> Self {
        let stride = n_particles * 3;
        let total_f64 = n_slots * stride;

        let flat_buf = gpu.create_f64_output_buffer(total_f64, "vel_ring_flat");

        let slots: Vec<wgpu::Buffer> = (0..n_slots)
            .map(|i| gpu.create_f64_output_buffer(stride, &format!("vel_ring_{i}")))
            .collect();

        Self {
            flat_buf,
            slots,
            n_slots,
            write_idx: 0,
            total_stored: 0,
            n_particles,
            stride,
        }
    }

    /// Store current velocity buffer into the ring via GPU→GPU copy.
    ///
    /// Copies into both the flat buffer (for batched VACF) and the slot
    /// buffer (for backward compat).
    pub fn store_snapshot(&mut self, gpu: &GpuF64, vel_buf: &wgpu::Buffer) {
        let byte_size = (self.stride * std::mem::size_of::<f64>()) as u64;
        let flat_offset = (self.write_idx * self.stride * std::mem::size_of::<f64>()) as u64;

        let mut encoder = gpu.begin_encoder("vel_ring_copy");
        encoder.copy_buffer_to_buffer(vel_buf, 0, &self.flat_buf, flat_offset, byte_size);
        encoder.copy_buffer_to_buffer(vel_buf, 0, &self.slots[self.write_idx], 0, byte_size);
        gpu.submit_encoder(encoder);

        self.write_idx = (self.write_idx + 1) % self.n_slots;
        self.total_stored += 1;
    }

    /// Store via encoder (for embedding in the MD production encoder).
    ///
    /// Caller must submit the encoder.
    pub fn store_snapshot_in_encoder(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        vel_buf: &wgpu::Buffer,
    ) {
        let byte_size = (self.stride * std::mem::size_of::<f64>()) as u64;
        let flat_offset = (self.write_idx * self.stride * std::mem::size_of::<f64>()) as u64;

        encoder.copy_buffer_to_buffer(vel_buf, 0, &self.flat_buf, flat_offset, byte_size);

        self.write_idx = (self.write_idx + 1) % self.n_slots;
        self.total_stored += 1;
    }

    /// Number of snapshots available for VACF computation.
    #[must_use]
    pub fn available(&self) -> usize {
        self.total_stored.min(self.n_slots)
    }

    /// Get slot buffer for a logical snapshot index.
    #[must_use]
    pub fn get_snapshot(&self, logical_idx: usize) -> &wgpu::Buffer {
        let available = self.available();
        assert!(logical_idx < available, "snapshot index out of range");
        let physical_idx = if self.total_stored <= self.n_slots {
            logical_idx
        } else {
            (self.write_idx + logical_idx) % self.n_slots
        };
        &self.slots[physical_idx]
    }
}

/// GPU VACF result: C(lag) at discrete lag times + D*.
#[derive(Clone, Debug)]
pub struct GpuVacf {
    /// Lag times (reduced units, ω_p⁻¹).
    pub t_values: Vec<f64>,
    /// Velocity autocorrelation C(t) / C(0).
    pub c_values: Vec<f64>,
    /// Self-diffusion coefficient D* from Green-Kubo integral.
    pub diffusion_coeff: f64,
}

/// Compute VACF entirely on GPU using the batched shader.
///
/// For each lag: ONE dispatch of `vacf_batch_f64.wgsl` (iterates over all
/// time origins inside the shader) + ONE `ReduceScalarPipeline::sum_f64`.
/// Total round trips: 2 × `n_lag` instead of 2 × `n_frames` × `n_lag`.
///
/// # Errors
///
/// Returns [`barracuda::error::BarracudaError`] if the GPU `ReduceScalarPipeline`
/// cannot be created or a reduction fails.
pub fn compute_vacf_gpu(
    gpu: &GpuF64,
    ring: &GpuVelocityRing,
    dt_dump: f64,
    max_lag: usize,
) -> Result<GpuVacf, barracuda::error::BarracudaError> {
    let n_frames = ring.available();
    let n_lag = max_lag.min(n_frames);
    let n = ring.n_particles;

    let pipeline = gpu.create_pipeline_f64(WGSL_VACF_BATCH_F64, "vacf_batch");
    let out_buf = gpu.create_f64_output_buffer(n, "vacf_batch_out");

    let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n)?;

    let workgroups = (n as u32).div_ceil(WORKGROUP_SIZE);

    let mut c_values = vec![0.0f64; n_lag];

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BatchParams {
        n: u32,
        n_frames: u32,
        lag: u32,
        stride: u32,
    }

    for (lag, c_val) in c_values.iter_mut().enumerate() {
        let n_origins = n_frames - lag;

        let params = BatchParams {
            n: n as u32,
            n_frames: n_frames as u32,
            lag: lag as u32,
            stride: ring.stride as u32,
        };
        let params_buf =
            gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "vacf_batch_params");

        let bg = gpu.create_bind_group(&pipeline, &[&ring.flat_buf, &out_buf, &params_buf]);
        gpu.dispatch(&pipeline, &bg, workgroups);

        let total = reducer.sum_f64(&out_buf)?;
        *c_val = total / (n as f64 * n_origins as f64);
    }

    let c0 = c_values[0].max(DIVISION_GUARD);
    let c_normalized: Vec<f64> = c_values.iter().map(|&c| c / c0).collect();

    let mut integral = 0.0;
    let mut d_star_max = 0.0;
    let mut plateau_count = 0;
    let plateau_window = (PLATEAU_DETECTION_TIME / dt_dump).ceil() as usize;

    for i in 1..n_lag {
        integral += 0.5 * dt_dump * (c_values[i - 1] + c_values[i]);
        let d_star_running = integral / 3.0;
        if d_star_running > d_star_max {
            d_star_max = d_star_running;
            plateau_count = 0;
        } else {
            plateau_count += 1;
            if plateau_count > plateau_window {
                break;
            }
        }
    }

    let t_values: Vec<f64> = (0..n_lag).map(|i| i as f64 * dt_dump).collect();

    Ok(GpuVacf {
        t_values,
        c_values: c_normalized,
        diffusion_coeff: d_star_max,
    })
}

/// Compute `σ_xy` on GPU for a single snapshot (positions + velocities on GPU).
///
/// # Errors
///
/// Returns [`barracuda::error::BarracudaError`] if GPU pipeline creation or
/// reduction fails.
pub fn compute_stress_xy_gpu(
    gpu: &GpuF64,
    pos_buf: &wgpu::Buffer,
    vel_buf: &wgpu::Buffer,
    n: usize,
    kappa: f64,
    mass: f64,
    box_side: f64,
) -> Result<f64, barracuda::error::BarracudaError> {
    let pipeline = gpu.create_pipeline_f64(WGSL_STRESS_VIRIAL_F64, "stress_virial");
    let out_buf = gpu.create_f64_output_buffer(n, "stress_out");

    let cutoff = box_side / 2.0;
    let params_data: [f64; 8] = [
        n as f64,
        kappa,
        mass,
        cutoff * cutoff,
        box_side,
        box_side,
        box_side,
        0.0,
    ];
    let params_buf = gpu.create_f64_buffer(&params_data, "stress_params");

    let bg = gpu.create_bind_group(&pipeline, &[pos_buf, vel_buf, &out_buf, &params_buf]);
    let workgroups = (n as u32).div_ceil(WORKGROUP_SIZE);
    gpu.dispatch(&pipeline, &bg, workgroups);

    let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n)?;
    reducer.sum_f64(&out_buf)
}
