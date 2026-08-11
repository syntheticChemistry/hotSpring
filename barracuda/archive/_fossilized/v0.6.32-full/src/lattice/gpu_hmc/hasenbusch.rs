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

use super::GpuF64;

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
