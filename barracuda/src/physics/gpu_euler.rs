// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated 1D Euler fluid solver with HLL Riemann solver (Paper 45).
//!
//! Promotes the spatial mesh Euler update from `kinetic_fluid.rs` to GPU.
//! Two-pass approach:
//! - Pass 1: Compute HLL fluxes at all cell interfaces (nx+1 interfaces).
//! - Pass 2: Apply conservative update to all cells.
//!
//! This becomes meaningful at larger spatial meshes (nx > ~256) where the
//! per-cell parallelism justifies GPU dispatch overhead.

use super::kinetic_fluid::{SodResult, run_sod_shock_tube};
use crate::gpu::GpuF64;

use barracuda::shaders::precision::ShaderTemplate;
use wgpu::util::DeviceExt;

const WGSL_EULER: &str = include_str!("shaders/euler_hll_f64.wgsl");

const GAMMA: f64 = 5.0 / 3.0;

/// Uniform parameter buffer (must match WGSL `EulerParams`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EulerParams {
    nx: u32,
    _pad: u32,
    dt: f64,
    dx: f64,
    gamma: f64,
}

/// GPU Euler pipeline with flux and update entry points.
pub struct GpuEulerPipeline {
    flux_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
}

impl GpuEulerPipeline {
    /// Compile Euler HLL shaders.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        let source = ShaderTemplate::with_math_f64_auto(WGSL_EULER);
        Self {
            flux_pipeline: gpu.create_pipeline_f64_entry(
                &source,
                "compute_hll_flux",
                "compute_hll_flux",
            ),
            update_pipeline: gpu.create_pipeline_f64_entry(&source, "euler_update", "euler_update"),
        }
    }
}

/// Result of the GPU Sod shock tube.
pub struct GpuSodResult {
    /// CPU-computed reference result.
    pub cpu: SodResult,
    /// GPU density field at final time.
    pub rho: Vec<f64>,
    /// GPU velocity field at final time.
    pub u: Vec<f64>,
    /// GPU pressure field at final time.
    pub p: Vec<f64>,
    /// Relative mass conservation error on GPU.
    pub mass_err: f64,
    /// Relative energy conservation error on GPU.
    pub energy_err: f64,
    /// GPU wall time (seconds).
    pub gpu_wall_seconds: f64,
    /// CPU wall time (seconds).
    pub cpu_wall_seconds: f64,
}

/// Run the Sod shock tube on GPU and compare with CPU.
#[must_use]
pub fn gpu_sod_shock_tube(
    gpu: &GpuF64,
    pipeline: &GpuEulerPipeline,
    nx: usize,
    t_final: f64,
) -> GpuSodResult {
    let dx = 1.0 / nx as f64;
    let x: Vec<f64> = (0..nx).map(|i| (i as f64 + 0.5) * dx).collect();

    // Initial conditions: Sod shock tube
    let rho: Vec<f64> = x
        .iter()
        .map(|&xi| if xi < 0.5 { 1.0 } else { 0.125 })
        .collect();
    let u_vel = vec![0.0f64; nx];
    let p: Vec<f64> = x
        .iter()
        .map(|&xi| if xi < 0.5 { 1.0 } else { 0.1 })
        .collect();

    // Convert to conserved variables
    let mut cons_data: Vec<f64> = Vec::with_capacity(3 * nx);
    let mut e_vec: Vec<f64> = Vec::with_capacity(nx);
    for i in 0..nx {
        e_vec.push(p[i] / (GAMMA - 1.0) + 0.5 * rho[i] * u_vel[i] * u_vel[i]);
    }
    // Layout: [rho_0..nx, rhou_0..nx, E_0..nx]
    cons_data.extend_from_slice(&rho);
    let rhou: Vec<f64> = rho.iter().zip(u_vel.iter()).map(|(&r, &u)| r * u).collect();
    cons_data.extend_from_slice(&rhou);
    cons_data.extend_from_slice(&e_vec);

    let total_mass_0: f64 = rho.iter().sum::<f64>() * dx;
    let total_e_0: f64 = e_vec.iter().sum::<f64>() * dx;

    let n_flux = 3 * (nx + 1);

    let gpu_start = std::time::Instant::now();

    let cons_buf = gpu
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("euler_cons"),
            contents: bytemuck::cast_slice(&cons_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

    let flux_buf = gpu.create_f64_output_buffer(n_flux, "euler_flux");

    // Time stepping with CFL
    let mut t = 0.0;
    let max_steps = (t_final / 1e-6).ceil() as usize + 1;
    for _ in 0..max_steps {
        if t >= t_final {
            break;
        }
        // Read current state for CFL
        let state = gpu
            .read_back_f64(&cons_buf, 3 * nx)
            .unwrap_or_else(|_| cons_data.clone());

        let max_speed = (0..nx)
            .map(|i| {
                let r = state[i].max(1e-30);
                let ru = state[nx + i];
                let e = state[2 * nx + i];
                let ui = ru / r;
                let pi = ((GAMMA - 1.0) * (e - 0.5 * r * ui * ui)).max(1e-30);
                let c = (GAMMA * pi / r).sqrt();
                ui.abs() + c
            })
            .fold(0.0_f64, f64::max);

        let dt = (0.4 * dx / max_speed.max(1e-30)).min(t_final - t);

        let params = EulerParams {
            nx: nx as u32,
            _pad: 0,
            dt,
            dx,
            gamma: GAMMA,
        };
        let param_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "euler_params");

        // Pass 1: HLL flux computation (nx+1 threads)
        let bg_flux =
            gpu.create_bind_group(&pipeline.flux_pipeline, &[&param_buf, &cons_buf, &flux_buf]);
        let wg_flux = ((nx as u32) + 1).div_ceil(64);
        gpu.dispatch(&pipeline.flux_pipeline, &bg_flux, wg_flux);

        // Pass 2: Conservative update (nx threads)
        let bg_update = gpu.create_bind_group(
            &pipeline.update_pipeline,
            &[&param_buf, &cons_buf, &flux_buf],
        );
        let wg_update = (nx as u32).div_ceil(64);
        gpu.dispatch(&pipeline.update_pipeline, &bg_update, wg_update);

        t += dt;
    }

    let gpu_wall = gpu_start.elapsed().as_secs_f64();

    // Read back final state
    let final_state = gpu
        .read_back_f64(&cons_buf, 3 * nx)
        .unwrap_or_else(|_| vec![0.0; 3 * nx]);

    let rho_final: Vec<f64> = final_state[..nx].to_vec();
    let rhou_final: Vec<f64> = final_state[nx..2 * nx].to_vec();
    let e_final: Vec<f64> = final_state[2 * nx..].to_vec();

    let u_final: Vec<f64> = rho_final
        .iter()
        .zip(rhou_final.iter())
        .map(|(&r, &ru)| if r.abs() > 1e-30 { ru / r } else { 0.0 })
        .collect();
    let p_final: Vec<f64> = rho_final
        .iter()
        .zip(u_final.iter())
        .zip(e_final.iter())
        .map(|((&r, &u), &e)| ((GAMMA - 1.0) * (e - 0.5 * r * u * u)).max(1e-30))
        .collect();

    let total_mass_f: f64 = rho_final.iter().sum::<f64>() * dx;
    let total_e_f: f64 = e_final.iter().sum::<f64>() * dx;

    let mass_err = (total_mass_f - total_mass_0).abs() / total_mass_0.abs().max(1e-30);
    let energy_err = (total_e_f - total_e_0).abs() / total_e_0.abs().max(1e-30);

    // CPU reference
    let cpu_start = std::time::Instant::now();
    let cpu = run_sod_shock_tube(nx, t_final);
    let cpu_wall = cpu_start.elapsed().as_secs_f64();

    GpuSodResult {
        cpu,
        rho: rho_final,
        u: u_final,
        p: p_final,
        mass_err,
        energy_err,
        gpu_wall_seconds: gpu_wall,
        cpu_wall_seconds: cpu_wall,
    }
}

/// Validate GPU Euler against CPU Sod shock tube.
#[must_use]
pub fn validate_gpu_euler(gpu: &GpuF64, pipeline: &GpuEulerPipeline) -> GpuSodResult {
    gpu_sod_shock_tube(gpu, pipeline, 400, 0.2)
}

#[cfg(test)]
mod tests {
    #[test]
    fn euler_params_layout() {
        assert_eq!(std::mem::size_of::<super::EulerParams>(), 4 + 4 + 8 + 8 + 8);
    }
}
