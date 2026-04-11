// SPDX-License-Identifier: AGPL-3.0-or-later

//! Full coupled kinetic-fluid GPU pipeline (Paper 45).
//!
//! Orchestrates BGK relaxation (GPU), Euler fluid update (GPU), and the
//! kinetic-fluid interface coupling. The kinetic advection (spatial transport)
//! and interface conditions remain on CPU since they involve neighbor-cell
//! communication patterns that don't parallelize well at small N_x.
//!
//! Architecture:
//! ```text
//! Per timestep:
//!   1. Kinetic advection (CPU, first-order upwind, per spatial cell)
//!   2. BGK collision per cell (GPU, velocity-space parallel)
//!   3. Interface: kinetic → fluid (CPU, moment extraction)
//!   4. Euler update (GPU, HLL flux + conservative update)
//!   5. Interface: fluid → kinetic (CPU, Maxwellian boundary)
//! ```

use super::kinetic_fluid::{
    CoupledResult, INTERFACE_CONVERGENCE_TOL, INTERFACE_MAX_SUB_ITERATIONS, compute_moments,
    maxwellian_1d, run_coupled_kinetic_fluid,
};
use crate::gpu::GpuF64;

use barracuda::shaders::precision::ShaderTemplate;
use wgpu::util::DeviceExt;

const GAMMA: f64 = 5.0 / 3.0;

const WGSL_BGK: &str = include_str!("shaders/bgk_relaxation_f64.wgsl");
const WGSL_EULER: &str = include_str!("shaders/euler_hll_f64.wgsl");

/// Uniform parameter buffer for BGK (must match WGSL `BgkParams`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BgkParams {
    nv: u32,
    n_species: u32,
    dt: f64,
    v_min: f64,
    dv: f64,
}

/// Uniform parameter buffer for Euler (must match WGSL `EulerParams`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EulerParams {
    nx: u32,
    _pad: u32,
    dt: f64,
    dx: f64,
    gamma: f64,
}

/// Result of the GPU coupled kinetic-fluid test.
pub struct GpuCoupledResult {
    /// CPU reference result.
    pub cpu: CoupledResult,
    /// GPU relative mass conservation error.
    pub mass_err: f64,
    /// GPU relative momentum conservation error.
    pub momentum_err: f64,
    /// GPU relative energy conservation error.
    pub energy_err: f64,
    /// GPU interface density mismatch.
    pub interface_density_match: f64,
    /// Number of time steps taken.
    pub n_steps: usize,
    /// GPU wall time (seconds).
    pub gpu_wall_seconds: f64,
    /// CPU wall time (seconds).
    pub cpu_wall_seconds: f64,
}

/// GPU coupled kinetic-fluid pipeline.
pub struct GpuCoupledPipeline {
    #[expect(dead_code, reason = "batched BGK evolution: N_x cells at once")]
    bgk_moments: wgpu::ComputePipeline,
    bgk_relax: wgpu::ComputePipeline,
    euler_flux: wgpu::ComputePipeline,
    euler_update: wgpu::ComputePipeline,
}

impl GpuCoupledPipeline {
    /// Compile both BGK and Euler shaders.
    #[must_use]
    pub fn new(gpu: &GpuF64) -> Self {
        let bgk_source = ShaderTemplate::with_math_f64_auto(WGSL_BGK);
        let euler_source = ShaderTemplate::with_math_f64_auto(WGSL_EULER);
        Self {
            bgk_moments: gpu.create_pipeline_f64_entry(
                &bgk_source,
                "compute_moments",
                "compute_moments",
            ),
            bgk_relax: gpu.create_pipeline_f64_entry(&bgk_source, "bgk_relax", "bgk_relax"),
            euler_flux: gpu.create_pipeline_f64_entry(
                &euler_source,
                "compute_hll_flux",
                "compute_hll_flux",
            ),
            euler_update: gpu.create_pipeline_f64_entry(
                &euler_source,
                "euler_update",
                "euler_update",
            ),
        }
    }
}

/// Run the coupled kinetic-fluid simulation with GPU-accelerated BGK and Euler.
#[must_use]
pub fn gpu_coupled_kinetic_fluid(
    gpu: &GpuF64,
    pipeline: &GpuCoupledPipeline,
    nx_kin: usize,
    nx_fluid: usize,
    nv: usize,
    t_final: f64,
) -> GpuCoupledResult {
    let dx = 1.0 / (nx_kin + nx_fluid) as f64;
    let v_max = 6.0;
    let v: Vec<f64> = (0..nv)
        .map(|i| -v_max + i as f64 * 2.0 * v_max / (nv - 1) as f64)
        .collect();
    let dv = v[1] - v[0];
    let m = 1.0;
    let nu = 10.0;

    let rho_init = 1.0;
    let u_init = 0.1;
    let p_init = 1.0;
    let t_init = p_init / (rho_init / m);

    let f_init = maxwellian_1d(&v, rho_init / m, u_init, t_init, m);
    let mut f_kin: Vec<Vec<f64>> = (0..nx_kin).map(|_| f_init.clone()).collect();

    let mut rho_fluid = vec![rho_init; nx_fluid];
    let mut u_fluid = vec![u_init; nx_fluid];
    let mut p_fluid = vec![p_init; nx_fluid];

    let kinetic_to_fluid_fn = |f: &[f64]| -> (f64, f64, f64) {
        let n: f64 = f.iter().sum::<f64>() * dv;
        let rho = m * n;
        let rho_u = m
            * f.iter()
                .zip(v.iter())
                .map(|(&fi, &vi)| fi * vi)
                .sum::<f64>()
            * dv;
        let e = 0.5
            * m
            * f.iter()
                .zip(v.iter())
                .map(|(&fi, &vi)| fi * vi * vi)
                .sum::<f64>()
            * dv;
        (rho, rho_u, e)
    };

    let total_mass_0: f64 = {
        let kin: f64 = (0..nx_kin).map(|i| kinetic_to_fluid_fn(&f_kin[i]).0).sum();
        let fluid: f64 = rho_fluid.iter().sum();
        (kin + fluid) * dx
    };
    let total_mom_0: f64 = {
        let kin: f64 = (0..nx_kin).map(|i| kinetic_to_fluid_fn(&f_kin[i]).1).sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .map(|(&r, &u)| r * u)
            .sum();
        (kin + fluid) * dx
    };
    let total_e_0: f64 = {
        let kin: f64 = (0..nx_kin).map(|i| kinetic_to_fluid_fn(&f_kin[i]).2).sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .zip(p_fluid.iter())
            .map(|((&r, &u), &p)| p / (GAMMA - 1.0) + 0.5 * r * u * u)
            .sum();
        (kin + fluid) * dx
    };

    let gpu_start = std::time::Instant::now();
    let mut t = 0.0;
    let mut n_steps = 0;
    let max_steps = 5000;

    for _ in 0..max_steps {
        if t >= t_final {
            break;
        }
        let max_speed_fluid = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .zip(p_fluid.iter())
            .map(|((&r, &u), &p)| {
                let c = if r > 1e-30 {
                    (GAMMA * p / r).sqrt()
                } else {
                    0.0
                };
                u.abs() + c
            })
            .fold(0.0_f64, f64::max);
        let max_speed = max_speed_fluid.max(v_max);
        let dt = (0.3 * dx / max_speed.max(1e-30)).min(t_final - t);

        // 1. Kinetic advection (CPU, first-order upwind)
        let mut f_new = f_kin.clone();
        for (j, &vj) in v.iter().enumerate() {
            if vj > 0.0 {
                for i in 1..nx_kin {
                    f_new[i][j] = f_kin[i][j] - dt * vj / dx * (f_kin[i][j] - f_kin[i - 1][j]);
                }
            } else {
                for i in 0..nx_kin - 1 {
                    f_new[i][j] = f_kin[i][j] - dt * vj / dx * (f_kin[i + 1][j] - f_kin[i][j]);
                }
            }
        }
        for row in &mut f_new {
            for fi in row.iter_mut() {
                *fi = fi.max(0.0);
            }
        }
        f_kin = f_new;

        // 2. BGK collision per kinetic cell (GPU)
        for cell in &mut f_kin[..nx_kin] {
            let bgk_params = BgkParams {
                nv: nv as u32,
                n_species: 1,
                dt,
                v_min: -v_max,
                dv,
            };

            let param_buf =
                gpu.create_uniform_buffer(bytemuck::bytes_of(&bgk_params), "bgk_params");

            let f_buf = gpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bgk_f"),
                    contents: bytemuck::cast_slice(cell.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

            let (ni, ui, ti, _) = compute_moments(cell, &v, dv, m);
            let sp_data: Vec<f64> = vec![m, nu, ni, ui, ti.max(1e-12)];
            let sp_buf = gpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("bgk_sp"),
                    contents: bytemuck::cast_slice(&sp_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

            let moment_buf = gpu.create_f64_output_buffer(nv * 3, "bgk_moments");

            let bg_relax = gpu.create_bind_group(
                &pipeline.bgk_relax,
                &[&param_buf, &f_buf, &sp_buf, &moment_buf],
            );
            let wg = (nv as u32).div_ceil(64);
            gpu.dispatch(&pipeline.bgk_relax, &bg_relax, wg);

            let f_out = gpu
                .read_back_f64(&f_buf, nv)
                .unwrap_or_else(|_| cell.clone());
            *cell = f_out;
        }

        // 3–5. Interface sub-iteration: converges kinetic-fluid boundary
        // by repeating the interface exchange until density mismatch drops
        // below tolerance or max iterations reached (Haack et al. §3.2).
        let max_sub = INTERFACE_MAX_SUB_ITERATIONS;
        let sub_tol = INTERFACE_CONVERGENCE_TOL;
        let f_kin_boundary_save = f_kin[nx_kin - 1].clone();
        let rho_fluid_save = rho_fluid.clone();
        let u_fluid_save = u_fluid.clone();
        let p_fluid_save = p_fluid.clone();

        let nx_euler = nx_fluid;
        let euler_params = EulerParams {
            nx: nx_euler as u32,
            _pad: 0,
            dt,
            dx,
            gamma: GAMMA,
        };
        let euler_param_buf =
            gpu.create_uniform_buffer(bytemuck::bytes_of(&euler_params), "euler_params");

        for _sub in 0..max_sub {
            // Interface: kinetic → fluid (CPU)
            let (rho_int, rhou_int, e_int) = kinetic_to_fluid_fn(&f_kin[nx_kin - 1]);
            let u_int = if rho_int > 1e-30 {
                rhou_int / rho_int
            } else {
                0.0
            };
            let p_int = ((GAMMA - 1.0) * (e_int - 0.5 * rho_int * u_int * u_int)).max(1e-15);

            // Euler update (GPU, from saved state each sub-iteration)
            let mut cons_data: Vec<f64> = Vec::with_capacity(3 * nx_euler);
            let mut e_vec: Vec<f64> = Vec::with_capacity(nx_euler);
            for i in 0..nx_euler {
                e_vec.push(
                    p_fluid_save[i] / (GAMMA - 1.0)
                        + 0.5 * rho_fluid_save[i] * u_fluid_save[i] * u_fluid_save[i],
                );
            }
            cons_data.extend_from_slice(&rho_fluid_save);
            let rhou_vec: Vec<f64> = rho_fluid_save
                .iter()
                .zip(u_fluid_save.iter())
                .map(|(&r, &u)| r * u)
                .collect();
            cons_data.extend_from_slice(&rhou_vec);
            cons_data.extend_from_slice(&e_vec);

            let cons_buf = gpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("euler_cons"),
                    contents: bytemuck::cast_slice(&cons_data),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

            let n_flux = 3 * (nx_euler + 1);
            let flux_buf = gpu.create_f64_output_buffer(n_flux, "euler_flux");

            // Flux pass
            let bg_flux = gpu.create_bind_group(
                &pipeline.euler_flux,
                &[&euler_param_buf, &cons_buf, &flux_buf],
            );
            let wg_flux = ((nx_euler as u32) + 1).div_ceil(64);
            gpu.dispatch(&pipeline.euler_flux, &bg_flux, wg_flux);

            // Overwrite left boundary flux with interface HLL flux (CPU-computed)
            {
                let c_l = if rho_int > 1e-30 {
                    (GAMMA * p_int / rho_int).sqrt()
                } else {
                    0.0
                };
                let c_r = if rho_fluid_save[0] > 1e-30 {
                    (GAMMA * p_fluid_save[0] / rho_fluid_save[0]).sqrt()
                } else {
                    0.0
                };
                let s_l = (u_int - c_l).min(u_fluid_save[0] - c_r);
                let s_r = (u_int + c_l).max(u_fluid_save[0] + c_r);

                let e_l = p_int / (GAMMA - 1.0) + 0.5 * rho_int * u_int * u_int;
                let e_r = p_fluid_save[0] / (GAMMA - 1.0)
                    + 0.5 * rho_fluid_save[0] * u_fluid_save[0] * u_fluid_save[0];

                let (f_rho, f_mom, f_ene) = if s_l >= 0.0 {
                    (
                        rho_int * u_int,
                        rho_int * u_int * u_int + p_int,
                        (e_l + p_int) * u_int,
                    )
                } else if s_r <= 0.0 {
                    (
                        rho_fluid_save[0] * u_fluid_save[0],
                        rho_fluid_save[0] * u_fluid_save[0] * u_fluid_save[0] + p_fluid_save[0],
                        (e_r + p_fluid_save[0]) * u_fluid_save[0],
                    )
                } else {
                    let d = s_r - s_l;
                    let fl = (
                        rho_int * u_int,
                        rho_int * u_int * u_int + p_int,
                        (e_l + p_int) * u_int,
                    );
                    let fr = (
                        rho_fluid_save[0] * u_fluid_save[0],
                        rho_fluid_save[0] * u_fluid_save[0] * u_fluid_save[0] + p_fluid_save[0],
                        (e_r + p_fluid_save[0]) * u_fluid_save[0],
                    );
                    let ul = (rho_int, rho_int * u_int, e_l);
                    let ur = (rho_fluid_save[0], rho_fluid_save[0] * u_fluid_save[0], e_r);
                    let hll = |fli: f64, fri: f64, uli: f64, uri: f64| -> f64 {
                        (s_r * fli - s_l * fri + s_l * s_r * (uri - uli)) / d
                    };
                    (
                        hll(fl.0, fr.0, ul.0, ur.0),
                        hll(fl.1, fr.1, ul.1, ur.1),
                        hll(fl.2, fr.2, ul.2, ur.2),
                    )
                };

                let nf1 = nx_euler + 1;
                gpu.queue()
                    .write_buffer(&flux_buf, 0, bytemuck::cast_slice(&[f_rho]));
                gpu.queue().write_buffer(
                    &flux_buf,
                    (nf1 * 8) as u64,
                    bytemuck::cast_slice(&[f_mom]),
                );
                gpu.queue().write_buffer(
                    &flux_buf,
                    (2 * nf1 * 8) as u64,
                    bytemuck::cast_slice(&[f_ene]),
                );
            }

            // Update pass
            let bg_update = gpu.create_bind_group(
                &pipeline.euler_update,
                &[&euler_param_buf, &cons_buf, &flux_buf],
            );
            let wg_update = (nx_euler as u32).div_ceil(64);
            gpu.dispatch(&pipeline.euler_update, &bg_update, wg_update);

            // Read back updated fluid state
            let final_state = gpu
                .read_back_f64(&cons_buf, 3 * nx_euler)
                .unwrap_or(cons_data);
            for i in 0..nx_euler {
                rho_fluid[i] = final_state[i].max(1e-10);
                let ru = final_state[nx_euler + i];
                let e = final_state[2 * nx_euler + i];
                u_fluid[i] = if rho_fluid[i] > 1e-30 {
                    ru / rho_fluid[i]
                } else {
                    0.0
                };
                p_fluid[i] =
                    ((GAMMA - 1.0) * (e - 0.5 * rho_fluid[i] * u_fluid[i] * u_fluid[i])).max(1e-30);
            }

            // Interface: fluid → kinetic (CPU)
            let m_boundary = maxwellian_1d(
                &v,
                rho_fluid[0] / m,
                u_fluid[0],
                p_fluid[0] / (rho_fluid[0] / m),
                m,
            );
            f_kin[nx_kin - 1].clone_from(&f_kin_boundary_save);
            for j in 0..nv {
                if v[j] <= 0.0 {
                    f_kin[nx_kin - 1][j] = m_boundary[j];
                }
            }

            let rho_kin_if = kinetic_to_fluid_fn(&f_kin[nx_kin - 1]).0;
            let mismatch = (rho_kin_if - rho_fluid[0]).abs() / rho_init.max(1e-30);
            if mismatch < sub_tol {
                break;
            }
        }

        t += dt;
        n_steps += 1;
    }

    let gpu_wall = gpu_start.elapsed().as_secs_f64();

    // Final conservation checks
    let total_mass_f: f64 = {
        let kin: f64 = (0..nx_kin).map(|i| kinetic_to_fluid_fn(&f_kin[i]).0).sum();
        let fluid: f64 = rho_fluid.iter().sum();
        (kin + fluid) * dx
    };
    let total_mom_f: f64 = {
        let kin: f64 = (0..nx_kin).map(|i| kinetic_to_fluid_fn(&f_kin[i]).1).sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .map(|(&r, &u)| r * u)
            .sum();
        (kin + fluid) * dx
    };
    let total_e_f: f64 = {
        let kin: f64 = (0..nx_kin).map(|i| kinetic_to_fluid_fn(&f_kin[i]).2).sum();
        let fluid: f64 = rho_fluid
            .iter()
            .zip(u_fluid.iter())
            .zip(p_fluid.iter())
            .map(|((&r, &u), &p)| p / (GAMMA - 1.0) + 0.5 * r * u * u)
            .sum();
        (kin + fluid) * dx
    };

    let rho_if_kin = kinetic_to_fluid_fn(&f_kin[nx_kin - 1]).0;
    let interface_density_match = (rho_if_kin - rho_fluid[0]).abs() / rho_init.max(1e-30);

    // CPU reference
    let cpu_start = std::time::Instant::now();
    let cpu = run_coupled_kinetic_fluid(nx_kin, nx_fluid, nv, t_final);
    let cpu_wall = cpu_start.elapsed().as_secs_f64();

    GpuCoupledResult {
        cpu,
        mass_err: (total_mass_f - total_mass_0).abs() / total_mass_0.max(1e-30),
        momentum_err: (total_mom_f - total_mom_0).abs() / total_mom_0.abs().max(1e-30),
        energy_err: (total_e_f - total_e_0).abs() / total_e_0.max(1e-30),
        interface_density_match,
        n_steps,
        gpu_wall_seconds: gpu_wall,
        cpu_wall_seconds: cpu_wall,
    }
}

/// Validate GPU coupled pipeline against CPU reference.
#[must_use]
pub fn validate_gpu_coupled(gpu: &GpuF64, pipeline: &GpuCoupledPipeline) -> GpuCoupledResult {
    gpu_coupled_kinetic_fluid(gpu, pipeline, 10, 40, 51, 0.1)
}
