// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-only MD transport simulation — unidirectional streaming.
//!
//! Modified simulation loop that stores velocity snapshots in a GPU ring
//! buffer instead of reading them back to CPU. VACF and D* computed
//! entirely on GPU. Only scalar transport coefficients cross `PCIe`.
//!
//! **Eliminated**: `O(n_snapshots` × N × 3 × 8) bytes of position/velocity
//! readback per case. For N=500, 4000 snapshots: 48 MB → 0.
//!
//! **Retained**: 8-byte KE/PE scalars for energy bookkeeping.

use crate::gpu::GpuF64;
use crate::md::config::MdConfig;
use crate::md::observables::transport_gpu::{compute_vacf_gpu, GpuVacf, GpuVelocityRing};
use crate::md::shaders;
use crate::md::simulation::{init_fcc_lattice, init_velocities, EnergyRecord};

use barracuda::pipeline::ReduceScalarPipeline;

use std::time::Instant;

/// Compute shader workgroup size (must match WGSL `@workgroup_size(N)`).
const WORKGROUP_SIZE: usize = 64;

/// Berendsen thermostat rescaling interval during equilibration (steps).
const THERMOSTAT_INTERVAL: usize = 10;

/// Default RNG seed for velocity initialization (reproducible baselines).
const DEFAULT_VELOCITY_SEED: u64 = 42;

/// Console progress reporting interval during production (steps).
const PROGRESS_REPORT_INTERVAL: usize = 5000;

/// Result of a GPU-only transport simulation.
pub struct GpuTransportResult {
    pub energy_history: Vec<EnergyRecord>,
    pub gpu_vacf: GpuVacf,
    pub d_star: f64,
    pub wall_time_s: f64,
    pub sim_time_s: f64,
    pub vacf_time_s: f64,
    pub steps_per_sec: f64,
    pub n_snapshots: usize,
}

/// Run MD + transport entirely on GPU.
///
/// # Errors
///
/// Returns device/shader errors if GPU initialization fails.
pub async fn run_transport_gpu(
    config: &MdConfig,
) -> Result<GpuTransportResult, crate::error::HotSpringError> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── GPU-Only Transport: {n} particles ──");
    println!(
        "    κ={}, Γ={}, T*={temperature:.6}",
        config.kappa, config.gamma
    );

    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let velocities = init_velocities(n, temperature, mass, DEFAULT_VELOCITY_SEED);

    let gpu = GpuF64::new().await?;
    if !gpu.has_f64 {
        return Err(crate::error::HotSpringError::NoShaderF64);
    }

    let force_pipeline = gpu.create_pipeline_f64(shaders::SHADER_YUKAWA_FORCE, "yukawa_force_f64");
    let kick_drift_pipeline =
        gpu.create_pipeline(shaders::SHADER_VV_KICK_DRIFT, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(shaders::SHADER_VV_HALF_KICK, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(shaders::SHADER_BERENDSEN, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(shaders::SHADER_KINETIC_ENERGY, "kinetic_energy_f64");

    let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n)?;

    let pos_buf = gpu.create_f64_output_buffer(n * 3, "positions");
    let vel_buf = gpu.create_f64_output_buffer(n * 3, "velocities");
    let force_buf = gpu.create_f64_output_buffer(n * 3, "forces");
    let pe_buf = gpu.create_f64_output_buffer(n, "pe_per_particle");
    let ke_buf = gpu.create_f64_output_buffer(n, "ke_per_particle");

    gpu.upload_f64(&pos_buf, &positions);
    gpu.upload_f64(&vel_buf, &velocities);

    let force_params = vec![
        n as f64,
        config.kappa,
        prefactor,
        config.rc * config.rc,
        box_side,
        box_side,
        box_side,
        0.0,
    ];
    let force_params_buf = gpu.create_f64_buffer(&force_params, "force_params");
    let vv_params = vec![
        n as f64, config.dt, mass, 0.0, box_side, box_side, box_side, 0.0,
    ];
    let vv_params_buf = gpu.create_f64_buffer(&vv_params, "vv_params");
    let hk_params = vec![n as f64, config.dt, mass, 0.0];
    let hk_params_buf = gpu.create_f64_buffer(&hk_params, "hk_params");
    let ke_params = vec![n as f64, mass, 0.0, 0.0];
    let ke_params_buf = gpu.create_f64_buffer(&ke_params, "ke_params");

    let workgroups = n.div_ceil(WORKGROUP_SIZE) as u32;

    let force_bg = gpu.create_bind_group(
        &force_pipeline,
        &[&pos_buf, &force_buf, &pe_buf, &force_params_buf],
    );
    let kick_drift_bg = gpu.create_bind_group(
        &kick_drift_pipeline,
        &[&pos_buf, &vel_buf, &force_buf, &vv_params_buf],
    );
    let half_kick_bg =
        gpu.create_bind_group(&half_kick_pipeline, &[&vel_buf, &force_buf, &hk_params_buf]);
    let ke_bg = gpu.create_bind_group(&ke_pipeline, &[&vel_buf, &ke_buf, &ke_params_buf]);

    gpu.dispatch(&force_pipeline, &force_bg, workgroups);

    // ── Equilibration ──
    println!("    Equilibrating ({} steps)...", config.equil_steps);
    let t_equil = Instant::now();
    let mut step = 0;
    while step < config.equil_steps {
        let batch_size = THERMOSTAT_INTERVAL.min(config.equil_steps - step);
        let mut encoder = gpu.begin_encoder("equil");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("equil_stream"),
                timestamp_writes: None,
            });
            for _ in 0..batch_size {
                pass.set_pipeline(&kick_drift_pipeline);
                pass.set_bind_group(0, &kick_drift_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&force_pipeline);
                pass.set_bind_group(0, &force_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&half_kick_pipeline);
                pass.set_bind_group(0, &half_kick_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            pass.set_pipeline(&ke_pipeline);
            pass.set_bind_group(0, &ke_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        gpu.submit_encoder(encoder);

        let total_ke = reducer.sum_f64(&ke_buf)?;
        let t_current = 2.0 * total_ke / (3.0 * n as f64);
        if t_current > 1e-30 {
            let ratio =
                (config.dt / config.berendsen_tau).mul_add(temperature / t_current - 1.0, 1.0);
            let scale = ratio.max(0.0).sqrt();
            let beren_params = vec![n as f64, scale, 0.0, 0.0];
            let beren_params_buf = gpu.create_f64_buffer(&beren_params, "beren_params");
            let beren_bg =
                gpu.create_bind_group(&berendsen_pipeline, &[&vel_buf, &beren_params_buf]);
            gpu.dispatch(&berendsen_pipeline, &beren_bg, workgroups);
        }
        step += batch_size;
    }
    println!("    Equilibration: {:.2}s", t_equil.elapsed().as_secs_f64());

    // ── Production with GPU velocity ring ──
    // Instead of reading back velocity snapshots to CPU,
    // copy velocity buffer directly into GPU ring slots.
    println!(
        "    Production ({} steps, GPU-resident snapshots)...",
        config.prod_steps
    );
    let t_prod = Instant::now();

    let n_dumps = config.prod_steps / config.dump_step;
    let snap_every = config.vel_snapshot_interval;
    let n_snapshots = n_dumps.div_ceil(snap_every);

    let mut ring = GpuVelocityRing::new(&gpu, n, n_snapshots);
    let mut energy_history = Vec::new();

    for dump_idx in 0..n_dumps {
        let need_snapshot = dump_idx % snap_every == 0;

        let mut encoder = gpu.begin_encoder("md_batch");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("md_stream"),
                timestamp_writes: None,
            });
            for _ in 0..config.dump_step {
                pass.set_pipeline(&kick_drift_pipeline);
                pass.set_bind_group(0, &kick_drift_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&force_pipeline);
                pass.set_bind_group(0, &force_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&half_kick_pipeline);
                pass.set_bind_group(0, &half_kick_bg, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            pass.set_pipeline(&ke_pipeline);
            pass.set_bind_group(0, &ke_bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // GPU→GPU velocity copy instead of GPU→staging→CPU readback
        if need_snapshot {
            ring.store_snapshot_in_encoder(&mut encoder, &vel_buf);
        }

        gpu.submit_encoder(encoder);

        // Only scalar readback: 16 bytes (KE + PE)
        let total_ke = reducer.sum_f64(&ke_buf)?;
        let total_pe = reducer.sum_f64(&pe_buf)?;
        let t_current = 2.0 * total_ke / (3.0 * n as f64);
        let step_end = (dump_idx + 1) * config.dump_step;

        energy_history.push(EnergyRecord {
            step: step_end - 1,
            ke: total_ke,
            pe: total_pe,
            total: total_ke + total_pe,
            temperature: t_current,
        });

        if step_end % PROGRESS_REPORT_INTERVAL < config.dump_step || step_end >= config.prod_steps {
            println!(
                "    Step {}: T*={:.6}, E={:.4}",
                step_end - 1,
                t_current,
                total_ke + total_pe,
            );
        }
    }
    let sim_time = t_prod.elapsed().as_secs_f64();
    let total_steps = config.equil_steps + config.prod_steps;
    let steps_per_sec = total_steps as f64 / (t_equil.elapsed().as_secs_f64() + sim_time);
    println!("    Production: {sim_time:.2}s ({steps_per_sec:.0} steps/s)");
    println!("    Snapshots stored in GPU ring: {}", ring.available());

    // ── GPU VACF computation ──
    println!("    Computing VACF on GPU...");
    let t_vacf = Instant::now();
    let dt_snap = config.dt * config.dump_step as f64 * config.vel_snapshot_interval as f64;
    let max_lag = (ring.available() / 2).max(10);
    let gpu_vacf = compute_vacf_gpu(&gpu, &ring, dt_snap, max_lag)?;
    let vacf_time = t_vacf.elapsed().as_secs_f64();
    println!(
        "    GPU VACF: {vacf_time:.2}s → D* = {:.4e}",
        gpu_vacf.diffusion_coeff
    );

    let wall_time = t_start.elapsed().as_secs_f64();

    Ok(GpuTransportResult {
        energy_history,
        gpu_vacf: gpu_vacf.clone(),
        d_star: gpu_vacf.diffusion_coeff,
        wall_time_s: wall_time,
        sim_time_s: sim_time,
        vacf_time_s: vacf_time,
        steps_per_sec,
        n_snapshots: ring.available(),
    })
}
