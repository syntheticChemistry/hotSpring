// SPDX-License-Identifier: AGPL-3.0-only

//! GPU MD Simulation Engine
//!
//! Runs a full Yukawa OCP molecular dynamics simulation on GPU using
//! f64 WGSL shaders. The MD loop is:
//!
//!   1. Half-kick + drift + PBC wrap  (fused kernel)
//!   2. Force computation              (Yukawa all-pairs kernel)
//!   3. Second half-kick               (kernel)
//!   4. (optional) Berendsen thermostat
//!
//! All particle data lives on GPU. CPU reads back energy/positions
//! only at dump intervals for observable computation.

mod init;
mod types;
mod verlet;

use barracuda::device::driver_profile::Fp64Strategy;
use barracuda::pipeline::ReduceScalarPipeline;

use crate::gpu::GpuF64;
pub use crate::md::celllist::{run_simulation_celllist, CellList};
use crate::md::config::MdConfig;
use crate::md::shaders;
use crate::tolerances::{DEFAULT_VELOCITY_SEED, MD_TEMPERATURE_FLOOR, THERMOSTAT_INTERVAL};

use std::time::Instant;

pub use init::{init_fcc_lattice, init_velocities};
pub use types::{BrainSummary, EnergyRecord, MdSimulation};
pub use verlet::{run_simulation_verlet, run_simulation_verlet_with_brain};

/// Run the full GPU MD simulation (all-pairs).
///
/// # Errors
///
/// Returns [`crate::error::HotSpringError::NoAdapter`] if no GPU adapter is found.
/// Returns [`crate::error::HotSpringError::NoShaderF64`] if the GPU lacks `SHADER_F64`.
/// Returns [`crate::error::HotSpringError::DeviceCreation`] or
/// [`crate::error::HotSpringError::Barracuda`] if GPU initialization or pipeline setup fails.
pub async fn run_simulation(
    config: &MdConfig,
) -> Result<MdSimulation, crate::error::HotSpringError> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── Initializing {n} particles ──");
    println!("    Box side: {box_side:.4} a_ws");
    println!(
        "    κ = {}, Γ = {}, T* = {:.6}",
        config.kappa, config.gamma, temperature
    );
    println!(
        "    rc = {} a_ws, dt* = {}, m* = {}",
        config.rc, config.dt, mass
    );
    println!("    Force prefactor: {prefactor:.4}");

    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let velocities = init_velocities(n, temperature, mass, DEFAULT_VELOCITY_SEED);

    println!("    Placed {n} particles on FCC lattice");

    let gpu = GpuF64::new().await?;
    if !gpu.has_f64 {
        return Err(crate::error::HotSpringError::NoShaderF64);
    }
    gpu.print_info();

    let strategy = gpu.driver_profile().fp64_strategy();
    let strategy_label = match strategy {
        Fp64Strategy::Native => "native f64",
        Fp64Strategy::Hybrid => "DF64 (FP32 core streaming)",
        Fp64Strategy::Concurrent => "concurrent f64 + DF64",
    };
    println!("  ── Compiling WGSL shaders ({strategy_label}) ──");
    let t_compile = Instant::now();

    let force_pipeline = match strategy {
        Fp64Strategy::Hybrid => {
            gpu.create_pipeline_df64(shaders::SHADER_YUKAWA_FORCE_DF64, "yukawa_force_df64")
        }
        Fp64Strategy::Native | Fp64Strategy::Concurrent => {
            gpu.create_pipeline_f64(shaders::SHADER_YUKAWA_FORCE, "yukawa_force_f64")
        }
    };
    let kick_drift_pipeline =
        gpu.create_pipeline(shaders::SHADER_VV_KICK_DRIFT, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(shaders::SHADER_VV_HALF_KICK, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(shaders::SHADER_BERENDSEN, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(shaders::SHADER_KINETIC_ENERGY, "kinetic_energy_f64");

    println!(
        "    Compiled 5 shaders in {:.1}ms",
        t_compile.elapsed().as_secs_f64() * 1000.0
    );

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

    use crate::tolerances::MD_WORKGROUP_SIZE;
    let workgroups = n.div_ceil(MD_WORKGROUP_SIZE) as u32;

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

    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();
    let thermostat_interval = THERMOSTAT_INTERVAL;
    let mut step = 0;

    while step < config.equil_steps {
        let batch_size = thermostat_interval.min(config.equil_steps - step);

        let mut encoder = gpu.begin_encoder("equil_batch");
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

        if t_current > MD_TEMPERATURE_FLOOR {
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

        if step % 1000 < thermostat_interval || step >= config.equil_steps {
            println!("    Step {step}: T* = {t_current:.6} (target {temperature:.6})");
        }
    }
    let equil_time = t_equil.elapsed().as_secs_f64();
    println!("    Equilibration complete in {equil_time:.2}s");

    println!("  ── Production ({} steps, streamed) ──", config.prod_steps);
    let t_prod = Instant::now();

    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    let pos_staging = gpu.create_staging_buffer(n * 3 * 8, "pos_staging");
    let vel_staging = gpu.create_staging_buffer(n * 3 * 8, "vel_staging");

    let n_dumps = config.prod_steps / config.dump_step;
    let snap_every = config.vel_snapshot_interval;

    for dump_idx in 0..n_dumps {
        let step_start = dump_idx * config.dump_step;
        let step_end = step_start + config.dump_step;
        let need_snapshot = dump_idx % snap_every == 0;

        let mut encoder = gpu.begin_encoder("md_batch");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("md_stream"),
                timestamp_writes: None,
            });

            for _ in step_start..step_end {
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
        if need_snapshot {
            encoder.copy_buffer_to_buffer(&pos_buf, 0, &pos_staging, 0, (n * 3 * 8) as u64);
            encoder.copy_buffer_to_buffer(&vel_buf, 0, &vel_staging, 0, (n * 3 * 8) as u64);
        }
        gpu.submit_encoder(encoder);

        let total_ke = reducer.sum_f64(&ke_buf)?;
        let total_pe = reducer.sum_f64(&pe_buf)?;
        let total_e = total_ke + total_pe;
        let t_current = 2.0 * total_ke / (3.0 * n as f64);

        energy_history.push(EnergyRecord {
            step: step_end - 1,
            ke: total_ke,
            pe: total_pe,
            total: total_e,
            temperature: t_current,
        });

        if need_snapshot {
            positions_snapshots.push(gpu.read_staging_f64(&pos_staging)?);
            velocity_snapshots.push(gpu.read_staging_f64(&vel_staging)?);
        }

        if step_end % 5000 < config.dump_step || step_end >= config.prod_steps {
            println!(
                "    Step {}: T*={:.6}, KE={:.4}, PE={:.4}, E={:.4}",
                step_end - 1,
                t_current,
                total_ke,
                total_pe,
                total_e
            );
        }
    }

    let remainder = config.prod_steps % config.dump_step;
    if remainder > 0 {
        let mut encoder = gpu.begin_encoder("md_tail");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("md_tail"),
                timestamp_writes: None,
            });
            for _ in 0..remainder {
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
        }
        gpu.submit_encoder(encoder);
    }

    let prod_time = t_prod.elapsed().as_secs_f64();
    let total_time = t_start.elapsed().as_secs_f64();
    let total_steps = config.equil_steps + config.prod_steps;
    let steps_per_sec = total_steps as f64 / total_time;

    println!("    Production complete in {prod_time:.2}s");
    println!("    Total: {total_time:.2}s ({steps_per_sec:.1} steps/s)");

    Ok(MdSimulation {
        config: config.clone(),
        energy_history,
        positions_snapshots,
        velocity_snapshots,
        rdf_histogram: Vec::new(),
        wall_time_s: total_time,
        steps_per_sec,
        brain_summary: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::md::config;

    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks_exact(8)
            .map(|chunk| {
                let bytes: [u8; 8] = std::array::from_fn(|i| chunk[i]);
                f64::from_le_bytes(bytes)
            })
            .collect()
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn read_back_f64_byte_conversion_roundtrip() {
        let original: Vec<f64> = vec![0.0, 1.0, -1.0, std::f64::consts::PI];
        let bytes: Vec<u8> = original.iter().flat_map(|v| v.to_le_bytes()).collect();
        let recovered = bytes_to_f64(&bytes);
        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn init_fcc_lattice_structure() {
        let (positions, n) = init_fcc_lattice(108, 10.0);
        assert!(n >= 108);
        assert_eq!(positions.len(), n * 3);
        let box_side = 10.0;
        for i in 0..n {
            assert!(positions[i * 3] >= 0.0 && positions[i * 3] <= box_side);
            assert!(positions[i * 3 + 1] >= 0.0 && positions[i * 3 + 1] <= box_side);
            assert!(positions[i * 3 + 2] >= 0.0 && positions[i * 3 + 2] <= box_side);
        }
    }

    #[test]
    fn init_fcc_lattice_4_atoms_per_cell() {
        let (_, n) = init_fcc_lattice(4, 2.0);
        assert_eq!(n, 4);
        let (_, n) = init_fcc_lattice(32, 4.0);
        assert_eq!(n, 32);
    }

    #[test]
    fn init_velocities_center_of_mass_zero() {
        let v = init_velocities(100, 1.0, 1.0, 42);
        let mut vx = 0.0;
        let mut vy = 0.0;
        let mut vz = 0.0;
        for i in 0..100 {
            vx += v[i * 3];
            vy += v[i * 3 + 1];
            vz += v[i * 3 + 2];
        }
        assert!(vx.abs() < 1e-10);
        assert!(vy.abs() < 1e-10);
        assert!(vz.abs() < 1e-10);
    }

    #[test]
    fn init_velocities_temperature() {
        let n = 500;
        let t_target = 2.0;
        let mass = 3.0;
        let v = init_velocities(n, t_target, mass, 12345);
        let ke: f64 = v.iter().map(|x| x * x).sum::<f64>() * 0.5 * mass;
        let t_actual = 2.0 * ke / (3.0 * n as f64);
        assert!(
            (t_actual - t_target).abs() < 0.05,
            "T* = {t_actual} (target {t_target})"
        );
    }

    #[test]
    fn cell_list_build() {
        let (positions, n) = init_fcc_lattice(64, 8.0);
        let box_side = 8.0;
        let rc = 2.0;
        let cl = CellList::build(&positions, n, box_side, rc);
        assert!(cl.n_cells_total > 0);
        assert_eq!(cl.sorted_indices.len(), n);
        let total: u32 = cl.cell_count.iter().sum();
        assert_eq!(total, n as u32);
    }

    #[test]
    fn energy_record_total() {
        let rec = EnergyRecord {
            step: 0,
            ke: 100.0,
            pe: -150.0,
            total: -50.0,
            temperature: 1.0,
        };
        assert!((rec.ke + rec.pe - rec.total).abs() < 1e-10);
    }

    #[test]
    fn config_validation_box_side() {
        let config = config::quick_test_case(256);
        let l = config.box_side();
        let expected = (4.0 * std::f64::consts::PI * 256.0 / 3.0).cbrt();
        assert!((l - expected).abs() < 1e-10);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn run_simulation_gpu() {
        let config = config::quick_test_case(64);
        assert!(config.n_particles > 0, "config should have particles");
        assert!(config.dump_step > 0, "config should have dump interval");
        assert!(config.box_side() > 0.0, "box side must be positive");
    }
}
