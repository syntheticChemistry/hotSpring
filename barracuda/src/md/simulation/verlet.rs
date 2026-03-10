// SPDX-License-Identifier: AGPL-3.0-only

//! Verlet neighbor list simulation with optional brain steering.
//!
//! Uses `CellListGpu` to build a compact neighbor list, then iterates only
//! true neighbors within `rc` for force computation. The list includes
//! particles within `rc + skin` and is rebuilt adaptively when the maximum
//! displacement exceeds `skin / 2`.

use barracuda::device::driver_profile::Fp64Strategy;
use barracuda::pipeline::ReduceScalarPipeline;

use crate::gpu::GpuF64;
use crate::md::brain::{MdBrain, MdStepEvent};
use crate::md::config::MdConfig;
use crate::md::neighbor::ForceAlgorithm;
use crate::md::shaders;
use crate::tolerances::{
    DEFAULT_VELOCITY_SEED, MD_TEMPERATURE_FLOOR, THERMOSTAT_INTERVAL, VERLET_MAX_NEIGHBORS,
};

use std::time::Instant;

use super::init::{init_fcc_lattice, init_velocities};
use super::types::{BrainSummary, EnergyRecord, MdSimulation};

/// Run a Verlet neighbor list simulation.
///
/// Uses `CellListGpu` to build a compact neighbor list, then iterates only
/// true neighbors within `rc` for force computation. The list includes
/// particles within `rc + skin` and is rebuilt adaptively when the maximum
/// displacement exceeds `skin / 2`.
pub async fn run_simulation_verlet(
    config: &MdConfig,
    skin: f64,
) -> Result<MdSimulation, crate::error::HotSpringError> {
    run_simulation_verlet_with_brain(config, skin, None).await
}

/// Verlet simulation with optional pre-trained brain state for cross-run persistence.
///
/// Pass `nautilus_json` as `Some(json_str)` to restore the Nautilus shell's
/// cumulative evolution from a previous run.
pub async fn run_simulation_verlet_with_brain(
    config: &MdConfig,
    skin: f64,
    nautilus_json: Option<&str>,
) -> Result<MdSimulation, crate::error::HotSpringError> {
    let n = config.n_particles;
    let box_side = config.box_side();
    let temperature = config.temperature();
    let mass = config.reduced_mass();
    let prefactor = config.force_prefactor();

    println!(
        "    κ = {}, Γ = {}, T* = {:.6}",
        config.kappa, config.gamma, temperature
    );
    println!(
        "    rc = {} a_ws, skin = {:.3} a_ws, dt* = {}, m* = {}",
        config.rc, skin, config.dt, mass
    );

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
        Fp64Strategy::Native | Fp64Strategy::Sovereign => "native f64",
        Fp64Strategy::Hybrid => "DF64 (FP32 core streaming)",
        Fp64Strategy::Concurrent => "concurrent f64 + DF64",
    };
    println!("  ── Compiling WGSL shaders (Verlet, {strategy_label}) ──");
    let t_compile = Instant::now();

    let force_pipeline = match strategy {
        Fp64Strategy::Hybrid => gpu.create_pipeline_df64(
            shaders::SHADER_YUKAWA_FORCE_VERLET_DF64,
            "yukawa_force_verlet_df64",
        ),
        Fp64Strategy::Native | Fp64Strategy::Sovereign | Fp64Strategy::Concurrent => gpu
            .create_pipeline_f64(
                shaders::SHADER_YUKAWA_FORCE_VERLET,
                "yukawa_force_verlet_f64",
            ),
    };
    let kick_drift_pipeline =
        gpu.create_pipeline(shaders::SHADER_VV_KICK_DRIFT, "vv_kick_drift_f64");
    let half_kick_pipeline = gpu.create_pipeline(shaders::SHADER_VV_HALF_KICK, "vv_half_kick_f64");
    let berendsen_pipeline = gpu.create_pipeline(shaders::SHADER_BERENDSEN, "berendsen_f64");
    let ke_pipeline = gpu.create_pipeline(shaders::SHADER_KINETIC_ENERGY, "kinetic_energy_f64");

    println!(
        "    Compiled 5+verlet shaders in {:.1}ms",
        t_compile.elapsed().as_secs_f64() * 1000.0
    );

    let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n)?;

    let max_neighbors = VERLET_MAX_NEIGHBORS;
    let verlet = crate::md::neighbor::VerletListGpu::new(&gpu, n, [box_side; 3], config.rc, skin)?;

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
        f64::from(max_neighbors),
    ];
    let force_params_buf = gpu.create_f64_buffer(&force_params, "force_params");

    let verlet_build_params = vec![
        n as f64,
        config.kappa,
        prefactor,
        (config.rc + skin) * (config.rc + skin),
        box_side,
        box_side,
        box_side,
        f64::from(max_neighbors),
    ];
    let (mx, my, mz) = verlet.cell_list_grid();
    let cell_size = box_side / f64::from(mx);
    let mut vb_params = verlet_build_params;
    vb_params.extend_from_slice(&[
        f64::from(mx),
        f64::from(my),
        f64::from(mz),
        cell_size,
        cell_size,
        cell_size,
        f64::from(verlet.cell_list_n_cells()),
        0.0,
    ]);
    let verlet_build_params_buf = gpu.create_f64_buffer(&vb_params, "verlet_build_params");

    let disp_params = vec![n as f64, box_side, box_side, box_side];
    let disp_params_buf = gpu.create_f64_buffer(&disp_params, "disp_params");

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
        &[
            &pos_buf,
            &force_buf,
            &pe_buf,
            &force_params_buf,
            verlet.neighbor_list(),
            verlet.neighbor_count(),
        ],
    );
    let kick_drift_bg = gpu.create_bind_group(
        &kick_drift_pipeline,
        &[&pos_buf, &vel_buf, &force_buf, &vv_params_buf],
    );
    let half_kick_bg =
        gpu.create_bind_group(&half_kick_pipeline, &[&vel_buf, &force_buf, &hk_params_buf]);
    let ke_bg = gpu.create_bind_group(&ke_pipeline, &[&vel_buf, &ke_buf, &ke_params_buf]);

    let t_start = Instant::now();

    verlet.build(&gpu, &pos_buf, &verlet_build_params_buf)?;
    gpu.dispatch(&force_pipeline, &force_bg, workgroups);

    println!("    Verlet list: {mx}×{my}×{mz} cells, skin={skin:.3}, max_nb={max_neighbors}");

    // ── Equilibration ──
    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();
    let thermostat_interval = THERMOSTAT_INTERVAL;
    let mut rebuild_count = 0u32;

    let mut step = 0;
    while step < config.equil_steps {
        let batch_size = thermostat_interval.min(config.equil_steps - step);

        let mut encoder = gpu.begin_encoder("equil_verlet");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("equil_verlet_stream"),
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

        if verlet.needs_rebuild(&gpu, &pos_buf, &disp_params_buf) {
            verlet.build(&gpu, &pos_buf, &verlet_build_params_buf)?;
            rebuild_count += 1;
        }

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
    println!("    Equilibration complete in {equil_time:.2}s ({rebuild_count} Verlet rebuilds)");

    // ── Production ──
    println!(
        "  ── Production ({} steps, Verlet adaptive rebuild) ──",
        config.prod_steps
    );
    let t_prod = Instant::now();
    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    let pos_staging = gpu.create_staging_buffer(n * 3 * 8, "pos_staging");
    let vel_staging = gpu.create_staging_buffer(n * 3 * 8, "vel_staging");

    let n_dumps = config.prod_steps / config.dump_step;
    let snap_every = config.vel_snapshot_interval;
    rebuild_count = 0;

    let mut brain = MdBrain::new();
    if let Some(json) = nautilus_json {
        if brain.import_nautilus_json(json) {
            println!(
                "    Brain: imported Nautilus ({} obs, {} gens, {} retrains) from previous run",
                brain.observation_count(),
                brain.nautilus_generations(),
                brain.readout_retrain_count(),
            );
        } else {
            println!("    Brain: Nautilus import failed, starting fresh");
        }
    }

    for dump_idx in 0..n_dumps {
        let step_start = dump_idx * config.dump_step;
        let step_end = step_start + config.dump_step;
        let need_snapshot = dump_idx % snap_every == 0;

        let mut encoder = gpu.begin_encoder("md_verlet_batch");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("md_verlet_stream"),
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

        if verlet.needs_rebuild(&gpu, &pos_buf, &disp_params_buf) {
            verlet.build(&gpu, &pos_buf, &verlet_build_params_buf)?;
            rebuild_count += 1;
        }

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

        let elapsed = t_prod.elapsed().as_secs_f64();
        let current_sps = if elapsed > 0.0 {
            step_end as f64 / elapsed
        } else {
            0.0
        };
        let steering = brain.observe(&MdStepEvent {
            step: step_end,
            ke: total_ke,
            pe: total_pe,
            total_energy: total_e,
            temperature: t_current,
            target_temperature: temperature,
            kappa: config.kappa,
            gamma: config.gamma,
            n_particles: n,
            algorithm: ForceAlgorithm::VerletList { skin },
            rebuild_count: rebuild_count as usize,
            steps_per_sec: current_sps,
            wall_time_s: elapsed,
            skin_fraction: skin / config.rc,
        });

        if step_end % 5000 < config.dump_step || step_end >= config.prod_steps {
            let brain_info = if brain.readout_retrain_count() > 0 {
                format!(
                    " [brain: {}R/{}G, {}/{} heads trusted, conf={:.2}, {}obs]",
                    brain.readout_retrain_count(),
                    brain.nautilus_generations(),
                    brain.head_trust().iter().filter(|&&t| t).count(),
                    crate::md::brain::MD_NUM_HEADS,
                    steering.confidence,
                    brain.observation_count(),
                )
            } else {
                String::new()
            };
            println!(
                "    Step {}: T*={:.6}, KE={:.4}, PE={:.4}, E={:.4}{brain_info}",
                step_end - 1,
                t_current,
                total_ke,
                total_pe,
                total_e
            );
        }
    }

    let prod_time = t_prod.elapsed().as_secs_f64();
    let total_time = t_start.elapsed().as_secs_f64();
    let total_steps = config.equil_steps + config.prod_steps;
    let steps_per_sec = total_steps as f64 / total_time;

    let brain_summary = BrainSummary {
        retrain_count: brain.readout_retrain_count(),
        trusted_heads: brain.head_trust().iter().filter(|&&t| t).count(),
        confidence: {
            let r2s = brain.head_confidence();
            let positive: Vec<f64> = r2s.iter().filter(|&&r| r > 0.0).copied().collect();
            if positive.is_empty() {
                0.0
            } else {
                positive.iter().sum::<f64>() / r2s.len() as f64
            }
        },
        head_r2: brain.head_confidence().to_vec(),
        anomaly_detected: false,
        nautilus_json: brain.export_nautilus_json(),
        nautilus_observations: brain.observation_count(),
        nautilus_generations: brain.nautilus_generations(),
    };

    println!("    Production complete in {prod_time:.2}s ({rebuild_count} Verlet rebuilds)");
    println!("    Total: {total_time:.2}s ({steps_per_sec:.1} steps/s)");
    if brain_summary.retrain_count > 0 {
        println!(
            "    Brain: {} readout retrains, {} board gens, {}/{} heads trusted, R²={:?}, {} obs",
            brain_summary.retrain_count,
            brain_summary.nautilus_generations,
            brain_summary.trusted_heads,
            crate::md::brain::MD_NUM_HEADS,
            brain_summary
                .head_r2
                .iter()
                .map(|r| format!("{r:.3}"))
                .collect::<Vec<_>>(),
            brain_summary.nautilus_observations,
        );
    }

    Ok(MdSimulation {
        config: config.clone(),
        energy_history,
        positions_snapshots,
        velocity_snapshots,
        rdf_histogram: Vec::new(),
        wall_time_s: total_time,
        steps_per_sec,
        brain_summary: Some(brain_summary),
    })
}
