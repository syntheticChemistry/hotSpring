// SPDX-License-Identifier: AGPL-3.0-only

//! Backend-agnostic MD engine using `GpuBackend` + `ComputeDispatch<B>`.
//!
//! This module provides the same Yukawa OCP physics as `md::simulation` but
//! dispatches through barraCuda's `GpuBackend` trait, enabling both wgpu/Vulkan
//! and sovereign (coralReef → DRM) backends from the same generic code.
//!
//! ## Cross-spring evolution
//!
//! The precision shaders (df64_core, df64_transcendentals) originated in
//! hotSpring Exp 028 and were absorbed upstream into barraCuda `a012076`.
//! coralReef Iter 33 validated that sovereign compilation of these shaders
//! bypasses the naga SPIR-V codegen bug that poisons DF64 transcendentals.
//!
//! `ComputeDispatch` (barraCuda `875e116`) routes the same WGSL source through
//! either Vulkan (naga → SPIR-V) or sovereign (coralReef → native SASS/GFX).
//! `BatchedComputeDispatch` (barraCuda `0649cd0`) batches the per-step
//! kick_drift + force + half_kick dispatches into a single GPU submission,
//! reducing host overhead from ~1.6ms to ~0.1ms amortized on Vulkan.

use barracuda::device::backend::GpuBackend;
use barracuda::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};

use crate::md::config::MdConfig;
use crate::md::shaders;
use crate::md::simulation::{init_fcc_lattice, init_velocities, EnergyRecord, MdSimulation};
use crate::tolerances::{DEFAULT_VELOCITY_SEED, MD_TEMPERATURE_FLOOR, MD_WORKGROUP_SIZE};

use std::time::Instant;

/// Backend-agnostic MD buffers.
struct MdBuffers<B: GpuBackend> {
    pos: B::Buffer,
    vel: B::Buffer,
    force: B::Buffer,
    pe: B::Buffer,
    ke: B::Buffer,
    force_params: B::Buffer,
    vv_params: B::Buffer,
    hk_params: B::Buffer,
    ke_params: B::Buffer,
}

/// Run a full Yukawa OCP MD simulation on any `GpuBackend`.
///
/// Uses `ComputeDispatch<B>` for all shader dispatch — works identically on
/// wgpu/Vulkan and sovereign/DRM backends. Energy reduction is CPU-side (sum
/// after readback) to avoid backend-specific reduce pipeline dependencies.
///
/// # Errors
///
/// Returns `String` on GPU allocation, dispatch, or readback failure.
pub fn run_simulation_generic<B: GpuBackend>(
    backend: &B,
    config: &MdConfig,
) -> Result<MdSimulation, String> {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let temperature = config.temperature();
    let mass = config.reduced_mass();

    println!("  ── MdEngine<{}> initializing {n} particles ──", backend.name());
    println!("    κ = {}, Γ = {}, T* = {temperature:.6}", config.kappa, config.gamma);

    let (positions, n_actual) = init_fcc_lattice(n, box_side);
    let n: usize = n_actual.min(n);
    let velocities = init_velocities(n, temperature, mass, DEFAULT_VELOCITY_SEED);
    println!("    Placed {n} particles on FCC lattice");

    let bufs = alloc_buffers(backend, n, config)?;

    backend.upload_f64(&bufs.pos, &positions);
    backend.upload_f64(&bufs.vel, &velocities);

    let wg = n.div_ceil(MD_WORKGROUP_SIZE) as u32;

    // Initial force computation
    let t_compile = Instant::now();
    dispatch_force(backend, &bufs, wg)?;
    println!(
        "    First dispatch in {:.1}ms (includes shader compile)",
        t_compile.elapsed().as_secs_f64() * 1000.0
    );

    // ── Equilibration ──
    println!("  ── Equilibration ({} steps) ──", config.equil_steps);
    let t_equil = Instant::now();
    let thermostat_interval = crate::tolerances::THERMOSTAT_INTERVAL;
    let mut step = 0;

    while step < config.equil_steps {
        let batch_size = thermostat_interval.min(config.equil_steps - step);

        for _ in 0..batch_size {
            dispatch_md_step(backend, &bufs, wg)?;
        }

        dispatch_ke(backend, &bufs, wg)?;
        let ke_data = backend
            .download_f64(&bufs.ke, n)
            .map_err(|e| format!("KE readback: {e}"))?;
        let total_ke: f64 = ke_data.iter().sum();
        let t_current = 2.0 * total_ke / (3.0 * n as f64);

        if t_current > MD_TEMPERATURE_FLOOR {
            let ratio =
                (config.dt / config.berendsen_tau).mul_add(temperature / t_current - 1.0, 1.0);
            let scale = ratio.max(0.0).sqrt();
            let beren_params = [n as f64, scale, 0.0, 0.0];
            let beren_buf = backend
                .alloc_buffer_f64_init("beren_params", &beren_params)
                .map_err(|e| format!("beren alloc: {e}"))?;
            dispatch_berendsen(backend, &bufs, &beren_buf, wg)?;
        }

        step += batch_size;
        if step % 1000 < thermostat_interval || step >= config.equil_steps {
            println!("    Step {step}: T* = {t_current:.6} (target {temperature:.6})");
        }
    }
    let equil_time = t_equil.elapsed().as_secs_f64();
    println!("    Equilibration complete in {equil_time:.2}s");

    // ── Production ──
    println!("  ── Production ({} steps) ──", config.prod_steps);
    let t_prod = Instant::now();

    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    let n_dumps = config.prod_steps / config.dump_step;
    let snap_every = config.vel_snapshot_interval;

    for dump_idx in 0..n_dumps {
        let step_start = dump_idx * config.dump_step;
        let step_end = step_start + config.dump_step;
        let need_snapshot = dump_idx % snap_every == 0;

        for _ in step_start..step_end {
            dispatch_md_step(backend, &bufs, wg)?;
        }

        dispatch_ke(backend, &bufs, wg)?;
        let ke_data = backend
            .download_f64(&bufs.ke, n)
            .map_err(|e| format!("KE readback: {e}"))?;
        let pe_data = backend
            .download_f64(&bufs.pe, n)
            .map_err(|e| format!("PE readback: {e}"))?;
        let total_ke: f64 = ke_data.iter().sum();
        let total_pe: f64 = pe_data.iter().sum();
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
            let pos = backend
                .download_f64(&bufs.pos, n * 3)
                .map_err(|e| format!("pos readback: {e}"))?;
            let vel = backend
                .download_f64(&bufs.vel, n * 3)
                .map_err(|e| format!("vel readback: {e}"))?;
            positions_snapshots.push(pos);
            velocity_snapshots.push(vel);
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
        for _ in 0..remainder {
            dispatch_md_step(backend, &bufs, wg)?;
        }
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

// ── Buffer allocation ───────────────────────────────────────────────────

fn alloc_buffers<B: GpuBackend>(
    backend: &B,
    n: usize,
    config: &MdConfig,
) -> Result<MdBuffers<B>, String> {
    let e = |msg: &str, e: barracuda::error::BarracudaError| format!("{msg}: {e}");

    let pos = backend
        .create_backend_buffer_f64(n * 3)
        .map_err(|err| e("pos", err))?;
    let vel = backend
        .create_backend_buffer_f64(n * 3)
        .map_err(|err| e("vel", err))?;
    let force = backend
        .create_backend_buffer_f64(n * 3)
        .map_err(|err| e("force", err))?;
    let pe = backend
        .create_backend_buffer_f64(n)
        .map_err(|err| e("pe", err))?;
    let ke = backend
        .create_backend_buffer_f64(n)
        .map_err(|err| e("ke", err))?;

    let box_side = config.box_side();
    let prefactor = config.force_prefactor();

    let force_params_data = [
        n as f64,
        config.kappa,
        prefactor,
        config.rc * config.rc,
        box_side,
        box_side,
        box_side,
        0.0,
    ];
    let force_params = backend
        .alloc_buffer_f64_init("force_params", &force_params_data)
        .map_err(|err| e("force_params", err))?;

    let mass = config.reduced_mass();
    let vv_params_data = [n as f64, config.dt, mass, 0.0, box_side, box_side, box_side, 0.0];
    let vv_params = backend
        .alloc_buffer_f64_init("vv_params", &vv_params_data)
        .map_err(|err| e("vv_params", err))?;

    let hk_params_data = [n as f64, config.dt, mass, 0.0];
    let hk_params = backend
        .alloc_buffer_f64_init("hk_params", &hk_params_data)
        .map_err(|err| e("hk_params", err))?;

    let ke_params_data = [n as f64, mass, 0.0, 0.0];
    let ke_params = backend
        .alloc_buffer_f64_init("ke_params", &ke_params_data)
        .map_err(|err| e("ke_params", err))?;

    Ok(MdBuffers {
        pos,
        vel,
        force,
        pe,
        ke,
        force_params,
        vv_params,
        hk_params,
        ke_params,
    })
}

// ── Dispatch helpers ────────────────────────────────────────────────────
// Each maps to one WGSL compute shader. Binding indices match the
// @group(0) @binding(N) declarations in the .wgsl files.

/// One Velocity-Verlet MD step: kick_drift → force → half_kick in a single
/// GPU submission via `BatchedComputeDispatch`. Amortizes per-dispatch
/// host-side overhead (~1.6ms → ~0.1ms on Vulkan).
fn dispatch_md_step<B: GpuBackend>(
    backend: &B,
    bufs: &MdBuffers<B>,
    wg: u32,
) -> Result<(), String> {
    let mut batch = BatchedComputeDispatch::new(backend);
    batch
        .push(
            ComputeDispatch::new(backend, "kick_drift")
                .shader(shaders::SHADER_VV_KICK_DRIFT, "main")
                .f64()
                .storage_rw(0, &bufs.pos)
                .storage_rw(1, &bufs.vel)
                .storage_read(2, &bufs.force)
                .storage_read(3, &bufs.vv_params)
                .dispatch(wg, 1, 1),
        )
        .map_err(|e| format!("batch kick_drift: {e}"))?;
    batch
        .push(
            ComputeDispatch::new(backend, "yukawa_force")
                .shader(shaders::SHADER_YUKAWA_FORCE, "main")
                .f64()
                .storage_read(0, &bufs.pos)
                .storage_rw(1, &bufs.force)
                .storage_rw(2, &bufs.pe)
                .storage_read(3, &bufs.force_params)
                .dispatch(wg, 1, 1),
        )
        .map_err(|e| format!("batch force: {e}"))?;
    batch
        .push(
            ComputeDispatch::new(backend, "half_kick")
                .shader(shaders::SHADER_VV_HALF_KICK, "main")
                .f64()
                .storage_rw(0, &bufs.vel)
                .storage_read(1, &bufs.force)
                .storage_read(2, &bufs.hk_params)
                .dispatch(wg, 1, 1),
        )
        .map_err(|e| format!("batch half_kick: {e}"))?;
    batch.submit().map_err(|e| format!("md_step batch: {e}"))
}

fn dispatch_force<B: GpuBackend>(
    backend: &B,
    bufs: &MdBuffers<B>,
    wg: u32,
) -> Result<(), String> {
    ComputeDispatch::new(backend, "yukawa_force")
        .shader(shaders::SHADER_YUKAWA_FORCE, "main")
        .f64()
        .storage_read(0, &bufs.pos)
        .storage_rw(1, &bufs.force)
        .storage_rw(2, &bufs.pe)
        .storage_read(3, &bufs.force_params)
        .dispatch(wg, 1, 1)
        .submit()
        .map_err(|e| format!("force dispatch: {e}"))
}

fn dispatch_ke<B: GpuBackend>(
    backend: &B,
    bufs: &MdBuffers<B>,
    wg: u32,
) -> Result<(), String> {
    ComputeDispatch::new(backend, "kinetic_energy")
        .shader(shaders::SHADER_KINETIC_ENERGY, "main")
        .f64()
        .storage_read(0, &bufs.vel)
        .storage_rw(1, &bufs.ke)
        .storage_read(2, &bufs.ke_params)
        .dispatch(wg, 1, 1)
        .submit()
        .map_err(|e| format!("ke dispatch: {e}"))
}

fn dispatch_berendsen<B: GpuBackend>(
    backend: &B,
    bufs: &MdBuffers<B>,
    beren_params: &B::Buffer,
    wg: u32,
) -> Result<(), String> {
    ComputeDispatch::new(backend, "berendsen")
        .shader(shaders::SHADER_BERENDSEN, "main")
        .f64()
        .storage_rw(0, &bufs.vel)
        .storage_read(1, beren_params)
        .dispatch(wg, 1, 1)
        .submit()
        .map_err(|e| format!("berendsen dispatch: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrip_sanity() {
        let config = MdConfig {
            label: "test".to_string(),
            n_particles: 32,
            kappa: 2.0,
            gamma: 160.0,
            dt: 0.005,
            rc: 6.0,
            equil_steps: 10,
            prod_steps: 10,
            dump_step: 5,
            berendsen_tau: 0.5,
            rdf_bins: 50,
            vel_snapshot_interval: 1,
        };
        assert!(config.n_particles > 0);
        assert!(config.box_side() > 0.0);
        assert!(config.force_prefactor() > 0.0);
    }
}
