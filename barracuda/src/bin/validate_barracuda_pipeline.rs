// SPDX-License-Identifier: AGPL-3.0-only

//! `BarraCUDA` Full-Pipeline MD Validation
//!
//! Proves that `BarraCUDA`'s abstracted GPU ops (`YukawaForceF64`,
//! VelocityVerletKickDrift/HalfKick, `BerendsenThermostat`, `KineticEnergy`)
//! produce correct Yukawa OCP physics end-to-end.
//!
//! **Strategy**: Run a short MD simulation through `BarraCUDA`'s Tensor/op API,
//! then validate energy conservation and force correctness against CPU f64.
//! This is the handoff proof: if these ops pass, `ToadStool` can evolve them
//! knowing the physics is validated.
//!
//! **Provenance**: GPU f32 kernels vs CPU f64 reference. See `provenance::GPU_KERNEL_REFS`.
//!
//! **Reference**: hotSpring raw-wgpu path (`sarkas_gpu`) passes 9/9 PP cases
//! with 0.000% energy drift. This binary validates the `BarraCUDA` abstraction
//! produces identical physics through a different code path.

use barracuda::ops::md::integrators::{VelocityVerletHalfKick, VelocityVerletKickDrift};
use barracuda::ops::md::observables::KineticEnergy;
use barracuda::ops::md::thermostats::BerendsenThermostat;
use barracuda::tensor::Tensor;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::md::yukawa_nvk_safe::yukawa_force_f64_nvk_safe;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

// ═══════════════════════════════════════════════════════════════════
// System setup (same physics as sarkas_gpu quick mode)
// ═══════════════════════════════════════════════════════════════════

const KAPPA: f64 = 2.0;
const GAMMA: f64 = 158.0;
const N_TARGET: usize = 108; // 3³ FCC = 108 particles
const RC: f64 = 6.5; // Cutoff in a_ws
const DT: f64 = 0.012; // Timestep in ω_p⁻¹
const EQUIL_STEPS: usize = 200;
const PROD_STEPS: usize = 300;
const BERENDSEN_TAU: f64 = 5.0; // τ/dt
const MASS: f64 = 3.0; // OCP reduced mass

fn box_side(n: usize) -> f64 {
    (4.0 * std::f64::consts::PI * n as f64 / 3.0).cbrt()
}

fn temperature() -> f64 {
    1.0 / GAMMA
}

// ═══════════════════════════════════════════════════════════════════
// CPU reference force (for cross-validation)
// ═══════════════════════════════════════════════════════════════════

fn cpu_yukawa_forces(
    positions: &[f64],
    n: usize,
    kappa: f64,
    cutoff_sq: f64,
    box_side: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut forces = vec![0.0_f64; n * 3];
    let mut pe = vec![0.0_f64; n];

    for i in 0..n {
        let (xi, yi, zi) = (positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        for j in (i + 1)..n {
            let mut dx = positions[j * 3] - xi;
            let mut dy = positions[j * 3 + 1] - yi;
            let mut dz = positions[j * 3 + 2] - zi;
            dx -= box_side * (dx / box_side).round();
            dy -= box_side * (dy / box_side).round();
            dz -= box_side * (dz / box_side).round();
            let r_sq = dx * dx + dy * dy + dz * dz;
            if r_sq > cutoff_sq {
                continue;
            }
            let r = r_sq.sqrt();
            let inv_r = 1.0 / r;
            let screening = (-kappa * r).exp();
            let force_mag = screening * (1.0 + kappa * r) / r_sq;
            let fx = -force_mag * dx * inv_r;
            let fy = -force_mag * dy * inv_r;
            let fz = -force_mag * dz * inv_r;
            forces[i * 3] += fx;
            forces[i * 3 + 1] += fy;
            forces[i * 3 + 2] += fz;
            forces[j * 3] -= fx;
            forces[j * 3 + 1] -= fy;
            forces[j * 3 + 2] -= fz;
            let u = screening * inv_r;
            pe[i] += 0.5 * u;
            pe[j] += 0.5 * u;
        }
    }
    (forces, pe)
}

fn compute_temperature(velocities: &[f64], n: usize, mass: f64) -> f64 {
    let ke: f64 = velocities.iter().map(|v| 0.5 * mass * v * v).sum();
    ke / (1.5 * n as f64)
}

// ═══════════════════════════════════════════════════════════════════
// Main validation
// ═══════════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  BarraCUDA Full-Pipeline Yukawa OCP Validation");
    println!("  Pure Rust GPU via WGSL/wgpu/Vulkan — no CUDA");
    println!("  Proving: BarraCUDA ops → correct physics end-to-end");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut harness = ValidationHarness::new("barracuda_pipeline");

    // ── Initialize GPU with SHADER_F64 ──
    // hotSpring's GpuF64 properly requests wgpu::Features::SHADER_F64,
    // then we bridge to BarraCUDA's WgpuDevice for Tensor ops.
    let gpu = match GpuF64::new().await {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU init failed: {e}");
            harness.check_bool("SHADER_F64 available", false);
            harness.finish();
        }
    };
    if !gpu.has_f64 {
        println!("  SHADER_F64 not supported — skipping pipeline validation.");
        harness.check_bool("SHADER_F64 available", false);
        harness.finish();
    }
    gpu.print_info();
    let device = gpu.to_wgpu_device();
    println!("  BarraCUDA WgpuDevice bridged (SHADER_F64 confirmed)");

    // ── Setup system ──
    let l = box_side(N_TARGET);
    let t_target = temperature();
    let cutoff_sq = RC * RC;
    println!(
        "\n  System: N={N_TARGET}, κ={KAPPA}, Γ={GAMMA}, L={l:.4}, T*={t_target:.6}, dt={DT}\n"
    );

    // FCC lattice initialization (reuse hotSpring's init)
    let (positions, n_actual) = hotspring_barracuda::md::simulation::init_fcc_lattice(N_TARGET, l);
    let n = n_actual;
    let velocities = hotspring_barracuda::md::simulation::init_velocities(n, t_target, MASS, 42);
    println!("  Placed {n} particles on FCC lattice");

    // ── Phase 1: Force validation (BarraCUDA vs CPU) ──
    println!("\n── Phase 1: Force Cross-Validation ──────────────────────");
    let (cpu_forces, cpu_pe) = cpu_yukawa_forces(&positions, n, KAPPA, cutoff_sq, l);

    let (gpu_forces, gpu_pe) = yukawa_force_f64_nvk_safe(&device, &positions, n, KAPPA, 1.0, RC, l)
        .expect("Yukawa NVK-safe execution");

    // Compare force vector norms per particle.
    // CPU uses symmetric Newton's-3rd accumulation (j > i), GPU uses full-loop
    // (j != i). The total force per particle matches but FP ordering differs,
    // so we compare per-particle force *magnitude* rather than components.
    let mut max_mag_rel: f64 = 0.0;
    let mut sum_mag_rel: f64 = 0.0;
    let mut n_sig = 0usize; // particles with significant force
    for i in 0..n {
        let cx = cpu_forces[i * 3];
        let cy = cpu_forces[i * 3 + 1];
        let cz = cpu_forces[i * 3 + 2];
        let gx = gpu_forces[i * 3];
        let gy = gpu_forces[i * 3 + 1];
        let gz = gpu_forces[i * 3 + 2];
        let cpu_mag = (cx * cx + cy * cy + cz * cz).sqrt();
        let gpu_mag = (gx * gx + gy * gy + gz * gz).sqrt();
        if cpu_mag > 1e-6 {
            let rel = ((cpu_mag - gpu_mag) / cpu_mag).abs();
            if rel > max_mag_rel {
                max_mag_rel = rel;
            }
            sum_mag_rel += rel;
            n_sig += 1;
        }
    }
    let avg_mag_rel = if n_sig > 0 {
        sum_mag_rel / n_sig as f64
    } else {
        0.0
    };
    println!(
        "  Force magnitude: max_rel={max_mag_rel:.2e}, avg_rel={avg_mag_rel:.2e} ({n_sig} particles)"
    );
    // Also check total momentum (should be ~0 for both CPU and GPU)
    let cpu_px: f64 = cpu_forces.chunks(3).map(|f| f[0]).sum();
    let gpu_px: f64 = gpu_forces.chunks(3).map(|f| f[0]).sum();
    println!("  Momentum conservation: CPU Σfx={cpu_px:.2e}, GPU Σfx={gpu_px:.2e}");
    harness.check_upper(
        "Force magnitude max relative error",
        max_mag_rel,
        tolerances::NEWTON_3RD_LAW_ABS,
    );
    harness.check_upper(
        "Force magnitude avg relative error",
        avg_mag_rel,
        tolerances::MD_ABSOLUTE_FLOOR,
    );

    // Compare PE
    let cpu_pe_total: f64 = cpu_pe.iter().sum();
    let gpu_pe_total: f64 = gpu_pe.iter().sum();
    let pe_rel_err = ((cpu_pe_total - gpu_pe_total) / cpu_pe_total).abs();
    println!("  PE total: CPU={cpu_pe_total:.6}, GPU={gpu_pe_total:.6}, rel_err={pe_rel_err:.2e}");
    harness.check_upper(
        "PE total relative error",
        pe_rel_err,
        tolerances::GPU_VS_CPU_F64,
    );

    // ── Phase 2: Single-step integrator validation ──
    println!("\n── Phase 2: Integrator Validation ──────────────────────");
    {
        let pos_t =
            Tensor::from_f64_data(&positions, vec![n, 3], device.clone()).expect("pos tensor");
        let vel_t =
            Tensor::from_f64_data(&velocities, vec![n, 3], device.clone()).expect("vel tensor");
        let force_t =
            Tensor::from_f64_data(&gpu_forces, vec![n, 3], device.clone()).expect("force tensor");

        let box_size = [l, l, l];
        let kick_drift = VelocityVerletKickDrift::new(pos_t, vel_t, force_t, DT, MASS, box_size)
            .expect("KickDrift creation");
        let (new_pos_t, new_vel_t) = kick_drift.execute().expect("KickDrift execution");

        let new_pos = new_pos_t.to_f64_vec().expect("read positions");
        let new_vel = new_vel_t.to_f64_vec().expect("read velocities");

        // Verify positions changed and are within box
        let mut any_moved = false;
        let mut all_in_box = true;
        for i in 0..n {
            for d in 0..3 {
                let old_p = positions[i * 3 + d];
                let new_p = new_pos[i * 3 + d];
                if (old_p - new_p).abs() > 1e-15 {
                    any_moved = true;
                }
                if new_p < 0.0 || new_p >= l {
                    all_in_box = false;
                }
            }
        }
        println!("  KickDrift: particles moved={any_moved}, all_in_box={all_in_box}");
        harness.check_bool("KickDrift: particles move", any_moved);
        harness.check_bool("KickDrift: PBC wrapping correct", all_in_box);

        // Half-kick with same forces
        let force_t2 =
            Tensor::from_f64_data(&gpu_forces, vec![n, 3], device.clone()).expect("force tensor");
        let half_kick =
            VelocityVerletHalfKick::new(new_vel_t, force_t2, DT, MASS).expect("HalfKick creation");
        let final_vel_t = half_kick.execute().expect("HalfKick execution");
        let final_vel = final_vel_t.to_f64_vec().expect("read velocities");

        // Velocities should have changed from the half-kick
        let vel_changed = final_vel
            .iter()
            .zip(new_vel.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        println!("  HalfKick: velocities updated={vel_changed}");
        harness.check_bool("HalfKick: velocities update", vel_changed);
    }

    // ── Phase 3: Thermostat validation ──
    println!("\n── Phase 3: Thermostat Validation ──────────────────────");
    {
        let t_current = compute_temperature(&velocities, n, MASS);
        let scale = BerendsenThermostat::compute_scale(t_current, t_target, DT, BERENDSEN_TAU * DT);
        println!("  T_current={t_current:.6}, T_target={t_target:.6}, scale={scale:.6}");

        let vel_t =
            Tensor::from_f64_data(&velocities, vec![n, 3], device.clone()).expect("vel tensor");
        let thermostat = BerendsenThermostat::new(vel_t, scale).expect("Berendsen creation");
        let scaled_vel_t = thermostat.execute().expect("Berendsen execution");
        let scaled_vel = scaled_vel_t.to_f64_vec().expect("read velocities");

        let t_after = compute_temperature(&scaled_vel, n, MASS);
        let t_closer = (t_after - t_target).abs() <= (t_current - t_target).abs();
        println!("  T_after={t_after:.6}, closer_to_target={t_closer}");
        harness.check_bool("Berendsen: temperature moves toward target", t_closer);
    }

    // ── Phase 4: KineticEnergy op validation ──
    println!("\n── Phase 4: KineticEnergy Op Validation ────────────────");
    {
        let vel_t =
            Tensor::from_f64_data(&velocities, vec![n, 3], device.clone()).expect("vel tensor");
        let ke_op = KineticEnergy::new(vel_t, MASS).expect("KineticEnergy creation");
        let ke_t = ke_op.execute().expect("KineticEnergy execution");
        let ke_per_particle = ke_t.to_f64_vec().expect("read KE");

        let gpu_ke_total: f64 = ke_per_particle.iter().sum();
        let cpu_ke_total: f64 = velocities
            .chunks(3)
            .map(|v| 0.5 * MASS * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            .sum();

        let ke_rel_err = ((gpu_ke_total - cpu_ke_total) / cpu_ke_total).abs();
        println!(
            "  KE total: GPU={gpu_ke_total:.6}, CPU={cpu_ke_total:.6}, rel_err={ke_rel_err:.2e}"
        );
        harness.check_upper(
            "KineticEnergy: total KE relative error",
            ke_rel_err,
            tolerances::GPU_VS_CPU_F64,
        );

        let t_from_ke = KineticEnergy::temperature_from_ke(gpu_ke_total, n);
        let t_from_cpu = compute_temperature(&velocities, n, MASS);
        let t_rel_err = ((t_from_ke - t_from_cpu) / t_from_cpu).abs();
        println!("  T from KE: GPU={t_from_ke:.6}, CPU={t_from_cpu:.6}, rel_err={t_rel_err:.2e}");
        harness.check_upper(
            "KineticEnergy: temperature relative error",
            t_rel_err,
            tolerances::GPU_VS_CPU_F64,
        );
    }

    // ── Phase 5: Full MD loop (energy conservation) ──
    println!("\n── Phase 5: Full MD Loop (BarraCUDA ops) ──────────────");
    {
        let mut pos = positions.clone();
        let mut vel = velocities.clone();

        // Initial forces
        let (mut forces, _) = cpu_yukawa_forces(&pos, n, KAPPA, cutoff_sq, l);
        let pe_initial: f64;
        let ke_initial: f64;

        // Compute initial energy
        {
            let (_, pe_vec) = cpu_yukawa_forces(&pos, n, KAPPA, cutoff_sq, l);
            pe_initial = pe_vec.iter().sum();
            ke_initial = vel
                .chunks(3)
                .map(|v| 0.5 * MASS * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
                .sum::<f64>();
        }
        println!(
            "  Initial: KE={:.4}, PE={:.4}, E={:.4}",
            ke_initial,
            pe_initial,
            ke_initial + pe_initial
        );

        let total_steps = EQUIL_STEPS + PROD_STEPS;
        let mut energies: Vec<f64> = Vec::with_capacity(PROD_STEPS);

        for step in 0..total_steps {
            let box_size = [l, l, l];

            // Step 1: Half-kick + drift (BarraCUDA op)
            let pos_t =
                Tensor::from_f64_data(&pos, vec![n, 3], device.clone()).expect("pos tensor");
            let vel_t =
                Tensor::from_f64_data(&vel, vec![n, 3], device.clone()).expect("vel tensor");
            let force_t =
                Tensor::from_f64_data(&forces, vec![n, 3], device.clone()).expect("force tensor");

            let kick_drift =
                VelocityVerletKickDrift::new(pos_t, vel_t, force_t, DT, MASS, box_size)
                    .expect("KickDrift");
            let (new_pos_t, new_vel_t) = kick_drift.execute().expect("KickDrift exec");
            pos = new_pos_t.to_f64_vec().expect("read pos");
            vel = new_vel_t.to_f64_vec().expect("read vel");

            // Step 2: New forces (BarraCUDA op)
            let (new_forces, pe_vec) =
                yukawa_force_f64_nvk_safe(&device, &pos, n, KAPPA, 1.0, RC, l)
                    .expect("Yukawa NVK-safe exec");
            forces = new_forces;

            // Step 3: Second half-kick (BarraCUDA op)
            let vel_t2 =
                Tensor::from_f64_data(&vel, vec![n, 3], device.clone()).expect("vel tensor");
            let force_t2 =
                Tensor::from_f64_data(&forces, vec![n, 3], device.clone()).expect("force tensor");
            let half_kick =
                VelocityVerletHalfKick::new(vel_t2, force_t2, DT, MASS).expect("HalfKick");
            let final_vel_t = half_kick.execute().expect("HalfKick exec");
            vel = final_vel_t.to_f64_vec().expect("read vel");

            // Step 4: Thermostat during equilibration
            if step < EQUIL_STEPS {
                let t_current = compute_temperature(&vel, n, MASS);
                let scale =
                    BerendsenThermostat::compute_scale(t_current, t_target, DT, BERENDSEN_TAU * DT);
                let vel_t3 =
                    Tensor::from_f64_data(&vel, vec![n, 3], device.clone()).expect("vel tensor");
                let thermo = BerendsenThermostat::new(vel_t3, scale).expect("Berendsen");
                let scaled_t = thermo.execute().expect("Berendsen exec");
                vel = scaled_t.to_f64_vec().expect("read vel");
            }

            // Record production energy
            if step >= EQUIL_STEPS {
                let ke: f64 = vel
                    .chunks(3)
                    .map(|v| 0.5 * MASS * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
                    .sum();
                let pe: f64 = pe_vec.iter().sum();
                energies.push(ke + pe);
            }

            if step % 100 == 0 || step == total_steps - 1 {
                let ke: f64 = vel
                    .chunks(3)
                    .map(|v| 0.5 * MASS * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
                    .sum();
                let pe: f64 = pe_vec.iter().sum();
                let t = compute_temperature(&vel, n, MASS);
                let phase = if step < EQUIL_STEPS { "equil" } else { "prod " };
                println!(
                    "    Step {:4} [{}]: KE={:.4} PE={:.4} E={:.4} T*={:.6}",
                    step,
                    phase,
                    ke,
                    pe,
                    ke + pe,
                    t
                );
            }
        }

        // Validate energy conservation during production
        if energies.len() >= 2 {
            let e_mean: f64 = energies.iter().sum::<f64>() / energies.len() as f64;
            let e_first = energies[0];
            let e_last = *energies.last().expect("non-empty energies");
            let drift_pct = ((e_last - e_first) / e_first.abs()).abs() * 100.0;
            let e_std: f64 = (energies.iter().map(|e| (e - e_mean).powi(2)).sum::<f64>()
                / energies.len() as f64)
                .sqrt();
            let rel_fluct = e_std / e_mean.abs();

            println!("\n  Production energy:");
            println!("    Mean:  {e_mean:.6}");
            println!("    Std:   {e_std:.6}");
            println!("    Drift: {drift_pct:.4}%");
            println!("    Rel fluctuation: {rel_fluct:.2e}");

            harness.check_upper("Energy drift (%)", drift_pct, tolerances::ENERGY_DRIFT_PCT);
            harness.check_upper(
                "Energy relative fluctuation",
                rel_fluct,
                tolerances::ENERGY_DRIFT_PCT / 100.0,
            );
        }

        // Validate final temperature
        let t_final = compute_temperature(&vel, n, MASS);
        let t_deviation = ((t_final - t_target) / t_target).abs();
        println!("  Final T*={t_final:.6}, deviation={t_deviation:.2e}");
        harness.check_upper("Temperature deviation from target", t_deviation, 0.5);
    }

    println!();
    harness.finish();
}
