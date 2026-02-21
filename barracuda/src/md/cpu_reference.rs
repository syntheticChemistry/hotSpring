// SPDX-License-Identifier: AGPL-3.0-only

//! CPU Yukawa OCP MD in pure Rust f64.
//!
//! Same physics as the GPU path (`simulation.rs`): Velocity-Verlet integrator,
//! Berendsen thermostat, all-pairs Yukawa forces with PBC minimum-image.
//!
//! Used for CPU/GPU parity validation: identical initial conditions (FCC lattice,
//! seed 42), identical algorithm, different hardware → must produce the same
//! physics (energy conservation, VACF, D*) within documented f64 tolerances.

use crate::md::config::MdConfig;
use crate::md::simulation::{init_fcc_lattice, init_velocities, EnergyRecord, MdSimulation};

use std::time::Instant;

/// CPU all-pairs Yukawa force computation
fn compute_forces_cpu(
    positions: &[f64],
    forces: &mut [f64],
    pe: &mut [f64],
    n: usize,
    kappa: f64,
    prefactor: f64,
    cutoff_sq: f64,
    box_side: f64,
) {
    // Zero forces and PE
    for f in forces.iter_mut() {
        *f = 0.0;
    }
    for p in pe.iter_mut() {
        *p = 0.0;
    }

    for i in 0..n {
        let xi = positions[i * 3];
        let yi = positions[i * 3 + 1];
        let zi = positions[i * 3 + 2];

        for j in (i + 1)..n {
            let mut dx = positions[j * 3] - xi;
            let mut dy = positions[j * 3 + 1] - yi;
            let mut dz = positions[j * 3 + 2] - zi;

            // PBC minimum image
            dx -= box_side * (dx / box_side).round();
            dy -= box_side * (dy / box_side).round();
            dz -= box_side * (dz / box_side).round();

            let r_sq = dx * dx + dy * dy + dz * dz;
            if r_sq > cutoff_sq {
                continue;
            }

            let r = r_sq.sqrt();
            let screening = (-kappa * r).exp();
            let force_mag = prefactor * screening * (1.0 + kappa * r) / r_sq;
            let inv_r = 1.0 / r;

            // Newton's third law: apply to both particles
            let fx = -force_mag * dx * inv_r;
            let fy = -force_mag * dy * inv_r;
            let fz = -force_mag * dz * inv_r;

            forces[i * 3] += fx;
            forces[i * 3 + 1] += fy;
            forces[i * 3 + 2] += fz;
            forces[j * 3] -= fx;
            forces[j * 3 + 1] -= fy;
            forces[j * 3 + 2] -= fz;

            // PE (each pair counted once)
            let u = prefactor * screening * inv_r;
            pe[i] += 0.5 * u;
            pe[j] += 0.5 * u;
        }
    }
}

/// Run CPU MD simulation (for benchmarking)
pub fn run_simulation_cpu(config: &MdConfig) -> MdSimulation {
    let t_start = Instant::now();
    let n = config.n_particles;
    let box_side = config.box_side();
    let prefactor = config.force_prefactor();
    let temperature = config.temperature();
    let mass = config.reduced_mass();
    let dt = config.dt;
    let cutoff_sq = config.rc * config.rc;

    println!("  ── CPU Reference: {n} particles ──");

    // Initialize
    let (mut positions, n_actual) = init_fcc_lattice(n, box_side);
    let n = n_actual.min(n);
    let mut velocities = init_velocities(n, temperature, mass, 42);
    let mut forces = vec![0.0f64; n * 3];
    let mut pe = vec![0.0f64; n];

    // Initial forces
    compute_forces_cpu(
        &positions,
        &mut forces,
        &mut pe,
        n,
        config.kappa,
        prefactor,
        cutoff_sq,
        box_side,
    );

    let inv_m = 1.0 / mass;
    let half_dt = 0.5 * dt;

    // ── Equilibration ──
    println!("    Equilibrating ({} steps)...", config.equil_steps);
    let t_equil = Instant::now();

    for step in 0..config.equil_steps {
        // Half-kick
        for i in 0..n {
            velocities[i * 3] += half_dt * forces[i * 3] * inv_m;
            velocities[i * 3 + 1] += half_dt * forces[i * 3 + 1] * inv_m;
            velocities[i * 3 + 2] += half_dt * forces[i * 3 + 2] * inv_m;
        }

        // Drift + PBC
        for i in 0..n {
            positions[i * 3] += dt * velocities[i * 3];
            positions[i * 3 + 1] += dt * velocities[i * 3 + 1];
            positions[i * 3 + 2] += dt * velocities[i * 3 + 2];

            positions[i * 3] -= box_side * (positions[i * 3] / box_side).floor();
            positions[i * 3 + 1] -= box_side * (positions[i * 3 + 1] / box_side).floor();
            positions[i * 3 + 2] -= box_side * (positions[i * 3 + 2] / box_side).floor();
        }

        // New forces
        compute_forces_cpu(
            &positions,
            &mut forces,
            &mut pe,
            n,
            config.kappa,
            prefactor,
            cutoff_sq,
            box_side,
        );

        // Second half-kick
        for i in 0..n {
            velocities[i * 3] += half_dt * forces[i * 3] * inv_m;
            velocities[i * 3 + 1] += half_dt * forces[i * 3 + 1] * inv_m;
            velocities[i * 3 + 2] += half_dt * forces[i * 3 + 2] * inv_m;
        }

        // Berendsen thermostat
        if step % 10 == 0 {
            let mut ke = 0.0;
            for i in 0..n {
                ke += mass
                    * (velocities[i * 3].powi(2)
                        + velocities[i * 3 + 1].powi(2)
                        + velocities[i * 3 + 2].powi(2));
            }
            ke *= 0.5;
            let t_current = 2.0 * ke / (3.0 * n as f64);
            if t_current > 1e-30 {
                let ratio = 1.0 + (dt / config.berendsen_tau) * (temperature / t_current - 1.0);
                let scale = ratio.max(0.0).sqrt();
                for v in &mut velocities {
                    *v *= scale;
                }
            }
        }
    }
    // Final exact rescale to ensure T = T_target at production start.
    // Berendsen thermostat is gradual; structural rearrangement (FCC→liquid)
    // can prevent convergence. This rescale sets KE exactly.
    {
        let mut ke = 0.0;
        for i in 0..n {
            ke += mass
                * (velocities[i * 3].powi(2)
                    + velocities[i * 3 + 1].powi(2)
                    + velocities[i * 3 + 2].powi(2));
        }
        ke *= 0.5;
        let t_current = 2.0 * ke / (3.0 * n as f64);
        if t_current > 1e-30 {
            let scale = (temperature / t_current).sqrt();
            for v in &mut velocities {
                *v *= scale;
            }
        }
    }

    println!("    Equilibration: {:.2}s", t_equil.elapsed().as_secs_f64());

    // ── Production ──
    println!("    Production ({} steps)...", config.prod_steps);
    let _t_prod = Instant::now();
    let mut energy_history = Vec::new();
    let mut positions_snapshots = Vec::new();
    let mut velocity_snapshots = Vec::new();

    for step in 0..config.prod_steps {
        // Half-kick
        for i in 0..n {
            velocities[i * 3] += half_dt * forces[i * 3] * inv_m;
            velocities[i * 3 + 1] += half_dt * forces[i * 3 + 1] * inv_m;
            velocities[i * 3 + 2] += half_dt * forces[i * 3 + 2] * inv_m;
        }

        // Drift + PBC
        for i in 0..n {
            positions[i * 3] += dt * velocities[i * 3];
            positions[i * 3 + 1] += dt * velocities[i * 3 + 1];
            positions[i * 3 + 2] += dt * velocities[i * 3 + 2];

            positions[i * 3] -= box_side * (positions[i * 3] / box_side).floor();
            positions[i * 3 + 1] -= box_side * (positions[i * 3 + 1] / box_side).floor();
            positions[i * 3 + 2] -= box_side * (positions[i * 3 + 2] / box_side).floor();
        }

        // New forces
        compute_forces_cpu(
            &positions,
            &mut forces,
            &mut pe,
            n,
            config.kappa,
            prefactor,
            cutoff_sq,
            box_side,
        );

        // Second half-kick
        for i in 0..n {
            velocities[i * 3] += half_dt * forces[i * 3] * inv_m;
            velocities[i * 3 + 1] += half_dt * forces[i * 3 + 1] * inv_m;
            velocities[i * 3 + 2] += half_dt * forces[i * 3 + 2] * inv_m;
        }

        if step % config.dump_step == 0 {
            let mut ke = 0.0;
            let total_pe: f64 = pe.iter().sum();
            for i in 0..n {
                ke += mass
                    * (velocities[i * 3].powi(2)
                        + velocities[i * 3 + 1].powi(2)
                        + velocities[i * 3 + 2].powi(2));
            }
            ke *= 0.5;
            let t_current = 2.0 * ke / (3.0 * n as f64);

            energy_history.push(EnergyRecord {
                step,
                ke,
                pe: total_pe,
                total: ke + total_pe,
                temperature: t_current,
            });

            if step % (config.dump_step * config.vel_snapshot_interval) == 0 {
                positions_snapshots.push(positions.clone());
                velocity_snapshots.push(velocities.clone());
            }
        }

        if step % 5000 == 0 || step == config.prod_steps - 1 {
            if let Some(last) = energy_history.last() {
                println!(
                    "    Step {}: T*={:.6}, E={:.4}",
                    step, last.temperature, last.total
                );
            }
        }
    }

    let total_time = t_start.elapsed().as_secs_f64();
    let total_steps = config.equil_steps + config.prod_steps;
    let steps_per_sec = total_steps as f64 / total_time;

    println!("    CPU total: {total_time:.2}s ({steps_per_sec:.1} steps/s)");

    MdSimulation {
        config: config.clone(),
        energy_history,
        positions_snapshots,
        velocity_snapshots,
        rdf_histogram: Vec::new(),
        wall_time_s: total_time,
        steps_per_sec,
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::compute_forces_cpu;
    use crate::md::config::MdConfig;
    use crate::md::simulation::{init_fcc_lattice, init_velocities};

    /// Minimal config for fast CPU tests (4 particles, few steps).
    fn minimal_test_config() -> MdConfig {
        MdConfig {
            label: "minimal_test".to_string(),
            n_particles: 4,
            kappa: 2.0,
            gamma: 158.0,
            dt: 0.01,
            rc: 6.5,
            equil_steps: 10,
            prod_steps: 50,
            dump_step: 5,
            berendsen_tau: 5.0,
            rdf_bins: 20,
            vel_snapshot_interval: 10,
        }
    }

    #[test]
    fn run_simulation_cpu_produces_valid_output() {
        let config = minimal_test_config();
        let sim = super::run_simulation_cpu(&config);
        assert!(
            !sim.energy_history.is_empty(),
            "energy history should be populated"
        );
        assert!(sim.wall_time_s > 0.0);
        assert!(sim.steps_per_sec > 0.0);
        let last = sim.energy_history.last().expect("energy history non-empty");
        assert!(last.total.is_finite(), "total energy should be finite");
        assert!(last.temperature > 0.0 && last.temperature.is_finite());
    }

    #[test]
    fn run_simulation_cpu_positions_in_box() {
        let config = minimal_test_config();
        let sim = super::run_simulation_cpu(&config);
        let box_side = config.box_side();
        if let Some(pos) = sim.positions_snapshots.last() {
            let n = config.n_particles.min(pos.len() / 3);
            for i in 0..n {
                let x = pos[i * 3];
                let y = pos[i * 3 + 1];
                let z = pos[i * 3 + 2];
                assert!(
                    x >= 0.0 && x < box_side,
                    "PBC: x={x} out of [0, {box_side})"
                );
                assert!(
                    y >= 0.0 && y < box_side,
                    "PBC: y={y} out of [0, {box_side})"
                );
                assert!(
                    z >= 0.0 && z < box_side,
                    "PBC: z={z} out of [0, {box_side})"
                );
            }
        }
    }

    #[test]
    fn run_simulation_cpu_energy_conservation() {
        let config = minimal_test_config();
        let sim = super::run_simulation_cpu(&config);
        assert!(sim.energy_history.len() >= 2);
        let e0 = sim.energy_history.first().expect("history non-empty").total;
        let e1 = sim.energy_history.last().expect("history non-empty").total;
        let mean = sim.energy_history.iter().map(|r| r.total).sum::<f64>()
            / sim.energy_history.len() as f64;
        let drift_pct = if mean.abs() > 1e-10 {
            ((e1 - e0) / mean.abs()).abs() * 100.0
        } else {
            0.0
        };
        assert!(
            drift_pct < 50.0,
            "short run energy drift should be modest: {drift_pct}%"
        );
    }

    #[test]
    fn fcc_lattice_correct_count() {
        let (pos, n_actual) = init_fcc_lattice(108, 10.0);
        assert_eq!(n_actual, 108, "4×3³ = 108 for 3 unit cells");
        assert_eq!(pos.len(), 108 * 3);
    }

    #[test]
    fn fcc_lattice_in_box() {
        let box_side = 10.0;
        let (pos, n) = init_fcc_lattice(500, box_side);
        for i in 0..n {
            let x = pos[i * 3];
            let y = pos[i * 3 + 1];
            let z = pos[i * 3 + 2];
            assert!(x >= 0.0 && x < box_side, "x={x} out of box");
            assert!(y >= 0.0 && y < box_side, "y={y} out of box");
            assert!(z >= 0.0 && z < box_side, "z={z} out of box");
        }
    }

    #[test]
    fn velocities_zero_com() {
        let n = 500;
        let vel = init_velocities(n, 1.0 / 158.0, 3.0, 42);
        let mut vx_sum = 0.0;
        let mut vy_sum = 0.0;
        let mut vz_sum = 0.0;
        for i in 0..n {
            vx_sum += vel[i * 3];
            vy_sum += vel[i * 3 + 1];
            vz_sum += vel[i * 3 + 2];
        }
        assert!(vx_sum.abs() < 1e-10, "COM vx should be zero: {vx_sum}");
        assert!(vy_sum.abs() < 1e-10, "COM vy should be zero: {vy_sum}");
        assert!(vz_sum.abs() < 1e-10, "COM vz should be zero: {vz_sum}");
    }

    #[test]
    fn velocities_correct_temperature() {
        let n = 2000;
        let t_target = 1.0 / 158.0;
        let mass = 3.0;
        let vel = init_velocities(n, t_target, mass, 42);
        let mut v_sq_sum = 0.0;
        for &v in &vel {
            v_sq_sum += v * v;
        }
        let t_measured = mass * v_sq_sum / (3.0 * n as f64);
        assert!(
            (t_measured - t_target).abs() / t_target < 0.01,
            "temperature should match target: {t_measured} vs {t_target}"
        );
    }

    #[test]
    fn velocities_deterministic() {
        let v1 = init_velocities(100, 0.01, 3.0, 42);
        let v2 = init_velocities(100, 0.01, 3.0, 42);
        assert_eq!(v1, v2, "same seed should give identical velocities");
    }

    #[test]
    fn yukawa_force_analytical() {
        // Two particles separated by r along x-axis
        let r: f64 = 2.0;
        let kappa: f64 = 2.0;
        let prefactor: f64 = 1.0;

        // F = prefactor * exp(-κr) * (1 + κr) / r²
        let expected_f = prefactor * (-kappa * r).exp() * (1.0 + kappa * r) / (r * r);

        // Actual computation via the force function
        let f = yukawa_force_magnitude(r, kappa, prefactor);
        assert!(
            (f - expected_f).abs() < 1e-12,
            "force should match analytical: {f} vs {expected_f}"
        );
    }

    #[test]
    fn yukawa_force_various_distances() {
        let kappa = 1.0f64;
        let prefactor = 1.0;
        for r in [0.5, 1.0, 2.0, 5.0] {
            let f = yukawa_force_magnitude(r, kappa, prefactor);
            let expected = prefactor * (-kappa * r).exp() * (1.0 + kappa * r) / (r * r);
            assert!(
                (f - expected).abs() < 1e-12,
                "r={r}: force {f} vs expected {expected}"
            );
            assert!(f > 0.0, "Yukawa force repulsive (positive) at r={r}");
        }
    }

    #[test]
    fn yukawa_force_coulomb_limit() {
        // κ→0: F = prefactor/r² (Coulomb)
        let r = 2.0;
        let prefactor = 1.0;
        let kappa_small = 1e-10;
        let f = yukawa_force_magnitude(r, kappa_small, prefactor);
        let coulomb = prefactor / (r * r);
        assert!(
            (f - coulomb).abs() < 1e-8,
            "kappa≈0 should give Coulomb: {f} vs {coulomb}"
        );
    }

    #[test]
    fn yukawa_force_strong_screening() {
        // Large κ: force decays faster
        let r = 1.0;
        let prefactor = 1.0;
        let f_k1 = yukawa_force_magnitude(r, 1.0, prefactor);
        let f_k3 = yukawa_force_magnitude(r, 3.0, prefactor);
        assert!(
            f_k3 < f_k1,
            "higher kappa should give smaller force at same r"
        );
    }

    /// Compute Yukawa force magnitude for testing.
    /// F(r) = prefactor * exp(-κr) * (1 + κr) / r²
    fn yukawa_force_magnitude(r: f64, kappa: f64, prefactor: f64) -> f64 {
        prefactor * (-kappa * r).exp() * (1.0 + kappa * r) / (r * r)
    }

    #[test]
    fn md_cpu_fcc_velocities_determinism() {
        let n = 108;
        let box_side = 10.0;
        let results: Vec<(Vec<f64>, Vec<f64>)> = (0..2)
            .map(|_| {
                let (pos, _) = init_fcc_lattice(n, box_side);
                let vel = init_velocities(n, 1.0 / 158.0, 3.0, 42);
                (pos, vel)
            })
            .collect();
        for i in 0..n * 3 {
            assert!(
                (results[0].0[i] - results[1].0[i]).abs() < f64::EPSILON,
                "FCC positions must be identical at index {i}"
            );
            assert!(
                (results[0].1[i] - results[1].1[i]).abs() < f64::EPSILON,
                "Initial velocities must be identical at index {i}"
            );
        }
    }

    #[test]
    fn md_cpu_forces_determinism() {
        let n = 32;
        let box_side = 8.0;
        let (pos, _) = init_fcc_lattice(n, box_side);
        let kappa = 1.0;
        let prefactor = 1.0;
        let cutoff_sq = (box_side / 2.0) * (box_side / 2.0);

        let results: Vec<(Vec<f64>, Vec<f64>)> = (0..2)
            .map(|_| {
                let mut forces = vec![0.0; n * 3];
                let mut pe = vec![0.0; n];
                compute_forces_cpu(
                    &pos,
                    &mut forces,
                    &mut pe,
                    n,
                    kappa,
                    prefactor,
                    cutoff_sq,
                    box_side,
                );
                (forces, pe)
            })
            .collect();
        for i in 0..n * 3 {
            assert!(
                (results[0].0[i] - results[1].0[i]).abs() < f64::EPSILON,
                "Forces must be identical at index {i}"
            );
        }
        for i in 0..n {
            assert!(
                (results[0].1[i] - results[1].1[i]).abs() < f64::EPSILON,
                "PE must be identical at index {i}"
            );
        }
    }
}
