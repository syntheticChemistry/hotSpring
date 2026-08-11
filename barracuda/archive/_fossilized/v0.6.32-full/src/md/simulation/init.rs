// SPDX-License-Identifier: AGPL-3.0-or-later

//! Particle initialization: FCC lattice placement and Maxwell-Boltzmann velocities.

use crate::tolerances::MD_BOX_MULLER_FLOOR;
use std::f64::consts::PI;

/// Initialize particles on an FCC lattice, then add Maxwell-Boltzmann velocities
#[must_use]
pub fn init_fcc_lattice(n: usize, box_side: f64) -> (Vec<f64>, usize) {
    // Find smallest n_cell such that 4*n_cell^3 >= n (FCC has 4 atoms per unit cell)
    let mut n_cell = 1usize;
    while 4 * n_cell * n_cell * n_cell < n {
        n_cell += 1;
    }
    let n_actual = 4 * n_cell * n_cell * n_cell;
    let a = box_side / n_cell as f64; // lattice constant

    // FCC basis vectors (in units of lattice constant a)
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];

    let mut positions = Vec::with_capacity(n_actual * 3);
    for ix in 0..n_cell {
        for iy in 0..n_cell {
            for iz in 0..n_cell {
                for b in &basis {
                    let x = (ix as f64 + b[0]) * a;
                    let y = (iy as f64 + b[1]) * a;
                    let z = (iz as f64 + b[2]) * a;
                    positions.push(x);
                    positions.push(y);
                    positions.push(z);
                }
            }
        }
    }

    // Trim to exactly n particles if n_actual > n
    positions.truncate(n * 3);
    (positions, n.min(n_actual))
}

/// Generate Maxwell-Boltzmann velocities for given temperature
/// T* = 1/Gamma in reduced units.  KE = (3/2) N T*
/// Per component: ⟨`v²_α`⟩ = `T*`/`m*`  →  σ = sqrt(`T*`/`m*`)
#[must_use]
pub fn init_velocities(n: usize, temperature: f64, mass: f64, seed: u64) -> Vec<f64> {
    // Simple Box-Muller PRNG for Gaussian distribution
    let sigma = (temperature / mass).sqrt(); // sqrt(T*/m*) per component
    let mut velocities = Vec::with_capacity(n * 3);

    // LCG random number generator (good enough for initialization)
    let mut rng_state = seed;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 33) as f64 / (1u64 << 31) as f64
    };

    for _ in 0..n {
        for _ in 0..3 {
            // Box-Muller transform
            let u1 = lcg_next(&mut rng_state).max(MD_BOX_MULLER_FLOOR);
            let u2 = lcg_next(&mut rng_state);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            velocities.push(z * sigma);
        }
    }

    // Remove center-of-mass velocity
    let mut vx_sum = 0.0;
    let mut vy_sum = 0.0;
    let mut vz_sum = 0.0;
    for i in 0..n {
        vx_sum += velocities[i * 3];
        vy_sum += velocities[i * 3 + 1];
        vz_sum += velocities[i * 3 + 2];
    }
    let n_f = n as f64;
    for i in 0..n {
        velocities[i * 3] -= vx_sum / n_f;
        velocities[i * 3 + 1] -= vy_sum / n_f;
        velocities[i * 3 + 2] -= vz_sum / n_f;
    }

    // Rescale to exact target temperature
    // KE = 0.5 * m * sum(v²), T = 2*KE/(3N) = m*sum(v²)/(3N)
    let mut v_sq_sum = 0.0;
    for v in &velocities {
        v_sq_sum += v * v;
    }
    let t_current = mass * v_sq_sum / (3.0 * n_f);
    let scale = (temperature / t_current).sqrt();
    for v in &mut velocities {
        *v *= scale;
    }

    velocities
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcc_lattice_n4_produces_4_positions() {
        let (positions, n_actual) = init_fcc_lattice(4, 1.0);
        assert_eq!(n_actual, 4);
        assert_eq!(positions.len(), 12);
    }

    #[test]
    fn fcc_lattice_n32_produces_32_positions() {
        let (positions, n_actual) = init_fcc_lattice(32, 10.0);
        assert_eq!(n_actual, 32);
        assert_eq!(positions.len(), 96);
    }

    #[test]
    fn fcc_lattice_positions_in_box() {
        let box_side = 5.0;
        let (positions, n_actual) = init_fcc_lattice(32, box_side);
        for i in 0..n_actual {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            assert!(x >= 0.0 && x < box_side, "x={x} out of box [0, {box_side})");
            assert!(y >= 0.0 && y < box_side, "y={y} out of box [0, {box_side})");
            assert!(z >= 0.0 && z < box_side, "z={z} out of box [0, {box_side})");
        }
    }

    #[test]
    fn fcc_lattice_no_overlapping_positions() {
        let (positions, n_actual) = init_fcc_lattice(32, 10.0);
        let min_dist_sq = 0.01_f64;
        for i in 0..n_actual {
            for j in (i + 1)..n_actual {
                let dx = positions[i * 3] - positions[j * 3];
                let dy = positions[i * 3 + 1] - positions[j * 3 + 1];
                let dz = positions[i * 3 + 2] - positions[j * 3 + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                assert!(r2 > min_dist_sq, "particles {i} and {j} overlap at r²={r2}");
            }
        }
    }

    #[test]
    fn velocities_zero_center_of_mass() {
        let n = 64;
        let velocities = init_velocities(n, 1.0, 1.0, 42);
        assert_eq!(velocities.len(), n * 3);

        let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
        for i in 0..n {
            vx += velocities[i * 3];
            vy += velocities[i * 3 + 1];
            vz += velocities[i * 3 + 2];
        }
        assert!(
            (vx / n as f64).abs() < 1e-12,
            "CoM vx should be ~0, got {}",
            vx / n as f64
        );
        assert!(
            (vy / n as f64).abs() < 1e-12,
            "CoM vy should be ~0, got {}",
            vy / n as f64
        );
        assert!(
            (vz / n as f64).abs() < 1e-12,
            "CoM vz should be ~0, got {}",
            vz / n as f64
        );
    }

    #[test]
    fn velocities_correct_temperature() {
        let n = 256;
        let target_temp = 1.0;
        let mass = 1.0;
        let velocities = init_velocities(n, target_temp, mass, 42);

        let v_sq_sum: f64 = velocities.iter().map(|v| v * v).sum();
        let measured_temp = mass * v_sq_sum / (3.0 * n as f64);
        assert!(
            (measured_temp - target_temp).abs() < 1e-10,
            "T should be {target_temp}, got {measured_temp}"
        );
    }

    #[test]
    fn velocities_deterministic() {
        let v1 = init_velocities(32, 1.0, 1.0, 42);
        let v2 = init_velocities(32, 1.0, 1.0, 42);
        assert_eq!(v1, v2, "same seed should produce same velocities");
    }

    #[test]
    fn velocities_different_seeds_differ() {
        let v1 = init_velocities(32, 1.0, 1.0, 42);
        let v2 = init_velocities(32, 1.0, 1.0, 99);
        assert_ne!(
            v1, v2,
            "different seeds should produce different velocities"
        );
    }
}
