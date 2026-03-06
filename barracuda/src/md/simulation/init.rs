// SPDX-License-Identifier: AGPL-3.0-only

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
