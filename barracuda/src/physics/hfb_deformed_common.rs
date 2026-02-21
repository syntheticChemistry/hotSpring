// SPDX-License-Identifier: AGPL-3.0-only

//! Shared physics kernels for deformed HFB (CPU and GPU paths).
//!
//! Extracts pure functions used identically in `hfb_deformed` and `hfb_deformed_gpu`:
//! - Nilsson-inspired deformation guess
//! - Beta₂ from quadrupole moment Q20
//! - RMS charge/matter radius from density

use crate::tolerances::{DEFORMATION_GUESS_GENERIC, DEFORMATION_GUESS_SD, DEFORMATION_GUESS_WEAK};
use std::f64::consts::PI;

/// Heuristic initial deformation β₂ based on nuclear chart (Nilsson model regions).
///
/// Formula regions:
/// - Doubly magic (Z,N near 2,8,20,28,50,82,126): β₂ = 0
/// - Actinides (A > 222): β₂ ≈ 0.25
/// - Rare earths (150 < A < 190): β₂ ≈ 0.28
/// - Light deformed sd-shell (20 < A < 28): β₂ ≈ 0.35
/// - Single magic: β₂ ≈ 0.05
/// - Generic: β₂ ≈ 0.15
#[must_use]
pub fn deformation_guess(z: usize, n: usize) -> f64 {
    let a = z + n;
    let magic = [2, 8, 20, 28, 50, 82, 126];
    let z_magic = magic.iter().any(|&m| (z as i32 - m).unsigned_abs() <= 2);
    let n_magic = magic.iter().any(|&m| (n as i32 - m).unsigned_abs() <= 2);

    if z_magic && n_magic {
        0.0 // doubly magic → spherical
    } else if a > 222 {
        0.25 // actinides
    } else if a > 150 && a < 190 {
        0.28 // rare earths
    } else if a > 20 && a < 28 {
        DEFORMATION_GUESS_SD
    } else if z_magic || n_magic {
        DEFORMATION_GUESS_WEAK
    } else {
        DEFORMATION_GUESS_GENERIC
    }
}

/// Deformation parameter β₂ from quadrupole moment Q₂₀.
///
/// Formula: β₂ = √(5π) · Q₂₀ / (3 · A · R₀²), with R₀ = 1.2 · A^(1/3) fm.
#[must_use]
pub fn beta2_from_q20(a: usize, q20: f64) -> f64 {
    let a_f = a as f64;
    let r0 = 1.2 * a_f.powf(1.0 / 3.0);
    (5.0 * PI).sqrt() * q20 / (3.0 * a_f * r0 * r0)
}

/// RMS radius from density on a cylindrical grid.
///
/// Formula: r_RMS = √(∫ ρ·r² dV / ∫ ρ dV).
/// Grid layout: ρ = (i_ρ + 1)·dρ, z = z_min + (i_z + 0.5)·dz, idx = i_ρ·n_z + i_z.
///
/// # Arguments
/// * `density` - Density at each grid point, length = n_rho × n_z
/// * `n_rho` - Number of radial points
/// * `n_z` - Number of axial points
/// * `d_rho` - Radial spacing (fm)
/// * `d_z` - Axial spacing (fm)
/// * `z_min` - z-coordinate of cell center at i_z=0 (fm)
#[must_use]
pub fn rms_radius(
    density: &[f64],
    n_rho: usize,
    n_z: usize,
    d_rho: f64,
    d_z: f64,
    z_min: f64,
) -> f64 {
    assert_eq!(density.len(), n_rho * n_z);

    let mut sum_r2 = 0.0;
    let mut sum_rho = 0.0;
    for i_rho in 0..n_rho {
        let rho = (i_rho + 1) as f64 * d_rho;
        let dv = 2.0 * PI * rho * d_rho * d_z;
        for i_z in 0..n_z {
            let z = z_min + (i_z as f64 + 0.5) * d_z;
            let r2 = rho * rho + z * z;
            let idx = i_rho * n_z + i_z;
            sum_r2 += density[idx] * r2 * dv;
            sum_rho += density[idx] * dv;
        }
    }
    if sum_rho > 0.0 {
        (sum_r2 / sum_rho).sqrt()
    } else {
        0.0
    }
}
