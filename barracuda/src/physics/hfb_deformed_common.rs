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
    let r0 = 1.2 * a_f.cbrt();
    (5.0 * PI).sqrt() * q20 / (3.0 * a_f * r0 * r0)
}

/// RMS radius from density on a cylindrical grid.
///
/// Formula: `r_RMS` = √(∫ ρ·r² dV / ∫ ρ dV).
/// Grid layout: ρ = (`i_ρ` + 1)·`d_ρ`, z = `z_min` + (`i_z` + 0.5)·`d_z`, idx = `i_ρ`·`n_z` + `i_z`.
///
/// # Arguments
/// * `density` - Density at each grid point, length = `n_rho` × `n_z`
/// * `n_rho` - Number of radial points
/// * `n_z` - Number of axial points
/// * `d_rho` - Radial spacing (fm)
/// * `d_z` - Axial spacing (fm)
/// * `z_min` - z-coordinate of cell center at `i_z`=0 (fm)
///
/// # Panics
///
/// Panics if `density.len() != n_rho * n_z`.
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
            let z = (i_z as f64 + 0.5).mul_add(d_z, z_min);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerances::{
        DEFORMATION_GUESS_GENERIC, DEFORMATION_GUESS_SD, DEFORMATION_GUESS_WEAK,
    };

    #[test]
    #[allow(clippy::float_cmp)]
    fn doubly_magic_nuclei_are_spherical() {
        assert_eq!(deformation_guess(8, 8), 0.0); // O-16
        assert_eq!(deformation_guess(20, 20), 0.0); // Ca-40
        assert_eq!(deformation_guess(20, 28), 0.0); // Ca-48
        assert_eq!(deformation_guess(82, 126), 0.0); // Pb-208
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn actinides_are_well_deformed() {
        assert_eq!(deformation_guess(92, 146), 0.25); // U-238
        assert_eq!(deformation_guess(94, 150), 0.25); // Pu-244
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn rare_earths_strongly_deformed() {
        assert_eq!(deformation_guess(66, 96), 0.28); // Dy-162
        assert_eq!(deformation_guess(68, 98), 0.28); // Er-166
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn sd_shell_nuclei_deformed() {
        assert_eq!(deformation_guess(12, 12), DEFORMATION_GUESS_SD); // Mg-24
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn single_magic_weakly_deformed() {
        assert_eq!(deformation_guess(50, 60), DEFORMATION_GUESS_WEAK); // Sn-110, Z magic
        assert_eq!(deformation_guess(40, 50), DEFORMATION_GUESS_WEAK); // Zr-90, N magic
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn generic_nuclei_moderate_deformation() {
        assert_eq!(deformation_guess(40, 56), DEFORMATION_GUESS_GENERIC); // Zr-96
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn beta2_from_q20_zero_for_spherical() {
        assert_eq!(beta2_from_q20(16, 0.0), 0.0);
    }

    #[test]
    fn beta2_from_q20_sign_follows_q20() {
        let b_pos = beta2_from_q20(16, 100.0);
        let b_neg = beta2_from_q20(16, -100.0);
        assert!(b_pos > 0.0);
        assert!(b_neg < 0.0);
        assert!((b_pos + b_neg).abs() < 1e-12, "should be antisymmetric");
    }

    #[test]
    fn beta2_from_q20_known_scaling() {
        let b_16 = beta2_from_q20(16, 50.0);
        let b_208 = beta2_from_q20(208, 50.0);
        assert!(
            b_16.abs() > b_208.abs(),
            "lighter nucleus → larger β₂ for same Q20"
        );
    }

    #[test]
    fn rms_radius_uniform_density() {
        let n_rho = 10;
        let n_z = 10;
        let d_rho = 0.5;
        let d_z = 0.5;
        let z_min = -2.5;
        let density = vec![1.0; n_rho * n_z];
        let r = rms_radius(&density, n_rho, n_z, d_rho, d_z, z_min);
        assert!(r > 0.0, "RMS radius of uniform density must be positive");
        assert!(r.is_finite());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn rms_radius_zero_density_returns_zero() {
        let density = vec![0.0; 25];
        let r = rms_radius(&density, 5, 5, 1.0, 1.0, -2.5);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn rms_radius_single_point_density() {
        let n_rho = 5;
        let n_z = 5;
        let d_rho = 1.0;
        let d_z = 1.0;
        let z_min = -2.5;
        let mut density = vec![0.0; n_rho * n_z];
        // Place density at i_rho=2, i_z=2 → rho=3.0, z=-2.5+2.5=0.0
        density[2 * n_z + 2] = 1.0;
        let r = rms_radius(&density, n_rho, n_z, d_rho, d_z, z_min);
        assert!(
            (r - 3.0).abs() < 0.01,
            "single point at rho=3 → r_rms ≈ 3, got {r}"
        );
    }
}
