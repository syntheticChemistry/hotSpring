// SPDX-License-Identifier: AGPL-3.0-only

//! Centralized constants for lattice field theory modules.
//!
//! Collects LCG PRNG parameters, SU(3) color constants, and numerical
//! guards used across `su3.rs`, `wilson.rs`, `hmc.rs`, `dirac.rs`, and `cg.rs`.

/// Number of colors in QCD (SU(3)).
pub const N_COLORS: usize = 3;

/// Number of spacetime dimensions.
pub const N_DIM: usize = 4;

/// LCG multiplier (Knuth MMIX).
///
/// Used for deterministic pseudo-random number generation in lattice
/// initialization (hot start) and HMC momentum refresh.
pub const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// LCG increment (Knuth MMIX).
pub const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;

/// Mantissa bits for LCG → uniform [0, 1) conversion.
///
/// `(seed >> 33) as f64 / (1u64 << 31) as f64` gives 31 bits of precision.
/// For Box-Muller, `(seed >> 11) as f64 / (1u64 << 53) as f64` gives 53 bits.
pub const LCG_53_DIVISOR: f64 = (1u64 << 53) as f64;

/// Division guard for lattice CG/reunitarization.
///
/// Prevents division by zero in vector norms, inner products, and
/// Gram-Schmidt orthonormalization. Well below any physical lattice scale.
pub const LATTICE_DIVISION_GUARD: f64 = 1e-30;

/// Hot-start perturbation scale for SU(3) link matrices.
///
/// Controls the magnitude of random anti-Hermitian perturbation added
/// to identity matrices during hot start. 1.5 gives a well-distributed
/// initial configuration that thermalizes quickly.
pub const HOT_START_EPSILON: f64 = 1.5;

/// Advance the LCG state by one step.
#[inline]
pub fn lcg_step(seed: &mut u64) {
    *seed = seed
        .wrapping_mul(LCG_MULTIPLIER)
        .wrapping_add(LCG_INCREMENT);
}

/// Generate a uniform f64 in [0, 1) from 53 bits of LCG state.
#[inline]
pub fn lcg_uniform_f64(seed: &mut u64) -> f64 {
    lcg_step(seed);
    (*seed >> 11) as f64 / LCG_53_DIVISOR
}

/// Box-Muller Gaussian deviate N(0, 1) from two LCG draws.
///
/// Uses the polar form: z = sqrt(-2 ln u1) cos(2π u2).
/// The `ln` argument is clamped to `LATTICE_DIVISION_GUARD` to avoid ln(0).
#[inline]
pub fn lcg_gaussian(seed: &mut u64) -> f64 {
    let u1 = lcg_uniform_f64(seed);
    let u2 = lcg_uniform_f64(seed);
    (-2.0 * u1.max(LATTICE_DIVISION_GUARD).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_step_deterministic() {
        let mut a = 42u64;
        let mut b = 42u64;
        lcg_step(&mut a);
        lcg_step(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn lcg_uniform_in_range() {
        let mut seed = 12345u64;
        for _ in 0..1000 {
            let v = lcg_uniform_f64(&mut seed);
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn n_colors_is_three() {
        assert_eq!(N_COLORS, 3);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn n_dim_is_four() {
        assert_eq!(N_DIM, 4);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // constants sanity check
    fn guards_are_positive() {
        assert!(LATTICE_DIVISION_GUARD > 0.0);
        assert!(HOT_START_EPSILON > 0.0);
    }

    #[test]
    fn lcg_gaussian_is_finite() {
        let mut seed = 99u64;
        for _ in 0..1000 {
            let g = lcg_gaussian(&mut seed);
            assert!(g.is_finite(), "Gaussian deviate must be finite: {g}");
        }
    }

    #[test]
    fn lcg_gaussian_mean_near_zero() {
        let mut seed = 42u64;
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| lcg_gaussian(&mut seed)).sum();
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "mean should be near 0, got {mean}");
    }
}
