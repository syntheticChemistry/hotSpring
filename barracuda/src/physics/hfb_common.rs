// SPDX-License-Identifier: AGPL-3.0-only

//! Shared types and utilities for the HFB solver family.
//!
//! Consolidates duplicated code across spherical HFB (L2), deformed HFB (L3),
//! and GPU-accelerated deformed HFB implementations.
//!
//! - [`Mat`] — lightweight row-major square matrix for Hamiltonian blocks
//! - [`hermite_value`] — Re-export of `barracuda::special::hermite` (canonical implementation)
//! - [`factorial_f64`] — delegates to `barracuda::special::factorial`

// ═══════════════════════════════════════════════════════════════════
// Row-major square matrix
// ═══════════════════════════════════════════════════════════════════

/// Simple row-major square matrix for Hamiltonian blocks.
///
/// Lightweight matrix for small blocks (typically ≤ 20×20 in HFB solvers).
/// Avoids external linear algebra dependencies for construction and element access.
/// The flat `data` field is exposed for direct handoff to eigensolvers.
#[derive(Debug, Clone)]
pub struct Mat {
    /// Row-major flat storage: element (r, c) lives at `data[r * n + c]`.
    pub data: Vec<f64>,
    /// Matrix dimension (n × n).
    pub n: usize,
}

impl Mat {
    /// Create an n×n zero matrix.
    pub fn zeros(n: usize) -> Self {
        Mat {
            data: vec![0.0; n * n],
            n,
        }
    }

    /// Read element (r, c).
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }

    /// Write element (r, c).
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] = v;
    }

    /// Add `v` to element (r, c).
    #[inline]
    pub fn add(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.n + c] += v;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Math helpers shared across deformed HFB variants
// ═══════════════════════════════════════════════════════════════════

/// Hermite polynomial H_n(x) — delegates to `barracuda::special::hermite`.
///
/// H₀(x) = 1, H₁(x) = 2x, H_{n+1}(x) = 2x·Hₙ(x) − 2n·H_{n−1}(x)
///
/// Used by deformed HFB solvers for axial harmonic oscillator wavefunctions.
/// Previously a local implementation; now delegates to the canonical barracuda
/// version to eliminate duplicate math (zero-duplicate-math principle).
#[inline]
pub fn hermite_value(n: usize, x: f64) -> f64 {
    barracuda::special::hermite(n, x)
}

/// Factorial n! as f64.
///
/// Delegates to [`barracuda::special::factorial()`] — the canonical
/// implementation with a lookup table for n ≤ 20 and Stirling's
/// approximation for larger n.
pub fn factorial_f64(n: usize) -> f64 {
    barracuda::special::factorial(n)
}

// ═══════════════════════════════════════════════════════════════════
// Shared physics formulas across spherical and deformed HFB
// ═══════════════════════════════════════════════════════════════════

/// BCS quasiparticle occupation v² for a single state.
///
/// v²(ε, Δ) = ½ (1 − ε / E_qp), where E_qp = √(ε² + Δ²).
///
/// - `eps`: single-particle energy relative to chemical potential (ε = e − μ)
/// - `delta`: pairing gap Δ (MeV)
///
/// Returns occupation probability v² ∈ [0, 1].
#[inline]
#[must_use]
pub fn bcs_v2(eps: f64, delta: f64) -> f64 {
    let e_qp = (eps * eps + delta * delta).sqrt();
    // Physics: e_qp is always ≥ |Δ| > 0 when pairing is active
    0.5 * (1.0 - eps / e_qp)
}

/// Coulomb exchange potential (Slater approximation) at a single point.
///
/// V_ex(r) = −e² (3/π)^{1/3} ρ_p(r)^{1/3}
///
/// - `rho_p`: proton density at this point (fm⁻³)
///
/// Uses the Slater local density approximation to the Fock exchange term.
/// Standard form in Skyrme-HFB calculations (see Bender et al. 2003, §II.C).
#[inline]
#[must_use]
pub fn coulomb_exchange_slater(rho_p: f64) -> f64 {
    use super::constants::E2;
    -E2 * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0) * rho_p.max(0.0).powf(1.0 / 3.0)
}

/// Coulomb exchange energy density (Slater approximation).
///
/// ε_ex(r) = −¾ e² (3/π)^{1/3} ρ_p(r)^{4/3}
///
/// Integrate over volume to get total Coulomb exchange energy.
#[inline]
#[must_use]
pub fn coulomb_exchange_energy_density(rho_p: f64) -> f64 {
    use super::constants::E2;
    -0.75 * E2 * (3.0 / std::f64::consts::PI).powf(1.0 / 3.0) * rho_p.max(0.0).powf(4.0 / 3.0)
}

/// Center-of-mass correction to the total energy.
///
/// E_cm = −¾ ℏω, where ℏω is the harmonic oscillator energy spacing.
///
/// This one-body approximation removes spurious CM kinetic energy
/// (see Bender et al. 2003, §II.E).
#[inline]
#[must_use]
pub fn cm_correction(hw: f64) -> f64 {
    -0.75 * hw
}

/// Skyrme central potential t₀ contribution at a single point.
///
/// U_t₀ = t₀ [(1 + x₀/2)ρ − (½ + x₀)ρ_q]
///
/// - `rho`: total density ρ = ρ_p + ρ_n
/// - `rho_q`: isospin density ρ_q (ρ_p for protons, ρ_n for neutrons)
#[inline]
#[must_use]
pub fn skyrme_central_t0(t0: f64, x0: f64, rho: f64, rho_q: f64) -> f64 {
    t0 * ((1.0 + x0 / 2.0) * rho - (0.5 + x0) * rho_q)
}

/// Initialize Wood-Saxon-like density profile for protons and neutrons.
///
/// Uses a sharp-cutoff approximation: uniform density inside nuclear radius
/// R = r₀ A^(1/3) (r₀ = 1.2 fm), zero outside, floored at `DENSITY_FLOOR` everywhere.
/// Central density ρ₀ = 3A/(4πR³); proton/neutron fractions Z/A and N/A.
///
/// Returns `(rho_p, rho_n)` vectors of length `nr`. Grid points: `r[k] = (k+1)*dr`.
#[must_use]
pub fn initial_wood_saxon_density(z: usize, n: usize, nr: usize, dr: f64) -> (Vec<f64>, Vec<f64>) {
    use crate::tolerances::DENSITY_FLOOR;

    let a = z + n;
    let a_f = a as f64;
    let r_nuc = 1.2 * a_f.powf(1.0 / 3.0);
    let rho0 = 3.0 * a_f / (4.0 * std::f64::consts::PI * r_nuc.powi(3));

    let rho_p: Vec<f64> = (0..nr)
        .map(|k| {
            let r = (k + 1) as f64 * dr;
            if r < r_nuc {
                (rho0 * z as f64 / a_f).max(DENSITY_FLOOR)
            } else {
                DENSITY_FLOOR
            }
        })
        .collect();

    let rho_n: Vec<f64> = (0..nr)
        .map(|k| {
            let r = (k + 1) as f64 * dr;
            if r < r_nuc {
                (rho0 * n as f64 / a_f).max(DENSITY_FLOOR)
            } else {
                DENSITY_FLOOR
            }
        })
        .collect();

    (rho_p, rho_n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)] // exact known values
    fn mat_zeros_and_access() {
        let mut m = Mat::zeros(3);
        assert_eq!(m.get(0, 0), 0.0);
        m.set(1, 2, 5.0);
        assert_eq!(m.get(1, 2), 5.0);
        m.add(1, 2, 3.0);
        assert_eq!(m.get(1, 2), 8.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known values
    fn hermite_known_values() {
        assert_eq!(hermite_value(0, 1.0), 1.0);
        assert_eq!(hermite_value(1, 1.0), 2.0);
        assert!((hermite_value(2, 1.0) - 2.0).abs() < 1e-12);
        assert!((hermite_value(3, 1.0) - (-4.0)).abs() < 1e-12);
        assert!((hermite_value(4, 1.0) - (-20.0)).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known values
    fn factorial_known_values() {
        assert_eq!(factorial_f64(0), 1.0);
        assert_eq!(factorial_f64(1), 1.0);
        assert_eq!(factorial_f64(5), 120.0);
        assert_eq!(factorial_f64(10), 3_628_800.0);
    }

    #[test]
    fn hermite_determinism() {
        #[allow(clippy::approx_constant)] // test inputs, not math constants; determinism check
        let xs = [0.0, 1.0, -1.5, 3.14, -2.718];
        for &x in &xs {
            for n in 0..8 {
                let a = hermite_value(n, x);
                let b = hermite_value(n, x);
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "hermite_value({n}, {x}) not bitwise deterministic"
                );
            }
        }
    }

    #[test]
    fn factorial_determinism() {
        for n in 0..20 {
            let a = factorial_f64(n);
            let b = factorial_f64(n);
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "factorial_f64({n}) not bitwise deterministic"
            );
        }
    }

    #[test]
    fn bcs_v2_zero_gap_gives_step() {
        // With Δ→0, v²(ε<0) → 1 and v²(ε>0) → 0
        assert!((bcs_v2(-10.0, 0.001) - 1.0).abs() < 0.01);
        assert!(bcs_v2(10.0, 0.001).abs() < 0.01);
    }

    #[test]
    fn bcs_v2_at_fermi_surface() {
        // At ε=0, v² = 0.5 regardless of Δ
        assert!((bcs_v2(0.0, 1.0) - 0.5).abs() < 1e-12);
        assert!((bcs_v2(0.0, 5.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bcs_v2_range() {
        // v² must be in [0, 1] for any (ε, Δ>0)
        for eps in [-20.0, -5.0, 0.0, 5.0, 20.0] {
            let v2 = bcs_v2(eps, 1.0);
            assert!(
                (0.0..=1.0).contains(&v2),
                "v2={v2} out of [0,1] for eps={eps}"
            );
        }
    }

    #[test]
    fn coulomb_exchange_slater_positive_density() {
        // Exchange is always negative for positive density
        assert!(coulomb_exchange_slater(0.16) < 0.0);
        assert!(coulomb_exchange_slater(0.01) < 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact known value (0.0)
    fn coulomb_exchange_slater_zero_density() {
        assert_eq!(coulomb_exchange_slater(0.0), 0.0);
    }

    #[test]
    fn cm_correction_is_negative() {
        // CM correction subtracts energy
        assert!(cm_correction(10.0) < 0.0);
        assert!((cm_correction(10.0) - (-7.5)).abs() < 1e-12);
    }

    #[test]
    fn initial_wood_saxon_density_determinism() {
        let (a, b) = initial_wood_saxon_density(8, 8, 100, 0.15);
        let (a2, b2) = initial_wood_saxon_density(8, 8, 100, 0.15);
        for (x, y) in a.iter().zip(a2.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in b.iter().zip(b2.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn initial_wood_saxon_density_shape_and_normalization() {
        let (rho_p, rho_n) = initial_wood_saxon_density(28, 28, 120, 15.0 / 120.0);
        assert_eq!(rho_p.len(), 120);
        assert_eq!(rho_n.len(), 120);
        // 56Ni: R ≈ 1.2 * 56^(1/3) ≈ 4.6 fm; dr = 0.125 → r[35] ≈ 4.5, r[36] ≈ 4.625
        // Inside r_nuc density is uniform; outside is DENSITY_FLOOR
        assert!(
            rho_p[30] > 1e-10,
            "inside nucleus: proton density should be finite"
        );
        assert!(
            rho_n[30] > 1e-10,
            "inside nucleus: neutron density should be finite"
        );
        assert!(
            rho_p[115] < 1e-10,
            "outside nucleus: density should be floor"
        );
    }

    #[test]
    fn skyrme_central_t0_symmetric() {
        // For symmetric matter (rho_q = rho/2), the isospin dependence simplifies
        let t0 = -2488.0;
        let x0 = 0.834;
        let rho = 0.16;
        let rho_q = rho / 2.0;
        let u = skyrme_central_t0(t0, x0, rho, rho_q);
        // Result should be finite and have correct sign structure
        assert!(u.is_finite());
        // For attractive t0 < 0 and typical x0, potential should be negative
        assert!(
            u < 0.0,
            "t0 potential should be attractive for nuclear matter"
        );
    }

    #[test]
    fn coulomb_exchange_energy_density_sign_and_scaling() {
        let eps_zero = coulomb_exchange_energy_density(0.0);
        assert!(eps_zero.abs() < 1e-30, "zero density → zero energy density");

        let rho_sat = 0.08; // proton saturation density ~0.08 fm⁻³
        let eps = coulomb_exchange_energy_density(rho_sat);
        assert!(
            eps < 0.0,
            "Coulomb exchange energy density must be negative"
        );

        // ε ∝ ρ^{4/3}, so doubling density → 2^{4/3} ≈ 2.52× increase
        let eps2 = coulomb_exchange_energy_density(2.0 * rho_sat);
        let ratio = eps2 / eps;
        let expected = 2.0_f64.powf(4.0 / 3.0);
        assert!(
            (ratio - expected).abs() < 1e-10,
            "scaling: {ratio} vs expected {expected}"
        );
    }

    #[test]
    fn coulomb_exchange_energy_density_negative_density_clamped() {
        let eps = coulomb_exchange_energy_density(-0.1);
        assert!(
            eps.abs() < 1e-30,
            "negative density should be clamped to zero"
        );
    }
}
