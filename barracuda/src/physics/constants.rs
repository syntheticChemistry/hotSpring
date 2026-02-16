// SPDX-License-Identifier: AGPL-3.0-only

//! Physical constants for nuclear structure calculations.
//!
//! Source: CODATA 2018 — Tiesinga et al., Rev. Mod. Phys. 93, 025010 (2021).
//! See PHYSICS.md §1 for full documentation.

/// ℏc — reduced Planck constant × speed of light (MeV·fm)
pub const HBAR_C: f64 = 197.3269804;

/// Average nucleon mass (m_p + m_n)/2 (MeV/c²)
pub const M_NUCLEON: f64 = 938.918;

/// Proton mass (MeV/c²)
pub const M_PROTON: f64 = 938.272046;

/// Neutron mass (MeV/c²)
pub const M_NEUTRON: f64 = 939.565378;

/// Coulomb constant e²/(4πε₀) (MeV·fm)
pub const E2: f64 = 1.4399764;

/// ℏ²/(2m_N) ≈ 20.735 MeV·fm² — kinetic energy prefactor
pub const HBAR2_2M: f64 = HBAR_C * HBAR_C / (2.0 * M_NUCLEON);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hbar_c_codata2018() {
        // CODATA 2018: ℏc = 197.3269804 ± 0.0000097 MeV·fm
        assert!(
            (HBAR_C - 197.3269804).abs() < 0.001,
            "ℏc should match CODATA 2018"
        );
    }

    #[test]
    fn nucleon_masses_physical() {
        assert!(M_PROTON > 938.0 && M_PROTON < 939.0);
        assert!(M_NEUTRON > 939.0 && M_NEUTRON < 940.0);
        assert!(M_NEUTRON > M_PROTON, "neutron heavier than proton");
        let avg = f64::midpoint(M_PROTON, M_NEUTRON);
        assert!(
            (avg - M_NUCLEON).abs() < 0.1,
            "M_NUCLEON should be average of proton and neutron"
        );
    }

    #[test]
    fn coulomb_constant_physical() {
        // e²/(4πε₀) ≈ 1.44 MeV·fm
        assert!(E2 > 1.43 && E2 < 1.45);
    }

    #[test]
    fn hbar2_2m_derived_correctly() {
        let expected = HBAR_C * HBAR_C / (2.0 * M_NUCLEON);
        assert!(
            (HBAR2_2M - expected).abs() < 1e-10,
            "HBAR2_2M should be ℏ²c²/(2·m_N·c²)"
        );
        // ~20.73 MeV·fm²
        assert!(HBAR2_2M > 20.0 && HBAR2_2M < 21.0);
    }
}
