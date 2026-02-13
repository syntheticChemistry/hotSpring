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

