//! Physical constants (CODATA 2018)

pub const HBAR_C: f64 = 197.3269804;       // MeV·fm
pub const M_NUCLEON: f64 = 938.918;         // MeV/c², average
pub const M_PROTON: f64 = 938.272046;       // MeV/c²
pub const M_NEUTRON: f64 = 939.565378;      // MeV/c²
pub const E2: f64 = 1.4399764;              // e²/(4πε₀) in MeV·fm
pub const HBAR2_2M: f64 = HBAR_C * HBAR_C / (2.0 * M_NUCLEON); // ≈ 20.735 MeV·fm²

