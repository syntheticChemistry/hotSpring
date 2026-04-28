// SPDX-License-Identifier: AGPL-3.0-or-later

//! Named cost estimate constants for biomeOS scheduling.
//!
//! Every numeric literal in [`crate::niche::cost_estimates`] is sourced
//! from here. Reference hardware: RTX 4070 12 GB + i9-12900K.

// Physics domain (GPU-heavy)
pub const LATTICE_QCD_MS: f64 = 500.0;
pub const LATTICE_QCD_BYTES: u64 = 536_870_912;
pub const GAUGE_UPDATE_MS: f64 = 50.0;
pub const GAUGE_UPDATE_BYTES: u64 = 268_435_456;
pub const HMC_TRAJECTORY_MS: f64 = 2000.0;
pub const HMC_TRAJECTORY_BYTES: u64 = 536_870_912;
pub const WILSON_DIRAC_MS: f64 = 100.0;
pub const WILSON_DIRAC_BYTES: u64 = 268_435_456;
pub const MOLECULAR_DYNAMICS_MS: f64 = 1000.0;
pub const MOLECULAR_DYNAMICS_BYTES: u64 = 134_217_728;
pub const FLUID_MS: f64 = 200.0;
pub const FLUID_BYTES: u64 = 67_108_864;
pub const NUCLEAR_EOS_MS: f64 = 100.0;
pub const NUCLEAR_EOS_BYTES: u64 = 33_554_432;
pub const THERMAL_MS: f64 = 50.0;
pub const THERMAL_BYTES: u64 = 16_777_216;
pub const RADIATION_MS: f64 = 50.0;
pub const RADIATION_BYTES: u64 = 16_777_216;

// Compute primitives
pub const DF64_MS: f64 = 10.0;
pub const DF64_BYTES: u64 = 8_388_608;
pub const CG_SOLVER_MS: f64 = 500.0;
pub const CG_SOLVER_BYTES: u64 = 268_435_456;
pub const GRADIENT_FLOW_MS: f64 = 200.0;
pub const GRADIENT_FLOW_BYTES: u64 = 134_217_728;
pub const F64_MS: f64 = 5.0;
pub const F64_BYTES: u64 = 4_194_304;

// Health / lifecycle (near-zero cost)
pub const HEALTH_CHECK_MS: f64 = 0.1;
pub const HEALTH_CHECK_BYTES: u64 = 64;
pub const HEALTH_READINESS_MS: f64 = 0.2;
pub const HEALTH_READINESS_BYTES: u64 = 128;
pub const CAPABILITIES_LIST_MS: f64 = 0.1;
pub const CAPABILITIES_LIST_BYTES: u64 = 256;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::assertions_on_constants,
        reason = "compile-time sanity check for tolerance constants"
    )]
    fn cost_latencies_are_positive() {
        assert!(LATTICE_QCD_MS > 0.0);
        assert!(HMC_TRAJECTORY_MS > 0.0);
        assert!(HEALTH_CHECK_MS > 0.0);
    }

    #[test]
    #[expect(
        clippy::assertions_on_constants,
        reason = "compile-time sanity check for tolerance constants"
    )]
    fn cost_bytes_are_positive() {
        assert!(LATTICE_QCD_BYTES > 0);
        assert!(HEALTH_CHECK_BYTES > 0);
        assert!(CAPABILITIES_LIST_BYTES > 0);
    }

    #[test]
    #[expect(
        clippy::assertions_on_constants,
        reason = "compile-time sanity check for tolerance constants"
    )]
    fn cost_latencies_are_ordered() {
        assert!(HEALTH_CHECK_MS < NUCLEAR_EOS_MS);
        assert!(NUCLEAR_EOS_MS < LATTICE_QCD_MS);
        assert!(LATTICE_QCD_MS < HMC_TRAJECTORY_MS);
    }
}
