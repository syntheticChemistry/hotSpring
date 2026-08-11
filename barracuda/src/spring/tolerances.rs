// SPDX-License-Identifier: AGPL-3.0-or-later

//! QCD-specific validation thresholds.
//!
//! These tolerances define pass/fail criteria for lattice QCD observables.
//! Each constant is documented with its physical justification and reference.

/// Expected plaquette values for SU(3) Wilson action (quenched).
/// Sources: Gattringer & Lang Table 4.1, Creutz (1983), APE collaboration.
pub mod plaquette {
    /// β=5.70, infinite volume extrapolation.
    pub const BETA_5_70: f64 = 0.5494;
    /// β=5.90, infinite volume extrapolation.
    pub const BETA_5_90: f64 = 0.5778;
    /// β=6.00, infinite volume extrapolation.
    pub const BETA_6_00: f64 = 0.5934;
    /// β=6.20, infinite volume extrapolation.
    pub const BETA_6_20: f64 = 0.6136;

    /// Tolerance for plaquette agreement (accounts for finite volume + stats).
    pub const TOLERANCE: f64 = 0.005;
}

/// Expected acceptance rates for properly tuned HMC.
pub mod acceptance {
    /// Minimum acceptable acceptance rate for production data.
    pub const MIN_PRODUCTION: f64 = 0.50;
    /// Target acceptance rate for step-size tuning.
    pub const TARGET: f64 = 0.70;
    /// Maximum (indicates step size too small, slow thermalization).
    pub const MAX_HEALTHY: f64 = 0.90;
}

/// Thermalization criteria.
pub mod thermalization {
    /// Minimum warmup trajectories for 16⁴ volumes.
    pub const WARMUP_16: usize = 200;
    /// Minimum warmup trajectories for 24⁴ volumes.
    pub const WARMUP_24: usize = 500;
    /// Minimum warmup trajectories for 32⁴ volumes.
    pub const WARMUP_32: usize = 1000;
}
