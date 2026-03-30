// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-species kinetic-fluid coupling for HED simulations.
//!
//! Implements the conservative multi-species BGK kinetic equation,
//! 1D Euler fluid solver, and kinetic-fluid coupling interface.
//! Validates conservation of mass, momentum, and energy across
//! the coupling boundary.
//!
//! # References
//!
//! - Haack, Murillo, Sagert & Chuna, J. Comput. Phys. (2024), DOI:10.1016/j.jcp.2024.112908
//! - Haack, Hauck, Murillo, J. Stat. Phys. 168:826 (2017)
//!
//! # Module structure
//!
//! - `maxwellian` — 1D Maxwellian and velocity-space moments
//! - `bgk` — Multi-species BGK relaxation
//! - `euler` — 1D Euler with HLL Riemann solver
//! - `coupling` — Kinetic-fluid interface coupling

/// 1D Maxwellian distribution and moment computation.
pub mod maxwellian;

/// Conservative multi-species BGK kinetic relaxation.
pub mod bgk;

/// 1D compressible Euler equations with HLL Riemann solver.
pub(crate) mod euler;

/// Kinetic-fluid interface coupling.
pub mod coupling;

/// Maximum sub-iterations for kinetic-fluid interface coupling (Haack et al. §3.2).
pub(crate) const INTERFACE_MAX_SUB_ITERATIONS: usize = 3;

/// Density mismatch tolerance for interface sub-iteration convergence.
pub(crate) const INTERFACE_CONVERGENCE_TOL: f64 = 0.01;

/// Adiabatic index for monatomic ideal gas.
const GAMMA: f64 = 5.0 / 3.0;

// Re-export public API to preserve backwards compatibility.
pub use bgk::{
    BgkRelaxationResult, BgkSpecies, bgk_relaxation_step, bgk_target_params, entropy_h,
    run_bgk_relaxation,
};
pub use coupling::{CoupledResult, run_coupled_kinetic_fluid};
pub use euler::{SodResult, run_sod_shock_tube};
pub use maxwellian::{compute_moments, maxwellian_1d};

/// Full validation result for Paper 45.
pub struct KineticFluidValidation {
    /// BGK relaxation results.
    pub bgk: BgkRelaxationResult,
    /// Sod shock tube results.
    pub sod: SodResult,
    /// Coupled kinetic-fluid results.
    pub coupled: CoupledResult,
}

/// Run all three phases of the kinetic-fluid validation.
#[must_use]
pub fn validate_kinetic_fluid() -> KineticFluidValidation {
    let bgk = run_bgk_relaxation(3000, 0.005);
    let sod = run_sod_shock_tube(400, 0.2);
    let coupled = run_coupled_kinetic_fluid(30, 30, 81, 0.05);
    KineticFluidValidation { bgk, sod, coupled }
}
