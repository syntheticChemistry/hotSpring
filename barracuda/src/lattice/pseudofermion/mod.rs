// SPDX-License-Identifier: AGPL-3.0-or-later

//! Pseudofermion action for dynamical fermion HMC (Papers 10-12).
//!
//! In lattice QCD with dynamical fermions, the fermion determinant
//! is represented via pseudofermion fields:
//!
//!   det(D†D) = ∫ Dφ†Dφ exp(−φ†(D†D)⁻¹φ)
//!
//! The pseudofermion action is `S_F` = φ†X where `(D†D)X = φ`, solved by CG.
//! The pseudofermion force (derivative of `S_F` with respect to gauge links)
//! drives the molecular dynamics evolution alongside the gauge force.
//!
//! # Algorithm
//!
//! 1. **Heat bath**: generate φ = D†η where η is Gaussian random
//! 2. **Action**: `S_F` = φ† (D†D)⁻¹ φ = X†(D†D)X where (D†D)X = φ
//! 3. **Force**: dS\_F/dU\_μ(x) computed from X via the Dirac operator derivative
//! 4. **HMC**: total force = gauge force + fermion force
//!
//! # Module structure
//!
//! | Submodule | Responsibility |
//! |-----------|---------------|
//! | `config` | All configuration types |
//! | `action` | Core heat bath, action, force primitives |
//! | `hasenbusch` | Mass-preconditioned HMC (2-level) |
//! | `dynamics` | Full dynamical fermion HMC trajectory |
//! | `adaptive` | Adaptive step control + warm start |
//! | `npu_steering` | NPU-guided force anomaly detection |
//! | `run_history` | Trajectory history bookkeeping |
//!
//! # References
//!
//! - Gottlieb et al., PRD 35, 2531 (1987) — pseudofermion HMC
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 8.1-8.3
//! - Clark & Kennedy, NPB Proc. Suppl. 129, 850 (2004) — RHMC
//! - Hasenbusch, PLB 519, 177 (2001) — mass preconditioning

mod action;
mod config;
mod dynamics;
mod hasenbusch;

#[cfg(feature = "barracuda-local")]
pub mod adaptive;
#[cfg(feature = "barracuda-local")]
pub mod npu_steering;
pub mod run_history;

#[cfg(test)]
mod tests;

// ── Config re-exports ─────────────────────────────────────────────────────────
pub use config::{DynamicalHmcConfig, HasenbuschConfig, HasenbuschHmcConfig, PseudofermionConfig};

// ── Core action re-exports ────────────────────────────────────────────────────
pub use action::{pseudofermion_action, pseudofermion_force, pseudofermion_heatbath};

// ── Hasenbusch re-exports ─────────────────────────────────────────────────────
pub use hasenbusch::{
    HasenbuschHmcResult, hasenbusch_heavy_action, hasenbusch_heavy_heatbath,
    hasenbusch_hmc_trajectory, hasenbusch_ratio_action, hasenbusch_ratio_force,
    hasenbusch_ratio_heatbath,
};

// ── Dynamics re-exports ───────────────────────────────────────────────────────
pub use dynamics::{DynamicalHmcResult, dynamical_hmc_trajectory};

// ── Adaptive / NPU re-exports ─────────────────────────────────────────────────
#[cfg(feature = "barracuda-local")]
pub use adaptive::{
    AdaptiveStepController, AdaptiveThermalizationResult, MassAnnealingSchedule, StageResult,
    WarmStartResult, dynamical_thermalize_adaptive, dynamical_thermalize_warm_start,
    dynamical_thermalize_warm_start_npu,
};
#[cfg(feature = "barracuda-local")]
pub use npu_steering::{HmcForceAnomalyDetector, NpuSteering};
