// SPDX-License-Identifier: AGPL-3.0-or-later

//! Configuration types for pseudofermion HMC variants.

use crate::lattice::hmc::IntegratorType;
use crate::tolerances::lattice::DYNAMICAL_CG_MAX_ITER;

/// Configuration for pseudofermion HMC.
#[derive(Clone, Debug)]
pub struct PseudofermionConfig {
    /// Fermion mass (staggered)
    pub mass: f64,
    /// CG tolerance for inversions
    pub cg_tol: f64,
    /// CG maximum iterations
    pub cg_max_iter: usize,
}

impl Default for PseudofermionConfig {
    fn default() -> Self {
        Self {
            mass: 0.1,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
        }
    }
}

/// Configuration for Hasenbusch mass preconditioning (Hasenbusch 2001, PLB 519, 177).
///
/// Two-level split: `det(D†D(m_light))` = `det(D†D(m_heavy))` × `det(D†D(m_light)/D†D(m_heavy))`.
/// Heavy sector is cheap (few CG iterations); ratio sector has smaller condition number than
/// the full light-mass operator → faster CG and smaller forces.
#[derive(Clone, Debug)]
pub struct HasenbuschConfig {
    /// Heavy (intermediate) mass — typically 0.3–0.5.
    pub heavy_mass: f64,
    /// Light (physical) mass — typically 0.01–0.1.
    pub light_mass: f64,
    /// CG tolerance for inversions.
    pub cg_tol: f64,
    /// CG maximum iterations per solve.
    pub cg_max_iter: usize,
    /// MD steps for the light (ratio) sector — more steps (expensive inversions).
    pub n_md_steps_light: usize,
    /// MD steps for the heavy sector — fewer steps (cheap inversions).
    pub n_md_steps_heavy: usize,
}

impl Default for HasenbuschConfig {
    fn default() -> Self {
        Self {
            heavy_mass: 0.4,
            light_mass: 0.1,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
            n_md_steps_light: 16,
            n_md_steps_heavy: 4,
        }
    }
}

/// Configuration for Hasenbusch HMC.
#[derive(Clone, Debug)]
pub struct HasenbuschHmcConfig {
    /// MD step size.
    pub dt: f64,
    /// RNG seed.
    pub seed: u64,
    /// Hasenbusch mass splitting parameters.
    pub hasenbusch: HasenbuschConfig,
    /// Inverse coupling β.
    pub beta: f64,
}

impl Default for HasenbuschHmcConfig {
    fn default() -> Self {
        Self {
            dt: 0.02,
            seed: 42,
            hasenbusch: HasenbuschConfig::default(),
            beta: 5.5,
        }
    }
}

/// Dynamical fermion HMC configuration.
#[derive(Clone, Debug)]
pub struct DynamicalHmcConfig {
    /// Number of MD steps per trajectory
    pub n_md_steps: usize,
    /// MD step size
    pub dt: f64,
    /// PRNG seed (mutated each trajectory)
    pub seed: u64,
    /// Pseudofermion configuration
    pub fermion: PseudofermionConfig,
    /// Gauge coupling (β)
    pub beta: f64,
    /// Number of staggered flavors (`N_f/4` for staggered; use 2 for 2-flavor)
    pub n_flavors_over_4: usize,
    /// Integrator type (default: Leapfrog for backward compat)
    pub integrator: IntegratorType,
}

impl Default for DynamicalHmcConfig {
    fn default() -> Self {
        Self {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            fermion: PseudofermionConfig::default(),
            beta: 5.5,
            n_flavors_over_4: 2,
            integrator: IntegratorType::Leapfrog,
        }
    }
}
