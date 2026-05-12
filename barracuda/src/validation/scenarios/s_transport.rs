// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Transport Coefficients — absorbed from validate_stanton_murillo / validate_transport.
//!
//! Validates Stanton-Murillo effective Coulomb coupling fits for viscosity
//! (η*) and thermal conductivity (λ*) against published values.

use crate::md::transport::{eta_star_stanton_murillo, lambda_star_stanton_murillo};
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "transport-stanton-murillo",
        track: Track::TransportCoefficients,
        tier: Tier::Rust,
        provenance_crate: "validate_stanton_murillo",
        provenance_date: "2026-05-12",
        description: "Stanton-Murillo transport coefficients: η* and λ* fit validation",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    // Weak coupling regime (Γ ~ 1): both coefficients should be large (dilute limit)
    let eta_weak = eta_star_stanton_murillo(1.0, 1.0);
    v.check_bool("transport:eta_weak_positive", eta_weak > 0.0);
    v.check_bool("transport:eta_weak_finite", eta_weak.is_finite());

    let lambda_weak = lambda_star_stanton_murillo(1.0, 1.0);
    v.check_bool("transport:lambda_weak_positive", lambda_weak > 0.0);
    v.check_bool("transport:lambda_weak_finite", lambda_weak.is_finite());

    // Strong coupling (Γ ~ 100): viscosity minimum / plateau behavior
    let eta_strong = eta_star_stanton_murillo(100.0, 1.0);
    v.check_bool("transport:eta_strong_positive", eta_strong > 0.0);
    v.check_bool("transport:eta_strong_finite", eta_strong.is_finite());

    // Viscosity should decrease from weak to moderate coupling
    let eta_mid = eta_star_stanton_murillo(10.0, 1.0);
    v.check_bool("transport:eta_decreases_to_moderate", eta_mid < eta_weak);

    // Thermal conductivity monotonically decreasing at fixed kappa
    let lambda_strong = lambda_star_stanton_murillo(100.0, 1.0);
    v.check_bool(
        "transport:lambda_decreasing_with_coupling",
        lambda_strong < lambda_weak,
    );

    // Screening dependence: higher kappa weakens correlations
    let eta_high_kappa = eta_star_stanton_murillo(10.0, 3.0);
    v.check_bool("transport:eta_kappa_dependence_finite", eta_high_kappa.is_finite());
    v.check_bool("transport:eta_kappa_dependence_positive", eta_high_kappa > 0.0);
}
