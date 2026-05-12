// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Dielectric Response — absorbed from validate_dielectric.
//!
//! Validates Mermin dielectric function against known limits: static
//! screening, high-frequency transparency, and dissipation physicality.

use crate::physics::dielectric::{PlasmaParams, epsilon_mermin};
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "dielectric-mermin",
        track: Track::DomainScience,
        tier: Tier::Rust,
        provenance_crate: "validate_dielectric",
        provenance_date: "2026-05-12",
        description: "Mermin dielectric function: static/high-freq limits and physicality",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    let params = PlasmaParams::from_coupling(1.0, 1.0);
    let nu = 0.1;

    // Static limit (omega -> 0): epsilon should be real and > 1 for small k
    let eps_static = epsilon_mermin(0.5, 1e-10, nu, &params);
    v.check_bool("dielectric:static_real_finite", eps_static.re.is_finite());
    v.check_bool("dielectric:static_imag_finite", eps_static.im.is_finite());
    v.check_bool("dielectric:static_screening", eps_static.re > 1.0);

    // High frequency (omega >> omega_p): epsilon -> 1
    let eps_high = epsilon_mermin(0.5, 100.0, nu, &params);
    v.check_bool("dielectric:high_freq_finite", eps_high.re.is_finite());
    v.check_abs("dielectric:high_freq_approaches_unity", eps_high.re, 1.0, 0.1);

    // Imaginary part positive for finite collisions (dissipation)
    let eps_mid = epsilon_mermin(1.0, 1.0, nu, &params);
    v.check_bool("dielectric:mid_freq_finite", eps_mid.re.is_finite());
    v.check_bool("dielectric:dissipation_positive_im", eps_mid.im >= 0.0);

    // Zero collision rate: should reduce toward collisionless limit
    let eps_collisionless = epsilon_mermin(1.0, 1.0, 0.0, &params);
    v.check_bool(
        "dielectric:collisionless_finite",
        eps_collisionless.re.is_finite(),
    );
    v.check_bool(
        "dielectric:collisionless_less_dissipation",
        eps_collisionless.im.abs() <= eps_mid.im.abs() + 1e-12,
    );
}
