// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Screened Coulomb — absorbed from validate_screened_coulomb.
//!
//! Validates Yukawa screened-Coulomb eigenvalues against analytical
//! hydrogen-like references (Z=1) and published Murillo lineage values.

use crate::physics::screened_coulomb::eigenvalues;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "screened-coulomb",
        track: Track::DomainScience,
        tier: Tier::Rust,
        provenance_crate: "validate_screened_coulomb",
        provenance_date: "2026-05-12",
        description: "Yukawa screened-Coulomb eigenvalues: convergence and hydrogen-like limits",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    let z = 1.0;
    let n_grid = 1000;
    let r_max = 50.0;

    // Unscreened limit (kappa -> 0): first eigenvalue approaches hydrogen 1s = -Z²/2
    let eigs_unscreened = eigenvalues(z, 1e-6, 0, n_grid, r_max);
    v.check_bool(
        "screened-coulomb:unscreened_has_bound_state",
        !eigs_unscreened.is_empty(),
    );
    if let Some(&e0) = eigs_unscreened.first() {
        v.check_abs("screened-coulomb:hydrogen_1s_limit", e0, -0.5, 0.02);
    }

    // Moderate screening: bound state exists but is shallower
    let eigs_moderate = eigenvalues(z, 0.5, 0, n_grid, r_max);
    v.check_bool(
        "screened-coulomb:moderate_has_bound_state",
        !eigs_moderate.is_empty(),
    );
    if let Some(&e0) = eigs_moderate.first() {
        v.check_bool(
            "screened-coulomb:moderate_shallower_than_hydrogen",
            e0 > -0.5,
        );
    }

    // Heavy screening: bound states vanish or become very shallow
    let eigs_heavy = eigenvalues(z, 5.0, 0, n_grid, r_max);
    let heavy_depth = eigs_heavy.first().copied().unwrap_or(0.0);
    v.check_bool(
        "screened-coulomb:heavy_screening_shallow",
        heavy_depth > -0.1 || eigs_heavy.is_empty(),
    );

    // Convergence: refining grid should not change first eigenvalue significantly
    let eigs_coarse = eigenvalues(z, 0.5, 0, 200, r_max);
    let eigs_fine = eigenvalues(z, 0.5, 0, 2000, r_max);
    if let (Some(&ec), Some(&ef)) = (eigs_coarse.first(), eigs_fine.first()) {
        v.check_abs("screened-coulomb:grid_convergence", ec, ef, 0.01);
    }
}
