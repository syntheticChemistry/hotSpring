// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Tolerance Ordering — validates centralized tolerance hierarchy.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "tolerance-ordering",
        track: Track::DomainScience,
        tier: Tier::Rust,
        provenance_crate: "hotspring_guidestone",
        provenance_date: "2026-05-09",
        description: "Centralized tolerance constants: ordering invariants and bounds",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::tolerances;

    v.check_bool(
        "tolerance:semf_rel_positive",
        tolerances::COMPOSITION_SEMF_PARITY_REL > 0.0,
    );
    v.check_bool(
        "tolerance:semf_rel_tight",
        tolerances::COMPOSITION_SEMF_PARITY_REL < 1.0,
    );
    v.check_bool(
        "tolerance:plaquette_abs_positive",
        tolerances::COMPOSITION_PLAQUETTE_PARITY_ABS > 0.0,
    );
    v.check_bool(
        "tolerance:plaquette_abs_tight",
        tolerances::COMPOSITION_PLAQUETTE_PARITY_ABS < 1.0,
    );
    v.check_bool(
        "tolerance:exact_tighter_than_iterative",
        tolerances::EXACT_F64 < tolerances::ITERATIVE_F64,
    );
}
