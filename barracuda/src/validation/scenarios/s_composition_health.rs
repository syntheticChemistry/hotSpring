// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Composition Health — absorbed from validate_nucleus_composition.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "composition-health",
        track: Track::CompositionParity,
        tier: Tier::Rust,
        provenance_crate: "validate_nucleus_composition",
        provenance_date: "2026-05-09",
        description: "NUCLEUS composition health: atomic type registry and standalone behavior",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::composition::{AtomicType, composition_health};
    use crate::primal_bridge::NucleusContext;

    v.check_bool(
        "composition:tower_domains",
        AtomicType::Tower.required_domains().len() == 2,
    );
    v.check_bool(
        "composition:node_domains",
        AtomicType::Node.required_domains().len() == 5,
    );
    v.check_bool(
        "composition:nest_domains",
        AtomicType::Nest.required_domains().len() == 6,
    );
    v.check_bool(
        "composition:nucleus_domains",
        AtomicType::FullNucleus.required_domains().len() == 9,
    );

    let ctx = NucleusContext {
        discovered: std::collections::HashMap::new(),
        family_id: "scenario-test".into(),
    };
    let health = composition_health(&ctx);
    v.check_bool(
        "composition:standalone_not_healthy",
        health["nucleus_health"] == false,
    );
}
