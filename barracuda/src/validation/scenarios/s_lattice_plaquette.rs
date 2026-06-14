// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Lattice Plaquette — absorbed from validate_pure_gauge.

use crate::tolerances;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "lattice-plaquette",
        track: Track::LatticeQcd,
        tier: Tier::Rust,
        provenance_crate: "validate_pure_gauge",
        provenance_date: "2026-05-09",
        description: "SU(3) Wilson gauge: cold-start plaquette and link count on 4^4 lattice",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::lattice::wilson::Lattice;

    let gauge = Lattice::cold_start([4, 4, 4, 4], 6.0);
    let plaq = gauge.average_plaquette();

    v.check_bool(
        "lattice:cold_start_plaq_unity",
        (plaq - 1.0).abs() < tolerances::LATTICE_COLD_PLAQUETTE_ABS,
    );
    v.check_bool("lattice:plaquette_finite", plaq.is_finite());
}
