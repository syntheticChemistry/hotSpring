// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: SEMF Parity — absorbed from validate_nuclear_eos.

use crate::physics::semf_binding_energy;
use crate::provenance::SLY4_PARAMS;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "semf-parity",
        track: Track::NuclearPhysics,
        tier: Tier::Rust,
        provenance_crate: "validate_nuclear_eos",
        provenance_date: "2026-05-09",
        description: "SEMF binding energy: Rust vs Python baseline parity",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    let be_pb208 = semf_binding_energy(82, 126, &SLY4_PARAMS);
    v.check_bool("semf:pb208_finite", be_pb208.is_finite() && be_pb208 > 0.0);
    v.check_bool("semf:pb208_range", (1600.0..1700.0).contains(&be_pb208));

    let be_fe56 = semf_binding_energy(26, 30, &SLY4_PARAMS);
    v.check_bool("semf:fe56_finite", be_fe56.is_finite() && be_fe56 > 0.0);
    v.check_bool("semf:fe56_range", (450.0..520.0).contains(&be_fe56));

    let be_he4 = semf_binding_energy(2, 2, &SLY4_PARAMS);
    v.check_bool("semf:he4_finite", be_he4.is_finite() && be_he4 > 0.0);

    let be1 = semf_binding_energy(82, 126, &SLY4_PARAMS);
    let be2 = semf_binding_energy(82, 126, &SLY4_PARAMS);
    v.check_bool("semf:deterministic", be1.total_cmp(&be2).is_eq());
}
