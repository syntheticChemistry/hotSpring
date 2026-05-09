// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: MD Yukawa OCP — absorbed from validate_md / sarkas_gpu.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "md-yukawa-ocp",
        track: Track::MolecularDynamics,
        tier: Tier::Rust,
        provenance_crate: "validate_md",
        provenance_date: "2026-05-09",
        description: "Yukawa OCP molecular dynamics: config construction and parameter validation",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::md::config::quick_test_case;

    let config = quick_test_case(108);
    v.check_bool("md:config_n_particles", config.n_particles == 108);
    v.check_bool("md:config_kappa_positive", config.kappa > 0.0);
    v.check_bool("md:config_gamma_positive", config.gamma > 0.0);
    v.check_bool("md:config_dt_positive", config.dt > 0.0);
    v.check_bool("md:box_side_positive", config.box_side() > 0.0);
    v.check_bool("md:temperature_positive", config.temperature() > 0.0);
    v.check_bool("md:number_density_positive", config.number_density() > 0.0);
}
