// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Gradient Flow — absorbed from validate_gradient_flow.
//!
//! Validates Wilson gradient flow on SU(3) gauge configurations:
//! action density at zero flow time, monotonic decrease under flow,
//! and numerical stability via the Euler integrator.

use crate::lattice::gradient_flow::{FlowIntegrator, run_flow};
use crate::lattice::wilson::Lattice;
use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "gradient-flow",
        track: Track::LatticeQcd,
        tier: Tier::Rust,
        provenance_crate: "validate_gradient_flow",
        provenance_date: "2026-05-12",
        description: "Wilson gradient flow: action density evolution and numerical stability",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    let dims = [4, 4, 4, 4];
    let beta = 6.0;
    let mut lattice = Lattice::cold_start(dims, beta);

    let action_cold = lattice.wilson_action();
    v.check_bool("gradient-flow:cold_start_finite", action_cold.is_finite());

    let measurements = run_flow(&mut lattice, FlowIntegrator::Euler, 0.01, 0.1, 1);

    v.check_bool(
        "gradient-flow:flow_steps_completed",
        measurements.len() >= 2,
    );

    if measurements.len() >= 2 {
        let first = measurements[0].energy_density;
        let last = measurements[measurements.len() - 1].energy_density;

        v.check_bool("gradient-flow:first_step_finite", first.is_finite());
        v.check_bool("gradient-flow:last_step_finite", last.is_finite());

        // Cold start: energy density stays near zero under flow
        v.check_bool("gradient-flow:cold_flow_stable", last.abs() < 1.0);
    }

    // Hot start: flow should smooth toward lower energy density
    let mut hot = Lattice::hot_start(dims, beta, 42);
    let hot_action_before = hot.wilson_action();
    let hot_measurements = run_flow(&mut hot, FlowIntegrator::Euler, 0.01, 0.2, 1);

    v.check_bool(
        "gradient-flow:hot_start_finite",
        hot_action_before.is_finite(),
    );

    if hot_measurements.len() >= 2 {
        let hot_first = hot_measurements[0].energy_density;
        let hot_last = hot_measurements[hot_measurements.len() - 1].energy_density;

        v.check_bool("gradient-flow:hot_flow_last_finite", hot_last.is_finite());
        v.check_bool(
            "gradient-flow:hot_flow_smoothing",
            hot_last <= hot_first + 1e-6,
        );
    }
}
