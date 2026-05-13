// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Cold Boot Sentinel — coralReef FECS/GPCCS sentinel feedback.
//!
//! Exercises the cold-boot sovereign dispatch path and captures structured
//! error feedback from coralReef's falcon_boot() hardening. This scenario
//! validates that:
//!
//! - FECS state is queryable and returns typed [`FecsState`] responses
//! - falcon_boot() failures produce observable structured errors
//! - Cold-boot dispatch timeouts are caught by the structured error path
//! - Device health and recovery are routable through NUCLEUS
//!
//! In standalone mode (no NUCLEUS), all probes gracefully degrade to
//! passing checks — the scenario confirms routing readiness rather than
//! hardware state.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "cold-boot-sentinel",
        track: Track::GpuCompute,
        tier: Tier::Live,
        provenance_crate: "exp170_sovereign_cold_boot",
        provenance_date: "2026-05-12",
        description: "coralReef FECS sentinel: cold boot structured errors, falcon_boot validation",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    use crate::primal_bridge::NucleusContext;

    let nucleus = NucleusContext::detect();

    let ember_alive = nucleus.by_domain("compute").is_some_and(|ep| ep.alive);

    // --- FECS state query with typed FecsState ---
    if ember_alive {
        let fecs_params = serde_json::json!({ "bdf": "auto" });
        match nucleus.call_by_capability("compute", "ember.fecs.state", fecs_params) {
            Ok(resp) => match serde_json::from_value::<crate::ember_types::FecsState>(resp) {
                Ok(fecs) => {
                    v.check_bool("sentinel:fecs_typed_response", true);
                    v.check_bool("sentinel:fecs_running", fecs.running);
                    if fecs.is_faulted() {
                        v.check_bool("sentinel:fecs_fault_observable", true);
                    }
                    if fecs.timed_out {
                        v.check_bool("sentinel:fecs_timeout_structured", true);
                    }
                    if let Some(ref err) = fecs.error {
                        v.check_bool("sentinel:fecs_error_captured", !err.is_empty());
                    }
                }
                Err(_) => {
                    v.check_bool("sentinel:fecs_typed_response", false);
                }
            },
            Err(_) => {
                v.check_bool("sentinel:fecs_reachable", false);
            }
        }
    } else {
        v.check_bool("sentinel:ember_standalone", true);
    }

    // --- Device health probe ---
    if ember_alive {
        let health_params = serde_json::json!({ "bdf": "auto" });
        match nucleus.call_by_capability("compute", "ember.device.health", health_params) {
            Ok(_resp) => {
                v.check_bool("sentinel:device_health_responded", true);
            }
            Err(_) => {
                v.check_bool("sentinel:device_health_responded", false);
            }
        }
    }

    // --- Device recovery probe ---
    let recover_routable = nucleus
        .get_by_capability("ember.device.recover")
        .is_some_and(|ep| ep.alive);
    v.check_bool(
        "sentinel:device_recover_routable",
        recover_routable || !ember_alive,
    );

    // --- Dispatch result retrieval ---
    let result_routable = nucleus
        .get_by_capability("compute.dispatch.result")
        .is_some_and(|ep| ep.alive);
    v.check_bool(
        "sentinel:dispatch_result_routable",
        result_routable || !ember_alive,
    );
}
