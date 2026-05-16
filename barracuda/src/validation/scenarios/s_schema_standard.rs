// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Schema Standard — validates canonical response shapes per Wave 20.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "schema-standard",
        track: Track::CompositionParity,
        tier: Tier::Rust,
        provenance_crate: "primalspring_wave20",
        provenance_date: "2026-05-16",
        description: "Wave 20 canonical response shape validation: capability.list envelope, signal registry",
    },
    run,
};

pub fn run(v: &mut ValidationHarness) {
    validate_capabilities_list_shape(v);
    validate_signal_registry(v);
    validate_niche_identity(v);
}

fn validate_capabilities_list_shape(v: &mut ValidationHarness) {
    let resp = crate::niche::capabilities_list_response();

    let caps = resp.get("capabilities").and_then(|c| c.as_array());
    v.check_bool(
        "schema:capability.list:capabilities_is_array",
        caps.is_some(),
    );

    if let Some(arr) = caps {
        v.check_bool(
            "schema:capability.list:capabilities_non_empty",
            !arr.is_empty(),
        );
        let all_strings = arr.iter().all(serde_json::Value::is_string);
        v.check_bool("schema:capability.list:all_string_entries", all_strings);
    }

    let count = resp.get("count").and_then(serde_json::Value::as_u64);
    v.check_bool("schema:capability.list:count_present", count.is_some());

    if let (Some(arr), Some(n)) = (caps, count) {
        v.check_bool(
            "schema:capability.list:count_matches_length",
            n as usize == arr.len(),
        );
    }

    let primal = resp.get("primal").and_then(|p| p.as_str());
    v.check_bool("schema:capability.list:primal_present", primal.is_some());
    v.check_bool(
        "schema:capability.list:primal_is_hotspring",
        primal == Some(crate::niche::NICHE_NAME),
    );
}

fn validate_signal_registry(v: &mut ValidationHarness) {
    let adopted = ["node.compute", "tower.publish", "nest.commit"];
    let candidates = ["nest.store"];

    let registry_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join("capability_registry.toml");

    let Ok(content) = std::fs::read_to_string(&registry_path) else {
        v.check_bool("schema:signal_registry:file_readable", false);
        return;
    };

    v.check_bool("schema:signal_registry:file_readable", true);

    for signal in &adopted {
        v.check_bool(
            &format!("schema:signal_registry:adopted:{signal}"),
            content.contains(signal),
        );
    }

    for signal in &candidates {
        v.check_bool(
            &format!("schema:signal_registry:candidate:{signal}"),
            content.contains(signal),
        );
    }
}

fn validate_niche_identity(v: &mut ValidationHarness) {
    v.check_bool(
        "schema:niche:name_non_empty",
        !crate::niche::NICHE_NAME.is_empty(),
    );
    v.check_bool(
        "schema:niche:domain_non_empty",
        !crate::niche::PRIMAL_DOMAIN.is_empty(),
    );
    v.check_bool(
        "schema:niche:local_capabilities_non_empty",
        !crate::niche::LOCAL_CAPABILITIES.is_empty(),
    );
    v.check_bool(
        "schema:niche:routed_capabilities_non_empty",
        !crate::niche::ROUTED_CAPABILITIES.is_empty(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::ValidationHarness;

    #[test]
    fn schema_standard_scenario_passes() {
        let mut harness = ValidationHarness::new("schema-standard-test");
        run(&mut harness);
        assert!(harness.all_passed(), "schema-standard scenario failed");
    }
}
