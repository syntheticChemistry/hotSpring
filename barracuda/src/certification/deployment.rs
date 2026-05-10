// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 6: NUCLEUS Deployment Validation.
//!
//! Validates that hotSpring can participate in a live NUCLEUS deployment:
//! - Deploy graphs are well-formed and reference all required primals
//! - biomeOS `composition.status` is reachable and healthy
//! - hotSpring methods can be dynamically registered via `method.register`
//! - skunkBat audit forwarding is wired

use primalspring::validation::ValidationResult;

use crate::ipc::biome_status;
use crate::ipc::method_register;

/// Required primals in every hotSpring deploy graph.
const REQUIRED_PRIMALS: &[&str] = &[
    "beardog",
    "songbird",
    "coralreef",
    "toadstool",
    "barracuda",
    "nestgate",
    "rhizocrypt",
    "loamspine",
    "sweetgrass",
];

/// Run all L6 deployment validation checks.
pub fn validate_deployment(v: &mut ValidationResult) {
    validate_deploy_graph_coverage(v);
    validate_biome_status(v);
    validate_method_registration(v);
    validate_skunkbat_wiring(v);
}

fn validate_deploy_graph_coverage(v: &mut ValidationResult) {
    let graphs_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../graphs");
    let graph_files: Vec<_> = std::fs::read_dir(&graphs_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();

    v.check_bool(
        "l6:deploy_graphs_exist",
        !graph_files.is_empty(),
        &format!("{} TOML graphs found", graph_files.len()),
    );

    v.check_bool(
        "l6:deploy_graphs_multiple_pipelines",
        graph_files.len() >= 3,
        &format!("{} graphs (need >= 3)", graph_files.len()),
    );

    for graph_path in &graph_files {
        let name = graph_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        if let Ok(contents) = std::fs::read_to_string(graph_path) {
            let has_skunkbat = contents.contains("skunkbat");
            v.check_bool(
                &format!("l6:graph_{name}:has_skunkbat"),
                has_skunkbat,
                if has_skunkbat {
                    "skunkBat node present"
                } else {
                    "missing skunkBat"
                },
            );

            let mut missing: Vec<&str> = Vec::new();
            for primal in REQUIRED_PRIMALS {
                if !contents.contains(primal) {
                    missing.push(primal);
                }
            }
            v.check_bool(
                &format!("l6:graph_{name}:all_primals_present"),
                missing.is_empty(),
                &if missing.is_empty() {
                    "all 9 core primals".into()
                } else {
                    format!("missing: {}", missing.join(", "))
                },
            );
        }
    }
}

fn validate_biome_status(v: &mut ValidationResult) {
    match biome_status::query_composition_status() {
        Some(status) => {
            v.check_bool("l6:biome_status:reachable", true, "biomeOS responding");
            v.check_bool(
                "l6:biome_status:healthy",
                status.is_healthy(),
                &format!(
                    "health={:.2} pressure={:.2}",
                    status.primal_health, status.resource_pressure
                ),
            );
        }
        None => {
            v.check_skip("l6:biome_status", "biomeOS not running");
        }
    }
}

fn validate_method_registration(v: &mut ValidationResult) {
    let count = method_register::HOTSPRING_METHODS.len();
    v.check_bool(
        "l6:method_registry:methods_defined",
        count >= 20,
        &format!("{count} methods defined"),
    );

    let all_dotted = method_register::HOTSPRING_METHODS
        .iter()
        .all(|(m, _)| m.contains('.'));
    v.check_bool(
        "l6:method_registry:all_dotted",
        all_dotted,
        "all use dotted notation",
    );

    let registered = method_register::register_all_methods();
    if registered > 0 {
        v.check_bool(
            "l6:method_registry:dynamic_registration",
            true,
            &format!("{registered}/{count} registered"),
        );
    } else {
        v.check_skip(
            "l6:method_registry:dynamic_registration",
            "biomeOS not running",
        );
    }
}

fn validate_skunkbat_wiring(v: &mut ValidationResult) {
    let graphs_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../graphs");
    let qcd_graph = graphs_dir.join("hotspring_qcd_deploy.toml");

    if let Ok(contents) = std::fs::read_to_string(qcd_graph) {
        v.check_bool(
            "l6:skunkbat:in_qcd_deploy_graph",
            contents.contains("skunkbat"),
            "skunkBat node in QCD graph",
        );
        v.check_bool(
            "l6:skunkbat:defense_capability",
            contents.contains("\"defense\""),
            "defense capability declared",
        );
    } else {
        v.check_bool(
            "l6:skunkbat:qcd_graph_readable",
            false,
            "graph file not found",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_primals_has_core_nine() {
        assert!(REQUIRED_PRIMALS.len() >= 9);
        assert!(REQUIRED_PRIMALS.contains(&"beardog"));
        assert!(REQUIRED_PRIMALS.contains(&"barracuda"));
        assert!(REQUIRED_PRIMALS.contains(&"rhizocrypt"));
    }

    #[test]
    fn deployment_validation_completes() {
        let mut v = ValidationResult::new("L6 test");
        validate_deployment(&mut v);
        assert!(v.passed > 0, "at least some L6 checks should pass");
    }
}
