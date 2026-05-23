// SPDX-License-Identifier: AGPL-3.0-or-later

//! Layer 6: NUCLEUS Deployment Validation.
//!
//! Validates that hotSpring can participate in a live NUCLEUS deployment:
//! - Deploy graphs are well-formed and reference all required primals
//! - biomeOS `composition.status` is reachable and healthy
//! - hotSpring methods can be dynamically registered via `method.register`
//! - Defense provider (audit forwarding) is wired

use primalspring::validation::ValidationResult;

use crate::ipc::biome_status;
use crate::ipc::method_register;

/// Required primals in every hotSpring deploy graph, derived from the
/// niche dependency table (single source of truth).
fn required_primals() -> Vec<&'static str> {
    crate::niche::DEPENDENCIES
        .iter()
        .filter(|d| d.required)
        .map(|d| d.name)
        .collect()
}

/// Run all L6 deployment validation checks.
pub fn validate_deployment(v: &mut ValidationResult) {
    validate_deploy_graph_coverage(v);
    validate_biome_status(v);
    validate_method_registration(v);
    validate_defense_wiring(v);
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
            let defense_primal =
                crate::niche::primal_name_for_domain("defense").unwrap_or("defense");
            let has_defense = contents.contains(defense_primal);
            v.check_bool(
                &format!("l6:graph_{name}:has_defense"),
                has_defense,
                if has_defense {
                    "defense provider present"
                } else {
                    "missing defense provider"
                },
            );

            let required = required_primals();
            let mut missing: Vec<&str> = Vec::new();
            for primal in &required {
                if !contents.contains(primal) {
                    missing.push(primal);
                }
            }
            v.check_bool(
                &format!("l6:graph_{name}:all_primals_present"),
                missing.is_empty(),
                &if missing.is_empty() {
                    format!("all {} required primals", required.len())
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

fn validate_defense_wiring(v: &mut ValidationResult) {
    let graphs_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../graphs");
    let qcd_graph = graphs_dir.join("hotspring_qcd_deploy.toml");
    let defense_primal = crate::niche::primal_name_for_domain("defense").unwrap_or("defense");

    if let Ok(contents) = std::fs::read_to_string(qcd_graph) {
        v.check_bool(
            "l6:defense:in_qcd_deploy_graph",
            contents.contains(defense_primal),
            "defense provider in QCD graph",
        );
        v.check_bool(
            "l6:defense:capability_declared",
            contents.contains("\"defense\""),
            "defense capability declared",
        );
    } else {
        v.check_bool(
            "l6:defense:qcd_graph_readable",
            false,
            "graph file not found",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_primals_derived_from_niche() {
        let required = required_primals();
        assert!(
            !required.is_empty(),
            "niche dependencies should include at least one required primal"
        );
        for name in &required {
            assert!(
                crate::niche::DEPENDENCIES
                    .iter()
                    .any(|d| d.name == *name && d.required),
                "{name} should be in DEPENDENCIES with required=true"
            );
        }
    }

    #[test]
    fn deployment_validation_completes() {
        let mut v = ValidationResult::new("L6 test");
        validate_deployment(&mut v);
        assert!(v.passed > 0, "at least some L6 checks should pass");
    }
}
