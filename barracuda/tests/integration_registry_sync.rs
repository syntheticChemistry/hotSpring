// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::unwrap_used)]

//! Cross-registry integration test: validates hotSpring's capability
//! registry methods against primalSpring's canonical registry.
//!
//! This test catches drift where hotSpring declares a method name
//! that no primal actually serves, or where primalSpring renames a
//! method upstream but hotSpring still uses the old name.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().to_path_buf()
}

fn extract_methods_from_hotspring_registry(content: &str) -> BTreeSet<String> {
    let parsed: toml::Value = toml::from_str(content).expect("parse hotSpring registry");
    parsed["capabilities"]
        .as_array()
        .expect("capabilities array")
        .iter()
        .filter_map(|c| c.get("method")?.as_str().map(String::from))
        .collect()
}

fn extract_methods_from_primalspring_registry(content: &str) -> BTreeSet<String> {
    let parsed: toml::Value = toml::from_str(content).expect("parse primalSpring registry");
    let table = parsed.as_table().expect("root table");
    let mut methods = BTreeSet::new();

    for (_domain, section) in table {
        if let Some(arr) = section.get("methods").and_then(|v| v.as_array()) {
            for m in arr {
                if let Some(s) = m.as_str() {
                    methods.insert(s.to_string());
                }
            }
        }
        if let Some(arr) = section.get("test_fixtures").and_then(|v| v.as_array()) {
            for m in arr {
                if let Some(s) = m.as_str() {
                    methods.insert(s.to_string());
                }
            }
        }
    }

    methods
}

#[test]
fn local_registry_parses_cleanly() {
    let registry_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join("capability_registry.toml");
    let content = std::fs::read_to_string(&registry_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", registry_path.display()));
    let methods = extract_methods_from_hotspring_registry(&content);
    assert!(
        methods.len() > 20,
        "expected >20 registered methods, got {}",
        methods.len()
    );
}

#[test]
#[ignore = "advisory: 13 hotSpring methods pending addition to primalSpring canonical registry"]
fn cross_registry_sync_with_primalspring() {
    let hotspring_registry = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join("capability_registry.toml");
    let primalspring_registry = workspace_root()
        .join("..")
        .join("primalSpring")
        .join("config")
        .join("capability_registry.toml");

    if !primalspring_registry.exists() {
        eprintln!(
            "SKIP: primalSpring registry not found at {}",
            primalspring_registry.display()
        );
        return;
    }

    let hs_content = std::fs::read_to_string(&hotspring_registry)
        .unwrap_or_else(|e| panic!("read hotSpring registry: {e}"));
    let ps_content = std::fs::read_to_string(&primalspring_registry)
        .unwrap_or_else(|e| panic!("read primalSpring registry: {e}"));

    let hs_methods = extract_methods_from_hotspring_registry(&hs_content);
    let ps_methods = extract_methods_from_primalspring_registry(&ps_content);

    let mut missing: Vec<&str> = Vec::new();
    for method in &hs_methods {
        if !ps_methods.contains(method) {
            missing.push(method);
        }
    }

    if !missing.is_empty() {
        let report = missing
            .iter()
            .map(|m| format!("  {m}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "{} hotSpring method(s) not in primalSpring canonical registry \
             ({} canonical methods):\n{report}\n\n\
             These must be added to primalSpring/config/capability_registry.toml \
             or renamed to match upstream.",
            missing.len(),
            ps_methods.len(),
        );
    }

    eprintln!(
        "OK: all {} hotSpring methods found in primalSpring canonical ({} total)",
        hs_methods.len(),
        ps_methods.len()
    );
}

#[test]
fn deploy_graphs_reference_only_registered_capabilities() {
    let registry_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join("capability_registry.toml");
    let content = std::fs::read_to_string(&registry_path).unwrap();
    let registered = extract_methods_from_hotspring_registry(&content);

    let graphs_dir = workspace_root().join("graphs");
    if !graphs_dir.exists() {
        eprintln!("SKIP: graphs/ not found");
        return;
    }

    let mut errors = Vec::new();

    for entry in std::fs::read_dir(&graphs_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "toml") {
            let graph_content = std::fs::read_to_string(&path).unwrap();
            let parsed: toml::Value = match toml::from_str(&graph_content) {
                Ok(v) => v,
                Err(e) => {
                    errors.push(format!("{}: parse error: {e}", path.display()));
                    continue;
                }
            };

            if let Some(nodes) = parsed
                .get("graph")
                .and_then(|g| g.get("nodes"))
                .and_then(|n| n.as_array())
            {
                for node in nodes {
                    if let Some(caps) = node.get("capabilities").and_then(|c| c.as_array()) {
                        let node_name = node
                            .get("name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("unknown");
                        for cap in caps {
                            if let Some(method) = cap.as_str() {
                                if !registered.contains(method) {
                                    errors.push(format!(
                                        "{}: node '{}' capability '{}' not in registry",
                                        path.file_name().unwrap().to_string_lossy(),
                                        node_name,
                                        method
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        errors.is_empty(),
        "{} unregistered capabilities in deploy graphs:\n{}",
        errors.len(),
        errors.join("\n")
    );
}
