// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scenario: Dark Forest Glacial Gate — validates deploy graph security invariants.
//!
//! Checks all deploy graphs in `graphs/` against the five pillars defined in
//! `infra/wateringHole/DARK_FOREST_GLACIAL_GATE_STANDARD.md`:
//!
//! 1. Zero metadata leakage (stripped binaries, no hardcoded hosts)
//! 2. Zero port exposure (UDS default, no TCP)
//! 3. Songbird as sole network surface
//! 4. BTSP crypto integrity (security_model enforced)
//! 5. Enclave computing (trust model, content routing)
//!
//! This is a structural check — reads TOML graph files at compile time
//! via `include_str!` and validates metadata fields.

use crate::validation::ValidationHarness;
use crate::validation::scenarios::registry::{Scenario, ScenarioMeta, Tier, Track};

pub const SCENARIO: Scenario = Scenario {
    meta: ScenarioMeta {
        id: "dark-forest-gate",
        track: Track::CompositionParity,
        tier: Tier::Rust,
        provenance_crate: "dark_forest_glacial_gate_standard",
        provenance_date: "2026-05-19",
        description: "Dark Forest Glacial Gate: deploy graph security invariants (5 pillars)",
    },
    run,
};

struct GraphCheck {
    name: &'static str,
    toml: &'static str,
}

const GRAPHS: &[GraphCheck] = &[
    GraphCheck {
        name: "md_deploy",
        toml: include_str!("../../../../graphs/hotspring_md_deploy.toml"),
    },
    GraphCheck {
        name: "nuclear_eos_deploy",
        toml: include_str!("../../../../graphs/hotspring_nuclear_eos_deploy.toml"),
    },
    GraphCheck {
        name: "plasma_deploy",
        toml: include_str!("../../../../graphs/hotspring_plasma_deploy.toml"),
    },
    GraphCheck {
        name: "plasma_md_deploy",
        toml: include_str!("../../../../graphs/hotspring_plasma_md_deploy.toml"),
    },
    GraphCheck {
        name: "qcd_deploy",
        toml: include_str!("../../../../graphs/hotspring_qcd_deploy.toml"),
    },
    GraphCheck {
        name: "sovereign_gpu_deploy",
        toml: include_str!("../../../../graphs/hotspring_sovereign_gpu_deploy.toml"),
    },
    GraphCheck {
        name: "spectral_deploy",
        toml: include_str!("../../../../graphs/hotspring_spectral_deploy.toml"),
    },
];

pub fn run(v: &mut ValidationHarness) {
    v.check_bool("dark_forest:graph_count", GRAPHS.len() == 7);

    for graph in GRAPHS {
        check_graph(v, graph);
    }
}

fn check_graph(v: &mut ValidationHarness, graph: &GraphCheck) {
    let name = graph.name;
    let toml = graph.toml;

    // Pillar 2: Zero port exposure — UDS only, no TCP
    let has_uds_only = toml.contains("transport = \"uds_only\"");
    v.check_bool(&format!("dark_forest:{name}:uds_only"), has_uds_only);

    let has_zero_tcp = toml.contains("tcp_ports = 0");
    v.check_bool(&format!("dark_forest:{name}:tcp_ports_zero"), has_zero_tcp);

    // Pillar 4: BTSP crypto integrity
    let has_btsp = toml.contains("security_model = \"btsp_enforced\"");
    v.check_bool(&format!("dark_forest:{name}:btsp_enforced"), has_btsp);

    let has_secure_default = toml.contains("secure_by_default = true");
    v.check_bool(
        &format!("dark_forest:{name}:secure_by_default"),
        has_secure_default,
    );

    // Pillar 3: Songbird as sole network surface — tower_atomic fragment present
    let has_tower = toml.contains("tower_atomic");
    v.check_bool(&format!("dark_forest:{name}:tower_atomic"), has_tower);

    // Pillar 1+3: No direct http/tls on non-songbird nodes
    let has_http_cap = toml.contains("\"http.") || toml.contains("\"tls.");
    v.check_bool(
        &format!("dark_forest:{name}:no_direct_http_tls"),
        !has_http_cap,
    );

    // Pillar 5: composition_model present
    let has_composition = toml.contains("composition_model = \"nucleated\"");
    v.check_bool(
        &format!("dark_forest:{name}:nucleated_composition"),
        has_composition,
    );
}
