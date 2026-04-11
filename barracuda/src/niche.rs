// SPDX-License-Identifier: AGPL-3.0-or-later

//! Niche deployment self-knowledge for hotSpring.
//!
//! A Spring is a niche validation domain — not a primal. It proves that
//! scientific Python baselines can be faithfully ported to sovereign
//! Rust + GPU compute using the ecoPrimals stack. The niche deploys as
//! a biomeOS graph (`hotspring_qcd_proto_nucleate.toml`) that composes
//! real primals (BearDog, Songbird, ToadStool, barraCuda, etc.).
//!
//! This module holds the niche's self-knowledge:
//! - Capability table (what the niche exposes via biomeOS)
//! - Semantic mappings (capability domain → physics methods)
//! - Primal dependencies (germination order)
//! - Proto-nucleate reference
//!
//! # Evolution
//!
//! The `hotspring_primal` binary exposes these capabilities via a
//! JSON-RPC server. The final form is graph-only deployment where
//! biomeOS orchestrates the niche directly from deploy graphs.

/// Niche identity.
pub const NICHE_NAME: &str = "hotspring";

/// Human-readable niche description for biomeOS.
pub const NICHE_DESCRIPTION: &str =
    "Computational physics validation: nuclear EOS, lattice QCD, GPU MD, transport coefficients";

/// Niche version (tracks the spring version).
pub const NICHE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Proto-nucleate graph defining this spring's NUCLEUS composition target.
pub const PROTO_NUCLEATE: &str = "primalSpring/graphs/downstream/hotspring_qcd_proto_nucleate.toml";

/// NUCLEUS particle profile.
pub const PARTICLE_PROFILE: &str = "proton_heavy";

/// NUCLEUS composition model.
pub const COMPOSITION_MODEL: &str = "nucleated";

/// NUCLEUS fragments this spring composes.
pub const FRAGMENTS: &[&str] = &["tower_atomic", "node_atomic", "nest_atomic"];

/// Science domain tag for biomeOS routing.
pub const SCIENCE_DOMAIN: &str = "high_performance_compute";

/// Primal dependency declaration.
pub struct NicheDependency {
    pub name: &'static str,
    pub role: &'static str,
    pub required: bool,
    pub capability_domain: &'static str,
}

/// Primals this niche depends on (germination order matters).
pub const DEPENDENCIES: &[NicheDependency] = &[
    NicheDependency {
        name: "beardog",
        role: "security",
        required: true,
        capability_domain: "crypto",
    },
    NicheDependency {
        name: "songbird",
        role: "discovery",
        required: true,
        capability_domain: "discovery",
    },
    NicheDependency {
        name: "coralreef",
        role: "shader_compile",
        required: false,
        capability_domain: "shader",
    },
    NicheDependency {
        name: "toadstool",
        role: "compute",
        required: false,
        capability_domain: "compute",
    },
    NicheDependency {
        name: "barracuda",
        role: "math",
        required: true,
        capability_domain: "math",
    },
    NicheDependency {
        name: "nestgate",
        role: "storage",
        required: false,
        capability_domain: "storage",
    },
    NicheDependency {
        name: "rhizocrypt",
        role: "dag",
        required: false,
        capability_domain: "dag",
    },
    NicheDependency {
        name: "loamspine",
        role: "ledger",
        required: false,
        capability_domain: "ledger",
    },
    NicheDependency {
        name: "sweetgrass",
        role: "attribution",
        required: false,
        capability_domain: "attribution",
    },
    NicheDependency {
        name: "squirrel",
        role: "inference",
        required: false,
        capability_domain: "ai",
    },
];

// When Squirrel is composed, `inference.complete`, `inference.embed`, and
// `inference.models` are consumed from the inference provider (neuralSpring via
// Squirrel discovery) — they are not listed in `CAPABILITIES` below, which only
// names capabilities this spring implements itself.

/// Capabilities this niche advertises via `capability.list`.
pub const CAPABILITIES: &[&str] = &[
    "physics.lattice_qcd",
    "physics.lattice_gauge_update",
    "physics.hmc_trajectory",
    "physics.wilson_dirac",
    "physics.molecular_dynamics",
    "physics.fluid",
    "physics.nuclear_eos",
    "physics.thermal",
    "physics.radiation",
    "compute.df64",
    "compute.cg_solver",
    "compute.gradient_flow",
    "compute.f64",
    "health.check",
    "health.liveness",
    "health.readiness",
    "capabilities.list",
];

/// Bonding policy for this niche's NUCLEUS composition.
pub const BOND_TYPE: &str = "Metallic";
/// Trust model: all primals within the same FAMILY_ID.
pub const TRUST_MODEL: &str = "InternalNucleus";

/// Estimated cost per invocation (scheduling hint for biomeOS).
pub const COST_ESTIMATE_MS: u64 = 500;

/// Conventional directory name for ecosystem IPC sockets.
pub const ECOSYSTEM_SOCKET_DIR: &str = "biomeos";

/// Operation dependency hints for biomeOS Pathway Learner parallelization.
///
/// Maps each capability to the data inputs it requires, enabling the
/// Pathway Learner to determine which operations can run in parallel.
#[must_use]
pub fn operation_dependencies() -> serde_json::Value {
    serde_json::json!({
        "physics.lattice_qcd":        { "requires": ["lattice_size", "beta", "mass"] },
        "physics.lattice_gauge_update":{ "requires": ["gauge_config", "beta"], "depends_on": ["physics.lattice_qcd"] },
        "physics.hmc_trajectory":     { "requires": ["gauge_config", "dt", "n_md_steps"] },
        "physics.wilson_dirac":       { "requires": ["gauge_config", "mass", "source_vector"] },
        "physics.molecular_dynamics": { "requires": ["particle_config", "coupling", "temperature"] },
        "physics.fluid":             { "requires": ["grid_dimensions", "initial_conditions"] },
        "physics.nuclear_eos":       { "requires": ["skyrme_params", "nucleus_z", "nucleus_a"] },
        "physics.thermal":           { "requires": ["temperature_range", "material_params"] },
        "physics.radiation":         { "requires": ["source_spectrum", "material_opacity"] },
        "compute.df64":              { "requires": ["tensors", "operation"] },
        "compute.cg_solver":         { "requires": ["sparse_matrix", "rhs_vector", "tolerance"], "depends_on": ["physics.wilson_dirac"] },
        "compute.gradient_flow":     { "requires": ["gauge_config", "flow_time"] },
        "compute.f64":               { "requires": ["tensors", "operation"] },
        "health.check":              { "requires": [] },
        "health.liveness":           { "requires": [] },
        "health.readiness":          { "requires": [] },
        "capabilities.list":         { "requires": [] },
    })
}

/// Cost estimates for biomeOS scheduling (reference hardware: RTX 4070 12 GB + i9-12900K).
#[must_use]
pub fn cost_estimates() -> serde_json::Value {
    serde_json::json!({
        "physics.lattice_qcd":         { "latency_ms": 500.0,  "cpu": "high",   "gpu": "required",  "memory_bytes": 536_870_912 },
        "physics.lattice_gauge_update":{ "latency_ms": 50.0,   "cpu": "low",    "gpu": "required",  "memory_bytes": 268_435_456 },
        "physics.hmc_trajectory":      { "latency_ms": 2000.0, "cpu": "medium", "gpu": "required",  "memory_bytes": 536_870_912 },
        "physics.wilson_dirac":        { "latency_ms": 100.0,  "cpu": "medium", "gpu": "preferred", "memory_bytes": 268_435_456 },
        "physics.molecular_dynamics":  { "latency_ms": 1000.0, "cpu": "high",   "gpu": "preferred", "memory_bytes": 134_217_728 },
        "physics.fluid":              { "latency_ms": 200.0,  "cpu": "high",   "gpu": "preferred", "memory_bytes": 67_108_864 },
        "physics.nuclear_eos":        { "latency_ms": 100.0,  "cpu": "high",   "gpu": "optional",  "memory_bytes": 33_554_432 },
        "physics.thermal":            { "latency_ms": 50.0,   "cpu": "medium", "memory_bytes": 16_777_216 },
        "physics.radiation":          { "latency_ms": 50.0,   "cpu": "medium", "memory_bytes": 16_777_216 },
        "compute.df64":               { "latency_ms": 10.0,   "cpu": "low",    "gpu": "required",  "memory_bytes": 8_388_608 },
        "compute.cg_solver":          { "latency_ms": 500.0,  "cpu": "low",    "gpu": "required",  "memory_bytes": 268_435_456 },
        "compute.gradient_flow":      { "latency_ms": 200.0,  "cpu": "low",    "gpu": "required",  "memory_bytes": 134_217_728 },
        "compute.f64":                { "latency_ms": 5.0,    "cpu": "low",    "gpu": "preferred", "memory_bytes": 4_194_304 },
        "health.check":               { "latency_ms": 0.1,    "cpu": "none",   "memory_bytes": 64 },
        "health.liveness":            { "latency_ms": 0.1,    "cpu": "none",   "memory_bytes": 64 },
        "health.readiness":           { "latency_ms": 0.2,    "cpu": "none",   "memory_bytes": 128 },
        "capabilities.list":          { "latency_ms": 0.1,    "cpu": "none",   "memory_bytes": 256 },
    })
}

/// Semantic mappings: short name to fully qualified capability.
///
/// Used by biomeOS `CapabilityTaxonomy` for cross-primal routing.
pub const SEMANTIC_MAPPINGS: &[(&str, &str)] = &[
    ("lattice_qcd", "physics.lattice_qcd"),
    ("gauge_update", "physics.lattice_gauge_update"),
    ("hmc", "physics.hmc_trajectory"),
    ("wilson_dirac", "physics.wilson_dirac"),
    ("molecular_dynamics", "physics.molecular_dynamics"),
    ("fluid", "physics.fluid"),
    ("nuclear_eos", "physics.nuclear_eos"),
    ("thermal", "physics.thermal"),
    ("radiation", "physics.radiation"),
    ("df64", "compute.df64"),
    ("cg_solver", "compute.cg_solver"),
    ("gradient_flow", "compute.gradient_flow"),
    ("f64", "compute.f64"),
    ("liveness", "health.liveness"),
    ("readiness", "health.readiness"),
    ("list", "capabilities.list"),
];

/// Resolve the biomeOS family ID from environment.
///
/// Priority: `FAMILY_ID` -> `BIOMEOS_FAMILY_ID` -> `"default"`.
#[must_use]
pub fn family_id() -> String {
    std::env::var("FAMILY_ID")
        .or_else(|_| std::env::var("BIOMEOS_FAMILY_ID"))
        .unwrap_or_else(|_| "default".to_string())
}

/// Socket directories in XDG-compliant priority order.
#[must_use]
pub fn socket_dirs() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let mut dirs = Vec::new();

    if let Ok(d) = std::env::var("BIOMEOS_SOCKET_DIR") {
        dirs.push(PathBuf::from(d));
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        dirs.push(PathBuf::from(xdg).join(ECOSYSTEM_SOCKET_DIR));
    }

    let user = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());
    dirs.push(std::env::temp_dir().join(format!("{ECOSYSTEM_SOCKET_DIR}-{user}")));
    dirs.push(std::env::temp_dir());

    dirs
}

/// Resolve the socket path for this niche's IPC server.
///
/// Explicit overrides checked first, then XDG-compliant directory chain.
#[must_use]
pub fn resolve_server_socket() -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Ok(explicit) = std::env::var("HOTSPRING_SOCKET") {
        return PathBuf::from(explicit);
    }
    if let Ok(explicit) = std::env::var("PRIMAL_SOCKET") {
        return PathBuf::from(explicit);
    }

    let fid = family_id();
    let sock_name = format!("hotspring-physics-{fid}.sock");

    for dir in socket_dirs() {
        if dir.is_dir() || std::fs::create_dir_all(&dir).is_ok() {
            return dir.join(&sock_name);
        }
    }

    std::env::temp_dir().join(sock_name)
}

/// Resolve the Neural API socket path (discovered by convention).
#[must_use]
pub fn resolve_neural_api_socket() -> Option<std::path::PathBuf> {
    if let Ok(explicit) = std::env::var("NEURAL_API_SOCKET") {
        let p = std::path::PathBuf::from(&explicit);
        if p.exists() {
            return Some(p);
        }
    }

    let fid = family_id();
    let sock_name = format!("neural-api-{fid}.sock");

    for dir in socket_dirs() {
        let p = dir.join(&sock_name);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn niche_constants_non_empty() {
        assert!(!NICHE_NAME.is_empty());
        assert!(!NICHE_DESCRIPTION.is_empty());
        assert!(!PROTO_NUCLEATE.is_empty());
        assert!(!CAPABILITIES.is_empty());
        assert!(!DEPENDENCIES.is_empty());
        assert!(!FRAGMENTS.is_empty());
        assert!(!SEMANTIC_MAPPINGS.is_empty());
    }

    #[test]
    fn capabilities_follow_semantic_naming() {
        for cap in CAPABILITIES {
            assert!(
                cap.contains('.'),
                "capability '{cap}' should follow domain.operation format"
            );
        }
    }

    #[test]
    fn capabilities_include_health_and_discovery() {
        assert!(CAPABILITIES.contains(&"health.liveness"));
        assert!(CAPABILITIES.contains(&"health.readiness"));
        assert!(CAPABILITIES.contains(&"capabilities.list"));
    }

    #[test]
    fn dependencies_include_core_primals() {
        let names: Vec<&str> = DEPENDENCIES.iter().map(|d| d.name).collect();
        assert!(names.contains(&"beardog"));
        assert!(names.contains(&"barracuda"));
        assert!(names.contains(&"toadstool"));
    }

    #[test]
    fn semantic_mappings_cover_capabilities() {
        for (_, full) in SEMANTIC_MAPPINGS {
            assert!(
                CAPABILITIES.contains(full),
                "semantic mapping target '{full}' not in CAPABILITIES"
            );
        }
    }

    #[test]
    fn operation_dependencies_covers_all_capabilities() {
        let deps = operation_dependencies();
        for cap in CAPABILITIES {
            assert!(
                deps.get(cap).is_some(),
                "missing dependency entry for {cap}"
            );
        }
    }

    #[test]
    fn cost_estimates_covers_all_capabilities() {
        let costs = cost_estimates();
        for cap in CAPABILITIES {
            assert!(costs.get(cap).is_some(), "missing cost entry for {cap}");
        }
    }

    #[test]
    fn socket_dirs_never_empty() {
        let dirs = socket_dirs();
        assert!(!dirs.is_empty(), "should always resolve at least one dir");
    }

    #[test]
    fn family_id_has_default() {
        let fid = family_id();
        assert!(!fid.is_empty());
    }

    #[test]
    fn niche_name_matches_convention() {
        assert_eq!(NICHE_NAME, "hotspring");
        assert!(NICHE_NAME.chars().all(|c| c.is_ascii_lowercase()));
    }
}
