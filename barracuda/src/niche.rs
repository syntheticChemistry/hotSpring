// SPDX-License-Identifier: AGPL-3.0-or-later

//! Niche deployment self-knowledge for hotSpring.
//!
//! A Spring is a niche validation domain — not a primal. It proves that
//! scientific Python baselines can be faithfully ported to sovereign
//! Rust + GPU compute using the ecoPrimals stack. The niche deploys as
//! a biomeOS graph (defined in `downstream_manifest.toml`) that composes
//! real primals (BearDog, Songbird, ToadStool, barraCuda, etc.).
//!
//! This module holds the niche's self-knowledge:
//! - Capability table (local vs routed — what we serve vs what we proxy)
//! - Semantic mappings (capability domain -> physics methods)
//! - Primal dependencies (germination order)
//! - Proto-nucleate reference
//! - biomeOS registration (lifecycle + capability advertisement)
//!
//! # Local vs Routed Capabilities
//!
//! `LOCAL_CAPABILITIES` lists methods that `hotspring_primal` actually
//! handles — callers get a real physics result. `ROUTED_CAPABILITIES`
//! lists methods that hotSpring understands but routes to canonical
//! providers (Squirrel for inference, provenance trio for DAG/ledger/
//! attribution, etc.). Callers reaching hotSpring directly for routed
//! methods should go through `capability.call` or the Neural API.
//!
//! # Standalone Mode
//!
//! Set `HOTSPRING_NO_NUCLEUS=1` to run without biomeOS or NUCLEUS primals.
//! Registration is skipped and IPC degrades gracefully.
//!
//! # Evolution
//!
//! The `hotspring_primal` binary exposes these capabilities via a
//! JSON-RPC server. The final form is graph-only deployment where
//! biomeOS orchestrates the niche directly from deploy graphs.

use std::path::Path;

use log::{info, warn};

/// Niche identity.
pub const NICHE_NAME: &str = "hotspring";

/// Primal domain for biomeOS routing.
pub const PRIMAL_DOMAIN: &str = "physics";

/// Human-readable niche description for biomeOS.
pub const NICHE_DESCRIPTION: &str =
    "Computational physics validation: nuclear EOS, lattice QCD, GPU MD, transport coefficients";

/// Niche version (tracks the spring version).
pub const NICHE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Proto-nucleate graph defining this spring's NUCLEUS composition target.
///
/// After v0.9.15 graph consolidation, the canonical source is the `[[downstream]]`
/// entry with `spring_name = "hotspring"` inside `downstream_manifest.toml`.
/// The standalone file was merged into the manifest.
pub const PROTO_NUCLEATE: &str = "primalSpring/graphs/downstream/downstream_manifest.toml";

/// NUCLEUS particle profile.
pub const PARTICLE_PROFILE: &str = "proton_heavy";

/// NUCLEUS composition model.
pub const COMPOSITION_MODEL: &str = "nucleated";

/// NUCLEUS fragments this spring composes.
pub const FRAGMENTS: &[&str] = &["tower_atomic", "node_atomic", "nest_atomic"];

/// Science domain tag for biomeOS routing.
pub const SCIENCE_DOMAIN: &str = "high_performance_compute";

/// Default registration target — discovered at runtime, not hardcoded.
/// Override via `BIOMEOS_PRIMAL` env var for non-standard deployments.
const REGISTRATION_TARGET: &str = "biomeos";

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

/// Capabilities this binary **locally serves** via `dispatch_request`.
///
/// These are the methods `hotspring_primal` actually handles — callers
/// get a real response, not a routing bounce. When registering with biomeOS,
/// these are claimed as "served here."
pub const LOCAL_CAPABILITIES: &[&str] = &[
    // Physics domain (core science)
    "physics.lattice_qcd",
    "physics.lattice_gauge_update",
    "physics.hmc_trajectory",
    "physics.wilson_dirac",
    "physics.molecular_dynamics",
    "physics.fluid",
    "physics.nuclear_eos",
    "physics.thermal",
    "physics.radiation",
    // Compute primitives exposed by hotSpring
    "compute.df64",
    "compute.cg_solver",
    "compute.gradient_flow",
    "compute.f64",
    // Composition health
    "composition.health",
    // Health probes
    "health.check",
    "health.liveness",
    "health.readiness",
    // Capability advertisement
    "capabilities.list",
    // MCP tool discovery
    "mcp.tools.list",
];

/// Ecosystem capabilities that hotSpring **coordinates but routes to
/// other primals** via biomeOS Neural API semantic routing.
///
/// These are registered as coordination metadata so biomeOS and springs
/// know hotSpring understands these domains. Callers reaching hotSpring
/// directly for these methods should go through `capability.call`.
///
/// Each entry names the canonical provider per `capability_registry.toml`.
pub const ROUTED_CAPABILITIES: &[(&str, &str)] = &[
    // Inference (Squirrel -> neuralSpring)
    ("inference.complete", "squirrel"),
    ("inference.embed", "squirrel"),
    ("inference.models", "squirrel"),
    // Crypto (BearDog)
    ("crypto.sign_ed25519", "beardog"),
    ("crypto.verify_ed25519", "beardog"),
    // Compute dispatch (ToadStool)
    ("compute.dispatch.submit", "toadstool"),
    ("compute.dispatch.capabilities", "toadstool"),
    // Shader compilation (coralReef)
    ("shader.compile.wgsl", "coralreef"),
    ("shader.compile.spirv", "coralreef"),
    // Storage (NestGate)
    ("storage.store", "nestgate"),
    ("storage.retrieve", "nestgate"),
    ("storage.list", "nestgate"),
    // Provenance DAG (rhizoCrypt)
    ("dag.session.create", "rhizocrypt"),
    ("dag.event.append", "rhizocrypt"),
    ("dag.merkle.root", "rhizocrypt"),
    ("dag.merkle.verify", "rhizocrypt"),
    // Ledger (loamSpine)
    ("spine.create", "loamspine"),
    ("entry.append", "loamspine"),
    ("session.commit", "loamspine"),
    ("certificate.mint", "loamspine"),
    // Attribution (sweetGrass)
    ("braid.create", "sweetgrass"),
    ("braid.commit", "sweetgrass"),
    ("provenance.graph", "sweetgrass"),
    ("provenance.export_provo", "sweetgrass"),
    ("attribution.chain", "sweetgrass"),
    // Discovery (Songbird)
    ("discovery.find_primals", "songbird"),
    ("discovery.announce", "songbird"),
];

/// Backward-compatible combined capability list (local + routed method names).
///
/// Prefer [`LOCAL_CAPABILITIES`] when you need to know what this binary
/// actually serves. This function is for `capability.list` responses.
#[must_use]
pub fn all_capabilities() -> Vec<&'static str> {
    let mut all = LOCAL_CAPABILITIES.to_vec();
    all.extend(ROUTED_CAPABILITIES.iter().map(|(method, _)| *method));
    all
}

/// Backward-compatible alias — points to [`LOCAL_CAPABILITIES`].
pub const CAPABILITIES: &[&str] = LOCAL_CAPABILITIES;

/// Bonding policy for this niche's NUCLEUS composition.
pub const BOND_TYPE: &str = "Metallic";
/// Trust model: all primals within the same FAMILY_ID.
pub const TRUST_MODEL: &str = "InternalNucleus";

/// Conventional directory name for ecosystem IPC sockets.
pub const ECOSYSTEM_SOCKET_DIR: &str = "biomeos";

/// Whether standalone mode is requested (no NUCLEUS / biomeOS).
#[must_use]
pub fn standalone_mode() -> bool {
    std::env::var("HOTSPRING_NO_NUCLEUS").is_ok()
}

/// Operation dependency hints for biomeOS Pathway Learner parallelization.
///
/// Maps each locally served capability to the data inputs it requires,
/// enabling the Pathway Learner to determine which operations can run
/// in parallel.
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
        "composition.health":        { "requires": [] },
        "health.check":              { "requires": [] },
        "health.liveness":           { "requires": [] },
        "health.readiness":          { "requires": [] },
        "capabilities.list":         { "requires": [] },
        "mcp.tools.list":            { "requires": [] },
    })
}

/// Cost estimates for biomeOS scheduling.
///
/// Reference hardware: RTX 4070 12 GB + i9-12900K. Numeric values are
/// sourced from [`crate::tolerances`] named constants where available.
#[must_use]
pub fn cost_estimates() -> serde_json::Value {
    use crate::tolerances::cost;
    serde_json::json!({
        "physics.lattice_qcd":         { "latency_ms": cost::LATTICE_QCD_MS,         "cpu": "high",   "gpu": "required",  "memory_bytes": cost::LATTICE_QCD_BYTES },
        "physics.lattice_gauge_update":{ "latency_ms": cost::GAUGE_UPDATE_MS,        "cpu": "low",    "gpu": "required",  "memory_bytes": cost::GAUGE_UPDATE_BYTES },
        "physics.hmc_trajectory":      { "latency_ms": cost::HMC_TRAJECTORY_MS,      "cpu": "medium", "gpu": "required",  "memory_bytes": cost::HMC_TRAJECTORY_BYTES },
        "physics.wilson_dirac":        { "latency_ms": cost::WILSON_DIRAC_MS,        "cpu": "medium", "gpu": "preferred", "memory_bytes": cost::WILSON_DIRAC_BYTES },
        "physics.molecular_dynamics":  { "latency_ms": cost::MOLECULAR_DYNAMICS_MS,  "cpu": "high",   "gpu": "preferred", "memory_bytes": cost::MOLECULAR_DYNAMICS_BYTES },
        "physics.fluid":              { "latency_ms": cost::FLUID_MS,               "cpu": "high",   "gpu": "preferred", "memory_bytes": cost::FLUID_BYTES },
        "physics.nuclear_eos":        { "latency_ms": cost::NUCLEAR_EOS_MS,         "cpu": "high",   "gpu": "optional",  "memory_bytes": cost::NUCLEAR_EOS_BYTES },
        "physics.thermal":            { "latency_ms": cost::THERMAL_MS,             "cpu": "medium", "memory_bytes": cost::THERMAL_BYTES },
        "physics.radiation":          { "latency_ms": cost::RADIATION_MS,           "cpu": "medium", "memory_bytes": cost::RADIATION_BYTES },
        "compute.df64":               { "latency_ms": cost::DF64_MS,               "cpu": "low",    "gpu": "required",  "memory_bytes": cost::DF64_BYTES },
        "compute.cg_solver":          { "latency_ms": cost::CG_SOLVER_MS,          "cpu": "low",    "gpu": "required",  "memory_bytes": cost::CG_SOLVER_BYTES },
        "compute.gradient_flow":      { "latency_ms": cost::GRADIENT_FLOW_MS,      "cpu": "low",    "gpu": "required",  "memory_bytes": cost::GRADIENT_FLOW_BYTES },
        "compute.f64":                { "latency_ms": cost::F64_MS,                "cpu": "low",    "gpu": "preferred", "memory_bytes": cost::F64_BYTES },
        "composition.health":         { "latency_ms": cost::HEALTH_CHECK_MS,       "cpu": "none",   "memory_bytes": cost::HEALTH_CHECK_BYTES },
        "health.check":               { "latency_ms": cost::HEALTH_CHECK_MS,       "cpu": "none",   "memory_bytes": cost::HEALTH_CHECK_BYTES },
        "health.liveness":            { "latency_ms": cost::HEALTH_CHECK_MS,       "cpu": "none",   "memory_bytes": cost::HEALTH_CHECK_BYTES },
        "health.readiness":           { "latency_ms": cost::HEALTH_READINESS_MS,   "cpu": "none",   "memory_bytes": cost::HEALTH_READINESS_BYTES },
        "capabilities.list":          { "latency_ms": cost::CAPABILITIES_LIST_MS,  "cpu": "none",   "memory_bytes": cost::CAPABILITIES_LIST_BYTES },
        "mcp.tools.list":             { "latency_ms": cost::CAPABILITIES_LIST_MS,  "cpu": "none",   "memory_bytes": cost::CAPABILITIES_LIST_BYTES },
    })
}

/// Semantic mappings: short name to fully qualified capability.
///
/// Used by biomeOS `CapabilityTaxonomy` for cross-primal routing.
/// Maps the physics domain's short operation names to fully qualified
/// capability methods.
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
    ("composition_health", "composition.health"),
    ("liveness", "health.liveness"),
    ("readiness", "health.readiness"),
    ("list", "capabilities.list"),
    ("tools", "mcp.tools.list"),
];

static FAMILY_ID_OVERRIDE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Set the family ID programmatically (thread-safe, first-write-wins).
///
/// Call before any socket resolution or primal discovery. This avoids
/// `unsafe { std::env::set_var }` in Edition 2024.
pub fn set_family_id(id: String) {
    FAMILY_ID_OVERRIDE.set(id).ok();
}

/// Resolve the biomeOS family ID.
///
/// Priority: `set_family_id()` override -> `FAMILY_ID` env -> `BIOMEOS_FAMILY_ID` env -> `"default"`.
#[must_use]
pub fn family_id() -> String {
    if let Some(id) = FAMILY_ID_OVERRIDE.get() {
        return id.clone();
    }
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

/// Register this niche's capabilities with biomeOS.
///
/// Discovers biomeOS at runtime via socket convention, then sends
/// `lifecycle.register` followed by `capability.register` for each
/// domain and individual capability.
///
/// Degrades gracefully if biomeOS is unreachable or if standalone mode
/// is active (`HOTSPRING_NO_NUCLEUS=1`). Physics must never depend on
/// registration success.
///
/// Absorbed from primalSpring `niche::register_with_target()` pattern.
pub fn register_with_target(our_socket: &Path) {
    if standalone_mode() {
        info!(target: "niche", "HOTSPRING_NO_NUCLEUS set — skipping registration");
        return;
    }

    let target = std::env::var("BIOMEOS_PRIMAL").unwrap_or_else(|_| REGISTRATION_TARGET.to_owned());

    let biomeos_socket = {
        let mut found = None;
        for dir in socket_dirs() {
            let fid = family_id();
            let candidate = dir.join(format!("{target}-{fid}.sock"));
            if candidate.exists() {
                found = Some(candidate);
                break;
            }
            let candidate = dir.join(format!("neural-api-{fid}.sock"));
            if candidate.exists() {
                found = Some(candidate);
                break;
            }
        }
        found
    };

    let Some(biomeos_path) = biomeos_socket else {
        info!(
            target: "niche",
            "biomeOS socket not discovered — registration deferred"
        );
        return;
    };

    let sock_str = our_socket.to_string_lossy().to_string();

    let reg_payload = serde_json::json!({
        "name": NICHE_NAME,
        "socket_path": &sock_str,
        "pid": std::process::id(),
        "domain": PRIMAL_DOMAIN,
        "version": NICHE_VERSION,
    });

    if let Ok(reg_json) = serde_json::to_string(&serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "lifecycle.register",
        "params": reg_payload,
    })) {
        match send_registration(&biomeos_path, &reg_json) {
            Ok(()) => info!(target: "biomeos", "registered with lifecycle manager"),
            Err(e) => warn!(target: "biomeos", "lifecycle.register failed (non-fatal): {e}"),
        }
    }

    let physics_mappings = physics_semantic_mappings();
    let domains: &[(&str, serde_json::Value)] = &[
        ("physics", physics_mappings),
        (
            "composition",
            serde_json::json!({ "health": "composition.health" }),
        ),
    ];

    for (domain, mappings) in domains {
        let mut payload = serde_json::json!({
            "primal": NICHE_NAME,
            "capability": domain,
            "socket": &sock_str,
            "semantic_mappings": mappings,
        });
        if *domain == "physics" {
            payload["operation_dependencies"] = operation_dependencies();
            payload["cost_estimates"] = cost_estimates();
        }
        let cap_json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "capability.register",
            "params": payload,
        });
        if let Ok(json_str) = serde_json::to_string(&cap_json) {
            let _ = send_registration(&biomeos_path, &json_str);
        }
    }

    let mut registered = 0u32;
    for cap in LOCAL_CAPABILITIES {
        let cap_json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "capability.register",
            "params": {
                "primal": NICHE_NAME,
                "capability": cap,
                "socket": &sock_str,
                "served_locally": true,
            },
        });
        if let Ok(json_str) = serde_json::to_string(&cap_json)
            && send_registration(&biomeos_path, &json_str).is_ok()
        {
            registered += 1;
        }
    }

    for (cap, provider) in ROUTED_CAPABILITIES {
        let cap_json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "capability.register",
            "params": {
                "primal": NICHE_NAME,
                "capability": cap,
                "socket": &sock_str,
                "served_locally": false,
                "canonical_provider": provider,
            },
        });
        if let Ok(json_str) = serde_json::to_string(&cap_json) {
            let _ = send_registration(&biomeos_path, &json_str);
        }
    }

    let total = LOCAL_CAPABILITIES.len() + ROUTED_CAPABILITIES.len();
    info!(
        target: "biomeos",
        "capabilities registered: {registered} local, {} routed, {total} total, {} domains",
        ROUTED_CAPABILITIES.len(),
        domains.len(),
    );
}

/// Physics-domain semantic mappings for `capability.register`.
#[must_use]
fn physics_semantic_mappings() -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (short, full) in SEMANTIC_MAPPINGS {
        if full.starts_with("physics.") || full.starts_with("compute.") {
            map.insert(
                (*short).to_owned(),
                serde_json::Value::String((*full).to_owned()),
            );
        }
    }
    serde_json::Value::Object(map)
}

/// Best-effort JSON-RPC send over Unix socket (fire-and-forget style).
fn send_registration(socket_path: &std::path::Path, json: &str) -> Result<(), std::io::Error> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    let mut stream = UnixStream::connect(socket_path)?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(json.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;
    let mut buf = vec![0u8; 4096];
    let _ = stream.read(&mut buf);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn niche_constants_non_empty() {
        assert!(!NICHE_NAME.is_empty());
        assert!(!NICHE_DESCRIPTION.is_empty());
        assert!(!PROTO_NUCLEATE.is_empty());
        assert!(!LOCAL_CAPABILITIES.is_empty());
        assert!(!ROUTED_CAPABILITIES.is_empty());
        assert!(!DEPENDENCIES.is_empty());
        assert!(!FRAGMENTS.is_empty());
        assert!(!SEMANTIC_MAPPINGS.is_empty());
    }

    #[test]
    fn all_capabilities_follow_semantic_naming() {
        let all = all_capabilities();
        for cap in &all {
            assert!(
                cap.contains('.'),
                "capability '{cap}' should follow domain.operation format"
            );
        }
    }

    #[test]
    fn no_duplicate_capabilities() {
        let mut seen = std::collections::HashSet::new();
        let all = all_capabilities();
        for cap in &all {
            assert!(seen.insert(cap), "duplicate capability: {cap}");
        }
    }

    #[test]
    fn local_and_routed_are_disjoint() {
        for (routed_method, _) in ROUTED_CAPABILITIES {
            assert!(
                !LOCAL_CAPABILITIES.contains(routed_method),
                "'{routed_method}' is in both LOCAL and ROUTED — pick one"
            );
        }
    }

    #[test]
    fn routed_capabilities_have_providers() {
        for (method, provider) in ROUTED_CAPABILITIES {
            assert!(
                !provider.is_empty(),
                "routed capability '{method}' has empty provider"
            );
        }
    }

    #[test]
    fn capabilities_include_health_and_discovery() {
        assert!(LOCAL_CAPABILITIES.contains(&"health.liveness"));
        assert!(LOCAL_CAPABILITIES.contains(&"health.readiness"));
        assert!(LOCAL_CAPABILITIES.contains(&"capabilities.list"));
    }

    #[test]
    fn dependencies_include_core_primals() {
        let names: Vec<&str> = DEPENDENCIES.iter().map(|d| d.name).collect();
        assert!(names.contains(&"beardog"));
        assert!(names.contains(&"barracuda"));
        assert!(names.contains(&"toadstool"));
    }

    #[test]
    fn semantic_mappings_cover_local_capabilities() {
        for (_, full) in SEMANTIC_MAPPINGS {
            assert!(
                LOCAL_CAPABILITIES.contains(full),
                "semantic mapping target '{full}' not in LOCAL_CAPABILITIES"
            );
        }
    }

    #[test]
    fn operation_dependencies_covers_all_local_capabilities() {
        let deps = operation_dependencies();
        for cap in LOCAL_CAPABILITIES {
            assert!(
                deps.get(cap).is_some(),
                "missing dependency entry for {cap}"
            );
        }
    }

    #[test]
    fn cost_estimates_covers_all_local_capabilities() {
        let costs = cost_estimates();
        for cap in LOCAL_CAPABILITIES {
            assert!(costs.get(cap).is_some(), "missing cost entry for {cap}");
        }
    }

    #[test]
    fn cost_estimates_have_latency_and_memory() {
        let costs = cost_estimates();
        let map = costs.as_object().expect("costs should be object");
        for (key, val) in map {
            assert!(
                val.get("latency_ms").is_some(),
                "cost estimate for '{key}' missing latency_ms"
            );
            assert!(
                val.get("memory_bytes").is_some(),
                "cost estimate for '{key}' missing memory_bytes"
            );
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

    #[test]
    fn registration_target_is_biomeos() {
        assert_eq!(REGISTRATION_TARGET, "biomeos");
    }

    #[test]
    fn register_with_target_graceful_when_biomeos_unreachable() {
        let sock = Path::new("/tmp/hotspring-niche-test-nonexistent.sock");
        register_with_target(sock);
    }

    #[test]
    fn capabilities_match_registry_toml() {
        let toml_str = include_str!("../config/capability_registry.toml");
        let parsed: toml::Value = toml::from_str(toml_str).expect("parse registry");
        let caps_in_toml: Vec<&str> = parsed["capabilities"]
            .as_array()
            .expect("capabilities array")
            .iter()
            .filter_map(|c| c.get("method")?.as_str())
            .collect();

        let all = all_capabilities();
        for code_cap in &all {
            assert!(
                caps_in_toml.contains(code_cap),
                "capability '{code_cap}' is in niche but missing from \
                 config/capability_registry.toml"
            );
        }
        for toml_cap in &caps_in_toml {
            assert!(
                all.contains(toml_cap),
                "capability '{toml_cap}' is in capability_registry.toml but missing from \
                 niche capabilities"
            );
        }
    }

    #[test]
    fn backward_compat_capabilities_alias() {
        assert!(std::ptr::eq(CAPABILITIES, LOCAL_CAPABILITIES));
    }
}
