// SPDX-License-Identifier: AGPL-3.0-or-later

//! Static capability tables, dependency declarations, and semantic mappings.
//!
//! Extracted from `niche.rs` so that registration and runtime logic are
//! not buried under 400 lines of data tables. These tables are the
//! niche's compile-time self-knowledge and change only when capabilities
//! are added or removed.

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
pub const PROTO_NUCLEATE: &str = "primalSpring/graphs/downstream/downstream_manifest.toml";

/// NUCLEUS particle profile.
pub const PARTICLE_PROFILE: &str = "proton_heavy";

/// NUCLEUS composition model.
pub const COMPOSITION_MODEL: &str = "nucleated";

/// NUCLEUS fragments this spring composes.
pub const FRAGMENTS: &[&str] = &["tower_atomic", "node_atomic", "nest_atomic"];

/// Science domain tag for biomeOS routing.
pub const SCIENCE_DOMAIN: &str = "high_performance_compute";

/// Bonding policy for this niche's NUCLEUS composition.
pub const BOND_TYPE: &str = "Metallic";

/// Trust model: all primals within the same FAMILY_ID.
pub const TRUST_MODEL: &str = "InternalNucleus";

// ── Dependencies ────────────────────────────────────────────────────

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
        name: "skunkbat",
        role: "defense",
        required: false,
        capability_domain: "defense",
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
    NicheDependency {
        name: "petaltongue",
        role: "visualization",
        required: false,
        capability_domain: "visualization",
    },
];

// ── Local capabilities ──────────────────────────────────────────────

/// Capabilities this binary **locally serves** via `dispatch_request`.
pub const LOCAL_CAPABILITIES: &[&str] = &[
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
    "composition.health",
    "health.check",
    "health.liveness",
    "health.readiness",
    "capabilities.list",
    "mcp.tools.list",
];

// ── Routed capabilities ─────────────────────────────────────────────

/// Ecosystem capabilities routed to other primals via biomeOS.
///
/// Each entry: `(method, canonical_provider)`.
pub const ROUTED_CAPABILITIES: &[(&str, &str)] = &[
    // Inference (Squirrel -> neuralSpring)
    ("inference.complete", "squirrel"),
    ("inference.embed", "squirrel"),
    ("inference.models", "squirrel"),
    // Defense / Audit (SkunkBat)
    ("security.audit_log", "skunkbat"),
    ("security.audit_event", "skunkbat"),
    ("defense.scan", "skunkbat"),
    ("defense.status", "skunkbat"),
    ("recon.scan", "skunkbat"),
    ("threat.assess", "skunkbat"),
    // Crypto (BearDog)
    ("crypto.sign_ed25519", "beardog"),
    ("crypto.verify_ed25519", "beardog"),
    // Compute dispatch (toadStool — operational)
    ("compute.dispatch.submit", "toadstool"),
    ("compute.dispatch.capabilities", "toadstool"),
    ("compute.dispatch.result", "toadstool"),
    ("compute.dispatch.status", "toadstool"),
    ("compute.dispatch.forward", "toadstool"),
    // Dispatch pipeline (toadStool — multi-stage ordered dispatch)
    ("compute.dispatch.pipeline.submit", "toadstool"),
    ("compute.dispatch.pipeline.status", "toadstool"),
    // Performance surface + multi-unit routing (toadStool — operational)
    ("compute.performance_surface.report", "toadstool"),
    ("compute.performance_surface.query", "toadstool"),
    ("compute.performance_surface.list", "toadstool"),
    ("compute.route.multi_unit", "toadstool"),
    // GPU introspection (toadStool — operational)
    ("gpu.query_info", "toadstool"),
    ("gpu.query_memory", "toadstool"),
    ("gpu.query_telemetry", "toadstool"),
    // Hardware learning (toadStool S237+ — operational)
    ("compute.hardware.observe", "toadstool"),
    ("compute.hardware.status", "toadstool"),
    ("compute.hardware.vfio_devices", "toadstool"),
    ("compute.hardware.distill", "toadstool"),
    ("compute.hardware.apply", "toadstool"),
    ("compute.hardware.share_recipe", "toadstool"),
    ("compute.hardware.auto_init", "toadstool"),
    ("compute.hardware.auto_init_all", "toadstool"),
    // Tier 2 Live Science API (toadStool — operational)
    ("toadstool.validate", "toadstool"),
    ("toadstool.list_workloads", "toadstool"),
    // Ember device listing (toadStool Phase A/B — operational)
    ("ember.status", "toadstool"),
    ("ember.list", "toadstool"),
    ("ember.reacquire", "toadstool"),
    // Device management (toadStool Phase C — operational S252+)
    ("device.list", "toadstool"),
    ("device.status", "toadstool"),
    ("device.reacquire", "toadstool"),
    ("device.swap", "toadstool"),
    ("device.warm_catch", "toadstool"),
    // VFIO PBDMA dispatch (toadStool S258-S262)
    ("device.vfio.open", "toadstool"),
    ("device.vfio.alloc", "toadstool"),
    ("device.vfio.roundtrip", "toadstool"),
    // GR context init (toadStool S262)
    ("device.gr.init", "toadstool"),
    ("compute.context.init", "toadstool"),
    // MMIO (toadStool S252)
    ("mmio.read32", "toadstool"),
    ("mmio.write32", "toadstool"),
    ("mmio.batch", "toadstool"),
    ("mmio.pramin.read32", "toadstool"),
    ("mmio.bar0.probe", "toadstool"),
    ("mmio.falcon.status", "toadstool"),
    // Ember lifecycle (toadStool Phase D)
    ("ember.warm_cycle", "toadstool"),
    ("ember.adopt_device", "toadstool"),
    ("ember.fecs.state", "toadstool"),
    ("ember.device.health", "toadstool"),
    ("ember.device.recover", "toadstool"),
    // Semantic aliases (toadStool S255)
    ("ember.swap", "toadstool"),
    ("sovereign.boot", "toadstool"),
    // Auth (toadStool MethodGate)
    ("auth.check", "toadstool"),
    ("auth.mode", "toadstool"),
    ("auth.peer_info", "toadstool"),
    // Provenance (toadStool)
    ("provenance.query", "toadstool"),
    ("provenance.get", "toadstool"),
    // Precision advisory (barraCuda v0.4.0)
    ("precision.route", "barracuda"),
    // Shader compilation (coralReef — pure compiler)
    ("shader.compile.wgsl", "coralreef"),
    ("shader.compile.spirv", "coralreef"),
    ("shader.compile.wgsl.multi", "coralreef"),
    ("shader.compile.gemm", "coralreef"),
    ("shader.compile.status", "coralreef"),
    ("shader.compile.capabilities", "coralreef"),
    // Shader dispatch (toadStool)
    ("shader.dispatch", "toadstool"),
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
    // Primal discovery (biomeOS — Wave 20)
    ("primal.list", "biomeos"),
    // Visualization (petalTongue)
    ("visualization.render", "petaltongue"),
    ("visualization.render.scene", "petaltongue"),
    ("visualization.render.stream", "petaltongue"),
    ("interaction.subscribe", "petaltongue"),
    ("interaction.poll", "petaltongue"),
];

// ── Derived helpers ─────────────────────────────────────────────────

/// All capabilities (local + routed method names).
#[must_use]
pub fn all_capabilities() -> Vec<&'static str> {
    let mut all = LOCAL_CAPABILITIES.to_vec();
    all.extend(ROUTED_CAPABILITIES.iter().map(|(method, _)| *method));
    all
}

/// Build the canonical `capability.list` response per Wave 20 schema.
///
/// Returns `{ "capabilities": [...], "count": N, "primal": "hotspring" }`.
/// Extra fields are allowed by the standard; this provides the required
/// canonical subset that downstream consumers (projectNUCLEUS,
/// projectFOUNDATION) depend on.
#[must_use]
pub fn capabilities_list_response() -> serde_json::Value {
    let caps = all_capabilities();
    serde_json::json!({
        "capabilities": caps,
        "count": caps.len(),
        "primal": NICHE_NAME,
    })
}

/// Backward-compatible alias — points to [`LOCAL_CAPABILITIES`].
pub const CAPABILITIES: &[&str] = LOCAL_CAPABILITIES;

// ── Semantic mappings ───────────────────────────────────────────────

/// Short name → fully qualified capability for biomeOS `CapabilityTaxonomy`.
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

// ── Cost + dependency metadata ──────────────────────────────────────

/// Operation dependency hints for biomeOS Pathway Learner parallelization.
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
/// Reference hardware: RTX 4070 12 GB + i9-12900K.
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

/// Physics-domain semantic mappings for `capability.register`.
#[must_use]
pub fn physics_semantic_mappings() -> serde_json::Value {
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
