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
}
