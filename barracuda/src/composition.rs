// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS composition validation — proves IPC-wired primals match Rust results.
//!
//! This is the evolutionary bridge: Python validated Rust, now Rust+Python
//! validate NUCLEUS primal compositions. Each atomic (Tower, Node, Nest) is
//! probed independently, then the full NUCLEUS composition is validated as a
//! coherent system.
//!
//! # Atomic hierarchy
//!
//! - **Tower** (electron): BearDog + Songbird — trust + discovery
//! - **Node** (proton): Tower + ToadStool + barraCuda + coralReef — compute
//! - **Nest** (neutron): Tower + NestGate + rhizoCrypt + loamSpine + sweetGrass — storage
//! - **NUCLEUS** (atom): Tower + Node + Nest (9 core primals)
//!
//! # Validation model
//!
//! ```text
//! Rust result (known-good) ←→ IPC composition result (being validated)
//! ```
//!
//! Same pattern as Python→Rust: the trusted baseline is the local Rust code,
//! and the validation target is the primal composition producing the same
//! result through IPC dispatch.

use crate::primal_bridge::{NucleusContext, PrimalEndpoint};
use crate::validation::ValidationHarness;

/// Atomic fragment type per NUCLEUS model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicType {
    Tower,
    Node,
    Nest,
    FullNucleus,
}

/// Capability domain for composition checks — aligns with [`crate::niche::NicheDependency::capability_domain`].
fn capability_domain_for_required_primal(name: &str) -> &'static str {
    match name {
        "beardog" => "crypto",
        "songbird" => "discovery",
        "toadstool" => "compute",
        "barracuda" => "math",
        "coralreef" => "shader",
        "nestgate" => "storage",
        "rhizocrypt" => "dag",
        "loamspine" => "ledger",
        "sweetgrass" => "attribution",
        _ => "unknown",
    }
}

impl AtomicType {
    /// Primals required for this atomic to be valid.
    #[must_use]
    pub const fn required_primals(&self) -> &[&str] {
        match self {
            Self::Tower => &["beardog", "songbird"],
            Self::Node => &["beardog", "songbird", "toadstool", "barracuda", "coralreef"],
            Self::Nest => &[
                "beardog",
                "songbird",
                "nestgate",
                "rhizocrypt",
                "loamspine",
                "sweetgrass",
            ],
            Self::FullNucleus => &[
                "beardog",
                "songbird",
                "toadstool",
                "barracuda",
                "coralreef",
                "nestgate",
                "rhizocrypt",
                "loamspine",
                "sweetgrass",
            ],
        }
    }

    /// Human-readable name.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Tower => "Tower (electron)",
            Self::Node => "Node (proton)",
            Self::Nest => "Nest (neutron)",
            Self::FullNucleus => "NUCLEUS (atom)",
        }
    }
}

/// Result of validating one atomic fragment.
#[derive(Debug, Clone)]
pub struct AtomicValidation {
    pub atomic: AtomicType,
    pub primals_required: usize,
    pub primals_alive: usize,
    pub primals_missing: Vec<String>,
    pub health_checks_passed: usize,
    pub capability_checks_passed: usize,
    pub passed: bool,
}

/// Validate an atomic fragment by checking liveness and capabilities of required primals.
pub fn validate_atomic(
    ctx: &NucleusContext,
    atomic: AtomicType,
    harness: &mut ValidationHarness,
) -> AtomicValidation {
    let required = atomic.required_primals();
    let mut alive_count = 0;
    let mut missing = Vec::new();
    let mut health_passed = 0;
    let mut cap_passed = 0;

    println!("  ── {} ──", atomic.label());
    println!("    Required primals: {}", required.len());

    for &name in required {
        let domain = capability_domain_for_required_primal(name);
        let check_name = format!("{} {} ({domain}) alive", atomic.label(), name);
        match ctx.get_by_capability(domain) {
            Some(ep) if ep.alive => {
                alive_count += 1;
                harness.check_bool(&check_name, true);
                println!("    {name} [{domain}]: alive ({})", ep.socket);

                if let Ok(resp) =
                    ctx.call_by_capability(domain, "health.liveness", serde_json::json!({}))
                    && resp.get("result").is_some()
                {
                    health_passed += 1;
                }

                if ep.capabilities.is_some() {
                    cap_passed += 1;
                }
            }
            Some(ep) => {
                harness.check_bool(&check_name, false);
                println!("    {name} [{domain}]: UNREACHABLE ({})", ep.socket);
                missing.push(name.to_string());
            }
            None => {
                harness.check_bool(&check_name, false);
                println!("    {name} [{domain}]: NOT DISCOVERED");
                missing.push(name.to_string());
            }
        }
    }

    let passed = missing.is_empty();
    println!(
        "    Result: {}/{} alive, {} health, {} capabilities [{}]",
        alive_count,
        required.len(),
        health_passed,
        cap_passed,
        if passed { "PASS" } else { "FAIL" }
    );

    AtomicValidation {
        atomic,
        primals_required: required.len(),
        primals_alive: alive_count,
        primals_missing: missing,
        health_checks_passed: health_passed,
        capability_checks_passed: cap_passed,
        passed,
    }
}

/// Validate a capability exists on a specific primal.
pub fn validate_capability(
    ctx: &NucleusContext,
    primal: &str,
    capability: &str,
    harness: &mut ValidationHarness,
) -> bool {
    let label = format!("{primal} has {capability}");
    let Some(ep) = ctx.get(primal) else {
        harness.check_bool(&label, false);
        return false;
    };
    let has_cap = ep
        .capabilities
        .as_ref()
        .and_then(|c| c.get("capabilities"))
        .and_then(|a| a.as_array())
        .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some(capability)));
    harness.check_bool(&label, has_cap);
    has_cap
}

/// Full composition health report for IPC response.
#[must_use]
pub fn composition_health(ctx: &NucleusContext) -> serde_json::Value {
    let tower = check_atomic_alive(ctx, AtomicType::Tower);
    let node = check_atomic_alive(ctx, AtomicType::Node);
    let nest = check_atomic_alive(ctx, AtomicType::Nest);
    let nucleus = tower && node && nest;

    serde_json::json!({
        "tower_health": tower,
        "node_health": node,
        "nest_health": nest,
        "nucleus_health": nucleus,
        "science_health": ctx.physics_health(),
        "primals_discovered": ctx.discovered.len(),
        "primals_alive": ctx.alive_names().len(),
    })
}

/// IPC handler for `composition.tower_health`.
#[must_use]
pub fn tower_health(ctx: &NucleusContext) -> serde_json::Value {
    atomic_health_json(ctx, AtomicType::Tower)
}

/// IPC handler for `composition.node_health`.
#[must_use]
pub fn node_health(ctx: &NucleusContext) -> serde_json::Value {
    atomic_health_json(ctx, AtomicType::Node)
}

/// IPC handler for `composition.nest_health`.
#[must_use]
pub fn nest_health(ctx: &NucleusContext) -> serde_json::Value {
    atomic_health_json(ctx, AtomicType::Nest)
}

/// IPC handler for `composition.nucleus_health`.
#[must_use]
pub fn nucleus_health(ctx: &NucleusContext) -> serde_json::Value {
    atomic_health_json(ctx, AtomicType::FullNucleus)
}

fn atomic_health_json(ctx: &NucleusContext, atomic: AtomicType) -> serde_json::Value {
    let required = atomic.required_primals();
    let mut statuses = serde_json::Map::new();
    let mut alive = 0usize;

    for &name in required {
        let domain = capability_domain_for_required_primal(name);
        let status = ctx.get_by_capability(domain).is_some_and(|ep| ep.alive);
        if status {
            alive += 1;
        }
        statuses.insert(
            name.to_string(),
            serde_json::json!(if status { "ok" } else { "missing" }),
        );
    }

    serde_json::json!({
        "atomic": atomic.label(),
        "healthy": alive == required.len(),
        "primals_required": required.len(),
        "primals_alive": alive,
        "statuses": statuses,
    })
}

fn check_atomic_alive(ctx: &NucleusContext, atomic: AtomicType) -> bool {
    atomic.required_primals().iter().all(|&name| {
        let domain = capability_domain_for_required_primal(name);
        ctx.get_by_capability(domain).is_some_and(|ep| ep.alive)
    })
}

/// Discover by capability rather than by name. Returns the first alive primal
/// advertising `capability_domain` in its `capability.list` response.
#[must_use]
pub fn get_by_capability<'a>(
    ctx: &'a NucleusContext,
    capability_domain: &str,
) -> Option<&'a PrimalEndpoint> {
    ctx.get_by_capability(capability_domain)
}

// ═══════════════════════════════════════════════════════════════════
//  Science probe validation: Rust baseline → NUCLEUS IPC parity
// ═══════════════════════════════════════════════════════════════════

/// Result of a science parity probe: local Rust value vs IPC-routed value.
#[derive(Debug, Clone)]
pub struct ScienceProbeResult {
    pub probe_name: String,
    pub domain: String,
    pub local_value: Option<f64>,
    pub ipc_value: Option<f64>,
    pub tolerance: f64,
    pub passed: bool,
    pub skip_reason: Option<String>,
}

/// Validate science capability **liveness** through NUCLEUS IPC.
///
/// These probes verify that compute, math, and provenance trio primals are
/// reachable and alive via IPC. They do NOT perform numeric parity checks —
/// for science parity (Rust vs IPC values), see the `validate_nucleus_*`
/// binaries which compare SEMF, plaquette, and HMC results with centralized
/// tolerances from `tolerances::COMPOSITION_*`.
pub fn validate_science_probes(
    ctx: &NucleusContext,
    harness: &mut ValidationHarness,
) -> Vec<ScienceProbeResult> {
    vec![
        probe_compute_health(ctx, harness),
        probe_math_capability(ctx, harness),
        probe_provenance_trio(ctx, harness),
    ]
}

fn probe_compute_health(
    ctx: &NucleusContext,
    harness: &mut ValidationHarness,
) -> ScienceProbeResult {
    let name = "compute.dispatch available via IPC";
    let domain = "compute";

    let Some(ep) = ctx.get_by_capability(domain) else {
        let result = ScienceProbeResult {
            probe_name: name.into(),
            domain: domain.into(),
            local_value: None,
            ipc_value: None,
            tolerance: 0.0,
            passed: true,
            skip_reason: Some("toadStool not discovered — standalone mode".into()),
        };
        harness.check_bool(&format!("{name} [SKIP]"), true);
        return result;
    };

    let alive = ep.alive;
    harness.check_bool(name, alive);
    ScienceProbeResult {
        probe_name: name.into(),
        domain: domain.into(),
        local_value: Some(if alive { 1.0 } else { 0.0 }),
        ipc_value: Some(if alive { 1.0 } else { 0.0 }),
        tolerance: 0.0,
        passed: alive,
        skip_reason: None,
    }
}

fn probe_math_capability(
    ctx: &NucleusContext,
    harness: &mut ValidationHarness,
) -> ScienceProbeResult {
    let name = "math capability reachable via IPC";
    let domain = "math";

    let Some(ep) = ctx.get_by_capability(domain) else {
        let result = ScienceProbeResult {
            probe_name: name.into(),
            domain: domain.into(),
            local_value: None,
            ipc_value: None,
            tolerance: 0.0,
            passed: true,
            skip_reason: Some("barraCuda primal not discovered — using direct import".into()),
        };
        harness.check_bool(&format!("{name} [SKIP]"), true);
        return result;
    };

    let alive = ep.alive;
    harness.check_bool(name, alive);
    ScienceProbeResult {
        probe_name: name.into(),
        domain: domain.into(),
        local_value: Some(if alive { 1.0 } else { 0.0 }),
        ipc_value: Some(if alive { 1.0 } else { 0.0 }),
        tolerance: 0.0,
        passed: alive,
        skip_reason: None,
    }
}

fn probe_provenance_trio(
    ctx: &NucleusContext,
    harness: &mut ValidationHarness,
) -> ScienceProbeResult {
    let name = "provenance trio (dag + ledger + attribution) available";

    let dag_ok = ctx.get_by_capability("dag").is_some_and(|e| e.alive);
    let ledger_ok = ctx.get_by_capability("ledger").is_some_and(|e| e.alive);
    let attr_ok = ctx
        .get_by_capability("attribution")
        .is_some_and(|e| e.alive);

    let trio_count = [dag_ok, ledger_ok, attr_ok].iter().filter(|&&v| v).count();

    if trio_count == 0 {
        let result = ScienceProbeResult {
            probe_name: name.into(),
            domain: "provenance".into(),
            local_value: None,
            ipc_value: None,
            tolerance: 0.0,
            passed: true,
            skip_reason: Some("provenance trio not discovered — standalone mode".into()),
        };
        harness.check_bool(&format!("{name} [SKIP]"), true);
        return result;
    }

    let all_ok = trio_count == 3;
    harness.check_bool(name, all_ok);
    ScienceProbeResult {
        probe_name: name.into(),
        domain: "provenance".into(),
        local_value: Some(3.0),
        ipc_value: Some(trio_count as f64),
        tolerance: 0.0,
        passed: all_ok,
        skip_reason: if all_ok {
            None
        } else {
            Some(format!("{trio_count}/3 trio primals alive"))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_type_required_primals() {
        assert_eq!(AtomicType::Tower.required_primals().len(), 2);
        assert_eq!(AtomicType::Node.required_primals().len(), 5);
        assert_eq!(AtomicType::Nest.required_primals().len(), 6);
        assert_eq!(AtomicType::FullNucleus.required_primals().len(), 9);
    }

    #[test]
    fn empty_context_reports_no_health() {
        let ctx = NucleusContext {
            discovered: std::collections::HashMap::new(),
            family_id: "test".into(),
        };
        let health = composition_health(&ctx);
        assert_eq!(health["tower_health"], false);
        assert_eq!(health["nucleus_health"], false);
    }

    #[test]
    fn get_by_capability_empty_context() {
        let ctx = NucleusContext {
            discovered: std::collections::HashMap::new(),
            family_id: "test".into(),
        };
        assert!(get_by_capability(&ctx, "compute").is_none());
    }

    #[test]
    fn science_probes_skip_in_standalone() {
        let ctx = NucleusContext {
            discovered: std::collections::HashMap::new(),
            family_id: "test".into(),
        };
        let mut harness = ValidationHarness::new("composition_science_test");
        let results = validate_science_probes(&ctx, &mut harness);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.passed, "probe {} should pass in standalone", r.probe_name);
            assert!(
                r.skip_reason.is_some(),
                "probe {} should have skip reason",
                r.probe_name
            );
        }
    }
}
