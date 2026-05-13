// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tier 2 Live Science API client — toadStool workload pre-flight and
//! barraCuda precision advisory via NUCLEUS IPC.
//!
//! Tier 2 convergence wires hotSpring into the upstream APIs that became
//! live with toadStool S250 and barraCuda `precision.rs`. Springs call
//! `toadstool.validate` for workload pre-flight and
//! `barracuda.precision.route` for precision advisory.
//!
//! Degrades gracefully: when toadStool or barraCuda primals are not
//! reachable, functions return `None` and callers fall through to local
//! validation.

use crate::primal_bridge::NucleusContext;
use serde::{Deserialize, Serialize};

/// Workload pre-flight result from `toadstool.validate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadPreflight {
    /// Whether the workload can be dispatched.
    pub valid: bool,
    /// Whether the required GPU is available.
    pub gpu_available: bool,
    /// Precision tier available for this workload.
    #[serde(default)]
    pub precision_tier: Option<String>,
    /// Estimated dispatch time in milliseconds.
    #[serde(default)]
    pub estimated_dispatch_time_ms: Option<u64>,
    /// Warnings about suboptimal configuration.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Capabilities required by this workload.
    #[serde(default)]
    pub required_capabilities: Vec<String>,
}

/// Workload catalog entry from `toadstool.list_workloads`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadEntry {
    pub id: serde_json::Value,
    #[serde(default)]
    pub job_type: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
}

/// Workload listing result from `toadstool.list_workloads`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadListing {
    pub jobs: Vec<serde_json::Value>,
    #[serde(default)]
    pub counts: serde_json::Value,
}

/// Precision routing advisory from `barracuda.precision.route` (v0.4.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionAdvisory {
    /// Recommended precision tier name (e.g. "F32", "DF64", "F64").
    #[serde(default, alias = "recommended_tier")]
    pub tier: Option<String>,
    /// Whether FMA fusion is safe for this domain.
    #[serde(default)]
    pub fma_safe: bool,
    /// Whether this tier needs sovereign compilation (coralReef).
    #[serde(default)]
    pub requires_compiler: bool,
    /// Recommended hardware dispatch hint ("compute", "tensor_core", etc.).
    #[serde(default)]
    pub hardware_hint: Option<String>,
    /// Human-readable rationale for the recommendation.
    #[serde(default)]
    pub rationale: Option<String>,
    /// Whether sovereign compile is needed vs wgpu polyfill.
    #[serde(default)]
    pub needs_sovereign_compile: bool,
    /// GPU adapter name (null if no GPU available).
    #[serde(default)]
    pub adapter: Option<String>,
    /// Active dispatch path: `"wgpu"`, `"sovereign"`, or `"unavailable"` (Sprint 64+).
    #[serde(default)]
    pub dispatch_path: Option<String>,
}

/// Pre-flight a workload via `toadstool.validate`.
///
/// Returns `None` when toadStool is unreachable or the method is not
/// available. Falls back to `TOADSTOOL_SOCKET` env var when NUCLEUS
/// discovery is not running.
pub fn workload_preflight(
    nucleus: &NucleusContext,
    workload_name: &str,
) -> Option<WorkloadPreflight> {
    let params = serde_json::json!({
        "workload": workload_name,
        "format": "json",
    });

    let resp = nucleus
        .call_by_capability("compute", "toadstool.validate", params)
        .ok()?;

    serde_json::from_value(resp).ok()
}

/// List workloads known to toadStool via `toadstool.list_workloads`.
///
/// Returns `None` when toadStool is unreachable.
pub fn list_workloads(nucleus: &NucleusContext) -> Option<WorkloadListing> {
    let params = serde_json::json!({});

    let resp = nucleus
        .call_by_capability("compute", "toadstool.list_workloads", params)
        .ok()?;

    serde_json::from_value(resp).ok()
}

/// Query precision routing advisory from barraCuda via `precision.route`.
///
/// `domain` is a snake_case physics domain name (e.g. "lattice_qcd",
/// "gradient_flow", "molecular_dynamics"). barraCuda v0.4.0 returns
/// `recommended_tier`, `fma_safe`, `requires_compiler`, `hardware_hint`,
/// `rationale`, `needs_sovereign_compile`, and `adapter`.
///
/// Returns `None` when barraCuda is unreachable or the method is not available.
pub fn precision_advisory(nucleus: &NucleusContext, domain: &str) -> Option<PrecisionAdvisory> {
    let params = serde_json::json!({
        "domain": domain,
    });

    let resp = nucleus
        .call_by_capability("math", "precision.route", params)
        .ok()?;

    serde_json::from_value(resp).ok()
}

/// Query toadStool's dispatch capabilities via `compute.dispatch.capabilities`.
///
/// S243+ returns `render_node` and `device_id` on DRM GPU objects, plus
/// `ember.phase` ("B" as of Phase B absorption) and `ember.held_devices`
/// count. Returns `None` when toadStool is unreachable.
pub fn dispatch_capabilities(nucleus: &NucleusContext) -> Option<serde_json::Value> {
    nucleus
        .call_by_capability(
            "compute",
            "compute.dispatch.capabilities",
            serde_json::json!({}),
        )
        .ok()
}

/// Check Tier 2 readiness for hotSpring's validation pipeline.
///
/// Returns a summary of which Tier 2 services are reachable and what
/// capabilities they report. Used by `hotspring_unibin status` and
/// validation pre-flight.
pub fn tier2_status(nucleus: &NucleusContext) -> Tier2Status {
    let toadstool_alive = nucleus.by_domain("compute").is_some_and(|ep| ep.alive);

    let barracuda_alive = nucleus.by_domain("math").is_some_and(|ep| ep.alive);

    let preflight_available = if toadstool_alive {
        workload_preflight(nucleus, "__probe__").is_some()
    } else {
        false
    };

    let precision_available = if barracuda_alive {
        precision_advisory(nucleus, "lattice_qcd").is_some()
    } else {
        false
    };

    Tier2Status {
        toadstool_alive,
        barracuda_alive,
        preflight_available,
        precision_available,
    }
}

/// Tier 2 readiness summary.
#[derive(Debug, Clone, Serialize)]
pub struct Tier2Status {
    /// Whether toadStool (compute domain) is reachable.
    pub toadstool_alive: bool,
    /// Whether barraCuda (math domain) is reachable.
    pub barracuda_alive: bool,
    /// Whether `toadstool.validate` responds to pre-flight probes.
    pub preflight_available: bool,
    /// Whether `precision.route` responds to advisory queries.
    pub precision_available: bool,
}

impl Tier2Status {
    /// True when both Tier 2 services are fully operational.
    #[must_use]
    pub fn fully_wired(&self) -> bool {
        self.preflight_available && self.precision_available
    }

    /// Record Tier 2 status as validation checks on a harness.
    pub fn check(&self, v: &mut crate::validation::ValidationHarness) {
        v.check_bool("tier2:toadstool_alive", self.toadstool_alive);
        v.check_bool("tier2:barracuda_alive", self.barracuda_alive);
        v.check_bool("tier2:preflight_available", self.preflight_available);
        v.check_bool("tier2:precision_available", self.precision_available);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier2_status_not_wired_when_empty() {
        let nucleus = NucleusContext::detect();
        let status = tier2_status(&nucleus);
        assert!(!status.fully_wired());
    }

    #[test]
    fn workload_preflight_returns_none_without_toadstool() {
        let nucleus = NucleusContext::detect();
        assert!(workload_preflight(&nucleus, "test-workload").is_none());
    }

    #[test]
    fn list_workloads_returns_none_without_toadstool() {
        let nucleus = NucleusContext::detect();
        assert!(list_workloads(&nucleus).is_none());
    }

    #[test]
    fn precision_advisory_returns_none_without_barracuda() {
        let nucleus = NucleusContext::empty("test");
        assert!(precision_advisory(&nucleus, "lattice_qcd").is_none());
    }

    #[test]
    fn dispatch_capabilities_returns_none_without_toadstool() {
        let nucleus = NucleusContext::empty("test");
        assert!(dispatch_capabilities(&nucleus).is_none());
    }

    #[test]
    fn preflight_struct_deserializes_minimal() {
        let json = serde_json::json!({
            "valid": true,
            "gpu_available": false,
        });
        let pf: WorkloadPreflight = serde_json::from_value(json).unwrap();
        assert!(pf.valid);
        assert!(!pf.gpu_available);
        assert!(pf.warnings.is_empty());
    }
}
