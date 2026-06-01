// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-gate compute dispatch via biomeOS `capability.call`.
//!
//! Extends the local dispatch pipeline with remote gate routing. When a
//! target gate is specified, calls route through `capability.call` which
//! biomeOS resolves via Songbird mesh to the remote gate's primals.
//!
//! # Routing
//!
//! ```text
//! hotSpring (biomeGate)
//!   → capability.call { gate: "strandGate", capability: "compute", operation: "dispatch.submit" }
//!   → biomeOS (local)
//!     → Songbird mesh :7700
//!       → biomeOS (strandGate)
//!         → toadStool → barraCuda GPU
//! ```
//!
//! # Prerequisites
//!
//! - Songbird federation active with `SONGBIRD_PEERS` seeded
//! - Target gate running NUCLEUS with compute trio
//! - biomeOS `capability.call` proxy operational (fixed Wave 67)

use crate::dag_provenance::blake3_hex;
use crate::error::HotSpringError;
use crate::primal_bridge::NucleusContext;
use serde::{Deserialize, Serialize};

/// A discovered remote gate with compute capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteGate {
    pub gate_name: String,
    pub capabilities: Vec<String>,
    pub peer_id: Option<String>,
}

/// Result of a cross-gate dispatch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossGateResult {
    pub gate: String,
    pub job_id: String,
    pub routed_via: String,
}

/// Discover remote gates with compute capabilities via Songbird mesh.
///
/// Queries `discovery.peers` and probes each for `compute.dispatch.capabilities`.
/// Returns gates that advertise GPU compute.
pub fn discover_compute_gates(nucleus: &NucleusContext) -> Vec<RemoteGate> {
    let mut gates = Vec::new();

    let params = serde_json::json!({});
    let Ok(resp) = nucleus.call_by_capability("discovery", "discovery.peers", params) else {
        return gates;
    };

    let peers = resp
        .as_array()
        .or_else(|| resp.get("peers").and_then(|v| v.as_array()));

    if let Some(peer_list) = peers {
        for peer in peer_list {
            let gate_name = peer
                .get("gate")
                .or_else(|| peer.get("node_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            let peer_id = peer.get("peer_id").and_then(|v| v.as_str()).map(String::from);

            if let Ok(caps_resp) = capability_call(
                nucleus,
                gate_name,
                "compute",
                "dispatch.capabilities",
                &serde_json::json!({}),
            ) {
                let capabilities: Vec<String> = caps_resp
                    .get("capabilities")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                if !capabilities.is_empty() {
                    gates.push(RemoteGate {
                        gate_name: gate_name.to_string(),
                        capabilities,
                        peer_id,
                    });
                }
            }
        }
    }

    gates
}

/// Route a `capability.call` to a specific remote gate via biomeOS.
///
/// biomeOS resolves the gate via Songbird mesh when not local.
pub fn capability_call(
    nucleus: &NucleusContext,
    gate: &str,
    capability: &str,
    operation: &str,
    args: &serde_json::Value,
) -> Result<serde_json::Value, HotSpringError> {
    let params = serde_json::json!({
        "gate": gate,
        "capability": capability,
        "operation": operation,
        "args": args,
    });

    nucleus
        .call_by_capability("orchestration", "capability.call", params)
        .map_err(|e| HotSpringError::Ipc(format!("capability.call to {gate}: {e}")))
}

/// Compile + submit a WGSL shader on a remote gate's compute trio.
///
/// Routes both compilation (coralReef) and dispatch (toadStool) through
/// the target gate's biomeOS via `capability.call`.
pub fn compile_and_submit_remote(
    nucleus: &NucleusContext,
    gate: &str,
    wgsl_source: &str,
    input_data: &[f64],
    bdf: Option<&str>,
) -> Result<CrossGateResult, HotSpringError> {
    let compile_resp = capability_call(
        nucleus,
        gate,
        "shader",
        "compile.wgsl",
        &serde_json::json!({ "wgsl_source": wgsl_source }),
    )?;

    let binary_b64 = compile_resp
        .get("binary_b64")
        .or_else(|| compile_resp.get("binary"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            HotSpringError::Ipc(format!("remote shader compile on {gate}: no binary returned"))
        })?;

    let input_hash = blake3_hex(&serde_json::to_vec(&input_data).unwrap_or_default());

    let mut submit_args = serde_json::json!({
        "binary_b64": binary_b64,
        "input": {
            "data": input_data,
            "format": "f64_array",
        },
        "input_hash": input_hash,
        "spring": "hotSpring",
        "dispatch_mode": "passthrough",
    });

    if let Some(si) = compile_resp.get("shader_info") {
        submit_args["shader_info"] = si.clone();
    }
    if let Some(bdf_val) = bdf {
        submit_args["bdf"] = serde_json::Value::String(bdf_val.to_string());
    }

    let resp = capability_call(nucleus, gate, "compute", "dispatch.submit", &submit_args)?;

    let job_id = resp
        .get("job_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            HotSpringError::Ipc(format!("remote dispatch on {gate}: missing job_id"))
        })?
        .to_string();

    Ok(CrossGateResult {
        gate: gate.to_string(),
        job_id,
        routed_via: "capability.call".to_string(),
    })
}

/// Retrieve the result of a cross-gate dispatch.
pub fn retrieve_result_remote(
    nucleus: &NucleusContext,
    gate: &str,
    job_id: &str,
) -> Result<serde_json::Value, HotSpringError> {
    capability_call(
        nucleus,
        gate,
        "compute",
        "dispatch.result",
        &serde_json::json!({ "job_id": job_id }),
    )
}

/// Query compute capabilities on a remote gate.
pub fn query_capabilities_remote(
    nucleus: &NucleusContext,
    gate: &str,
) -> Result<Vec<String>, HotSpringError> {
    let resp = capability_call(
        nucleus,
        gate,
        "compute",
        "dispatch.capabilities",
        &serde_json::json!({}),
    )?;

    Ok(resp
        .get("capabilities")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default())
}

/// Cross-gate dispatch with optional ionic lease trust layer.
///
/// When `require_lease` is true, negotiates a GPU lease via BearDog
/// ionic bonding before dispatching (GAP-HS-005 integration).
pub fn dispatch_with_lease(
    nucleus: &NucleusContext,
    gate: &str,
    wgsl_source: &str,
    input_data: &[f64],
    bdf: Option<&str>,
    require_lease: bool,
) -> Result<CrossGateResult, HotSpringError> {
    if require_lease {
        let lease_terms = crate::ipc::ionic_lease::GpuLeaseTerms {
            lessee_family: nucleus.family_id.clone(),
            lessor_family: gate.to_string(),
            gpu_adapter: "default".to_string(),
            ttl_seconds: 3600,
            max_dispatches: Some(100),
            workload_type: "compchem".to_string(),
            precision: "f64".to_string(),
        };

        match crate::ipc::ionic_lease::negotiate_gpu_lease(nucleus, lease_terms) {
            Ok(lease) => {
                log::info!(
                    "Ionic GPU lease sealed: contract_id={}, expires={}",
                    lease.contract_id,
                    lease.expires_at
                );
            }
            Err(e) => {
                log::warn!("Ionic lease failed (proceeding without trust): {e}");
                if require_lease {
                    return Err(e);
                }
            }
        }
    }

    compile_and_submit_remote(nucleus, gate, wgsl_source, input_data, bdf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_gate_serializes() {
        let gate = RemoteGate {
            gate_name: "strandGate".into(),
            capabilities: vec!["gpu_f32".into(), "gpu_f64".into()],
            peer_id: Some("peer-001".into()),
        };
        let json = serde_json::to_string(&gate).expect("serialize");
        assert!(json.contains("strandGate"));
        assert!(json.contains("gpu_f64"));
    }

    #[test]
    fn cross_gate_result_serializes() {
        let result = CrossGateResult {
            gate: "strandGate".into(),
            job_id: "job-001".into(),
            routed_via: "capability.call".into(),
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("capability.call"));
    }

    #[test]
    fn discover_returns_empty_without_nucleus() {
        let ctx = NucleusContext::detect();
        let gates = discover_compute_gates(&ctx);
        assert!(gates.is_empty());
    }

    #[test]
    fn capability_call_fails_without_nucleus() {
        let ctx = NucleusContext::detect();
        let result = capability_call(&ctx, "strandGate", "compute", "test", &serde_json::json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn compile_and_submit_remote_fails_without_nucleus() {
        let ctx = NucleusContext::detect();
        let result = compile_and_submit_remote(&ctx, "strandGate", "/* nop */", &[1.0], None);
        assert!(result.is_err());
    }
}
