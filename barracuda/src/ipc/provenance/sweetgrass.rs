// SPDX-License-Identifier: AGPL-3.0-or-later

//! sweetGrass IPC client — W3C PROV-O attribution braids.
//!
//! Creates provenance braids linking computation results to their scientific
//! context (papers, datasets, experiments) through sweetGrass's PROV-O engine.
//!
//! Wire contract reference: `sweet-grass-core/src/braid/mod.rs`
//! → `braid.create { data_hash, mime_type, size, name, description, tags, ... }`.
//! Alias: `attribution.braid` → `braid.create` (GAP-36).

use crate::primal_bridge::{NucleusContext, send_jsonrpc};
use serde::{Deserialize, Serialize};

/// Parameters for creating a sweetGrass braid.
///
/// Maps to sweetGrass `braid.create` canonical params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraidParams {
    /// BLAKE3 hash of the content being attributed.
    pub data_hash: String,
    /// MIME type of the content (e.g. "application/json").
    pub mime_type: String,
    /// Size of the content in bytes.
    pub size: u64,
    /// Human-readable name for the braid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Description of the braid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Tags for categorization.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Source DAG session ID (links to rhizoCrypt).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_session: Option<String>,
    /// Source merkle root (links to dehydrated DAG).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_merkle_root: Option<String>,
}

/// Reproduction status for hotSpring-specific braid metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BraidStatus {
    Pass,
    Fail,
    Partial,
}

/// Response from sweetGrass `braid.create`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraidResult {
    /// The created braid ID (URN format: `urn:braid:...`).
    pub braid_id: String,
}

/// Create a provenance braid in sweetGrass.
///
/// Uses `call_by_capability("attribution", ...)` for capability-based routing.
/// Falls back to direct socket RPC if NUCLEUS routing is unavailable.
pub fn create_braid(params: &BraidParams) -> Option<BraidResult> {
    let ctx = NucleusContext::detect();
    let wire_params = serde_json::to_value(params).ok()?;

    if let Ok(resp) = ctx.call_by_capability("attribution", "braid.create", wire_params.clone()) {
        if let Some(bid) = resp.get("braid_id").and_then(|v| v.as_str()) {
            return Some(BraidResult {
                braid_id: bid.to_string(),
            });
        }
        return serde_json::from_value(resp).ok();
    }

    let socket = ctx
        .by_domain("attribution")
        .filter(|ep| ep.alive)
        .map(|ep| std::path::PathBuf::from(&ep.socket))?;
    let resp = send_jsonrpc(&socket, "braid.create", &wire_params).ok()?;
    if let Some(bid) = resp.get("braid_id").and_then(|v| v.as_str()) {
        return Some(BraidResult {
            braid_id: bid.to_string(),
        });
    }
    serde_json::from_value(resp).ok()
}

/// Record a contributor for an existing braid.
///
/// Uses `contribution.record` to add an agent contribution.
pub fn record_contribution(
    nucleus: &NucleusContext,
    data_hash: &str,
    agent: &str,
    role: &str,
) -> Option<serde_json::Value> {
    let params = serde_json::json!({
        "hash": data_hash,
        "agent": agent,
        "role": role,
    });
    nucleus
        .call_by_capability("attribution", "contribution.record", params)
        .ok()
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions")]
    use super::*;

    #[test]
    fn braid_params_serializes() {
        let params = BraidParams {
            data_hash: "abc123".into(),
            mime_type: "application/json".into(),
            size: 1024,
            name: Some("QCD plaquette scan".into()),
            description: None,
            tags: vec!["lattice_qcd".into(), "hotspring".into()],
            source_session: Some("sess-001".into()),
            source_merkle_root: Some("deadbeef".into()),
        };
        let json = serde_json::to_string(&params).expect("serialize");
        assert!(json.contains("abc123"));
        assert!(json.contains("application/json"));
        assert!(json.contains("lattice_qcd"));
    }

    #[test]
    fn create_returns_none_when_not_running() {
        let params = BraidParams {
            data_hash: "test".into(),
            mime_type: "text/plain".into(),
            size: 0,
            name: None,
            description: None,
            tags: vec![],
            source_session: None,
            source_merkle_root: None,
        };
        assert!(create_braid(&params).is_none());
    }

    #[test]
    fn braid_status_serde_roundtrip() {
        let pass = serde_json::to_string(&BraidStatus::Pass).expect("serialize BraidStatus::Pass");
        assert_eq!(pass, "\"pass\"");
    }
}
