// SPDX-License-Identifier: AGPL-3.0-or-later

//! loamSpine IPC client — immutable append-only ledger for provenance.
//!
//! Records computation provenance in a loamSpine spine via `entry.append`.
//! Each entry links a rhizoCrypt DAG session/merkle root to a specific
//! experiment, creating an immutable permanent record.
//!
//! Wire contract reference: `loam-spine-api/src/jsonrpc/mod.rs`
//! → `entry.append { spine_id, data, metadata }`.
//! Legacy alias: `session.commit` (for full trio coordination).

use crate::primal_bridge::{NucleusContext, send_jsonrpc};
use serde::{Deserialize, Serialize};

/// A ledger entry for loamSpine provenance.
///
/// Maps to loamSpine's `entry.append` params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpineEntry {
    /// Target spine ID (from `spine.create` or default experiment spine).
    pub spine_id: String,
    /// Entry data — canonical JSON of the provenance record.
    pub data: serde_json::Value,
    /// Metadata key-value pairs.
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

/// Response from loamSpine `entry.append`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryAppendResult {
    /// Entry index in the spine.
    #[serde(default)]
    pub index: u64,
    /// Entry hash (BLAKE3 of canonical data).
    #[serde(default)]
    pub hash: String,
}

/// Append a provenance entry to loamSpine.
///
/// Uses `call_by_capability("ledger", ...)` for capability-based routing.
/// Falls back to direct socket RPC if NUCLEUS routing is unavailable.
pub fn append_entry(entry: &SpineEntry) -> Option<EntryAppendResult> {
    let ctx = NucleusContext::detect();
    let params = serde_json::json!({
        "spine_id": entry.spine_id,
        "data": entry.data,
        "metadata": entry.metadata,
    });

    if let Ok(resp) = ctx.call_by_capability("ledger", "entry.append", params.clone()) {
        return serde_json::from_value(resp).ok();
    }

    let socket = ctx
        .by_domain("ledger")
        .filter(|ep| ep.alive)
        .map(|ep| std::path::PathBuf::from(&ep.socket))?;
    let resp = send_jsonrpc(&socket, "entry.append", &params).ok()?;
    serde_json::from_value(resp).ok()
}

/// Commit a provenance session to loamSpine via `session.commit`.
///
/// Higher-level operation that finalizes a trio session (DAG → spine → braid).
/// Returns `None` if loamSpine is not available.
pub fn commit_session(
    nucleus: &NucleusContext,
    session_id: &str,
    merkle_root: &str,
) -> Option<serde_json::Value> {
    let params = serde_json::json!({
        "session_id": session_id,
        "merkle_root": merkle_root,
    });

    nucleus
        .call_by_capability("ledger", "session.commit", params)
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spine_entry_serializes() {
        let entry = SpineEntry {
            spine_id: "spine-001".into(),
            data: serde_json::json!({
                "experiment_id": "exp-001",
                "dag_session": "dag-abc",
                "merkle_root": "deadbeef",
            }),
            metadata: [("spring".into(), "hotSpring".into())]
                .into_iter()
                .collect(),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("spine-001"));
        assert!(json.contains("deadbeef"));
    }

    #[test]
    fn append_returns_none_when_not_running() {
        let entry = SpineEntry {
            spine_id: "test".into(),
            data: serde_json::json!({}),
            metadata: std::collections::HashMap::new(),
        };
        assert!(append_entry(&entry).is_none());
    }
}
