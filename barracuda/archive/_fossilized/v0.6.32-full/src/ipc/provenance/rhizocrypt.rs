// SPDX-License-Identifier: AGPL-3.0-or-later

//! rhizoCrypt IPC client — DAG-based computation trace.
//!
//! Appends computation events to the rhizoCrypt DAG for immutable provenance.
//! Uses `dag.event.append` (canonical wire method) with `EventType::Custom`
//! payloads. Each physics computation produces an event vertex anchored in
//! the session's Merkle DAG.
//!
//! Wire contract reference: `rhizo-crypt-rpc/src/service_types.rs`
//! → `AppendEventRequest { session_id, event_type, agent, parents, metadata, payload_ref }`.

use crate::primal_bridge::{NucleusContext, send_jsonrpc};
use serde::{Deserialize, Serialize};

/// A computation event for the rhizoCrypt DAG.
///
/// Maps to rhizoCrypt's `AppendEventRequest` with `EventType::Custom`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagEvent {
    /// Active session ID (from `dag.session.create`).
    pub session_id: String,
    /// Event type — wrapped as `{ "Custom": { "kind": ..., "data": ... } }`.
    pub kind: String,
    /// blake3 hash of the computation output (stored as metadata).
    pub output_hash: Option<String>,
    /// Optional parent vertex IDs (empty = append to frontier).
    #[serde(default)]
    pub parents: Vec<String>,
    /// Free-form metadata key-value pairs.
    #[serde(default)]
    pub metadata: Vec<(String, String)>,
}

/// Response from rhizoCrypt `dag.event.append`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagAppendResult {
    /// Vertex ID of the appended event (BLAKE3 of canonical CBOR).
    pub vertex_id: String,
}

/// Append a computation event to the rhizoCrypt DAG.
///
/// Uses `call_by_capability("dag", ...)` for capability-based routing.
/// Falls back to direct socket RPC if NUCLEUS routing is unavailable.
pub fn append_event(event: &DagEvent) -> Option<DagAppendResult> {
    let ctx = NucleusContext::detect();

    let mut metadata_pairs: Vec<serde_json::Value> = event
        .metadata
        .iter()
        .map(|(k, v)| serde_json::json!([k, v]))
        .collect();
    if let Some(ref h) = event.output_hash {
        metadata_pairs.push(serde_json::json!(["output_hash", h]));
    }

    let params = serde_json::json!({
        "session_id": event.session_id,
        "event_type": { "Custom": { "kind": event.kind, "data": {} } },
        "parents": event.parents,
        "metadata": metadata_pairs,
    });

    if let Ok(resp) = ctx.call_by_capability("dag", "dag.event.append", params.clone()) {
        if let Some(vid) = resp.as_str() {
            return Some(DagAppendResult {
                vertex_id: vid.to_string(),
            });
        }
        return serde_json::from_value(resp).ok();
    }

    let socket = ctx
        .by_domain("dag")
        .filter(|ep| ep.alive)
        .map(|ep| std::path::PathBuf::from(&ep.socket))?;
    let resp = send_jsonrpc(&socket, "dag.event.append", &params).ok()?;
    if let Some(vid) = resp.as_str() {
        return Some(DagAppendResult {
            vertex_id: vid.to_string(),
        });
    }
    serde_json::from_value(resp).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_event_serializes() {
        let e = DagEvent {
            session_id: "sess-001".into(),
            kind: "physics.nuclear_eos".into(),
            output_hash: Some("abc123".into()),
            parents: vec![],
            metadata: vec![("spring".into(), "hotSpring".into())],
        };
        let json = serde_json::to_string(&e).expect("serialize");
        assert!(json.contains("nuclear_eos"));
    }

    #[test]
    fn append_returns_none_when_not_running() {
        let e = DagEvent {
            session_id: "test".into(),
            kind: "test".into(),
            output_hash: None,
            parents: vec![],
            metadata: vec![],
        };
        assert!(append_event(&e).is_none());
    }
}
