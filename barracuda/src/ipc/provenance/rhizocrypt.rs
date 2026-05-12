// SPDX-License-Identifier: AGPL-3.0-or-later

//! rhizoCrypt IPC client — DAG-based computation trace.
//!
//! Submits blake3-hashed computation witnesses to the rhizoCrypt DAG for
//! immutable provenance recording. Each physics computation produces a
//! witness that is anchored in the DAG.

use crate::primal_bridge::{NucleusContext, send_jsonrpc};
use serde::{Deserialize, Serialize};

/// A computation witness for the rhizoCrypt DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagWitness {
    /// blake3 hash of the computation output.
    pub hash: String,
    /// Method that produced this result.
    pub method: String,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
    /// Optional parent witness hash (chain provenance).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

/// Response from rhizoCrypt `dag.submit_witness`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagSubmitResult {
    pub accepted: bool,
    pub dag_node_id: String,
}

/// Submit a computation witness to rhizoCrypt.
///
/// Uses `call_by_capability("dag", ...)` for fully capability-based transport.
/// Falls back to direct socket RPC if NUCLEUS routing is unavailable but the
/// endpoint was discovered.
pub fn submit_witness(witness: &DagWitness) -> Option<DagSubmitResult> {
    let ctx = NucleusContext::detect();
    let params = serde_json::to_value(witness).ok()?;

    if let Ok(resp) = ctx.call_by_capability("dag", "dag.submit_witness", params.clone()) {
        return serde_json::from_value(resp).ok();
    }

    let socket = ctx
        .by_domain("dag")
        .filter(|ep| ep.alive)
        .map(|ep| std::path::PathBuf::from(&ep.socket))?;
    let resp = send_jsonrpc(&socket, "dag.submit_witness", &params).ok()?;
    serde_json::from_value(resp).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_witness_serializes() {
        let w = DagWitness {
            hash: "abc123".into(),
            method: "physics.nuclear_eos".into(),
            timestamp: "2026-05-10T09:00:00Z".into(),
            parent: None,
        };
        let json = serde_json::to_string(&w);
        assert!(json.is_ok());
    }

    #[test]
    fn submit_returns_none_when_not_running() {
        let w = DagWitness {
            hash: "test".into(),
            method: "test".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            parent: None,
        };
        assert!(submit_witness(&w).is_none());
    }
}
