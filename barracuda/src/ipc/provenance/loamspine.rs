// SPDX-License-Identifier: AGPL-3.0-or-later

//! loamSpine IPC client — distributed ledger for provenance records.
//!
//! Records computation provenance in the loamSpine ledger. Each ledger entry
//! links a DAG witness (from rhizoCrypt) to a specific experiment or
//! validation run.

use crate::primal_bridge::send_jsonrpc;
use serde::{Deserialize, Serialize};

/// A ledger entry for loamSpine provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Experiment or validation ID.
    pub experiment_id: String,
    /// rhizoCrypt DAG node ID.
    pub dag_node_id: String,
    /// Spring that produced the result.
    pub spring: String,
    /// Method used.
    pub method: String,
    /// Metadata (free-form key-value).
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

/// Response from loamSpine `ledger.record`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerRecordResult {
    pub recorded: bool,
    pub ledger_id: String,
}

/// Record a provenance entry in loamSpine.
///
/// Discovery: `by_domain("ledger")` → NUCLEUS capability routing.
pub fn record_entry(entry: &LedgerEntry) -> Option<LedgerRecordResult> {
    let ctx = crate::primal_bridge::NucleusContext::detect();
    let socket = ctx
        .by_domain("ledger")
        .filter(|ep| ep.alive)
        .map(|ep| std::path::PathBuf::from(&ep.socket))?;

    let params = serde_json::to_value(entry).ok()?;
    let resp = send_jsonrpc(&socket, "ledger.record", &params).ok()?;
    serde_json::from_value(resp).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_entry_serializes() {
        let entry = LedgerEntry {
            experiment_id: "exp-001".into(),
            dag_node_id: "dag-abc".into(),
            spring: "hotspring".into(),
            method: "physics.nuclear_eos".into(),
            metadata: std::collections::HashMap::new(),
        };
        let json = serde_json::to_string(&entry);
        assert!(json.is_ok());
    }

    #[test]
    fn record_returns_none_when_not_running() {
        let entry = LedgerEntry {
            experiment_id: "test".into(),
            dag_node_id: "test".into(),
            spring: "hotspring".into(),
            method: "test".into(),
            metadata: std::collections::HashMap::new(),
        };
        assert!(record_entry(&entry).is_none());
    }
}
