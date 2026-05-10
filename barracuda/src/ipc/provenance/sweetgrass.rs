// SPDX-License-Identifier: AGPL-3.0-or-later

//! sweetGrass IPC client — attribution braid for experiment→paper linking.
//!
//! Links experiment results to published papers and baseCamp reproductions
//! through sweetGrass's attribution braid. Each braid entry connects a
//! computation witness to its scientific provenance.

use crate::primal_bridge::send_jsonrpc;
use serde::{Deserialize, Serialize};

/// An attribution braid entry for sweetGrass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraidEntry {
    /// The computation witness hash (from rhizoCrypt DAG).
    pub witness_hash: String,
    /// Paper reference (DOI or arXiv ID).
    pub paper_ref: String,
    /// Specific figure/table/equation being reproduced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    /// Spring that produced the reproduction.
    pub spring: String,
    /// Pass/fail status of the reproduction.
    pub status: BraidStatus,
}

/// Reproduction status for a braid entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BraidStatus {
    Pass,
    Fail,
    Partial,
}

/// Response from sweetGrass `attribution.braid`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraidResult {
    pub braided: bool,
    pub braid_id: String,
}

/// Submit an attribution braid entry to sweetGrass.
pub fn submit_braid(entry: &BraidEntry) -> Option<BraidResult> {
    let socket = crate::niche::socket_dirs()
        .into_iter()
        .map(|d| d.join("biomeos/sweetgrass.sock"))
        .find(|p| p.exists())?;

    let params = serde_json::to_value(entry).ok()?;
    let resp = send_jsonrpc(&socket, "attribution.braid", &params).ok()?;
    serde_json::from_value(resp).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braid_entry_serializes() {
        let entry = BraidEntry {
            witness_hash: "abc123".into(),
            paper_ref: "arXiv:2405.07871".into(),
            artifact: Some("Figure 3".into()),
            spring: "hotspring".into(),
            status: BraidStatus::Pass,
        };
        let json = serde_json::to_string(&entry);
        assert!(json.is_ok());
    }

    #[test]
    fn submit_returns_none_when_not_running() {
        let entry = BraidEntry {
            witness_hash: "test".into(),
            paper_ref: "test".into(),
            artifact: None,
            spring: "hotspring".into(),
            status: BraidStatus::Pass,
        };
        assert!(submit_braid(&entry).is_none());
    }

    #[test]
    fn braid_status_serde_roundtrip() {
        let pass = serde_json::to_string(&BraidStatus::Pass).expect("serialize BraidStatus::Pass");
        assert_eq!(pass, "\"pass\"");
    }
}
