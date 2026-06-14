// SPDX-License-Identifier: AGPL-3.0-or-later

//! rhizoCrypt DAG session with witness emission.
//!
//! When rhizoCrypt is available (detected via [`crate::primal_bridge::NucleusContext`]), creates
//! an ephemeral DAG session for the run, appends events for each pipeline
//! phase, and dehydrates to a merkle root on exit.
//!
//! **Witness integration (v2):** Each pipeline phase can produce a
//! `WireWitnessRef` hash witness (blake3 of the output). On dehydration,
//! the merkle root can optionally be signed by the crypto provider, producing a
//! signature witness. All witnesses are collected for downstream
//! loamSpine commit and sweetGrass braid.
//!
//! When rhizoCrypt is absent, all operations are no-ops and the session
//! fields stay `None` in the receipt.
//!
//! ## Trio Transaction Semantics
//!
//! Per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md` (absorbed Wave 20):
//!
//! - The trio commit flow (DAG → spine → braid) is **not atomic**.
//! - **DAG without braid** = valid partial provenance.
//! - **Braid without spine** = attribution without permanence.
//! - **No rollback** — DAG sessions are append-only.
//! - **Partial state must be reported** — `commit_provenance` returns
//!   `primals_reached` listing which trio components succeeded.
//! - **Domain logic must not fail** on partial provenance — provenance
//!   is enrichment, not a gate.

use crate::primal_bridge::NucleusContext;
use crate::witness::WireWitnessRef;
use serde::{Deserialize, Serialize};

/// Active DAG session handle.
///
/// Created via [`DagSession::begin`], appended to with [`DagSession::append`],
/// and finalized with [`DagSession::dehydrate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagSession {
    pub session_id: String,
    pub events_appended: usize,
    pub merkle_root: Option<String>,
    #[serde(default)]
    pub witnesses: Vec<WireWitnessRef>,
}

/// Metadata for a single DAG event (one pipeline phase).
#[derive(Debug, Clone, Serialize)]
pub struct DagEvent {
    pub phase: String,
    pub input_hash: Option<String>,
    pub output_hash: Option<String>,
    pub wall_seconds: f64,
    pub summary: serde_json::Value,
}

/// Provenance fields to embed in the JSON receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagProvenance {
    pub dag_session_id: String,
    pub merkle_root: String,
    pub events_count: usize,
    #[serde(default)]
    pub witnesses: Vec<WireWitnessRef>,
}

impl DagSession {
    /// Begin a new DAG session via `dag.session.create`.
    ///
    /// Returns `None` if rhizoCrypt is not available.
    pub fn begin(nucleus: &NucleusContext, label: &str) -> Option<Self> {
        let params = serde_json::json!({
            "description": format!("hotSpring: {label}"),
            "session_type": "General",
        });

        let resp = nucleus
            .call_by_capability("dag", "dag.session.create", params)
            .ok()?;
        // rhizoCrypt returns session ID as plain UUID string or nested object
        let session_id = resp.as_str().map(String::from).or_else(|| {
            resp.get("session_id")
                .and_then(serde_json::Value::as_str)
                .map(String::from)
        })?;

        println!("  rhizoCrypt: DAG session {session_id}");

        Some(Self {
            session_id,
            events_appended: 0,
            merkle_root: None,
            witnesses: Vec::new(),
        })
    }

    /// Append an event to the DAG. If `output_hash` is present, a hash
    /// witness is automatically emitted.
    pub fn append(&mut self, nucleus: &NucleusContext, event: DagEvent) {
        if let Some(ref hash) = event.output_hash {
            self.witnesses.push(WireWitnessRef::hash(
                "hotspring:pipeline",
                hash,
                Some(&format!("dag:{}:phase:{}", self.session_id, event.phase)),
            ));
        }

        self.witnesses.push(WireWitnessRef::checkpoint(
            "hotspring:pipeline",
            &format!("dag:{}:phase:{}", self.session_id, event.phase),
        ));

        let mut metadata: Vec<serde_json::Value> = vec![
            serde_json::json!(["phase", event.phase]),
            serde_json::json!(["wall_seconds", event.wall_seconds.to_string()]),
            serde_json::json!(["spring", "hotSpring"]),
        ];
        if let Some(ref h) = event.input_hash {
            metadata.push(serde_json::json!(["input_hash", h]));
        }
        if let Some(ref h) = event.output_hash {
            metadata.push(serde_json::json!(["output_hash", h]));
        }

        let params = serde_json::json!({
            "session_id": self.session_id,
            "event_type": { "Custom": { "kind": event.phase, "data": event.summary } },
            "parents": [],
            "metadata": metadata,
        });

        match nucleus.call_by_capability("dag", "dag.event.append", params) {
            Ok(_) => {
                self.events_appended += 1;
            }
            Err(e) => {
                log::warn!("rhizoCrypt: append failed — {e}");
            }
        }
    }

    /// Dehydrate the session to a merkle root.
    ///
    /// When the crypto provider is available, signs the merkle root and adds a
    /// signature witness. Returns the complete provenance with all
    /// witnesses collected during the session.
    pub fn dehydrate(mut self, nucleus: &NucleusContext) -> DagProvenance {
        let params = serde_json::json!({
            "session_id": self.session_id,
        });

        let root = match nucleus.call_by_capability("dag", "dag.merkle.root", params) {
            Ok(resp) => resp
                .as_str()
                .map(String::from)
                .or_else(|| {
                    resp.get("merkle_root")
                        .and_then(serde_json::Value::as_str)
                        .map(String::from)
                })
                .unwrap_or_else(|| "unknown".to_string()),
            Err(e) => {
                log::warn!("rhizoCrypt: dehydrate failed — {e}");
                "error".to_string()
            }
        };

        self.merkle_root = Some(root.clone());

        self.witnesses.push(WireWitnessRef::hash(
            "hotspring:pipeline",
            &root,
            Some(&format!("dag:{}:merkle_root", self.session_id)),
        ));

        if let Some(sig_witness) = try_sign_merkle_root(nucleus, &root, &self.session_id) {
            self.witnesses.push(sig_witness);
        }

        println!(
            "  rhizoCrypt: session {} dehydrated → {} ({} witnesses)",
            self.session_id,
            &root[..root.len().min(16)],
            self.witnesses.len(),
        );

        DagProvenance {
            dag_session_id: self.session_id,
            merkle_root: root,
            events_count: self.events_appended,
            witnesses: self.witnesses,
        }
    }
}

/// Attempt to sign the merkle root via the crypto domain provider.
/// Returns `None` if no crypto primal is available or signing fails.
fn try_sign_merkle_root(
    nucleus: &NucleusContext,
    merkle_root: &str,
    session_id: &str,
) -> Option<WireWitnessRef> {
    use crate::base64_encode::encode as b64_encode;

    let message_b64 = b64_encode(merkle_root.as_bytes());

    let params = serde_json::json!({
        "message": message_b64,
    });

    let resp = nucleus
        .call_by_capability("crypto", "crypto.sign_ed25519", params)
        .ok()?;
    let sig_b64 = resp
        .get("result")
        .and_then(|r| r.get("signature"))
        .and_then(serde_json::Value::as_str)?;

    let family = &nucleus.family_id;
    let crypto_name = crate::niche::primal_name_for_domain("crypto").unwrap_or("crypto");
    println!("  {crypto_name}: signed merkle root ({family})");

    Some(WireWitnessRef::beardog_signature(
        family,
        sig_b64,
        Some(&format!("dag:{session_id}:merkle_sign")),
    ))
}

/// Commit provenance via the Wave 20 `nest.commit` signal.
///
/// Dispatches `nest.commit` which biomeOS decomposes into:
/// `event.append` → `crypto.sign` → `content.put` → `session.commit` → `braid.create`.
/// Falls back to direct `entry.append` + `braid.create` multi-call
/// if the signal is unavailable (pre-v3.57 biomeOS).
///
/// **Trio semantics:** The commit flow is not atomic. Each leg is attempted
/// independently. The return value includes `primals_reached` listing which
/// trio components succeeded, per `PROVENANCE_TRIO_INTEGRATION_GUIDE.md`.
/// Partial completion is valid: DAG without braid, braid without spine.
/// This function never returns `None` on reachability failure — it returns
/// partial state so callers can inspect `primals_reached`.
///
/// **Wiring status (May 2026):** Scaffolding ready for integration.
/// Call after [`DagSession::dehydrate`] to commit the session's merkle root
/// to the ledger and braid provenance.
pub fn commit_provenance(
    nucleus: &NucleusContext,
    provenance: &DagProvenance,
    experiment_id: &str,
    paper_ref: Option<&str>,
) -> Option<serde_json::Value> {
    let mut primals_reached: Vec<&str> = Vec::new();

    let signal_params = serde_json::json!({
        "session_id": provenance.dag_session_id,
        "merkle_root": provenance.merkle_root,
        "events_count": provenance.events_count,
        "experiment_id": experiment_id,
        "spring": "hotSpring",
        "paper_ref": paper_ref,
    });

    let dispatch_params = serde_json::json!({
        "signal": "nest.commit",
        "params": signal_params,
    });

    if let Ok(resp) =
        nucleus.call_by_capability("orchestration", "signal.dispatch", dispatch_params)
    {
        log::info!("nest.commit signal dispatched for {experiment_id}");
        primals_reached.push("biomeOS");
        primals_reached.push("rhizoCrypt");
        primals_reached.push("loamSpine");
        primals_reached.push("sweetGrass");
        return resp.get("result").cloned().map(|mut r| {
            if let serde_json::Value::Object(ref mut m) = r {
                m.insert("primals_reached".into(), serde_json::json!(primals_reached));
            }
            r
        });
    }

    log::info!("nest.commit signal unavailable, falling back to multi-call");

    // DAG already committed via DagSession — DAG leg is always reached
    primals_reached.push("rhizoCrypt");

    let ledger_params = serde_json::json!({
        "spine_id": format!("hotspring-{experiment_id}"),
        "data": {
            "experiment_id": experiment_id,
            "dag_session": provenance.dag_session_id,
            "merkle_root": provenance.merkle_root,
            "events_count": provenance.events_count,
            "spring": "hotSpring",
        },
        "metadata": {
            "spring": "hotSpring",
            "experiment_id": experiment_id,
        },
    });
    if nucleus
        .call_by_capability("ledger", "entry.append", ledger_params)
        .is_ok()
    {
        primals_reached.push("loamSpine");
    }

    // sweetGrass braid: canonical braid.create with data_hash/mime_type/size
    let receipt_json = serde_json::to_string(&serde_json::json!({
        "experiment_id": experiment_id,
        "merkle_root": provenance.merkle_root,
        "spring": "hotSpring",
        "paper_ref": paper_ref,
    }))
    .unwrap_or_default();
    let braid_params = serde_json::json!({
        "data_hash": provenance.merkle_root,
        "mime_type": "application/json",
        "size": receipt_json.len(),
        "name": format!("hotSpring:{experiment_id}"),
        "description": paper_ref.map(|r| format!("Reproduction of {r}")),
        "tags": ["hotSpring", "provenance"],
        "source_session": provenance.dag_session_id,
        "source_merkle_root": provenance.merkle_root,
    });
    if nucleus
        .call_by_capability("attribution", "braid.create", braid_params)
        .is_ok()
    {
        primals_reached.push("sweetGrass");
    }

    Some(serde_json::json!({
        "committed": true,
        "fallback": true,
        "experiment_id": experiment_id,
        "primals_reached": primals_reached,
    }))
}

/// Compute BLAKE3 hash of a byte slice and return hex string.
///
/// Used to hash GPU compute inputs/outputs for DAG events and witnesses.
pub fn blake3_hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "hash/encoding tests use expect on known vectors"
)]
mod tests {
    use super::*;

    #[test]
    fn blake3_hex_deterministic() {
        let h1 = blake3_hex(b"hello world");
        let h2 = blake3_hex(b"hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn blake3_hex_differs_for_different_input() {
        let h1 = blake3_hex(b"hello");
        let h2 = blake3_hex(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn base64_encode_known_values() {
        assert_eq!(crate::base64_encode::encode(b"hello"), "aGVsbG8=");
        assert_eq!(crate::base64_encode::encode(b""), "");
        assert_eq!(crate::base64_encode::encode(b"a"), "YQ==");
        assert_eq!(crate::base64_encode::encode(b"ab"), "YWI=");
        assert_eq!(crate::base64_encode::encode(b"abc"), "YWJj");
    }

    #[test]
    fn dag_provenance_serializes() {
        let prov = DagProvenance {
            dag_session_id: "test-session".into(),
            merkle_root: "deadbeef".into(),
            events_count: 3,
            witnesses: vec![WireWitnessRef::hash("test", "abc123", None)],
        };
        let json = serde_json::to_string(&prov).expect("serialize");
        assert!(json.contains("witnesses"));
        assert!(json.contains("blake3:abc123"));
    }
}
