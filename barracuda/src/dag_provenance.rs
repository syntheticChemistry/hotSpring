// SPDX-License-Identifier: AGPL-3.0-or-later

//! rhizoCrypt DAG session with witness emission.
//!
//! When rhizoCrypt is available (detected via [`crate::primal_bridge::NucleusContext`]), creates
//! an ephemeral DAG session for the run, appends events for each pipeline
//! phase, and dehydrates to a merkle root on exit.
//!
//! **Witness integration (v2):** Each pipeline phase can produce a
//! `WireWitnessRef` hash witness (blake3 of the output). On dehydration,
//! the merkle root can optionally be signed by BearDog, producing a
//! signature witness. All witnesses are collected for downstream
//! loamSpine commit and sweetGrass braid.
//!
//! When rhizoCrypt is absent, all operations are no-ops and the session
//! fields stay `None` in the receipt.

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
    /// Begin a new DAG session via `dag.create_session`.
    ///
    /// Returns `None` if rhizoCrypt is not available.
    pub fn begin(nucleus: &NucleusContext, label: &str) -> Option<Self> {
        let params = serde_json::json!({
            "label": label,
            "spring": "hotSpring",
        });

        let resp = nucleus
            .call_by_capability("dag", "dag.session.create", params)
            .ok()?;
        let session_id = resp
            .get("result")
            .and_then(|r| r.get("session_id"))
            .and_then(serde_json::Value::as_str)?
            .to_string();

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

        let params = serde_json::json!({
            "session_id": self.session_id,
            "event": {
                "phase": event.phase,
                "input_hash": event.input_hash,
                "output_hash": event.output_hash,
                "wall_seconds": event.wall_seconds,
                "summary": event.summary,
            },
        });

        match nucleus.call_by_capability("dag", "dag.event.append", params) {
            Ok(_) => {
                self.events_appended += 1;
            }
            Err(e) => {
                eprintln!("  rhizoCrypt: append failed — {e}");
            }
        }
    }

    /// Dehydrate the session to a merkle root.
    ///
    /// When BearDog is available, signs the merkle root and adds a
    /// signature witness. Returns the complete provenance with all
    /// witnesses collected during the session.
    pub fn dehydrate(mut self, nucleus: &NucleusContext) -> DagProvenance {
        let params = serde_json::json!({
            "session_id": self.session_id,
        });

        let root = match nucleus.call_by_capability("dag", "dag.merkle.root", params) {
            Ok(resp) => resp
                .get("result")
                .and_then(|r| r.get("merkle_root"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("unknown")
                .to_string(),
            Err(e) => {
                eprintln!("  rhizoCrypt: dehydrate failed — {e}");
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

/// Attempt to sign the merkle root via BearDog. Returns `None` if
/// BearDog is not available or signing fails.
fn try_sign_merkle_root(
    nucleus: &NucleusContext,
    merkle_root: &str,
    session_id: &str,
) -> Option<WireWitnessRef> {
    use base64_encode::encode as b64_encode;

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
    println!("  BearDog: signed merkle root ({family})");

    Some(WireWitnessRef::beardog_signature(
        family,
        sig_b64,
        Some(&format!("dag:{session_id}:merkle_sign")),
    ))
}

/// Minimal base64 encoding (standard alphabet, no padding dependency).
mod base64_encode {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        for chunk in input.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
            let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
            let triple = (b0 << 16) | (b1 << 8) | b2;

            out.push(ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
            if chunk.len() > 1 {
                out.push(ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(ALPHABET[(triple & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
        out
    }
}

/// Compute BLAKE3 hash of a byte slice and return hex string.
///
/// Used to hash GPU compute inputs/outputs for DAG events and witnesses.
pub fn blake3_hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
        assert_eq!(base64_encode::encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode::encode(b""), "");
        assert_eq!(base64_encode::encode(b"a"), "YQ==");
        assert_eq!(base64_encode::encode(b"ab"), "YWI=");
        assert_eq!(base64_encode::encode(b"abc"), "YWJj");
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
