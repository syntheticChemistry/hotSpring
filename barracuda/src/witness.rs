// SPDX-License-Identifier: AGPL-3.0-or-later

//! `WireWitnessRef` — self-describing provenance events.
//!
//! Implements the witness wire type from `ATTESTATION_ENCODING_STANDARD.md`
//! v2.0.0. Each witness carries its own metadata (kind, encoding, algorithm,
//! tier) so the provenance trio can transport it opaquely.
//!
//! hotSpring uses witnesses for:
//! - `kind: "hash"` — blake3 of GPU compute output tensors
//! - `kind: "checkpoint"` — pipeline phase boundaries
//! - `kind: "signature"` — BearDog Ed25519 signatures (when available)

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Self-describing provenance event per `ATTESTATION_ENCODING_STANDARD.md`.
///
/// Each trio primal owns its own copy (primal sovereignty). This is
/// hotSpring's copy — the JSON shape is the interop contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireWitnessRef {
    /// Who/what produced this witness (e.g. "hotspring:pipeline").
    pub agent: String,

    /// Discriminant: "signature", "hash", "checkpoint", "marker", "timestamp".
    #[serde(default = "default_kind")]
    pub kind: String,

    /// Opaque evidence payload (sig bytes, hash hex, marker text).
    #[serde(default)]
    pub evidence: String,

    /// Nanosecond Unix timestamp.
    #[serde(default)]
    pub witnessed_at: u64,

    /// How `evidence` is encoded: "hex", "base64", "utf8", "none".
    #[serde(default = "default_encoding")]
    pub encoding: String,

    /// Crypto algorithm when `kind = "signature"` (e.g. "ed25519").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,

    /// Trust tier: "local", "gateway", "anchor", "external", "open".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,

    /// Freeform context (experiment ID, pipeline phase, thread).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

fn default_kind() -> String {
    "signature".into()
}

fn default_encoding() -> String {
    "hex".into()
}

fn now_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

impl WireWitnessRef {
    /// Create a blake3 hash observation witness.
    ///
    /// Used after GPU compute completes: hash the output tensor, record
    /// as a witness so re-running the compute can compare hashes.
    #[must_use]
    pub fn hash(agent: &str, blake3_hex: &str, context: Option<&str>) -> Self {
        Self {
            agent: agent.into(),
            kind: "hash".into(),
            evidence: format!("blake3:{blake3_hex}"),
            witnessed_at: now_nanos(),
            encoding: "utf8".into(),
            algorithm: None,
            tier: Some("open".into()),
            context: context.map(Into::into),
        }
    }

    /// Create a pipeline checkpoint witness (no crypto).
    #[must_use]
    pub fn checkpoint(agent: &str, context: &str) -> Self {
        Self {
            agent: agent.into(),
            kind: "checkpoint".into(),
            evidence: String::new(),
            witnessed_at: now_nanos(),
            encoding: "none".into(),
            algorithm: None,
            tier: Some("open".into()),
            context: Some(context.into()),
        }
    }

    /// Create a signature witness from BearDog's `crypto.sign_ed25519` response.
    ///
    /// Maps BearDog fields to WireWitnessRef per the trio witness harvest
    /// handoff: evidence = base64 signature, encoding = "base64",
    /// algorithm = "ed25519", tier = "local".
    #[must_use]
    pub fn beardog_signature(gate_id: &str, signature_base64: &str, context: Option<&str>) -> Self {
        Self {
            agent: format!("beardog:{gate_id}"),
            kind: "signature".into(),
            evidence: signature_base64.into(),
            witnessed_at: now_nanos(),
            encoding: "base64".into(),
            algorithm: Some("ed25519".into()),
            tier: Some("local".into()),
            context: context.map(Into::into),
        }
    }

    /// Create a bare timestamp witness (attests only that an event occurred).
    #[must_use]
    pub fn timestamp(agent: &str, context: Option<&str>) -> Self {
        Self {
            agent: agent.into(),
            kind: "timestamp".into(),
            evidence: String::new(),
            witnessed_at: now_nanos(),
            encoding: "none".into(),
            algorithm: None,
            tier: Some("open".into()),
            context: context.map(Into::into),
        }
    }

    /// Convert to JSON value for embedding in IPC params.
    pub fn to_json(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_witness_round_trip() {
        let w = WireWitnessRef::hash(
            "hotspring:pipeline",
            "abc123def456",
            Some("experiment:152:gpu_output"),
        );
        assert_eq!(w.kind, "hash");
        assert_eq!(w.encoding, "utf8");
        assert!(w.evidence.starts_with("blake3:"));
        assert_eq!(w.tier.as_deref(), Some("open"));
        assert!(w.witnessed_at > 0);

        let json = serde_json::to_string(&w).expect("serialize");
        let back: WireWitnessRef = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.kind, "hash");
        assert_eq!(back.evidence, w.evidence);
    }

    #[test]
    fn checkpoint_witness_fields() {
        let w = WireWitnessRef::checkpoint("hotspring:pipeline", "phase:dispatch:submit");
        assert_eq!(w.kind, "checkpoint");
        assert_eq!(w.encoding, "none");
        assert!(w.evidence.is_empty());
        assert_eq!(w.context.as_deref(), Some("phase:dispatch:submit"));
    }

    #[test]
    fn beardog_signature_witness_fields() {
        let w = WireWitnessRef::beardog_signature(
            "biomegate",
            "AAAA==",
            Some("experiment:152:merkle_sign"),
        );
        assert_eq!(w.kind, "signature");
        assert_eq!(w.encoding, "base64");
        assert_eq!(w.algorithm.as_deref(), Some("ed25519"));
        assert_eq!(w.tier.as_deref(), Some("local"));
        assert_eq!(w.agent, "beardog:biomegate");
    }

    #[test]
    fn timestamp_witness_fields() {
        let w = WireWitnessRef::timestamp("hotspring:pipeline", Some("pipeline:start"));
        assert_eq!(w.kind, "timestamp");
        assert_eq!(w.encoding, "none");
        assert!(w.evidence.is_empty());
    }

    #[test]
    fn default_kind_is_signature() {
        let json = r#"{"agent":"test"}"#;
        let w: WireWitnessRef = serde_json::from_str(json).expect("deserialize minimal");
        assert_eq!(w.kind, "signature");
        assert_eq!(w.encoding, "hex");
    }

    #[test]
    fn to_json_succeeds() {
        let w = WireWitnessRef::hash("test", "deadbeef", None);
        let v = w.to_json().expect("to_json");
        assert_eq!(v["kind"], "hash");
    }
}
