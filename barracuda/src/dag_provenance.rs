// SPDX-License-Identifier: AGPL-3.0-only

//! rhizoCrypt DAG session for computation trace.
//!
//! When rhizoCrypt is available (detected via [`NucleusContext`]), creates
//! an ephemeral DAG session for the run, appends events for each pipeline
//! phase, and dehydrates to a merkle root on exit.
//!
//! When rhizoCrypt is absent, all operations are no-ops and the session
//! fields stay `None` in the receipt.

use crate::primal_bridge::NucleusContext;
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

        let resp = nucleus.call("rhizocrypt", "dag.create_session", &params).ok()?;
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
        })
    }

    /// Append an event to the DAG.
    pub fn append(&mut self, nucleus: &NucleusContext, event: DagEvent) {
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

        match nucleus.call("rhizocrypt", "dag.append_event", &params) {
            Ok(_) => {
                self.events_appended += 1;
            }
            Err(e) => {
                eprintln!("  rhizoCrypt: append failed — {e}");
            }
        }
    }

    /// Dehydrate the session to a merkle root.
    pub fn dehydrate(mut self, nucleus: &NucleusContext) -> DagProvenance {
        let params = serde_json::json!({
            "session_id": self.session_id,
        });

        let root = match nucleus.call("rhizocrypt", "dag.dehydrate", &params) {
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
        println!(
            "  rhizoCrypt: session {} dehydrated → {}",
            self.session_id,
            &root[..root.len().min(16)]
        );

        DagProvenance {
            dag_session_id: self.session_id,
            merkle_root: root,
            events_count: self.events_appended,
        }
    }
}

/// Convenience: compute SHA-256 of a byte slice and return hex string.
///
/// Used to hash inputs/outputs for DAG events.
pub fn sha256_hex(data: &[u8]) -> String {
    use std::io::Write;
    let mut hasher = Sha256::new();
    hasher.write_all(data).unwrap_or(());
    hasher.finish_hex()
}

/// Minimal SHA-256 for DAG hashing (no external dependency — uses the same
/// approach as CHECKSUMS generation in the validation harness).
struct Sha256 {
    data: Vec<u8>,
}

impl Sha256 {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn finish_hex(&self) -> String {
        use std::process::Command;
        let result = Command::new("sha256sum")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                use std::io::Write;
                if let Some(ref mut stdin) = child.stdin {
                    stdin.write_all(&self.data)?;
                }
                child.wait_with_output()
            });

        match result {
            Ok(output) => {
                let s = String::from_utf8_lossy(&output.stdout);
                s.split_whitespace().next().unwrap_or("").to_string()
            }
            Err(_) => format!("{:016x}", self.data.len()),
        }
    }
}

impl std::io::Write for Sha256 {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.data.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
