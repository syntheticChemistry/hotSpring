// SPDX-License-Identifier: AGPL-3.0-or-later

//! Provenance braid pipeline for hotSpring guidestone artifacts.
//!
//! Bridges the gap between computation (compchem module) and the
//! lithoSpore → sporePrint content pipeline by:
//!
//! 1. Creating a DAG session for the guidestone validation run
//! 2. Appending per-module events (BLAKE3-hashed FES outputs)
//! 3. Dehydrating to a merkle root
//! 4. Committing provenance via the trio pipeline
//! 5. Emitting a canonical `FermentTranscript` for lithoSpore ingestion
//!
//! Wire format: `pseudospore-core::FermentTranscript` (flat JSON).

use crate::dag_provenance::{DagEvent, DagProvenance, DagSession, blake3_hex};
use crate::primal_bridge::NucleusContext;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Canonical ferment transcript — flat JSON for `provenance/ferment_transcript.json`.
///
/// Must match `pseudospore-core::braid_envelope::FermentTranscript` wire format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FermentTranscript {
    pub dataset_id: String,
    pub spring: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spring_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub braid_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dag_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dag_merkle_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spine_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub computation: Option<serde_json::Value>,
}

/// Result of a guidestone provenance pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuideStoneProvenance {
    pub transcript: FermentTranscript,
    pub dag: Option<DagProvenance>,
    pub primals_reached: Vec<String>,
}

/// Module-level computation event for DAG recording.
pub struct ModuleEvent {
    /// Module name (e.g. "01_free_xylose_1d").
    pub name: String,
    /// Output file path for BLAKE3 hashing.
    pub output_path: Option<String>,
    /// Output bytes for BLAKE3 hashing (if file not available).
    pub output_bytes: Option<Vec<u8>>,
    /// Wall-clock seconds for this module's validation.
    pub wall_seconds: f64,
    /// Module-specific summary data.
    pub summary: serde_json::Value,
}

/// Run the full provenance braid pipeline for a guidestone artifact.
///
/// Attempts the complete trio pipeline when NUCLEUS is available,
/// gracefully degrades to local-only transcript when offline.
pub fn run_guidestone_provenance(
    dataset_id: &str,
    version: &str,
    paper_ref: Option<&str>,
    modules: &[ModuleEvent],
) -> GuideStoneProvenance {
    let now = chrono_iso8601_now();
    let nucleus = NucleusContext::detect();

    let input_hashes = serde_json::Map::new();
    let mut output_hashes = serde_json::Map::new();

    let dag = if let Some(mut session) = DagSession::begin(&nucleus, dataset_id) {
        for module in modules {
            let output_hash = compute_output_hash(module);

            if let Some(ref h) = output_hash {
                output_hashes.insert(module.name.clone(), serde_json::json!(format!("blake3:{h}")));
            }

            session.append(
                &nucleus,
                DagEvent {
                    phase: module.name.clone(),
                    input_hash: None,
                    output_hash,
                    wall_seconds: module.wall_seconds,
                    summary: module.summary.clone(),
                },
            );
        }

        Some(session.dehydrate(&nucleus))
    } else {
        for module in modules {
            if let Some(ref h) = compute_output_hash(module) {
                output_hashes.insert(module.name.clone(), serde_json::json!(format!("blake3:{h}")));
            }
        }
        None
    };

    let mut primals_reached = Vec::new();
    let mut braid_id = None;
    let mut spine_id = None;

    if let Some(ref prov) = dag {
        primals_reached.push("rhizoCrypt".to_string());

        if let Some(commit_result) =
            crate::dag_provenance::commit_provenance(&nucleus, prov, dataset_id, paper_ref)
        {
            if let Some(reached) = commit_result.get("primals_reached") {
                if let Some(arr) = reached.as_array() {
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            if !primals_reached.contains(&s.to_string()) {
                                primals_reached.push(s.to_string());
                            }
                        }
                    }
                }
            }
            braid_id = commit_result
                .get("braid_id")
                .and_then(|v| v.as_str())
                .map(String::from);
            spine_id = commit_result
                .get("spine_id")
                .and_then(|v| v.as_str())
                .map(String::from);
        }
    }

    let computation = serde_json::json!({
        "tool": "hotSpring sovereign compchem",
        "substrate": "GPU (WGSL f64 shaders via barraCuda)",
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "modules_complete": modules.len(),
    });

    let transcript = FermentTranscript {
        dataset_id: format!("{dataset_id}_v{version}"),
        spring: "hotSpring".to_string(),
        spring_version: Some(version.to_string()),
        braid_id,
        dag_session_id: dag.as_ref().map(|d| d.dag_session_id.clone()),
        dag_merkle_root: dag
            .as_ref()
            .map(|d| format!("blake3:{}", d.merkle_root)),
        spine_id,
        timestamp: Some(now),
        computation: Some(computation),
    };

    GuideStoneProvenance {
        transcript,
        dag,
        primals_reached,
    }
}

/// Write the ferment transcript to the pseudoSpore provenance directory.
pub fn write_ferment_transcript(
    provenance_dir: &Path,
    result: &GuideStoneProvenance,
) -> std::io::Result<()> {
    std::fs::create_dir_all(provenance_dir)?;

    let transcript_path = provenance_dir.join("ferment_transcript.json");
    let json = serde_json::to_string_pretty(&result.transcript)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&transcript_path, json)?;

    let braids_dir = provenance_dir.join("braids");
    std::fs::create_dir_all(&braids_dir)?;

    let braid_filename = format!(
        "{}.json",
        result
            .transcript
            .dataset_id
            .replace(['/', ' '], "_")
    );
    let braid_json = serde_json::to_string_pretty(&serde_json::json!({
        "dataset_id": result.transcript.dataset_id,
        "spring": "hotSpring",
        "braid_id": result.transcript.braid_id,
        "dag_session_id": result.transcript.dag_session_id,
        "dag_merkle_root": result.transcript.dag_merkle_root,
        "spine_id": result.transcript.spine_id,
        "timestamp": result.transcript.timestamp,
        "computation": result.transcript.computation,
        "primals_reached": result.primals_reached,
    }))
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(braids_dir.join(braid_filename), braid_json)?;

    Ok(())
}

fn compute_output_hash(module: &ModuleEvent) -> Option<String> {
    if let Some(ref bytes) = module.output_bytes {
        return Some(blake3_hex(bytes));
    }
    if let Some(ref path) = module.output_path {
        if let Ok(bytes) = std::fs::read(path) {
            return Some(blake3_hex(&bytes));
        }
    }
    None
}

fn chrono_iso8601_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let s = time_secs % 60;

    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let year_days = if is_leap(y) { 366 } else { 365 };
        if remaining_days < year_days {
            break;
        }
        remaining_days -= year_days;
        y += 1;
    }
    let leap = is_leap(y);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0usize;
    for &md in &month_days {
        if remaining_days < md {
            break;
        }
        remaining_days -= md;
        m += 1;
    }
    format!(
        "{y:04}-{:02}-{:02}T{hours:02}:{mins:02}:{s:02}Z",
        m + 1,
        remaining_days + 1,
    )
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ferment_transcript_serializes_flat() {
        let ft = FermentTranscript {
            dataset_id: "cazyme_fel_v1.7.0".into(),
            spring: "hotSpring".into(),
            spring_version: Some("0.6.32".into()),
            braid_id: Some("braid-hotspring-cazyme-001".into()),
            dag_session_id: Some("dag-sess-001".into()),
            dag_merkle_root: Some("blake3:deadbeef".into()),
            spine_id: None,
            timestamp: Some("2026-06-01T14:00:00Z".into()),
            computation: Some(serde_json::json!({
                "tool": "hotSpring sovereign compchem",
                "modules_complete": 6,
            })),
        };
        let json = serde_json::to_string(&ft).expect("serialize");
        assert!(json.contains("cazyme_fel_v1.7.0"));
        assert!(json.contains("hotSpring"));
        assert!(!json.contains("spine_id")); // None fields skipped
    }

    #[test]
    fn ferment_transcript_roundtrip() {
        let ft = FermentTranscript {
            dataset_id: "test_v0.1.0".into(),
            spring: "hotSpring".into(),
            spring_version: None,
            braid_id: None,
            dag_session_id: None,
            dag_merkle_root: None,
            spine_id: None,
            timestamp: None,
            computation: None,
        };
        let json = serde_json::to_string(&ft).expect("serialize");
        let parsed: FermentTranscript = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.dataset_id, "test_v0.1.0");
        assert_eq!(parsed.spring, "hotSpring");
    }

    #[test]
    fn module_event_hash_from_bytes() {
        let module = ModuleEvent {
            name: "01_free_xylose_1d".into(),
            output_path: None,
            output_bytes: Some(b"test FES data".to_vec()),
            wall_seconds: 1.5,
            summary: serde_json::json!({"rmsd": 0.5}),
        };
        let hash = compute_output_hash(&module);
        assert!(hash.is_some());
        assert_eq!(hash.as_ref().unwrap().len(), 64);
    }

    #[test]
    fn offline_provenance_has_empty_primals() {
        let result = run_guidestone_provenance("test_artifact", "0.1.0", None, &[]);
        assert!(result.primals_reached.is_empty());
        assert!(result.dag.is_none());
        assert_eq!(result.transcript.spring, "hotSpring");
    }

    #[test]
    fn iso8601_timestamp_is_valid_format() {
        let ts = chrono_iso8601_now();
        assert!(ts.starts_with("20"));
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
    }

    #[test]
    fn write_ferment_transcript_creates_files() {
        let result = GuideStoneProvenance {
            transcript: FermentTranscript {
                dataset_id: "test_v0.1.0".into(),
                spring: "hotSpring".into(),
                spring_version: Some("0.6.32".into()),
                braid_id: None,
                dag_session_id: None,
                dag_merkle_root: None,
                spine_id: None,
                timestamp: Some("2026-06-01T00:00:00Z".into()),
                computation: None,
            },
            dag: None,
            primals_reached: vec![],
        };

        let tmp = std::env::temp_dir().join("hotspring_braid_test");
        let _ = std::fs::remove_dir_all(&tmp);
        write_ferment_transcript(&tmp, &result).expect("write");
        assert!(tmp.join("ferment_transcript.json").exists());
        assert!(tmp.join("braids/test_v0.1.0.json").exists());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
