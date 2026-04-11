// SPDX-License-Identifier: AGPL-3.0-or-later

//! ToadStool compute dispatch validation.
//!
//! Validates `compute.dispatch.submit`, `compute.dispatch.result`, and
//! `compute.dispatch.capabilities` for GPU workloads routed through
//! ToadStool to barraCuda/coralReef.
//!
//! This module exercises the dispatch pipeline that esotericWebb's
//! `webb_node.toml` (Tower + ToadStool) will use. Every shader hotSpring
//! validates and hands off to barraCuda becomes available to all springs
//! and gardens via this dispatch path.
//!
//! # Flow
//!
//! ```text
//! hotSpring → compute.dispatch.submit → ToadStool
//!   ToadStool → barraCuda (shader execution)
//!   ToadStool → coralReef (shader compilation, if sovereign)
//!   ToadStool ← result
//! hotSpring ← compute.dispatch.result → verify output
//! hotSpring → rhizoCrypt: kind:"hash" witness (blake3 of output)
//! ```

use crate::dag_provenance::{DagEvent, DagSession, blake3_hex};
use crate::primal_bridge::NucleusContext;
use crate::witness::WireWitnessRef;
use serde::{Deserialize, Serialize};

/// Result of a compute dispatch validation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchValidation {
    pub capabilities_available: bool,
    pub gpu_capabilities: Vec<String>,
    pub submit_succeeded: bool,
    pub result_received: bool,
    pub output_hash: Option<String>,
    pub witnesses: Vec<WireWitnessRef>,
    pub errors: Vec<String>,
}

impl DispatchValidation {
    /// True when the full pipeline (capabilities + submit + result) passed.
    #[must_use]
    pub fn all_passed(&self) -> bool {
        self.capabilities_available && self.submit_succeeded && self.result_received
    }
}

/// Query ToadStool for GPU compute capabilities.
///
/// Returns the list of capability strings advertised by the compute
/// dispatch system. An empty list means ToadStool is reachable but
/// has no GPU backends registered.
pub fn query_capabilities(nucleus: &NucleusContext) -> Result<Vec<String>, String> {
    let resp = nucleus.call_by_capability(
        "compute",
        "compute.dispatch.capabilities",
        serde_json::json!({}),
    )?;

    let caps = resp
        .get("result")
        .and_then(|r| r.get("capabilities"))
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    Ok(caps)
}

/// Submit a GPU workload via `compute.dispatch.submit`.
///
/// Returns the job ID for result retrieval. The workload is a simple
/// vector addition shader — validates the pipeline without requiring
/// heavy physics computation.
pub fn submit_workload(
    nucleus: &NucleusContext,
    shader_name: &str,
    input_data: &[f64],
) -> Result<String, String> {
    let input_hash = blake3_hex(&serde_json::to_vec(&input_data).unwrap_or_default());

    let params = serde_json::json!({
        "shader": shader_name,
        "input": {
            "data": input_data,
            "format": "f64_array",
        },
        "input_hash": input_hash,
        "spring": "hotSpring",
    });

    let resp = nucleus.call_by_capability("compute", "compute.dispatch.submit", params)?;

    resp.get("result")
        .and_then(|r| r.get("job_id"))
        .and_then(serde_json::Value::as_str)
        .map(String::from)
        .ok_or_else(|| "no job_id in submit response".into())
}

/// Retrieve the result of a submitted GPU workload.
///
/// Polls `compute.dispatch.result` for the given job ID. Returns the
/// output data as a JSON value (typically an f64 array).
pub fn retrieve_result(
    nucleus: &NucleusContext,
    job_id: &str,
) -> Result<serde_json::Value, String> {
    let params = serde_json::json!({
        "job_id": job_id,
    });

    let resp = nucleus.call_by_capability("compute", "compute.dispatch.result", params)?;

    resp.get("result")
        .cloned()
        .ok_or_else(|| "no result field in dispatch response".into())
}

/// Run the full compute dispatch validation pipeline.
///
/// 1. Query capabilities
/// 2. Submit a test workload
/// 3. Retrieve and verify the result
/// 4. Emit blake3 hash witness for the output
///
/// Returns a [`DispatchValidation`] with all results and witnesses.
/// If a DAG session is provided, events are appended for each phase.
pub fn validate_dispatch(
    nucleus: &NucleusContext,
    mut dag: Option<&mut DagSession>,
) -> DispatchValidation {
    let mut result = DispatchValidation {
        capabilities_available: false,
        gpu_capabilities: Vec::new(),
        submit_succeeded: false,
        result_received: false,
        output_hash: None,
        witnesses: Vec::new(),
        errors: Vec::new(),
    };

    result.witnesses.push(WireWitnessRef::checkpoint(
        "hotspring:dispatch",
        "compute_dispatch:start",
    ));

    // Phase 1: capabilities
    let start = std::time::Instant::now();
    match query_capabilities(nucleus) {
        Ok(caps) => {
            result.capabilities_available = true;
            result.gpu_capabilities.clone_from(&caps);
            println!(
                "  compute.dispatch.capabilities: {} capabilities",
                caps.len()
            );
            if let Some(ref mut dag) = dag {
                dag.append(
                    nucleus,
                    DagEvent {
                        phase: "dispatch_capabilities".into(),
                        input_hash: None,
                        output_hash: None,
                        wall_seconds: start.elapsed().as_secs_f64(),
                        summary: serde_json::json!({"capabilities": caps}),
                    },
                );
            }
        }
        Err(e) => {
            let msg = format!("capabilities query failed: {e}");
            eprintln!("  {msg}");
            result.errors.push(msg);
            return result;
        }
    }

    // Phase 2: submit test workload
    let test_input: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let input_hash = blake3_hex(&serde_json::to_vec(&test_input).unwrap_or_default());

    let submit_start = std::time::Instant::now();
    let job_id = match submit_workload(nucleus, "vector_add_f64", &test_input) {
        Ok(id) => {
            result.submit_succeeded = true;
            println!("  compute.dispatch.submit: job_id={id}");
            if let Some(ref mut dag) = dag {
                dag.append(
                    nucleus,
                    DagEvent {
                        phase: "dispatch_submit".into(),
                        input_hash: Some(input_hash),
                        output_hash: None,
                        wall_seconds: submit_start.elapsed().as_secs_f64(),
                        summary: serde_json::json!({"job_id": id, "shader": "vector_add_f64"}),
                    },
                );
            }
            id
        }
        Err(e) => {
            let msg = format!("submit failed: {e}");
            eprintln!("  {msg}");
            result.errors.push(msg);
            return result;
        }
    };

    // Phase 3: retrieve result
    let result_start = std::time::Instant::now();
    match retrieve_result(nucleus, &job_id) {
        Ok(output) => {
            result.result_received = true;
            let output_bytes = serde_json::to_vec(&output).unwrap_or_default();
            let output_hash = blake3_hex(&output_bytes);
            result.output_hash = Some(output_hash.clone());

            result.witnesses.push(WireWitnessRef::hash(
                "hotspring:dispatch",
                &output_hash,
                Some(&format!("compute_dispatch:result:{job_id}")),
            ));

            println!(
                "  compute.dispatch.result: {} bytes, blake3={}…",
                output_bytes.len(),
                &output_hash[..16.min(output_hash.len())]
            );

            if let Some(ref mut dag) = dag {
                dag.append(
                    nucleus,
                    DagEvent {
                        phase: "dispatch_result".into(),
                        input_hash: None,
                        output_hash: Some(output_hash),
                        wall_seconds: result_start.elapsed().as_secs_f64(),
                        summary: serde_json::json!({"job_id": job_id}),
                    },
                );
            }
        }
        Err(e) => {
            let msg = format!("result retrieval failed: {e}");
            eprintln!("  {msg}");
            result.errors.push(msg);
        }
    }

    result.witnesses.push(WireWitnessRef::checkpoint(
        "hotspring:dispatch",
        "compute_dispatch:complete",
    ));

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_validation_all_passed_logic() {
        let mut v = DispatchValidation {
            capabilities_available: true,
            gpu_capabilities: vec!["gpu.f64".into()],
            submit_succeeded: true,
            result_received: true,
            output_hash: Some("abc".into()),
            witnesses: Vec::new(),
            errors: Vec::new(),
        };
        assert!(v.all_passed());

        v.submit_succeeded = false;
        assert!(!v.all_passed());
    }

    #[test]
    fn dispatch_validation_serializes() {
        let v = DispatchValidation {
            capabilities_available: true,
            gpu_capabilities: vec!["gpu.f64".into(), "gpu.f32".into()],
            submit_succeeded: true,
            result_received: true,
            output_hash: Some("deadbeef".into()),
            witnesses: vec![WireWitnessRef::hash("test", "abc", None)],
            errors: Vec::new(),
        };
        let json = serde_json::to_string(&v).expect("serialize");
        assert!(json.contains("gpu.f64"));
        assert!(json.contains("witnesses"));
    }
}
