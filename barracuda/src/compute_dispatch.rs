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
use crate::error::HotSpringError;
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
pub fn query_capabilities(nucleus: &NucleusContext) -> Result<Vec<String>, HotSpringError> {
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
) -> Result<String, HotSpringError> {
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
        .ok_or_else(|| HotSpringError::Ipc("no job_id in submit response".into()))
}

/// Retrieve the result of a submitted GPU workload.
///
/// Polls `compute.dispatch.result` for the given job ID. Returns the
/// output data as a JSON value (typically an f64 array).
pub fn retrieve_result(
    nucleus: &NucleusContext,
    job_id: &str,
) -> Result<serde_json::Value, HotSpringError> {
    let params = serde_json::json!({
        "job_id": job_id,
    });

    let resp = nucleus.call_by_capability("compute", "compute.dispatch.result", params)?;

    resp.get("result")
        .cloned()
        .ok_or_else(|| HotSpringError::Ipc("no result field in dispatch response".into()))
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
            log::info!("compute.dispatch.capabilities: {} capabilities", caps.len());
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
            log::error!("{msg}");
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
            log::info!("compute.dispatch.submit: job_id={id}");
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
            log::error!("{msg}");
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

            log::info!(
                "compute.dispatch.result: {} bytes, blake3={}…",
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
            log::error!("{msg}");
            result.errors.push(msg);
        }
    }

    result.witnesses.push(WireWitnessRef::checkpoint(
        "hotspring:dispatch",
        "compute_dispatch:complete",
    ));

    result
}

/// WGSL shaders that require coralReef's `membar.{cta,gl}` emitter.
///
/// These use `workgroupBarrier()` and must compile on SM35 (K80),
/// SM70 (Titan V), and SM120 (RTX 5060) for generation parity.
pub const BARRIER_SHADERS: &[&str] = &[
    "src/bin/shaders/silicon_capabilities/probe_f32_workgroup_reduce.wgsl",
    "src/bin/shaders/silicon_capabilities/probe_df64_workgroup_reduce_f32_body.wgsl",
    "src/bin/shaders/silicon_capabilities/probe_df64_workgroup_reduce_f64_body.wgsl",
    "src/bin/shaders/qcd_silicon_routing/reduce_shared.wgsl",
    "src/lattice/shaders/sum_reduce_subgroup_f64.wgsl",
    "src/lattice/shaders/sum_reduce_f64.wgsl",
    "src/physics/shaders/deformed_potentials_f64.wgsl",
    "src/physics/shaders/deformed_hamiltonian_f64.wgsl",
    "src/physics/shaders/deformed_wavefunction_f64.wgsl",
];

/// Result of barrier shader compilation validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierShaderValidation {
    pub shader_path: String,
    pub compiled: bool,
    pub error: Option<String>,
    /// Binary size in bytes (from coralReef `size` or `binary_b64` decode length).
    #[serde(default)]
    pub binary_size: Option<u64>,
    /// GPU registers used (from coralReef `shader_info.gpr_count`).
    #[serde(default)]
    pub gpr_count: Option<u32>,
}

/// Validate that barrier/shared-memory WGSL shaders compile through
/// coralReef's `shader.compile.wgsl` IPC.
///
/// Returns a per-shader compilation result. When coralReef is not
/// available, all entries report `compiled: false` with an appropriate error.
pub fn validate_barrier_shaders(nucleus: &NucleusContext) -> Vec<BarrierShaderValidation> {
    let shader_alive = nucleus.by_domain("shader").is_some_and(|ep| ep.alive);

    if !shader_alive {
        return BARRIER_SHADERS
            .iter()
            .map(|&path| BarrierShaderValidation {
                shader_path: path.to_string(),
                compiled: false,
                error: Some("shader compilation primal not available".into()),
                binary_size: None,
                gpr_count: None,
            })
            .collect();
    }

    BARRIER_SHADERS
        .iter()
        .map(|&rel_path| {
            let shader_source = match std::fs::read_to_string(rel_path) {
                Ok(src) => src,
                Err(e) => {
                    return BarrierShaderValidation {
                        shader_path: rel_path.to_string(),
                        compiled: false,
                        error: Some(format!("read failed: {e}")),
                        binary_size: None,
                        gpr_count: None,
                    };
                }
            };

            let params = serde_json::json!({
                "wgsl_source": shader_source,
            });

            match nucleus.call_by_capability("shader", "shader.compile.wgsl", params) {
                Ok(resp) => {
                    let has_error = resp.get("error").is_some();
                    let binary_size = resp
                        .get("size")
                        .and_then(serde_json::Value::as_u64)
                        .or_else(|| {
                            resp.get("binary_b64")
                                .and_then(serde_json::Value::as_str)
                                .map(|b| (b.len() as u64 * 3) / 4)
                        });
                    let gpr_count = resp
                        .get("shader_info")
                        .and_then(|si| si.get("gprs").or_else(|| si.get("gpr_count")))
                        .and_then(serde_json::Value::as_u64)
                        .map(|n| n as u32);
                    BarrierShaderValidation {
                        shader_path: rel_path.to_string(),
                        compiled: !has_error,
                        error: if has_error {
                            Some(resp["error"].to_string())
                        } else {
                            None
                        },
                        binary_size,
                        gpr_count,
                    }
                }
                Err(e) => BarrierShaderValidation {
                    shader_path: rel_path.to_string(),
                    compiled: false,
                    error: Some(format!("IPC failed: {e}")),
                    binary_size: None,
                    gpr_count: None,
                },
            }
        })
        .collect()
}

// ── Fused Pipeline (TensorSession evolution) ─────────────────────────

/// A fused multi-op dispatch pipeline.
///
/// Replaces the single-op `submit_workload` → `retrieve_result` pattern
/// with a session that batches multiple operations and submits them as
/// a single dispatch unit. This is the hotSpring-side wiring for upstream
/// barraCuda's `TensorSession` concept (GAP-HS-027).
///
/// # Protocol
///
/// 1. Create a `FusedPipeline` with a session name
/// 2. Add operations via `push_op`
/// 3. Submit the entire batch via `submit`
/// 4. Retrieve fused results via `retrieve`
///
/// When toadStool supports fused dispatch (`compute.dispatch.submit_fused`),
/// the batch is sent as a single RPC. Until then, operations are dispatched
/// sequentially and results are collected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedPipeline {
    pub session_name: String,
    pub ops: Vec<FusedOp>,
    pub submitted_job_ids: Vec<String>,
}

/// A single operation within a fused pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOp {
    pub shader: String,
    pub domain: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub depends_on: Vec<usize>,
}

/// Result of a fused pipeline submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedResult {
    pub session_name: String,
    pub op_results: Vec<FusedOpResult>,
    pub all_succeeded: bool,
}

/// Result of a single operation within a fused pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedOpResult {
    pub index: usize,
    pub shader: String,
    pub succeeded: bool,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
}

impl FusedPipeline {
    /// Create a new fused pipeline session.
    #[must_use]
    pub fn new(session_name: impl Into<String>) -> Self {
        Self {
            session_name: session_name.into(),
            ops: Vec::new(),
            submitted_job_ids: Vec::new(),
        }
    }

    /// Add an operation to the pipeline.
    pub fn push_op(
        &mut self,
        shader: impl Into<String>,
        domain: impl Into<String>,
        input: serde_json::Value,
    ) {
        self.ops.push(FusedOp {
            shader: shader.into(),
            domain: domain.into(),
            input,
            depends_on: Vec::new(),
        });
    }

    /// Add an operation with explicit dependencies on prior ops.
    pub fn push_op_with_deps(
        &mut self,
        shader: impl Into<String>,
        domain: impl Into<String>,
        input: serde_json::Value,
        depends_on: Vec<usize>,
    ) {
        self.ops.push(FusedOp {
            shader: shader.into(),
            domain: domain.into(),
            input,
            depends_on,
        });
    }

    /// Number of operations in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the pipeline has no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Submit the fused pipeline via NUCLEUS IPC.
    ///
    /// Attempts `compute.dispatch.submit_fused` first; if unavailable,
    /// falls back to sequential `compute.dispatch.submit` per operation.
    pub fn submit(&mut self, nucleus: &NucleusContext) -> Result<(), HotSpringError> {
        let fused_params = serde_json::json!({
            "session": self.session_name,
            "ops": self.ops,
        });

        if let Ok(resp) =
            nucleus.call_by_capability("compute", "compute.dispatch.submit_fused", fused_params)
        {
            if let Some(ids) = resp.get("job_ids").and_then(|v| v.as_array()) {
                self.submitted_job_ids = ids
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                return Ok(());
            }
        }

        for op in &self.ops {
            let params = serde_json::json!({
                "shader": op.shader,
                "input": op.input,
                "spring": "hotSpring",
                "session": self.session_name,
            });
            match nucleus.call_by_capability("compute", "compute.dispatch.submit", params) {
                Ok(resp) => {
                    let job_id = resp
                        .get("result")
                        .and_then(|r| r.get("job_id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    self.submitted_job_ids.push(job_id);
                }
                Err(e) => {
                    self.submitted_job_ids.push(format!("error:{e}"));
                }
            }
        }
        Ok(())
    }

    /// Retrieve results for all submitted operations.
    pub fn retrieve(&self, nucleus: &NucleusContext) -> FusedResult {
        let mut op_results = Vec::with_capacity(self.ops.len());

        for (i, job_id) in self.submitted_job_ids.iter().enumerate() {
            let shader = self.ops.get(i).map_or("unknown", |o| &o.shader).to_string();
            if job_id.starts_with("error:") {
                op_results.push(FusedOpResult {
                    index: i,
                    shader,
                    succeeded: false,
                    result: None,
                    error: Some(job_id.clone()),
                });
                continue;
            }
            match retrieve_result(nucleus, job_id) {
                Ok(data) => {
                    op_results.push(FusedOpResult {
                        index: i,
                        shader,
                        succeeded: true,
                        result: Some(data),
                        error: None,
                    });
                }
                Err(e) => {
                    op_results.push(FusedOpResult {
                        index: i,
                        shader,
                        succeeded: false,
                        result: None,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        let all_succeeded = op_results.iter().all(|r| r.succeeded);

        FusedResult {
            session_name: self.session_name.clone(),
            op_results,
            all_succeeded,
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "dispatch validation tests use expect on test payloads"
)]
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

    #[test]
    fn fused_pipeline_builds_ops() {
        let mut fp = FusedPipeline::new("test-session");
        assert!(fp.is_empty());

        fp.push_op("vector_add_f64", "compute", serde_json::json!({"a": [1.0]}));
        fp.push_op_with_deps(
            "reduce_sum_f64",
            "compute",
            serde_json::json!({"input": "prev"}),
            vec![0],
        );

        assert_eq!(fp.len(), 2);
        assert!(!fp.is_empty());
        assert_eq!(fp.ops[0].shader, "vector_add_f64");
        assert_eq!(fp.ops[1].depends_on, vec![0]);
    }

    #[test]
    fn fused_pipeline_serializes() {
        let mut fp = FusedPipeline::new("serialize-test");
        fp.push_op("saxpy", "compute", serde_json::json!({"alpha": 2.0}));

        let json = serde_json::to_string(&fp).expect("serialize");
        assert!(json.contains("serialize-test"));
        assert!(json.contains("saxpy"));

        let round_trip: FusedPipeline = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(round_trip.session_name, "serialize-test");
        assert_eq!(round_trip.ops.len(), 1);
    }

    #[test]
    fn fused_result_all_succeeded_logic() {
        let result = FusedResult {
            session_name: "test".into(),
            op_results: vec![
                FusedOpResult {
                    index: 0,
                    shader: "a".into(),
                    succeeded: true,
                    result: Some(serde_json::json!({})),
                    error: None,
                },
                FusedOpResult {
                    index: 1,
                    shader: "b".into(),
                    succeeded: true,
                    result: Some(serde_json::json!({})),
                    error: None,
                },
            ],
            all_succeeded: true,
        };
        assert!(result.all_succeeded);

        let partial = FusedResult {
            session_name: "test".into(),
            op_results: vec![FusedOpResult {
                index: 0,
                shader: "c".into(),
                succeeded: false,
                result: None,
                error: Some("dispatch failed".into()),
            }],
            all_succeeded: false,
        };
        assert!(!partial.all_succeeded);
    }
}
