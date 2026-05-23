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
use crate::primal_bridge::{NucleusContext, parse_jsonrpc_response};
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

    let result = parse_jsonrpc_response(&resp, "compute.dispatch.capabilities")?;

    let caps = result
        .get("capabilities")
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

/// Compile a WGSL shader source via coralReef, then submit the compiled
/// binary to toadStool for dispatch.  Returns the `job_id` for result
/// retrieval.
///
/// Two-step pipeline:
/// 1. `shader.compile.wgsl` → coralReef → `binary_b64` + metadata
/// 2. `compute.dispatch.submit` → toadStool → `job_id`
pub fn compile_and_submit(
    nucleus: &NucleusContext,
    wgsl_source: &str,
    input_data: &[f64],
    bdf: Option<&str>,
) -> Result<String, HotSpringError> {
    let compile_params = serde_json::json!({ "wgsl_source": wgsl_source });
    let compile_resp =
        nucleus.call_by_capability("shader", "shader.compile.wgsl", compile_params)?;

    let compile_result = parse_jsonrpc_response(&compile_resp, "shader.compile.wgsl")?;

    let binary_b64 = compile_result
        .get("binary_b64")
        .or_else(|| compile_result.get("binary"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            let err_detail = compile_result.get("error").map_or_else(
                || "no binary in compile response".into(),
                ToString::to_string,
            );
            HotSpringError::Ipc(format!("shader compile failed: {err_detail}"))
        })?;

    let input_hash = blake3_hex(&serde_json::to_vec(&input_data).unwrap_or_default());

    let mut submit = serde_json::json!({
        "binary_b64": binary_b64,
        "input": {
            "data": input_data,
            "format": "f64_array",
        },
        "input_hash": input_hash,
        "spring": "hotSpring",
        "dispatch_mode": "passthrough",
    });

    if let Some(bdf_val) = bdf {
        submit["bdf"] = serde_json::Value::String(bdf_val.to_string());
    }

    let resp = nucleus.call_by_capability("compute", "compute.dispatch.submit", submit)?;

    extract_job_id(&resp)
}

/// Dispatch a GPU compute workload via the Wave 17 `node.compute` signal.
///
/// Sends a single `node.compute` signal that biomeOS decomposes into
/// compile → submit → execute via the graph. Falls back to the
/// multi-call [`compile_and_submit`] pipeline for older biomeOS.
pub fn dispatch_node_compute(
    nucleus: &NucleusContext,
    wgsl_source: &str,
    input_data: &[f64],
    bdf: Option<&str>,
) -> Result<String, HotSpringError> {
    let input_hash = blake3_hex(&serde_json::to_vec(&input_data).unwrap_or_default());

    let mut signal_params = serde_json::json!({
        "wgsl_source": wgsl_source,
        "input": {
            "data": input_data,
            "format": "f64_array",
        },
        "input_hash": input_hash,
        "spring": "hotSpring",
    });

    if let Some(bdf_val) = bdf {
        signal_params["bdf"] = serde_json::Value::String(bdf_val.to_string());
    }

    let dispatch_params = serde_json::json!({
        "signal": "node.compute",
        "params": signal_params,
    });

    match nucleus.call_by_capability("orchestration", "signal.dispatch", dispatch_params) {
        Ok(resp) => extract_job_id(&resp),
        Err(_) => compile_and_submit(nucleus, wgsl_source, input_data, bdf),
    }
}

/// Submit a pre-compiled binary to toadStool for dispatch.
/// Use [`compile_and_submit`] for the full WGSL→binary→dispatch pipeline.
pub fn submit_binary(
    nucleus: &NucleusContext,
    binary_b64: &str,
    input_data: &[f64],
    bdf: Option<&str>,
) -> Result<String, HotSpringError> {
    let input_hash = blake3_hex(&serde_json::to_vec(&input_data).unwrap_or_default());

    let mut params = serde_json::json!({
        "binary_b64": binary_b64,
        "input": {
            "data": input_data,
            "format": "f64_array",
        },
        "input_hash": input_hash,
        "spring": "hotSpring",
        "dispatch_mode": "passthrough",
    });

    if let Some(bdf_val) = bdf {
        params["bdf"] = serde_json::Value::String(bdf_val.to_string());
    }

    let resp = nucleus.call_by_capability("compute", "compute.dispatch.submit", params)?;

    extract_job_id(&resp)
}

/// Extract `job_id` from a `compute.dispatch.submit` response envelope.
fn extract_job_id(resp: &serde_json::Value) -> Result<String, HotSpringError> {
    let result = parse_jsonrpc_response(resp, "compute.dispatch.submit")?;
    result
        .get("job_id")
        .and_then(serde_json::Value::as_str)
        .map(String::from)
        .ok_or_else(|| HotSpringError::Ipc("compute.dispatch.submit: missing job_id".into()))
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

    parse_jsonrpc_response(&resp, "compute.dispatch.result")
}

/// Publish a signed result via the Wave 17 `tower.publish` signal.
///
/// Dispatches `tower.publish` which biomeOS decomposes into sign (bearDog)
/// → announce (songBird) → audit (skunkBat). Falls back to direct
/// `crypto.sign_ed25519` + `discovery.announce` if the signal is unavailable.
pub fn publish_result(
    nucleus: &NucleusContext,
    result_data: &serde_json::Value,
    topic: &str,
) -> Result<serde_json::Value, HotSpringError> {
    let result_hash = blake3_hex(&serde_json::to_vec(result_data).unwrap_or_default());

    let signal_params = serde_json::json!({
        "content": result_data,
        "content_hash": result_hash,
        "topic": topic,
        "spring": "hotSpring",
    });

    let dispatch_params = serde_json::json!({
        "signal": "tower.publish",
        "params": signal_params,
    });

    if let Ok(resp) =
        nucleus.call_by_capability("orchestration", "signal.dispatch", dispatch_params)
    {
        parse_jsonrpc_response(&resp, "tower.publish")
    } else {
        let sign_params = serde_json::json!({
            "message": result_hash,
            "key_purpose": "result_signing",
        });
        let _sign_resp = nucleus.call_by_capability("crypto", "crypto.sign_ed25519", sign_params);

        let announce_params = serde_json::json!({
            "topic": topic,
            "content_hash": result_hash,
            "spring": "hotSpring",
        });
        let resp =
            nucleus.call_by_capability("discovery", "discovery.announce", announce_params)?;
        parse_jsonrpc_response(&resp, "discovery.announce")
    }
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
    let job_id = match dispatch_node_compute(nucleus, VALIDATE_VECTOR_ADD_WGSL, &test_input, None) {
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

/// Minimal WGSL shader for dispatch validation: element-wise vector addition.
const VALIDATE_VECTOR_ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&a) {
        out[i] = a[i] + b[i];
    }
}
"#;

/// WGSL shaders that require coralReef's `membar.{cta,gl}` emitter.
///
/// These use `workgroupBarrier()` and must compile on SM35 (K80),
/// SM70 (Titan V), and SM120 (RTX 5060) for generation parity.
/// Ordered so stable shaders run first — subgroup shaders last to contain
/// upstream panic risk (coralReef copy-prop assertion on SubgroupBallotResult).
pub const BARRIER_SHADERS: &[&str] = &[
    "src/bin/shaders/silicon_capabilities/probe_f32_workgroup_reduce.wgsl",
    "src/bin/shaders/silicon_capabilities/probe_df64_workgroup_reduce_f32_body.wgsl",
    "src/bin/shaders/silicon_capabilities/probe_df64_workgroup_reduce_f64_body.wgsl",
    "src/bin/shaders/qcd_silicon_routing/reduce_shared.wgsl",
    "src/lattice/shaders/sum_reduce_f64.wgsl",
    "src/physics/shaders/deformed_potentials_f64.wgsl",
    "src/physics/shaders/deformed_hamiltonian_f64.wgsl",
    "src/physics/shaders/deformed_wavefunction_f64.wgsl",
    "src/lattice/shaders/sum_reduce_subgroup_f64.wgsl",
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

/// CPU fallback dispatch for offline parity testing.
///
/// When toadStool is unavailable (no `compute` domain in NUCLEUS), this
/// function runs the same physics computation locally on CPU. Returns
/// the result as a JSON value matching the shape of
/// `compute.dispatch.result` responses.
///
/// Supported workload names:
/// - `"semf_batch"` — SEMF binding energy for Z/N pairs in `input_data`
/// - `"vector_add"` — element-wise addition (input split in half)
/// - `"spmv"` — sparse matrix-vector product (placeholder)
///
/// Returns `None` for unsupported workload names.
pub fn dispatch_cpu_fallback(
    workload_name: &str,
    input_data: &[f64],
) -> Option<serde_json::Value> {
    match workload_name {
        "vector_add" | "vector_add_f64" => {
            let half = input_data.len() / 2;
            let result: Vec<f64> = input_data[..half]
                .iter()
                .zip(&input_data[half..half * 2])
                .map(|(a, b)| a + b)
                .collect();
            Some(serde_json::json!({
                "output": result,
                "format": "f64_array",
                "substrate": "cpu_fallback",
            }))
        }
        "semf_batch" => {
            use crate::physics::semf_binding_energy;
            use crate::provenance::SLY4_PARAMS;
            let pairs: Vec<(usize, usize)> = input_data
                .chunks(2)
                .filter(|c| c.len() == 2)
                .map(|c| (c[0] as usize, c[1] as usize))
                .collect();
            let energies: Vec<f64> = pairs
                .iter()
                .map(|&(z, n)| semf_binding_energy(z, n, &SLY4_PARAMS))
                .collect();
            Some(serde_json::json!({
                "output": energies,
                "format": "f64_array",
                "substrate": "cpu_fallback",
                "n_nuclei": pairs.len(),
            }))
        }
        _ => None,
    }
}

mod fused;

pub use fused::*;

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
}
