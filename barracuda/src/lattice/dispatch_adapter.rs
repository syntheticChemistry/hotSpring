// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lattice QCD GPU dispatch adapter — maps QCD kernel bind groups to toadStool's
//! `compute.dispatch.submit` wire format via NUCLEUS IPC.
//!
//! Orchestrates coralReef shader compilation and toadStool buffer lifecycle for
//! Wilson plaquette, gauge force, and other lattice WGSL kernels whose bindings
//! follow the standard four-buffer layout:
//!
//! | Binding | Role |
//! |---------|------|
//! | 0 | uniform params |
//! | 1 | gauge links (f64) |
//! | 2 | neighbor table (u32) |
//! | 3 | output (f64) |

use std::collections::HashMap;
use std::time::Instant;

use barracuda::ops::lattice::shader_sources::lattice_shader_source;
use serde_json::Value;

use crate::base64_encode;
use crate::compute_dispatch::retrieve_result;
use crate::error::HotSpringError;
use crate::primal_bridge::{NucleusContext, parse_jsonrpc_response};

/// Default inverse bare coupling β = 6/g² for quenched QCD convenience dispatches.
const DEFAULT_BETA: f64 = 6.0;

/// Workgroup count scale for lattice volume (`ceil(sites / LATTICE_DISPATCH_SCALE)`).
const LATTICE_DISPATCH_SCALE: usize = 256;

/// Maximum polls of `compute.dispatch.result` before giving up.
const RESULT_POLL_ATTEMPTS: u32 = 60;

/// Pause between result polls (milliseconds).
const RESULT_POLL_INTERVAL_MS: u64 = 50;

/// Maps lattice shader bind groups to toadStool dispatch IPC.
pub struct LatticeDispatchAdapter {
    nucleus: NucleusContext,
    shader_cache: HashMap<String, String>,
}

/// Parameters for a single lattice GPU dispatch.
pub struct LatticeDispatchParams {
    pub shader_name: String,
    pub bdf: Option<String>,
    pub lattice_dims: [u32; 4],
    pub links: Vec<f64>,
    pub neighbors: Vec<u32>,
    pub params: Vec<u8>,
    pub output_elements: usize,
}

/// Result of a completed lattice GPU dispatch.
pub struct LatticeDispatchResult {
    pub output: Vec<f64>,
    pub job_id: String,
    pub elapsed_ms: u64,
}

impl LatticeDispatchAdapter {
    /// Create an adapter bound to the given NUCLEUS context.
    #[must_use]
    pub fn new(nucleus: NucleusContext) -> Self {
        Self {
            nucleus,
            shader_cache: HashMap::new(),
        }
    }

    /// Compile a lattice WGSL shader via coralReef, caching the binary by name.
    pub fn compile_shader(&mut self, name: &str) -> Result<String, HotSpringError> {
        if let Some(cached) = self.shader_cache.get(name) {
            return Ok(cached.clone());
        }

        let wgsl = lattice_shader_source(name).ok_or_else(|| {
            HotSpringError::Ipc(format!("unknown lattice shader: {name}"))
        })?;

        let compile_params = serde_json::json!({ "wgsl_source": wgsl });
        let compile_resp = self
            .nucleus
            .call_by_capability("shader", "shader.compile.wgsl", compile_params)?;

        let compile_result = parse_jsonrpc_response(&compile_resp, "shader.compile.wgsl")?;

        let binary_b64 = compile_result
            .get("binary_b64")
            .or_else(|| compile_result.get("binary"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                let err_detail = compile_result.get("error").map_or_else(
                    || "no binary in compile response".into(),
                    ToString::to_string,
                );
                HotSpringError::Ipc(format!("shader compile failed: {err_detail}"))
            })?
            .to_string();

        self.shader_cache
            .insert(name.to_string(), binary_b64.clone());
        Ok(binary_b64)
    }

    /// Run a lattice dispatch: compile (if needed), submit buffers, poll result.
    pub fn dispatch(
        &mut self,
        params: LatticeDispatchParams,
    ) -> Result<LatticeDispatchResult, HotSpringError> {
        let started = Instant::now();
        let binary_b64 = self.compile_shader(&params.shader_name)?;

        let volume = lattice_volume(params.lattice_dims);
        let dispatch_dims = dispatch_dims_for_sites(volume);

        let buffers = serialize_lattice_buffers(&params)?;
        let job_id = submit_lattice_dispatch(
            &self.nucleus,
            &binary_b64,
            params.bdf.as_deref(),
            dispatch_dims,
            &buffers,
        )?;

        let output = poll_dispatch_output(&self.nucleus, &job_id, params.output_elements)?;

        Ok(LatticeDispatchResult {
            output,
            job_id,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// Dispatch `wilson_plaquette_f64` and return the average plaquette.
    pub fn dispatch_plaquette(
        &mut self,
        links: Vec<f64>,
        neighbors: Vec<u32>,
        lattice_dims: [u32; 4],
    ) -> Result<f64, HotSpringError> {
        let volume = lattice_volume(lattice_dims);
        let params = LatticeDispatchParams {
            shader_name: "wilson_plaquette_f64".into(),
            bdf: None,
            lattice_dims,
            links,
            neighbors,
            params: make_plaq_uniform_params(volume as u32),
            output_elements: volume,
        };

        let result = self.dispatch(params)?;
        let sum: f64 = result.output.iter().sum();
        let n_plaq = 6.0 * volume as f64;
        Ok(sum / n_plaq)
    }

    /// Dispatch `su3_gauge_force_f64` and return the flattened force field.
    pub fn dispatch_gauge_force(
        &mut self,
        links: Vec<f64>,
        neighbors: Vec<u32>,
        lattice_dims: [u32; 4],
    ) -> Result<Vec<f64>, HotSpringError> {
        let volume = lattice_volume(lattice_dims);
        let n_links = volume * 4;
        let params = LatticeDispatchParams {
            shader_name: "su3_gauge_force_f64".into(),
            bdf: None,
            lattice_dims,
            links,
            neighbors,
            params: make_force_uniform_params(volume as u32, DEFAULT_BETA),
            output_elements: n_links * 18,
        };

        self.dispatch(params).map(|r| r.output)
    }
}

/// Total lattice sites from `[Nt, Nx, Ny, Nz]`.
#[must_use]
pub fn lattice_volume(dims: [u32; 4]) -> usize {
    dims.iter().map(|&d| d as usize).product()
}

/// Workgroup grid for a site-parallel lattice kernel.
#[must_use]
pub fn dispatch_dims_for_sites(total_sites: usize) -> [u32; 3] {
    let wg = total_sites.div_ceil(LATTICE_DISPATCH_SCALE).max(1) as u32;
    [wg, 1, 1]
}

/// Uniform buffer for `wilson_plaquette_f64` (`PlaqParams { volume, pad×3 }`).
#[must_use]
pub fn make_plaq_uniform_params(volume: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&volume.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes
}

/// Uniform buffer for `su3_gauge_force_f64` (`ForceParams { volume, pad, beta }`).
#[must_use]
pub fn make_force_uniform_params(volume: u32, beta: f64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&volume.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&beta.to_le_bytes());
    bytes
}

/// Serialize QCD bind groups into toadStool `buffers[]` wire format.
pub fn serialize_lattice_buffers(
    params: &LatticeDispatchParams,
) -> Result<Value, HotSpringError> {
    let params_b64 = base64_encode::encode(&params.params);
    let links_bytes = f64_slice_to_bytes(&params.links);
    let links_b64 = base64_encode::encode(&links_bytes);
    let neighbors_bytes = u32_slice_to_bytes(&params.neighbors);
    let neighbors_b64 = base64_encode::encode(&neighbors_bytes);
    let output_bytes = params.output_elements * size_of::<f64>();

    Ok(serde_json::json!([
        {
            "size": params.params.len(),
            "direction": "input",
            "data_b64": params_b64,
            "domain": "lattice",
        },
        {
            "size": links_bytes.len(),
            "direction": "input",
            "data_b64": links_b64,
            "domain": "lattice",
        },
        {
            "size": neighbors_bytes.len(),
            "direction": "input",
            "data_b64": neighbors_b64,
            "domain": "lattice",
        },
        {
            "size": output_bytes,
            "direction": "output",
        },
    ]))
}

fn f64_slice_to_bytes(data: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * size_of::<f64>());
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn u32_slice_to_bytes(data: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * size_of::<u32>());
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn f64_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f64>, HotSpringError> {
    if !bytes.len().is_multiple_of(size_of::<f64>()) {
        return Err(HotSpringError::Ipc(format!(
            "output byte length {} is not f64-aligned",
            bytes.len()
        )));
    }
    bytes
        .chunks_exact(size_of::<f64>())
        .map(|chunk| {
            let arr: [u8; 8] = chunk.try_into().map_err(|_| {
                HotSpringError::Ipc("invalid f64 chunk".into())
            })?;
            Ok(f64::from_le_bytes(arr))
        })
        .collect()
}

/// Map hotSpring buffer directions to toadStool cylinder lifecycle names.
fn buffers_for_ipc(buffers: &Value) -> Value {
    let Some(arr) = buffers.as_array() else {
        return buffers.clone();
    };

    let mapped: Vec<Value> = arr
        .iter()
        .map(|buf| {
            let mut out = buf.clone();
            if let Some(dir) = buf.get("direction").and_then(Value::as_str) {
                let mapped_dir = match dir {
                    "input" => "in",
                    "output" => "out",
                    other => other,
                };
                out["direction"] = Value::String(mapped_dir.to_string());
            }
            out
        })
        .collect();

    Value::Array(mapped)
}

fn submit_lattice_dispatch(
    nucleus: &NucleusContext,
    binary_b64: &str,
    bdf: Option<&str>,
    dispatch_dims: [u32; 3],
    buffers: &Value,
) -> Result<String, HotSpringError> {
    let mut submit = serde_json::json!({
        "binary_b64": binary_b64,
        "dispatch_dims": dispatch_dims,
        "buffers": buffers_for_ipc(buffers),
        "spring": "hotSpring",
        "dispatch_mode": "passthrough",
    });

    if let Some(bdf_val) = bdf {
        submit["bdf"] = Value::String(bdf_val.to_string());
    }

    let resp = nucleus.call_by_capability("compute", "compute.dispatch.submit", submit)?;

    let result = parse_jsonrpc_response(&resp, "compute.dispatch.submit")?;
    result
        .get("job_id")
        .and_then(Value::as_str)
        .map(String::from)
        .ok_or_else(|| HotSpringError::Ipc("compute.dispatch.submit: missing job_id".into()))
}

fn poll_dispatch_output(
    nucleus: &NucleusContext,
    job_id: &str,
    output_elements: usize,
) -> Result<Vec<f64>, HotSpringError> {
    for attempt in 0..RESULT_POLL_ATTEMPTS {
        let envelope = retrieve_result(nucleus, job_id)?;

        if let Some(output) = extract_output_from_envelope(&envelope, output_elements)? {
            return Ok(output);
        }

        let status = envelope
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if status == "failed" {
            let err = envelope
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("dispatch failed");
            return Err(HotSpringError::Ipc(format!(
                "compute.dispatch.result: {err}"
            )));
        }

        if attempt + 1 < RESULT_POLL_ATTEMPTS {
            std::thread::sleep(std::time::Duration::from_millis(RESULT_POLL_INTERVAL_MS));
        }
    }

    Err(HotSpringError::Ipc(format!(
        "compute.dispatch.result: timed out waiting for job {job_id}"
    )))
}

fn extract_output_from_envelope(
    envelope: &Value,
    output_elements: usize,
) -> Result<Option<Vec<f64>>, HotSpringError> {
    let output = envelope
        .get("output")
        .or_else(|| envelope.get("result"));

    let Some(output) = output else {
        return Ok(None);
    };

    if let Some(data_b64) = output.get("data_b64").and_then(Value::as_str) {
        let bytes = base64_encode::decode(data_b64.as_bytes())
            .map_err(|e| HotSpringError::Ipc(format!("output base64 decode: {e}")))?;
        return Ok(Some(f64_bytes_to_vec(&bytes)?));
    }

    if let Some(buffers) = output.get("buffers").and_then(Value::as_array) {
        if let Some(last) = buffers.last() {
            if let Some(data_b64) = last.get("data_b64").and_then(Value::as_str) {
                let bytes = base64_encode::decode(data_b64.as_bytes()).map_err(|e| {
                    HotSpringError::Ipc(format!("buffer output base64 decode: {e}"))
                })?;
                let mut values = f64_bytes_to_vec(&bytes)?;
                values.truncate(output_elements);
                return Ok(Some(values));
            }
        }
    }

    if let Some(arr) = output.get("data").and_then(Value::as_array) {
        let values: Vec<f64> = arr
            .iter()
            .filter_map(Value::as_f64)
            .collect();
        if !values.is_empty() {
            return Ok(Some(values));
        }
    }

    if let Some(arr) = output.as_array() {
        let values: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
        if !values.is_empty() {
            return Ok(Some(values));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_volume_from_dims() {
        assert_eq!(lattice_volume([4, 4, 4, 4]), 256);
        assert_eq!(lattice_volume([8, 4, 4, 4]), 512);
    }

    #[test]
    fn dispatch_dims_scales_with_volume() {
        assert_eq!(dispatch_dims_for_sites(256), [1, 1, 1]);
        assert_eq!(dispatch_dims_for_sites(257), [2, 1, 1]);
        assert_eq!(dispatch_dims_for_sites(512), [2, 1, 1]);
    }

    #[test]
    fn plaq_uniform_params_layout() {
        let bytes = make_plaq_uniform_params(256);
        assert_eq!(bytes.len(), 16);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 256);
    }

    #[test]
    fn force_uniform_params_includes_beta() {
        let bytes = make_force_uniform_params(64, 6.0);
        assert_eq!(bytes.len(), 16);
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 64);
        assert!((f64::from_le_bytes(bytes[8..16].try_into().unwrap()) - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn serialize_lattice_buffers_wire_format() {
        let params = LatticeDispatchParams {
            shader_name: "wilson_plaquette_f64".into(),
            bdf: None,
            lattice_dims: [4, 4, 4, 4],
            links: vec![1.0, 0.0],
            neighbors: vec![1, 2, 3],
            params: make_plaq_uniform_params(256),
            output_elements: 256,
        };

        let buffers = serialize_lattice_buffers(&params).expect("serialize");
        let arr = buffers.as_array().expect("array");
        assert_eq!(arr.len(), 4);

        assert_eq!(arr[0]["direction"], "input");
        assert_eq!(arr[0]["domain"], "lattice");
        assert!(arr[0]["data_b64"].is_string());
        assert_eq!(arr[0]["size"], 16);

        assert_eq!(arr[1]["direction"], "input");
        assert_eq!(arr[1]["size"], 16);
        let links_decoded = base64_encode::decode(
            arr[1]["data_b64"]
                .as_str()
                .expect("links b64")
                .as_bytes(),
        )
        .expect("decode links");
        assert_eq!(links_decoded.len(), 16);

        assert_eq!(arr[2]["direction"], "input");
        assert_eq!(arr[2]["size"], 12);

        assert_eq!(arr[3]["direction"], "output");
        assert_eq!(arr[3]["size"], 256 * 8);
        assert!(arr[3].get("data_b64").is_none());
    }

    #[test]
    fn buffers_for_ipc_maps_direction_aliases() {
        let wire = serde_json::json!([
            { "direction": "input", "size": 4 },
            { "direction": "output", "size": 8 },
        ]);
        let ipc = buffers_for_ipc(&wire);
        assert_eq!(ipc[0]["direction"], "in");
        assert_eq!(ipc[1]["direction"], "out");
    }

    #[test]
    fn f64_roundtrip_bytes() {
        let src = vec![1.0, -2.5, std::f64::consts::PI];
        let bytes = f64_slice_to_bytes(&src);
        let back = f64_bytes_to_vec(&bytes).expect("decode");
        assert_eq!(back, src);
    }
}
