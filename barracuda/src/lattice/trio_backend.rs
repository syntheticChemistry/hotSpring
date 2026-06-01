// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU HMC backends: local wgpu (`GpuF64`) and sovereign trio IPC
//! (coralReef compile + toadStool dispatch).
//!
//! [`TrioGpuBackend`] routes shader compilation through `shader.compile.wgsl`
//! and kernel dispatch through `compute.dispatch.submit` /
//! `compute.dispatch.result`. [`LocalGpuBackend`] preserves the existing
//! in-process wgpu path for parity testing and offline development.
//!
//! Wiring into `gpu_hmc` state machines is deferred — this module exposes the
//! [`GpuHmcBackend`] trait and both implementations only.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::HotSpringError;
use crate::gpu::GpuF64;
use crate::primal_bridge::{NucleusContext, parse_jsonrpc_response};

/// Buffer role for trio dispatch IPC and the shared HMC backend trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferDirection {
    /// Upload-only (`direction: "in"` on the wire).
    Input,
    /// Download-only (`direction: "out"`).
    Output,
    /// Upload + download (`direction: "inout"`).
    InputOutput,
}

/// Storage buffer descriptor for trio IPC dispatch.
#[derive(Debug, Clone)]
pub struct DispatchBuffer {
    /// Host-side bytes for input / in-out buffers; `None` for output-only.
    pub data: Option<Vec<u8>>,
    /// Buffer size in bytes (required even for output-only buffers).
    pub size: u64,
    pub direction: BufferDirection,
}

/// Completed trio dispatch with readback payloads.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// Readback bytes for each output / in-out buffer, in submission order.
    pub outputs: Vec<Vec<u8>>,
    pub elapsed_ms: u64,
}

/// Compiled kernel handle shared by both backends.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub name: String,
}

/// Buffer descriptor for the shared HMC backend trait.
#[derive(Debug, Clone)]
pub struct GpuHmcBuffer {
    pub binding: u32,
    pub direction: BufferDirection,
    pub size: u64,
    pub data: Option<Vec<u8>>,
}

/// Swappable GPU backend for lattice HMC workloads.
pub trait GpuHmcBackend {
    fn compile(&mut self, name: &str, source: &str) -> Result<CompiledKernel, HotSpringError>;

    fn dispatch(
        &self,
        kernel: &CompiledKernel,
        buffers: &[GpuHmcBuffer],
        dims: [u32; 3],
    ) -> Result<(), HotSpringError>;

    fn readback(&self, buffer: &GpuHmcBuffer) -> Result<Vec<f64>, HotSpringError>;
}

struct CompiledShader {
    binary: Vec<u8>,
    shader_info: Option<serde_json::Value>,
}

/// Sovereign trio backend: coralReef WGSL compile + toadStool dispatch IPC.
pub struct TrioGpuBackend {
    nucleus: NucleusContext,
    bdf: Option<String>,
    compiled_cache: HashMap<String, CompiledShader>,
    last_result: RefCell<Option<DispatchResult>>,
    /// Bindings of output/in-out buffers from the last dispatch, in readback order.
    last_output_bindings: RefCell<Vec<u32>>,
}

impl TrioGpuBackend {
    #[must_use]
    pub fn new(bdf: Option<String>) -> Self {
        Self {
            nucleus: NucleusContext::detect(),
            bdf,
            compiled_cache: HashMap::new(),
            last_result: RefCell::new(None),
            last_output_bindings: RefCell::new(Vec::new()),
        }
    }

    #[must_use]
    pub fn with_nucleus(nucleus: NucleusContext, bdf: Option<String>) -> Self {
        Self {
            nucleus,
            bdf,
            compiled_cache: HashMap::new(),
            last_result: RefCell::new(None),
            last_output_bindings: RefCell::new(Vec::new()),
        }
    }

    /// Compile a shader via coralReef IPC, caching the result.
    pub fn compile(
        &mut self,
        shader_name: &str,
        wgsl_source: &str,
    ) -> Result<&[u8], HotSpringError> {
        self.ensure_compiled(shader_name, wgsl_source)
    }

    fn ensure_compiled(
        &mut self,
        shader_name: &str,
        wgsl_source: &str,
    ) -> Result<&[u8], HotSpringError> {
        if !self.compiled_cache.contains_key(shader_name) {
            let compile_params = serde_json::json!({ "wgsl_source": wgsl_source });
            let compile_resp = self
                .nucleus
                .call_by_capability("shader", "shader.compile.wgsl", compile_params)?;

            let compile_result =
                parse_jsonrpc_response(&compile_resp, "shader.compile.wgsl")?;

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

            let binary = crate::base64_encode::decode(binary_b64.as_bytes()).map_err(|e| {
                HotSpringError::Ipc(format!("shader.compile.wgsl: binary_b64 decode: {e}"))
            })?;

            let shader_info = compile_result.get("shader_info").cloned();

            self.compiled_cache.insert(
                shader_name.to_string(),
                CompiledShader {
                    binary,
                    shader_info,
                },
            );
        }

        Ok(&self
            .compiled_cache
            .get(shader_name)
            .map_or(&[][..], |entry| entry.binary.as_slice()))
    }

    /// Submit a compiled kernel with buffers to toadStool.
    pub fn dispatch(
        &self,
        compiled_binary: &[u8],
        buffers: Vec<DispatchBuffer>,
        workgroup_dims: [u32; 3],
    ) -> Result<String, HotSpringError> {
        let binary_b64 = crate::base64_encode::encode(compiled_binary);

        let mut submit = serde_json::json!({
            "binary_b64": binary_b64,
            "dispatch_dims": workgroup_dims,
            "buffers": serialize_buffers(&buffers),
            "spring": "hotSpring",
            "dispatch_mode": "passthrough",
        });

        if let Some(bdf_val) = &self.bdf {
            submit["bdf"] = serde_json::Value::String(bdf_val.clone());
        }

        let resp = self
            .nucleus
            .call_by_capability("compute", "compute.dispatch.submit", submit)?;

        let result = parse_jsonrpc_response(&resp, "compute.dispatch.submit")?;
        extract_job_id_from_result(&result)
    }

    /// Poll for dispatch result.
    pub fn poll_result(&self, job_id: &str) -> Result<DispatchResult, HotSpringError> {
        let params = serde_json::json!({ "job_id": job_id });
        let resp = self
            .nucleus
            .call_by_capability("compute", "compute.dispatch.result", params)?;

        let result = parse_jsonrpc_response(&resp, "compute.dispatch.result")?;
        parse_dispatch_result(&result)
    }

    /// Compile + dispatch + readback in one call.
    pub fn run_kernel(
        &mut self,
        shader_name: &str,
        wgsl_source: &str,
        buffers: Vec<DispatchBuffer>,
        workgroup_dims: [u32; 3],
    ) -> Result<DispatchResult, HotSpringError> {
        let binary = self.compile(shader_name, wgsl_source)?.to_vec();
        let shader_info = self
            .compiled_cache
            .get(shader_name)
            .and_then(|entry| entry.shader_info.clone());

        let job_id = self.dispatch_with_shader_info(
            &binary,
            shader_info.as_ref(),
            &buffers,
            workgroup_dims,
        )?;

        let dispatch_result = self.poll_result(&job_id)?;
        self.store_dispatch_outcome(&buffers, dispatch_result.clone());
        Ok(dispatch_result)
    }

    fn store_dispatch_outcome(&self, buffers: &[DispatchBuffer], result: DispatchResult) {
        let output_bindings: Vec<u32> = buffers
            .iter()
            .enumerate()
            .filter(|(_, buf)| {
                matches!(
                    buf.direction,
                    BufferDirection::Output | BufferDirection::InputOutput
                )
            })
            .map(|(binding, _)| binding as u32)
            .collect();
        *self.last_output_bindings.borrow_mut() = output_bindings;
        *self.last_result.borrow_mut() = Some(result);
    }

    fn dispatch_with_shader_info(
        &self,
        compiled_binary: &[u8],
        shader_info: Option<&serde_json::Value>,
        buffers: &[DispatchBuffer],
        workgroup_dims: [u32; 3],
    ) -> Result<String, HotSpringError> {
        let binary_b64 = crate::base64_encode::encode(compiled_binary);

        let mut submit = serde_json::json!({
            "binary_b64": binary_b64,
            "dispatch_dims": workgroup_dims,
            "buffers": serialize_buffers(&buffers),
            "spring": "hotSpring",
            "dispatch_mode": "passthrough",
        });

        if let Some(bdf_val) = &self.bdf {
            submit["bdf"] = serde_json::Value::String(bdf_val.clone());
        }
        if let Some(si) = shader_info {
            submit["shader_info"] = si.clone();
        }

        let resp = self
            .nucleus
            .call_by_capability("compute", "compute.dispatch.submit", submit)?;

        let result = parse_jsonrpc_response(&resp, "compute.dispatch.submit")?;
        extract_job_id_from_result(&result)
    }
}

impl GpuHmcBackend for TrioGpuBackend {
    fn compile(&mut self, name: &str, source: &str) -> Result<CompiledKernel, HotSpringError> {
        self.ensure_compiled(name, source)?;
        Ok(CompiledKernel {
            name: name.to_string(),
        })
    }

    fn dispatch(
        &self,
        kernel: &CompiledKernel,
        buffers: &[GpuHmcBuffer],
        dims: [u32; 3],
    ) -> Result<(), HotSpringError> {
        let entry = self.compiled_cache.get(&kernel.name).ok_or_else(|| {
            HotSpringError::InvalidOperation(format!(
                "kernel {} not compiled on trio backend",
                kernel.name
            ))
        })?;

        let dispatch_buffers: Vec<DispatchBuffer> =
            buffers.iter().map(gpu_hmc_buffer_to_dispatch).collect();

        let job_id = self.dispatch_with_shader_info(
            &entry.binary,
            entry.shader_info.as_ref(),
            &dispatch_buffers,
            dims,
        )?;

        let result = self.poll_result(&job_id)?;
        self.store_dispatch_outcome(&dispatch_buffers, result);
        Ok(())
    }

    fn readback(&self, buffer: &GpuHmcBuffer) -> Result<Vec<f64>, HotSpringError> {
        let output_index =
            output_buffer_index(&self.last_output_bindings.borrow(), buffer)?;
        let last = self.last_result.borrow();
        let dispatch_result = last.as_ref().ok_or_else(|| {
            HotSpringError::InvalidOperation("no trio dispatch result to read back".into())
        })?;

        let bytes = dispatch_result.outputs.get(output_index).ok_or_else(|| {
            HotSpringError::GpuCompute(format!(
                "no readback for binding {} ({} outputs)",
                buffer.binding,
                dispatch_result.outputs.len()
            ))
        })?;

        bytes_to_f64(bytes)
    }
}

/// In-process wgpu backend wrapping [`GpuF64`].
pub struct LocalGpuBackend {
    gpu: GpuF64,
    pipelines: RefCell<HashMap<String, wgpu::ComputePipeline>>,
    readbacks: RefCell<HashMap<u32, Vec<f64>>>,
}

impl LocalGpuBackend {
    #[must_use]
    pub fn new(gpu: GpuF64) -> Self {
        Self {
            gpu,
            pipelines: RefCell::new(HashMap::new()),
            readbacks: RefCell::new(HashMap::new()),
        }
    }

    #[must_use]
    pub fn gpu(&self) -> &GpuF64 {
        &self.gpu
    }
}

impl GpuHmcBackend for LocalGpuBackend {
    fn compile(&mut self, name: &str, source: &str) -> Result<CompiledKernel, HotSpringError> {
        let pipeline = self.gpu.create_pipeline_f64(source, name);
        self.pipelines
            .borrow_mut()
            .insert(name.to_string(), pipeline);
        Ok(CompiledKernel {
            name: name.to_string(),
        })
    }

    fn dispatch(
        &self,
        kernel: &CompiledKernel,
        buffers: &[GpuHmcBuffer],
        dims: [u32; 3],
    ) -> Result<(), HotSpringError> {
        let pipeline = self
            .pipelines
            .borrow()
            .get(&kernel.name)
            .cloned()
            .ok_or_else(|| {
                HotSpringError::InvalidOperation(format!(
                    "kernel {} not compiled on local backend",
                    kernel.name
                ))
            })?;

        let mut wgpu_buffers = Vec::with_capacity(buffers.len());
        for (idx, buf) in buffers.iter().enumerate() {
            let label = format!("hmc_buf_{}", buf.binding);
            let wgpu_buf = if let Some(ref data) = buf.data {
                self.gpu.create_storage_buffer_init(data, &label)
            } else {
                self.gpu
                    .create_storage_buffer_init(&vec![0u8; buf.size as usize], &label)
            };
            if idx as u32 != buf.binding {
                return Err(HotSpringError::InvalidOperation(format!(
                    "buffer binding {} must match slice index {idx}",
                    buf.binding
                )));
            }
            wgpu_buffers.push(wgpu_buf);
        }

        let refs: Vec<&wgpu::Buffer> = wgpu_buffers.iter().collect();
        let bind_group = self.gpu.create_bind_group(&pipeline, &refs);

        let mut encoder = self.gpu.begin_encoder("gpu_hmc_dispatch");
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_hmc_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dims[0], dims[1], dims[2]);
        }
        self.gpu.submit_encoder(encoder);

        let mut readbacks = self.readbacks.borrow_mut();
        readbacks.clear();
        for (buf, wgpu_buf) in buffers.iter().zip(wgpu_buffers.iter()) {
            if matches!(
                buf.direction,
                BufferDirection::Output | BufferDirection::InputOutput
            ) {
                let f64_count = (buf.size / 8) as usize;
                let staging = self
                    .gpu
                    .create_staging_buffer(buf.size as usize, "hmc_readback");
                let mut copy_encoder = self.gpu.begin_encoder("hmc_readback");
                copy_encoder.copy_buffer_to_buffer(
                    wgpu_buf,
                    0,
                    &staging,
                    0,
                    buf.size,
                );
                self.gpu.submit_encoder(copy_encoder);
                let values = self.gpu.read_staging_f64_n(&staging, f64_count)?;
                readbacks.insert(buf.binding, values);
            }
        }

        Ok(())
    }

    fn readback(&self, buffer: &GpuHmcBuffer) -> Result<Vec<f64>, HotSpringError> {
        self.readbacks
            .borrow()
            .get(&buffer.binding)
            .cloned()
            .ok_or_else(|| {
                HotSpringError::GpuCompute(format!(
                    "no local readback cached for binding {}",
                    buffer.binding
                ))
            })
    }
}

/// Serialize buffer descriptors for `compute.dispatch.submit`.
///
/// Wire format follows the toadStool trio contract:
/// `{ binding, size, direction, data_b64?, usage }`.
#[must_use]
pub fn serialize_buffers(buffers: &[DispatchBuffer]) -> serde_json::Value {
    let entries: Vec<serde_json::Value> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buf)| {
            let direction = direction_to_wire(buf.direction);
            let mut entry = serde_json::json!({
                "binding": binding,
                "size": buf.size,
                "direction": direction,
                "usage": "storage",
            });
            if let Some(ref data) = buf.data {
                entry["data_b64"] = serde_json::Value::String(crate::base64_encode::encode(data));
            }
            entry
        })
        .collect();
    serde_json::Value::Array(entries)
}

fn direction_to_wire(direction: BufferDirection) -> &'static str {
    match direction {
        BufferDirection::Input => "in",
        BufferDirection::Output => "out",
        BufferDirection::InputOutput => "inout",
    }
}

fn gpu_hmc_buffer_to_dispatch(buf: &GpuHmcBuffer) -> DispatchBuffer {
    DispatchBuffer {
        data: buf.data.clone(),
        size: buf.size,
        direction: buf.direction,
    }
}

fn output_buffer_index(
    output_bindings: &[u32],
    buffer: &GpuHmcBuffer,
) -> Result<usize, HotSpringError> {
    if !matches!(
        buffer.direction,
        BufferDirection::Output | BufferDirection::InputOutput
    ) {
        return Err(HotSpringError::InvalidOperation(format!(
            "binding {} is not an output buffer",
            buffer.binding
        )));
    }
    output_bindings
        .iter()
        .position(|&binding| binding == buffer.binding)
        .ok_or_else(|| {
            HotSpringError::GpuCompute(format!(
                "binding {} was not part of the last trio dispatch outputs",
                buffer.binding
            ))
        })
}

fn extract_job_id_from_result(result: &serde_json::Value) -> Result<String, HotSpringError> {
    result
        .get("job_id")
        .and_then(serde_json::Value::as_str)
        .map(String::from)
        .ok_or_else(|| HotSpringError::Ipc("compute.dispatch.submit: missing job_id".into()))
}

fn parse_dispatch_result(result: &serde_json::Value) -> Result<DispatchResult, HotSpringError> {
    if let Some(status) = result.get("status").and_then(serde_json::Value::as_str)
        && status == "failed"
    {
        let msg = result
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("dispatch failed");
        return Err(HotSpringError::Ipc(format!(
            "compute.dispatch.result: {msg}"
        )));
    }

    let output = result.get("output").unwrap_or(result);
    let outputs = parse_output_buffers(output)?;

    let elapsed_ms = result
        .get("metadata")
        .and_then(|m| m.get("elapsed_ms"))
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            result.get("timing").and_then(|timing| {
                let dispatch = timing
                    .get("dispatch_ms")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let readback = timing
                    .get("readback_ms")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                Some(dispatch + readback)
            })
        })
        .unwrap_or(0);

    Ok(DispatchResult {
        outputs,
        elapsed_ms,
    })
}

fn parse_output_buffers(output: &serde_json::Value) -> Result<Vec<Vec<u8>>, HotSpringError> {
    if let Some(buffers) = output.get("buffers").and_then(serde_json::Value::as_array) {
        let mut out = Vec::with_capacity(buffers.len());
        for (i, buf) in buffers.iter().enumerate() {
            if let Some(err) = buf.get("error").and_then(serde_json::Value::as_str) {
                return Err(HotSpringError::Ipc(format!("buffer[{i}] readback: {err}")));
            }
            let b64 = buf
                .get("data_b64")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    HotSpringError::Ipc(format!("buffer[{i}] missing data_b64 in result"))
                })?;
            let bytes = crate::base64_encode::decode(b64.as_bytes()).map_err(|e| {
                HotSpringError::Ipc(format!("buffer[{i}] data_b64 decode: {e}"))
            })?;
            out.push(bytes);
        }
        return Ok(out);
    }

    if let Some(outputs) = output.get("outputs").and_then(serde_json::Value::as_array) {
        let mut out = Vec::with_capacity(outputs.len());
        for (i, item) in outputs.iter().enumerate() {
            let b64 = item.as_str().ok_or_else(|| {
                HotSpringError::Ipc(format!("outputs[{i}] is not a base64 string"))
            })?;
            let bytes = crate::base64_encode::decode(b64.as_bytes()).map_err(|e| {
                HotSpringError::Ipc(format!("outputs[{i}] decode: {e}"))
            })?;
            out.push(bytes);
        }
        return Ok(out);
    }

    Ok(Vec::new())
}

fn bytes_to_f64(data: &[u8]) -> Result<Vec<f64>, HotSpringError> {
    if !data.len().is_multiple_of(8) {
        return Err(HotSpringError::GpuCompute(format!(
            "readback size {} is not a multiple of 8",
            data.len()
        )));
    }
    let mut out = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        let arr: [u8; 8] = chunk
            .try_into()
            .map_err(|_| HotSpringError::GpuCompute("invalid f64 chunk".into()))?;
        out.push(f64::from_le_bytes(arr));
    }
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "buffer serialization tests use expect on constructed JSON"
)]
mod tests {
    use super::*;

    #[test]
    fn serialize_input_buffer_includes_data_b64() {
        let json = serialize_buffers(&[DispatchBuffer {
            data: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            size: 4,
            direction: BufferDirection::Input,
        }]);
        let arr = json.as_array().expect("array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["binding"], 0);
        assert_eq!(arr[0]["size"], 4);
        assert_eq!(arr[0]["direction"], "in");
        assert_eq!(arr[0]["usage"], "storage");
        assert!(arr[0]["data_b64"].is_string());
    }

    #[test]
    fn serialize_output_buffer_omits_data_b64() {
        let json = serialize_buffers(&[DispatchBuffer {
            data: None,
            size: 128,
            direction: BufferDirection::Output,
        }]);
        let arr = json.as_array().expect("array");
        assert_eq!(arr[0]["direction"], "out");
        assert!(arr[0].get("data_b64").is_none());
    }

    #[test]
    fn serialize_inout_buffer_uses_inout_direction() {
        let json = serialize_buffers(&[DispatchBuffer {
            data: Some(vec![1, 2, 3, 4]),
            size: 4,
            direction: BufferDirection::InputOutput,
        }]);
        let arr = json.as_array().expect("array");
        assert_eq!(arr[0]["direction"], "inout");
        assert!(arr[0]["data_b64"].is_string());
    }

    #[test]
    fn serialize_multiple_buffers_preserves_binding_order() {
        let json = serialize_buffers(&[
            DispatchBuffer {
                data: Some(vec![1]),
                size: 1,
                direction: BufferDirection::Input,
            },
            DispatchBuffer {
                data: None,
                size: 8,
                direction: BufferDirection::Output,
            },
        ]);
        let arr = json.as_array().expect("array");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["binding"], 0);
        assert_eq!(arr[1]["binding"], 1);
        assert_eq!(arr[1]["direction"], "out");
    }

    #[test]
    fn parse_dispatch_result_decodes_buffer_array() {
        let payload = serde_json::json!({
            "status": "completed",
            "output": {
                "buffers": [
                    { "size": 4, "data_b64": "AQIDBA==" }
                ]
            },
            "timing": { "dispatch_ms": 3, "readback_ms": 2 }
        });
        let parsed = parse_dispatch_result(&payload).expect("parse");
        assert_eq!(parsed.outputs, vec![vec![1, 2, 3, 4]]);
        assert_eq!(parsed.elapsed_ms, 5);
    }
}
