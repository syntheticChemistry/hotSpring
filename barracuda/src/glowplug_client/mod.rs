// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC client for toadStool **device management** RPCs over the
//! NUCLEUS Unix socket.
//!
//! This client wraps device lifecycle methods: `device.list`, `device.get`,
//! `device.swap`, `device.warm_catch`, `device.vfio.*`, `device.reset`,
//! `device.resurrect`, `capture.training`, and `sovereign.boot`.
//!
//! **Compute dispatch** (shader submission + result retrieval) is in
//! [`crate::compute_dispatch`] — use [`crate::compute_dispatch::compile_and_submit`]
//! for the full WGSL→binary→hardware pipeline.
//!
//! Post-excision (coralReef Sprint 9, May 2026): toadStool is the sole
//! provider for device management, lifecycle orchestration, and sovereign
//! dispatch. The `compute` NUCLEUS domain resolves to toadStool's socket.
//! Legacy glowplug daemon is no longer separate; discovery falls through
//! gracefully if toadStool is not yet available.
//!
//! All calls use [`crate::primal_bridge::send_jsonrpc`] (JSON-RPC 2.0, newline-framed).
//!
//! # Module structure
//!
//! - **`types`** — Protocol types: request options, response structs,
//!   error variants, wire-format helpers.
//! - **This module** — `GlowplugClient` impl (device lifecycle RPCs),
//!   free helper functions, and protocol tests.

mod types;

pub use types::*;

use std::path::{Path, PathBuf};

use crate::ember_types::{
    Bar0RangeResult, DeviceLifecycleResult, ExperimentLifecycleResult, GlowplugRegisterDumpResult,
};
use crate::primal_bridge::{NucleusContext, send_jsonrpc};

/// Connected glowplug endpoint (socket path only; each RPC opens a short-lived connection).
#[derive(Debug, Clone)]
pub struct GlowplugClient {
    socket: PathBuf,
}

impl GlowplugClient {
    /// Build a client from a discovered toadStool compute endpoint.
    ///
    /// Looks up the `compute` NUCLEUS domain (toadStool), then falls
    /// back to `shader` (coralReef compiler) for backward compat.
    pub fn from_nucleus(nucleus: &NucleusContext) -> Result<Self, GlowplugError> {
        let ep = nucleus
            .get_by_capability("compute")
            .or_else(|| nucleus.get_by_capability("shader"))
            .ok_or(GlowplugError::NoComputeEndpoint)?;
        if !ep.alive {
            return Err(GlowplugError::EndpointNotAlive);
        }
        Ok(Self {
            socket: PathBuf::from(&ep.socket),
        })
    }

    /// Build a client from an explicit socket path.
    #[must_use]
    pub fn from_socket(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
        }
    }

    /// Raw RPC: send method+params, handle JSON-RPC error envelope.
    fn call(
        &self,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, GlowplugError> {
        rpc_result(&self.socket, method, params)
    }

    /// Try NUCLEUS capability routing first, then direct socket call.
    fn call_with_nucleus_fallback(
        &self,
        domain: &str,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, GlowplugError> {
        let ctx = NucleusContext::detect();
        if let Ok(resp) = ctx.call_by_capability(domain, method, params.clone()) {
            if let Some(result) = resp.get("result").cloned() {
                return Ok(result);
            }
        }
        self.call(method, params)
    }

    // ── Device lifecycle ─────────────────────────────────────────────

    /// `device.list` — enumerate PCI devices the daemon knows about.
    ///
    /// Handles two toadStool response formats:
    /// - Full rows: `{ "devices": [ { "bdf": "...", ... }, ... ] }`
    /// - BDF-only:  `{ "devices": [ "0000:02:00.0", ... ] }` (older daemons)
    pub fn list_devices(&self) -> Result<Vec<GlowplugDeviceSummary>, GlowplugError> {
        let v = self.call("device.list", &serde_json::json!({}))?;
        let arr = v
            .get("devices")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                GlowplugError::InvalidPayload("device.list missing devices array".into())
            })?;
        let mut out = Vec::with_capacity(arr.len());
        for dev in arr {
            if let Some(bdf_str) = dev.as_str() {
                match self.get_device(bdf_str) {
                    Ok(detail) => out.push(GlowplugDeviceSummary {
                        bdf: detail.bdf,
                        vendor: detail.vendor.unwrap_or_default(),
                        name: detail.name,
                        personality: detail.personality.unwrap_or_else(|| "unknown".into()),
                        protected: detail.protected.unwrap_or(false),
                        health: GlowplugDeviceHealthSummary {
                            vram_alive: detail.vram_alive.unwrap_or(false),
                            domains_faulted: detail.domains_faulted.unwrap_or(0),
                        },
                    }),
                    Err(_) => out.push(GlowplugDeviceSummary {
                        bdf: bdf_str.to_string(),
                        vendor: String::new(),
                        name: None,
                        personality: "unknown".to_string(),
                        protected: false,
                        health: GlowplugDeviceHealthSummary {
                            vram_alive: false,
                            domains_faulted: 0,
                        },
                    }),
                }
            } else {
                let row: GlowplugListRow = serde_json::from_value(dev.clone())
                    .map_err(|e| GlowplugError::InvalidPayload(format!("device row: {e}")))?;
                out.push(row.into());
            }
        }
        Ok(out)
    }

    /// `device.get` — full detail for one device by BDF.
    pub fn get_device(&self, bdf: &str) -> Result<GlowplugDeviceDetail, GlowplugError> {
        let v = self.call_with_nucleus_fallback(
            "compute",
            "device.get",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.get: {e}")))
    }

    /// `health.liveness` / `health.check` — daemon health.
    pub fn daemon_health(&self) -> Result<GlowplugDaemonHealth, GlowplugError> {
        let v = self.call("health.liveness", &serde_json::json!({}))?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("health.liveness: {e}")))
    }

    /// `device.swap` — swap a device from current driver to `target_driver`.
    pub fn device_swap(
        &self,
        bdf: &str,
        target_driver: &str,
    ) -> Result<DeviceLifecycleResult, GlowplugError> {
        let v = self.call_with_nucleus_fallback(
            "compute",
            "device.swap",
            &serde_json::json!({
                "bdf": bdf,
                "target": target_driver,
            }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.swap: {e}")))
    }

    /// `device.register_dump` — raw PCI BAR0 register dump.
    pub fn register_dump(&self, bdf: &str) -> Result<GlowplugRegisterDumpResult, GlowplugError> {
        let v = self.call("device.register_dump", &serde_json::json!({ "bdf": bdf }))?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("register_dump: {e}")))
    }

    /// `device.bar0_range` — read a contiguous BAR0 register range.
    pub fn bar0_range(
        &self,
        bdf: &str,
        start: u64,
        count: u64,
    ) -> Result<Bar0RangeResult, GlowplugError> {
        let v = self.call(
            "device.bar0_range",
            &serde_json::json!({
                "bdf": bdf,
                "start": start,
                "count": count,
            }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("bar0_range: {e}")))
    }

    /// `device.experiment_lifecycle` — create/destroy experiment sessions.
    pub fn experiment_lifecycle(
        &self,
        bdf: &str,
        action: &str,
    ) -> Result<ExperimentLifecycleResult, GlowplugError> {
        let v = self.call(
            "device.experiment_lifecycle",
            &serde_json::json!({
                "bdf": bdf,
                "action": action,
            }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("experiment_lifecycle: {e}")))
    }

    /// `compute.dispatch` — compile-then-dispatch a shader kernel.
    pub fn dispatch(
        &self,
        bdf: &str,
        kernel: &[u8],
        buffers: &[Vec<u8>],
        output_sizes: &[u64],
        options: &GlowplugDispatchOptions,
    ) -> Result<Vec<Vec<u8>>, GlowplugError> {
        let params = build_dispatch_params(bdf, kernel, buffers, output_sizes, options);
        let result = self.call_with_nucleus_fallback("compute", "compute.dispatch", &params)?;
        decode_dispatch_outputs(&result)
    }

    /// `device.reset` — issue a secondary bus reset (SBR) on the device.
    pub fn device_reset(&self, bdf: &str) -> Result<DeviceLifecycleResult, GlowplugError> {
        let v = self.call_with_nucleus_fallback(
            "compute",
            "device.reset",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.reset: {e}")))
    }

    /// `device.resurrect` — resurrect a faulted device (re-probe + re-init).
    pub fn device_resurrect(&self, bdf: &str) -> Result<DeviceLifecycleResult, GlowplugError> {
        let v = self.call_with_nucleus_fallback(
            "compute",
            "device.resurrect",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.resurrect: {e}")))
    }

    /// `device.health` — detailed per-device health (domain breakdown, fault history).
    pub fn device_health(&self, bdf: &str) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback(
            "compute",
            "device.health",
            &serde_json::json!({ "bdf": bdf }),
        )
    }

    /// `daemon.status` — glowplug daemon status (mode, uptime, fleet info).
    pub fn daemon_status(&self) -> Result<serde_json::Value, GlowplugError> {
        self.call("daemon.status", &serde_json::json!({}))
    }

    // ── Sovereign boot + training capture ────────────────────────────

    /// `capture.training` — capture a training recipe by observing an external
    /// driver's memory initialization on a cold GPU.
    pub fn capture_training(
        &self,
        bdf: &str,
        warm_driver: Option<&str>,
    ) -> Result<CaptureTrainingResult, GlowplugError> {
        let mut params = serde_json::json!({"bdf": bdf});
        if let Some(driver) = warm_driver {
            params["warm_driver"] = serde_json::Value::String(driver.to_string());
        }
        let v = self.call("capture.training", &params)?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("capture.training: {e}")))
    }

    /// `device.warm_catch` — catch a warm GPU after driver handoff.
    pub fn device_warm_catch(&self, bdf: &str) -> Result<WarmCatchResult, GlowplugError> {
        let v = self.call_with_nucleus_fallback(
            "compute",
            "device.warm_catch",
            &serde_json::json!({"bdf": bdf}),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.warm_catch: {e}")))
    }

    /// `device.vfio.open` — open VFIO device and create PBDMA channel.
    pub fn device_vfio_open(&self, bdf: &str) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback(
            "compute",
            "device.vfio.open",
            &serde_json::json!({"bdf": bdf}),
        )
    }

    /// `device.vfio.alloc` — allocate a DMA buffer for PBDMA dispatch.
    pub fn device_vfio_alloc(
        &self,
        bdf: &str,
        size: u64,
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback(
            "compute",
            "device.vfio.alloc",
            &serde_json::json!({"bdf": bdf, "size": size}),
        )
    }

    /// `device.vfio.roundtrip` — DMA buffer roundtrip test.
    pub fn device_vfio_roundtrip(
        &self,
        bdf: &str,
        data: &[u8],
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback(
            "compute",
            "device.vfio.roundtrip",
            &serde_json::json!({
                "bdf": bdf,
                "data_b64": crate::base64_encode::encode(data),
            }),
        )
    }

    /// `device.gr.init` — initialize GR compute context on warm-caught GPU.
    pub fn device_gr_init(
        &self,
        bdf: &str,
        method_entries: &[[u32; 2]],
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback(
            "compute",
            "device.gr.init",
            &serde_json::json!({
                "bdf": bdf,
                "method_entries": method_entries,
            }),
        )
    }

    /// `compute.context.init` — alias for [`Self::device_gr_init`].
    pub fn compute_context_init(
        &self,
        bdf: &str,
        method_entries: &[[u32; 2]],
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback(
            "compute",
            "compute.context.init",
            &serde_json::json!({
                "bdf": bdf,
                "method_entries": method_entries,
            }),
        )
    }

    /// `device.vfio.roundtrip` with optional GR init.
    pub fn device_vfio_roundtrip_with_gr_init(
        &self,
        bdf: &str,
        binary_b64: &str,
        shader_info: &serde_json::Value,
        dispatch_dims: [u32; 3],
        gr_init_entries: &[[u32; 2]],
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback(
            "compute",
            "device.vfio.roundtrip",
            &serde_json::json!({
                "bdf": bdf,
                "gr_init_entries": gr_init_entries,
                "binary_b64": binary_b64,
                "shader_info": shader_info,
                "dispatch_dims": dispatch_dims,
            }),
        )
    }

    /// `shader.compile.gemm` — compile a tensor-core GEMM kernel via coralReef.
    pub fn compile_gemm(
        &self,
        m: u32,
        n: u32,
        k: u32,
        arch: &str,
        precision: &str,
    ) -> Result<serde_json::Value, GlowplugError> {
        let nucleus = NucleusContext::detect();
        nucleus
            .call_by_capability(
                "shader",
                "shader.compile.gemm",
                serde_json::json!({
                    "m": m,
                    "n": n,
                    "k": k,
                    "arch": arch,
                    "precision": precision,
                }),
            )
            .map_err(|e| GlowplugError::Transport(format!("shader.compile.gemm: {e}")))
    }

    /// `sovereign.boot` — full orchestrated sovereign boot.
    pub fn sovereign_boot(&self, bdf: &str) -> Result<SovereignBootResult, GlowplugError> {
        let v = self.call_with_nucleus_fallback(
            "compute",
            "sovereign.boot",
            &serde_json::json!({"bdf": bdf}),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("sovereign.boot: {e}")))
    }

    /// `sovereign.init` — staged sovereign GPU initialization pipeline.
    ///
    /// This is the correct entry for full sovereign boot (PMC ramp → DEVINIT →
    /// SEC2 ACR → PMU → FECS), unlike `sovereign.boot` which was historically
    /// misrouted to driver swap.
    pub fn sovereign_init(
        &self,
        bdf: &str,
        opts: &SovereignInitOptions,
    ) -> Result<serde_json::Value, GlowplugError> {
        let mut params = serde_json::json!({ "bdf": bdf });
        if let Some(ref halt) = opts.halt_before {
            params["halt_before"] = serde_json::Value::String(halt.clone());
        }
        if let Some(ref vbios) = opts.vbios_rom_path {
            params["vbios_rom_path"] = serde_json::Value::String(vbios.clone());
        }
        if let Some(ref engine) = opts.engine_init_path {
            params["engine_init_path"] = serde_json::Value::String(engine.clone());
        }
        if opts.skip_gr_init {
            params["skip_gr_init"] = serde_json::Value::Bool(true);
        }
        self.call_with_nucleus_fallback("compute", "sovereign.init", &params)
    }

    /// Raw JSON-RPC to the compute/sovereign endpoint (NUCLEUS-routed when available).
    pub fn rpc_call(
        &self,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call_with_nucleus_fallback("compute", method, params)
    }
}

// ── Free functions ──────────────────────────────────────────────────

fn rpc_result(
    socket: &Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, GlowplugError> {
    let v = send_jsonrpc(socket, method, params)
        .map_err(|e| GlowplugError::Transport(e.to_string()))?;
    if let Some(err) = v.get("error") {
        let code = err
            .get("code")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(-1);
        let message = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("rpc error")
            .to_string();
        return Err(GlowplugError::JsonRpc { code, message });
    }
    v.get("result").cloned().ok_or(GlowplugError::MissingResult)
}

fn build_dispatch_params(
    bdf: &str,
    kernel: &[u8],
    buffers: &[Vec<u8>],
    output_sizes: &[u64],
    options: &GlowplugDispatchOptions,
) -> serde_json::Value {
    let shader_b64 = crate::base64_encode::encode(kernel);
    let inputs: Vec<String> = buffers
        .iter()
        .map(|b| crate::base64_encode::encode(b))
        .collect();
    serde_json::json!({
        "bdf": bdf,
        "shader": shader_b64,
        "inputs": inputs,
        "output_sizes": output_sizes,
        "dims": [options.dims[0], options.dims[1], options.dims[2]],
        "workgroup": [options.workgroup[0], options.workgroup[1], options.workgroup[2]],
        "shared_mem": options.shared_mem,
        "kernel_name": options.kernel_name,
    })
}

fn decode_dispatch_outputs(result: &serde_json::Value) -> Result<Vec<Vec<u8>>, GlowplugError> {
    let outputs = result
        .get("outputs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            GlowplugError::InvalidPayload("dispatch result missing outputs array".into())
        })?;
    let mut out = Vec::with_capacity(outputs.len());
    for (i, o) in outputs.iter().enumerate() {
        let s = o.as_str().ok_or_else(|| {
            GlowplugError::InvalidPayload(format!("outputs[{i}] is not a base64 string"))
        })?;
        let bytes = crate::base64_encode::decode(s.as_bytes())
            .map_err(|e| GlowplugError::OutputDecode(e.to_string()))?;
        out.push(bytes);
    }
    Ok(out)
}

/// Build the JSON-RPC request object (for tests and tooling).
#[must_use]
pub fn jsonrpc_request_object(method: &str, params: &serde_json::Value) -> serde_json::Value {
    crate::primal_bridge::jsonrpc_request(method, params.clone())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "protocol tests use expect on constructed dispatch params"
)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_params_shape_matches_glowplug_protocol() {
        let kernel = b".version 7.0\n";
        let inputs = vec![vec![1u8, 2, 3], vec![4u8, 5]];
        let opts = GlowplugDispatchOptions {
            dims: [64, 1, 1],
            workgroup: [128, 1, 1],
            kernel_name: "k".to_string(),
            shared_mem: 0,
        };
        let params = build_dispatch_params("0000:01:00.0", kernel, &inputs, &[4096u64], &opts);

        assert_eq!(params["bdf"], "0000:01:00.0");
        assert_eq!(params["dims"], serde_json::json!([64, 1, 1]));
        assert_eq!(params["workgroup"], serde_json::json!([128, 1, 1]));
        assert_eq!(params["kernel_name"], "k");
        assert_eq!(params["output_sizes"], serde_json::json!([4096]));
        assert!(params["shader"].as_str().is_some());
        let inputs_json = params["inputs"].as_array().expect("inputs array");
        assert_eq!(inputs_json.len(), 2);
    }

    #[test]
    fn jsonrpc_request_wraps_method_and_params() {
        let params = serde_json::json!({ "bdf": "0000:aa:00.0" });
        let req = jsonrpc_request_object("device.get", &params);
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["method"], "device.get");
        assert_eq!(req["id"], 1);
        assert_eq!(req["params"], params);
    }

    #[test]
    fn list_devices_wire_envelope() {
        let params = serde_json::json!({});
        let req = jsonrpc_request_object("device.list", &params);
        let line = serde_json::to_string(&req).expect("serialize");
        assert!(line.contains("device.list"));
    }

    #[test]
    fn oracle_capture_params() {
        let v = serde_json::json!({
            "bdf": "0000:01:00.0",
            "max_channels": 8u64,
        });
        let req = jsonrpc_request_object("device.oracle_capture", &v);
        assert_eq!(req["method"], "device.oracle_capture");
        assert_eq!(req["params"]["max_channels"], 8);
    }

    #[test]
    fn health_methods_use_empty_object_params() {
        for method in ["health.check", "health.liveness", "health.readiness"] {
            let req = jsonrpc_request_object(method, &serde_json::json!({}));
            assert_eq!(req["params"], serde_json::json!({}), "{method}");
        }
    }

    #[test]
    fn capture_training_params_shape() {
        let params = serde_json::json!({
            "bdf": "0000:03:00.0",
            "warm_driver": "nouveau",
        });
        let req = jsonrpc_request_object("capture.training", &params);
        assert_eq!(req["method"], "capture.training");
        assert_eq!(req["params"]["bdf"], "0000:03:00.0");
        assert_eq!(req["params"]["warm_driver"], "nouveau");
    }

    #[test]
    fn capture_training_params_auto_driver() {
        let params = serde_json::json!({"bdf": "0000:03:00.0"});
        let req = jsonrpc_request_object("capture.training", &params);
        assert!(req["params"].get("warm_driver").is_none());
    }

    #[test]
    fn sovereign_boot_params_shape() {
        let params = serde_json::json!({"bdf": "0000:03:00.0"});
        let req = jsonrpc_request_object("sovereign.boot", &params);
        assert_eq!(req["method"], "sovereign.boot");
        assert_eq!(req["params"]["bdf"], "0000:03:00.0");
    }

    #[test]
    fn capture_training_result_deserializes() {
        let json = serde_json::json!({
            "bdf": "0000:03:00.0",
            "warm_driver": "nouveau",
            "recipe_path": "/var/lib/toadstool/training/gv100.json",
            "total_writes": 342,
            "success": true,
            "summary": "captured 342 training writes for gv100",
            "steps": [
                {"name": "cold_snapshot", "status": "ok", "detail": "boot0=0x140000a1", "duration_ms": 12}
            ]
        });
        let result: CaptureTrainingResult =
            serde_json::from_value(json).expect("deserialize CaptureTrainingResult");
        assert!(result.success);
        assert_eq!(result.total_writes, 342);
        assert_eq!(result.steps.len(), 1);
    }

    #[test]
    fn warm_catch_params_shape() {
        let params = serde_json::json!({"bdf": "0000:02:00.0"});
        let req = jsonrpc_request_object("device.warm_catch", &params);
        assert_eq!(req["method"], "device.warm_catch");
        assert_eq!(req["params"]["bdf"], "0000:02:00.0");
    }

    #[test]
    fn warm_catch_result_deserializes() {
        let json = serde_json::json!({
            "bdf": "0000:02:00.0",
            "fecs_ready": true,
            "chip_id": 0x1d81,
            "summary": "GV100 warm FECS detected — compute-ready"
        });
        let result: WarmCatchResult =
            serde_json::from_value(json).expect("deserialize WarmCatchResult");
        assert!(result.fecs_ready);
        assert_eq!(result.chip_id, Some(0x1d81));
        assert_eq!(result.bdf, "0000:02:00.0");
    }

    #[test]
    fn warm_catch_result_deserializes_minimal() {
        let json = serde_json::json!({ "bdf": "0000:4b:00.0" });
        let result: WarmCatchResult =
            serde_json::from_value(json).expect("deserialize minimal WarmCatchResult");
        assert!(!result.fecs_ready);
        assert!(result.chip_id.is_none());
    }

    #[test]
    fn sovereign_boot_result_deserializes() {
        let json = serde_json::json!({
            "bdf": "0000:03:00.0",
            "initial_driver": "vfio-pci",
            "warm_cycle_performed": false,
            "final_driver": "vfio-pci",
            "sovereign_init": {"all_ok": true, "compute_ready": false},
            "success": false,
            "summary": "sovereign pipeline halted at: WAKE_MEMORY_CONTROLLER",
            "steps": []
        });
        let result: SovereignBootResult =
            serde_json::from_value(json).expect("deserialize SovereignBootResult");
        assert!(!result.success);
        assert!(result.sovereign_init.is_some());
    }
}
