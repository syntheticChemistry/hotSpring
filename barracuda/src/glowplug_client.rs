// SPDX-License-Identifier: AGPL-3.0-or-later

//! JSON-RPC client for [coral-glowplug](https://github.com/biomegate/ecoPrimals) over the
//! NUCLEUS Unix socket (`NucleusContext::coralreef` or an explicit path).
//!
//! All calls use [`crate::primal_bridge::send_jsonrpc`] (JSON-RPC 2.0, newline-framed).

use base64::Engine;
use serde::Deserialize;
use std::fmt;
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

/// Options for [`GlowplugClient::dispatch`] (grid, workgroup, entry symbol).
#[derive(Debug, Clone)]
pub struct GlowplugDispatchOptions {
    /// `[grid_x, grid_y, grid_z]` dispatch dimensions.
    pub dims: [u32; 3],
    /// `[threads_x, threads_y, threads_z]` per-block (workgroup) size.
    pub workgroup: [u32; 3],
    /// CUDA kernel / entry point name in the shader module.
    pub kernel_name: String,
    /// Dynamic shared memory bytes (default 0).
    pub shared_mem: u32,
}

impl Default for GlowplugDispatchOptions {
    fn default() -> Self {
        Self {
            dims: [1, 1, 1],
            workgroup: [256, 1, 1],
            kernel_name: "main_kernel".to_string(),
            shared_mem: 0,
        }
    }
}

/// One row from `device.list` — BDF, vendor id, display name, coarse health.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlowplugDeviceSummary {
    pub bdf: String,
    /// PCI vendor id as `0xABCD` (same information as JSON `vendor_id`).
    pub vendor: String,
    pub name: Option<String>,
    /// Driver / backend personality string (e.g. `nvidia`, `cuda`).
    pub personality: String,
    /// True when the device is display-attached and swap-immune.
    pub protected: bool,
    pub health: GlowplugDeviceHealthSummary,
}

/// Health fields exposed for quick listing (mirrors glowplug `DeviceInfo` health slice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlowplugDeviceHealthSummary {
    pub vram_alive: bool,
    pub domains_faulted: usize,
}

/// Full `device.get` payload (structured like coral-glowplug `DeviceInfo`).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GlowplugDeviceDetail {
    pub bdf: String,
    pub name: Option<String>,
    pub chip: String,
    pub vendor_id: u16,
    pub device_id: u16,
    pub personality: String,
    pub role: Option<String>,
    pub power: String,
    pub vram_alive: bool,
    pub domains_alive: usize,
    pub domains_faulted: usize,
    pub has_vfio_fd: bool,
    pub pci_link_width: Option<u8>,
    #[serde(default)]
    pub protected: bool,
}

/// Daemon response for `health.check` / `health.liveness` (same JSON shape in glowplug today).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GlowplugDaemonHealth {
    pub alive: bool,
    pub name: String,
    pub device_count: usize,
    pub healthy_count: usize,
}

/// Errors from glowplug RPC or payload handling.
#[derive(Debug)]
pub enum GlowplugError {
    /// No `coralreef` / `coral-glowplug` entry in [`NucleusContext`].
    NoCoralreefEndpoint,
    /// Socket path exists but the primal failed liveness at discovery time.
    EndpointNotAlive,
    /// Low-level transport / JSON parse (`send_jsonrpc` message).
    Transport(String),
    /// JSON-RPC `error` object.
    JsonRpc { code: i64, message: String },
    /// Successful envelope but missing `result`.
    MissingResult,
    /// Base64 decode of a dispatch output buffer.
    OutputDecode(String),
    /// `serde_json` shape mismatch.
    InvalidPayload(String),
}

impl fmt::Display for GlowplugError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlowplugError::NoCoralreefEndpoint => {
                write!(f, "no coralreef / coral-glowplug primal in NucleusContext")
            }
            GlowplugError::EndpointNotAlive => write!(f, "coralreef primal socket is not alive"),
            GlowplugError::Transport(s) => write!(f, "transport: {s}"),
            GlowplugError::JsonRpc { code, message } => write!(f, "json-rpc {code}: {message}"),
            GlowplugError::MissingResult => write!(f, "json-rpc response missing result"),
            GlowplugError::OutputDecode(s) => write!(f, "output base64 decode: {s}"),
            GlowplugError::InvalidPayload(s) => write!(f, "invalid payload: {s}"),
        }
    }
}

impl std::error::Error for GlowplugError {}

impl GlowplugClient {
    /// Build a client from a discovered coralReef / coral-glowplug endpoint.
    pub fn from_nucleus(nucleus: &NucleusContext) -> Result<Self, GlowplugError> {
        let ep = nucleus
            .by_domain("shader")
            .ok_or(GlowplugError::NoCoralreefEndpoint)?;
        if !ep.alive {
            return Err(GlowplugError::EndpointNotAlive);
        }
        Ok(Self {
            socket: PathBuf::from(&ep.socket),
        })
    }

    /// Build a client for an explicit Unix socket path (no liveness probe here).
    pub fn from_socket(path: &Path) -> Self {
        Self {
            socket: path.to_path_buf(),
        }
    }

    /// Underlying socket path.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket
    }

    /// Raw JSON-RPC call (same framing as [`send_jsonrpc`]).
    pub fn call(
        &self,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, GlowplugError> {
        rpc_result(&self.socket, method, params)
    }

    /// `health.check` — full daemon health snapshot.
    pub fn health(&self) -> Result<GlowplugDaemonHealth, GlowplugError> {
        let v = self.call("health.check", &serde_json::json!({}))?;
        serde_json::from_value(v).map_err(|e| GlowplugError::InvalidPayload(e.to_string()))
    }

    /// `health.liveness` — lightweight probe (same JSON shape as `health.check` in glowplug).
    pub fn health_liveness(&self) -> Result<GlowplugDaemonHealth, GlowplugError> {
        let v = self.call("health.liveness", &serde_json::json!({}))?;
        serde_json::from_value(v).map_err(|e| GlowplugError::InvalidPayload(e.to_string()))
    }

    /// `health.readiness` — reserved in the method table; may be unsupported on older daemons.
    pub fn health_readiness(&self) -> Result<serde_json::Value, GlowplugError> {
        self.call("health.readiness", &serde_json::json!({}))
    }

    /// `device.list` — managed GPUs/devices.
    pub fn list_devices(&self) -> Result<Vec<GlowplugDeviceSummary>, GlowplugError> {
        let v = self.call("device.list", &serde_json::json!({}))?;
        let rows: Vec<GlowplugListRow> = serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.list: {e}")))?;
        Ok(rows.into_iter().map(GlowplugDeviceSummary::from).collect())
    }

    /// `device.get` — full record for one BDF.
    pub fn device_status(&self, bdf: &str) -> Result<GlowplugDeviceDetail, GlowplugError> {
        let v = self.call(
            "device.get",
            &serde_json::json!({
                "bdf": bdf,
            }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.get: {e}")))
    }

    /// `device.swap` — hot-swap driver personality.
    pub fn device_swap(
        &self,
        bdf: &str,
        target: &str,
        trace: bool,
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call(
            "device.swap",
            &serde_json::json!({
                "bdf": bdf,
                "target": target,
                "trace": trace,
            }),
        )
    }

    /// `device.dispatch` — run `kernel` bytes (e.g. PTX) with `buffers` as inputs.
    ///
    /// `output_sizes` lists byte lengths for each output buffer the kernel writes (protocol requirement).
    /// Grid and workgroup use [`GlowplugDispatchOptions`]; see [`GlowplugDispatchOptions::default`].
    pub fn dispatch(
        &self,
        bdf: &str,
        kernel: &[u8],
        buffers: &[Vec<u8>],
        output_sizes: &[u64],
    ) -> Result<Vec<Vec<u8>>, GlowplugError> {
        self.dispatch_with_options(
            bdf,
            kernel,
            buffers,
            output_sizes,
            &GlowplugDispatchOptions::default(),
        )
    }

    /// `device.dispatch` with explicit grid, workgroup, and kernel name.
    pub fn dispatch_with_options(
        &self,
        bdf: &str,
        kernel: &[u8],
        buffers: &[Vec<u8>],
        output_sizes: &[u64],
        options: &GlowplugDispatchOptions,
    ) -> Result<Vec<Vec<u8>>, GlowplugError> {
        let params = build_dispatch_params(bdf, kernel, buffers, output_sizes, options);
        let v = self.call("device.dispatch", &params)?;
        decode_dispatch_outputs(&v)
    }

    /// `device.oracle_capture` — MMU page table capture (privileged daemon path).
    pub fn oracle_capture(
        &self,
        bdf: &str,
        max_channels: u64,
    ) -> Result<serde_json::Value, GlowplugError> {
        self.call(
            "device.oracle_capture",
            &serde_json::json!({
                "bdf": bdf,
                "max_channels": max_channels,
            }),
        )
    }

    // ── Register / BAR0 access ──────────────────────────────────────

    /// `device.register_dump` — full named-register dump for a device.
    pub fn register_dump(&self, bdf: &str) -> Result<GlowplugRegisterDumpResult, GlowplugError> {
        let v = self.call("device.register_dump", &serde_json::json!({ "bdf": bdf }))?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("register_dump: {e}")))
    }

    /// `device.register_snapshot` — snapshot of key registers (lighter than full dump).
    pub fn register_snapshot(
        &self,
        bdf: &str,
    ) -> Result<GlowplugRegisterDumpResult, GlowplugError> {
        let v = self.call(
            "device.register_snapshot",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("register_snapshot: {e}")))
    }

    /// `device.read_bar0_range` — read a raw BAR0 byte range.
    pub fn read_bar0_range(
        &self,
        bdf: &str,
        start: u32,
        length: u32,
    ) -> Result<Bar0RangeResult, GlowplugError> {
        let v = self.call(
            "device.read_bar0_range",
            &serde_json::json!({ "bdf": bdf, "start": start, "length": length }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("read_bar0_range: {e}")))
    }

    // ── Device lifecycle ────────────────────────────────────────────

    /// `device.experiment_start` — mark a device as under active experiment.
    pub fn experiment_start(&self, bdf: &str) -> Result<ExperimentLifecycleResult, GlowplugError> {
        let v = self.call(
            "device.experiment_start",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("experiment_start: {e}")))
    }

    /// `device.experiment_end` — release a device from active experiment.
    pub fn experiment_end(&self, bdf: &str) -> Result<ExperimentLifecycleResult, GlowplugError> {
        let v = self.call("device.experiment_end", &serde_json::json!({ "bdf": bdf }))?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("experiment_end: {e}")))
    }

    /// `device.reset` — request a device reset (FLR or engine reset depending on chip).
    pub fn device_reset(&self, bdf: &str) -> Result<DeviceLifecycleResult, GlowplugError> {
        let v = self.call("device.reset", &serde_json::json!({ "bdf": bdf }))?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.reset: {e}")))
    }

    /// `device.resurrect` — resurrect a dead/faulted device via warm cycle + fd restore.
    pub fn device_resurrect(&self, bdf: &str) -> Result<DeviceLifecycleResult, GlowplugError> {
        let v = self.call("device.resurrect", &serde_json::json!({ "bdf": bdf }))?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("device.resurrect: {e}")))
    }

    /// `device.health` — detailed per-device health (domain breakdown, fault history).
    pub fn device_health(&self, bdf: &str) -> Result<serde_json::Value, GlowplugError> {
        self.call("device.health", &serde_json::json!({ "bdf": bdf }))
    }

    /// `daemon.status` — glowplug daemon status (mode, uptime, fleet info).
    pub fn daemon_status(&self) -> Result<serde_json::Value, GlowplugError> {
        self.call("daemon.status", &serde_json::json!({}))
    }

    // ── Sovereign boot + training capture ────────────────────────────

    /// `capture.training` — capture a training recipe by observing an external
    /// driver's memory initialization on a cold GPU.
    ///
    /// Orchestrates: cold BAR0 snapshot → swap to warm driver (with mmiotrace)
    /// → settle → warm BAR0 snapshot → diff → save recipe JSON. The recipe is
    /// stored at `/var/lib/coralreef/training/{chip}.json` for sovereign replay.
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

    /// `sovereign.boot` — full orchestrated sovereign boot: detect driver →
    /// warm if needed → swap to vfio → run SovereignInit pipeline.
    ///
    /// Routes through glowplug for full lifecycle coordination.
    pub fn sovereign_boot(
        &self,
        bdf: &str,
    ) -> Result<SovereignBootResult, GlowplugError> {
        let v = self.call("sovereign.boot", &serde_json::json!({"bdf": bdf}))?;
        serde_json::from_value(v)
            .map_err(|e| GlowplugError::InvalidPayload(format!("sovereign.boot: {e}")))
    }
}

/// Result of `capture.training` — training recipe capture flow.
#[derive(Debug, Clone, Deserialize)]
pub struct CaptureTrainingResult {
    pub bdf: String,
    pub warm_driver: String,
    pub recipe_path: Option<String>,
    pub total_writes: usize,
    pub success: bool,
    pub summary: String,
    pub steps: Vec<BootStepResult>,
}

/// Result of `sovereign.boot` — full orchestrated sovereign boot.
#[derive(Debug, Clone, Deserialize)]
pub struct SovereignBootResult {
    pub bdf: String,
    pub initial_driver: Option<String>,
    pub warm_cycle_performed: bool,
    pub final_driver: Option<String>,
    pub sovereign_init: Option<serde_json::Value>,
    pub success: bool,
    pub summary: String,
    pub steps: Vec<BootStepResult>,
}

/// A single step in an orchestrated boot or capture flow.
#[derive(Debug, Clone, Deserialize)]
pub struct BootStepResult {
    pub name: String,
    pub status: String,
    pub detail: Option<String>,
    pub duration_ms: u64,
}

#[derive(Debug, Deserialize)]
struct GlowplugListRow {
    bdf: String,
    name: Option<String>,
    vendor_id: u16,
    personality: String,
    #[serde(default)]
    protected: bool,
    vram_alive: bool,
    domains_faulted: usize,
}

impl From<GlowplugListRow> for GlowplugDeviceSummary {
    fn from(row: GlowplugListRow) -> Self {
        GlowplugDeviceSummary {
            bdf: row.bdf,
            vendor: format!("0x{:04X}", row.vendor_id),
            name: row.name,
            personality: row.personality,
            protected: row.protected,
            health: GlowplugDeviceHealthSummary {
                vram_alive: row.vram_alive,
                domains_faulted: row.domains_faulted,
            },
        }
    }
}

fn rpc_result(
    socket: &Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, GlowplugError> {
    let v = send_jsonrpc(socket, method, params).map_err(GlowplugError::Transport)?;
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
    let b64 = base64::engine::general_purpose::STANDARD;
    let shader_b64 = b64.encode(kernel);
    let inputs: Vec<String> = buffers.iter().map(|b| b64.encode(b)).collect();
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
    let b64 = base64::engine::general_purpose::STANDARD;
    let mut out = Vec::with_capacity(outputs.len());
    for (i, o) in outputs.iter().enumerate() {
        let s = o.as_str().ok_or_else(|| {
            GlowplugError::InvalidPayload(format!("outputs[{i}] is not a base64 string"))
        })?;
        let bytes = b64
            .decode(s.as_bytes())
            .map_err(|e| GlowplugError::OutputDecode(e.to_string()))?;
        out.push(bytes);
    }
    Ok(out)
}

/// Build the JSON-RPC request object (for tests and tooling). Matches [`send_jsonrpc`] wire format.
#[must_use]
pub fn jsonrpc_request_object(method: &str, params: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
            "recipe_path": "/var/lib/coralreef/training/gv100.json",
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
