// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal toadStool performance surface reporter.
//!
//! Sends silicon probe results to toadStool's `compute.performance_surface.report`
//! JSON-RPC method via Unix socket. Degrades gracefully when toadStool is not
//! running — measurements are constructed but only logged locally.
//!
//! Protocol: JSON-RPC 2.0 over Unix domain socket (newline-delimited).
//! Socket path: `toadstool_socket` — wateringHole IPC v3.1:
//! `$XDG_RUNTIME_DIR/biomeos/toadstool-<FAMILY_ID>.sock` with `/tmp` when `XDG_RUNTIME_DIR`
//! is unset, or override with `TOADSTOOL_SOCKET`.

use crate::primal_bridge::NucleusContext;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
const TOADSTOOL_JSONRPC_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
#[cfg(unix)]
const TOADSTOOL_JSONRPC_REQUEST_ID: i64 = 1;

/// A performance measurement matching toadStool's `PerformanceMeasurement` schema.
///
/// Fields are wire-compatible with `toadstool_core::silicon::PerformanceMeasurement`.
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMeasurement {
    /// Operation identifier (e.g. `"reduce_f64"`, `"df64_workgroup_reduce"`).
    pub operation: String,
    /// Silicon unit category (`"shader"` for GPU compute shader units).
    pub silicon_unit: String,
    /// Precision mode tested (e.g. `"df64_workgroup"`, `"f64_scalar_storage"`).
    pub precision_mode: String,
    /// Measured throughput in GFLOPS (0.0 for correctness-only probes).
    pub throughput_gflops: f64,
    /// Tolerance achieved by this probe (absolute error of probe result).
    pub tolerance_achieved: f64,
    /// GPU model identifier (adapter name from wgpu).
    pub gpu_model: String,
    /// Which spring/binary produced this measurement.
    pub measured_by: String,
    /// Measurement timestamp (epoch seconds).
    pub timestamp: u64,
}

/// Current epoch timestamp in seconds.
#[must_use]
pub fn epoch_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Resolve toadStool's JSON-RPC Unix socket path.
///
/// Discovery order:
/// 1. `TOADSTOOL_SOCKET` env override (CI/lab)
/// 2. NUCLEUS `by_domain("compute")` capability discovery
/// 3. `niche::socket_dirs()` filesystem scan (legacy fallback)
fn toadstool_socket() -> String {
    if let Ok(p) = std::env::var("TOADSTOOL_SOCKET") {
        return p;
    }
    let nucleus = crate::primal_bridge::NucleusContext::detect();
    if let Some(ep) = nucleus.by_domain("compute").filter(|ep| ep.alive) {
        return ep.socket.clone();
    }
    let family = crate::niche::family_id();
    let sock_name = format!("toadstool-{family}.sock");
    for dir in crate::niche::socket_dirs() {
        let candidate = dir.join(&sock_name);
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
    format!("{runtime_dir}/biomeos/{sock_name}")
}

/// Send a JSON-RPC 2.0 request over a Unix domain socket.
#[cfg(unix)]
fn send_jsonrpc(
    socket_path: &std::path::Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket_path).map_err(|e| format!("connect: {e}"))?;

    stream
        .set_read_timeout(Some(TOADSTOOL_JSONRPC_READ_TIMEOUT))
        .map_err(|e| format!("timeout: {e}"))?;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": TOADSTOOL_JSONRPC_REQUEST_ID
    });

    let mut request_bytes = serde_json::to_vec(&request).map_err(|e| format!("serialize: {e}"))?;
    request_bytes.push(b'\n');

    stream
        .write_all(&request_bytes)
        .map_err(|e| format!("write: {e}"))?;
    stream.flush().map_err(|e| format!("flush: {e}"))?;

    let mut response = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                response.extend_from_slice(&buf[..n]);
                if response.contains(&b'\n') {
                    break;
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) => return Err(format!("read: {e}")),
        }
    }

    serde_json::from_slice(&response).map_err(|e| format!("parse response: {e}"))
}

#[cfg(not(unix))]
fn send_jsonrpc(
    _socket_path: &std::path::Path,
    _method: &str,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    Err("toadStool JSON-RPC over Unix socket not available on this platform".into())
}

/// A GPU capability record returned by toadStool's compute.capability_query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapability {
    pub adapter_name: String,
    pub vendor: String,
    pub vram_mb: u64,
    pub f64_support: bool,
    pub precision_tier: String,
    pub available: bool,
}

/// NUCLEUS-aware compute capabilities from toadStool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    pub gpus: Vec<GpuCapability>,
    pub source: String,
}

/// Query toadStool for available compute capabilities (NUCLEUS silicon view).
///
/// When toadStool is available, this returns the cluster-wide GPU inventory.
/// Falls back to `None` when toadStool is absent — callers should use
/// local `GpuF64::enumerate_adapters()` as a fallback.
pub fn query_capabilities(nucleus: &NucleusContext) -> Option<ComputeCapabilities> {
    let compute_ep = nucleus.get_by_capability("compute")?;
    let source = compute_ep.name.clone();
    let params = serde_json::json!({
        "spring": "hotSpring",
        "require_f64": true,
    });

    let resp = nucleus
        .call_by_capability("compute", "compute.capability_query", params)
        .ok()?;

    let result = resp.get("result")?;
    let gpus_val = result.get("gpus")?;
    let gpus: Vec<GpuCapability> = serde_json::from_value(gpus_val.clone()).ok()?;

    println!(
        "  compute primal ({source}): {} GPU(s) via NUCLEUS capability query",
        gpus.len()
    );
    for g in &gpus {
        println!(
            "    {} ({}, {}MB, f64={}, {})",
            g.adapter_name, g.vendor, g.vram_mb, g.f64_support, g.precision_tier
        );
    }

    Some(ComputeCapabilities { gpus, source })
}

/// Register a validated shader with toadStool for cross-spring absorption.
///
/// Sends `compute.shader.register` JSON-RPC when toadStool is available.
pub fn register_shader(
    nucleus: &NucleusContext,
    name: &str,
    version: &str,
    precision_tier: &str,
    receipt_json: &serde_json::Value,
) -> Result<(), String> {
    let params = serde_json::json!({
        "shader_name": name,
        "shader_version": version,
        "precision_tier": precision_tier,
        "spring": "hotSpring",
        "guidestone_receipt": receipt_json,
    });

    let resp = nucleus.call_by_capability("compute", "compute.shader.register", params)?;

    let status = resp
        .get("result")
        .and_then(|r| r.get("status"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");

    println!("  toadStool: shader.register({name} v{version}) → {status}");
    Ok(())
}

/// Resolve compute dispatch socket: use `NucleusContext` capability table when provided,
/// otherwise bootstrap via `toadstool_socket` (name-based path).
fn resolve_compute_report_socket(nucleus: Option<&NucleusContext>) -> String {
    if let Some(ctx) = nucleus
        && let Some(ep) = ctx.get_by_capability("compute")
    {
        return ep.socket.clone();
    }
    toadstool_socket()
}

/// Report a batch of performance measurements to the compute primal.
///
/// Prefers `call_by_capability("compute", ...)` when NUCLEUS context is
/// available; falls back to direct socket IPC via `toadstool_socket()`.
pub fn report_to_toadstool_with_nucleus(
    nucleus: Option<&NucleusContext>,
    measurements: &[PerformanceMeasurement],
) {
    let use_capability =
        nucleus.is_some_and(|ctx| ctx.by_domain("compute").is_some_and(|ep| ep.alive));

    if use_capability {
        println!(
            "  Reporting {} measurement(s) to toadStool via NUCLEUS capability",
            measurements.len(),
        );
    } else {
        let socket = resolve_compute_report_socket(nucleus);
        println!(
            "  Reporting {} measurement(s) to toadStool at {socket} (direct IPC)",
            measurements.len(),
        );
    }

    for m in measurements {
        let params = match serde_json::to_value(m) {
            Ok(v) => v,
            Err(e) => {
                println!("    serialize error: {e}");
                continue;
            }
        };

        let result = if let Some(ctx) = nucleus.filter(|_| use_capability) {
            ctx.call_by_capability("compute", "compute.performance_surface.report", params)
                .map_err(|e| format!("{e}"))
        } else {
            let socket_path = PathBuf::from(resolve_compute_report_socket(nucleus));
            send_jsonrpc(&socket_path, "compute.performance_surface.report", &params)
        };

        match result {
            Ok(resp) => {
                let status = resp
                    .get("result")
                    .and_then(|r| r.get("status"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown");
                println!(
                    "    {} / {} / {} → {}",
                    m.operation, m.gpu_model, m.precision_mode, status
                );
            }
            Err(e) => {
                println!("    {} / {} → report failed: {e}", m.operation, m.gpu_model);
            }
        }
    }
}

/// Same as [`report_to_toadstool_with_nucleus`] with no nucleus context (bootstrap socket only).
pub fn report_to_toadstool(measurements: &[PerformanceMeasurement]) {
    report_to_toadstool_with_nucleus(None, measurements);
}
