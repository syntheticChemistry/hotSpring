// SPDX-License-Identifier: AGPL-3.0-only

//! Minimal toadStool performance surface reporter.
//!
//! Sends silicon probe results to toadStool's `compute.performance_surface.report`
//! JSON-RPC method via Unix socket. Degrades gracefully when toadStool is not
//! running — measurements are constructed but only logged locally.
//!
//! Protocol: JSON-RPC 2.0 over Unix domain socket (newline-delimited).
//! Socket discovery: `TOADSTOOL_SOCKET` → `XDG_RUNTIME_DIR` → `/tmp`.

use serde::Serialize;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Discover toadStool's JSON-RPC Unix socket path.
///
/// Checks `TOADSTOOL_SOCKET` (with `.jsonrpc.sock` extension),
/// `XDG_RUNTIME_DIR/biomeos/toadstool.jsonrpc.sock` (toadStool's default),
/// `XDG_RUNTIME_DIR/toadstool.jsonrpc.sock` (flat layout), and `/tmp/toadstool.jsonrpc.sock`.
fn discover_socket() -> Option<PathBuf> {
    for var in &["TOADSTOOL_SOCKET", "PRIMAL_SOCKET"] {
        if let Ok(p) = std::env::var(var) {
            let sock = PathBuf::from(&p).with_extension("jsonrpc.sock");
            if sock.exists() {
                return Some(sock);
            }
            let direct = PathBuf::from(&p);
            if direct.exists() {
                return Some(direct);
            }
        }
    }

    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let xdg = PathBuf::from(xdg);
        // toadStool serves under biomeos/ subdirectory
        let biomeos = xdg.join("biomeos/toadstool.jsonrpc.sock");
        if biomeos.exists() {
            return Some(biomeos);
        }
        let flat = xdg.join("toadstool.jsonrpc.sock");
        if flat.exists() {
            return Some(flat);
        }
    }

    let tmp = PathBuf::from("/tmp/toadstool.jsonrpc.sock");
    if tmp.exists() {
        return Some(tmp);
    }

    None
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

    let mut stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect: {e}"))?;

    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .map_err(|e| format!("timeout: {e}"))?;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
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

/// Report a batch of performance measurements to toadStool.
///
/// Connects to the toadStool JSON-RPC socket and sends each measurement via
/// `compute.performance_surface.report`. If toadStool is not running, logs
/// a message and returns silently.
pub fn report_to_toadstool(measurements: &[PerformanceMeasurement]) {
    let socket = match discover_socket() {
        Some(s) => s,
        None => {
            println!("  toadStool socket not found — measurements logged locally only");
            for m in measurements {
                println!(
                    "    {}: {} / {} / {} → tol={:.2e}",
                    m.gpu_model, m.operation, m.silicon_unit, m.precision_mode, m.tolerance_achieved
                );
            }
            return;
        }
    };

    println!(
        "  Reporting {} measurement(s) to toadStool at {}",
        measurements.len(),
        socket.display()
    );

    for m in measurements {
        let params = match serde_json::to_value(m) {
            Ok(v) => v,
            Err(e) => {
                println!("    serialize error: {e}");
                continue;
            }
        };
        match send_jsonrpc(&socket, "compute.performance_surface.report", &params) {
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
                println!(
                    "    {} / {} → report failed: {e}",
                    m.operation, m.gpu_model
                );
            }
        }
    }
}
