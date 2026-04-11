// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-ember JSON-RPC client: MMIO, falcon, SEC2, PRAMIN, DMA, and flood testing.
//!
//! [`EmberClient`] wraps a single Unix socket to one coral-ember instance.
//! [`FleetEmberHub`] manages a stable client per socket. [`flood_test`]
//! stress-tests an ember with concurrent RPCs.

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ember_types::{
    CircuitBreakerStatus, DmaCleanupResult, DmaPrepareResult, FalconPollResult, FalconStartResult,
    FalconUploadResult, MmioBatchOp, MmioBatchResult, MmioReadResult, MmioWriteResult,
    PraminReadResult, PraminWriteResult, Sec2PrepareResult,
};
use crate::primal_bridge::send_jsonrpc;

fn jsonrpc_ok_result(resp: &serde_json::Value) -> Result<serde_json::Value, String> {
    if let Some(err) = resp.get("error") {
        return Err(format!("JSON-RPC error: {err}"));
    }
    resp.get("result")
        .cloned()
        .ok_or_else(|| "JSON-RPC response missing result".to_string())
}

/// Client for one ember instance (single Unix socket).
#[derive(Debug, Clone)]
pub struct EmberClient {
    socket_path: PathBuf,
}

impl EmberClient {
    /// Connect to an ember instance (stores the socket path; connection is per RPC).
    #[must_use]
    pub fn connect(socket_path: impl AsRef<Path>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
        }
    }

    /// Build a JSON-RPC 2.0 object (without sending) — used by tests and tooling.
    #[must_use]
    pub fn jsonrpc_request(id: u64, method: &str, params: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": id,
        })
    }

    /// Serialized `ember.warm_cycle` request for `bdf`.
    #[must_use]
    pub fn warm_cycle_request(bdf: &str) -> serde_json::Value {
        Self::jsonrpc_request(1, "ember.warm_cycle", &serde_json::json!({ "bdf": bdf }))
    }

    /// Serialized `ember.adopt_device` request for `bdf`.
    #[must_use]
    pub fn adopt_device_request(bdf: &str) -> serde_json::Value {
        Self::jsonrpc_request(1, "ember.adopt_device", &serde_json::json!({ "bdf": bdf }))
    }

    /// Socket path for this client.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// `ember.status` — aggregate status (devices, uptime, per-device hints).
    pub fn status(&self) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(&self.socket_path, "ember.status", &serde_json::json!({}))?;
        jsonrpc_ok_result(&v)
    }

    /// Ember health check — same payload as [`Self::status`] (`ember.status`).
    pub fn health(&self) -> Result<serde_json::Value, String> {
        self.status()
    }

    /// `ember.list` — BDFs currently held.
    pub fn list_devices(&self) -> Result<Vec<String>, String> {
        let v = send_jsonrpc(&self.socket_path, "ember.list", &serde_json::json!({}))?;
        let result = jsonrpc_ok_result(&v)?;
        let arr = result
            .get("devices")
            .and_then(|x| x.as_array())
            .ok_or_else(|| "ember.list: missing devices array".to_string())?;
        let mut out = Vec::new();
        for x in arr {
            let s = x
                .as_str()
                .ok_or_else(|| "ember.list: non-string device entry".to_string())?;
            out.push(s.to_string());
        }
        Ok(out)
    }

    /// `ember.adopt_device` — JSON-RPC only (current coral-ember opens VFIO server-side).
    pub fn adopt_device(&self, bdf: &str) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.adopt_device",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        jsonrpc_ok_result(&v)
    }

    /// `ember.warm_cycle` — driver warm cycle for `bdf`.
    pub fn warm_cycle(&self, bdf: &str) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.warm_cycle",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        jsonrpc_ok_result(&v)
    }

    /// `ember.device.health` — per-device health.
    pub fn device_health(&self, bdf: &str) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.device.health",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        jsonrpc_ok_result(&v)
    }

    /// `ember.device.recover` — attempt MMIO recovery for a faulted device.
    pub fn device_recover(&self, bdf: &str) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.device.recover",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        jsonrpc_ok_result(&v)
    }

    // ── MMIO (fork-isolated, circuit-breaker-protected) ─────────────

    /// `ember.mmio.read` — read a single 32-bit BAR0 register.
    pub fn mmio_read(&self, bdf: &str, offset: u32) -> Result<MmioReadResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.mmio.read",
            &serde_json::json!({ "bdf": bdf, "offset": offset }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("mmio.read parse: {e}"))
    }

    /// `ember.mmio.write` — write a single 32-bit BAR0 register.
    pub fn mmio_write(&self, bdf: &str, offset: u32, value: u32) -> Result<MmioWriteResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.mmio.write",
            &serde_json::json!({ "bdf": bdf, "offset": offset, "value": value }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("mmio.write parse: {e}"))
    }

    /// `ember.mmio.batch` — execute a sequence of read/write ops in one fork.
    pub fn mmio_batch(&self, bdf: &str, ops: &[MmioBatchOp]) -> Result<MmioBatchResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.mmio.batch",
            &serde_json::json!({ "bdf": bdf, "ops": ops }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("mmio.batch parse: {e}"))
    }

    /// `ember.mmio.circuit_breaker` — query or reset the MMIO circuit breaker.
    pub fn mmio_circuit_breaker(
        &self,
        bdf: &str,
        action: Option<&str>,
    ) -> Result<CircuitBreakerStatus, String> {
        let mut params = serde_json::json!({ "bdf": bdf });
        if let Some(a) = action {
            params["action"] = serde_json::json!(a);
        }
        let v = send_jsonrpc(&self.socket_path, "ember.mmio.circuit_breaker", &params)?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("circuit_breaker parse: {e}"))
    }

    // ── Falcon (IMEM/DMEM upload, CPU start, poll) ──────────────────

    /// `ember.falcon.upload_imem` — upload firmware code to falcon IMEM via PIO.
    pub fn falcon_upload_imem(
        &self,
        bdf: &str,
        base: u32,
        imem_addr: u32,
        code: &[u8],
        start_tag: u32,
        secure: bool,
    ) -> Result<FalconUploadResult, String> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.falcon.upload_imem",
            &serde_json::json!({
                "bdf": bdf,
                "base": base,
                "imem_addr": imem_addr,
                "code_b64": b64.encode(code),
                "start_tag": start_tag,
                "secure": secure,
            }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("falcon.upload_imem parse: {e}"))
    }

    /// `ember.falcon.upload_dmem` — upload data to falcon DMEM.
    pub fn falcon_upload_dmem(
        &self,
        bdf: &str,
        base: u32,
        dmem_addr: u32,
        data: &[u8],
    ) -> Result<FalconUploadResult, String> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.falcon.upload_dmem",
            &serde_json::json!({
                "bdf": bdf,
                "base": base,
                "dmem_addr": dmem_addr,
                "data_b64": b64.encode(data),
            }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("falcon.upload_dmem parse: {e}"))
    }

    /// `ember.falcon.start_cpu` — trigger STARTCPU and read back PC/EXCI/CPUCTL.
    pub fn falcon_start_cpu(&self, bdf: &str, base: u32) -> Result<FalconStartResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.falcon.start_cpu",
            &serde_json::json!({ "bdf": bdf, "base": base }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("falcon.start_cpu parse: {e}"))
    }

    /// `ember.falcon.poll` — poll falcon state until halt/mailbox/timeout.
    pub fn falcon_poll(
        &self,
        bdf: &str,
        base: u32,
        timeout_ms: u32,
        mailbox_sentinel: u32,
    ) -> Result<FalconPollResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.falcon.poll",
            &serde_json::json!({
                "bdf": bdf,
                "base": base,
                "timeout_ms": timeout_ms,
                "mailbox_sentinel": mailbox_sentinel,
            }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("falcon.poll parse: {e}"))
    }

    // ── SEC2 ────────────────────────────────────────────────────────

    /// `ember.sec2.prepare_physical` — PMC reset + instance bind + PHYS_VID path.
    pub fn sec2_prepare_physical(&self, bdf: &str) -> Result<Sec2PrepareResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.sec2.prepare_physical",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("sec2.prepare parse: {e}"))
    }

    // ── PRAMIN (bulk VRAM staging) ──────────────────────────────────

    /// `ember.pramin.read` — read VRAM at `vram_addr` for `length` bytes.
    pub fn pramin_read(
        &self,
        bdf: &str,
        vram_addr: u64,
        length: u32,
    ) -> Result<Vec<u8>, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.pramin.read",
            &serde_json::json!({
                "bdf": bdf,
                "vram_addr": vram_addr,
                "length": length,
            }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        let parsed: PraminReadResult =
            serde_json::from_value(result).map_err(|e| format!("pramin.read parse: {e}"))?;
        let b64 = base64::engine::general_purpose::STANDARD;
        b64.decode(parsed.data_b64.as_bytes())
            .map_err(|e| format!("pramin.read base64 decode: {e}"))
    }

    /// `ember.pramin.write` — write `data` to VRAM at `vram_addr`.
    pub fn pramin_write(
        &self,
        bdf: &str,
        vram_addr: u64,
        data: &[u8],
    ) -> Result<PraminWriteResult, String> {
        let b64 = base64::engine::general_purpose::STANDARD;
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.pramin.write",
            &serde_json::json!({
                "bdf": bdf,
                "vram_addr": vram_addr,
                "data_b64": b64.encode(data),
            }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("pramin.write parse: {e}"))
    }

    // ── DMA lifecycle ───────────────────────────────────────────────

    /// `ember.prepare_dma` — quiesce + AER mask + optional bus master enable.
    pub fn prepare_dma(&self, bdf: &str, bus_master: bool) -> Result<DmaPrepareResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.prepare_dma",
            &serde_json::json!({ "bdf": bdf, "bus_master": bus_master }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("prepare_dma parse: {e}"))
    }

    /// `ember.cleanup_dma` — decontaminate + restore AER + disable bus master.
    pub fn cleanup_dma(&self, bdf: &str) -> Result<DmaCleanupResult, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.cleanup_dma",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        let result = jsonrpc_ok_result(&v)?;
        serde_json::from_value(result).map_err(|e| format!("cleanup_dma parse: {e}"))
    }

    // ── FECS (Kepler/Volta PGRAPH engine) ─────────────────────────

    /// `ember.fecs.state` — read FECS falcon status registers.
    pub fn fecs_state(&self, bdf: &str) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.fecs.state",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        jsonrpc_ok_result(&v)
    }

    // ── Journal / Policy ────────────────────────────────────────────

    /// `ember.journal.query` — query the per-device event journal.
    pub fn journal_query(&self, bdf: &str) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.journal.query",
            &serde_json::json!({ "bdf": bdf }),
        )?;
        jsonrpc_ok_result(&v)
    }

    /// `ember.policy.list` — list active MMIO safety policies.
    pub fn policy_list(&self) -> Result<serde_json::Value, String> {
        let v = send_jsonrpc(
            &self.socket_path,
            "ember.policy.list",
            &serde_json::json!({}),
        )?;
        jsonrpc_ok_result(&v)
    }

    /// Receive JSON result for `ember.adopt_device`, then any `SCM_RIGHTS` fds on the same stream.
    #[cfg(not(all(unix, feature = "low-level")))]
    pub fn adopt_device_recv_scm_rights(
        &self,
        _bdf: &str,
    ) -> Result<(serde_json::Value, Vec<std::os::fd::OwnedFd>), String> {
        Err(
            "SCM_RIGHTS fd reception requires recvmsg with CMSG_SPACE ancillary buffers; build \
             hotspring-barracuda with the `low-level` feature (rustix net APIs)."
                .to_string(),
        )
    }

    /// Receive JSON + ancillary fds after `ember.adopt_device` (same connection).
    #[cfg(all(unix, feature = "low-level"))]
    pub fn adopt_device_recv_scm_rights(
        &self,
        bdf: &str,
    ) -> Result<(serde_json::Value, Vec<std::os::fd::OwnedFd>), String> {
        adopt_device_recv_scm_rights_impl(&self.socket_path, bdf)
    }
}

#[cfg(all(unix, feature = "low-level"))]
fn adopt_device_recv_scm_rights_impl(
    socket_path: &Path,
    bdf: &str,
) -> Result<(serde_json::Value, Vec<std::os::fd::OwnedFd>), String> {
    use std::io::{Read, Write};
    use std::mem::MaybeUninit;
    use std::os::fd::AsFd;
    use std::os::unix::net::UnixStream;

    use rustix::io::IoSliceMut;
    use rustix::net::{
        RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, recvmsg,
    };

    let mut stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .map_err(|e| format!("timeout: {e}"))?;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "ember.adopt_device",
        "params": { "bdf": bdf },
        "id": 1,
    });
    let mut request_bytes =
        serde_json::to_vec(&request).map_err(|e| format!("serialize: {e}"))?;
    request_bytes.push(b'\n');
    stream
        .write_all(&request_bytes)
        .map_err(|e| format!("write: {e}"))?;
    stream.flush().map_err(|e| format!("flush: {e}"))?;

    let mut line_buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = stream
            .read(&mut chunk)
            .map_err(|e| format!("read JSON line: {e}"))?;
        if n == 0 {
            break;
        }
        line_buf.extend_from_slice(&chunk[..n]);
        if line_buf.contains(&b'\n') {
            break;
        }
    }
    let line_end = line_buf
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| "adopt_device: no newline in response".to_string())?;
    let json_line = &line_buf[..line_end];
    let resp: serde_json::Value =
        serde_json::from_slice(json_line).map_err(|e| format!("parse JSON-RPC: {e}"))?;
    let result = jsonrpc_ok_result(&resp)?;

    let mut buf = [0u8; 256];
    let mut iov = [IoSliceMut::new(&mut buf)];
    let fd_space = 8 * std::mem::size_of::<std::os::fd::OwnedFd>() + 64;
    let mut ancillary_space = vec![MaybeUninit::uninit(); fd_space];
    let mut control = RecvAncillaryBuffer::new(&mut ancillary_space);
    let msg_result = recvmsg(&stream.as_fd(), &mut iov, &mut control, RecvFlags::empty())
        .map_err(|e| format!("recvmsg (SCM_RIGHTS): {e}"))?;

    let mut fds = Vec::new();
    for msg in control.drain() {
        if let RecvAncillaryMessage::ScmRights(rights) = msg {
            fds.extend(rights);
        }
    }

    let _ = msg_result.bytes;
    Ok((result, fds))
}

/// Holds one [`EmberClient`] per distinct Unix socket (fleet isolation).
#[derive(Debug, Default)]
pub struct FleetEmberHub {
    clients: HashMap<PathBuf, EmberClient>,
}

impl FleetEmberHub {
    /// Return a stable client for `socket_path`, creating it on first use.
    #[must_use]
    pub fn client_for_socket(&mut self, socket_path: impl AsRef<Path>) -> &EmberClient {
        let p = socket_path.as_ref().to_path_buf();
        self.clients
            .entry(p.clone())
            .or_insert_with(|| EmberClient::connect(p))
    }

    /// Convenience: client for a fleet route's socket.
    #[must_use]
    pub fn client_for_route(&mut self, route: &crate::fleet_client::FleetDeviceRoute) -> &EmberClient {
        self.client_for_socket(&route.socket_path)
    }

    /// Number of distinct sockets with materialized clients.
    #[must_use]
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    /// `true` when no clients have been cached yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

/// Configuration for [`flood_test`].
#[derive(Debug, Clone)]
pub struct FloodTestConfig {
    /// Target ember socket to flood.
    pub target_socket: PathBuf,
    /// Number of concurrent threads hammering the target.
    pub concurrency: usize,
    /// Total number of requests to send (spread across threads).
    pub total_requests: usize,
    /// Per-request timeout.
    pub request_timeout: std::time::Duration,
}

impl Default for FloodTestConfig {
    fn default() -> Self {
        Self {
            target_socket: PathBuf::new(),
            concurrency: 50,
            total_requests: 500,
            request_timeout: std::time::Duration::from_secs(5),
        }
    }
}

/// Result of a single RPC attempt during a flood test.
#[derive(Debug, Clone)]
pub struct FloodRequestResult {
    /// Monotonic request index.
    pub index: usize,
    /// Thread id that issued this request.
    pub thread_id: usize,
    /// Wall-clock latency for this request.
    pub latency: std::time::Duration,
    /// `true` if ember returned a valid JSON-RPC result.
    pub success: bool,
    /// Error message when `!success`.
    pub error: Option<String>,
}

/// Aggregate flood test results.
#[derive(Debug, Clone)]
pub struct FloodTestResult {
    /// All individual request results.
    pub requests: Vec<FloodRequestResult>,
    /// Total wall-clock time for the entire flood.
    pub total_duration: std::time::Duration,
    /// Count of successful requests.
    pub success_count: usize,
    /// Count of failed/timed-out requests.
    pub failure_count: usize,
    /// Median latency of successful requests.
    pub median_latency: std::time::Duration,
    /// p99 latency of successful requests.
    pub p99_latency: std::time::Duration,
}

/// Flood a target ember socket with concurrent `ember.status` RPCs.
///
/// Spawns `config.concurrency` threads, each sending a share of
/// `config.total_requests` in a tight loop. Returns timing and success/failure
/// data for analysis by the validation harness.
pub fn flood_test(config: &FloodTestConfig) -> FloodTestResult {
    let per_thread = config.total_requests / config.concurrency.max(1);
    let remainder = config.total_requests % config.concurrency.max(1);

    let start = std::time::Instant::now();
    let results: Vec<Vec<FloodRequestResult>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for tid in 0..config.concurrency {
            let count = if tid < remainder {
                per_thread + 1
            } else {
                per_thread
            };
            let socket = config.target_socket.clone();
            let timeout = config.request_timeout;
            handles.push(s.spawn(move || {
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let idx = tid * per_thread + i;
                    let req_start = std::time::Instant::now();
                    let result = send_jsonrpc_with_timeout(&socket, "ember.status", timeout);
                    let latency = req_start.elapsed();
                    let (success, error) = match result {
                        Ok(v) => {
                            if v.get("result").is_some() {
                                (true, None)
                            } else {
                                let msg = v
                                    .get("error")
                                    .map(|e| e.to_string())
                                    .unwrap_or_else(|| "no result".into());
                                (false, Some(msg))
                            }
                        }
                        Err(e) => (false, Some(e)),
                    };
                    out.push(FloodRequestResult {
                        index: idx,
                        thread_id: tid,
                        latency,
                        success,
                        error,
                    });
                }
                out
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap_or_default()).collect()
    });

    let total_duration = start.elapsed();
    let mut all: Vec<FloodRequestResult> = results.into_iter().flatten().collect();
    let success_count = all.iter().filter(|r| r.success).count();
    let failure_count = all.len() - success_count;

    let mut ok_latencies: Vec<std::time::Duration> =
        all.iter().filter(|r| r.success).map(|r| r.latency).collect();
    ok_latencies.sort();

    let median_latency = ok_latencies
        .get(ok_latencies.len() / 2)
        .copied()
        .unwrap_or_default();
    let p99_latency = ok_latencies
        .get((ok_latencies.len() as f64 * 0.99) as usize)
        .copied()
        .unwrap_or_default();

    all.sort_by_key(|r| r.index);

    FloodTestResult {
        requests: all,
        total_duration,
        success_count,
        failure_count,
        median_latency,
        p99_latency,
    }
}

/// Like [`send_jsonrpc`] but with an explicit connect+read timeout.
fn send_jsonrpc_with_timeout(
    socket_path: &Path,
    method: &str,
    timeout: std::time::Duration,
) -> Result<serde_json::Value, String> {
    use std::io::{BufRead, Write};
    use std::os::unix::net::UnixStream;

    let stream =
        UnixStream::connect(socket_path).map_err(|e| format!("connect {}: {e}", socket_path.display()))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| format!("set timeout: {e}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| format!("set write timeout: {e}"))?;

    let req = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": {},
        "id": 1,
    });
    let mut buf = serde_json::to_vec(&req).map_err(|e| format!("serialize: {e}"))?;
    buf.push(b'\n');
    (&stream)
        .write_all(&buf)
        .map_err(|e| format!("write: {e}"))?;

    let mut reader = std::io::BufReader::new(&stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("read: {e}"))?;
    serde_json::from_str(line.trim()).map_err(|e| format!("parse: {e}"))
}

/// Check if a socket is still accepting `ember.status` RPCs.
pub fn verify_ember_alive(socket_path: &Path) -> Result<std::time::Duration, String> {
    let start = std::time::Instant::now();
    let resp = send_jsonrpc_with_timeout(
        socket_path,
        "ember.status",
        std::time::Duration::from_secs(5),
    )?;
    let latency = start.elapsed();
    if resp.get("result").is_some() {
        Ok(latency)
    } else {
        Err(format!("ember not alive: {resp}"))
    }
}

/// Extract the PID of the ember process at a given socket from `ember.status`.
///
/// Returns `None` if the response doesn't include `pid`.
pub fn extract_ember_pid(socket_path: &Path) -> Option<u32> {
    let resp = send_jsonrpc_with_timeout(
        socket_path,
        "ember.status",
        std::time::Duration::from_secs(5),
    )
    .ok()?;
    resp.get("result")
        .and_then(|r| r.get("pid"))
        .and_then(|p| p.as_u64())
        .map(|p| p as u32)
}
