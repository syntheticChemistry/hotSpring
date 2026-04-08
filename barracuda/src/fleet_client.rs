// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-ember fleet discovery and JSON-RPC routing for coral-ember.
//!
//! Reads the fleet discovery file written by coral-glowplug (`mode: fleet`) and
//! routes GPU work to per-device ember Unix sockets. JSON-RPC calls use
//! [`crate::primal_bridge::send_jsonrpc`].
//!
//! ## Discovery file
//!
//! Default path: `$XDG_RUNTIME_DIR/biomeos/coral-ember-fleet.json` (falls back to
//! `/tmp` when `XDG_RUNTIME_DIR` is unset). Override with `EMBER_FLEET_FILE`.
//!
//! ## `SCM_RIGHTS` / `ember.adopt_device`
//!
//! When the protocol passes file descriptors after the JSON-RPC line, reception
//! requires `sendmsg`/`recvmsg` with ancillary `CMSG_SPACE` buffers. The default
//! build does not receive fds; use [`EmberClient::adopt_device_recv_scm_rights`]
//! with the `low-level` feature (pulls in `rustix` net APIs).

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

/// Environment variable overriding the fleet discovery JSON path.
pub const EMBER_FLEET_FILE_ENV: &str = "EMBER_FLEET_FILE";

/// Relative path under `XDG_RUNTIME_DIR` for fleet discovery.
pub const FLEET_FILE_REL: &str = "biomeos/coral-ember-fleet.json";

/// Physics domain tag: lattice QCD workloads (Wilson, staggered, HMC, …).
pub const DOMAIN_LATTICE_QCD: &str = "lattice_qcd";
/// Molecular dynamics / Yukawa OCP style workloads.
pub const DOMAIN_MD: &str = "molecular_dynamics";
/// Transport / hydrodynamics style workloads.
pub const DOMAIN_TRANSPORT: &str = "transport";
/// Default / catch-all routing bucket.
pub const DOMAIN_DEFAULT: &str = "default";

/// Raw fleet file payload (glowplug-compatible + optional extended metadata).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FleetFile {
    /// Should be `"fleet"` when written by glowplug fleet mode.
    #[serde(default)]
    pub mode: Option<String>,
    /// BDF → ember Unix socket path (primary routing table).
    #[serde(default)]
    pub routes: HashMap<String, String>,
    /// Count of live standby embers (informational).
    #[serde(default)]
    pub standby_count: Option<u32>,
    /// Optional extended per-device rows (vendor, health hints, physics domains).
    #[serde(default)]
    pub devices: Vec<FleetDeviceRecord>,
}

/// Optional extended metadata for a fleet member (not required for routing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FleetDeviceRecord {
    pub bdf: String,
    #[serde(default)]
    pub socket: Option<String>,
    #[serde(default)]
    pub vendor: Option<String>,
    /// Free-form health hint from the writer (e.g. `"Alive"`, `"Degraded"`).
    #[serde(default)]
    pub health: Option<String>,
    /// Physics domains this device is tagged for (see [`DOMAIN_*`] constants).
    #[serde(default)]
    pub physics_domains: Option<Vec<String>>,
    /// When set, this row is a hot standby for the given primary BDF.
    #[serde(default)]
    pub hot_standby_of: Option<String>,
    /// Last experiment did not clean up; router should not schedule new work here.
    #[serde(default)]
    pub experiment_dirty: bool,
    /// Device needs a driver warm cycle before trusted adoption.
    #[serde(default)]
    pub needs_warm_cycle: bool,
}

/// Resolved fleet discovery: JSON contents + path it was loaded from.
#[derive(Debug, Clone)]
pub struct FleetDiscovery {
    path: PathBuf,
    file: FleetFile,
}

impl FleetDiscovery {
    /// Resolve the fleet file path.
    ///
    /// Search order:
    /// 1. `EMBER_FLEET_FILE` env var (explicit override)
    /// 2. `$XDG_RUNTIME_DIR/biomeos/coral-ember-fleet.json`
    /// 3. `/tmp/biomeos/coral-ember-fleet.json` (glowplug's default write path)
    #[must_use]
    pub fn resolve_path() -> PathBuf {
        if let Ok(p) = std::env::var(EMBER_FLEET_FILE_ENV) {
            return PathBuf::from(p);
        }
        if let Ok(runtime) = std::env::var("XDG_RUNTIME_DIR") {
            let xdg_path = PathBuf::from(runtime).join(FLEET_FILE_REL);
            if xdg_path.exists() {
                return xdg_path;
            }
        }
        PathBuf::from("/tmp").join(FLEET_FILE_REL)
    }

    /// Read and parse the fleet discovery file at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let file: FleetFile =
            serde_json::from_slice(&bytes).map_err(|e| format!("parse fleet JSON: {e}"))?;
        Ok(Self { path, file })
    }

    /// Load the fleet file when present; if the path does not exist, return `Ok(None)` (graceful
    /// degradation). Other I/O or JSON errors are propagated.
    pub fn load_if_present(path: impl AsRef<Path>) -> Result<Option<Self>, String> {
        let path = path.as_ref().to_path_buf();
        match std::fs::read(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("read {}: {e}", path.display())),
            Ok(bytes) => {
                let file: FleetFile = serde_json::from_slice(&bytes)
                    .map_err(|e| format!("parse fleet JSON: {e}"))?;
                Ok(Some(Self { path, file }))
            }
        }
    }

    /// Load from [`Self::resolve_path`].
    pub fn load_default() -> Result<Self, String> {
        Self::load(Self::resolve_path())
    }

    /// Path that was loaded.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Parsed fleet document.
    #[must_use]
    pub fn file(&self) -> &FleetFile {
        &self.file
    }

    /// Consume into the parsed file.
    #[must_use]
    pub fn into_file(self) -> FleetFile {
        self.file
    }
}

/// One routable ember with optional discovery metadata and probe state.
#[derive(Debug, Clone)]
pub struct FleetDeviceRoute {
    pub bdf: String,
    pub socket_path: PathBuf,
    pub vendor: Option<String>,
    pub health_hint: Option<String>,
    pub physics_domains: Option<Vec<String>>,
    pub hot_standby_of: Option<String>,
    pub experiment_dirty: bool,
    pub needs_warm_cycle: bool,
    /// `None` until [`FleetRouter::probe_all`] or [`FleetRouter::discover`] runs.
    pub reachable: Option<bool>,
}

impl FleetDeviceRoute {
    fn from_record(r: &FleetDeviceRecord, routes: &HashMap<String, String>) -> Option<Self> {
        let socket = r
            .socket
            .clone()
            .or_else(|| routes.get(&r.bdf).cloned())?;
        Some(Self {
            bdf: r.bdf.clone(),
            socket_path: PathBuf::from(socket),
            vendor: r.vendor.clone(),
            health_hint: r.health.clone(),
            physics_domains: r.physics_domains.clone(),
            hot_standby_of: r.hot_standby_of.clone(),
            experiment_dirty: r.experiment_dirty,
            needs_warm_cycle: r.needs_warm_cycle,
            reachable: None,
        })
    }

    fn from_routes_only(bdf: &str, socket: &str) -> Self {
        Self {
            bdf: bdf.to_string(),
            socket_path: PathBuf::from(socket),
            vendor: None,
            health_hint: None,
            physics_domains: None,
            hot_standby_of: None,
            experiment_dirty: false,
            needs_warm_cycle: false,
            reachable: None,
        }
    }
}

/// Build [`FleetDeviceRoute`] list from a [`FleetFile`].
fn expand_devices(file: &FleetFile) -> Vec<FleetDeviceRoute> {
    let mut out = Vec::new();
    if !file.devices.is_empty() {
        for d in &file.devices {
            if let Some(row) = FleetDeviceRoute::from_record(d, &file.routes) {
                out.push(row);
            }
        }
        // Include any BDF in routes missing from devices[]
        let seen: std::collections::HashSet<String> =
            out.iter().map(|x| x.bdf.clone()).collect();
        for (bdf, sock) in &file.routes {
            if !seen.contains(bdf) {
                out.push(FleetDeviceRoute::from_routes_only(bdf, sock));
            }
        }
    } else {
        let mut bdfs: Vec<&String> = file.routes.keys().collect();
        bdfs.sort();
        for bdf in bdfs {
            if let Some(sock) = file.routes.get(bdf) {
                out.push(FleetDeviceRoute::from_routes_only(bdf, sock));
            }
        }
    }
    out.sort_by(|a, b| a.bdf.cmp(&b.bdf));
    out
}

/// Routes GPU work using fleet discovery + optional socket probes.
#[derive(Debug, Clone)]
pub struct FleetRouter {
    devices: Vec<FleetDeviceRoute>,
}

impl FleetRouter {
    /// Build from an already-parsed [`FleetFile`] (no probing).
    #[must_use]
    pub fn from_fleet_file(file: &FleetFile) -> Self {
        Self {
            devices: expand_devices(file),
        }
    }

    /// Load discovery from the default / env-configured path and probe each socket.
    pub fn discover() -> Result<Self, String> {
        let disc = FleetDiscovery::load_default()?;
        let mut router = Self::from_fleet_file(disc.file());
        router.probe_all()?;
        Ok(router)
    }

    /// Probe every known socket with `ember.status` and set [`FleetDeviceRoute::reachable`].
    pub fn probe_all(&mut self) -> Result<(), String> {
        for d in &mut self.devices {
            d.reachable = Some(probe_ember_socket(&d.socket_path));
        }
        Ok(())
    }

    /// Route to a specific PCI BDF.
    #[must_use]
    pub fn route_by_bdf(&self, bdf: &str) -> Option<&FleetDeviceRoute> {
        self.devices.iter().find(|d| d.bdf == bdf)
    }

    /// Pick a device for `domain` using metadata + probe results.
    ///
    /// Scoring: domain tag match, reachable probe, health hint, vendor heuristics
    /// (NVIDIA often preferred for `DOMAIN_LATTICE_QCD`).
    #[must_use]
    pub fn route_by_capability(&self, domain: &str) -> Option<&FleetDeviceRoute> {
        let mut best: Option<(&FleetDeviceRoute, i32)> = None;
        for d in &self.devices {
            let score = Self::score_for_domain(d, domain);
            let better = match best {
                None => true,
                Some((_, s)) => score > s,
            };
            if better {
                best = Some((d, score));
            }
        }
        best.map(|(d, _)| d)
    }

    fn score_for_domain(d: &FleetDeviceRoute, domain: &str) -> i32 {
        let domain_l = domain.to_ascii_lowercase();
        let mut score: i32 = 0;
        if d.reachable == Some(true) {
            score += 100;
        }
        if let Some(domains) = &d.physics_domains {
            if domains.iter().any(|x| x.to_ascii_lowercase() == domain_l) {
                score += 80;
            }
        }
        if matches!(
            d.health_hint.as_deref().map(str::to_ascii_lowercase).as_deref(),
            Some("alive") | Some("ok") | Some("pristine")
        ) {
            score += 40;
        }
        if domain_l == DOMAIN_LATTICE_QCD || domain_l == "qcd" {
            if d.vendor
                .as_deref()
                .is_some_and(|v| v.to_ascii_lowercase().contains("nvidia"))
            {
                score += 25;
            }
        }
        if domain_l == DOMAIN_DEFAULT || domain.is_empty() {
            score += 5;
        }
        score
    }

    fn find_clean_standby_for(&self, primary_bdf: &str) -> Option<&FleetDeviceRoute> {
        self.devices.iter().find(|d| {
            d.hot_standby_of.as_deref() == Some(primary_bdf)
                && !d.experiment_dirty
                && !d.needs_warm_cycle
        })
    }

    /// Route for `domain` while respecting fault hints: [`FleetDeviceRoute::experiment_dirty`]
    /// entries are skipped; [`FleetDeviceRoute::needs_warm_cycle`] prefers a clean hot standby
    /// when one exists, otherwise reports warm-cycle requirement.
    #[must_use]
    pub fn route_resilient(&self, domain: &str) -> ResilientRoute<'_> {
        let eligible: Vec<&FleetDeviceRoute> = self
            .devices
            .iter()
            .filter(|d| !d.experiment_dirty)
            .collect();
        if eligible.is_empty() {
            return ResilientRoute::NoEligibleDevice;
        }
        let mut best: Option<(&FleetDeviceRoute, i32)> = None;
        for d in eligible {
            let score = Self::score_for_domain(d, domain);
            let better = match best {
                None => true,
                Some((_, s)) => score > s,
            };
            if better {
                best = Some((d, score));
            }
        }
        let Some((best, _)) = best else {
            return ResilientRoute::NoEligibleDevice;
        };
        if best.needs_warm_cycle {
            if let Some(standby) = self.find_clean_standby_for(&best.bdf) {
                return ResilientRoute::Routed {
                    device: standby,
                    adopted_from_faulted_primary: true,
                };
            }
            return ResilientRoute::WarmCycleRequired { device: best };
        }
        ResilientRoute::Routed {
            device: best,
            adopted_from_faulted_primary: false,
        }
    }

    /// All routes (sorted by BDF).
    #[must_use]
    pub fn devices(&self) -> &[FleetDeviceRoute] {
        &self.devices
    }
}

/// Result of [`FleetRouter::route_resilient`]: either a usable device, an explicit warm-cycle
/// request, or no routable device.
#[derive(Debug, Clone, Copy)]
pub enum ResilientRoute<'a> {
    /// Work should run on `device`. When `adopted_from_faulted_primary`, the primary had
    /// `needs_warm_cycle` and a clean hot standby was selected instead.
    Routed {
        device: &'a FleetDeviceRoute,
        adopted_from_faulted_primary: bool,
    },
    /// No non-dirty device was available for the domain.
    NoEligibleDevice,
    /// The best-scoring device requires a warm cycle and no clean standby exists.
    WarmCycleRequired {
        device: &'a FleetDeviceRoute,
    },
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
    pub fn client_for_route(&mut self, route: &FleetDeviceRoute) -> &EmberClient {
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

/// True if `ember.status` returns a JSON-RPC result on this socket.
#[must_use]
pub fn probe_ember_socket(socket_path: &Path) -> bool {
    send_jsonrpc(socket_path, "ember.status", &serde_json::json!({}))
        .map(|resp| resp.get("result").is_some())
        .unwrap_or(false)
}

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
    ///
    /// For ancillary `SCM_RIGHTS` fds on the same Unix session, see
    /// [`Self::adopt_device_recv_scm_rights`].
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
    ///
    /// Without the `low-level` feature this always returns
    /// `Err` — receiving fds needs `recvmsg` ancillary buffers (`rustix::net`).
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

    // Ancillary fds may follow in a separate datagram on the same SOCK_STREAM;
    // attempt one recvmsg (mirrors coral-ember `recv_with_fds`).
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

    // No second message, or JSON-only response (current ember): `fds` may be empty.
    let _ = msg_result.bytes;
    Ok((result, fds))
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
///
/// The target ember is expected to degrade or die under sufficient load;
/// this is intentional and part of the sacrificial architecture proof.
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
///
/// Useful for verifying isolation during flood tests — call on OTHER embers
/// to confirm they're unaffected by the flood on the target.
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    const SAMPLE_FLEET_JSON: &str = r#"{
        "mode": "fleet",
        "routes": {
            "0000:03:00.0": "/run/coralreef/ember-0000-03-00.0.sock",
            "0000:4c:00.0": "/run/coralreef/ember-0000-4c-00.0.sock"
        },
        "standby_count": 1,
        "devices": [
            {
                "bdf": "0000:03:00.0",
                "vendor": "NVIDIA Corporation",
                "health": "Alive",
                "physics_domains": ["lattice_qcd", "default"]
            },
            {
                "bdf": "0000:4c:00.0",
                "vendor": "NVIDIA Corporation",
                "health": "Degraded",
                "physics_domains": ["molecular_dynamics"]
            }
        ]
    }"#;

    #[test]
    fn fleet_file_parses_routes_and_devices() -> Result<(), String> {
        let f: FleetFile =
            serde_json::from_str(SAMPLE_FLEET_JSON).map_err(|e| e.to_string())?;
        assert_eq!(f.mode.as_deref(), Some("fleet"));
        assert_eq!(f.routes.len(), 2);
        assert_eq!(f.devices.len(), 2);
        let router = FleetRouter::from_fleet_file(&f);
        assert_eq!(router.devices().len(), 2);
        Ok(())
    }

    #[test]
    fn route_by_bdf_finds_socket() -> Result<(), String> {
        let f: FleetFile =
            serde_json::from_str(SAMPLE_FLEET_JSON).map_err(|e| e.to_string())?;
        let router = FleetRouter::from_fleet_file(&f);
        let d = router
            .route_by_bdf("0000:03:00.0")
            .ok_or_else(|| "missing bdf 03".to_string())?;
        assert_eq!(
            d.socket_path,
            PathBuf::from("/run/coralreef/ember-0000-03-00.0.sock")
        );
        Ok(())
    }

    #[test]
    fn route_by_capability_prefers_domain_and_reachable() -> Result<(), String> {
        let f: FleetFile =
            serde_json::from_str(SAMPLE_FLEET_JSON).map_err(|e| e.to_string())?;
        let mut router = FleetRouter::from_fleet_file(&f);
        for d in &mut router.devices {
            if d.bdf == "0000:03:00.0" {
                d.reachable = Some(true);
            } else {
                d.reachable = Some(false);
            }
        }
        let pick = router
            .route_by_capability(DOMAIN_LATTICE_QCD)
            .ok_or_else(|| "no route".to_string())?;
        assert_eq!(pick.bdf, "0000:03:00.0");
        Ok(())
    }

    #[test]
    fn expand_devices_routes_only_without_devices_array() {
        let f = FleetFile {
            mode: Some("fleet".into()),
            routes: HashMap::from([(
                "0000:01:00.0".into(),
                "/tmp/e0.sock".into(),
            )]),
            standby_count: None,
            devices: vec![],
        };
        let router = FleetRouter::from_fleet_file(&f);
        assert_eq!(router.devices().len(), 1);
        assert_eq!(router.devices()[0].bdf, "0000:01:00.0");
    }

    #[test]
    fn jsonrpc_ok_result_errors_on_rpc_error_field() {
        let v = serde_json::json!({"error": {"code": -1, "message": "no"}, "id": 1});
        assert!(jsonrpc_ok_result(&v).is_err());
    }

    #[test]
    fn ember_warm_cycle_request_shape() {
        let v = EmberClient::warm_cycle_request("0000:03:00.0");
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["method"], "ember.warm_cycle");
        assert_eq!(v["params"]["bdf"], "0000:03:00.0");
    }

    #[test]
    fn fleet_ember_hub_distinct_sockets() {
        let mut hub = FleetEmberHub::default();
        assert_eq!(
            hub.client_for_socket("/tmp/a.sock").socket_path(),
            Path::new("/tmp/a.sock")
        );
        assert_eq!(
            hub.client_for_socket("/tmp/b.sock").socket_path(),
            Path::new("/tmp/b.sock")
        );
        assert_eq!(hub.len(), 2);
        let first = hub.client_for_socket("/tmp/a.sock").socket_path().to_path_buf();
        let again = hub.client_for_socket("/tmp/a.sock").socket_path().to_path_buf();
        assert_eq!(first, again);
        assert_eq!(hub.len(), 2);
    }

    #[test]
    fn route_resilient_adopts_hot_standby_when_primary_needs_warm() -> Result<(), String> {
        let json = r#"{
            "mode": "fleet",
            "routes": {
                "0000:03:00.0": "/tmp/p.sock",
                "0000:04:00.0": "/tmp/s.sock"
            },
            "devices": [
                {
                    "bdf": "0000:03:00.0",
                    "physics_domains": ["lattice_qcd"],
                    "needs_warm_cycle": true
                },
                {
                    "bdf": "0000:04:00.0",
                    "hot_standby_of": "0000:03:00.0",
                    "physics_domains": ["lattice_qcd"]
                }
            ]
        }"#;
        let f: FleetFile = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let router = FleetRouter::from_fleet_file(&f);
        match router.route_resilient(DOMAIN_LATTICE_QCD) {
            ResilientRoute::Routed {
                device,
                adopted_from_faulted_primary,
            } => {
                assert_eq!(device.bdf, "0000:04:00.0");
                assert!(adopted_from_faulted_primary);
            }
            _ => return Err("expected routed standby".to_string()),
        }
        Ok(())
    }

    #[test]
    fn route_resilient_skips_experiment_dirty_only_fleet() -> Result<(), String> {
        let json = r#"{
            "mode": "fleet",
            "routes": { "0000:01:00.0": "/tmp/x.sock" },
            "devices": [
                {
                    "bdf": "0000:01:00.0",
                    "experiment_dirty": true,
                    "needs_warm_cycle": true,
                    "physics_domains": ["lattice_qcd"]
                }
            ]
        }"#;
        let f: FleetFile = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let router = FleetRouter::from_fleet_file(&f);
        assert!(matches!(
            router.route_resilient(DOMAIN_LATTICE_QCD),
            ResilientRoute::NoEligibleDevice
        ));
        Ok(())
    }

    #[test]
    fn load_if_present_missing_ok() -> Result<(), String> {
        let p = std::env::temp_dir().join("hotspring-fleet-absent-xxxxxxxx");
        assert!(FleetDiscovery::load_if_present(&p)?.is_none());
        Ok(())
    }

    #[test]
    fn flood_test_config_defaults() {
        let cfg = FloodTestConfig::default();
        assert_eq!(cfg.concurrency, 50);
        assert_eq!(cfg.total_requests, 500);
    }

    #[test]
    fn flood_test_against_missing_socket_all_fail() {
        let cfg = FloodTestConfig {
            target_socket: PathBuf::from("/tmp/hotspring-test-absent-flood.sock"),
            concurrency: 2,
            total_requests: 4,
            request_timeout: std::time::Duration::from_millis(200),
        };
        let result = flood_test(&cfg);
        assert_eq!(result.success_count, 0);
        assert_eq!(result.failure_count, 4);
        assert!(result.total_duration < std::time::Duration::from_secs(10));
    }

    #[test]
    fn verify_ember_alive_missing_socket_fails() {
        let p = Path::new("/tmp/hotspring-test-absent-ember.sock");
        assert!(verify_ember_alive(p).is_err());
    }

    #[test]
    fn extract_ember_pid_missing_socket_returns_none() {
        let p = Path::new("/tmp/hotspring-test-absent-pid.sock");
        assert!(extract_ember_pid(p).is_none());
    }
}
