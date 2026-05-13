// SPDX-License-Identifier: AGPL-3.0-or-later

//! Multi-ember fleet discovery and JSON-RPC routing.
//!
//! Discovers GPU compute endpoints via two paths:
//! 1. **toadStool** (modern): NUCLEUS capability routing → `compute.sock`
//!    at `$XDG_RUNTIME_DIR/biomeos/compute.sock` (or `$TOADSTOOL_SOCKET`)
//! 2. **coral-ember** (legacy): Fleet JSON written by coral-glowplug `mode: fleet`
//!    at `$XDG_RUNTIME_DIR/biomeos/coral-ember-fleet.json` (or `$EMBER_FLEET_FILE`)
//!
//! Per-ember RPC methods live in [`crate::fleet_ember`].
//! toadStool-specific dispatch in [`crate::fleet_toadstool`] (feature-gated).
//!
//! ## Post-excision note (May 2026)
//!
//! coralReef Sprint 9 excised coral-ember/coral-glowplug. The legacy diesel
//! discovery paths (`/run/coralreef/ember-*.sock`) are retained for backward
//! compatibility but the modern path is toadStool's compute socket.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::HotSpringError;
use crate::primal_bridge::send_jsonrpc;

pub use crate::fleet_ember::{
    EmberClient, FleetEmberHub, FloodRequestResult, FloodTestConfig, FloodTestResult,
    extract_ember_pid, flood_test, verify_ember_alive,
};

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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    /// Physics domains this device is tagged for (see `DOMAIN_*` constants).
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
    /// 3. `<temp_dir>/biomeos/coral-ember-fleet.json` (platform temp via `std::env::temp_dir()`)
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
        std::env::temp_dir().join(FLEET_FILE_REL)
    }

    /// Read and parse the fleet discovery file at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, HotSpringError> {
        let path = path.as_ref().to_path_buf();
        let bytes = std::fs::read(&path)?;
        let file: FleetFile = serde_json::from_slice(&bytes)?;
        Ok(Self { path, file })
    }

    /// Load the fleet file when present; if the path does not exist, return `Ok(None)` (graceful
    /// degradation). Other I/O or JSON errors are propagated.
    pub fn load_if_present(path: impl AsRef<Path>) -> Result<Option<Self>, HotSpringError> {
        let path = path.as_ref().to_path_buf();
        match std::fs::read(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
            Ok(bytes) => {
                let file: FleetFile = serde_json::from_slice(&bytes)?;
                Ok(Some(Self { path, file }))
            }
        }
    }

    /// Load from [`Self::resolve_path`].
    pub fn load_default() -> Result<Self, HotSpringError> {
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
        let socket = r.socket.clone().or_else(|| routes.get(&r.bdf).cloned())?;
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
    if file.devices.is_empty() {
        let mut bdfs: Vec<&str> = file.routes.keys().map(String::as_str).collect();
        bdfs.sort_unstable();
        for bdf in bdfs {
            if let Some(sock) = file.routes.get(bdf) {
                out.push(FleetDeviceRoute::from_routes_only(bdf, sock));
            }
        }
    } else {
        for d in &file.devices {
            if let Some(row) = FleetDeviceRoute::from_record(d, &file.routes) {
                out.push(row);
            }
        }
        let seen: std::collections::HashSet<String> = out.iter().map(|x| x.bdf.clone()).collect();
        for (bdf, sock) in &file.routes {
            if !seen.contains(bdf) {
                out.push(FleetDeviceRoute::from_routes_only(bdf, sock));
            }
        }
    }
    out.sort_by(|a, b| a.bdf.cmp(&b.bdf));
    out
}

/// True if `ember.status` returns a JSON-RPC result on this socket.
///
/// toadStool S237+ (Phase A/B) serves `ember.status` and `ember.list`
/// as live JSON-RPC methods via sysfs GPU enumeration. The `ember.phase`
/// field in `compute.dispatch.capabilities` reports "B" as of S239.
#[must_use]
pub fn probe_ember_socket(socket_path: &Path) -> bool {
    send_jsonrpc(socket_path, "ember.status", &serde_json::json!({}))
        .is_ok_and(|resp| resp.get("result").is_some())
}

/// Resolve the toadStool compute socket path.
///
/// Precedence: `TOADSTOOL_SOCKET` env → `$XDG_RUNTIME_DIR/biomeos/compute.sock`
/// → `/tmp/biomeos/compute.sock`.
fn toadstool_compute_socket() -> PathBuf {
    if let Ok(sock) = std::env::var("TOADSTOOL_SOCKET") {
        return PathBuf::from(sock);
    }
    let base = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".to_owned());
    PathBuf::from(base).join("biomeos").join("compute.sock")
}

/// Well-known toadStool runtime directory for ember sockets.
const TOADSTOOL_RUN_DEFAULT: &str = "/run/toadstool";

/// Resolve the toadStool runtime directory for ember sockets.
///
/// Precedence: `TOADSTOOL_RUN_DIR` → `CORALREEF_RUN_DIR` (deprecated) →
/// `$XDG_RUNTIME_DIR/toadstool` → `/run/toadstool`.
fn toadstool_run_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("TOADSTOOL_RUN_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("CORALREEF_RUN_DIR") {
        log::warn!("CORALREEF_RUN_DIR is deprecated — use TOADSTOOL_RUN_DIR");
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        let base = PathBuf::from(&xdg);
        let p = base.join("toadstool");
        if p.is_dir() {
            return p;
        }
        let legacy = base.join("coralreef");
        if legacy.is_dir() {
            return legacy;
        }
    }
    PathBuf::from(TOADSTOOL_RUN_DEFAULT)
}

/// Discover the compute socket for a specific BDF.
///
/// Discovery chain:
/// 1. NUCLEUS capability routing (`compute` domain → toadStool `ember.list`)
/// 2. toadStool compute socket direct probe (`ember.list` on `compute.sock`)
/// 3. toadStool run directory: `toadstool-ember-*.sock` or legacy `ember-*.sock`
///
/// Returns the socket path of whichever endpoint holds the requested BDF.
#[must_use]
pub fn discover_diesel_ember_socket(bdf: &str) -> Option<PathBuf> {
    let ctx = crate::primal_bridge::NucleusContext::detect();

    // 1. NUCLEUS capability routing
    if let Some(ep) = ctx.by_domain("compute") {
        let socket = PathBuf::from(&ep.socket);
        if ep.alive {
            if let Ok(resp) = send_jsonrpc(&socket, "ember.list", &serde_json::json!({})) {
                if resp
                    .get("result")
                    .and_then(|r| r.get("devices"))
                    .and_then(|d| d.as_array())
                    .is_some_and(|arr| arr.iter().any(|d| d.as_str() == Some(bdf)))
                {
                    return Some(socket);
                }
            }
        }
    }

    // 2. toadStool compute socket direct probe
    let ts_sock = toadstool_compute_socket();
    if ts_sock.exists() {
        if let Ok(resp) = send_jsonrpc(&ts_sock, "ember.list", &serde_json::json!({})) {
            if resp
                .get("result")
                .and_then(|r| r.get("devices"))
                .and_then(|d| d.as_array())
                .is_some_and(|arr| arr.iter().any(|d| d.as_str() == Some(bdf)))
            {
                return Some(ts_sock);
            }
        }
    }

    // 3. toadStool run directory (toadstool-ember-*.sock) + legacy (ember-*.sock)
    let run_dir = toadstool_run_dir();
    if let Ok(entries) = std::fs::read_dir(run_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            let is_ember = (name_str.starts_with("toadstool-ember-")
                || name_str.starts_with("ember-"))
                && name_str.ends_with(".sock");
            if !is_ember {
                continue;
            }
            let path = entry.path();
            if let Ok(resp) = send_jsonrpc(&path, "ember.list", &serde_json::json!({})) {
                if let Some(devices) = resp
                    .get("result")
                    .and_then(|r| r.get("devices"))
                    .and_then(|d| d.as_array())
                {
                    for dev in devices {
                        if dev.as_str().is_some_and(|s| s == bdf) {
                            return Some(path);
                        }
                    }
                }
            }
        }
    }
    None
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
    pub fn discover() -> Result<Self, HotSpringError> {
        let disc = FleetDiscovery::load_default()?;
        let mut router = Self::from_fleet_file(disc.file());
        router.probe_all()?;
        Ok(router)
    }

    /// Probe every known socket with `ember.status` and set [`FleetDeviceRoute::reachable`].
    pub fn probe_all(&mut self) -> Result<(), HotSpringError> {
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
        if let Some(domains) = &d.physics_domains
            && domains.iter().any(|x| x.to_ascii_lowercase() == domain_l)
        {
            score += 80;
        }
        if matches!(
            d.health_hint
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("alive" | "ok" | "pristine")
        ) {
            score += 40;
        }
        if (domain_l == DOMAIN_LATTICE_QCD || domain_l == "qcd")
            && d.vendor
                .as_deref()
                .is_some_and(|v| v.to_ascii_lowercase().contains("nvidia"))
        {
            score += 25;
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

    /// Route for `domain` while respecting fault hints.
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

/// Result of [`FleetRouter::route_resilient`].
#[derive(Debug, Clone, Copy)]
pub enum ResilientRoute<'a> {
    Routed {
        device: &'a FleetDeviceRoute,
        adopted_from_faulted_primary: bool,
    },
    NoEligibleDevice,
    WarmCycleRequired {
        device: &'a FleetDeviceRoute,
    },
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
        let f: FleetFile = serde_json::from_str(SAMPLE_FLEET_JSON).map_err(|e| e.to_string())?;
        assert_eq!(f.mode.as_deref(), Some("fleet"));
        assert_eq!(f.routes.len(), 2);
        assert_eq!(f.devices.len(), 2);
        let router = FleetRouter::from_fleet_file(&f);
        assert_eq!(router.devices().len(), 2);
        Ok(())
    }

    #[test]
    fn route_by_bdf_finds_socket() -> Result<(), String> {
        let f: FleetFile = serde_json::from_str(SAMPLE_FLEET_JSON).map_err(|e| e.to_string())?;
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
        let f: FleetFile = serde_json::from_str(SAMPLE_FLEET_JSON).map_err(|e| e.to_string())?;
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
            routes: HashMap::from([("0000:01:00.0".into(), "/tmp/e0.sock".into())]),
            standby_count: None,
            devices: vec![],
        };
        let router = FleetRouter::from_fleet_file(&f);
        assert_eq!(router.devices().len(), 1);
        assert_eq!(router.devices()[0].bdf, "0000:01:00.0");
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
        let first = hub
            .client_for_socket("/tmp/a.sock")
            .socket_path()
            .to_path_buf();
        let again = hub
            .client_for_socket("/tmp/a.sock")
            .socket_path()
            .to_path_buf();
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
