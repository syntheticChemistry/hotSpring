// SPDX-License-Identifier: AGPL-3.0-or-later

//! NUCLEUS primal discovery — runtime detection of available primals.
//!
//! Scans `$XDG_RUNTIME_DIR/biomeos/` for `*.sock` files matching
//! `{name}-{family}.sock`, then probes each with `health.liveness` and
//! optionally `capability.list`. Degrades gracefully: when no primals
//! are present, everything works standalone (bare guideStone).
//!
//! Set `HOTSPRING_NO_NUCLEUS=1` to skip all primal detection (useful
//! for headless HPC or isolated benchmarks).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const JSONRPC_SOCKET_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);
const JSONRPC_READ_BUFFER_BYTES: usize = 4096;
const JSONRPC_REQUEST_ID: i64 = 1;

/// Max consecutive IPC failures before marking an endpoint dead.
const CIRCUIT_BREAKER_THRESHOLD: u32 = 3;
/// Cooldown before re-probing a dead endpoint (seconds).
const CIRCUIT_BREAKER_COOLDOWN_SECS: u64 = 30;

/// A detected primal with its socket path and optional capability payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalEndpoint {
    pub name: String,
    pub socket: String,
    pub alive: bool,
    /// Result of `capability.list` when the primal supports it and liveness passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<serde_json::Value>,
    /// Consecutive IPC failure count (circuit breaker).
    #[serde(skip)]
    pub fail_count: u32,
    /// When the endpoint was marked dead (for cooldown-based re-probe).
    #[serde(skip)]
    pub dead_since: Option<std::time::Instant>,
}

/// Runtime snapshot of which NUCLEUS primals are reachable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NucleusContext {
    /// Discovered primals keyed by logical name from the socket stem (before `-{family}`).
    pub discovered: HashMap<String, PrimalEndpoint>,
    pub family_id: String,
}

impl NucleusContext {
    /// Probe all sockets under the biomeos runtime directory and build a context.
    ///
    /// Returns an empty context when `HOTSPRING_NO_NUCLEUS=1` is set, on non-Unix
    /// platforms, or when the biomeos directory is missing or empty.
    #[must_use]
    pub fn detect() -> Self {
        let family = crate::niche::family_id();

        if std::env::var("HOTSPRING_NO_NUCLEUS").is_ok_and(|v| v == "1") {
            return Self::empty(&family);
        }

        #[cfg(not(unix))]
        {
            return Self::empty(&family);
        }

        #[cfg(unix)]
        {
            let mut discovered = HashMap::new();
            for dir in crate::niche::socket_dirs() {
                for path in collect_biomeos_socks(&dir, &family) {
                    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    let primal_name = if let Some((name, _sock_family)) = stem.rsplit_once('-') {
                        name
                    } else {
                        // Single-name socket (e.g. compute.sock, math.sock) —
                        // use the stem as-is; aliases will map to canonical primal.
                        stem
                    };
                    let ep = probe_socket(&path, primal_name);
                    discovered.insert(primal_name.to_string(), ep);
                }
            }

            Self {
                discovered,
                family_id: family,
            }
        }
    }

    /// Build an empty context with no discovered primals (for testing).
    #[must_use]
    pub fn empty(family: &str) -> Self {
        Self {
            discovered: HashMap::new(),
            family_id: family.to_string(),
        }
    }

    /// Re-scan socket directories and re-probe all endpoints.
    pub fn refresh(&mut self) {
        *self = Self::detect();
    }

    /// Record an IPC failure for a primal. After [`CIRCUIT_BREAKER_THRESHOLD`]
    /// consecutive failures the endpoint is marked dead with a cooldown.
    pub fn record_failure(&mut self, primal: &str) {
        let canonical = self.resolve_canonical_name(primal);
        if let Some(ep) = self.discovered.get_mut(&canonical) {
            ep.fail_count += 1;
            if ep.fail_count >= CIRCUIT_BREAKER_THRESHOLD {
                ep.alive = false;
                ep.dead_since = Some(std::time::Instant::now());
            }
        }
    }

    /// Record a successful IPC call — resets the failure counter.
    pub fn record_success(&mut self, primal: &str) {
        let canonical = self.resolve_canonical_name(primal);
        if let Some(ep) = self.discovered.get_mut(&canonical) {
            ep.fail_count = 0;
            ep.dead_since = None;
        }
    }

    /// Check if a dead endpoint's cooldown has expired and re-probe it.
    pub fn maybe_reprobe(&mut self, primal: &str) -> bool {
        let canonical = self.resolve_canonical_name(primal);
        let should_reprobe = self.discovered.get(&canonical).is_some_and(|ep| {
            !ep.alive
                && ep
                    .dead_since
                    .is_some_and(|t| t.elapsed().as_secs() >= CIRCUIT_BREAKER_COOLDOWN_SECS)
        });
        if should_reprobe {
            if let Some(ep) = self.discovered.get(&canonical) {
                let path = PathBuf::from(&ep.socket);
                let fresh = probe_socket(&path, &canonical);
                if let Some(entry) = self.discovered.get_mut(&canonical) {
                    entry.alive = fresh.alive;
                    entry.capabilities = fresh.capabilities;
                    entry.fail_count = 0;
                    entry.dead_since = None;
                }
            }
        }
        self.discovered.get(&canonical).is_some_and(|ep| ep.alive)
    }

    fn resolve_canonical_name(&self, primal: &str) -> String {
        if self.discovered.contains_key(primal) {
            return primal.to_string();
        }
        for alias in known_aliases(primal) {
            if self.discovered.contains_key(alias) {
                return alias.to_string();
            }
        }
        primal.to_string()
    }

    /// Look up a primal by the logical name used in socket filenames.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&PrimalEndpoint> {
        self.resolve_endpoint(name)
    }

    fn resolve_endpoint(&self, primal: &str) -> Option<&PrimalEndpoint> {
        if let Some(ep) = self.discovered.get(primal) {
            return Some(ep);
        }
        for alias in known_aliases(primal) {
            if let Some(ep) = self.discovered.get(alias) {
                return Some(ep);
            }
        }
        None
    }

    /// Discover by capability domain. Preferred over named accessors.
    ///
    /// Returns first alive primal whose `capability.list` includes a string
    /// starting with `domain`. Use `"compute"` for toadStool, `"crypto"` for
    /// bearDog, `"shader"` for coralReef, etc.
    #[must_use]
    pub fn by_domain(&self, domain: &str) -> Option<&PrimalEndpoint> {
        self.get_by_capability(domain)
    }

    /// Names of all alive primals (for banner / manifest), sorted for stable output.
    #[must_use]
    pub fn alive_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .discovered
            .iter()
            .filter(|(_, ep)| ep.alive)
            .map(|(k, _)| k.as_str())
            .collect();
        names.sort_unstable();
        names
    }

    /// True when at least one primal is reachable.
    #[must_use]
    pub fn any_alive(&self) -> bool {
        !self.alive_names().is_empty()
    }

    /// All discovered endpoints `(name, endpoint)` sorted by name.
    #[must_use]
    pub fn all_endpoints(&self) -> Vec<(&str, &PrimalEndpoint)> {
        let mut v: Vec<(&str, &PrimalEndpoint)> = self
            .discovered
            .iter()
            .map(|(k, ep)| (k.as_str(), ep))
            .collect();
        v.sort_by_key(|(k, _)| *k);
        v
    }

    /// Print a banner line showing discovered primals.
    pub fn print_banner(&self) {
        let names = self.alive_names();
        if names.is_empty() {
            println!("  NUCLEUS: standalone (no primals detected)");
        } else {
            println!(
                "  NUCLEUS: {} primal(s) — {}",
                names.len(),
                names.join(", ")
            );
        }
    }

    /// Discover by capability rather than by name: first alive primal whose
    /// `capability.list` includes a capability string starting with `capability_domain`.
    #[must_use]
    pub fn get_by_capability(&self, capability_domain: &str) -> Option<&PrimalEndpoint> {
        let has_typed_cap = |cap_key: &str, ep: &&PrimalEndpoint| {
            ep.capabilities
                .as_ref()
                .and_then(|c| c.get(cap_key))
                .and_then(|a| a.as_array())
                .is_some_and(|arr| {
                    arr.iter().any(|v| {
                        v.get("type")
                            .and_then(|t| t.as_str())
                            .is_some_and(|t| t == capability_domain)
                    })
                })
        };

        // First pass: exact `provided_capabilities[].type` match (strongest signal)
        let by_type = self
            .discovered
            .values()
            .find(|ep| ep.alive && has_typed_cap("provided_capabilities", ep));
        if by_type.is_some() {
            return by_type;
        }

        // Second pass: `capabilities[].type` match (toadStool format)
        let by_cap_obj = self
            .discovered
            .values()
            .find(|ep| ep.alive && has_typed_cap("capabilities", ep));
        if by_cap_obj.is_some() {
            return by_cap_obj;
        }

        // Third pass: `domains` array (barraCuda style)
        let by_domain = self.discovered.values().find(|ep| {
            ep.alive
                && ep
                    .capabilities
                    .as_ref()
                    .and_then(|c| c.get("domains"))
                    .and_then(|a| a.as_array())
                    .is_some_and(|arr| {
                        arr.iter()
                            .any(|v| v.as_str().is_some_and(|s| s == capability_domain))
                    })
        });
        if by_domain.is_some() {
            return by_domain;
        }

        // Legacy: flat `capabilities` array of strings (test harness format, prefix match)
        self.discovered.values().find(|ep| {
            ep.alive
                && ep
                    .capabilities
                    .as_ref()
                    .and_then(|c| c.get("capabilities"))
                    .and_then(|a| a.as_array())
                    .is_some_and(|arr| {
                        arr.iter()
                            .any(|v| v.as_str().is_some_and(|s| s.starts_with(capability_domain)))
                    })
        })
    }

    /// JSON-RPC call routed by capability domain (see [`Self::get_by_capability`]).
    ///
    /// When multiple primals match the same domain, prefers the one whose
    /// registered `methods` list contains the exact method being called.
    /// This disambiguates e.g. `shader.compile.wgsl` → coralReef (has method)
    /// vs toadStool (has `shader.dispatch` but not `shader.compile.wgsl`).
    pub fn call_by_capability(
        &self,
        capability_domain: &str,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, crate::error::HotSpringError> {
        let ep = self
            .get_by_exact_method(method)
            .or_else(|| self.get_by_capability(capability_domain))
            .ok_or_else(|| {
                crate::error::HotSpringError::Ipc(format!(
                    "no alive primal for capability domain: {capability_domain}"
                ))
            })?;
        self.call(&ep.name, method, &params)
    }

    /// Find the alive primal whose `methods` list contains the exact method name.
    fn get_by_exact_method(&self, method: &str) -> Option<&PrimalEndpoint> {
        self.discovered.values().find(|ep| {
            ep.alive
                && ep
                    .capabilities
                    .as_ref()
                    .and_then(|c| c.get("methods"))
                    .and_then(|a| a.as_array())
                    .is_some_and(|arr| arr.iter().any(|v| v.as_str().is_some_and(|s| s == method)))
        })
    }

    /// Send a JSON-RPC call to a specific primal.
    ///
    /// Retries once on connection reset (daemon restart during call).
    pub fn call(
        &self,
        primal: &str,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, crate::error::HotSpringError> {
        let ep = self.resolve_endpoint(primal).ok_or_else(|| {
            crate::error::HotSpringError::Ipc(format!("unknown primal: {primal}"))
        })?;

        if !ep.alive {
            return Err(crate::error::HotSpringError::Ipc(format!(
                "{primal} socket exists but health check failed"
            )));
        }

        let path = PathBuf::from(&ep.socket);
        match send_jsonrpc(&path, method, params) {
            Ok(v) => Ok(v),
            Err(crate::error::HotSpringError::Ipc(ref msg)) if is_retriable_ipc(msg) => {
                std::thread::sleep(std::time::Duration::from_secs(2));
                send_jsonrpc(&path, method, params)
            }
            Err(e) => Err(e),
        }
    }

    /// Like [`call`] but tracks failures for circuit-breaker semantics.
    ///
    /// After [`CIRCUIT_BREAKER_THRESHOLD`] consecutive failures, the endpoint
    /// is marked dead and will not be retried until the cooldown expires
    /// (at which point it's automatically re-probed).
    pub fn call_tracked(
        &mut self,
        primal: &str,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, crate::error::HotSpringError> {
        self.maybe_reprobe(primal);
        match self.call(primal, method, params) {
            Ok(v) => {
                self.record_success(primal);
                Ok(v)
            }
            Err(e) => {
                self.record_failure(primal);
                Err(e)
            }
        }
    }

    /// Generate a `composition.physics_health` response per `COMPOSITION_HEALTH_STANDARD.md`.
    #[must_use]
    pub fn physics_health(&self) -> serde_json::Value {
        let alive = self.alive_names();
        let compute_ready = self.get_by_capability("compute").is_some_and(|e| e.alive);
        let gpu_ready = self.get_by_capability("shader").is_some_and(|e| e.alive);
        let trio_ready = self.get_by_capability("dag").is_some_and(|e| e.alive)
            && self.get_by_capability("ledger").is_some_and(|e| e.alive)
            && self
                .get_by_capability("attribution")
                .is_some_and(|e| e.alive);

        let mut subsystems = serde_json::Map::new();
        for (name, ep) in self.all_endpoints() {
            let status = if ep.alive { "ok" } else { "unavailable" };
            subsystems.insert(name.to_string(), serde_json::json!(status));
        }

        serde_json::json!({
            "healthy": compute_ready,
            "deploy_graph": "physics_pipeline",
            "subsystems": subsystems,
            "compute_dispatch": compute_ready,
            "gpu_backend": gpu_ready,
            "provenance_trio": trio_ready,
            "primals_alive": alive.len(),
        })
    }
}

/// Build a JSON-RPC 2.0 request envelope.
pub fn jsonrpc_request(method: &str, params: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": JSONRPC_REQUEST_ID,
        "method": method,
        "params": params,
    })
}

/// Parse a JSON-RPC 2.0 response envelope.
///
/// Extracts the `result` value on success or returns a typed
/// `HotSpringError::Ipc` with code/message on JSON-RPC `error` objects.
/// `method_hint` is included in error messages for diagnostics.
pub fn parse_jsonrpc_response(
    resp: &serde_json::Value,
    method_hint: &str,
) -> Result<serde_json::Value, crate::error::HotSpringError> {
    if let Some(err) = resp.get("error") {
        let code = err
            .get("code")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(-1);
        let message = err
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown error");
        return Err(crate::error::HotSpringError::Ipc(format!(
            "{method_hint}: JSON-RPC error {code}: {message}"
        )));
    }
    resp.get("result").cloned().ok_or_else(|| {
        crate::error::HotSpringError::Ipc(format!(
            "{method_hint}: JSON-RPC response missing result"
        ))
    })
}

#[cfg(unix)]
fn collect_biomeos_socks(base: &Path, family: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(base) else {
        return out;
    };
    for e in entries.flatten() {
        let path = e.path();
        if path.extension().and_then(|s| s.to_str()) != Some("sock") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some((_primal, sock_family)) = name.rsplit_once('-') {
            // {primal}-{family}.sock — only include if family matches
            if sock_family == family {
                out.push(path);
            }
        } else {
            // {name}.sock (no family suffix) — e.g. compute.sock, math.sock
            // These are single-instance primal sockets; always include.
            out.push(path);
        }
    }
    out.sort();
    out
}

#[cfg(unix)]
fn probe_socket(path: &Path, logical_name: &str) -> PrimalEndpoint {
    let socket = path.to_string_lossy().into_owned();
    let alive = send_jsonrpc(path, "health.liveness", &serde_json::json!({}))
        .is_ok_and(|resp| resp.get("result").is_some());

    let capabilities = if alive {
        send_jsonrpc(path, "capability.list", &serde_json::json!({}))
            .ok()
            .and_then(|r| r.get("result").cloned())
    } else {
        None
    };

    PrimalEndpoint {
        name: logical_name.to_string(),
        socket,
        alive,
        capabilities,
        fail_count: 0,
        dead_since: None,
    }
}

/// Connection errors that indicate a daemon restart — worth one retry.
fn is_retriable_ipc(msg: &str) -> bool {
    msg.contains("connect:")
        || msg.contains("Connection reset")
        || msg.contains("Broken pipe")
        || msg.contains("Connection refused")
}

/// Send a JSON-RPC 2.0 request over a Unix domain socket with 2s timeout.
#[cfg(unix)]
pub fn send_jsonrpc(
    socket_path: &std::path::Path,
    method: &str,
    params: &serde_json::Value,
) -> Result<serde_json::Value, crate::error::HotSpringError> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket_path)
        .map_err(|e| crate::error::HotSpringError::Ipc(format!("connect: {e}")))?;

    stream
        .set_read_timeout(Some(JSONRPC_SOCKET_READ_TIMEOUT))
        .map_err(|e| crate::error::HotSpringError::Ipc(format!("timeout: {e}")))?;

    let request = jsonrpc_request(method, params.clone());

    let mut request_bytes = serde_json::to_vec(&request)?;
    request_bytes.push(b'\n');

    stream.write_all(&request_bytes)?;
    stream.flush()?;

    let mut response = Vec::new();
    let mut buf = [0u8; JSONRPC_READ_BUFFER_BYTES];
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
            Err(e) => return Err(e.into()),
        }
    }

    Ok(serde_json::from_slice(&response)?)
}

/// Compiled-in fallback aliases (used when `capability_registry.toml` is missing).
const DEFAULT_ALIASES: &[(&str, &[&str])] = &[
    (
        "toadstool",
        &["toadstool-server", "toadstool-glowplug", "compute"],
    ),
    (
        "coralreef",
        &[
            "coralreef-core",
            "coralreef-core-default",
            "coral-glowplug",
            "shader",
        ],
    ),
    ("barracuda", &["barracuda-core", "math"]),
];

/// Lazily-loaded alias table from `config/capability_registry.toml`.
fn loaded_aliases() -> &'static HashMap<String, Vec<String>> {
    use std::sync::OnceLock;
    static ALIASES: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    ALIASES.get_or_init(load_aliases_from_toml)
}

fn load_aliases_from_toml() -> HashMap<String, Vec<String>> {
    let candidates = [
        PathBuf::from("barracuda/config/capability_registry.toml"),
        PathBuf::from("config/capability_registry.toml"),
    ];
    for path in &candidates {
        if let Ok(contents) = std::fs::read_to_string(path) {
            if let Ok(table) = contents.parse::<toml::Table>() {
                if let Some(aliases_val) = table.get("primal_aliases") {
                    if let Some(tbl) = aliases_val.as_table() {
                        let mut map = HashMap::new();
                        for (k, v) in tbl {
                            if let Some(arr) = v.as_array() {
                                let names: Vec<String> = arr
                                    .iter()
                                    .filter_map(|s| s.as_str().map(String::from))
                                    .collect();
                                map.insert(k.clone(), names);
                            }
                        }
                        return map;
                    }
                }
            }
        }
    }
    let mut map = HashMap::new();
    for &(k, aliases) in DEFAULT_ALIASES {
        map.insert(
            k.to_string(),
            aliases.iter().map(|s| (*s).to_string()).collect(),
        );
    }
    map
}

fn known_aliases(name: &str) -> Vec<&str> {
    loaded_aliases()
        .get(name)
        .map(|v| v.iter().map(String::as_str).collect())
        .unwrap_or_default()
}

#[cfg(not(unix))]
pub fn send_jsonrpc(
    _socket_path: &std::path::Path,
    _method: &str,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, crate::error::HotSpringError> {
    Err(crate::error::HotSpringError::Ipc(
        "Unix socket IPC not available on this platform".into(),
    ))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "test assertions")]
    use super::*;

    #[test]
    fn empty_context_has_no_discovered_primals() {
        let ctx = NucleusContext::empty("test-family");
        assert!(ctx.discovered.is_empty());
        assert_eq!(ctx.family_id, "test-family");
    }

    #[test]
    fn by_domain_returns_none_on_empty_context() {
        let ctx = NucleusContext::empty("fam");
        assert!(ctx.by_domain("compute").is_none());
    }

    #[test]
    fn alive_names_empty_on_empty_context() {
        let ctx = NucleusContext::empty("fam");
        assert!(ctx.alive_names().is_empty());
    }

    #[test]
    fn get_unknown_name_returns_none() {
        let ctx = NucleusContext::empty("fam");
        assert!(ctx.get("no-such-primal").is_none());
    }

    #[test]
    fn by_domain_finds_manually_constructed_endpoint_with_capabilities() {
        let caps = serde_json::json!({
            "capabilities": ["compute.batch", "compute.other"],
        });
        let ep = PrimalEndpoint {
            name: "toad".to_string(),
            socket: "/tmp/toad-test.sock".to_string(),
            alive: true,
            capabilities: Some(caps),
            fail_count: 0,
            dead_since: None,
        };
        let mut discovered = HashMap::new();
        let socket = ep.socket.clone();
        discovered.insert("toad".to_string(), ep);
        let ctx = NucleusContext {
            discovered,
            family_id: "fam".to_string(),
        };
        let found = ctx.by_domain("compute").expect("compute primal");
        assert_eq!(found.name, "toad");
        assert_eq!(found.socket, socket);
        assert!(found.alive);
    }

    #[test]
    fn get_by_capability_matches_domain_prefix_in_capabilities_list() {
        let caps = serde_json::json!({
            "capabilities": ["crypto.sign_ed25519"],
        });
        let ep = PrimalEndpoint {
            name: "bear".to_string(),
            socket: "/tmp/bear-test.sock".to_string(),
            alive: true,
            capabilities: Some(caps),
            fail_count: 0,
            dead_since: None,
        };
        let mut discovered = HashMap::new();
        discovered.insert("bear".to_string(), ep);
        let ctx = NucleusContext {
            discovered,
            family_id: "fam".to_string(),
        };
        let found = ctx.get_by_capability("crypto").expect("crypto capability");
        assert_eq!(found.name, "bear");
    }
}
