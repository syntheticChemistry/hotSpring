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

/// A detected primal with its socket path and optional capability payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalEndpoint {
    pub name: String,
    pub socket: String,
    pub alive: bool,
    /// Result of `capability.list` when the primal supports it and liveness passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<serde_json::Value>,
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
                    let Some((primal_name, _sock_family)) = stem.rsplit_once('-') else {
                        continue;
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

    fn empty(family: &str) -> Self {
        Self {
            discovered: HashMap::new(),
            family_id: family.to_string(),
        }
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
        // coral-glowplug is the fleet orchestrator that also serves shader capabilities;
        // fall through when the direct name isn't found.
        for &alias in known_aliases(primal) {
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
    pub fn call_by_capability(
        &self,
        capability_domain: &str,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, crate::error::HotSpringError> {
        let ep = self.get_by_capability(capability_domain).ok_or_else(|| {
            crate::error::HotSpringError::Ipc(format!(
                "no alive primal for capability domain: {capability_domain}"
            ))
        })?;
        self.call(&ep.name, method, &params)
    }

    /// Send a JSON-RPC call to a specific primal.
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

        send_jsonrpc(&PathBuf::from(&ep.socket), method, params)
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
        let Some((_primal, sock_family)) = name.rsplit_once('-') else {
            continue;
        };
        if sock_family != family {
            continue;
        }
        out.push(path);
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
    }
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

/// Known socket-name aliases for primals that may register under alternative names.
/// Data-driven: no hardcoded if/else chains.
const PRIMAL_ALIASES: &[(&str, &[&str])] = &[
    (
        "toadstool",
        &["toadstool-server", "toadstool-glowplug", "compute"],
    ),
    ("coralreef", &["coral-glowplug", "shader"]),
    ("barracuda", &["barracuda-core", "math"]),
];

fn known_aliases(name: &str) -> &'static [&'static str] {
    PRIMAL_ALIASES
        .iter()
        .find(|(k, _)| *k == name)
        .map_or(&[], |(_, aliases)| aliases)
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
