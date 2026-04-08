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
        let family = std::env::var("FAMILY_ID").unwrap_or_else(|_| "default".into());

        if std::env::var("HOTSPRING_NO_NUCLEUS").map_or(false, |v| v == "1") {
            return Self::empty(&family);
        }

        #[cfg(not(unix))]
        {
            return Self::empty(&family);
        }

        #[cfg(unix)]
        {
            let runtime_dir =
                std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
            let base = format!("{runtime_dir}/biomeos");

            let mut discovered = HashMap::new();
            for path in collect_biomeos_socks(Path::new(&base), &family) {
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let Some((primal_name, _sock_family)) = stem.rsplit_once('-') else {
                    continue;
                };
                let ep = probe_socket(&path, primal_name);
                discovered.insert(primal_name.to_string(), ep);
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
        if primal == "coralreef" {
            return self.discovered.get("coral-glowplug");
        }
        None
    }

    /// Convenience: toadStool compute primal.
    #[must_use]
    pub fn toadstool(&self) -> Option<&PrimalEndpoint> {
        self.discovered.get("toadstool")
    }

    /// Convenience: bearDog signing primal.
    #[must_use]
    pub fn beardog(&self) -> Option<&PrimalEndpoint> {
        self.discovered.get("beardog")
    }

    /// Convenience: rhizoCrypt DAG primal.
    #[must_use]
    pub fn rhizocrypt(&self) -> Option<&PrimalEndpoint> {
        self.discovered.get("rhizocrypt")
    }

    /// Convenience: loamSpine commit primal.
    #[must_use]
    pub fn loamspine(&self) -> Option<&PrimalEndpoint> {
        self.discovered.get("loamspine")
    }

    /// Convenience: sweetgrass provenance primal.
    #[must_use]
    pub fn sweetgrass(&self) -> Option<&PrimalEndpoint> {
        self.discovered.get("sweetgrass")
    }

    /// Convenience: coralReef / coral-glowplug GPU sovereign path (glowplug socket naming).
    #[must_use]
    pub fn coralreef(&self) -> Option<&PrimalEndpoint> {
        self.discovered
            .get("coralreef")
            .or_else(|| self.discovered.get("coral-glowplug"))
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
            println!("  NUCLEUS: {} primal(s) — {}", names.len(), names.join(", "));
        }
    }

    /// Send a JSON-RPC call to a specific primal.
    pub fn call(
        &self,
        primal: &str,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let ep = self
            .resolve_endpoint(primal)
            .ok_or_else(|| format!("unknown primal: {primal}"))?;

        if !ep.alive {
            return Err(format!("{primal} socket exists but health check failed"));
        }

        send_jsonrpc(&PathBuf::from(&ep.socket), method, params)
    }

    /// Generate a `composition.physics_health` response per `COMPOSITION_HEALTH_STANDARD.md`.
    #[must_use]
    pub fn physics_health(&self) -> serde_json::Value {
        let alive = self.alive_names();
        let compute_ready = self.toadstool().is_some_and(|e| e.alive);
        let gpu_ready = self.coralreef().is_some_and(|e| e.alive);
        let trio_ready = self.rhizocrypt().is_some_and(|e| e.alive)
            && self.loamspine().is_some_and(|e| e.alive)
            && self.sweetgrass().is_some_and(|e| e.alive);

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
        .map(|resp| resp.get("result").is_some())
        .unwrap_or(false);

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
) -> Result<serde_json::Value, String> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(socket_path).map_err(|e| format!("connect: {e}"))?;

    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
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
pub fn send_jsonrpc(
    _socket_path: &std::path::Path,
    _method: &str,
    _params: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    Err("Unix socket IPC not available on this platform".into())
}
