// SPDX-License-Identifier: AGPL-3.0-only

//! NUCLEUS primal discovery — runtime detection of available primals.
//!
//! Probes wateringHole IPC v3.1 Unix sockets to discover which primals
//! are running. Degrades gracefully: when no primals are present,
//! everything works standalone (bare guideStone).
//!
//! Set `HOTSPRING_NO_NUCLEUS=1` to skip all primal detection (useful
//! for headless HPC or isolated benchmarks).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A detected primal with its socket path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalEndpoint {
    pub name: String,
    pub socket: String,
    pub alive: bool,
}

/// Runtime snapshot of which NUCLEUS primals are reachable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NucleusContext {
    pub toadstool: Option<PrimalEndpoint>,
    pub beardog: Option<PrimalEndpoint>,
    pub rhizocrypt: Option<PrimalEndpoint>,
    pub loamspine: Option<PrimalEndpoint>,
    pub sweetgrass: Option<PrimalEndpoint>,
    pub family_id: String,
}

impl NucleusContext {
    /// Probe all known primal sockets and build a context.
    ///
    /// Returns an empty context (all `None`) when `HOTSPRING_NO_NUCLEUS=1`
    /// is set or when running on non-Unix platforms.
    #[must_use]
    pub fn detect() -> Self {
        let family = std::env::var("FAMILY_ID").unwrap_or_else(|_| "default".into());

        if std::env::var("HOTSPRING_NO_NUCLEUS").map_or(false, |v| v == "1") {
            return Self::empty(&family);
        }

        let runtime_dir =
            std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
        let base = format!("{runtime_dir}/biomeos");

        let primals = [
            ("toadstool", format!("{base}/toadstool-{family}.sock"), "compute.health"),
            ("beardog", format!("{base}/beardog-{family}.sock"), "health.liveness"),
            ("rhizocrypt", format!("{base}/rhizocrypt-{family}.sock"), "dag.health"),
            ("loamspine", format!("{base}/loamspine-{family}.sock"), "commit.health"),
            ("sweetgrass", format!("{base}/sweetgrass-{family}.sock"), "provenance.health"),
        ];

        let mut ctx = Self::empty(&family);

        for (name, socket, method) in &primals {
            if let Some(ep) = probe_primal(name, socket, method) {
                match *name {
                    "toadstool" => ctx.toadstool = Some(ep),
                    "beardog" => ctx.beardog = Some(ep),
                    "rhizocrypt" => ctx.rhizocrypt = Some(ep),
                    "loamspine" => ctx.loamspine = Some(ep),
                    "sweetgrass" => ctx.sweetgrass = Some(ep),
                    _ => {}
                }
            }
        }

        ctx
    }

    fn empty(family: &str) -> Self {
        Self {
            toadstool: None,
            beardog: None,
            rhizocrypt: None,
            loamspine: None,
            sweetgrass: None,
            family_id: family.to_string(),
        }
    }

    /// Names of all alive primals (for banner / manifest).
    #[must_use]
    pub fn alive_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for (name, ep) in self.all_endpoints() {
            if let Some(e) = ep {
                if e.alive {
                    names.push(name);
                }
            }
        }
        names
    }

    /// True when at least one primal is reachable.
    #[must_use]
    pub fn any_alive(&self) -> bool {
        !self.alive_names().is_empty()
    }

    fn all_endpoints(&self) -> Vec<(&str, &Option<PrimalEndpoint>)> {
        vec![
            ("toadstool", &self.toadstool),
            ("beardog", &self.beardog),
            ("rhizocrypt", &self.rhizocrypt),
            ("loamspine", &self.loamspine),
            ("sweetgrass", &self.sweetgrass),
        ]
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
        let ep = match primal {
            "toadstool" => &self.toadstool,
            "beardog" => &self.beardog,
            "rhizocrypt" => &self.rhizocrypt,
            "loamspine" => &self.loamspine,
            "sweetgrass" => &self.sweetgrass,
            _ => return Err(format!("unknown primal: {primal}")),
        };

        let ep = ep.as_ref().ok_or_else(|| format!("{primal} not available"))?;
        if !ep.alive {
            return Err(format!("{primal} socket exists but health check failed"));
        }

        send_jsonrpc(&PathBuf::from(&ep.socket), method, params)
    }
}

fn probe_primal(name: &str, socket: &str, health_method: &str) -> Option<PrimalEndpoint> {
    let path = PathBuf::from(socket);
    if !path.exists() {
        return None;
    }

    let alive = send_jsonrpc(&path, health_method, &serde_json::json!({}))
        .map(|resp| resp.get("result").is_some())
        .unwrap_or(false);

    Some(PrimalEndpoint {
        name: name.to_string(),
        socket: socket.to_string(),
        alive,
    })
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
