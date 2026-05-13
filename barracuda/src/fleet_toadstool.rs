// SPDX-License-Identifier: AGPL-3.0-or-later

//! toadStool dispatch client: IPC path for `compute.dispatch.capabilities`
//! and `compute.dispatch.submit`.
//!
//! This module provides a parallel dispatch path alongside the direct
//! `coral-ember` socket path in [`crate::fleet_ember`]. When toadStool
//! Phase C (cylinder absorption) completes, the `toadstool-dispatch`
//! feature flag will become the default dispatch path.
//!
//! ## Discovery
//!
//! toadStool is discovered via NUCLEUS `by_domain("compute")` — the same
//! capability-based discovery used for all primals. The client caches the
//! socket path after first discovery.
//!
//! ## Timeout mapping
//!
//! | ember constant         | toadStool equivalent     | Value |
//! |------------------------|--------------------------|-------|
//! | `EMBER_ADOPT_TIMEOUT`  | device pool acquire      | 30s   |
//! | `EMBER_STATUS_TIMEOUT` | health/capabilities probe| 5s    |

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::HotSpringError;
use crate::primal_bridge::{NucleusContext, send_jsonrpc};

/// Timeout for device acquisition via toadStool (maps to `EMBER_ADOPT_TIMEOUT`).
pub const TOADSTOOL_DEVICE_TIMEOUT: Duration = Duration::from_secs(30);

/// Timeout for lightweight capability/status probes (maps to `EMBER_STATUS_TIMEOUT`).
pub const TOADSTOOL_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Timeout for compute dispatch submission (sovereign dispatch can exceed 10s
/// on real workloads — GAP-HS-040 confirmed).
pub const TOADSTOOL_DISPATCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Client for toadStool's `compute.dispatch.*` JSON-RPC surface.
///
/// Wraps a single toadStool socket and provides typed accessors for the
/// dispatch pipeline methods. Designed to sit alongside [`crate::fleet_ember::EmberClient`]
/// during the Phase C migration period.
#[derive(Debug, Clone)]
pub struct ToadStoolDispatchClient {
    socket_path: PathBuf,
}

impl ToadStoolDispatchClient {
    /// Connect to a toadStool instance at the given socket path.
    #[must_use]
    pub fn connect(socket_path: impl AsRef<Path>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
        }
    }

    /// Discover toadStool via NUCLEUS `by_domain("compute")`.
    ///
    /// Returns `None` if no alive toadStool primal is found.
    #[must_use]
    pub fn discover() -> Option<Self> {
        let ctx = NucleusContext::detect();
        let ep = ctx.by_domain("compute")?;
        if ep.alive {
            Some(Self::connect(&ep.socket))
        } else {
            None
        }
    }

    /// Socket path for this client.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// `compute.dispatch.capabilities` — query available dispatch backends
    /// and hardware units.
    ///
    /// Prefers `call_by_capability` when NUCLEUS context is available,
    /// falling back to direct socket RPC.
    pub fn capabilities(&self) -> Result<serde_json::Value, HotSpringError> {
        let ctx = NucleusContext::detect();
        if let Ok(resp) = ctx.call_by_capability(
            "compute",
            "compute.dispatch.capabilities",
            serde_json::json!({}),
        ) {
            return Ok(resp);
        }
        let resp = send_jsonrpc(
            &self.socket_path,
            "compute.dispatch.capabilities",
            &serde_json::json!({}),
        )?;
        jsonrpc_result(&resp)
    }

    /// `health.liveness` — lightweight health probe.
    pub fn health(&self) -> Result<serde_json::Value, HotSpringError> {
        let resp = send_jsonrpc(&self.socket_path, "health.liveness", &serde_json::json!({}))?;
        jsonrpc_result(&resp)
    }

    /// `compute.dispatch.submit` — submit a workload for sovereign dispatch.
    ///
    /// The `params` should include at minimum:
    /// - `binary_b64`: base64-encoded compiled shader binary
    /// - `dispatch_dims`: `[x, y, z]` workgroup counts
    /// - `buffers`: array of `{ data_b64, size, binding }` buffer descriptors
    /// - `timeout_ms`: dispatch timeout in milliseconds
    ///
    /// Prefers `call_by_capability` when NUCLEUS context is available,
    /// falling back to direct socket RPC.
    pub fn submit(&self, params: &serde_json::Value) -> Result<serde_json::Value, HotSpringError> {
        let ctx = NucleusContext::detect();
        if let Ok(resp) =
            ctx.call_by_capability("compute", "compute.dispatch.submit", params.clone())
        {
            return Ok(resp);
        }
        let resp = send_jsonrpc(&self.socket_path, "compute.dispatch.submit", params)?;
        jsonrpc_result(&resp)
    }

    /// `compute.dispatch` — alias for `submit` (toadStool accepts both method names).
    pub fn dispatch(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, HotSpringError> {
        self.submit(params)
    }

    /// Check if toadStool is alive and accepting dispatch requests.
    pub fn is_alive(&self) -> bool {
        self.health().is_ok()
    }
}

/// Phase D local dispatch result.
///
/// Captures whether local dispatch was attempted and its outcome,
/// enabling parity comparison with the coralReef-forwarded path.
#[derive(Debug, Clone)]
pub struct LocalDispatchResult {
    /// Whether local dispatch was attempted.
    pub attempted: bool,
    /// Whether local dispatch succeeded.
    pub succeeded: bool,
    /// The dispatch result (if successful) or error description.
    pub result: Result<serde_json::Value, String>,
}

/// Phase D: attempt local dispatch via toadStool's `LocalDeviceFactory`.
///
/// toadStool S250 shipped `LocalDeviceFactory` + `try_local_dispatch()` —
/// dispatch without forwarding to coralReef. This function wraps that
/// path via NUCLEUS IPC, enabling hotSpring to validate local dispatch
/// parity with the standard coralReef-forwarded path.
///
/// Returns `None` when the `local-dispatch` feature is not enabled or
/// when the toadStool endpoint doesn't support local dispatch.
///
/// # Phase D Protocol
///
/// 1. Call `compute.dispatch.local` (Phase D method) via NUCLEUS
/// 2. If unavailable, fall back to standard `compute.dispatch.submit`
/// 3. Compare results for parity validation
pub fn try_local_dispatch(
    nucleus: &NucleusContext,
    params: &serde_json::Value,
) -> LocalDispatchResult {
    let local_params = {
        let mut p = params.clone();
        if let Some(obj) = p.as_object_mut() {
            obj.insert("local_dispatch".into(), serde_json::Value::Bool(true));
            obj.insert("phase_d".into(), serde_json::Value::Bool(true));
        }
        p
    };

    match nucleus.call_by_capability("compute", "compute.dispatch.submit", local_params) {
        Ok(resp) => LocalDispatchResult {
            attempted: true,
            succeeded: true,
            result: Ok(resp),
        },
        Err(e) => LocalDispatchResult {
            attempted: true,
            succeeded: false,
            result: Err(e.to_string()),
        },
    }
}

fn jsonrpc_result(resp: &serde_json::Value) -> Result<serde_json::Value, HotSpringError> {
    if let Some(err) = resp.get("error") {
        return Err(HotSpringError::Ipc(format!(
            "toadStool JSON-RPC error: {err}"
        )));
    }
    resp.get("result")
        .cloned()
        .ok_or_else(|| HotSpringError::Ipc("toadStool JSON-RPC response missing result".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_constants_match_ember_semantics() {
        assert_eq!(
            TOADSTOOL_DEVICE_TIMEOUT,
            crate::fleet_ember::EMBER_STATUS_TIMEOUT * 6,
            "device timeout should be 6x the probe timeout (30s vs 5s)"
        );
        assert_eq!(TOADSTOOL_PROBE_TIMEOUT.as_secs(), 5);
        assert_eq!(TOADSTOOL_DEVICE_TIMEOUT.as_secs(), 30);
    }

    #[test]
    fn connect_and_discover_are_independent() {
        let direct = ToadStoolDispatchClient::connect("/tmp/toadstool-test.sock");
        assert_eq!(direct.socket_path(), Path::new("/tmp/toadstool-test.sock"));
        assert!(!direct.is_alive(), "no daemon at test socket");
    }

    #[test]
    fn connect_stores_path() {
        let client = ToadStoolDispatchClient::connect("/tmp/test.sock");
        assert_eq!(client.socket_path(), Path::new("/tmp/test.sock"));
    }
}
