// SPDX-License-Identifier: AGPL-3.0-or-later

//! biomeOS `composition.status` IPC client (v3.51).
//!
//! Queries the local biomeOS instance for composition health metrics:
//! `{ active_users, primal_health, resource_pressure }`.
//!
//! Used by hotSpring's monitoring and health paths to report NUCLEUS-level
//! composition state alongside spring-local validation results.

use crate::primal_bridge::{NucleusContext, send_jsonrpc};
use serde::{Deserialize, Serialize};

/// Composition status response from biomeOS v3.51 `composition.status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionStatus {
    /// Number of active users/sessions on this NUCLEUS instance.
    pub active_users: u32,
    /// Aggregate primal health (0.0 = unhealthy, 1.0 = fully healthy).
    pub primal_health: f64,
    /// Resource pressure indicator (0.0 = idle, 1.0 = saturated).
    pub resource_pressure: f64,
}

impl CompositionStatus {
    /// Whether the composition is in a healthy state for accepting work.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.primal_health > 0.5 && self.resource_pressure < 0.9
    }
}

/// Query biomeOS `composition.status` via JSON-RPC.
///
/// Prefers `call_by_capability("composition", ...)` for fully capability-based
/// transport. Falls back to `BIOMEOS_SOCKET` env var, then conventional
/// socket-dir scanning for CI/lab environments where NUCLEUS discovery
/// is not running.
///
/// Returns `None` if biomeOS is unreachable or the method is not available
/// (pre-v3.51 instances).
pub fn query_composition_status() -> Option<CompositionStatus> {
    let nucleus = NucleusContext::detect();
    let params = serde_json::json!({});

    if let Ok(resp) = nucleus.call_by_capability("composition", "composition.status", params.clone())
    {
        return serde_json::from_value(resp).ok();
    }

    let socket: std::path::PathBuf = if let Ok(p) = std::env::var("BIOMEOS_SOCKET") {
        let path = std::path::PathBuf::from(p);
        if path.exists() { path } else { return None; }
    } else {
        crate::niche::socket_dirs()
            .into_iter()
            .map(|d| d.join("biomeos/biomeos.sock"))
            .find(|p| p.exists())?
    };

    let response = send_jsonrpc(&socket, "composition.status", &params).ok()?;
    serde_json::from_value(response).ok()
}

/// Record composition status as validation checks on a harness.
pub fn check_composition_status(v: &mut crate::validation::ValidationHarness) {
    match query_composition_status() {
        Some(status) => {
            v.check_bool("composition.status:reachable", true);
            v.check_bool("composition.status:healthy", status.is_healthy());
            v.check_lower(
                "composition.status:primal_health",
                status.primal_health,
                0.5,
            );
            v.check_upper(
                "composition.status:resource_pressure",
                status.resource_pressure,
                0.9,
            );
        }
        None => {
            v.check_bool("composition.status:reachable (biomeOS not running)", false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_status_healthy_when_normal() {
        let status = CompositionStatus {
            active_users: 3,
            primal_health: 0.95,
            resource_pressure: 0.2,
        };
        assert!(status.is_healthy());
    }

    #[test]
    fn composition_status_unhealthy_when_degraded() {
        let status = CompositionStatus {
            active_users: 0,
            primal_health: 0.3,
            resource_pressure: 0.1,
        };
        assert!(!status.is_healthy());
    }

    #[test]
    fn composition_status_unhealthy_under_pressure() {
        let status = CompositionStatus {
            active_users: 10,
            primal_health: 0.9,
            resource_pressure: 0.95,
        };
        assert!(!status.is_healthy());
    }

    #[test]
    fn query_returns_none_when_biomeos_not_running() {
        assert!(query_composition_status().is_none());
    }
}
