// SPDX-License-Identifier: AGPL-3.0-or-later

//! skunkBat `security.audit_log` IPC client.
//!
//! Provides cursor-based audit event polling from the skunkBat primal via
//! JSON-RPC. When Phase 3 ships, audit events forwarded here automatically
//! propagate to rhizoCrypt DAG + sweetGrass braid.

use crate::primal_bridge::send_jsonrpc;
use serde::{Deserialize, Serialize};

/// A single audit event from the skunkBat audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Monotonic sequence number within the audit log.
    pub seq: u64,
    /// ISO-8601 timestamp of the event.
    pub timestamp: String,
    /// Event category (e.g. "auth", "access", "anomaly").
    pub category: String,
    /// Human-readable description.
    pub message: String,
    /// Severity level (info/warn/error/critical).
    #[serde(default = "default_severity")]
    pub severity: String,
    /// Originating primal or spring, if known.
    #[serde(default)]
    pub source: Option<String>,
}

fn default_severity() -> String {
    "info".into()
}

/// Response from `security.audit_log`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogResponse {
    /// Audit events matching the query.
    pub events: Vec<AuditEvent>,
    /// Latest sequence number in the full audit log.
    pub latest_seq: u64,
    /// Number of events returned.
    pub count: usize,
}

impl AuditLogResponse {
    /// Whether the caller is fully caught up with the audit stream.
    #[must_use]
    pub fn is_current(&self) -> bool {
        self.events.last().is_none_or(|e| e.seq >= self.latest_seq)
    }
}

/// Query skunkBat `security.audit_log` via JSON-RPC.
///
/// `since_seq` is a cursor — events with `seq > since_seq` are returned.
/// `limit` caps the number of events (server max: 1000).
///
/// Returns `None` if skunkBat is unreachable or the socket does not exist.
pub fn query_audit_log(since_seq: u64, limit: u64) -> Option<AuditLogResponse> {
    let socket = crate::niche::socket_dirs()
        .into_iter()
        .map(|d| d.join("skunkbat/skunkbat.sock"))
        .find(|p| p.exists())?;

    let params = serde_json::json!({
        "since_seq": since_seq,
        "limit": limit.min(1000),
    });
    let response = send_jsonrpc(&socket, "security.audit_log", &params).ok()?;

    serde_json::from_value(response).ok()
}

/// Convenience: query the latest audit events (up to 100, from the beginning).
pub fn query_latest_audit(limit: u64) -> Option<AuditLogResponse> {
    query_audit_log(0, limit)
}

/// Record audit log reachability as validation checks on a harness.
pub fn check_audit_log(v: &mut crate::validation::ValidationHarness) {
    match query_audit_log(0, 1) {
        Some(resp) => {
            v.check_bool("security.audit_log:reachable", true);
            v.check_bool(
                "security.audit_log:has_events",
                resp.count > 0 || resp.latest_seq > 0,
            );
        }
        None => {
            v.check_bool("security.audit_log:reachable (skunkBat not running)", false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_event_deserialize() {
        let json = serde_json::json!({
            "seq": 42,
            "timestamp": "2026-05-11T00:00:00Z",
            "category": "auth",
            "message": "peer authenticated",
            "severity": "info",
            "source": "hotSpring"
        });
        let event: AuditEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.seq, 42);
        assert_eq!(event.category, "auth");
        assert_eq!(event.source.as_deref(), Some("hotSpring"));
    }

    #[test]
    fn audit_event_default_severity() {
        let json = serde_json::json!({
            "seq": 1,
            "timestamp": "2026-05-11T00:00:00Z",
            "category": "access",
            "message": "resource accessed"
        });
        let event: AuditEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.severity, "info");
        assert!(event.source.is_none());
    }

    #[test]
    fn audit_log_response_is_current() {
        let resp = AuditLogResponse {
            events: vec![AuditEvent {
                seq: 100,
                timestamp: "2026-05-11T00:00:00Z".into(),
                category: "test".into(),
                message: "test event".into(),
                severity: "info".into(),
                source: None,
            }],
            latest_seq: 100,
            count: 1,
        };
        assert!(resp.is_current());
    }

    #[test]
    fn audit_log_response_not_current() {
        let resp = AuditLogResponse {
            events: vec![AuditEvent {
                seq: 50,
                timestamp: "2026-05-11T00:00:00Z".into(),
                category: "test".into(),
                message: "test event".into(),
                severity: "info".into(),
                source: None,
            }],
            latest_seq: 100,
            count: 1,
        };
        assert!(!resp.is_current());
    }

    #[test]
    fn empty_response_is_current() {
        let resp = AuditLogResponse {
            events: vec![],
            latest_seq: 0,
            count: 0,
        };
        assert!(resp.is_current());
    }

    #[test]
    fn query_returns_none_when_skunkbat_not_running() {
        assert!(query_audit_log(0, 10).is_none());
    }
}
