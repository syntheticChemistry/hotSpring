// SPDX-License-Identifier: AGPL-3.0-or-later

//! Typed response structs for coral-ember IPC (JSON-RPC 2.0).
//!
//! Maps the JSON payloads returned by ember's fork-isolated MMIO, falcon,
//! SEC2, PRAMIN, and DMA handlers into Rust types for use by experiment
//! binaries and the [`crate::fleet_client::EmberClient`].

use serde::{Deserialize, Serialize};

// ── MMIO ────────────────────────────────────────────────────────────────

/// Result of `ember.mmio.read`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MmioReadResult {
    pub value: u32,
}

/// Result of `ember.mmio.write`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MmioWriteResult {
    pub ok: bool,
}

/// A single operation within an `ember.mmio.batch` request.
#[derive(Debug, Clone, Serialize)]
pub struct MmioBatchOp {
    /// `"r"` for read, `"w"` for write.
    #[serde(rename = "type")]
    pub op_type: String,
    pub offset: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<u32>,
}

impl MmioBatchOp {
    #[must_use]
    pub fn read(offset: u32) -> Self {
        Self {
            op_type: "r".into(),
            offset,
            value: None,
        }
    }

    #[must_use]
    pub fn write(offset: u32, value: u32) -> Self {
        Self {
            op_type: "w".into(),
            offset,
            value: Some(value),
        }
    }
}

/// Full result of `ember.mmio.batch`.
///
/// Ember returns a heterogeneous array: raw `u32` for reads, `true` for writes,
/// `{"error": "..."}` for failures. Use [`MmioBatchResult::read_value`] to
/// extract read results by index.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MmioBatchResult {
    pub results: Vec<serde_json::Value>,
}

impl MmioBatchResult {
    /// Extract the u32 read value at `index`, or `None` if the entry is a write
    /// result, error, or out of bounds.
    #[must_use]
    pub fn read_value(&self, index: usize) -> Option<u32> {
        self.results.get(index)?.as_u64().map(|v| v as u32)
    }

    /// Check if the entry at `index` is an error object.
    #[must_use]
    pub fn is_error(&self, index: usize) -> bool {
        self.results
            .get(index)
            .and_then(|v| v.as_object())
            .and_then(|o| o.get("error"))
            .is_some()
    }

    /// Number of entries in the batch result.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// True if the batch returned no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

/// Result of `ember.mmio.circuit_breaker`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CircuitBreakerStatus {
    pub bdf: String,
    pub tripped: Option<bool>,
    pub fault_count: Option<u32>,
    pub consecutive_faults: Option<u32>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

// ── Falcon ──────────────────────────────────────────────────────────────

/// Result of `ember.falcon.upload_imem`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FalconUploadResult {
    pub ok: bool,
    pub bytes: Option<usize>,
}

/// Result of `ember.falcon.start_cpu`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FalconStartResult {
    pub ok: bool,
    pub pc: Option<u32>,
    pub exci: Option<u32>,
    pub cpuctl: Option<u32>,
}

/// A snapshot during `ember.falcon.poll`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FalconPollSnapshot {
    pub pc: Option<u32>,
    pub cpuctl: Option<u32>,
    pub mailbox0: Option<u32>,
    pub mailbox1: Option<u32>,
    pub sctl: Option<u32>,
    pub elapsed_ms: Option<u64>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Result of `ember.falcon.poll`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FalconPollResult {
    pub snapshots: Vec<FalconPollSnapshot>,
    pub pc_trace: Vec<u32>,
    #[serde(rename = "final")]
    pub final_state: Option<FalconPollSnapshot>,
}

// ── SEC2 ────────────────────────────────────────────────────────────────

/// Result of `ember.sec2.prepare_physical`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Sec2PrepareResult {
    pub ok: bool,
    #[serde(default)]
    pub notes: Vec<String>,
}

// ── PRAMIN ──────────────────────────────────────────────────────────────

/// Result of `ember.pramin.read`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PraminReadResult {
    pub data_b64: String,
    pub length: usize,
}

/// Result of `ember.pramin.write`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PraminWriteResult {
    pub ok: bool,
    pub bytes_written: usize,
}

// ── DMA ─────────────────────────────────────────────────────────────────

/// Result of `ember.prepare_dma`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DmaPrepareResult {
    pub bdf: String,
    pub ok: bool,
    pub pmc_before: Option<String>,
    pub pmc_after: Option<String>,
}

/// Result of `ember.cleanup_dma`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DmaCleanupResult {
    pub bdf: String,
    pub ok: bool,
    pub decontaminated: Option<bool>,
    pub decontaminate_result: Option<String>,
}

// ── Glowplug device lifecycle ───────────────────────────────────────────

/// Result of `device.register_dump` / `device.register_snapshot`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GlowplugRegisterDumpResult {
    pub bdf: String,
    pub registers: Vec<GlowplugRegisterEntry>,
    pub timestamp: Option<String>,
}

/// One register in a glowplug dump.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GlowplugRegisterEntry {
    pub offset: u32,
    pub name: Option<String>,
    pub value: u32,
}

/// Result of `device.read_bar0_range`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Bar0RangeResult {
    pub bdf: String,
    pub start: u32,
    pub length: u32,
    pub data_b64: String,
}

/// Result of `device.experiment_start` / `device.experiment_end`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExperimentLifecycleResult {
    #[serde(default)]
    pub bdf: Option<String>,
    #[serde(default)]
    pub ok: Option<bool>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

/// Result of `device.reset` / `device.resurrect`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeviceLifecycleResult {
    #[serde(default)]
    pub bdf: Option<String>,
    #[serde(default)]
    pub ok: Option<bool>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmio_read_result_deserializes() {
        let json = r#"{"value": 3735928559}"#;
        let r: MmioReadResult = serde_json::from_str(json).expect("parse");
        assert_eq!(r.value, 0xDEAD_BEEF);
    }

    #[test]
    fn mmio_batch_op_serializes_read() {
        let op = MmioBatchOp::read(0x000000);
        let json = serde_json::to_string(&op).expect("serialize");
        assert!(json.contains(r#""type":"r""#));
        assert!(!json.contains("value"));
    }

    #[test]
    fn mmio_batch_op_serializes_write() {
        let op = MmioBatchOp::write(0x000200, 0xFF);
        let json = serde_json::to_string(&op).expect("serialize");
        assert!(json.contains(r#""type":"w""#));
        assert!(json.contains("255"));
    }

    #[test]
    fn falcon_start_result_deserializes() {
        let json = r#"{"ok": true, "pc": 86, "exci": 0, "cpuctl": 2}"#;
        let r: FalconStartResult = serde_json::from_str(json).expect("parse");
        assert!(r.ok);
        assert_eq!(r.pc, Some(86));
    }

    #[test]
    fn falcon_poll_result_deserializes_with_snapshots() {
        let json = r#"{"snapshots": [{"pc": 0, "elapsed_ms": 10}], "pc_trace": [0, 86], "final": {"pc": 86}}"#;
        let r: FalconPollResult = serde_json::from_str(json).expect("parse");
        assert_eq!(r.snapshots.len(), 1);
        assert_eq!(r.pc_trace, vec![0, 86]);
    }

    #[test]
    fn sec2_prepare_result_deserializes() {
        let json = r#"{"ok": true, "notes": ["reset ok", "bind complete"]}"#;
        let r: Sec2PrepareResult = serde_json::from_str(json).expect("parse");
        assert!(r.ok);
        assert_eq!(r.notes.len(), 2);
    }

    #[test]
    fn dma_prepare_result_deserializes() {
        let json = r#"{"bdf": "0000:03:00.0", "ok": true, "pmc_before": "0x00000001", "pmc_after": "0x00000003"}"#;
        let r: DmaPrepareResult = serde_json::from_str(json).expect("parse");
        assert!(r.ok);
        assert_eq!(r.pmc_before.as_deref(), Some("0x00000001"));
    }

    #[test]
    fn pramin_read_result_deserializes() {
        let json = r#"{"data_b64": "AAAA", "length": 3}"#;
        let r: PraminReadResult = serde_json::from_str(json).expect("parse");
        assert_eq!(r.length, 3);
    }
}
