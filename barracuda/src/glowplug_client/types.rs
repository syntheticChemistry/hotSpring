// SPDX-License-Identifier: AGPL-3.0-or-later

//! Protocol types for the toadStool device management RPC client.
//!
//! These are pure data definitions: request option structs, response
//! deserialisation targets, error variants, and wire-format helpers.
//! The [`super::GlowplugClient`] impl that uses them lives in the
//! parent module.

use serde::Deserialize;
use std::fmt;

pub(super) const DEFAULT_DISPATCH_DIMS: [u32; 3] = [1, 1, 1];
pub(super) const DEFAULT_DISPATCH_WORKGROUP: [u32; 3] = [256, 1, 1];
pub(super) const DEFAULT_KERNEL_ENTRY_NAME: &str = "main_kernel";

// ── Request types ───────────────────────────────────────────────────

/// Options for [`super::GlowplugClient::dispatch`] (grid, workgroup, entry symbol).
#[derive(Debug, Clone)]
pub struct GlowplugDispatchOptions {
    /// `[grid_x, grid_y, grid_z]` dispatch dimensions.
    pub dims: [u32; 3],
    /// `[threads_x, threads_y, threads_z]` per-block (workgroup) size.
    pub workgroup: [u32; 3],
    /// CUDA kernel / entry point name in the shader module.
    pub kernel_name: String,
    /// Dynamic shared memory bytes (default 0).
    pub shared_mem: u32,
}

impl Default for GlowplugDispatchOptions {
    fn default() -> Self {
        Self {
            dims: DEFAULT_DISPATCH_DIMS,
            workgroup: DEFAULT_DISPATCH_WORKGROUP,
            kernel_name: DEFAULT_KERNEL_ENTRY_NAME.to_string(),
            shared_mem: 0,
        }
    }
}

// ── Device list / detail types ──────────────────────────────────────

/// One row from `device.list` — BDF, vendor id, display name, coarse health.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlowplugDeviceSummary {
    pub bdf: String,
    /// PCI vendor id as `0xABCD` (same information as JSON `vendor_id`).
    pub vendor: String,
    pub name: Option<String>,
    /// Driver / backend personality string (e.g. `nvidia`, `cuda`).
    pub personality: String,
    /// True when the device is display-attached and swap-immune.
    pub protected: bool,
    pub health: GlowplugDeviceHealthSummary,
}

/// Health fields exposed for quick listing (mirrors glowplug `DeviceInfo` health slice).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlowplugDeviceHealthSummary {
    pub vram_alive: bool,
    pub domains_faulted: usize,
}

/// Full `device.get` payload (structured like toadStool `DeviceInfo`).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GlowplugDeviceDetail {
    pub bdf: String,
    pub name: Option<String>,
    pub chip: String,
    pub vendor_id: u16,
    pub device_id: u16,
    pub personality: String,
    pub role: Option<String>,
    pub power: String,
    pub vram_alive: bool,
    pub domains_alive: usize,
    pub domains_faulted: usize,
    pub has_vfio_fd: bool,
    pub pci_link_width: Option<u8>,
    #[serde(default)]
    pub protected: bool,
}

/// Daemon response for `health.check` / `health.liveness`.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GlowplugDaemonHealth {
    pub alive: bool,
    pub name: String,
    pub device_count: usize,
    pub healthy_count: usize,
}

// ── Error type ──────────────────────────────────────────────────────

/// Errors from glowplug RPC or payload handling.
#[derive(Debug)]
pub enum GlowplugError {
    /// No `compute` provider (toadStool) found in [`crate::primal_bridge::NucleusContext`].
    NoComputeEndpoint,
    /// Socket path exists but the primal failed liveness at discovery time.
    EndpointNotAlive,
    /// Low-level transport / JSON parse (`send_jsonrpc` message).
    Transport(String),
    /// JSON-RPC `error` object.
    JsonRpc { code: i64, message: String },
    /// Successful envelope but missing `result`.
    MissingResult,
    /// Base64 decode of a dispatch output buffer.
    OutputDecode(String),
    /// `serde_json` shape mismatch.
    InvalidPayload(String),
}

impl fmt::Display for GlowplugError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlowplugError::NoComputeEndpoint => {
                write!(f, "no compute provider (toadStool) in NucleusContext")
            }
            GlowplugError::EndpointNotAlive => write!(f, "compute provider socket is not alive"),
            GlowplugError::Transport(s) => write!(f, "transport: {s}"),
            GlowplugError::JsonRpc { code, message } => write!(f, "json-rpc {code}: {message}"),
            GlowplugError::MissingResult => write!(f, "json-rpc response missing result"),
            GlowplugError::OutputDecode(s) => write!(f, "output base64 decode: {s}"),
            GlowplugError::InvalidPayload(s) => write!(f, "invalid payload: {s}"),
        }
    }
}

impl std::error::Error for GlowplugError {}

// ── Response types ──────────────────────────────────────────────────

/// Result of `capture.training` — training recipe capture flow.
#[derive(Debug, Clone, Deserialize)]
pub struct CaptureTrainingResult {
    pub bdf: String,
    pub warm_driver: String,
    pub recipe_path: Option<String>,
    pub total_writes: usize,
    pub success: bool,
    pub summary: String,
    pub steps: Vec<BootStepResult>,
}

/// Result of `device.warm_catch` — warm GPU catch after driver handoff.
///
/// S258: when FECS is warm and `open_vfio()` succeeds, `vfio_open` is `true`
/// and `channel_id` contains the PFIFO channel ID for PBDMA dispatch.
#[derive(Debug, Clone, Deserialize)]
pub struct WarmCatchResult {
    pub bdf: String,
    #[serde(default)]
    pub fecs_ready: bool,
    #[serde(default)]
    pub chip_id: Option<u32>,
    #[serde(default)]
    pub capabilities: Option<serde_json::Value>,
    #[serde(default)]
    pub summary: Option<String>,
    /// Whether `open_vfio()` succeeded (S258 PBDMA channel ready).
    #[serde(default)]
    pub vfio_open: bool,
    /// PFIFO channel ID if VFIO dispatch is initialized.
    #[serde(default)]
    pub channel_id: Option<u32>,
}

/// Result of `sovereign.boot` — full orchestrated sovereign boot.
#[derive(Debug, Clone, Deserialize)]
pub struct SovereignBootResult {
    pub bdf: String,
    pub initial_driver: Option<String>,
    pub warm_cycle_performed: bool,
    pub final_driver: Option<String>,
    pub sovereign_init: Option<serde_json::Value>,
    pub success: bool,
    pub summary: String,
    pub steps: Vec<BootStepResult>,
}

/// A single step in an orchestrated boot or capture flow.
#[derive(Debug, Clone, Deserialize)]
pub struct BootStepResult {
    pub name: String,
    pub status: String,
    pub detail: Option<String>,
    pub duration_ms: u64,
}

// ── Internal wire-format types ──────────────────────────────────────

#[derive(Debug, Deserialize)]
pub(super) struct GlowplugListRow {
    pub bdf: String,
    pub name: Option<String>,
    pub vendor_id: u16,
    pub personality: String,
    #[serde(default)]
    pub protected: bool,
    pub vram_alive: bool,
    pub domains_faulted: usize,
}

impl From<GlowplugListRow> for GlowplugDeviceSummary {
    fn from(row: GlowplugListRow) -> Self {
        GlowplugDeviceSummary {
            bdf: row.bdf,
            vendor: format!("0x{:04X}", row.vendor_id),
            name: row.name,
            personality: row.personality,
            protected: row.protected,
            health: GlowplugDeviceHealthSummary {
                vram_alive: row.vram_alive,
                domains_faulted: row.domains_faulted,
            },
        }
    }
}
