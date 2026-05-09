// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consolidated IPC module for hotSpring NUCLEUS primal interactions.
//!
//! This module provides the unified entry point for all primal IPC in hotSpring.
//! It consolidates discovery, composition validation, and domain-specific clients
//! (glowplug, ember, squirrel, toadStool, receipt signing) into a single `ipc::`
//! namespace.
//!
//! # Architecture
//!
//! ```text
//! ipc::discovery     — NucleusContext, PrimalEndpoint, send_jsonrpc
//! ipc::composition   — AtomicType, validate_atomic, composition_health
//! ipc::glowplug      — GlowplugClient (coral-glowplug shader dispatch)
//! ipc::ember         — EmberClient, FleetEmberHub (coral-ember MMIO/falcon)
//! ipc::fleet         — FleetClient (multi-ember fleet discovery)
//! ipc::squirrel      — SquirrelClient (inference via neuralSpring/squirrel)
//! ipc::toadstool     — ToadStool performance surface reporter
//! ipc::signing       — Receipt signing via bearDog crypto.sign_ed25519
//! ```
//!
//! # Migration
//!
//! Previous scattered modules are re-exported through this namespace:
//!
//! | Old path | New path |
//! |----------|----------|
//! | `crate::primal_bridge::NucleusContext` | `crate::ipc::discovery::NucleusContext` |
//! | `crate::composition::AtomicType` | `crate::ipc::composition::AtomicType` |
//! | `crate::glowplug_client::GlowplugClient` | `crate::ipc::glowplug::GlowplugClient` |
//! | `crate::fleet_ember::EmberClient` | `crate::ipc::ember::EmberClient` |

/// NUCLEUS primal discovery — socket scanning, liveness probing, capability queries.
///
/// Re-exports from [`crate::primal_bridge`].
pub mod discovery {
    pub use crate::primal_bridge::{NucleusContext, PrimalEndpoint, jsonrpc_request, send_jsonrpc};
}

/// NUCLEUS composition validation — atomic hierarchy, health, science probes.
///
/// Re-exports from [`crate::composition`].
pub mod composition {
    pub use crate::composition::{
        AtomicType, AtomicValidation, ScienceProbeResult, composition_health, get_by_capability,
        nest_health, node_health, nucleus_health, tower_health, validate_atomic,
        validate_capability, validate_science_probes,
    };
}

/// coral-glowplug JSON-RPC client (shader dispatch, device listing, BAR0 probes).
///
/// Re-exports from [`crate::glowplug_client`].
pub mod glowplug {
    pub use crate::glowplug_client::{
        CaptureTrainingResult, GlowplugClient, GlowplugDaemonHealth, GlowplugDeviceDetail,
        GlowplugDeviceHealthSummary, GlowplugDeviceSummary, GlowplugDispatchOptions, GlowplugError,
        SovereignBootResult,
    };
}

/// coral-ember per-instance JSON-RPC client (MMIO, falcon, SEC2, PRAMIN, DMA).
///
/// Re-exports from [`crate::fleet_ember`].
pub mod ember {
    pub use crate::fleet_ember::{EmberClient, FleetEmberHub};
}

/// Multi-ember fleet discovery and per-socket routing.
///
/// Re-exports from [`crate::fleet_client`].
pub mod fleet {
    pub use crate::fleet_client::*;
}

/// Squirrel / neuralSpring inference JSON-RPC client.
///
/// Re-exports from [`crate::squirrel_client`].
pub mod squirrel {
    pub use crate::squirrel_client::*;
}

/// ToadStool performance surface reporter.
///
/// Re-exports from [`crate::toadstool_report`].
pub mod toadstool {
    pub use crate::toadstool_report::*;
}

/// Receipt signing via bearDog `crypto.sign_ed25519` JSON-RPC.
///
/// Re-exports from [`crate::receipt_signing`].
pub mod signing {
    pub use crate::receipt_signing::*;
}
