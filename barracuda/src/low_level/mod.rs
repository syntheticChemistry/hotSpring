// SPDX-License-Identifier: AGPL-3.0-or-later

//! Low-level hardware abstractions (requires `low-level` feature).
//!
//! # Deprecated — use toadStool RPCs instead
//!
//! This module duplicates MMIO functionality that now lives in toadStool
//! (`toadstool_cylinder::vfio::sysfs_bar0`, `nv::registers::falcon`).
//! Experiment binaries and validation tools should route GPU access through
//! ember/glowplug JSON-RPC rather than direct BAR0 mmap:
//!
//! | Operation        | RPC method                 |
//! |------------------|----------------------------|
//! | Register read    | `mmio.read32`              |
//! | Register write   | `mmio.write32`             |
//! | Batch MMIO       | `mmio.batch`               |
//! | Falcon IMEM/DMEM | `ember.falcon.upload_*`    |
//! | Falcon poll      | `ember.falcon.poll`        |
//!
//! Use [`crate::bin_helpers::sovereignty::connect`] (`connect_ember`,
//! `connect_glowplug`) from experiment binaries.
//!
//! Files in this module are retained as legacy reference behind the
//! `low-level` feature gate (not enabled by default).

pub mod bar0;
pub mod falcon;

pub use bar0::{Bar0Domain, Bar0Error, Bar0Map, Bar0View, DenyEntry, SafeBar0};
pub use falcon::{DEAD_LINK, FalconSnapshot};
