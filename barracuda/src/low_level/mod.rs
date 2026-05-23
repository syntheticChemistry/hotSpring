// SPDX-License-Identifier: AGPL-3.0-or-later

//! Low-level hardware abstractions (requires `low-level` feature).
//!
//! Contains safe RAII wrappers around unsafe OS primitives (mmap, MMIO) so
//! the unsafe surface is audited in one place and callers use safe APIs.

pub mod bar0;

pub use bar0::{Bar0Domain, Bar0Map, Bar0View, SafeBar0};
