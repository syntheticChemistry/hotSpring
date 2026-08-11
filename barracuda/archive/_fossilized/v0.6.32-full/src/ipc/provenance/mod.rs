// SPDX-License-Identifier: AGPL-3.0-or-later

//! Per-trio provenance IPC modules (rhizoCrypt, loamSpine, sweetGrass).
//!
//! Following ludoSpring's reference pattern: each provenance primal has a
//! dedicated IPC module for type-safe cross-primal calls.
//!
//! # Trio Architecture
//!
//! ```text
//! rhizoCrypt  — DAG-based computation trace (blake3 witnesses)
//! loamSpine   — Distributed ledger for provenance records
//! sweetGrass  — Attribution braid linking experiments to papers
//! ```

pub mod loamspine;
pub mod rhizocrypt;
pub mod sweetgrass;
