// SPDX-License-Identifier: AGPL-3.0-only

// Clippy pedantic/nursery + physics-specific allows are in [workspace.lints.clippy]
// in Cargo.toml. Library code must propagate errors, not panic:
#![deny(clippy::expect_used, clippy::unwrap_used)]

//! hotSpring Nuclear EOS — `BarraCuda` validation environment
//!
//! Validates `BarraCuda` library against Python/scipy controls using nuclear
//! equation-of-state workloads (Skyrme energy density functional) and GPU
//! molecular dynamics (Yukawa OCP).
//!
//! # Evolution path
//!
//! ```text
//! Python baseline → Rust validation → GPU acceleration → sovereign pipeline
//! ```
//!
//! # Architecture
//!
//! - **`provenance`** — traces every hardcoded value to its Python origin
//! - **`tolerances`** — centralized, justified validation thresholds
//! - **`validation`** — harness for pass/fail binaries (exit 0/1)
//! - **`discovery`** — capability-based data path resolution
//! - **`data`** — AME2020 experimental data and Skyrme parameter bounds
//! - **`physics`** — SEMF, nuclear matter, spherical HFB, deformed HFB
//! - **`error`** — typed errors for GPU/simulation failure modes
//! - **`gpu`** — GPU FP64 device wrapper (`SHADER_F64` via wgpu/Vulkan)
//! - **`md`** — GPU molecular dynamics (f64 WGSL Yukawa OCP)
//! - **`prescreen`** — NMP cascade filter for L2 heterogeneous pipeline
//! - **`bench`** — benchmark harness (RAPL, nvidia-smi, JSON reports)
//!
//! # Validation binaries
//!
//! All binaries follow the hotSpring pattern: hardcoded expected values
//! with provenance, explicit pass/fail against documented tolerances,
//! exit code 0 (pass) or 1 (fail).
//!
//! # License
//!
//! AGPL-3.0 — see LICENSE in repository root.

pub mod bench;
pub mod data;
pub mod discovery;
pub mod error;
pub mod gpu;
pub mod lattice;
pub mod md;
pub mod physics;
pub mod prescreen;
pub mod provenance;
pub mod spectral;
pub mod tolerances;
pub mod validation;
