// SPDX-License-Identifier: AGPL-3.0-only

// Clippy pedantic/nursery + physics-specific allows are in [workspace.lints.clippy]
// in Cargo.toml. Library code must propagate errors, not panic:
#![deny(clippy::expect_used, clippy::unwrap_used)]
#![warn(missing_docs)]

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

/// Benchmark harness (RAPL energy, `nvidia-smi`, JSON reports).
pub mod bench;
/// AME2020 experimental data, Skyrme parameter bounds, and chi-squared.
pub mod data;
/// Capability-based data-path resolution (zero hardcoded absolute paths).
pub mod discovery;
/// Typed errors for GPU, simulation, and data-loading failure modes.
pub mod error;
/// GPU FP64 compute wrapper (`SHADER_F64` via wgpu/Vulkan).
pub mod gpu;
/// Lattice QCD: SU(3), Wilson action, HMC, Dirac, CG, Abelian Higgs.
pub mod lattice;
/// GPU molecular dynamics (f64 WGSL Yukawa OCP, cell-list, transport).
pub mod md;
/// NPU experiment campaign.
///
/// Trajectory generation, dataset builders, evaluators, and placement strategies.
pub mod npu_experiments;
/// Shared helpers for nuclear EOS validation binaries (L1/L2).
pub mod nuclear_eos_helpers;
/// Nuclear structure: SEMF, nuclear matter, spherical/deformed HFB, BCS.
pub mod physics;
/// L2 heterogeneous pipeline: L1 data gen, classifier training, L2 objective.
pub mod pipeline;
/// NMP cascade filter for L2 heterogeneous pipeline.
pub mod prescreen;
/// Shared types and infrastructure for production lattice QCD binaries.
pub mod production;
/// Traces every hardcoded value to its Python origin (script, commit, date).
pub mod provenance;
/// Physics proxy pipeline (Anderson 3D, Z(3) Potts) for NPU training.
pub mod proxy;
/// Spectral theory re-exports from `barracuda::spectral`.
pub mod spectral;
/// Centralized, justified validation thresholds (~170 constants).
pub mod tolerances;
/// Two-Temperature Model (laser-plasma 0D ODE solver).
pub mod ttm;
/// Pass/fail harness for validation binaries (exit 0/1).
pub mod validation;
