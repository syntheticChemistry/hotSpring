// SPDX-License-Identifier: AGPL-3.0-only

// ── Clippy configuration ──
// Physics code necessarily uses usize↔f64 casts, similar variable names
// (vx/vy/vz, rho_p/rho_n), and single-character physics variables (r, k, q).
// These are idiomatic in computational physics, not bugs.
// Workspace-level [lints.clippy] in Cargo.toml covers all targets (lib + bins).
#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::use_self,
    clippy::redundant_else,
    clippy::inline_always,
    clippy::struct_field_names,
    clippy::items_after_statements,
    clippy::wildcard_imports,
    clippy::unreadable_literal,
    clippy::imprecise_flops,
    clippy::implicit_hasher,
    clippy::needless_pass_by_value,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::option_if_let_else,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::assigning_clones,
    clippy::manual_clamp,
    clippy::manual_let_else,
    clippy::significant_drop_tightening,
    clippy::collection_is_never_read,
    clippy::needless_for_each,
    clippy::no_effect_underscore_binding,
    clippy::unused_self,
    clippy::used_underscore_binding,
    clippy::float_cmp
)]

//! hotSpring Nuclear EOS — BarraCUDA validation environment
//!
//! Validates BarraCUDA library against Python/scipy controls using nuclear
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
//! - **`gpu`** — GPU FP64 device wrapper (SHADER_F64 via wgpu/Vulkan)
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
