// SPDX-License-Identifier: AGPL-3.0-or-later

// Clippy pedantic/nursery + physics-specific allows are in [workspace.lints.clippy]
// in Cargo.toml. Library code must propagate errors, not panic:
#![deny(unsafe_code)]
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

// Shared `bin_helpers` modules use `hotspring_barracuda::...` paths; alias this crate so those
// resolve the same way as in `src/bin` targets.
extern crate self as hotspring_barracuda;

/// Minimal zero-dependency base64 codec (RFC 4648).
pub mod base64_encode;
/// Minimal inline `block_on` — replaces `pollster` crate.
pub mod block_on;
/// Benchmark harness (RAPL energy, `nvidia-smi`, JSON reports).
#[cfg(feature = "barracuda-local")]
pub mod bench;
/// Shared infrastructure for `src/bin` targets (refactored validation suites).
#[cfg(feature = "barracuda-local")]
pub mod bin_helpers;
/// Composition certification engine — absorbed guideStone organelle (L0–L5).
pub mod certification;
/// Computational chemistry: topology parsing, force field bridge, parity validation.
pub mod compchem;
/// NUCLEUS composition validation — atomic health, capability routing, science probes.
pub mod composition;
/// ToadStool compute dispatch validation (submit/result/capabilities).
pub mod compute_dispatch;
/// rhizoCrypt DAG session for computation trace with blake3 + witnesses.
pub mod dag_provenance;
/// AME2020 experimental data, Skyrme parameter bounds, and chi-squared.
pub mod data;
/// Heterogeneous dual-GPU device pair (precise + throughput brains).
#[cfg(feature = "barracuda-local")]
pub mod device_pair;
/// Capability-based data-path resolution (zero hardcoded absolute paths).
pub mod discovery;
/// Dual-dispatch executor for heterogeneous GPU workloads.
#[cfg(feature = "barracuda-local")]
pub mod dual_dispatch;
/// Dual-card cooperative pipeline profiler (Split BCS, Split HMC, Redundant).
#[cfg(feature = "barracuda-local")]
pub mod dual_pipeline_eval;
/// Typed response structs for toadstool-ember IPC (MMIO, falcon, SEC2, PRAMIN, DMA).
pub mod ember_types;
/// Typed errors for GPU, simulation, and data-loading failure modes.
pub mod error;
/// toadstool-ember multi-instance fleet discovery and per-socket JSON-RPC routing.
pub mod fleet_client;
/// Per-ember JSON-RPC client: MMIO, falcon, SEC2, PRAMIN, DMA, flood testing.
pub mod fleet_ember;
/// toadStool dispatch client: IPC path for compute.dispatch.capabilities/submit.
/// Parallel to fleet_ember — preparation for toadStool Phase C migration.
#[cfg(feature = "toadstool-dispatch")]
pub mod fleet_toadstool;
/// JSON-RPC client for toadStool compute dispatch.
///
/// Covers `device.dispatch`, `device.list`, health checks, and related helpers.
pub mod glowplug_client;
/// GPU FP64 compute wrapper (`SHADER_F64` via wgpu/Vulkan).
#[cfg(feature = "barracuda-local")]
pub mod gpu;
/// Hardware calibration: safe per-tier probe + capability mask.
#[cfg(feature = "barracuda-local")]
pub mod hardware_calibration;
/// Consolidated IPC module: discovery, composition, glowplug, ember, squirrel, signing.
pub mod ipc;
/// Lattice QCD: SU(3), Wilson action, HMC, Dirac, CG, Abelian Higgs.
pub mod lattice;
/// Low-level PCI BAR0 MMIO and Falcon register map (legacy).
///
/// **Deprecated**: canonical MMIO lives in toadStool cylinder (`sysfs_bar0`,
/// `nv::registers::falcon`). Experiment binaries should use ember/glowplug RPCs
/// via [`bin_helpers::sovereignty::connect`] instead of direct mmap.
///
/// Contains `unsafe` blocks for mmap/`read_volatile`/`write_volatile`.
/// Gated behind `low-level` feature (not in default build).
#[cfg(feature = "low-level")]
#[expect(unsafe_code, reason = "MMIO mmap requires unsafe; audited surface confined to bar0.rs")]
#[deprecated(
    since = "0.6.32",
    note = "Use toadStool ember/glowplug RPCs. See bin_helpers/sovereignty/connect.rs."
)]
pub mod low_level;

/// MCP (Model Context Protocol) tool definitions for AI/LLM integration.
pub mod mcp_tools;
/// GPU molecular dynamics (f64 WGSL Yukawa OCP, cell-list, transport).
pub mod md;
/// Mixed-hardware pipeline infrastructure for metalForge integration.
pub mod mixed_hardware;
/// Niche deployment self-knowledge: capabilities, dependencies, proto-nucleate.
pub mod niche;
/// NPU experiment campaign.
///
/// Trajectory generation, dataset builders, evaluators, and placement strategies.
#[cfg(feature = "barracuda-local")]
pub mod npu_experiments;
/// Shared helpers for nuclear EOS validation binaries (L1/L2).
#[cfg(feature = "barracuda-local")]
pub mod nuclear_eos_helpers;
/// Nuclear structure: SEMF, nuclear matter, spherical/deformed HFB, BCS.
pub mod physics;
/// L2 heterogeneous pipeline: L1 data gen, classifier training, L2 objective.
#[cfg(feature = "barracuda-local")]
pub mod pipeline;
/// Full physics pipeline end-to-end profiler (MD, HMC, BCS, dielectric).
#[cfg(feature = "barracuda-local")]
pub mod pipeline_eval;
/// Self-routing precision brain: data-driven tier selection from calibration.
#[cfg(feature = "barracuda-local")]
pub mod precision_brain;
/// Per-shader precision/throughput profiler across all 3 tiers.
#[cfg(feature = "barracuda-local")]
pub mod precision_eval;
/// Precision routing: capability-aware shader compilation (local toadStool evolution).
#[cfg(feature = "barracuda-local")]
pub mod precision_routing;
/// NMP cascade filter for L2 heterogeneous pipeline.
#[cfg(feature = "barracuda-local")]
pub mod prescreen;
/// NUCLEUS primal discovery — runtime detection of available primals.
pub mod primal_bridge;
/// JSON-RPC server for NUCLEUS deploy graph integration.
pub mod serve;
/// Shared types and infrastructure for production lattice QCD binaries.
#[cfg(feature = "barracuda-local")]
pub mod production;
/// Shared statistical helpers for production binaries (delegates to barraCuda `stats`).
#[cfg(feature = "barracuda-local")]
pub mod production_support;
/// Traces every hardcoded value to its Python origin (script, commit, date).
pub mod provenance;
/// Physics proxy pipeline (Anderson 3D, Z(3) Potts) for NPU training.
#[cfg(feature = "barracuda-local")]
pub mod proxy;
/// Ed25519 receipt signing via bearDog `crypto.sign_ed25519` JSON-RPC.
pub mod receipt_signing;
/// Spectral theory re-exports from `barracuda::spectral`.
#[cfg(feature = "barracuda-local")]
pub mod spectral;
/// Squirrel / neuralSpring `inference.*` JSON-RPC client (NUCLEUS capability discovery).
pub mod squirrel_client;
/// Streaming compute dispatch for GPU-resident physics (local toadStool evolution).
#[cfg(feature = "barracuda-local")]
pub mod streaming_dispatch;
/// JSONL telemetry reader for petalTongue visualization integration.
pub mod telemetry_reader;
/// Minimal toadStool performance surface reporter (JSON-RPC over Unix socket).
pub mod toadstool_report;
/// Centralized, justified validation thresholds (~170 constants).
pub mod tolerances;
/// PCIe transfer cost profiler per GPU card.
#[cfg(feature = "barracuda-local")]
pub mod transfer_eval;
/// Two-Temperature Model (laser-plasma 0D ODE solver).
pub mod ttm;
/// Pass/fail harness for validation binaries (exit 0/1/2).
pub mod validation;
/// `WireWitnessRef` — self-describing provenance events (ATTESTATION_ENCODING_STANDARD v2).
pub mod witness;
/// Workload planner for heterogeneous dual-GPU dispatch.
#[cfg(feature = "barracuda-local")]
pub mod workload_planner;

/// Vendor-agnostic register maps for GPU reverse engineering.
pub mod register_maps;
