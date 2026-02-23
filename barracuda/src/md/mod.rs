// SPDX-License-Identifier: AGPL-3.0-only

//! GPU Molecular Dynamics — f64 Yukawa OCP on consumer GPU
//!
//! Full MD simulation pipeline matching Sarkas PP Yukawa studies.
//! Reference: Choi, Dharuman, Murillo, Phys. Rev. E 100, 013206 (2019).
//!
//! # Architecture
//!
//! Production simulation runs GPU-resident: particle data stays on GPU between
//! substeps, CPU reads back only at dump intervals for observables. All shader
//! compilation routes through `ToadStool`'s `WgslOptimizer` + `GpuDriverProfile`
//! for hardware-accurate ILP scheduling, fossil substitution, and driver-aware
//! exp/log patching (NVK workaround via barracuda).
//!
//! ## Upstream `ToadStool` capabilities (v0.5.16 audit)
//!
//! | Capability | ToadStool location | hotSpring status |
//! |---|---|---|
//! | `ReduceScalarPipeline` | `barracuda::pipeline` | Wired (KE/PE reduction) |
//! | `WgslOptimizer` | `barracuda::shaders::optimizer` | Wired (all shaders via `ShaderTemplate`) |
//! | `GpuDriverProfile` | `barracuda::device::capabilities` | Wired (shader compile) |
//! | `StatefulPipeline` | `barracuda::staging` | Available — `run_iterations()` / `run_until_converged()` |
//! | `CellListGpu` | `barracuda::ops::md::neighbor` | **Migrated** (v0.6.2) — local `GpuCellList` deleted, using upstream |
//! | NAK eigensolve shader | `shaders/linalg/batched_eigh_nak_optimized_f64.wgsl` | Absorbed from hotSpring |
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `config` | Sarkas-style config (κ, Γ, DSF parameters) |
//! | `observables` | Energy validation, CPU RDF/VACF, GPU SSF (via barracuda) |
//! | `simulation` | GPU-resident MD loop with cell-list support |
//! | `shaders` | WGSL shader sources for cell-list + inline kernels |
//! | `transport` | Daligault (2012) D* fit, Green-Kubo integration |
//! | `cpu_reference` | CPU Yukawa force for cross-validation |

/// Cell-list neighbor search (GPU-resident via upstream `CellListGpu`).
pub mod celllist;
/// Sarkas-style MD configuration (kappa, Gamma, DSF parameters).
pub mod config;
/// CPU Yukawa force reference for GPU cross-validation.
pub mod cpu_reference;
/// Diagnostic helpers for force comparison (celllist_diag).
pub mod diag;
/// Energy, RDF, SSF, VACF, transport observables.
pub mod observables;
/// Real NPU hardware adapter (BrainChip Akida AKD1000).
#[cfg(feature = "npu-hw")]
pub mod npu_hw;
/// Echo State Network (ESN) reservoir for transport coefficient prediction.
pub mod reservoir;
/// WGSL shader sources loaded from `.wgsl` files (zero inline).
pub mod shaders;
/// GPU-resident MD simulation loop with cell-list support.
pub mod simulation;
/// GPU transport pipeline: batched VACF, Green-Kubo D*.
pub mod simulation_transport_gpu;
/// Daligault (2012) D* fit, Stanton-Murillo transport coefficients.
pub mod transport;
