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
//! compilation routes through ToadStool's `WgslOptimizer` + `GpuDriverProfile`
//! for hardware-accurate ILP scheduling, fossil substitution, and driver-aware
//! exp/log patching (NVK workaround via barracuda).
//!
//! ## Upstream ToadStool capabilities (v0.5.16 audit)
//!
//! | Capability | ToadStool location | hotSpring status |
//! |---|---|---|
//! | `ReduceScalarPipeline` | `barracuda::pipeline` | Wired (KE/PE reduction) |
//! | `WgslOptimizer` | `barracuda::shaders::optimizer` | Wired (all shaders via `ShaderTemplate`) |
//! | `GpuDriverProfile` | `barracuda::device::capabilities` | Wired (shader compile) |
//! | `StatefulPipeline` | `barracuda::staging` | Available — `run_iterations()` / `run_until_converged()` |
//! | `CellListGpu` | `barracuda::ops::md::neighbor` | Not used (prefix_sum BGL mismatch — see handoff) |
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

pub mod celllist;
pub mod config;
pub mod cpu_reference;
pub mod observables;
pub mod shaders;
pub mod simulation;
pub mod transport;
