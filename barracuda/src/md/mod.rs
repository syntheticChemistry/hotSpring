// SPDX-License-Identifier: AGPL-3.0-only

//! GPU Molecular Dynamics — f64 Yukawa OCP on consumer GPU
//!
//! Full MD simulation pipeline matching Sarkas PP Yukawa studies.
//! Reference: Choi, Dharuman, Murillo, Phys. Rev. E 100, 013206 (2019).
//!
//! # Architecture
//!
//! Production simulation runs GPU-resident: particle data stays on GPU between
//! substeps, CPU reads back only at dump intervals for observables. Shader
//! compilation uses [`crate::gpu::GpuF64::create_pipeline_f64`] for driver-aware patching
//! (NVK `exp(f64)` workaround via barracuda).
//!
//! Barracuda ops (`barracuda::ops::md::*`) are used by validation binaries
//! which don't need GPU-resident performance. Production simulation keeps
//! local WGSL for the cell-list force kernel (not yet in barracuda GPU path).
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `config` | Sarkas-style config (κ, Γ, DSF parameters) |
//! | `observables` | Energy validation, CPU RDF/VACF, GPU SSF (via barracuda) |
//! | `simulation` | GPU-resident MD loop with cell-list support |
//! | `shaders` | WGSL shader sources for cell-list + inline kernels |
//! | `transport` | Daligault (2012) D* fit, Green-Kubo integration |
//! | `cpu_reference` | CPU Yukawa force for cross-validation |

pub mod config;
pub mod cpu_reference;
pub mod observables;
pub mod shaders;
pub mod simulation;
pub mod transport;
