// SPDX-License-Identifier: AGPL-3.0-only

//! GPU Molecular Dynamics — f64 Yukawa OCP on consumer GPU
//!
//! Full MD simulation pipeline matching Sarkas PP Yukawa studies.
//! Reference: Choi, Dharuman, Murillo, Phys. Rev. E 100, 013206 (2019).
//!
//! # Deprecation status
//!
//! MD force, integrator, thermostat, and observable shaders have been
//! absorbed by `barracuda::ops::md`. The following submodules are
//! **deprecated** and retained as fossil record:
//!
//! | Module | Absorbed by | Status |
//! |--------|-------------|--------|
//! | `shaders` | `barracuda::ops::md::*` | Deprecated — local .wgsl copies |
//! | `cpu_reference` | barracuda CPU forces | Deprecated — benchmarking reference |
//!
//! The following submodules remain **active** (hotSpring-specific):
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `config` | Sarkas-style config (κ, Γ, DSF parameters) |
//! | `observables` | Energy validation, CPU RDF/VACF, GPU SSF (via barracuda) |
//! | `simulation` | MD loop orchestration (should migrate to barracuda ops) |

pub mod config;
pub mod cpu_reference;
pub mod observables;
pub mod shaders;
pub mod simulation;
pub mod yukawa_nvk_safe;
