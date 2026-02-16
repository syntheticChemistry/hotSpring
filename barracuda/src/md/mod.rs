// SPDX-License-Identifier: AGPL-3.0-only

//! GPU Molecular Dynamics — f64 Yukawa OCP on consumer GPU
//!
//! Full MD simulation pipeline matching Sarkas PP Yukawa studies:
//!   - Yukawa pairwise force (native f64 WGSL builtins)
//!   - Velocity-Verlet symplectic integrator
//!   - Periodic boundary conditions (minimum image)
//!   - Berendsen thermostat (equilibration)
//!   - Cell-list O(N) force calculation for N > ~2,000
//!   - Observables: energy, RDF, VACF, SSF
//!
//! All physics runs on GPU. CPU handles I/O and observable accumulation.
//!
//! Reference: Choi, Dharuman, Murillo, Phys. Rev. E 100, 013206 (2019).

pub mod config;
pub mod cpu_reference;
pub mod observables;
pub mod shaders;
pub mod simulation;
