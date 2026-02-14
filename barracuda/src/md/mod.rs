//! GPU Molecular Dynamics — f64 Yukawa OCP on consumer GPU
//!
//! Full MD simulation pipeline matching Sarkas PP Yukawa studies:
//!   - Yukawa pairwise force (f64 WGSL via math_f64)
//!   - Velocity-Verlet symplectic integrator
//!   - Periodic boundary conditions (minimum image)
//!   - Berendsen thermostat (equilibration)
//!   - Observables: energy, RDF, VACF, SSF
//!
//! All physics runs on GPU. CPU handles I/O and observable accumulation.

pub mod shaders;
pub mod simulation;
pub mod config;
pub mod cpu_reference;
pub mod observables;
