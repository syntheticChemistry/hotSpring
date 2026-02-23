// SPDX-License-Identifier: AGPL-3.0-only

//! Lattice field theory — SU(3) gauge theory on consumer GPUs.
//!
//! Evolution path from plasma MD to lattice QCD:
//!
//! | Component | Plasma MD analogue | Lattice QCD |
//! |-----------|-------------------|-------------|
//! | State | Particle positions | SU(3) link variables |
//! | Force law | Yukawa potential | Plaquette staples |
//! | Integrator | Velocity-Verlet | Leapfrog HMC |
//! | Observables | Energy, VACF, D* | Plaquette avg, Polyakov loop |
//!
//! The math is universal — only the force law and state representation change.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `complex_f64` | Complex f64 arithmetic (extracted from barracuda FFT) |
//! | `su3` | SU(3) 3×3 complex matrix operations |
//! | `wilson` | Wilson gauge action: plaquettes, staples, force |
//! | `hmc` | Hybrid Monte Carlo integrator |
//! | `dirac` | Staggered Dirac operator |
//! | `cg` | Conjugate gradient solver |
//! | `pseudofermion` | Pseudofermion action and dynamical fermion HMC |
//! | `abelian_higgs` | U(1) gauge + complex scalar Higgs (1+1)D |
//!
//! # References
//!
//! - `Creutz`, "Quarks, Gluons and Lattices" (1983)
//! - Gattringer & Lang, "Quantum Chromodynamics on the Lattice" (2010)
//! - `HotQCD` Collaboration, Bazavov et al., PRD 90, 094503 (2014)

/// U(1) gauge + complex scalar Higgs field theory in (1+1)D.
pub mod abelian_higgs;
/// Conjugate gradient solver for D-dagger-D on GPU.
pub mod cg;
/// Complex f64 arithmetic (re, im) with WGSL shader template.
pub mod complex_f64;
/// LCG PRNG, lattice constants, and shared numerical guards.
pub mod constants;
/// Hadronic correlator and susceptibility measurements.
pub mod correlator;
/// Staggered Dirac operator (GPU SpMV via WGSL).
pub mod dirac;
/// HotQCD EOS tables (Bazavov et al. PRD 90, 094503).
pub mod eos_tables;
/// Pure GPU HMC: all math on GPU via fp64 WGSL shaders.
pub mod gpu_hmc;
/// Hybrid Monte Carlo integrator (Cayley SU(3) exponential).
pub mod hmc;
/// Multi-GPU lattice dispatcher (CPU-threaded, evolution target).
pub mod multi_gpu;
/// Pseudofermion action and dynamical fermion HMC.
pub mod pseudofermion;
/// SU(3) 3x3 complex matrix operations (group, algebra, Cayley).
pub mod su3;
/// Wilson gauge action: plaquettes, staples, gauge force.
pub mod wilson;
