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
//! | `abelian_higgs` | U(1) gauge + complex scalar Higgs (1+1)D |
//!
//! # References
//!
//! - `Creutz`, "Quarks, Gluons and Lattices" (1983)
//! - Gattringer & Lang, "Quantum Chromodynamics on the Lattice" (2010)
//! - `HotQCD` Collaboration, Bazavov et al., PRD 90, 094503 (2014)

pub mod abelian_higgs;
pub mod cg;
pub mod complex_f64;
pub mod constants;
pub mod dirac;
pub mod eos_tables;
pub mod hmc;
pub mod multi_gpu;
pub mod su3;
pub mod wilson;
