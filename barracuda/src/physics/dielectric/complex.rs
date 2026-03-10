// SPDX-License-Identifier: AGPL-3.0-only

//! Complex number type for plasma response calculations.
//!
//! Re-exports barraCuda's `Complex64` as `Complex` — single source of truth
//! for complex arithmetic across the ecoPrimals pipeline.

pub use barracuda::ops::lattice::cpu_complex::Complex64 as Complex;
