// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parity validation: sovereign FEL reconstruction vs GROMACS reference.
//!
//! Loads PLUMED HILLS files, reconstructs F(θ) using our metadynamics
//! engine, and compares against GROMACS `sum_hills --mintozero` output.

pub mod fes_parity;

pub use fes_parity::{FesParity, FesParityResult};
