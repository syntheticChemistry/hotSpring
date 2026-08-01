// SPDX-License-Identifier: AGPL-3.0-or-later

//! CAZyme conformational free energy landscape validation module.
//!
//! Reconstructs F(θ) from PLUMED HILLS files using Gaussian kernel summation
//! (well-tempered metadynamics). Validates topology: basin count, barrier
//! ranges, ground state identity.
//!
//! This is the Tier 2 (Rust) implementation paired with the Tier 1 Python
//! notebook at `notebooks/cazyme_fel/puckering_fel.py`. Both must produce
//! identical results (MATCH in ParityReport).
//!
//! Numeric constants are anchored in:
//!   pseudoSpore/derivations/threshold_calibration.toml

mod binding;
mod constants;
mod cross_landscape;
mod interp;
mod ks;
mod parse;
mod parity;
mod reconstruct;
mod topology;
mod types;
mod validation;

#[cfg(test)]
mod tests;

pub use binding::check_binding_distance;
pub use cross_landscape::{compare_free_bound, compare_free_bound_2d};
pub use ks::{compare_colvar_distributions, ks_two_sample};
pub use parse::{
    parse_binding_colvar, parse_colvar_theta, parse_fes, parse_fes_2d, parse_hills,
    parse_hills_2d,
};
pub use parity::{check_parity, check_parity_2d};
pub use reconstruct::{reconstruct_fes, reconstruct_fes_2d};
pub use topology::{find_barriers, find_basins};
pub use types::{
    Barrier, Basin, BasinDiff, BindingCheck, CrossLandscapeReport, CrossLandscapeVerdict,
    FesResult, FesResult2D, Hills, Hills2D, KsTestResult, ParityCheck, ValidationResult,
};
pub use validation::{run_validation, run_validation_2d, run_validation_2d_with_bounds};
