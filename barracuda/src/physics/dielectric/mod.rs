// SPDX-License-Identifier: AGPL-3.0-only

//! Conservative dielectric functions from the BGK equation.
//!
//! Implements the Mermin dielectric function and related plasma response
//! functions for classical one-component plasmas (OCP). This module
//! provides the theoretical foundation for dynamic structure factor S(k,ω)
//! computation, complementing the MD-derived DSF from Papers 1/5.
//!
//! # Module structure
//!
//! - `complex` — re-exports barraCuda's `Complex64` as `Complex`
//! - `plasma_dispersion` — plasma parameters; Z(z)/W(z) delegated to barraCuda
//! - `response` — Mermin dielectric, DSF, f-sum rule, DC conductivity, validation
//!
//! # References
//!
//! - Mermin, Phys. Rev. B 1, 2362 (1970)
//! - Chuna & Murillo, Phys. Rev. E 111, 035206 (2024), arXiv:2405.07871
//! - Stanton & Murillo, Phys. Rev. E 91, 033104 (2015)

pub mod complex;
pub mod plasma_dispersion;
pub mod response;

pub use complex::Complex;
pub use plasma_dispersion::{
    PlasmaParams, chi0_classical, epsilon_vlasov, plasma_dispersion_w, plasma_dispersion_w_stable,
    plasma_dispersion_z,
};
pub use response::{
    DielectricValidation, conductivity_dc, debye_screening, dynamic_structure_factor,
    dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
    f_sum_rule_integral, f_sum_rule_integral_completed, validate_dielectric,
};
