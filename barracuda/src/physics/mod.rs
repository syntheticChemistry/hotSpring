//! Nuclear physics: Skyrme EDF, SEMF, spherical HFB
//!
//! Direct Rust ports of the Python control implementations.
//! Uses BarraCUDA special functions (gamma, laguerre) and numerical
//! methods (gradient_1d, trapz) wherever possible.

pub mod constants;
pub mod nuclear_matter;
pub mod semf;
pub mod hfb;

pub use constants::*;
pub use nuclear_matter::{nuclear_matter_properties, NuclearMatterProps};
pub use semf::semf_binding_energy;
pub use hfb::SphericalHFB;

