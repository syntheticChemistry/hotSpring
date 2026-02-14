//! Nuclear physics: Skyrme EDF, SEMF, spherical HFB, deformed HFB
//!
//! Direct Rust ports of the Python control implementations.
//! Uses BarraCUDA special functions (gamma, laguerre) and numerical
//! methods (gradient_1d, trapz) wherever possible.
//!
//! Levels:
//!   L1 — SEMF (Semi-Empirical Mass Formula)
//!   L2 — Spherical HFB (Hartree-Fock-Bogoliubov)
//!   L3 — Axially-deformed HFB (Nilsson basis)
//!   L4 — Beyond-mean-field (GCM, Fayans, etc.) [future]

pub mod constants;
pub mod nuclear_matter;
pub mod semf;
pub mod hfb;
pub mod hfb_gpu;
pub mod hfb_deformed;

pub use constants::*;
pub use nuclear_matter::{nuclear_matter_properties, NuclearMatterProps};
pub use semf::semf_binding_energy;
pub use hfb::{SphericalHFB, binding_energy_l2};
pub use hfb_gpu::binding_energies_l2_gpu;
pub use hfb_deformed::DeformedHFB;

