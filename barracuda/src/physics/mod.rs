// SPDX-License-Identifier: AGPL-3.0-only

//! Physics modules: nuclear structure, atomic physics, dense plasma theory.
//!
//! Direct Rust ports of the Python control implementations.
//! Uses BarraCUDA special functions (gamma, laguerre) and numerical
//! methods (gradient_1d, trapz) wherever possible.
//!
//! Domains:
//!   Nuclear EOS — SEMF (L1), spherical HFB (L2), deformed HFB (L3)
//!   Dense plasma — screened Coulomb bound states (Murillo & Weisheit 1998)

pub mod bcs_gpu;
pub mod constants;
pub mod hfb;
pub mod hfb_common;
pub mod hfb_deformed;
pub mod hfb_deformed_common;
pub mod hfb_deformed_gpu;
pub mod hfb_gpu;
pub mod hfb_gpu_resident;
pub(crate) mod hfb_gpu_types;
pub mod nuclear_matter;
pub mod screened_coulomb;
pub mod semf;

pub use constants::*;
pub use hfb::{binding_energy_l2, SphericalHFB};
pub use hfb_deformed::DeformedHFB;
pub use hfb_deformed_gpu::binding_energies_l3_gpu_auto;
pub use hfb_gpu::binding_energies_l2_gpu;
pub use hfb_gpu_resident::binding_energies_l2_gpu_resident;
pub use nuclear_matter::{nuclear_matter_properties, NuclearMatterProps};
pub use semf::semf_binding_energy;
