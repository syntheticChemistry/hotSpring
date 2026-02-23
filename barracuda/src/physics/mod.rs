// SPDX-License-Identifier: AGPL-3.0-only

//! Physics modules: nuclear structure, atomic physics, dense plasma theory.
//!
//! Direct Rust ports of the Python control implementations.
//! Uses `BarraCuda` special functions (`gamma`, `laguerre`) and numerical
//! methods (`gradient_1d`, `trapz`) wherever possible.
//!
//! Domains:
//!   Nuclear EOS — SEMF (L1), spherical HFB (L2), deformed HFB (L3)
//!   Dense plasma — screened Coulomb bound states (Murillo & Weisheit 1998)

/// BCS pairing gap solver on GPU (bisection in WGSL).
pub mod bcs_gpu;
/// Physical constants (CODATA 2018, nuclear, Skyrme).
pub mod constants;
/// Spherical Hartree-Fock-Bogoliubov (L2 nuclear EOS).
pub mod hfb;
/// Shared HFB utilities: Woods-Saxon radii, deformation estimation.
pub mod hfb_common;
/// Deformed HFB in axial symmetry (L3 nuclear EOS).
pub mod hfb_deformed;
/// Shared constants and types for deformed HFB (CPU + GPU).
pub mod hfb_deformed_common;
/// Deformed HFB with GPU-accelerated Hamiltonian build.
pub mod hfb_deformed_gpu;
/// GPU-batched HFB eigensolve via `BatchedEighGpu`.
pub mod hfb_gpu;
/// Fully GPU-resident HFB pipeline (potentials → H → eigensolve → density → energy).
pub mod hfb_gpu_resident;
/// GPU buffer types and bind-group layouts for HFB pipelines.
pub(crate) mod hfb_gpu_types;
/// Nuclear matter properties at saturation (NMP): E/A, K, S, L.
pub mod nuclear_matter;
/// Screened Coulomb bound states (Murillo & Weisheit 1998).
pub mod screened_coulomb;
/// Semi-Empirical Mass Formula (L1 nuclear EOS, von Weizsacker/Evans).
pub mod semf;

pub use constants::*;
pub use hfb::{binding_energy_l2, SphericalHFB};
pub use hfb_deformed::DeformedHFB;
pub use hfb_deformed_gpu::binding_energies_l3_gpu_auto;
pub use hfb_gpu::binding_energies_l2_gpu;
pub use hfb_gpu_resident::binding_energies_l2_gpu_resident;
pub use nuclear_matter::{nuclear_matter_properties, NuclearMatterProps};
pub use semf::semf_binding_energy;
