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

/// Average-atom model for warm dense matter (Paper 33 — atoMEC).
pub mod average_atom;
/// BCS pairing gap solver on GPU (bisection in WGSL).
pub mod bcs_gpu;
/// Physical constants (CODATA 2018, nuclear, Skyrme).
pub mod constants;
/// BGK dielectric functions (Chuna & Murillo 2024, arXiv:2405.07871).
pub mod dielectric;
/// Multi-component Mermin dielectric for electron-ion plasmas (Paper 44 extension).
pub mod dielectric_multicomponent;
/// Militzer FPEOS database: first-principles EOS tables for WDM (Paper 32).
pub mod fpeos;
/// Full GPU coupled kinetic-fluid pipeline (Paper 45).
pub mod gpu_coupled_kinetic_fluid;
/// GPU-accelerated BGK dielectric (batched Mermin on GPU).
pub mod gpu_dielectric;
/// GPU-accelerated multi-component Mermin dielectric (Paper 44 extension).
pub mod gpu_dielectric_multicomponent;
/// GPU-accelerated 1D Euler with HLL Riemann solver (Paper 45).
pub mod gpu_euler;
/// GPU-accelerated BGK relaxation for multi-species kinetic plasma (Paper 45).
pub mod gpu_kinetic_fluid;
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
/// Multi-species kinetic-fluid coupling (Haack, Murillo, Sagert & Chuna 2024).
pub mod kinetic_fluid;
/// Nuclear matter properties at saturation (NMP): E/A, K, S, L.
pub mod nuclear_matter;
/// Screened Coulomb bound states (Murillo & Weisheit 1998).
pub mod screened_coulomb;
/// Semi-Empirical Mass Formula (L1 nuclear EOS, von Weizsacker/Evans).
pub mod semf;

pub use constants::*;
pub use hfb::{SphericalHFB, binding_energy_l2};
pub use hfb_deformed::DeformedHFB;
pub use hfb_deformed_gpu::binding_energies_l3_gpu_auto;
pub use hfb_gpu::binding_energies_l2_gpu;
pub use hfb_gpu_resident::binding_energies_l2_gpu_resident;
pub use nuclear_matter::{NuclearMatterProps, nuclear_matter_properties};
pub use semf::semf_binding_energy;
