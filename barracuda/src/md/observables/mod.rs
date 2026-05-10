// SPDX-License-Identifier: AGPL-3.0-or-later

//! Observable computation for MD validation
//!
//! Computes RDF, VACF, SSF, and energy metrics from simulation snapshots.
//! CPU path computes from GPU-generated snapshots.
//! GPU path uses toadstool's `SsfGpu` for O(N) GPU-accelerated S(k).

#[cfg(feature = "barracuda-local")]
pub mod energy;
pub mod rdf;
#[cfg(feature = "barracuda-local")]
pub mod ssf;
#[cfg(feature = "barracuda-local")]
pub mod summary;
#[cfg(feature = "barracuda-local")]
pub mod transport;
#[cfg(feature = "barracuda-local")]
pub mod transport_gpu;
#[cfg(feature = "barracuda-local")]
pub mod vacf;

#[cfg(feature = "barracuda-local")]
pub use energy::{EnergyValidation, validate_energy};
pub use rdf::{Rdf, compute_rdf};
#[cfg(feature = "barracuda-local")]
pub use ssf::{compute_ssf, compute_ssf_gpu};
#[cfg(feature = "barracuda-local")]
pub use summary::{print_observable_summary, print_observable_summary_with_gpu};
#[cfg(feature = "barracuda-local")]
pub use transport::{
    HeatAcf, StressAcf, compute_heat_acf, compute_heat_current, compute_stress_acf,
    compute_stress_xy,
};
#[cfg(feature = "barracuda-local")]
pub use transport_gpu::{
    GpuVacf, GpuVelocityRing, WGSL_STRESS_VIRIAL_F64, WGSL_VACF_BATCH_F64, WGSL_VACF_DOT_F64,
    compute_stress_xy_gpu, compute_vacf_gpu,
};
#[cfg(feature = "barracuda-local")]
pub use vacf::{Vacf, compute_d_star_msd, compute_vacf, compute_vacf_upstream_gpu};
