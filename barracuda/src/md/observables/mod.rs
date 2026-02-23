// SPDX-License-Identifier: AGPL-3.0-only

//! Observable computation for MD validation
//!
//! Computes RDF, VACF, SSF, and energy metrics from simulation snapshots.
//! CPU path computes from GPU-generated snapshots.
//! GPU path uses toadstool's `SsfGpu` for O(N) GPU-accelerated S(k).

pub mod energy;
pub mod rdf;
pub mod ssf;
pub mod summary;
pub mod transport;
pub mod transport_gpu;
pub mod vacf;

pub use energy::{validate_energy, EnergyValidation};
pub use rdf::{compute_rdf, Rdf};
pub use ssf::{compute_ssf, compute_ssf_gpu};
pub use summary::{print_observable_summary, print_observable_summary_with_gpu};
pub use transport::{
    compute_heat_acf, compute_heat_current, compute_stress_acf, compute_stress_xy, HeatAcf,
    StressAcf,
};
pub use transport_gpu::{
    compute_stress_xy_gpu, compute_vacf_gpu, GpuVacf, GpuVelocityRing, WGSL_STRESS_VIRIAL_F64,
    WGSL_VACF_BATCH_F64, WGSL_VACF_DOT_F64,
};
pub use vacf::{compute_d_star_msd, compute_vacf, Vacf};
