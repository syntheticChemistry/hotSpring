// SPDX-License-Identifier: AGPL-3.0-or-later

//! Overnight Chuna validation: substrate harness types and paper 43/44/45 suites.

mod harness;
mod papers;

pub use harness::{SubstrateResults, max_lattice_l};
pub use papers::{
    cpu_quenched_pretherm, paper_43_convergence, paper_43_dynamical, paper_43_production,
    paper_44_cpu, paper_44_gpu, paper_44_multicomponent_cpu, paper_44_multicomponent_gpu,
    paper_45_gpu_bgk, paper_45_gpu_coupled, paper_45_gpu_euler,
};
