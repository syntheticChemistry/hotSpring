// SPDX-License-Identifier: AGPL-3.0-or-later

//! Helpers extracted from `nuclear_eos_gpu`: L1 SEMF validation/optimization and L2 HFB phases.

pub mod hetero;
pub mod l1_semf;
pub mod l2_hfb;
pub mod l2_pipeline;

pub use hetero::{
    CliConfig, HeteroResult, parse_cli, run_direct_l2, run_hetero_mode, run_heterogeneous_l2,
    run_plain_l2, run_screen_l2,
};
pub use l1_semf::{
    L1DirectSamplerResult, L1SemfResult, l1_chi2_gpu, run_l1_direct_sampler, run_l1_lhs_sweep,
    run_l1_semf_cpu_vs_gpu, run_l1_semf_pure_gpu,
};
pub use l2_hfb::{L2HfbResult, print_summary_table, run_l2_direct_sampler, run_l2_hfb_baseline};
pub use l2_pipeline::{
    L2PipelineCli, SeedResult, compute_l2_binding_energies, decompose_chi2, l1_objective_nmp,
    l2_objective_nmp, run_l2_seed_pipeline, run_statistical_analysis, validate_nmp_constraints,
};
