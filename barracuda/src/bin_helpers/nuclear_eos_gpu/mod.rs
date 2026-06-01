// SPDX-License-Identifier: AGPL-3.0-or-later

//! Helpers extracted from `nuclear_eos_gpu`: L1 SEMF validation/optimization and L2 HFB phases.

pub mod l1_semf;
pub mod l2_hfb;
pub mod hetero;

pub use l1_semf::{
    l1_chi2_gpu, run_l1_direct_sampler, run_l1_lhs_sweep, run_l1_semf_cpu_vs_gpu,
    run_l1_semf_pure_gpu, L1DirectSamplerResult, L1SemfResult,
};
pub use l2_hfb::{print_summary_table, run_l2_direct_sampler, run_l2_hfb_baseline, L2HfbResult};
pub use hetero::{
    parse_cli, run_direct_l2, run_hetero_mode, run_heterogeneous_l2, run_plain_l2, run_screen_l2,
    CliConfig, HeteroResult,
};
