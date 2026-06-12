// SPDX-License-Identifier: AGPL-3.0-or-later

//! QCD silicon benchmark helpers: WGSL proxy kernels and result reporting.

pub mod kernels;
pub mod report;

pub use kernels::bench_kernel;
pub use report::{
    KernelSpec, classify_silicon_opportunity, df64_kernel_specs, fp32_kernel_specs,
    print_trajectory_cost_model,
};
