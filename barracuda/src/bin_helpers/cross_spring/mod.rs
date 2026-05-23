// SPDX-License-Identifier: AGPL-3.0-or-later

//! Helpers extracted from `bench_cross_spring_evolution`: per-domain CPU/GPU benchmarks
//! and shared support utilities.

pub mod cpu_misc;
pub mod cpu_specials;
pub mod cpu_stats;
pub mod gpu_linalg;
pub mod gpu_md;
pub mod gpu_optimize;
pub mod gpu_stats;
pub mod support;

pub use cpu_misc::{bench_fma_precision_routing, bench_neighbor_precompute};
pub use cpu_specials::{bench_special_functions_cpu, bench_stable_specials_cpu};
pub use cpu_stats::bench_spectral_stats_cpu;
pub use gpu_linalg::bench_gemm_transpose_gpu;
pub use gpu_md::{bench_stress_virial_gpu, bench_vacf_gpu_vs_cpu};
pub use gpu_optimize::bench_nelder_mead_gpu;
pub use gpu_stats::{
    bench_autocorrelation_gpu, bench_chi_squared_gpu, bench_correlation_gpu,
    bench_linear_regression_gpu, bench_matrix_correlation_gpu, bench_mean_variance_gpu,
};
pub use support::{run_guarded, try_create_device};
