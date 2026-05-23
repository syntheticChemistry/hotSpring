// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-Spring Evolution Benchmark
//!
//! Exercises GPU/CPU ops that evolved through cross-spring absorption in
//! toadStool/barracuda, benchmarks modern vs legacy paths, and documents
//! the provenance of each shader/primitive.
//!
//! # Cross-Spring Shader Provenance (synced to barraCuda v0.3.5, coralReef Iter 54)
//!
//! | Op | Origin Spring | Session | Notes |
//! |----|---------------|---------|-------|
//! | VACF batch GPU | hotSpring | S70+ | Batched ACF design from MD transport |
//! | Stress virial GPU | hotSpring | S70+ | Green-Kubo viscosity building block |
//! | Autocorrelation GPU | hotSpring+wetSpring | S96 | General 1D ACF — single dispatch |
//! | Mean+Variance GPU | Kokkos/hotSpring | S86+ | Welford single-pass fused shader |
//! | Correlation GPU | Kokkos/wetSpring | S86+ | 5-accumulator Pearson single-pass |
//! | Chi-squared GPU | groundSpring V74 | S93 | CDF + quantile via regularized gamma |
//! | Linear regression GPU | neuralSpring | S25 | `stats/linear_regression_f64.wgsl` |
//! | Matrix correlation GPU | neuralSpring | S25 | `stats/matrix_correlation_f64.wgsl` |
//! | DF64 transcendentals | hotSpring+wetSpring | S71+ | gamma_f64, erf_f64, trig_f64, DF64 |
//! | Special functions CPU | hotSpring | S25–S68 | gamma, erf, bessel, hermite, laguerre |
//! | Spectral stats (⟨r⟩) | hotSpring→barraCuda | S78 | level_spacing_ratio, bandwidth, κ |
//! | SpectralAnalysis+RMT | hotSpring→barraCuda | S78 | Marchenko–Pastur phase classifier |
//! | Anderson 3D proxy | hotSpring | S25–S78 | Lanczos + level statistics + CG predictor |
//! | NeighborMode 4D | hotSpring→barraCuda | S80 | Precomputed periodic neighbor table |
//! | Batched Nelder-Mead GPU | neuralSpring | S79 | Parallel optimization on GPU |
//! | Nautilus brain | hotSpring+neuralSpring | S79 | Evolutionary reservoir for QCD steering |
//! | Nuclear shaders (7) | hotSpring | S93+ | SEMF, chi2, deformed HFB, spin-orbit |
//! | Bio HMM GPU | wetSpring+neuralSpring | S93 | Log-domain forward/backward |
//! | FFT radix-2 GPU | groundSpring | S93 | Cooley-Tukey f64 butterfly |
//! | **FmaPolicy** | hotSpring+coralReef | Iter30 | FMA contraction control for precision |
//! | **Stable GPU specials** | wetSpring+hotSpring | Sprint2 | log1p, expm1, erfc, bessel_j0-1 |
//! | **GemmF64 transpose** | neuralSpring | Sprint6 | A^T*B without materializing transpose |
//! | **PrecisionTier+Domain** | hotSpring v0.6.25 | Sprint2 | F32/DF64/F64/F64Precise routing |

use std::panic;

#[path = "../bin_helpers/cross_spring/mod.rs"]
mod cross_spring;

use cross_spring::{
    bench_autocorrelation_gpu, bench_chi_squared_gpu, bench_correlation_gpu,
    bench_fma_precision_routing, bench_gemm_transpose_gpu, bench_linear_regression_gpu,
    bench_matrix_correlation_gpu, bench_mean_variance_gpu, bench_neighbor_precompute,
    bench_nelder_mead_gpu, bench_special_functions_cpu, bench_spectral_stats_cpu,
    bench_stable_specials_cpu, bench_stress_virial_gpu, bench_vacf_gpu_vs_cpu, run_guarded,
    try_create_device,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Spring Evolution Benchmark — barraCuda v0.3.5           ║");
    println!("║  hotSpring × wetSpring × neuralSpring × groundSpring           ║");
    println!("║  + coralReef FMA policy + stable GPU specials + precision tiers ║");
    println!("║  → barraCuda (math is universal, precision is silicon)          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

    bench_special_functions_cpu();
    bench_spectral_stats_cpu();
    bench_neighbor_precompute();
    bench_fma_precision_routing();
    bench_stable_specials_cpu();

    let gpu_device = rt.block_on(try_create_device());
    if let Some(device) = gpu_device {
        println!("  GPU: {} (f64 capable)", device.adapter_info().name);
        println!();
        bench_vacf_gpu_vs_cpu(&device);
        run_guarded(
            "Autocorrelation GPU",
            panic::AssertUnwindSafe(|| bench_autocorrelation_gpu(&device)),
        );
        run_guarded(
            "Mean+Variance GPU",
            panic::AssertUnwindSafe(|| bench_mean_variance_gpu(&device)),
        );
        run_guarded(
            "Correlation GPU",
            panic::AssertUnwindSafe(|| bench_correlation_gpu(&device)),
        );
        run_guarded(
            "Chi-squared GPU",
            panic::AssertUnwindSafe(|| bench_chi_squared_gpu(&device)),
        );
        run_guarded(
            "Linear Regression GPU",
            panic::AssertUnwindSafe(|| bench_linear_regression_gpu(&device)),
        );
        run_guarded(
            "Matrix Correlation GPU",
            panic::AssertUnwindSafe(|| bench_matrix_correlation_gpu(&device)),
        );
        run_guarded(
            "Stress Virial GPU",
            panic::AssertUnwindSafe(|| bench_stress_virial_gpu(&device)),
        );
        run_guarded(
            "Batched Nelder-Mead GPU",
            panic::AssertUnwindSafe(|| {
                rt.block_on(bench_nelder_mead_gpu(&device));
            }),
        );
        run_guarded(
            "GemmF64 Transpose",
            panic::AssertUnwindSafe(|| bench_gemm_transpose_gpu(&device)),
        );
    } else {
        println!("  GPU unavailable — skipping GPU benchmarks");
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Benchmark complete — all cross-spring pathways exercised       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
