// SPDX-License-Identifier: AGPL-3.0-only

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
use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Spring Evolution Benchmark — barraCuda v0.3.5           ║");
    println!("║  hotSpring × wetSpring × neuralSpring × groundSpring           ║");
    println!("║  + coralReef FMA policy + stable GPU specials + precision tiers ║");
    println!("║  → barraCuda (math is universal, precision is silicon)          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

    // Phase 1: CPU special functions (no GPU needed)
    bench_special_functions_cpu();

    // Phase 1b: Spectral stats (CPU, upstream from hotSpring)
    bench_spectral_stats_cpu();

    // Phase 1c: Neighbor table precompute (CPU)
    bench_neighbor_precompute();

    // Phase 1d: FMA policy + precision tier routing (coralReef Iter 30)
    bench_fma_precision_routing();

    // Phase 1e: Stable GPU special functions (wetSpring+hotSpring)
    bench_stable_specials_cpu();

    // Phase 2: GPU ops (if available)
    let gpu_device = rt.block_on(try_create_device());
    if let Some(device) = gpu_device {
        println!("  GPU: {} (f64 capable)", device.adapter_info().name);
        println!();
        bench_vacf_gpu_vs_cpu(&device);
        run_guarded(
            "Autocorrelation GPU",
            panic::AssertUnwindSafe(|| {
                bench_autocorrelation_gpu(&device);
            }),
        );
        run_guarded(
            "Mean+Variance GPU",
            panic::AssertUnwindSafe(|| {
                bench_mean_variance_gpu(&device);
            }),
        );
        run_guarded(
            "Correlation GPU",
            panic::AssertUnwindSafe(|| {
                bench_correlation_gpu(&device);
            }),
        );
        run_guarded(
            "Chi-squared GPU",
            panic::AssertUnwindSafe(|| {
                bench_chi_squared_gpu(&device);
            }),
        );
        run_guarded(
            "Linear Regression GPU",
            panic::AssertUnwindSafe(|| {
                bench_linear_regression_gpu(&device);
            }),
        );
        run_guarded(
            "Matrix Correlation GPU",
            panic::AssertUnwindSafe(|| {
                bench_matrix_correlation_gpu(&device);
            }),
        );
        run_guarded(
            "Stress Virial GPU",
            panic::AssertUnwindSafe(|| {
                bench_stress_virial_gpu(&device);
            }),
        );
        run_guarded(
            "Batched Nelder-Mead GPU",
            panic::AssertUnwindSafe(|| {
                rt.block_on(bench_nelder_mead_gpu(&device));
            }),
        );
        run_guarded(
            "GemmF64 Transpose",
            panic::AssertUnwindSafe(|| {
                bench_gemm_transpose_gpu(&device);
            }),
        );
    } else {
        println!("  GPU unavailable — skipping GPU benchmarks");
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Benchmark complete — all cross-spring pathways exercised       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

fn run_guarded(label: &str, f: impl FnOnce() + panic::UnwindSafe) {
    if let Err(e) = panic::catch_unwind(f) {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        let short = if msg.len() > 120 { &msg[..120] } else { &msg };
        println!("  SKIP {label}: shader/driver incompatibility — {short}...");
        println!();
    }
}

async fn try_create_device() -> Option<Arc<WgpuDevice>> {
    println!("═══ Initializing GPU ═══");
    match WgpuDevice::new_f64_capable().await {
        Ok(dev) => Some(Arc::new(dev)),
        Err(_) => match WgpuDevice::new().await {
            Ok(dev) => Some(Arc::new(dev)),
            Err(_) => None,
        },
    }
}

// ── Phase 1: CPU Special Functions ──────────────────────────────────────────

fn bench_special_functions_cpu() {
    println!("═══ Phase 1: Special Functions (CPU) ═══");
    println!("  Provenance: hotSpring → toadStool S25–S68 (gamma, erf, bessel, hermite, laguerre)");
    println!();

    let n_evals = 100_000;

    // Gamma function
    let t = Instant::now();
    let mut gamma_sum = 0.0f64;
    for i in 1..=n_evals {
        let x = (f64::from(i) / f64::from(n_evals)).mul_add(10.0, 0.5);
        gamma_sum += barracuda::special::gamma(x).unwrap_or(0.0);
    }
    let gamma_us = t.elapsed().as_micros();
    println!(
        "  gamma(x)     : {n_evals} evals in {gamma_us} µs ({:.1} ns/eval) — checksum={gamma_sum:.6e}",
        gamma_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Error function
    let t = Instant::now();
    let mut erf_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)).mul_add(6.0, -3.0);
        erf_sum += barracuda::special::erf(x);
    }
    let erf_us = t.elapsed().as_micros();
    println!(
        "  erf(x)       : {n_evals} evals in {erf_us} µs ({:.1} ns/eval) — checksum={erf_sum:.6e}",
        erf_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Bessel J0
    let t = Instant::now();
    let mut j0_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)) * 20.0;
        j0_sum += barracuda::special::bessel_j0(x);
    }
    let j0_us = t.elapsed().as_micros();
    println!(
        "  bessel_j0(x) : {n_evals} evals in {j0_us} µs ({:.1} ns/eval) — checksum={j0_sum:.6e}",
        j0_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Hermite polynomial H_10(x) — nuclear HFB wavefunctions
    let t = Instant::now();
    let mut herm_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)).mul_add(8.0, -4.0);
        herm_sum += barracuda::special::hermite(10, x);
    }
    let herm_us = t.elapsed().as_micros();
    println!(
        "  hermite(10,x): {n_evals} evals in {herm_us} µs ({:.1} ns/eval) — checksum={herm_sum:.6e}",
        herm_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Laguerre polynomial L_5^(0.5)(x) — nuclear deformed basis
    let t = Instant::now();
    let mut lag_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)) * 10.0;
        lag_sum += barracuda::special::laguerre(5, 0.5, x);
    }
    let lag_us = t.elapsed().as_micros();
    println!(
        "  laguerre(5,x): {n_evals} evals in {lag_us} µs ({:.1} ns/eval) — checksum={lag_sum:.6e}",
        lag_us as f64 * 1000.0 / f64::from(n_evals)
    );

    println!();
}

// ── Phase 2a: VACF GPU vs CPU ───────────────────────────────────────────────

fn bench_vacf_gpu_vs_cpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2a: VACF — GPU (barracuda::ops::md) vs CPU ═══");
    println!("  Provenance: hotSpring MD transport → toadStool S70+ (batched ACF shader)");
    println!();

    for &n_atoms in &[64, 256, 1024] {
        let n_frames = 200;
        let n_lags = 100;

        let velocities: Vec<f64> = (0..n_frames * n_atoms * 3)
            .map(|i| (i as f64 * 0.7).sin() * 2.0)
            .collect();

        // GPU path (upstream barracuda)
        let t = Instant::now();
        let gpu_result =
            barracuda::ops::md::compute_vacf_batch(device, &velocities, n_atoms, n_frames, n_lags);
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        // CPU path (hotSpring local)
        let vel_snapshots: Vec<Vec<f64>> = velocities
            .chunks(n_atoms * 3)
            .map(<[f64]>::to_vec)
            .collect();
        let t = Instant::now();
        let cpu_result = hotspring_barracuda::md::observables::compute_vacf(
            &vel_snapshots,
            n_atoms,
            0.01,
            n_lags,
        );
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let speedup = cpu_ms / gpu_ms.max(0.001);
        let gpu_ok = if let Ok(ref g) = gpu_result {
            format!("C(0)={:.4e}", g[0])
        } else {
            "FAIL".to_string()
        };

        println!(
            "  N={n_atoms:>5}, {n_frames} frames, {n_lags} lags: GPU={gpu_ms:.1}ms CPU={cpu_ms:.1}ms (×{speedup:.1}) [{gpu_ok}, CPU D*={:.4e}]",
            cpu_result.diffusion_coeff
        );
    }
    println!();
}

// ── Phase 2b: Autocorrelation GPU (hotSpring+wetSpring → barraCuda) ──────────

fn bench_autocorrelation_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2b: Autocorrelation GPU (barracuda::ops::autocorrelation_f64_wgsl) ═══");
    println!("  Provenance: hotSpring batched VACF design + wetSpring time-series pattern");
    println!("  Cross-spring: single-dispatch C(lag) used by all springs for spectral analysis");
    println!();

    use barracuda::ops::autocorrelation_f64_wgsl::AutocorrelationF64;

    let acf_op =
        AutocorrelationF64::new(Arc::clone(device)).unwrap_or_else(|e| panic!("acf init: {e}"));

    for &(n, max_lag) in &[(1_000, 100), (10_000, 500), (100_000, 1_000)] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();

        let t = Instant::now();
        let result = acf_op.autocorrelation(&data, max_lag);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(c) => format!("C(0)={:.4e}, C(1)={:.4e}", c[0], c.get(1).unwrap_or(&0.0)),
            Err(e) => format!("ERR: {e}"),
        };

        println!("  n={n:>7}, lags={max_lag:>5}: {ms:.2}ms [{status}]");
    }
    println!();
}

// ── Phase 2c: Mean+Variance GPU (Kokkos/hotSpring → barraCuda) ──────────────

fn bench_mean_variance_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2c: Mean+Variance GPU (barracuda::ops::variance_f64_wgsl) ═══");
    println!("  Provenance: Kokkos parallel_reduce pattern, refined by hotSpring Welford");
    println!("  Cross-spring: all springs use for observable statistics (plaquette, energy, etc.)");
    println!();

    use barracuda::ops::variance_f64_wgsl::VarianceF64;

    let var_op = VarianceF64::new(Arc::clone(device)).unwrap_or_else(|e| panic!("var init: {e}"));

    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin() * 10.0).collect();

        // GPU
        let t = Instant::now();
        let gpu_result = var_op.mean_variance(&data, 1);
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        // CPU reference
        let t = Instant::now();
        let mean_cpu: f64 = data.iter().sum::<f64>() / n as f64;
        let var_cpu: f64 =
            data.iter().map(|x| (x - mean_cpu).powi(2)).sum::<f64>() / (n - 1) as f64;
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &gpu_result {
            Ok([mean, var]) => {
                let mean_err = (mean - mean_cpu).abs();
                let var_err = if var_cpu > 0.0 {
                    ((var - var_cpu) / var_cpu).abs()
                } else {
                    0.0
                };
                format!(
                    "GPU mean={mean:.6e} var={var:.6e} | Δmean={mean_err:.1e} Δvar={var_err:.1e}"
                )
            }
            Err(e) => format!("ERR: {e}"),
        };

        let speedup = cpu_ms / gpu_ms.max(0.001);
        println!("  n={n:>8}: GPU={gpu_ms:.2}ms CPU={cpu_ms:.2}ms (×{speedup:.1}) [{status}]");
    }
    println!();
}

// ── Phase 2d: Correlation GPU (Kokkos/wetSpring → barraCuda) ────────────────

fn bench_correlation_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2d: Correlation GPU (barracuda::ops::correlation_f64_wgsl) ═══");
    println!("  Provenance: Kokkos 5-accumulator Pearson, refined by wetSpring bio-stats");
    println!("  Cross-spring: groundSpring validation, neuralSpring model evaluation,");
    println!("    hotSpring observable correlations (plaquette vs Polyakov, KE vs PE)");
    println!();

    use barracuda::ops::correlation_f64_wgsl::CorrelationF64;

    let corr_op =
        CorrelationF64::new(Arc::clone(device)).unwrap_or_else(|e| panic!("corr init: {e}"));

    for &n in &[1_000, 10_000, 100_000] {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.01).sin() * 0.9 + (i as f64 * 0.03).cos() * 0.1)
            .collect();

        let t = Instant::now();
        let result = corr_op.correlation_full(&x, &y);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(cr) => format!(
                "r={:.6}, r²={:.6}, cov={:.4e}",
                cr.pearson_r,
                cr.r_squared(),
                cr.covariance()
            ),
            Err(e) => format!("ERR: {e}"),
        };

        println!("  n={n:>7}: {ms:.2}ms [{status}]");
    }
    println!();
}

// ── Phase 2e: Chi-squared GPU (groundSpring V74 → barraCuda) ────────────────

fn bench_chi_squared_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2e: Chi-squared GPU (barracuda::special::chi_squared) ═══");
    println!("  Provenance: groundSpring V74 noise validation → barraCuda S93");
    println!("  Cross-spring: hotSpring nuclear χ² fits, wetSpring enrichment,");
    println!("    neuralSpring model goodness-of-fit");
    println!();

    use barracuda::special::{chi_squared_cdf, chi_squared_quantile, chi_squared_statistic};

    // CPU path benchmarks
    for &n in &[10, 100, 1000] {
        let observed: Vec<f64> = (0..n)
            .map(|i| 10.0 + (i as f64 * 0.1).sin() * 2.0)
            .collect();
        let expected: Vec<f64> = vec![10.0; n];

        let t = Instant::now();
        let chi2 = chi_squared_statistic(&observed, &expected);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match chi2 {
            Ok(val) => {
                let df = (n - 1) as f64;
                let p = chi_squared_cdf(val, df).unwrap_or(f64::NAN);
                let q95 = chi_squared_quantile(0.95, df).unwrap_or(f64::NAN);
                format!("χ²={val:.2}, p={p:.4}, q95={q95:.2}, df={df}")
            }
            Err(e) => format!("ERR: {e}"),
        };

        println!("  n={n:>5}: {ms:.3}ms [{status}]");
    }

    // Fused GPU path (when available)
    use barracuda::ops::fused_chi_squared_f64::FusedChiSquaredGpu;
    let observed: Vec<f64> = (0..1000)
        .map(|i| 10.0 + (i as f64 * 0.1).sin() * 2.0)
        .collect();
    let expected: Vec<f64> = vec![10.0; 1000];
    let t = Instant::now();
    let gpu_result = FusedChiSquaredGpu::execute(Arc::clone(device), &observed, &expected);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    match gpu_result {
        Ok(r) => println!(
            "  GPU fused (n=1000): {ms:.2}ms [χ²={:.2}, p={:.4}]",
            r.statistic, r.p_value
        ),
        Err(e) => println!("  GPU fused: SKIP ({e})"),
    }
    println!();
}

// ── Phase 2f: Linear Regression GPU ─────────────────────────────────────────

fn bench_linear_regression_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2b: Linear Regression GPU (barracuda::ops::stats_f64) ═══");
    println!("  Provenance: neuralSpring baseCamp V18 → toadStool S25 (stats/linear_regression_f64.wgsl)");
    println!();

    for &(b, n, k) in &[(10, 100, 3), (100, 500, 5), (1000, 1000, 8)] {
        // Design matrix [b, n, k] and response [b, n]
        let x: Vec<f64> = (0..b * n * k)
            .map(|i| (f64::from(i) * 0.3).sin() + 1.0)
            .collect();
        let y: Vec<f64> = (0..b * n)
            .map(|i| (f64::from(i) * 0.5).cos() * 3.0)
            .collect();

        let t = Instant::now();
        let result = barracuda::ops::stats_f64::linear_regression(
            device, &x, &y, b as u32, n as u32, k as u32,
        );
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(coeffs) => format!("β[0]={:.4e}", coeffs[0]),
            Err(e) => format!("ERR: {e}"),
        };

        println!("  batch={b:>5}, n={n:>5}, k={k}: {ms:.1}ms [{status}]");
    }
    println!();
}

// ── Phase 2c: Matrix Correlation GPU ────────────────────────────────────────

fn bench_matrix_correlation_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2c: Matrix Correlation GPU (barracuda::ops::stats_f64) ═══");
    println!("  Provenance: neuralSpring baseCamp V18 → toadStool S25 (stats/matrix_correlation_f64.wgsl)");
    println!();

    for &(n, p) in &[(100, 10), (500, 20), (2000, 50)] {
        let data: Vec<f64> = (0..n * p)
            .map(|i| (f64::from(i) * 0.13).sin() * 5.0)
            .collect();

        let t = Instant::now();
        let result =
            barracuda::ops::stats_f64::matrix_correlation(device, &data, n as u32, p as u32);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(corr) => format!("R[0,0]={:.4}, shape={}×{}", corr[0], p, p),
            Err(e) => format!("ERR: {e}"),
        };

        println!("  n={n:>5}, p={p:>3}: {ms:.1}ms [{status}]");
    }
    println!();
}

// ── Phase 1b: Spectral Statistics (CPU) ──────────────────────────────────

fn bench_spectral_stats_cpu() {
    println!("═══ Phase 1b: Spectral Statistics (CPU) ═══");
    println!("  Provenance: hotSpring Anderson proxy → toadStool S78 (spectral/stats.rs)");
    println!("  Cross-spring: level_spacing_ratio born in hotSpring, absorbed into toadStool,");
    println!("    now used by wetSpring (bio spectral) and neuralSpring (RMT classifier)");
    println!();

    use barracuda::spectral::{
        anderson_3d, find_all_eigenvalues, lanczos, level_spacing_ratio, spectral_bandwidth,
        spectral_condition_number, SpectralAnalysis, GOE_R, POISSON_R,
    };

    for &(l, w, label) in &[
        (6_usize, 4.0_f64, "weak disorder (extended)"),
        (6, 16.0, "moderate disorder (critical)"),
        (6, 30.0, "strong disorder (localized)"),
        (8, 4.0, "8^3 weak disorder"),
        (10, 10.0, "10^3 moderate disorder"),
    ] {
        let t = Instant::now();
        let h = anderson_3d(l, l, l, w, 42);
        let n = l * l * l;
        let tri = lanczos(&h, n.min(h.n), 42);
        let eigs = find_all_eigenvalues(&tri.alpha, &tri.beta);
        let lanczos_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let r = level_spacing_ratio(&eigs);
        let bw = spectral_bandwidth(&eigs);
        let kappa = spectral_condition_number(&eigs);
        let gamma_est = if bw > 0.0 { n as f64 / bw } else { 1.0 };
        let analysis = SpectralAnalysis::from_eigenvalues(eigs.clone(), gamma_est);
        let stats_us = t.elapsed().as_micros();

        let phase_r = if r > 0.48 {
            "extended"
        } else if r < 0.42 {
            "localized"
        } else {
            "critical"
        };
        let phase_mp = format!("{:?}", analysis.phase);

        println!(
            "  L={l}, W={w:>4.0} ({label:30}): <r>={r:.4} (GOE={GOE_R:.4}, Poisson={POISSON_R:.4})"
        );
        println!("    BW={bw:.2}, kappa={kappa:.1e}, phase_r={phase_r}, phase_MP={phase_mp}");
        println!(
            "    Lanczos: {lanczos_ms:.1}ms | Stats: {stats_us}us | n_eig={}",
            eigs.len()
        );
    }
    println!();
}

// ── Phase 1c: Neighbor Table Precompute ────────────────────────────────────

fn bench_neighbor_precompute() {
    println!("═══ Phase 1c: Neighbor Table Precompute ═══");
    println!("  Provenance: hotSpring build_neighbors (HMC) -> toadStool S80 NeighborMode::precompute_periodic_4d");
    println!("  Note: hotSpring idx = t*V3 + x*V2 + y*Nz + z (z fastest)");
    println!("        toadStool idx = t*V3 + z*V2 + y*Nx + x (x fastest)");
    println!();

    use barracuda::ops::lattice::NeighborMode;

    for &dims in &[
        [4u32, 4, 4, 4],
        [8, 8, 8, 8],
        [12, 12, 12, 12],
        [16, 16, 16, 16],
    ] {
        let vol = dims.iter().product::<u32>() as usize;
        let t = Instant::now();
        let mode = NeighborMode::precompute_periodic_4d(dims);
        let us = t.elapsed().as_micros();

        let table_kb = match &mode {
            NeighborMode::PrecomputedBuffer(v) => v.len() * 4 / 1024,
            NeighborMode::OnTheFly => 0,
        };

        let t_hs = Instant::now();
        let lat = hotspring_barracuda::lattice::wilson::Lattice::cold_start(
            [
                dims[0] as usize,
                dims[1] as usize,
                dims[2] as usize,
                dims[3] as usize,
            ],
            6.0,
        );
        let hs_table = hotspring_barracuda::lattice::gpu_hmc::build_neighbors(&lat);
        let us_hs = t_hs.elapsed().as_micros();

        println!(
            "  {}^4 (vol={vol:>6}): toadStool={us:>5}us, hotSpring={us_hs:>5}us, table={table_kb}KB, entries={}",
            dims[0],
            hs_table.len()
        );
    }

    // Verify hotSpring table self-consistency (fwd(bwd(s)) == s)
    let lat = hotspring_barracuda::lattice::wilson::Lattice::cold_start([4, 4, 4, 4], 6.0);
    let nbr = hotspring_barracuda::lattice::gpu_hmc::build_neighbors(&lat);
    let vol = lat.volume();
    let mut ok = true;
    for s in 0..vol {
        for mu in 0..4 {
            let fwd = nbr[s * 8 + mu * 2] as usize;
            let bwd = nbr[s * 8 + mu * 2 + 1] as usize;
            if nbr[fwd * 8 + mu * 2 + 1] as usize != s {
                ok = false;
            }
            if nbr[bwd * 8 + mu * 2] as usize != s {
                ok = false;
            }
        }
    }
    println!(
        "  Inverse consistency (4^4 hotSpring): {}",
        if ok { "PASS" } else { "FAIL" }
    );
    println!();
}

// ── Phase 1d: FMA Policy + Precision Tier Routing ──────────────────────────

fn bench_fma_precision_routing() {
    println!("═══ Phase 1d: FMA Policy + Precision Tier Routing ═══");
    println!("  Provenance: hotSpring v0.6.25 precision brain → barraCuda Sprint 2");
    println!("  coralReef Iter 30: FmaPolicy::Separate splits fma→mul+add for bit-exact QCD");
    println!("  Cross-spring: all springs benefit from domain-aware precision routing");
    println!();

    use barracuda::device::fma_policy::{domain_requires_separate_fma, FmaPolicy};
    use barracuda::device::precision_tier::{PhysicsDomain, PrecisionTier};

    let domains = [
        (PhysicsDomain::LatticeQcd, "LatticeQcd"),
        (PhysicsDomain::GradientFlow, "GradientFlow"),
        (PhysicsDomain::NuclearEos, "NuclearEOS"),
        (PhysicsDomain::MolecularDynamics, "MolecularDynamics"),
        (PhysicsDomain::Dielectric, "Dielectric"),
        (PhysicsDomain::KineticFluid, "KineticFluid"),
        (PhysicsDomain::Statistics, "Statistics"),
        (PhysicsDomain::Bioinformatics, "Bioinformatics"),
    ];

    for (domain, label) in &domains {
        let needs_separate = domain_requires_separate_fma(domain);
        let policy = if needs_separate {
            FmaPolicy::Separate
        } else {
            FmaPolicy::Contract
        };
        println!("  {label:20} → FMA={policy}, separate_required={needs_separate}");
    }

    println!();

    let tiers = [
        PrecisionTier::F32,
        PrecisionTier::DF64,
        PrecisionTier::F64,
        PrecisionTier::F64Precise,
    ];

    for tier in &tiers {
        println!(
            "  {tier:12} → {bits} mantissa bits",
            bits = tier.mantissa_bits()
        );
    }

    println!();
}

// ── Phase 1e: Stable GPU Special Functions ─────────────────────────────────

fn bench_stable_specials_cpu() {
    println!("═══ Phase 1e: Stable GPU Special Functions (CPU reference) ═══");
    println!("  Provenance: wetSpring+hotSpring → barraCuda Sprint 2");
    println!("  Cross-spring: log1p/expm1/erfc/J₀-1 avoid catastrophic cancellation");
    println!("  hotSpring uses: screened Coulomb (erfc), dielectric (log1p), BCS (expm1)");
    println!("  wetSpring uses: HMM log-domain (log1p), diversity (erfc)");
    println!();

    use barracuda::special::stable_gpu::{bessel_j0_minus1_f64, erfc_f64, expm1_f64, log1p_f64};

    let n = 100_000;

    let t = Instant::now();
    let mut sum = 0.0_f64;
    for i in 0..n {
        sum += log1p_f64(f64::from(i) * 1e-10);
    }
    let us = t.elapsed().as_micros();
    println!("  log1p(x)      : {n} evals in {us}µs — checksum={sum:.6e}");

    let t = Instant::now();
    sum = 0.0;
    for i in 0..n {
        sum += expm1_f64(f64::from(i) * 1e-10);
    }
    let us = t.elapsed().as_micros();
    println!("  expm1(x)      : {n} evals in {us}µs — checksum={sum:.6e}");

    let t = Instant::now();
    sum = 0.0;
    for i in 0..n {
        let x = f64::from(i) / f64::from(n) * 6.0;
        sum += erfc_f64(x);
    }
    let us = t.elapsed().as_micros();
    println!("  erfc(x)       : {n} evals in {us}µs — checksum={sum:.6e}");

    let t = Instant::now();
    sum = 0.0;
    for i in 0..n {
        let x = f64::from(i) / f64::from(n) * 0.1;
        sum += bessel_j0_minus1_f64(x);
    }
    let us = t.elapsed().as_micros();
    println!("  J₀(x)-1       : {n} evals in {us}µs — checksum={sum:.6e}");

    // Validate stable vs naive near cancellation
    let x_small = 1e-14;
    let stable = log1p_f64(x_small);
    let naive = (1.0 + x_small).ln();
    let rel_err = if stable.abs() > 0.0 {
        ((stable - naive) / stable).abs()
    } else {
        0.0
    };
    println!();
    println!("  Cancellation test: log1p(1e-14)={stable:.6e}, ln(1+1e-14)={naive:.6e}, rel_err={rel_err:.2e}");
    println!(
        "  → stable wins: {}",
        if rel_err < 1e-10 {
            "both accurate at this level"
        } else {
            "stable avoids cancellation"
        }
    );
    println!();
}

// ── Phase 2d: Stress Virial GPU ─────────────────────────────────────────────

fn bench_stress_virial_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2d: Stress Virial GPU (barracuda::ops::md) ═══");
    println!("  Provenance: hotSpring MD transport → toadStool S70+ (ComputeDispatch)");
    println!();

    for &n_atoms in &[100, 500, 2000] {
        let positions: Vec<f64> = (0..n_atoms * 3)
            .map(|i| (i as f64 * 0.01).sin() * 5.0)
            .collect();
        let velocities: Vec<f64> = (0..n_atoms * 3)
            .map(|i| (i as f64 * 0.02).cos() * 0.5)
            .collect();
        let forces: Vec<f64> = (0..n_atoms * 3)
            .map(|i| (i as f64 * 0.03).sin() * 0.1)
            .collect();
        let masses: Vec<f64> = vec![1.0; n_atoms];
        let volume = 1000.0;

        let t = Instant::now();
        let result = barracuda::ops::md::compute_stress_virial(
            device,
            &positions,
            &velocities,
            &forces,
            &masses,
            volume,
        );
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(sigma) => format!("σ_xx={:.4e}, σ_xy={:.4e}", sigma[0], sigma[3]),
            Err(e) => format!("ERR: {e}"),
        };

        println!("  N={n_atoms:>5}: {ms:.1}ms [{status}]");
    }
    println!();
}

// ── Phase 2e: Batched Nelder-Mead GPU ───────────────────────────────────────

async fn bench_nelder_mead_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2e: Batched Nelder-Mead GPU (barracuda::optimize) ═══");
    println!("  Provenance: neuralSpring parameter optimization → toadStool S79");
    println!("  Cross-spring: hotSpring HMC parameter tuning benefits from GPU batch optimizer");
    println!();

    use barracuda::optimize::{batched_nelder_mead_gpu, BatchNelderMeadConfig};

    for &(n_problems, dims) in &[(10_usize, 2_usize), (100, 3), (1000, 2)] {
        let config = BatchNelderMeadConfig {
            dims,
            max_iters: 200,
            tol: 1e-8,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        };

        let n_vertices = dims + 1;
        let simplices: Vec<f64> = (0..n_problems * n_vertices * dims)
            .map(|i| {
                let problem = i / (n_vertices * dims);
                let vertex = (i / dims) % n_vertices;
                let dim = i % dims;
                if vertex == 0 {
                    (problem as f64 * 0.1).sin()
                } else if dim == vertex - 1 {
                    1.0 + (problem as f64 * 0.1).sin()
                } else {
                    (problem as f64 * 0.1).sin()
                }
            })
            .collect();

        let t = Instant::now();
        let result = batched_nelder_mead_gpu(device, &config, n_problems, &simplices, |points| {
            points
                .chunks(dims)
                .map(|p| p.iter().map(|x| x * x).sum::<f64>())
                .collect()
        })
        .await;
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(results) => {
                let converged = results.iter().filter(|r| r.converged).count();
                let best = results.first().map_or(f64::NAN, |r| r.best_value);
                format!("{converged}/{n_problems} converged, best={best:.2e}")
            }
            Err(e) => format!("ERR: {e}"),
        };

        println!("  n={n_problems:>5}, dims={dims}: {ms:.1}ms [{status}]");
    }
    println!();
}

// ── Phase 2f: GemmF64 Transpose (neuralSpring → barraCuda Sprint 6) ─────────

fn bench_gemm_transpose_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2f: GemmF64 Transpose (barracuda::ops::linalg::GemmF64) ═══");
    println!("  Provenance: neuralSpring Tikhonov/least-squares → barraCuda Sprint 6");
    println!("  Cross-spring: A^T*B without materializing transpose — Gram matrices,");
    println!("    normal equations, covariance. Used by neuralSpring regression,");
    println!("    hotSpring surrogate fitting, groundSpring least-squares");
    println!();

    use barracuda::ops::linalg::gemm_f64::GemmF64;

    for &(m, k, n) in &[(64_usize, 128, 32), (256, 512, 64), (512, 1024, 128)] {
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.01).sin()).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.02).cos()).collect();

        // Standard A*B
        let t = Instant::now();
        let result_ab = GemmF64::execute(Arc::clone(device), &a, &b, m, k, n, 1);
        let ms_ab = t.elapsed().as_secs_f64() * 1000.0;

        // A^T*B (storage: k×m, logically transposed to m×k, then multiplied by k×n)
        let a_for_trans: Vec<f64> = (0..k * m).map(|i| (i as f64 * 0.01).sin()).collect();
        let t = Instant::now();
        let result_atb = GemmF64::execute_gemm_ex(
            Arc::clone(device),
            &a_for_trans,
            &b,
            m,
            k,
            n,
            1,
            1.0,
            0.0,
            true,
            false,
        );
        let ms_atb = t.elapsed().as_secs_f64() * 1000.0;

        let ab_ok = result_ab
            .as_ref()
            .map_or_else(|e| format!("ERR: {e}"), |v| format!("ok, len={}", v.len()));
        let atb_ok = result_atb
            .as_ref()
            .map_or_else(|e| format!("ERR: {e}"), |v| format!("ok, len={}", v.len()));
        println!(
            "  {m}×{k} * {k}×{n}: A*B={ms_ab:.1}ms [{ab_ok}] | A^T*B={ms_atb:.1}ms [{atb_ok}]"
        );
    }
    println!();
}
