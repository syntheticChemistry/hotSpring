// SPDX-License-Identifier: AGPL-3.0-or-later

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

pub fn bench_autocorrelation_gpu(device: &Arc<WgpuDevice>) {
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

pub fn bench_mean_variance_gpu(device: &Arc<WgpuDevice>) {
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

pub fn bench_correlation_gpu(device: &Arc<WgpuDevice>) {
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

pub fn bench_chi_squared_gpu(device: &Arc<WgpuDevice>) {
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

pub fn bench_linear_regression_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2b: Linear Regression GPU (barracuda::ops::stats_f64) ═══");
    println!(
        "  Provenance: neuralSpring baseCamp V18 → toadStool S25 (stats/linear_regression_f64.wgsl)"
    );
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

pub fn bench_matrix_correlation_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2c: Matrix Correlation GPU (barracuda::ops::stats_f64) ═══");
    println!(
        "  Provenance: neuralSpring baseCamp V18 → toadStool S25 (stats/matrix_correlation_f64.wgsl)"
    );
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
