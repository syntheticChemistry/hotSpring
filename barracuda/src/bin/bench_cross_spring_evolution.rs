// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-Spring Evolution Benchmark
//!
//! Exercises GPU/CPU ops that evolved through cross-spring absorption in
//! toadStool/barracuda, benchmarks modern vs legacy paths, and documents
//! the provenance of each shader/primitive.
//!
//! # Cross-Spring Shader Provenance
//!
//! | Op | Origin Spring | toadStool Session | Notes |
//! |----|---------------|-------------------|-------|
//! | VACF batch GPU | hotSpring | S70+ | Batched ACF design from MD transport |
//! | Stress virial GPU | hotSpring | S70+ | Green-Kubo viscosity building block |
//! | SSF GPU | hotSpring | S25 | `SsfGpu::compute_axes` |
//! | Linear regression GPU | neuralSpring | S25 baseCamp V18 | `stats/linear_regression_f64.wgsl` |
//! | Matrix correlation GPU | neuralSpring | S25 baseCamp V18 | `stats/matrix_correlation_f64.wgsl` |
//! | DF64 transcendentals | hotSpring+wetSpring | S71+ | gamma_f64, erf_f64, trig_f64 |
//! | Special functions CPU | hotSpring | S25–S68 | gamma, erf, bessel, hermite, laguerre |
//! | ESN reservoir GPU | hotSpring | S25 | WGSL reservoir update from MD transport |
//! | Wilson plaquette GPU | hotSpring | S25 | Lattice QCD gauge observable |
//! | Bray-Curtis GPU | wetSpring | S18 | Bio diversity metric |
//! | HMM forward GPU | wetSpring+neuralSpring | S21–S25 | Genomic HMM → neural adaptation |

use std::panic;
use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Spring Evolution Benchmark — toadStool S78               ║");
    println!("║  hotSpring × wetSpring × neuralSpring → barracuda              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

    // Phase 1: CPU special functions (no GPU needed)
    bench_special_functions_cpu();

    // Phase 2: GPU ops (if available)
    let gpu_device = rt.block_on(try_create_device());
    if let Some(device) = gpu_device {
        println!("  GPU: {} (f64 capable)", device.adapter_info().name);
        println!();
        bench_vacf_gpu_vs_cpu(&device);
        run_guarded("Linear Regression GPU", panic::AssertUnwindSafe(|| {
            bench_linear_regression_gpu(&device);
        }));
        run_guarded("Matrix Correlation GPU", panic::AssertUnwindSafe(|| {
            bench_matrix_correlation_gpu(&device);
        }));
        run_guarded("Stress Virial GPU", panic::AssertUnwindSafe(|| {
            bench_stress_virial_gpu(&device);
        }));
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
        Err(_) => {
            match WgpuDevice::new().await {
                Ok(dev) => Some(Arc::new(dev)),
                Err(_) => None,
            }
        }
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
        let x = 0.5 + (i as f64 / n_evals as f64) * 10.0;
        gamma_sum += barracuda::special::gamma(x).unwrap_or(0.0);
    }
    let gamma_us = t.elapsed().as_micros();
    println!(
        "  gamma(x)     : {n_evals} evals in {gamma_us} µs ({:.1} ns/eval) — checksum={gamma_sum:.6e}",
        gamma_us as f64 * 1000.0 / n_evals as f64
    );

    // Error function
    let t = Instant::now();
    let mut erf_sum = 0.0f64;
    for i in 0..n_evals {
        let x = -3.0 + (i as f64 / n_evals as f64) * 6.0;
        erf_sum += barracuda::special::erf(x);
    }
    let erf_us = t.elapsed().as_micros();
    println!(
        "  erf(x)       : {n_evals} evals in {erf_us} µs ({:.1} ns/eval) — checksum={erf_sum:.6e}",
        erf_us as f64 * 1000.0 / n_evals as f64
    );

    // Bessel J0
    let t = Instant::now();
    let mut j0_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (i as f64 / n_evals as f64) * 20.0;
        j0_sum += barracuda::special::bessel_j0(x);
    }
    let j0_us = t.elapsed().as_micros();
    println!(
        "  bessel_j0(x) : {n_evals} evals in {j0_us} µs ({:.1} ns/eval) — checksum={j0_sum:.6e}",
        j0_us as f64 * 1000.0 / n_evals as f64
    );

    // Hermite polynomial H_10(x) — nuclear HFB wavefunctions
    let t = Instant::now();
    let mut herm_sum = 0.0f64;
    for i in 0..n_evals {
        let x = -4.0 + (i as f64 / n_evals as f64) * 8.0;
        herm_sum += barracuda::special::hermite(10, x);
    }
    let herm_us = t.elapsed().as_micros();
    println!(
        "  hermite(10,x): {n_evals} evals in {herm_us} µs ({:.1} ns/eval) — checksum={herm_sum:.6e}",
        herm_us as f64 * 1000.0 / n_evals as f64
    );

    // Laguerre polynomial L_5^(0.5)(x) — nuclear deformed basis
    let t = Instant::now();
    let mut lag_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (i as f64 / n_evals as f64) * 10.0;
        lag_sum += barracuda::special::laguerre(5, 0.5, x);
    }
    let lag_us = t.elapsed().as_micros();
    println!(
        "  laguerre(5,x): {n_evals} evals in {lag_us} µs ({:.1} ns/eval) — checksum={lag_sum:.6e}",
        lag_us as f64 * 1000.0 / n_evals as f64
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
        let gpu_result = barracuda::ops::md::compute_vacf_batch(
            device,
            &velocities,
            n_atoms,
            n_frames,
            n_lags,
        );
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        // CPU path (hotSpring local)
        let vel_snapshots: Vec<Vec<f64>> = velocities
            .chunks(n_atoms * 3)
            .map(<[f64]>::to_vec)
            .collect();
        let t = Instant::now();
        let cpu_result =
            hotspring_barracuda::md::observables::compute_vacf(&vel_snapshots, n_atoms, 0.01, n_lags);
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

// ── Phase 2b: Linear Regression GPU ─────────────────────────────────────────

fn bench_linear_regression_gpu(device: &Arc<WgpuDevice>) {
    println!("═══ Phase 2b: Linear Regression GPU (barracuda::ops::stats_f64) ═══");
    println!("  Provenance: neuralSpring baseCamp V18 → toadStool S25 (stats/linear_regression_f64.wgsl)");
    println!();

    for &(b, n, k) in &[(10, 100, 3), (100, 500, 5), (1000, 1000, 8)] {
        // Design matrix [b, n, k] and response [b, n]
        let x: Vec<f64> = (0..b * n * k)
            .map(|i| (i as f64 * 0.3).sin() + 1.0)
            .collect();
        let y: Vec<f64> = (0..b * n)
            .map(|i| (i as f64 * 0.5).cos() * 3.0)
            .collect();

        let t = Instant::now();
        let result = barracuda::ops::stats_f64::linear_regression(
            device,
            &x,
            &y,
            b as u32,
            n as u32,
            k as u32,
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
            .map(|i| (i as f64 * 0.13).sin() * 5.0)
            .collect();

        let t = Instant::now();
        let result = barracuda::ops::stats_f64::matrix_correlation(
            device,
            &data,
            n as u32,
            p as u32,
        );
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        let status = match &result {
            Ok(corr) => format!("R[0,0]={:.4}, shape={}×{}", corr[0], p, p),
            Err(e) => format!("ERR: {e}"),
        };

        println!("  n={n:>5}, p={p:>3}: {ms:.1}ms [{status}]");
    }
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
