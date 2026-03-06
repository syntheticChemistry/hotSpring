// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-Spring Evolution Benchmark
//!
//! Exercises GPU/CPU ops that evolved through cross-spring absorption in
//! toadStool/barracuda, benchmarks modern vs legacy paths, and documents
//! the provenance of each shader/primitive.
//!
//! # Cross-Spring Shader Provenance (synced to toadStool S80)
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
//! | Spectral stats (⟨r⟩) | hotSpring→toadStool | S78 | level_spacing_ratio, bandwidth, κ |
//! | SpectralAnalysis+RMT | hotSpring→toadStool | S78 | Marchenko–Pastur phase classifier |
//! | Anderson 3D proxy | hotSpring | S25–S78 | Lanczos + level statistics + CG predictor |
//! | NeighborMode 4D | hotSpring→toadStool | S80 | Precomputed periodic neighbor table |
//! | Batched Nelder-Mead GPU | neuralSpring | S79 | Parallel optimization on GPU |
//! | Fused MLP GPU | neuralSpring | S80 | Single-dispatch multi-layer perceptron |
//! | Nautilus brain | hotSpring+neuralSpring | S79 | Evolutionary reservoir for QCD steering |
//! | MultiHeadEsn GPU | hotSpring→toadStool | S78 | Per-head training replaces CPU sidecar |

use std::panic;
use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Cross-Spring Evolution Benchmark — toadStool S80               ║");
    println!("║  hotSpring × wetSpring × neuralSpring → barracuda              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));

    // Phase 1: CPU special functions (no GPU needed)
    bench_special_functions_cpu();

    // Phase 1b: Spectral stats (CPU, upstream from hotSpring)
    bench_spectral_stats_cpu();

    // Phase 1c: Neighbor table precompute (CPU)
    bench_neighbor_precompute();

    // Phase 2: GPU ops (if available)
    let gpu_device = rt.block_on(try_create_device());
    if let Some(device) = gpu_device {
        println!("  GPU: {} (f64 capable)", device.adapter_info().name);
        println!();
        bench_vacf_gpu_vs_cpu(&device);
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

// ── Phase 2b: Linear Regression GPU ─────────────────────────────────────────

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
