// SPDX-License-Identifier: AGPL-3.0-only

//! GPU FP64 benchmark: side-by-side comparison across GPUs and drivers.
//!
//! Exercises three GPU-intensive workloads at varying batch sizes:
//! 1. BCS bisection (root-finding on GPU)
//! 2. Batched eigensolve (Jacobi rotations on GPU)
//! 3. L2 HFB SCF pipeline (full physics on real nuclei)
//!
//! Run on each GPU:
//!   `HOTSPRING_GPU_ADAPTER=4070`  cargo run --release --bin `bench_gpu_fp64`
//!   `HOTSPRING_GPU_ADAPTER=titan` cargo run --release --bin `bench_gpu_fp64`
//!
//! Provenance: analytical benchmarks with synthetic Hamiltonians (1, 2)
//! and real AME2020 nuclei with `SLy4` parameters (3).
//! No Python baseline — this measures GPU throughput, not physics fidelity.

use barracuda::ops::linalg::BatchedEighGpu;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::physics::bcs_gpu::BcsBisectionGpu;
use hotspring_barracuda::physics::hfb_gpu_resident::binding_energies_l2_gpu_resident;
use hotspring_barracuda::provenance;
use std::sync::Arc;
use std::time::Instant;

const WARMUP_ROUNDS: usize = 3;
const MEASURE_ROUNDS: usize = 10;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = rt.block_on(GpuF64::new()).expect("GPU init failed");

    println!("═══════════════════════════════════════════════════════════");
    println!("  GPU FP64 Benchmark");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    gpu.print_info();
    GpuF64::print_available_adapters();
    println!();

    bench_bcs(&gpu);
    println!();
    bench_eigensolve(&gpu);

    // L2 pipeline is CPU-bound (eigensolve readback dominates).
    // Only run with --l2 flag for full end-to-end validation.
    if std::env::args().any(|a| a == "--l2") {
        println!();
        bench_l2_pipeline(&gpu);
    } else {
        println!();
        println!("── Skipping L2 pipeline (pass --l2 to include) ──────────");
        println!("  (L2 is currently CPU-bound: eigensolve readback dominates)");
    }
}

/// Benchmark 1: BCS bisection on GPU at varying batch sizes.
fn bench_bcs(gpu: &GpuF64) {
    println!("── Benchmark 1: BCS Bisection ──────────────────────────");
    println!(
        "  {:>8} {:>10} {:>12} {:>12}",
        "batch", "wall (ms)", "per-nucleus", "throughput"
    );

    let bcs = BcsBisectionGpu::new(gpu, 100, 1e-12);
    let n_levels = 20;

    for &batch in &[8, 32, 128, 512, 2048, 8192] {
        let eigenvalues = generate_eigenvalues(batch, n_levels);
        let delta: Vec<f64> = vec![2.0; batch];
        let target_n: Vec<f64> = (0..batch).map(|i| (i % 10 + 2) as f64).collect();
        let lower = vec![-50.0_f64; batch];
        let upper = vec![200.0_f64; batch];

        // Warmup
        for _ in 0..WARMUP_ROUNDS {
            let _ = bcs.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
        }

        // Measure
        let t0 = Instant::now();
        for _ in 0..MEASURE_ROUNDS {
            let _ = bcs.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
        }
        let elapsed = t0.elapsed().as_secs_f64() / MEASURE_ROUNDS as f64;
        let per_nucleus_us = elapsed * 1e6 / batch as f64;
        let throughput = batch as f64 / elapsed;

        println!(
            "  {:>8} {:>10.3} {:>10.1} μs {:>10.0}/s",
            batch,
            elapsed * 1e3,
            per_nucleus_us,
            throughput
        );
    }
}

/// Benchmark 2: Batched eigensolve on GPU at varying batch/dim sizes.
fn bench_eigensolve(gpu: &GpuF64) {
    println!("── Benchmark 2: Batched Eigensolve ─────────────────────");
    println!(
        "  {:>8} {:>6} {:>10} {:>12} {:>12}",
        "batch", "dim", "wall (ms)", "per-matrix", "throughput"
    );

    let device: Arc<barracuda::device::WgpuDevice> = gpu.to_wgpu_device();

    for &(batch, dim) in &[
        (8, 20),
        (32, 20),
        (128, 20),
        (512, 20),
        (8, 30),
        (32, 30),
        (128, 30),
        (512, 30),
    ] {
        let matrices = generate_symmetric_matrices(batch, dim);

        // Warmup
        for _ in 0..WARMUP_ROUNDS {
            let _ = BatchedEighGpu::execute_single_dispatch(
                device.clone(),
                &matrices,
                dim,
                batch,
                200,
                1e-12,
            );
        }

        // Measure
        let t0 = Instant::now();
        for _ in 0..MEASURE_ROUNDS {
            let _ = BatchedEighGpu::execute_single_dispatch(
                device.clone(),
                &matrices,
                dim,
                batch,
                200,
                1e-12,
            );
        }
        let elapsed = t0.elapsed().as_secs_f64() / MEASURE_ROUNDS as f64;
        let per_matrix_us = elapsed * 1e6 / batch as f64;
        let throughput = batch as f64 / elapsed;

        println!(
            "  {:>8} {:>6} {:>10.3} {:>10.1} μs {:>10.0}/s",
            batch,
            dim,
            elapsed * 1e3,
            per_matrix_us,
            throughput
        );
    }
}

/// Benchmark 3: Full L2 HFB pipeline on real nuclei (GPU-resident).
fn bench_l2_pipeline(gpu: &GpuF64) {
    println!("── Benchmark 3: L2 HFB Pipeline — GPU Resident ─────────");

    let device = gpu.to_wgpu_device();

    // Standard L2 nuclei: 18 focused set from the optimization pipeline
    let nuclei: Vec<(usize, usize)> = vec![
        (8, 8),
        (14, 14),
        (20, 20),
        (20, 28),
        (28, 28),
        (28, 30),
        (28, 32),
        (28, 34),
        (34, 42),
        (38, 50),
        (40, 50),
        (40, 52),
        (40, 54),
        (42, 50),
        (42, 52),
        (48, 62),
        (48, 64),
        (50, 82),
    ];

    let params = provenance::SLY4_PARAMS;
    let n_nuclei = nuclei.len();

    // Warmup — GPU-resident: potentials, H-build, density, mixing all on GPU
    let max_iter = 30; // Keep low for benchmarking (focus on GPU dispatch throughput)
    println!("  Warming up (1 eval, {n_nuclei} nuclei, {max_iter} iters, GPU-resident)...");
    let _ = binding_energies_l2_gpu_resident(&device, &nuclei, &params, max_iter, 0.05, 0.3)
        .expect("GPU-resident HFB warmup failed");

    // Measure
    let rounds = 2;
    println!("  Measuring ({rounds} evals)...");
    let t0 = Instant::now();
    for r in 0..rounds {
        let result =
            binding_energies_l2_gpu_resident(&device, &nuclei, &params, max_iter, 0.05, 0.3)
                .expect("GPU-resident HFB benchmark failed");
        if r == rounds - 1 {
            let converged = result.results.iter().filter(|r| r.3).count();
            println!("  Converged: {converged}/{n_nuclei} (in {max_iter} iters)");
        }
    }
    let elapsed = t0.elapsed().as_secs_f64() / f64::from(rounds);

    println!("  Wall per eval:   {:.1} ms", elapsed * 1e3);
    println!(
        "  Per nucleus:     {:.1} μs",
        elapsed * 1e6 / n_nuclei as f64
    );
    println!(
        "  Throughput:      {:.0} nuclei/s",
        n_nuclei as f64 / elapsed
    );
    println!("  Nuclei per eval: {n_nuclei}");
    println!("  Pipeline: GPU-resident (potentials, H, density, mixing on GPU)");
}

/// Generate synthetic eigenvalues: harmonic oscillator levels with jitter.
fn generate_eigenvalues(batch: usize, n_levels: usize) -> Vec<f64> {
    let mut vals = Vec::with_capacity(batch * n_levels);
    for b in 0..batch {
        for j in 0..n_levels {
            let base = (j as f64 + 0.5) * 5.0;
            let jitter = (b as f64 * 0.1).sin() * 0.5;
            vals.push(base + jitter);
        }
    }
    vals
}

/// Generate symmetric positive-definite matrices for eigensolve benchmarking.
fn generate_symmetric_matrices(batch: usize, dim: usize) -> Vec<f64> {
    let mut matrices = Vec::with_capacity(batch * dim * dim);
    for b in 0..batch {
        for i in 0..dim {
            for j in 0..dim {
                let diag = if i == j {
                    (i as f64 + 1.0).mul_add(10.0, b as f64 * 0.1)
                } else {
                    0.0
                };
                let off = if i == j {
                    0.0
                } else {
                    ((i + j) as f64).mul_add(0.3, b as f64 * 0.01).sin() * 0.5
                };
                matrices.push(diag + off);
            }
        }
    }
    // Symmetrize
    for b in 0..batch {
        let base = b * dim * dim;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = f64::midpoint(matrices[base + i * dim + j], matrices[base + j * dim + i]);
                matrices[base + i * dim + j] = val;
                matrices[base + j * dim + i] = val;
            }
        }
    }
    matrices
}
