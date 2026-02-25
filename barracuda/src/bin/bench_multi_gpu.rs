// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-GPU cooperative benchmark: distribute work across two GPUs.
//!
//! Demonstrates three modes:
//!   1. **Comparative**: Same workload on each card (side-by-side)
//!   2. **Cooperative**: Split batches across both cards simultaneously
//!   3. **Specialized**: Route workloads to the best card for each task type
//!
//! GPU selection via environment variables (node-agnostic):
//!   - `HOTSPRING_GPU_PRIMARY`   — Card A name substring (default: `"4070"`)
//!   - `HOTSPRING_GPU_SECONDARY` — Card B name substring (default: `"titan"`)
//!
//! Example (biomeGate):
//!   HOTSPRING_GPU_PRIMARY=3090 HOTSPRING_GPU_SECONDARY=titan \
//!     cargo run --release --bin bench_multi_gpu
//!
//! Or source a node profile:
//!   HOTSPRING_GPU_PRIMARY=0 HOTSPRING_GPU_SECONDARY=1 cargo run --release --bin bench_multi_gpu

use barracuda::ops::linalg::BatchedEighGpu;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::physics::bcs_gpu::BcsBisectionGpu;
use std::sync::Arc;
use std::time::Instant;

const WARMUP: usize = 3;
const MEASURE: usize = 10;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    println!("═══════════════════════════════════════════════════════════");
    println!("  Multi-GPU FP64 Cooperative Benchmark");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let primary_name = std::env::var("HOTSPRING_GPU_PRIMARY").unwrap_or_else(|_| "4070".into());
    let secondary_name =
        std::env::var("HOTSPRING_GPU_SECONDARY").unwrap_or_else(|_| "titan".into());

    println!("  Config: primary={primary_name:?}, secondary={secondary_name:?}");
    println!();

    // Initialize both GPUs
    #[allow(deprecated)] // set_var deprecated in edition 2024; safe here (single-threaded init)
    let gpu_primary = rt.block_on(async {
        std::env::set_var("HOTSPRING_GPU_ADAPTER", &primary_name);
        GpuF64::new().await
    });
    #[allow(deprecated)]
    std::env::set_var("HOTSPRING_GPU_ADAPTER", &secondary_name);
    let gpu_secondary = rt.block_on(GpuF64::new());

    let gpu_primary = match gpu_primary {
        Ok(g) => {
            println!("  Card A: {} (SHADER_F64={})", g.adapter_name, g.has_f64);
            g
        }
        Err(e) => {
            eprintln!("  Card A ({primary_name}): UNAVAILABLE ({e})");
            return;
        }
    };
    let gpu_secondary = match gpu_secondary {
        Ok(g) => {
            println!("  Card B: {} (SHADER_F64={})", g.adapter_name, g.has_f64);
            g
        }
        Err(e) => {
            eprintln!("  Card B ({secondary_name}): UNAVAILABLE ({e})");
            return;
        }
    };
    println!();

    // Phase 1: Comparative BCS bisection
    comparative_bcs(&gpu_primary, &gpu_secondary);
    println!();

    // Phase 2: Cooperative BCS — split batch across both cards
    cooperative_bcs(&gpu_primary, &gpu_secondary);
    println!();

    // Phase 3: Cooperative eigensolve — split batch across both cards
    cooperative_eigensolve(&gpu_primary, &gpu_secondary);
    println!();

    // Phase 4: Specialized routing — each card does what it's best at
    specialized_routing(&gpu_primary, &gpu_secondary);
}

/// Phase 1: Same BCS workload on each card for comparison.
fn comparative_bcs(gpu_a: &GpuF64, gpu_b: &GpuF64) {
    println!("── Phase 1: Comparative BCS (same workload, each card) ──");
    let batch = 4096;
    let n_levels = 20;

    let eigenvalues = generate_eigenvalues(batch, n_levels);
    let delta: Vec<f64> = vec![2.0; batch];
    let target_n: Vec<f64> = (0..batch).map(|i| (i % 10 + 2) as f64).collect();
    let lower = vec![-50.0_f64; batch];
    let upper = vec![200.0_f64; batch];

    let time_a = bench_bcs_single(
        gpu_a,
        &eigenvalues,
        &delta,
        &target_n,
        &lower,
        &upper,
        batch,
        n_levels,
    );
    let time_b = bench_bcs_single(
        gpu_b,
        &eigenvalues,
        &delta,
        &target_n,
        &lower,
        &upper,
        batch,
        n_levels,
    );

    println!(
        "  {:<30} {:>10.3} ms  ({:.0}/s)",
        gpu_a.adapter_name,
        time_a * 1e3,
        batch as f64 / time_a
    );
    println!(
        "  {:<30} {:>10.3} ms  ({:.0}/s)",
        gpu_b.adapter_name,
        time_b * 1e3,
        batch as f64 / time_b
    );
    let ratio = time_a / time_b;
    if ratio > 1.0 {
        println!("  → Card B is {ratio:.1}× faster");
    } else {
        println!("  → Card A is {:.1}× faster", 1.0 / ratio);
    }
}

/// Phase 2: Split BCS batch across both cards simultaneously.
fn cooperative_bcs(gpu_a: &GpuF64, gpu_b: &GpuF64) {
    println!("── Phase 2: Cooperative BCS (split across both cards) ────");
    let total_batch = 8192;
    let n_levels = 20;

    let eigenvalues = generate_eigenvalues(total_batch, n_levels);
    let delta: Vec<f64> = vec![2.0; total_batch];
    let target_n: Vec<f64> = (0..total_batch).map(|i| (i % 10 + 2) as f64).collect();
    let lower = vec![-50.0_f64; total_batch];
    let upper = vec![200.0_f64; total_batch];

    // Single card: all on card A
    let t_single = bench_bcs_single(
        gpu_a,
        &eigenvalues,
        &delta,
        &target_n,
        &lower,
        &upper,
        total_batch,
        n_levels,
    );

    // Cooperative: split 50/50 across both cards using threads
    let bcs_a = BcsBisectionGpu::new(gpu_a, 100, 1e-12);
    let bcs_b = BcsBisectionGpu::new(gpu_b, 100, 1e-12);

    let half = total_batch / 2;
    let split_evals_a = eigenvalues[..half * n_levels].to_vec();
    let split_evals_b = eigenvalues[half * n_levels..].to_vec();
    let split_delta_a = delta[..half].to_vec();
    let split_delta_b = delta[half..].to_vec();
    let split_tn_a = target_n[..half].to_vec();
    let split_tn_b = target_n[half..].to_vec();
    let split_lo_a = lower[..half].to_vec();
    let split_lo_b = lower[half..].to_vec();
    let split_hi_a = upper[..half].to_vec();
    let split_hi_b = upper[half..].to_vec();

    // Warmup
    for _ in 0..WARMUP {
        let _ = bcs_a.solve_bcs(
            &split_lo_a,
            &split_hi_a,
            &split_evals_a,
            &split_delta_a,
            &split_tn_a,
        );
        let _ = bcs_b.solve_bcs(
            &split_lo_b,
            &split_hi_b,
            &split_evals_b,
            &split_delta_b,
            &split_tn_b,
        );
    }

    // Measure cooperative: both cards work simultaneously
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        std::thread::scope(|s| {
            s.spawn(|| {
                let _ = bcs_a.solve_bcs(
                    &split_lo_a,
                    &split_hi_a,
                    &split_evals_a,
                    &split_delta_a,
                    &split_tn_a,
                );
            });
            s.spawn(|| {
                let _ = bcs_b.solve_bcs(
                    &split_lo_b,
                    &split_hi_b,
                    &split_evals_b,
                    &split_delta_b,
                    &split_tn_b,
                );
            });
        });
    }
    let t_coop = t0.elapsed().as_secs_f64() / MEASURE as f64;

    println!("  Total batch:     {total_batch}");
    println!(
        "  Single card (A): {:.3} ms  ({:.0}/s)",
        t_single * 1e3,
        total_batch as f64 / t_single
    );
    println!(
        "  Cooperative:     {:.3} ms  ({:.0}/s)",
        t_coop * 1e3,
        total_batch as f64 / t_coop
    );
    let speedup = t_single / t_coop;
    println!("  → Cooperative speedup: {speedup:.2}×");
}

/// Phase 3: Split eigensolve batch across both cards.
fn cooperative_eigensolve(gpu_a: &GpuF64, gpu_b: &GpuF64) {
    println!("── Phase 3: Cooperative Eigensolve (split across cards) ──");
    let total_batch = 256;
    let dim = 20;

    let all_matrices = generate_symmetric_matrices(total_batch, dim);
    let half = total_batch / 2;
    let mats_a = all_matrices[..half * dim * dim].to_vec();
    let mats_b = all_matrices[half * dim * dim..].to_vec();

    let dev_a: Arc<barracuda::device::WgpuDevice> = gpu_a.to_wgpu_device();
    let dev_b: Arc<barracuda::device::WgpuDevice> = gpu_b.to_wgpu_device();

    // Warmup
    for _ in 0..WARMUP {
        let _ = BatchedEighGpu::execute_single_dispatch(
            dev_a.clone(),
            &all_matrices,
            dim,
            total_batch,
            200,
            1e-12,
        );
    }

    // Single card
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let _ = BatchedEighGpu::execute_single_dispatch(
            dev_a.clone(),
            &all_matrices,
            dim,
            total_batch,
            200,
            1e-12,
        );
    }
    let t_single = t0.elapsed().as_secs_f64() / MEASURE as f64;

    // Warmup cooperative
    for _ in 0..WARMUP {
        let _ =
            BatchedEighGpu::execute_single_dispatch(dev_a.clone(), &mats_a, dim, half, 200, 1e-12);
        let _ =
            BatchedEighGpu::execute_single_dispatch(dev_b.clone(), &mats_b, dim, half, 200, 1e-12);
    }

    // Cooperative
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let dev_a2 = dev_a.clone();
        let dev_b2 = dev_b.clone();
        let mats_a2 = mats_a.clone();
        let mats_b2 = mats_b.clone();
        std::thread::scope(|s| {
            s.spawn(move || {
                let _ = BatchedEighGpu::execute_single_dispatch(
                    dev_a2, &mats_a2, dim, half, 200, 1e-12,
                );
            });
            s.spawn(move || {
                let _ = BatchedEighGpu::execute_single_dispatch(
                    dev_b2, &mats_b2, dim, half, 200, 1e-12,
                );
            });
        });
    }
    let t_coop = t0.elapsed().as_secs_f64() / MEASURE as f64;

    println!("  Total batch:     {total_batch} × {dim}×{dim} matrices");
    println!(
        "  Single card (A): {:.3} ms  ({:.0}/s)",
        t_single * 1e3,
        total_batch as f64 / t_single
    );
    println!(
        "  Cooperative:     {:.3} ms  ({:.0}/s)",
        t_coop * 1e3,
        total_batch as f64 / t_coop
    );
    let speedup = t_single / t_coop;
    println!("  → Cooperative speedup: {speedup:.2}×");
}

/// Phase 4: Route each workload to the card that's best at it.
fn specialized_routing(gpu_a: &GpuF64, gpu_b: &GpuF64) {
    println!("── Phase 4: Specialized Routing (best card per task) ─────");
    println!("  Strategy: BCS→{} (fp64 throughput),", gpu_b.adapter_name);
    println!(
        "            Eigensolve→{} (low dispatch latency)",
        gpu_a.adapter_name
    );

    let bcs_batch = 4096;
    let eig_batch = 128;
    let n_levels = 20;
    let dim = 20;

    let eigenvalues = generate_eigenvalues(bcs_batch, n_levels);
    let delta: Vec<f64> = vec![2.0; bcs_batch];
    let target_n: Vec<f64> = (0..bcs_batch).map(|i| (i % 10 + 2) as f64).collect();
    let lower = vec![-50.0_f64; bcs_batch];
    let upper = vec![200.0_f64; bcs_batch];
    let matrices = generate_symmetric_matrices(eig_batch, dim);

    let bcs_secondary = BcsBisectionGpu::new(gpu_b, 100, 1e-12);
    let dev_primary: Arc<barracuda::device::WgpuDevice> = gpu_a.to_wgpu_device();

    // Warmup
    for _ in 0..WARMUP {
        let _ = bcs_secondary.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
        let _ = BatchedEighGpu::execute_single_dispatch(
            dev_primary.clone(),
            &matrices,
            dim,
            eig_batch,
            200,
            1e-12,
        );
    }

    // Sequential: both tasks on card A
    let t0 = Instant::now();
    let bcs_a = BcsBisectionGpu::new(gpu_a, 100, 1e-12);
    for _ in 0..MEASURE {
        let _ = bcs_a.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
        let _ = BatchedEighGpu::execute_single_dispatch(
            dev_primary.clone(),
            &matrices,
            dim,
            eig_batch,
            200,
            1e-12,
        );
    }
    let t_seq = t0.elapsed().as_secs_f64() / MEASURE as f64;

    // Specialized: BCS on secondary, eigensolve on primary, simultaneously
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let dev = dev_primary.clone();
        let mats = matrices.clone();
        std::thread::scope(|s| {
            s.spawn(|| {
                let _ = bcs_secondary.solve_bcs(&lower, &upper, &eigenvalues, &delta, &target_n);
            });
            s.spawn(move || {
                let _ =
                    BatchedEighGpu::execute_single_dispatch(dev, &mats, dim, eig_batch, 200, 1e-12);
            });
        });
    }
    let t_spec = t0.elapsed().as_secs_f64() / MEASURE as f64;

    println!("  Sequential (A only): {:.3} ms", t_seq * 1e3);
    println!("  Specialized (both):  {:.3} ms", t_spec * 1e3);
    let speedup = t_seq / t_spec;
    println!("  → Specialized speedup: {speedup:.2}×");
}

fn bench_bcs_single(
    gpu: &GpuF64,
    eigenvalues: &[f64],
    delta: &[f64],
    target_n: &[f64],
    lower: &[f64],
    upper: &[f64],
    batch: usize,
    _n_levels: usize,
) -> f64 {
    let bcs = BcsBisectionGpu::new(gpu, 100, 1e-12);
    for _ in 0..WARMUP {
        let _ = bcs.solve_bcs(lower, upper, eigenvalues, delta, target_n);
    }
    let t0 = Instant::now();
    for _ in 0..MEASURE {
        let _ = bcs.solve_bcs(lower, upper, eigenvalues, delta, target_n);
    }
    let _ = batch; // used only for reporting
    t0.elapsed().as_secs_f64() / MEASURE as f64
}

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
