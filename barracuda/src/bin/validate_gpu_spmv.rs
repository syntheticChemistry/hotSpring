// SPDX-License-Identifier: AGPL-3.0-only

//! GPU SpMV (CSR Sparse Matrix-Vector Product) Validation
//!
//! Proves that the WGSL CSR SpMV shader produces identical f64 results
//! to the CPU `CsrMatrix::spmv()` across all spectral theory matrix types.
//! This is the P1 GPU primitive that enables:
//!   - GPU Lanczos eigensolve (Kachkovskiy spectral theory)
//!   - GPU Dirac operator (Bazavov lattice QCD)
//!
//! The shader is a direct port: one GPU thread per matrix row, same
//! algorithm as the CPU reference. f64 precision throughout.
//!
//! Exit code 0 = all checks pass, exit code 1 = any check fails.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::spectral::{self, anderson_2d, anderson_3d, CsrMatrix, WGSL_SPMV_CSR_F64};
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpMVParams {
    n: u32,
    nnz: u32,
    pad0: u32,
    pad1: u32,
}

fn identity_csr(n: usize) -> CsrMatrix {
    CsrMatrix {
        n,
        row_ptr: (0..=n).collect(),
        col_idx: (0..n).collect(),
        values: vec![1.0; n],
    }
}

fn gpu_spmv(
    gpu: &GpuF64,
    pipeline: &wgpu::ComputePipeline,
    matrix: &CsrMatrix,
    x: &[f64],
) -> Vec<f64> {
    let n = matrix.n;
    let nnz = matrix.nnz();

    let row_ptr_u32: Vec<u32> = matrix.row_ptr.iter().map(|&v| v as u32).collect();
    let col_idx_u32: Vec<u32> = matrix.col_idx.iter().map(|&v| v as u32).collect();

    let params = SpMVParams {
        n: n as u32,
        nnz: nnz as u32,
        pad0: 0,
        pad1: 0,
    };

    let params_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "spmv_params");
    let row_ptr_buf = gpu.create_u32_buffer(&row_ptr_u32, "row_ptr");
    let col_idx_buf = gpu.create_u32_buffer(&col_idx_u32, "col_idx");
    let values_buf = gpu.create_f64_buffer(&matrix.values, "values");
    let x_buf = gpu.create_f64_buffer(x, "x_vec");
    let y_buf = gpu.create_f64_output_buffer(n, "y_vec");

    let bind_group =
        gpu.create_bind_group(pipeline, &[&params_buf, &row_ptr_buf, &col_idx_buf, &values_buf, &x_buf, &y_buf]);

    let workgroups = (n as u32).div_ceil(64);
    gpu.dispatch(pipeline, &bind_group, workgroups);

    gpu.read_back_f64(&y_buf, n).expect("GPU readback failed")
}

fn cpu_spmv(matrix: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; matrix.n];
    matrix.spmv(x, &mut y);
    y
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GPU SpMV Validation — CSR Sparse Matrix-Vector Product    ║");
    println!("║  WGSL f64 shader vs CPU CsrMatrix::spmv() reference       ║");
    println!("║  P1 primitive: GPU Lanczos + GPU Dirac foundation          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("gpu_spmv");

    // ── GPU device ──────────────────────────────────────────────────
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU not available: {e}");
            println!("  (GPU validation requires SHADER_F64 — skipping)");
            harness.check_bool("GPU available", false);
            harness.finish();
        }
    };

    println!("  GPU: {} (f64={})", gpu.adapter_name, gpu.has_f64);
    println!();

    let pipeline = gpu.create_pipeline_f64(WGSL_SPMV_CSR_F64, "spmv_csr_f64");

    // ══════════════════════════════════════════════════════════════
    //  Check 1: Identity matrix (I*x = x, exact parity)
    // ══════════════════════════════════════════════════════════════
    println!("═══ Check 1: Identity matrix (N=100) ══════════════════════════");
    let n = 100;
    let eye = identity_csr(n);
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * 0.01).collect();
    let gpu_y = gpu_spmv(&gpu, &pipeline, &eye, &x);
    let err = max_abs_diff(&x, &gpu_y);
    println!("  I*x max |error|: {err:.2e} (expect 0)");
    harness.check_upper("Identity SpMV max error", err, 1e-15);

    // ══════════════════════════════════════════════════════════════
    //  Check 2: 2D Anderson (10×10, W=4)
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 2: 2D Anderson (10×10, W=4) ═══════════════════════");
    let a2d = anderson_2d(10, 10, 4.0, 42);
    let x2d: Vec<f64> = (0..a2d.n).map(|i| ((i as f64) * 0.37).sin()).collect();
    let cpu_y2d = cpu_spmv(&a2d, &x2d);
    let gpu_y2d = gpu_spmv(&gpu, &pipeline, &a2d, &x2d);
    let err2d = max_abs_diff(&cpu_y2d, &gpu_y2d);
    println!("  N={}, nnz={}, max |error|: {err2d:.2e}", a2d.n, a2d.nnz());
    harness.check_upper("2D Anderson SpMV (10x10)", err2d, 1e-14);

    // ══════════════════════════════════════════════════════════════
    //  Check 3: 3D Anderson (5×5×5, W=6)
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 3: 3D Anderson (5×5×5, W=6) ══════════════════════");
    let a3d = anderson_3d(5, 5, 5, 6.0, 77);
    let x3d: Vec<f64> = (0..a3d.n).map(|i| ((i as f64) * 0.23).cos()).collect();
    let cpu_y3d = cpu_spmv(&a3d, &x3d);
    let gpu_y3d = gpu_spmv(&gpu, &pipeline, &a3d, &x3d);
    let err3d = max_abs_diff(&cpu_y3d, &gpu_y3d);
    println!("  N={}, nnz={}, max |error|: {err3d:.2e}", a3d.n, a3d.nnz());
    harness.check_upper("3D Anderson SpMV (5x5x5)", err3d, 1e-14);

    // ══════════════════════════════════════════════════════════════
    //  Check 4: Clean 2D lattice (no disorder — deterministic)
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 4: Clean 2D lattice (16×16, W=0) ═════════════════");
    let clean = spectral::clean_2d_lattice(16, 16);
    let x_clean: Vec<f64> = (0..clean.n).map(|i| 1.0 / (i as f64 + 1.0)).collect();
    let cpu_y_clean = cpu_spmv(&clean, &x_clean);
    let gpu_y_clean = gpu_spmv(&gpu, &pipeline, &clean, &x_clean);
    let err_clean = max_abs_diff(&cpu_y_clean, &gpu_y_clean);
    println!(
        "  N={}, nnz={}, max |error|: {err_clean:.2e}",
        clean.n,
        clean.nnz()
    );
    harness.check_upper("Clean lattice SpMV (16x16)", err_clean, 1e-14);

    // ══════════════════════════════════════════════════════════════
    //  Check 5: Iterated SpMV — A²x via two GPU dispatches
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 5: Iterated SpMV A²x (8×8) ══════════════════════");
    let a_iter = anderson_2d(8, 8, 3.0, 99);
    let x_iter: Vec<f64> = (0..a_iter.n).map(|i| ((i as f64) * 0.11).sin()).collect();
    let cpu_ax = cpu_spmv(&a_iter, &x_iter);
    let cpu_aax = cpu_spmv(&a_iter, &cpu_ax);
    let gpu_ax = gpu_spmv(&gpu, &pipeline, &a_iter, &x_iter);
    let gpu_aax = gpu_spmv(&gpu, &pipeline, &a_iter, &gpu_ax);
    let err_iter = max_abs_diff(&cpu_aax, &gpu_aax);
    println!(
        "  N={}, max |error| after 2 iterations: {err_iter:.2e}",
        a_iter.n
    );
    harness.check_upper("Iterated SpMV A^2 x", err_iter, 1e-13);

    // ══════════════════════════════════════════════════════════════
    //  Check 6: Large 2D Anderson (32×32 = 1024, W=4)
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 6: Large 2D Anderson (32×32 = 1024) ══════════════");
    let a_large = anderson_2d(32, 32, 4.0, 123);
    let x_large: Vec<f64> = (0..a_large.n)
        .map(|i| ((i as f64) * 0.07).sin())
        .collect();
    let cpu_y_large = cpu_spmv(&a_large, &x_large);
    let gpu_y_large = gpu_spmv(&gpu, &pipeline, &a_large, &x_large);
    let err_large = max_abs_diff(&cpu_y_large, &gpu_y_large);
    println!(
        "  N={}, nnz={}, max |error|: {err_large:.2e}",
        a_large.n,
        a_large.nnz()
    );
    harness.check_upper("Large 2D Anderson SpMV (32x32)", err_large, 1e-14);

    // ══════════════════════════════════════════════════════════════
    //  Check 7: Strong disorder 3D (W=20, near all-localized)
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 7: Strong disorder 3D (6×6×6, W=20) ═════════════");
    let a_strong = anderson_3d(6, 6, 6, 20.0, 55);
    let x_strong: Vec<f64> = (0..a_strong.n)
        .map(|i| ((i as f64) * 0.13).cos())
        .collect();
    let cpu_y_strong = cpu_spmv(&a_strong, &x_strong);
    let gpu_y_strong = gpu_spmv(&gpu, &pipeline, &a_strong, &x_strong);
    let err_strong = max_abs_diff(&cpu_y_strong, &gpu_y_strong);
    println!(
        "  N={}, nnz={}, max |error|: {err_strong:.2e}",
        a_strong.n,
        a_strong.nnz()
    );
    harness.check_upper("Strong disorder 3D SpMV (W=20)", err_strong, 1e-14);

    // ══════════════════════════════════════════════════════════════
    //  Check 8: Very large 2D Anderson (64×64 = 4096) + timing
    // ══════════════════════════════════════════════════════════════
    println!("\n═══ Check 8: Very large SpMV (64×64 = 4096) + timing ═══════");
    let a_bench = anderson_2d(64, 64, 4.0, 999);
    let x_bench: Vec<f64> = (0..a_bench.n)
        .map(|i| ((i as f64) * 0.03).sin())
        .collect();

    // Correctness
    let cpu_y_bench = cpu_spmv(&a_bench, &x_bench);
    let gpu_y_bench = gpu_spmv(&gpu, &pipeline, &a_bench, &x_bench);
    let err_bench = max_abs_diff(&cpu_y_bench, &gpu_y_bench);
    println!(
        "  N={}, nnz={}, max |error|: {err_bench:.2e}",
        a_bench.n,
        a_bench.nnz()
    );
    harness.check_upper("Very large SpMV correctness (64x64)", err_bench, 1e-14);

    // Timing (informational — includes buffer allocation overhead)
    let n_reps = 100;
    let t_cpu = Instant::now();
    let mut y_tmp = vec![0.0; a_bench.n];
    for _ in 0..n_reps {
        a_bench.spmv(&x_bench, &mut y_tmp);
    }
    let cpu_us = t_cpu.elapsed().as_micros();

    let t_gpu = Instant::now();
    for _ in 0..n_reps {
        let _ = gpu_spmv(&gpu, &pipeline, &a_bench, &x_bench);
    }
    let gpu_us = t_gpu.elapsed().as_micros();

    println!(
        "  Timing ({n_reps} reps, includes buffer alloc): CPU={cpu_us}µs, GPU={gpu_us}µs"
    );
    println!(
        "  NOTE: GPU advantage requires N>100k and persistent buffers (streaming Lanczos)."
    );
    println!("        At N=4096, per-dispatch overhead dominates. This test is about CORRECTNESS.");

    // ══════════════════════════════════════════════════════════════
    //  Verdict
    // ══════════════════════════════════════════════════════════════
    println!();
    println!("═══ Summary ══════════════════════════════════════════════════");
    println!("  GPU SpMV shader: identical f64 results across all matrix types");
    println!("  CPU CsrMatrix::spmv() → GPU WGSL_SPMV_CSR_F64: math is portable");
    println!("  Next: GPU Lanczos (streaming SpMV), then GPU Dirac (lattice QCD)");
    println!();

    harness.finish();
}
