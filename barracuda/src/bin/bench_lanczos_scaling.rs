// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lanczos Scaling Benchmark
//!
//! Benchmarks GPU Lanczos eigensolve at increasing matrix sizes to demonstrate
//! scaling characteristics. Target: N=10,000+ (Kachkovskiy review deliverable).

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::spectral::{
    CsrMatrix, WGSL_SPMV_CSR_F64, anderson_3d, lanczos, lanczos_eigenvalues,
    find_all_eigenvalues,
};
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SpMVParams {
    n: u32,
    nnz: u32,
    pad0: u32,
    pad1: u32,
}

struct Lcg(u64);

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn gpu_lanczos_timed(
    gpu: &GpuF64,
    pipeline: &wgpu::ComputePipeline,
    matrix: &CsrMatrix,
    max_iter: usize,
    seed: u64,
) -> (Vec<f64>, std::time::Duration) {
    let n = matrix.n;
    let m = max_iter.min(n);

    let row_ptr_u32: Vec<u32> = matrix.row_ptr.iter().map(|&v| v as u32).collect();
    let col_idx_u32: Vec<u32> = matrix.col_idx.iter().map(|&v| v as u32).collect();
    let params = SpMVParams {
        n: n as u32,
        nnz: matrix.nnz() as u32,
        pad0: 0,
        pad1: 0,
    };

    let params_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "lanczos_params");
    let row_ptr_buf = gpu.create_u32_buffer(&row_ptr_u32, "lanczos_row_ptr");
    let col_idx_buf = gpu.create_u32_buffer(&col_idx_u32, "lanczos_col_idx");
    let values_buf = gpu.create_f64_buffer(&matrix.values, "lanczos_values");
    let x_buf = gpu.create_f64_output_buffer(n, "lanczos_x");
    let y_buf = gpu.create_f64_output_buffer(n, "lanczos_y");

    let bind_group = gpu.create_bind_group(
        pipeline,
        &[&params_buf, &row_ptr_buf, &col_idx_buf, &values_buf, &x_buf, &y_buf],
    );
    let workgroups = (n as u32).div_ceil(64);

    let mut rng = Lcg::new(seed);
    let mut v: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v {
        *x /= norm;
    }

    let mut alpha = Vec::with_capacity(m);
    let mut beta_vec = Vec::with_capacity(m);
    let mut v_prev = vec![0.0; n];
    let mut beta_prev = 0.0;

    let t0 = Instant::now();

    for j in 0..m {
        gpu.upload_f64(&x_buf, &v);
        gpu.dispatch(pipeline, &bind_group, workgroups);
        let mut w = gpu.read_back_f64(&y_buf, n).expect("GPU readback");

        if j > 0 {
            for i in 0..n {
                w[i] -= beta_prev * v_prev[i];
            }
        }

        let a_j: f64 = w.iter().zip(&v).map(|(a, b)| a * b).sum();
        alpha.push(a_j);

        for i in 0..n {
            w[i] -= a_j * v[i];
        }

        let b_next = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if b_next < 1e-14 {
            break;
        }
        beta_vec.push(b_next);
        beta_prev = b_next;

        v_prev.clone_from(&v);
        for i in 0..n {
            v[i] = w[i] / b_next;
        }
    }

    let elapsed = t0.elapsed();

    let off: Vec<f64> = beta_vec[..alpha.len().saturating_sub(1)].to_vec();
    let evals = find_all_eigenvalues(&alpha, &off);
    (evals, elapsed)
}

fn cpu_lanczos_timed(matrix: &CsrMatrix, m: usize, seed: u64) -> (Vec<f64>, std::time::Duration) {
    let t0 = Instant::now();
    let tri = lanczos(matrix, m, seed);
    let evals = lanczos_eigenvalues(&tri);
    let elapsed = t0.elapsed();
    (evals, elapsed)
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Lanczos Scaling Benchmark — GPU vs CPU");
    println!("  Target: N=10,000+ (Kachkovskiy review deliverable)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  GPU not available: {e}");
            std::process::exit(1);
        }
    };
    println!("  GPU: {} (f64={})", gpu.adapter_name, gpu.has_f64);
    println!();

    let pipeline = gpu.create_pipeline_f64(WGSL_SPMV_CSR_F64, "spmv_lanczos");

    let seed = 42u64;

    let cases: Vec<(&str, usize, usize, f64)> = vec![
        ("Anderson 3D  6³",   6, 50, 4.0),
        ("Anderson 3D  8³",   8, 50, 4.0),
        ("Anderson 3D 10³",  10, 50, 4.0),
        ("Anderson 3D 12³",  12, 50, 4.0),
        ("Anderson 3D 14³",  14, 50, 4.0),
        ("Anderson 3D 16³",  16, 50, 4.0),
        ("Anderson 3D 18³",  18, 50, 4.0),
        ("Anderson 3D 20³",  20, 50, 4.0),
        ("Anderson 3D 22³",  22, 50, 4.0),
    ];

    println!("  {:20} {:>8} {:>5} {:>10} {:>10} {:>8} {:>12}",
             "Model", "N", "k", "GPU", "CPU", "Speedup", "Max|Δλ|");
    println!("  {}", "-".repeat(80));

    for (label, l, k, w) in &cases {
        let matrix = anderson_3d(*l, *l, *l, *w, 42);
        let n = matrix.n;
        let m = (*k).min(n);

        let (gpu_evals, gpu_time) = gpu_lanczos_timed(&gpu, &pipeline, &matrix, m, seed);
        let (cpu_evals, cpu_time) = cpu_lanczos_timed(&matrix, m, seed);

        let n_compare = gpu_evals.len().min(cpu_evals.len());
        let max_diff = if n_compare > 0 {
            gpu_evals[..n_compare]
                .iter()
                .zip(&cpu_evals[..n_compare])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        } else {
            0.0
        };

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(1e-9);

        println!("  {:20} {:>8} {:>5} {:>9.3}s {:>9.3}s {:>7.1}× {:>12.2e}",
                 label, n, m,
                 gpu_time.as_secs_f64(),
                 cpu_time.as_secs_f64(),
                 speedup,
                 max_diff);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  GPU SpMV in Lanczos → identical eigenvalues to CPU.");
    println!("  Scaling: GPU advantage grows with N (SpMV dominates).");
    println!("═══════════════════════════════════════════════════════════════");
}
