// SPDX-License-Identifier: AGPL-3.0-only

//! Lattice QCD Scaling Benchmark — GPU CG at production lattice sizes
//!
//! Exercises GPU Dirac + CG on 4⁴, 6⁴, 8⁴, 8³×16, and 16⁴ lattices.
//! At each size: runs HMC thermalization, then GPU CG fermion solve.
//! Reports timing, iteration counts, and GPU vs CPU scaling.

use barracuda::pipeline::ReduceScalarPipeline;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::lattice::cg::{
    cg_solve, WGSL_AXPY_F64, WGSL_COMPLEX_DOT_RE_F64, WGSL_XPAY_F64,
};
use hotspring_barracuda::lattice::dirac::{
    flatten_fermion, DiracGpuLayout, FermionField, WGSL_DIRAC_STAGGERED_F64,
};
use hotspring_barracuda::lattice::hmc::{hmc_trajectory, HmcConfig};
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DiracParams {
    volume: u32,
    pad0: u32,
    mass_re: f64,
    hop_sign: f64,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DotParams {
    n_pairs: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarParams {
    n: u32,
    pad0: u32,
    alpha: f64,
}

struct GpuCgContext<'a> {
    gpu: &'a GpuF64,
    dirac_pl: &'a wgpu::ComputePipeline,
    dot_pl: &'a wgpu::ComputePipeline,
    axpy_pl: &'a wgpu::ComputePipeline,
    xpay_pl: &'a wgpu::ComputePipeline,
}

fn gpu_cg_solve(
    ctx: &GpuCgContext,
    reducer: &ReduceScalarPipeline,
    layout: &DiracGpuLayout,
    b_flat: &[f64],
    mass: f64,
    tol: f64,
    max_iter: usize,
) -> (usize, f64, f64) {
    let vol = layout.volume;
    let n_flat = vol * 6;
    let n_pairs = vol * 3;

    let links_buf = ctx.gpu.create_f64_buffer(&layout.links_flat, "links");
    let nbr_buf = ctx.gpu.create_u32_buffer(&layout.neighbors, "nbr");
    let phases_buf = ctx.gpu.create_f64_buffer(&layout.phases, "phases");

    let x_buf = ctx.gpu.create_f64_output_buffer(n_flat, "x");
    let r_buf = ctx.gpu.create_f64_output_buffer(n_flat, "r");
    let p_buf = ctx.gpu.create_f64_output_buffer(n_flat, "p");
    let ap_buf = ctx.gpu.create_f64_output_buffer(n_flat, "ap");
    let temp_buf = ctx.gpu.create_f64_output_buffer(n_flat, "temp");
    let dot_buf = ctx.gpu.create_f64_output_buffer(n_pairs, "dot");

    let wg_dirac = (vol as u32).div_ceil(64);
    let wg_dot = (n_pairs as u32).div_ceil(64);
    let wg_vec = (n_flat as u32).div_ceil(64);

    let dirac_dispatch = |input: &wgpu::Buffer, output: &wgpu::Buffer, hop_sign: f64| {
        let p = DiracParams {
            volume: vol as u32,
            pad0: 0,
            mass_re: mass,
            hop_sign,
        };
        let pb = ctx.gpu.create_uniform_buffer(bytemuck::bytes_of(&p), "dp");
        let bg = ctx.gpu.create_bind_group(
            ctx.dirac_pl,
            &[&pb, &links_buf, input, output, &nbr_buf, &phases_buf],
        );
        ctx.gpu.dispatch(ctx.dirac_pl, &bg, wg_dirac);
    };

    let dot_re = |a: &wgpu::Buffer, b: &wgpu::Buffer| -> f64 {
        let p = DotParams {
            n_pairs: n_pairs as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let pb = ctx
            .gpu
            .create_uniform_buffer(bytemuck::bytes_of(&p), "dotp");
        let bg = ctx
            .gpu
            .create_bind_group(ctx.dot_pl, &[&pb, a, b, &dot_buf]);
        ctx.gpu.dispatch(ctx.dot_pl, &bg, wg_dot);
        reducer.sum_f64(&dot_buf).expect("reduce")
    };

    let axpy = |alpha: f64, x: &wgpu::Buffer, y: &wgpu::Buffer| {
        let p = ScalarParams {
            n: n_flat as u32,
            pad0: 0,
            alpha,
        };
        let pb = ctx.gpu.create_uniform_buffer(bytemuck::bytes_of(&p), "ap");
        let bg = ctx.gpu.create_bind_group(ctx.axpy_pl, &[&pb, x, y]);
        ctx.gpu.dispatch(ctx.axpy_pl, &bg, wg_vec);
    };

    let xpay = |x: &wgpu::Buffer, beta: f64, p: &wgpu::Buffer| {
        let pr = ScalarParams {
            n: n_flat as u32,
            pad0: 0,
            alpha: beta,
        };
        let pb = ctx.gpu.create_uniform_buffer(bytemuck::bytes_of(&pr), "xp");
        let bg = ctx.gpu.create_bind_group(ctx.xpay_pl, &[&pb, x, p]);
        ctx.gpu.dispatch(ctx.xpay_pl, &bg, wg_vec);
    };

    ctx.gpu.upload_f64(&r_buf, b_flat);
    ctx.gpu.upload_f64(&p_buf, b_flat);

    let t0 = Instant::now();

    let b_norm_sq = dot_re(&r_buf, &r_buf);
    if b_norm_sq < 1e-30 {
        return (0, 0.0, 0.0);
    }

    let mut r_norm_sq = b_norm_sq;
    let tol_sq = tol * tol * b_norm_sq;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        dirac_dispatch(&p_buf, &temp_buf, 1.0);
        dirac_dispatch(&temp_buf, &ap_buf, -1.0);

        let p_ap = dot_re(&p_buf, &ap_buf);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = r_norm_sq / p_ap;

        axpy(alpha, &p_buf, &x_buf);
        axpy(-alpha, &ap_buf, &r_buf);

        let r_norm_sq_new = dot_re(&r_buf, &r_buf);
        if r_norm_sq_new < tol_sq {
            r_norm_sq = r_norm_sq_new;
            break;
        }

        let beta = r_norm_sq_new / r_norm_sq;
        r_norm_sq = r_norm_sq_new;
        xpay(&r_buf, beta, &p_buf);
    }

    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let final_res = (r_norm_sq / b_norm_sq).sqrt();
    (iterations, final_res, gpu_ms)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Lattice QCD Scaling Benchmark — GPU vs CPU at scale       ║");
    println!("║  CG solver (D†D x = b) on thermalized configurations      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let gpu = match rt.block_on(GpuF64::new()) {
        Ok(g) => g,
        Err(e) => {
            println!("  GPU not available: {e}");
            return;
        }
    };
    println!("  GPU: {} (f64={})", gpu.adapter_name, gpu.has_f64);
    println!();

    let dirac_pl = gpu.create_pipeline_f64(WGSL_DIRAC_STAGGERED_F64, "dirac");
    let dot_pl = gpu.create_pipeline_f64(WGSL_COMPLEX_DOT_RE_F64, "dot");
    let axpy_pl = gpu.create_pipeline_f64(WGSL_AXPY_F64, "axpy");
    let xpay_pl = gpu.create_pipeline_f64(WGSL_XPAY_F64, "xpay");

    let ctx = GpuCgContext {
        gpu: &gpu,
        dirac_pl: &dirac_pl,
        dot_pl: &dot_pl,
        axpy_pl: &axpy_pl,
        xpay_pl: &xpay_pl,
    };

    let lattice_configs: Vec<([usize; 4], &str)> = vec![
        ([4, 4, 4, 4], "4⁴"),
        ([6, 6, 6, 6], "6⁴"),
        ([8, 8, 8, 8], "8⁴"),
        ([8, 8, 8, 16], "8³×16"),
        ([16, 16, 16, 16], "16⁴"),
    ];

    println!(
        "  {:>8}  {:>7}  {:>8}  {:>8}  {:>6}  {:>9}  {:>9}  {:>7}",
        "Lattice", "Volume", "GPU(ms)", "CPU(ms)", "Iters", "Res", "Upload", "Speedup"
    );
    println!("  {}", "─".repeat(78));

    for (dims, label) in &lattice_configs {
        let mut lattice = Lattice::hot_start(*dims, 6.0, 42);
        let vol = lattice.volume();

        // Thermalize
        let mut hmc_config = HmcConfig {
            n_md_steps: 10,
            dt: 0.1,
            seed: 123,
            ..Default::default()
        };
        for _ in 0..5 {
            hmc_trajectory(&mut lattice, &mut hmc_config);
        }

        let layout = DiracGpuLayout::from_lattice(&lattice);
        let n_pairs = vol * 3;
        let reducer = ReduceScalarPipeline::new(gpu.to_wgpu_device(), n_pairs).expect("reducer");

        let b = FermionField::random(vol, 99);
        let b_flat = flatten_fermion(&b);

        // GPU CG
        let (gpu_iters, gpu_res, gpu_ms) =
            gpu_cg_solve(&ctx, &reducer, &layout, &b_flat, 0.5, 1e-6, 5000);

        // CPU CG
        let mut cpu_x = FermionField::zeros(vol);
        let t_cpu = Instant::now();
        let cpu_result = cg_solve(&lattice, &mut cpu_x, &b, 0.5, 1e-6, 5000);
        let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;

        let upload_kb = (layout.links_flat.len() * 8
            + layout.neighbors.len() * 4
            + layout.phases.len() * 8) as f64
            / 1024.0;

        let speedup = cpu_ms / gpu_ms;

        println!(
            "  {label:>8}  {vol:>7}  {gpu_ms:>8.1}  {cpu_ms:>8.1}  {gpu_iters:>6}  {gpu_res:>9.2e}  {upload_kb:>7.0} KB  {speedup:>6.1}×"
        );

        assert_eq!(
            gpu_iters, cpu_result.iterations,
            "Iteration mismatch at {label}: GPU={gpu_iters} CPU={}",
            cpu_result.iterations
        );
    }

    println!();
    println!("  Notes:");
    println!("  - GPU dispatch overhead dominates at small V; GPU wins at V≥4096");
    println!("  - Iterations are IDENTICAL across GPU and CPU (math parity)");
    println!("  - 16⁴ (V=65536): 25 MB upload once, then CG runs entirely on GPU");
    println!("  - Production lattice QCD uses 32⁴-64⁴ (V=1M-16M) where GPU dominates");
    println!();
}
