// SPDX-License-Identifier: AGPL-3.0-or-later

//! Nuclear EOS — GPU FP64 Validation (`SHADER_F64` on RTX 4070)
//!
//! Three-way comparison:
//!   1. Python/SciPy control  (reference numbers from prior runs)
//!   2. `BarraCuda` CPU         (existing L1/L2 path)
//!   3. `BarraCuda` GPU         (NEW: f64 compute shaders via wgpu/Vulkan)
//!
//! L1: Batched SEMF on GPU (all nuclei in one dispatch)
//! L2: CPU HFB with GPU-accelerated density/potential (per-SCF-iteration)
//!
//! Run: cargo run --release --bin `nuclear_eos_gpu`

use barracuda::ops::physics::{WGSL_CHI2_BATCH, WGSL_SEMF_BATCH, WGSL_SEMF_PURE_GPU};
use hotspring_barracuda::bench::{BenchReport, HardwareInventory};
use hotspring_barracuda::bin_helpers::nuclear_eos_gpu::{
    print_summary_table, run_l1_direct_sampler, run_l1_lhs_sweep, run_l1_semf_cpu_vs_gpu,
    run_l1_semf_pure_gpu, run_l2_direct_sampler, run_l2_hfb_baseline,
};
use hotspring_barracuda::data;
use hotspring_barracuda::gpu::GpuF64;

use std::time::Instant;

/// Batched SEMF: one thread per nucleus, evaluates Bethe-Weizsacker formula
const SHADER_SEMF_BATCH: &str = WGSL_SEMF_BATCH;

/// PURE-GPU SEMF: uses `math_f64` library — NO CPU precomputation needed
const SHADER_SEMF_PURE_GPU: &str = WGSL_SEMF_PURE_GPU;

/// Batched chi2: per-nucleus squared residual
const SHADER_CHI2: &str = WGSL_CHI2_BATCH;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  hotSpring GPU FP64 — Nuclear EOS L1 + L2                   ║");
    println!("║  Three-way: Python → BarraCuda CPU → BarraCuda GPU          ║");
    println!("║  + Substrate Benchmark (time / energy / hardware)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let hw = HardwareInventory::detect_local();
    hw.print();
    println!();

    let mut report = BenchReport::new(hw);

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime creation");
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("Failed to create GPU device");
    print!("  ");
    gpu.print_info();
    if !gpu.has_f64 {
        println!("\n  SHADER_F64 not supported — cannot run FP64 science compute.");
        println!("  Check GPU capabilities with: cargo run --bin f64_builtin_test");
        return;
    }
    println!();

    let ctx = data::load_eos_context().expect("Failed to load EOS context");
    let bounds = &ctx.bounds;

    let mut sorted_nuclei: Vec<((usize, usize), (f64, f64))> =
        ctx.exp_data.iter().map(|(&k, &v)| (k, v)).collect();
    sorted_nuclei.sort_by_key(|&((z, n), _)| (z, n));
    let n_nuclei = sorted_nuclei.len();

    println!("  Nuclei: {n_nuclei} (AME2020 selected)");
    println!("  Parameters: {} dimensions (Skyrme)", bounds.len());
    println!();

    let t_compile = Instant::now();
    let semf_pipeline = gpu.create_pipeline(SHADER_SEMF_BATCH, "SEMF_batch_f64");
    let chi2_pipeline = gpu.create_pipeline(SHADER_CHI2, "chi2_batch_f64");
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;
    println!("  GPU shader compilation: {compile_ms:.1}ms");
    println!();

    let nuclei_flat: Vec<f64> = sorted_nuclei
        .iter()
        .flat_map(|&((z, n), _)| {
            let a = (z + n) as f64;
            vec![
                z as f64,
                n as f64,
                a.powf(2.0 / 3.0),
                a.cbrt(),
                a.sqrt(),
                if z % 2 == 0 { 1.0 } else { 0.0 },
                if n % 2 == 0 { 1.0 } else { 0.0 },
            ]
        })
        .collect();
    let b_exp_vec: Vec<f64> = sorted_nuclei.iter().map(|&(_, (b, _))| b).collect();
    let sigma_vec: Vec<f64> = b_exp_vec.iter().map(|&b| (0.01 * b).max(2.0)).collect();

    let nuclei_buf = gpu.create_f64_buffer(&nuclei_flat, "nuclei_ZN");
    let b_exp_buf = gpu.create_f64_buffer(&b_exp_vec, "B_exp");
    let sigma_buf = gpu.create_f64_buffer(&sigma_vec, "sigma");

    let l1 = run_l1_semf_cpu_vs_gpu(
        &gpu,
        &mut report,
        &sorted_nuclei,
        n_nuclei,
        &semf_pipeline,
        &chi2_pipeline,
        &nuclei_buf,
        &b_exp_buf,
        &sigma_buf,
        &b_exp_vec,
        &sigma_vec,
    );

    run_l1_semf_pure_gpu(
        &gpu,
        &mut report,
        &sorted_nuclei,
        n_nuclei,
        &l1,
        SHADER_SEMF_PURE_GPU,
    );

    run_l1_lhs_sweep(
        &gpu,
        &mut report,
        &sorted_nuclei,
        bounds,
        &semf_pipeline,
        &chi2_pipeline,
        &nuclei_buf,
        &b_exp_buf,
        &sigma_buf,
        n_nuclei,
    );

    let l1_ds = run_l1_direct_sampler(gpu, &mut report, &sorted_nuclei, bounds);

    let l2 = run_l2_hfb_baseline(&mut report, &sorted_nuclei);
    let l2_opt_chi2 = run_l2_direct_sampler(
        &mut report,
        &ctx,
        bounds,
        l1_ds.device,
        l2.l2_count,
    );

    print_summary_table(
        l1_ds.cpu_full_chi2,
        l1_ds.gpu_full_chi2,
        l2_opt_chi2,
        l1.max_diff,
        &l1_ds.gpu_arc,
    );

    report.print_summary();
    report.save_and_print();
    println!();
}
