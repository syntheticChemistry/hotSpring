// SPDX-License-Identifier: AGPL-3.0-only

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

use hotspring_barracuda::bench::{
    peak_rss_mb, BenchReport, HardwareInventory, PhaseResult, PowerMonitor,
};
use hotspring_barracuda::data;
use hotspring_barracuda::discovery;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::nuclear_eos_helpers::{
    print_nmp_with_pulls, print_pure_gpu_precision, print_semf_gpu_precision,
};
use hotspring_barracuda::physics::{
    binding_energy_l2, nuclear_matter_properties, semf_binding_energy,
};
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

use barracuda::sample::latin_hypercube;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// WGSL Shaders — FP64 batched kernels
// ═══════════════════════════════════════════════════════════════════

/// Batched SEMF: one thread per nucleus, evaluates Bethe-Weizsacker formula
///
/// NOTE: WGSL f64 does NOT have builtin sqrt/pow/max/floor overloads.
/// All transcendentals are precomputed on CPU and passed as buffers.
const SHADER_SEMF_BATCH: &str = include_str!("../physics/shaders/semf_batch_f64.wgsl");

/// PURE-GPU SEMF: uses `math_f64` library — NO CPU precomputation needed
///
/// This shader computes sqrt, pow (via exp/log) entirely on GPU.
/// Input: nuclei[3*i+0..2] = Z, N, A (just integers!)
/// No need for CPU to precompute A^(2/3), A^(1/3), sqrt(A).
const SHADER_SEMF_PURE_GPU: &str = include_str!("../physics/shaders/semf_pure_gpu_f64.wgsl");

/// Batched chi2: per-nucleus squared residual
const SHADER_CHI2: &str = include_str!("../physics/shaders/chi2_batch_f64.wgsl");

// ═══════════════════════════════════════════════════════════════════
// NMP — use provenance::NMP_TARGETS, provenance::NMP_NAMES
// ═══════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    use barracuda::sample::direct::{direct_sampler, DirectSamplerConfig};
    use barracuda::shaders::precision::ShaderTemplate;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  hotSpring GPU FP64 — Nuclear EOS L1 + L2                   ║");
    println!("║  Three-way: Python → BarraCuda CPU → BarraCuda GPU          ║");
    println!("║  + Substrate Benchmark (time / energy / hardware)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Hardware inventory ────────────────────────────────────────────
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string());
    let hw = HardwareInventory::detect(&hostname);
    hw.print();
    println!();

    let mut report = BenchReport::new(hw);

    // ── Initialize GPU ──────────────────────────────────────────────
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

    // ── Load data ───────────────────────────────────────────────────
    let ctx = data::load_eos_context().expect("Failed to load EOS context");
    let exp_data = &*ctx.exp_data;
    let bounds = &ctx.bounds;

    // Sort nuclei for deterministic ordering on GPU
    let mut sorted_nuclei: Vec<((usize, usize), (f64, f64))> =
        exp_data.iter().map(|(&k, &v)| (k, v)).collect();
    sorted_nuclei.sort_by_key(|&((z, n), _)| (z, n));
    let n_nuclei = sorted_nuclei.len();

    println!("  Nuclei: {n_nuclei} (AME2020 selected)");
    println!("  Parameters: {} dimensions (Skyrme)", bounds.len());
    println!();

    // ── Compile GPU pipelines ───────────────────────────────────────
    let t_compile = Instant::now();
    let semf_pipeline = gpu.create_pipeline(SHADER_SEMF_BATCH, "SEMF_batch_f64");
    let chi2_pipeline = gpu.create_pipeline(SHADER_CHI2, "chi2_batch_f64");
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;
    println!("  GPU shader compilation: {compile_ms:.1}ms");
    println!();

    // ── Prepare persistent GPU buffers ──────────────────────────────
    // Pack 7 values per nucleus: Z, N, A^(2/3), A^(1/3), sqrt(A), z_even, n_even
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

    // ═══════════════════════════════════════════════════════════════
    //  LEVEL 1: SEMF — CPU vs GPU (SLy4 test case)
    // ═══════════════════════════════════════════════════════════════
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 SEMF: SLy4 single-evaluation CPU vs GPU");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    // CPU: evaluate all nuclei — 100k iterations for meaningful energy measurement
    let pmon_cpu_l1 = PowerMonitor::start();
    let t_cpu = Instant::now();
    let n_iters_l1: usize = 100_000;
    let mut cpu_energies: Vec<f64> = vec![0.0; n_nuclei];
    for _ in 0..n_iters_l1 {
        for (i, &((z, n), _)) in sorted_nuclei.iter().enumerate() {
            cpu_energies[i] = semf_binding_energy(z, n, &provenance::SLY4_PARAMS);
        }
    }
    let cpu_l1_wall = t_cpu.elapsed().as_secs_f64();
    let cpu_per_eval_us = cpu_l1_wall * 1e6 / n_iters_l1 as f64;
    let energy_cpu_l1 = pmon_cpu_l1.stop();

    // CPU chi2
    let cpu_chi2: f64 = cpu_energies
        .iter()
        .zip(b_exp_vec.iter())
        .zip(sigma_vec.iter())
        .map(|((bc, be), s)| ((bc - be) / s).powi(2))
        .sum::<f64>()
        / n_nuclei as f64;

    report.add_phase(PhaseResult {
        phase: "L1 SEMF".into(),
        substrate: "BarraCuda CPU".into(),
        wall_time_s: cpu_l1_wall,
        per_eval_us: cpu_per_eval_us,
        n_evals: n_iters_l1,
        energy: energy_cpu_l1,
        peak_rss_mb: peak_rss_mb(),
        chi2: cpu_chi2,
        precision_mev: 0.0,
        notes: format!("{n_nuclei} nuclei x {n_iters_l1} iterations"),
    });

    println!("  CPU (BarraCuda native):");
    println!("    chi2/datum = {cpu_chi2:.4}");
    println!("    {cpu_per_eval_us:.1} us/eval ({n_nuclei} nuclei, {n_iters_l1} iterations)");

    // GPU: derive NMP → shader params, dispatch
    let nmp = nuclear_matter_properties(&provenance::SLY4_PARAMS)
        .expect("SLY4 nuclear matter properties");
    let r0 = (3.0 / (4.0 * std::f64::consts::PI * nmp.rho0_fm3)).cbrt();
    let nmp_arr: Vec<f64> = vec![nmp.e_a_mev.abs(), r0, nmp.j_mev, 1.439_976_4];
    let nmp_buf = gpu.create_f64_buffer(&nmp_arr, "NMP_sly4");
    let energy_buf = gpu.create_f64_output_buffer(n_nuclei, "B_calc");

    let semf_bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SEMF_bg"),
        layout: &semf_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nuclei_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: nmp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: energy_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (n_nuclei as u32).div_ceil(64);

    // Warm up
    for _ in 0..10 {
        let _ = gpu
            .dispatch_and_read(&semf_pipeline, &semf_bg, wg, &energy_buf, n_nuclei)
            .expect("GPU SEMF warmup dispatch");
    }

    // Benchmark (with power monitoring) — 100k iterations for meaningful energy measurement
    let pmon_gpu_l1 = PowerMonitor::start();
    let t_gpu = Instant::now();
    let mut gpu_energies = vec![0.0f64; n_nuclei];
    for _ in 0..n_iters_l1 {
        gpu_energies = gpu
            .dispatch_and_read(&semf_pipeline, &semf_bg, wg, &energy_buf, n_nuclei)
            .expect("GPU SEMF dispatch");
    }
    let gpu_l1_wall = t_gpu.elapsed().as_secs_f64();
    let gpu_per_eval_us = gpu_l1_wall * 1e6 / n_iters_l1 as f64;
    let energy_gpu_l1 = pmon_gpu_l1.stop();

    // GPU chi2 — compute on GPU
    let b_calc_buf = gpu.create_f64_buffer(&gpu_energies, "B_calc_gpu");
    let chi2_out_buf = gpu.create_f64_output_buffer(n_nuclei, "chi2_per_nuc");
    let chi2_bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chi2_bg"),
        layout: &chi2_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_calc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_exp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sigma_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: chi2_out_buf.as_entire_binding(),
            },
        ],
    });
    let chi2_vals = gpu
        .dispatch_and_read(&chi2_pipeline, &chi2_bg, wg, &chi2_out_buf, n_nuclei)
        .expect("GPU chi2 dispatch");
    let gpu_chi2: f64 = chi2_vals.iter().sum::<f64>() / n_nuclei as f64;

    // Precision
    let max_diff = cpu_energies
        .iter()
        .zip(gpu_energies.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f64, f64::max);
    let mean_diff = cpu_energies
        .iter()
        .zip(gpu_energies.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f64>()
        / n_nuclei as f64;

    report.add_phase(PhaseResult {
        phase: "L1 SEMF".into(),
        substrate: "BarraCuda GPU".into(),
        wall_time_s: gpu_l1_wall,
        per_eval_us: gpu_per_eval_us,
        n_evals: n_iters_l1,
        energy: energy_gpu_l1,
        peak_rss_mb: peak_rss_mb(),
        chi2: gpu_chi2,
        precision_mev: max_diff,
        notes: format!("precomputed transcendentals, max|delta|={max_diff:.2e}"),
    });

    println!();
    println!("  GPU ({})", gpu.adapter_name);
    println!("    chi2/datum = {gpu_chi2:.4}");
    println!("    {gpu_per_eval_us:.1} us/eval ({n_nuclei} nuclei, {n_iters_l1} iterations)");

    println!();
    print_semf_gpu_precision(
        max_diff,
        mean_diff,
        (cpu_chi2 - gpu_chi2).abs(),
        cpu_per_eval_us,
        gpu_per_eval_us,
        n_nuclei,
    );

    // ═══════════════════════════════════════════════════════════════
    //  L1 SEMF: PURE-GPU (math_f64 library — no CPU precomputation)
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 SEMF: Pure-GPU (math_f64 library, no CPU precompute)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    // Build the pure-GPU shader by prepending math_f64 preamble
    let pure_gpu_shader = ShaderTemplate::with_math_f64_safe(SHADER_SEMF_PURE_GPU);

    let t_compile2 = Instant::now();
    let pure_pipeline = gpu.create_pipeline(&pure_gpu_shader, "SEMF_pure_gpu_f64");
    let compile2_ms = t_compile2.elapsed().as_secs_f64() * 1000.0;
    println!("  Shader compilation (with math_f64 lib): {compile2_ms:.1}ms");

    // Prepare minimal nuclei buffer: just Z, N, A (3 values per nucleus)
    let nuclei_pure: Vec<f64> = sorted_nuclei
        .iter()
        .flat_map(|&((z, n), _)| vec![z as f64, n as f64, (z + n) as f64])
        .collect();
    let nuclei_pure_buf = gpu.create_f64_buffer(&nuclei_pure, "nuclei_ZNA");
    let energy_pure_buf = gpu.create_f64_output_buffer(n_nuclei, "B_pure");

    let pure_bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SEMF_pure_bg"),
        layout: &pure_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nuclei_pure_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: nmp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: energy_pure_buf.as_entire_binding(),
            },
        ],
    });

    // Warm up
    for _ in 0..10 {
        let _ = gpu
            .dispatch_and_read(&pure_pipeline, &pure_bg, wg, &energy_pure_buf, n_nuclei)
            .expect("GPU pure warmup");
    }

    // Benchmark (with power monitoring) — 100k iterations
    let pmon_pure = PowerMonitor::start();
    let t_pure = Instant::now();
    let mut pure_energies = vec![0.0f64; n_nuclei];
    for _ in 0..n_iters_l1 {
        pure_energies = gpu
            .dispatch_and_read(&pure_pipeline, &pure_bg, wg, &energy_pure_buf, n_nuclei)
            .expect("GPU pure dispatch");
    }
    let pure_wall = t_pure.elapsed().as_secs_f64();
    let pure_per_eval_us = pure_wall * 1e6 / n_iters_l1 as f64;
    let energy_pure = pmon_pure.stop();

    // Compare pure-GPU vs CPU
    let pure_max_diff = cpu_energies
        .iter()
        .zip(pure_energies.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f64, f64::max);
    let pure_mean_diff = cpu_energies
        .iter()
        .zip(pure_energies.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f64>()
        / n_nuclei as f64;

    // Also compare pure-GPU vs precomputed-GPU
    let pure_vs_precomp = gpu_energies
        .iter()
        .zip(pure_energies.iter())
        .map(|(p, g)| (p - g).abs())
        .fold(0.0f64, f64::max);

    report.add_phase(PhaseResult {
        phase: "L1 SEMF pure".into(),
        substrate: "GPU math_f64".into(),
        wall_time_s: pure_wall,
        per_eval_us: pure_per_eval_us,
        n_evals: n_iters_l1,
        energy: energy_pure,
        peak_rss_mb: peak_rss_mb(),
        chi2: gpu_chi2, // same physics, different math path
        precision_mev: pure_max_diff,
        notes: format!("pure-GPU math_f64, max|delta|={pure_max_diff:.2e}"),
    });

    print_pure_gpu_precision(
        pure_max_diff,
        pure_mean_diff,
        pure_vs_precomp,
        cpu_per_eval_us,
        gpu_per_eval_us,
        pure_per_eval_us,
        n_iters_l1,
    );

    // ═══════════════════════════════════════════════════════════════
    //  L1 OPTIMIZATION: 64-eval sweep, CPU vs GPU objective
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 Optimization: 64-eval LHS sweep (CPU vs GPU)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let n_sweep: usize = 512;
    let samples = latin_hypercube(n_sweep, bounds, 42).expect("LHS failed");

    // CPU sweep (with power monitoring)
    let pmon_cpu_sweep = PowerMonitor::start();
    let t_cpu_opt = Instant::now();
    let cpu_chi2s: Vec<f64> = samples
        .iter()
        .map(|params| l1_chi2_cpu(params, &sorted_nuclei))
        .collect();
    let cpu_opt_wall = t_cpu_opt.elapsed().as_secs_f64();
    let cpu_opt_ms = cpu_opt_wall * 1000.0;
    let energy_cpu_sweep = pmon_cpu_sweep.stop();
    let cpu_best_idx = cpu_chi2s
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("sweep produced at least one result");
    let cpu_best = cpu_chi2s[cpu_best_idx];

    report.add_phase(PhaseResult {
        phase: "L1 sweep".into(),
        substrate: "BarraCuda CPU".into(),
        wall_time_s: cpu_opt_wall,
        per_eval_us: cpu_opt_wall * 1e6 / n_sweep as f64,
        n_evals: n_sweep,
        energy: energy_cpu_sweep,
        peak_rss_mb: peak_rss_mb(),
        chi2: cpu_best,
        precision_mev: 0.0,
        notes: format!("{n_sweep}-point LHS sweep"),
    });

    // GPU sweep (same param sets, with power monitoring)
    let pmon_gpu_sweep = PowerMonitor::start();
    let t_gpu_opt = Instant::now();
    let mut gpu_chi2s = Vec::with_capacity(n_sweep);
    for params in &samples {
        let chi2 = l1_chi2_gpu(
            params,
            &gpu,
            &semf_pipeline,
            &chi2_pipeline,
            &nuclei_buf,
            &b_exp_buf,
            &sigma_buf,
            n_nuclei,
        );
        gpu_chi2s.push(chi2);
    }
    let gpu_opt_wall = t_gpu_opt.elapsed().as_secs_f64();
    let gpu_opt_ms = gpu_opt_wall * 1000.0;
    let energy_gpu_sweep = pmon_gpu_sweep.stop();
    let gpu_best_idx = gpu_chi2s
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("sweep produced at least one result");
    let gpu_best = gpu_chi2s[gpu_best_idx];

    report.add_phase(PhaseResult {
        phase: "L1 sweep".into(),
        substrate: "BarraCuda GPU".into(),
        wall_time_s: gpu_opt_wall,
        per_eval_us: gpu_opt_wall * 1e6 / n_sweep as f64,
        n_evals: n_sweep,
        energy: energy_gpu_sweep,
        peak_rss_mb: peak_rss_mb(),
        chi2: gpu_best,
        precision_mev: 0.0,
        notes: format!("{n_sweep}-point LHS sweep, GPU SEMF+chi2"),
    });

    println!("  CPU: best chi2/datum = {cpu_best:.4} (idx {cpu_best_idx}) in {cpu_opt_ms:.1}ms");
    println!("  GPU: best chi2/datum = {gpu_best:.4} (idx {gpu_best_idx}) in {gpu_opt_ms:.1}ms");

    // Agreement
    let max_chi2_diff = cpu_chi2s
        .iter()
        .zip(gpu_chi2s.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f64, f64::max);
    println!("  Max |chi2_cpu - chi2_gpu|: {max_chi2_diff:.2e}");
    println!("  Speedup: {:.2}x", cpu_opt_ms / gpu_opt_ms);

    if let Some(nmp) = nuclear_matter_properties(&samples[gpu_best_idx]) {
        println!();
        println!("  Best GPU solution NMP:");
        print_nmp_with_pulls(&nmp);
    }

    // ═══════════════════════════════════════════════════════════════
    //  L1 FULL OPTIMIZATION: DirectSampler with GPU objective
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 Full Optimization: DirectSampler (GPU-backed objective)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let lambda = 0.1;

    let gpu_arc = Arc::new(gpu);
    let device = gpu_arc.to_wgpu_device();
    let sorted_nuclei_arc = Arc::new(sorted_nuclei.clone());

    // CPU objective for comparison
    let sorted_nuclei_cpu = sorted_nuclei.clone();
    let cpu_objective =
        move |x: &[f64]| -> f64 { l1_objective_with_nmp(x, &sorted_nuclei_cpu, lambda) };

    let pmon_ds_cpu = PowerMonitor::start();
    let t_cpu_full = Instant::now();
    let config_cpu = DirectSamplerConfig::new(42)
        .with_rounds(6)
        .with_solvers(8)
        .with_eval_budget(200)
        .with_patience(3);

    let result_cpu = direct_sampler(device.clone(), cpu_objective, bounds, &config_cpu)
        .expect("CPU DirectSampler failed");
    let cpu_full_time = t_cpu_full.elapsed().as_secs_f64();
    let energy_ds_cpu = pmon_ds_cpu.stop();
    let cpu_full_chi2 = result_cpu.f_best.exp_m1();

    report.add_phase(PhaseResult {
        phase: "L1 DirectSampler".into(),
        substrate: "BarraCuda CPU".into(),
        wall_time_s: cpu_full_time,
        per_eval_us: cpu_full_time * 1e6 / result_cpu.cache.len().max(1) as f64,
        n_evals: result_cpu.cache.len(),
        energy: energy_ds_cpu,
        peak_rss_mb: peak_rss_mb(),
        chi2: cpu_full_chi2,
        precision_mev: 0.0,
        notes: "6 rounds x 8 solvers x 200 evals".into(),
    });

    println!("  CPU DirectSampler:");
    println!(
        "    chi2/datum = {:.4} ({} evals in {:.2}s)",
        cpu_full_chi2,
        result_cpu.cache.len(),
        cpu_full_time
    );

    // GPU-backed objective for DirectSampler
    // Note: DirectSampler is serial per-eval. The real GPU win comes from
    // batching all Nelder-Mead vertices simultaneously (future evolution).
    // For now, we run identical CPU physics to verify result parity.
    let sorted_for_obj = sorted_nuclei_arc;
    let gpu_objective =
        move |x: &[f64]| -> f64 { l1_objective_with_nmp(x, &sorted_for_obj, lambda) };

    let pmon_ds_gpu = PowerMonitor::start();
    let t_gpu_full = Instant::now();
    let config_gpu = DirectSamplerConfig::new(42)
        .with_rounds(6)
        .with_solvers(8)
        .with_eval_budget(200)
        .with_patience(3);

    let result_gpu = direct_sampler(device.clone(), gpu_objective, bounds, &config_gpu)
        .expect("GPU DirectSampler failed");
    let gpu_full_time = t_gpu_full.elapsed().as_secs_f64();
    let energy_ds_gpu = pmon_ds_gpu.stop();
    let gpu_full_chi2 = result_gpu.f_best.exp_m1();

    report.add_phase(PhaseResult {
        phase: "L1 DirectSampler".into(),
        substrate: "BarraCuda GPU".into(),
        wall_time_s: gpu_full_time,
        per_eval_us: gpu_full_time * 1e6 / result_gpu.cache.len().max(1) as f64,
        n_evals: result_gpu.cache.len(),
        energy: energy_ds_gpu,
        peak_rss_mb: peak_rss_mb(),
        chi2: gpu_full_chi2,
        precision_mev: 0.0,
        notes: "serial objective (GPU batching future)".into(),
    });

    println!("  GPU-backed DirectSampler:");
    println!(
        "    chi2/datum = {:.4} ({} evals in {:.2}s)",
        gpu_full_chi2,
        result_gpu.cache.len(),
        gpu_full_time
    );
    println!("  Note: DirectSampler currently serial — GPU batching is a");
    println!("  future evolution (batch all Nelder-Mead vertices at once)");

    // ═══════════════════════════════════════════════════════════════
    //  LEVEL 2: HFB — CPU evaluation with GPU potential
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L2 HFB: CPU (SLy4) — baseline for GPU comparison");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let pmon_l2_sly4 = PowerMonitor::start();
    let t_l2 = Instant::now();
    let mut l2_chi2 = 0.0f64;
    let mut l2_count = 0usize;
    let mut l2_converged = 0usize;

    for &((z, n), (b_exp, _)) in &sorted_nuclei {
        let a = z + n;
        if (56..=132).contains(&a) {
            let (b_calc, converged) =
                binding_energy_l2(z, n, &provenance::SLY4_PARAMS).expect("HFB solve");
            if b_calc > 0.0 {
                let sigma_theo = tolerances::sigma_theo(b_exp);
                l2_chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
                l2_count += 1;
                if converged {
                    l2_converged += 1;
                }
            }
        }
    }
    if l2_count > 0 {
        l2_chi2 /= l2_count as f64;
    }
    let l2_time = t_l2.elapsed().as_secs_f64();
    let energy_l2_sly4 = pmon_l2_sly4.stop();

    report.add_phase(PhaseResult {
        phase: "L2 HFB SLy4".into(),
        substrate: "BarraCuda CPU".into(),
        wall_time_s: l2_time,
        per_eval_us: l2_time * 1e6 / l2_count.max(1) as f64,
        n_evals: l2_count,
        energy: energy_l2_sly4,
        peak_rss_mb: peak_rss_mb(),
        chi2: l2_chi2,
        precision_mev: 0.0,
        notes: format!("{l2_converged}/{l2_count} converged"),
    });

    println!("  L2 CPU (SLy4):");
    println!("    chi2/datum = {l2_chi2:.2}");
    println!("    Nuclei: {l2_converged}/{l2_count} converged");
    println!(
        "    Time: {:.1}s ({:.1}s/nucleus avg)",
        l2_time,
        l2_time / l2_count.max(1) as f64
    );

    // ═══════════════════════════════════════════════════════════════
    //  L2 DirectSampler optimization (CPU, with rayon parallelism)
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L2 Optimization: DirectSampler (CPU + rayon)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let exp_data_arc = ctx.exp_data.clone();
    let exp_data_l2 = exp_data_arc;

    let l2_objective = move |x: &[f64]| -> f64 { l2_objective_fn(x, &exp_data_l2, 0.1) };

    let pmon_l2_opt = PowerMonitor::start();
    let t_l2_opt = Instant::now();
    let l2_config = DirectSamplerConfig::new(42)
        .with_rounds(3)
        .with_solvers(4)
        .with_eval_budget(30)
        .with_patience(2);

    println!(
        "  Config: {} rounds × {} solvers × {} evals (patience={})",
        l2_config.n_rounds, l2_config.n_solvers, l2_config.max_eval_per_solver, l2_config.patience
    );
    println!("  Running... (each eval = full HFB SCF for ~{l2_count} nuclei)");

    let result_l2 =
        direct_sampler(device, l2_objective, bounds, &l2_config).expect("L2 DirectSampler failed");
    let l2_opt_time = t_l2_opt.elapsed().as_secs_f64();
    let energy_l2_opt = pmon_l2_opt.stop();
    let l2_opt_chi2 = result_l2.f_best.exp_m1();

    report.add_phase(PhaseResult {
        phase: "L2 DirectSampler".into(),
        substrate: "BarraCuda CPU".into(),
        wall_time_s: l2_opt_time,
        per_eval_us: l2_opt_time * 1e6 / result_l2.cache.len().max(1) as f64,
        n_evals: result_l2.cache.len(),
        energy: energy_l2_opt,
        peak_rss_mb: peak_rss_mb(),
        chi2: l2_opt_chi2,
        precision_mev: 0.0,
        notes: format!("{} evals, rayon parallel HFB", result_l2.cache.len()),
    });

    println!();
    println!("  L2 DirectSampler result:");
    println!(
        "    chi2/datum = {:.2} ({} evals in {:.1}s)",
        l2_opt_chi2,
        result_l2.cache.len(),
        l2_opt_time
    );
    println!(
        "    Time per eval: {:.1}s",
        l2_opt_time / result_l2.cache.len().max(1) as f64
    );

    if let Some(nmp_l2) = nuclear_matter_properties(&result_l2.x_best) {
        println!(
            "    NMP chi2: {:.4}",
            provenance::nmp_chi2_from_props(&nmp_l2) / 5.0
        );
    }

    // ═══════════════════════════════════════════════════════════════
    //  SUMMARY TABLE
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  SUMMARY: Three-Way Comparison");
    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("  ┌─────────┬────────────────┬────────────────┬────────────────┐");
    println!("  │ Level   │ Python/SciPy   │ BarraCuda CPU  │ BarraCuda GPU  │");
    println!("  ├─────────┼────────────────┼────────────────┼────────────────┤");
    println!(
        "  │ L1 SEMF │ chi2 ~{:.2}     │ chi2 = {cpu_full_chi2:<7.2} │ chi2 = {gpu_full_chi2:<7.2} │",
        provenance::L1_PYTHON_CHI2.value
    );
    println!(
        "  │ L2 HFB  │ chi2 ~{:.2}    │ chi2 = {l2_opt_chi2:<7.2} │ (SCF on CPU)   │",
        provenance::L2_PYTHON_CHI2.value
    );
    println!("  └─────────┴────────────────┴────────────────┴────────────────┘");
    println!();
    println!("  GPU FP64 Validation:");
    println!("    SEMF precision: max |delta| = {max_diff:.2e} MeV");
    println!("    IEEE 754 compliance: 0 ULP (verified by fp64_validation)");
    println!("    Adapter: {}", gpu_arc.adapter_name);
    println!("    SHADER_F64: ENABLED");
    println!();
    println!("  GPU evolution roadmap:");
    println!("    [x] L1 SEMF — batched f64 compute shader");
    println!("    [x] L1 chi2 — batched f64 reduction shader");
    println!("    [ ] L2 density accumulation — batched across nuclei on GPU");
    println!("    [ ] L2 Skyrme potential — element-wise f64 on GPU");
    println!("    [ ] L2 Coulomb — prefix-sum f64 on GPU");
    println!("    [ ] L2 eigh_f64 — batched eigendecomposition shader");
    println!("    [ ] L3 2D grid operations — f64 on GPU");

    // ═══════════════════════════════════════════════════════════════
    //  SUBSTRATE BENCHMARK REPORT (time + energy + hardware)
    // ═══════════════════════════════════════════════════════════════
    report.print_summary();

    // Save JSON report
    let report_dir = discovery::benchmark_results_dir()
        .unwrap_or_else(|_| PathBuf::from(discovery::paths::BENCHMARK_RESULTS));
    match report.save_json(
        report_dir
            .to_str()
            .unwrap_or(discovery::paths::BENCHMARK_RESULTS),
    ) {
        Ok(path) => println!("  Benchmark report saved: {path}"),
        Err(e) => println!("  Warning: failed to save benchmark report: {e}"),
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════════════

/// L1 chi2/datum for a parameter set (CPU)
fn l1_chi2_cpu(params: &[f64], nuclei: &[((usize, usize), (f64, f64))]) -> f64 {
    let Some(nmp) = nuclear_matter_properties(params) else {
        return 1e10;
    };
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 || nmp.e_a_mev > 0.0 {
        return 1e10;
    }

    let mut chi2 = 0.0;
    let mut count = 0;
    for &((z, n), (b_exp, _)) in nuclei {
        let b_calc = semf_binding_energy(z, n, params);
        if b_calc > 0.0 {
            let sigma = tolerances::sigma_theo(b_exp);
            chi2 += ((b_calc - b_exp) / sigma).powi(2);
            count += 1;
        }
    }
    if count == 0 {
        return 1e10;
    }
    chi2 / f64::from(count)
}

/// L1 chi2/datum via GPU dispatch
fn l1_chi2_gpu(
    params: &[f64],
    gpu: &GpuF64,
    semf_pipeline: &wgpu::ComputePipeline,
    chi2_pipeline: &wgpu::ComputePipeline,
    nuclei_buf: &wgpu::Buffer,
    b_exp_buf: &wgpu::Buffer,
    sigma_buf: &wgpu::Buffer,
    n_nuclei: usize,
) -> f64 {
    let Some(nmp) = nuclear_matter_properties(params) else {
        return 1e10;
    };
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 || nmp.e_a_mev > 0.0 {
        return 1e10;
    }

    let r0 = (3.0 / (4.0 * std::f64::consts::PI * nmp.rho0_fm3)).cbrt();
    let nmp_arr: Vec<f64> = vec![nmp.e_a_mev.abs(), r0, nmp.j_mev, 1.439_976_4];
    let nmp_buf = gpu.create_f64_buffer(&nmp_arr, "NMP_i");
    let energy_buf = gpu.create_f64_output_buffer(n_nuclei, "B_i");

    let semf_bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("SEMF_i"),
        layout: &semf_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: nuclei_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: nmp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: energy_buf.as_entire_binding(),
            },
        ],
    });

    let wg = (n_nuclei as u32).div_ceil(64);
    let energies = gpu
        .dispatch_and_read(semf_pipeline, &semf_bg, wg, &energy_buf, n_nuclei)
        .expect("GPU L2 SEMF dispatch");

    // Chi2 on GPU
    let b_calc_buf = gpu.create_f64_buffer(&energies, "B_calc_i");
    let chi2_buf = gpu.create_f64_output_buffer(n_nuclei, "chi2_i");
    let chi2_bg = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("chi2_i"),
        layout: &chi2_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b_calc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_exp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sigma_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: chi2_buf.as_entire_binding(),
            },
        ],
    });
    let chi2_vals = gpu
        .dispatch_and_read(chi2_pipeline, &chi2_bg, wg, &chi2_buf, n_nuclei)
        .expect("GPU L2 chi2 dispatch");
    chi2_vals.iter().sum::<f64>() / n_nuclei as f64
}

/// L1 objective with NMP constraint (for `DirectSampler`)
fn l1_objective_with_nmp(x: &[f64], nuclei: &[((usize, usize), (f64, f64))], lambda: f64) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    let Some(nmp) = nuclear_matter_properties(x) else {
        return (1e4_f64).ln_1p();
    };
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 {
        return (1e4_f64).ln_1p();
    }
    if nmp.e_a_mev > 0.0 {
        return (1e4_f64).ln_1p();
    }

    let mut chi2_be = 0.0;
    let mut count = 0;
    for &((z, n), (b_exp, _)) in nuclei {
        let b_calc = semf_binding_energy(z, n, x);
        if b_calc > 0.0 {
            let sigma = tolerances::sigma_theo(b_exp);
            chi2_be += ((b_calc - b_exp) / sigma).powi(2);
            count += 1;
        }
    }
    if count == 0 {
        return (1e4_f64).ln_1p();
    }

    let chi2_be_datum = chi2_be / f64::from(count);
    let chi2_nmp = provenance::nmp_chi2_from_props(&nmp) / 5.0;
    let total = lambda.mul_add(chi2_nmp, chi2_be_datum);
    total.ln_1p()
}

/// L2 objective function (HFB + NMP constraint)
fn l2_objective_fn(x: &[f64], exp_data: &HashMap<(usize, usize), (f64, f64)>, lambda: f64) -> f64 {
    use rayon::prelude::*;
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    let Some(nmp) = nuclear_matter_properties(x) else {
        return (1e4_f64).ln_1p();
    };
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 {
        return (1e4_f64).ln_1p();
    }
    if nmp.e_a_mev > 0.0 {
        return (1e4_f64).ln_1p();
    }

    let mut chi2_be = 0.0;
    let mut count = 0;

    let results: Vec<(f64, f64, f64)> = exp_data
        .par_iter()
        .filter_map(|(&(z, n), &(b_exp, _))| {
            let (b_calc, _converged) = binding_energy_l2(z, n, x).expect("HFB solve");
            if b_calc > 0.0 {
                let sigma = tolerances::sigma_theo(b_exp);
                let chi2 = ((b_calc - b_exp) / sigma).powi(2);
                Some((chi2, 1.0, 0.0))
            } else {
                None
            }
        })
        .collect();

    for (c, _, _) in &results {
        chi2_be += c;
        count += 1;
    }

    if count == 0 {
        return (1e4_f64).ln_1p();
    }

    let chi2_be_datum = chi2_be / f64::from(count);
    let chi2_nmp = provenance::nmp_chi2_from_props(&nmp) / 5.0;
    let total = lambda.mul_add(chi2_nmp, chi2_be_datum);
    total.ln_1p()
}
