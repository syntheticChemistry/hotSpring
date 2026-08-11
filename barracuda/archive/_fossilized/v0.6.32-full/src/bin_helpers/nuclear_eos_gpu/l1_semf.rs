// SPDX-License-Identifier: AGPL-3.0-or-later

//! L1 SEMF validation, LHS sweep, and DirectSampler optimization phases.

use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;
use barracuda::sample::direct::{DirectSamplerConfig, direct_sampler};
use barracuda::sample::latin_hypercube;
use barracuda::shaders::precision::ShaderTemplate;
use hotspring_barracuda::bench::{BenchReport, PhaseResult, PowerMonitor, peak_rss_mb};
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::nuclear_eos_helpers::{
    l1_chi2_cpu_nuclei, l1_objective_nmp_nuclei, print_nmp_with_pulls, print_pure_gpu_precision,
    print_semf_gpu_precision,
};
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::provenance;

pub struct L1SemfResult {
    pub cpu_energies: Vec<f64>,
    pub gpu_energies: Vec<f64>,
    pub gpu_chi2: f64,
    pub max_diff: f64,
    pub cpu_per_eval_us: f64,
    pub gpu_per_eval_us: f64,
    pub nmp_buf: wgpu::Buffer,
    pub wg: u32,
}

pub struct L1DirectSamplerResult {
    pub cpu_full_chi2: f64,
    pub gpu_full_chi2: f64,
    pub gpu_arc: Arc<GpuF64>,
    pub device: Arc<WgpuDevice>,
}

/// L1 SEMF: SLy4 single-evaluation CPU vs GPU (precomputed transcendentals path).
pub fn run_l1_semf_cpu_vs_gpu(
    gpu: &GpuF64,
    report: &mut BenchReport,
    sorted_nuclei: &[((usize, usize), (f64, f64))],
    n_nuclei: usize,
    semf_pipeline: &wgpu::ComputePipeline,
    chi2_pipeline: &wgpu::ComputePipeline,
    nuclei_buf: &wgpu::Buffer,
    b_exp_buf: &wgpu::Buffer,
    sigma_buf: &wgpu::Buffer,
    b_exp_vec: &[f64],
    sigma_vec: &[f64],
) -> L1SemfResult {
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 SEMF: SLy4 single-evaluation CPU vs GPU");
    println!("══════════════════════════════════════════════════════════════");
    println!();

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

    for _ in 0..10 {
        let _ = gpu
            .dispatch_and_read(semf_pipeline, &semf_bg, wg, &energy_buf, n_nuclei)
            .expect("GPU SEMF warmup dispatch");
    }

    let pmon_gpu_l1 = PowerMonitor::start();
    let t_gpu = Instant::now();
    let mut gpu_energies = vec![0.0f64; n_nuclei];
    for _ in 0..n_iters_l1 {
        gpu_energies = gpu
            .dispatch_and_read(semf_pipeline, &semf_bg, wg, &energy_buf, n_nuclei)
            .expect("GPU SEMF dispatch");
    }
    let gpu_l1_wall = t_gpu.elapsed().as_secs_f64();
    let gpu_per_eval_us = gpu_l1_wall * 1e6 / n_iters_l1 as f64;
    let energy_gpu_l1 = pmon_gpu_l1.stop();

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
        .dispatch_and_read(chi2_pipeline, &chi2_bg, wg, &chi2_out_buf, n_nuclei)
        .expect("GPU chi2 dispatch");
    let gpu_chi2: f64 = chi2_vals.iter().sum::<f64>() / n_nuclei as f64;

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

    L1SemfResult {
        cpu_energies,
        gpu_energies,
        gpu_chi2,
        max_diff,
        cpu_per_eval_us,
        gpu_per_eval_us,
        nmp_buf,
        wg,
    }
}

/// L1 SEMF: pure-GPU path using math_f64 library (no CPU precomputation).
pub fn run_l1_semf_pure_gpu(
    gpu: &GpuF64,
    report: &mut BenchReport,
    sorted_nuclei: &[((usize, usize), (f64, f64))],
    n_nuclei: usize,
    l1: &L1SemfResult,
    shader_semf_pure_gpu: &str,
) {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 SEMF: Pure-GPU (math_f64 library, no CPU precompute)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let pure_gpu_shader = ShaderTemplate::with_math_f64_safe(shader_semf_pure_gpu);

    let t_compile2 = Instant::now();
    let pure_pipeline = gpu.create_pipeline(&pure_gpu_shader, "SEMF_pure_gpu_f64");
    let compile2_ms = t_compile2.elapsed().as_secs_f64() * 1000.0;
    println!("  Shader compilation (with math_f64 lib): {compile2_ms:.1}ms");

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
                resource: l1.nmp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: energy_pure_buf.as_entire_binding(),
            },
        ],
    });

    for _ in 0..10 {
        let _ = gpu
            .dispatch_and_read(&pure_pipeline, &pure_bg, l1.wg, &energy_pure_buf, n_nuclei)
            .expect("GPU pure warmup");
    }

    let n_iters_l1: usize = 100_000;
    let pmon_pure = PowerMonitor::start();
    let t_pure = Instant::now();
    let mut pure_energies = vec![0.0f64; n_nuclei];
    for _ in 0..n_iters_l1 {
        pure_energies = gpu
            .dispatch_and_read(&pure_pipeline, &pure_bg, l1.wg, &energy_pure_buf, n_nuclei)
            .expect("GPU pure dispatch");
    }
    let pure_wall = t_pure.elapsed().as_secs_f64();
    let pure_per_eval_us = pure_wall * 1e6 / n_iters_l1 as f64;
    let energy_pure = pmon_pure.stop();

    let pure_max_diff = l1
        .cpu_energies
        .iter()
        .zip(pure_energies.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f64, f64::max);
    let pure_mean_diff = l1
        .cpu_energies
        .iter()
        .zip(pure_energies.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f64>()
        / n_nuclei as f64;

    let pure_vs_precomp = l1
        .gpu_energies
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
        chi2: l1.gpu_chi2,
        precision_mev: pure_max_diff,
        notes: format!("pure-GPU math_f64, max|delta|={pure_max_diff:.2e}"),
    });

    print_pure_gpu_precision(
        pure_max_diff,
        pure_mean_diff,
        pure_vs_precomp,
        l1.cpu_per_eval_us,
        l1.gpu_per_eval_us,
        pure_per_eval_us,
        n_iters_l1,
    );
}

/// L1 optimization: 512-point LHS sweep comparing CPU vs GPU chi2 objective.
pub fn run_l1_lhs_sweep(
    gpu: &GpuF64,
    report: &mut BenchReport,
    sorted_nuclei: &[((usize, usize), (f64, f64))],
    bounds: &[(f64, f64)],
    semf_pipeline: &wgpu::ComputePipeline,
    chi2_pipeline: &wgpu::ComputePipeline,
    nuclei_buf: &wgpu::Buffer,
    b_exp_buf: &wgpu::Buffer,
    sigma_buf: &wgpu::Buffer,
    n_nuclei: usize,
) {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 Optimization: 64-eval LHS sweep (CPU vs GPU)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let n_sweep: usize = 512;
    let samples = latin_hypercube(n_sweep, bounds, 42).expect("LHS failed");

    let pmon_cpu_sweep = PowerMonitor::start();
    let t_cpu_opt = Instant::now();
    let cpu_chi2s: Vec<f64> = samples
        .iter()
        .map(|params| l1_chi2_cpu_nuclei(params, sorted_nuclei))
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

    let pmon_gpu_sweep = PowerMonitor::start();
    let t_gpu_opt = Instant::now();
    let mut gpu_chi2s = Vec::with_capacity(n_sweep);
    for params in &samples {
        let chi2 = l1_chi2_gpu(
            params,
            gpu,
            semf_pipeline,
            chi2_pipeline,
            nuclei_buf,
            b_exp_buf,
            sigma_buf,
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
}

/// L1 full optimization via DirectSampler (CPU vs GPU-backed objective).
pub fn run_l1_direct_sampler(
    gpu: GpuF64,
    report: &mut BenchReport,
    sorted_nuclei: &[((usize, usize), (f64, f64))],
    bounds: &[(f64, f64)],
) -> L1DirectSamplerResult {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  L1 Full Optimization: DirectSampler (GPU-backed objective)");
    println!("══════════════════════════════════════════════════════════════");
    println!();

    let lambda = 0.1;

    let gpu_arc = Arc::new(gpu);
    let device = gpu_arc.to_wgpu_device();
    let sorted_nuclei_arc = Arc::new(sorted_nuclei.to_vec());

    let sorted_nuclei_cpu = sorted_nuclei.to_vec();
    let cpu_objective =
        move |x: &[f64]| -> f64 { l1_objective_nmp_nuclei(x, &sorted_nuclei_cpu, lambda) };

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

    let sorted_for_obj = sorted_nuclei_arc;
    let gpu_objective =
        move |x: &[f64]| -> f64 { l1_objective_nmp_nuclei(x, &sorted_for_obj, lambda) };

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

    L1DirectSamplerResult {
        cpu_full_chi2,
        gpu_full_chi2,
        gpu_arc,
        device,
    }
}

/// L1 chi2/datum via GPU dispatch.
pub fn l1_chi2_gpu(
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
