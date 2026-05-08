// SPDX-License-Identifier: AGPL-3.0-or-later

//! Systematic CPU vs GPU parity validation across all physics domains.
//!
//! Runs every physics pipeline through both CPU and GPU paths and reports
//! L2 norms. This is the "correct" leg of "correct, fast, cheap."
//!
//! Domains:
//!   - Lattice QCD: plaquette, gradient flow E(t)
//!   - Plasma dielectric: Mermin ε(k,ω), DSF S(k,ω)
//!   - Multi-component Mermin: electron-ion
//!   - Kinetic-fluid: BGK relaxation, Euler shock, coupled interface
//!   - Nuclear EOS: GPU SEMF binding energy vs CPU
//!   - Spectral: GPU SpMV vs CPU on Anderson 2D

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

fn main() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut harness = ValidationHarness::new("cpu_gpu_parity");
    let mut telem = TelemetryWriter::discover("cpu_gpu_parity_telemetry.jsonl");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  BarraCuda CPU vs GPU Parity — All Physics Domains         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let total_start = Instant::now();

    // ─── Domain 1: Lattice QCD (CPU only — GPU HMC parity via existing bins) ───
    println!("━━━ Domain 1: Lattice QCD ━━━\n");
    validate_lattice_qcd(&mut harness, &mut telem);

    // ─── GPU-dependent domains ───
    match rt.block_on(GpuF64::new()) {
        Ok(gpu) => {
            if gpu.has_f64 {
                println!("\n━━━ Domain 2: Plasma Dielectric ━━━\n");
                validate_dielectric(&mut harness, &gpu, &mut telem);

                println!("\n━━━ Domain 3: Multi-component Mermin ━━━\n");
                validate_multicomp(&mut harness, &gpu, &mut telem);

                println!("\n━━━ Domain 4: BGK Relaxation ━━━\n");
                validate_bgk(&mut harness, &gpu, &mut telem);

                println!("\n━━━ Domain 5: Euler HLL ━━━\n");
                validate_euler(&mut harness, &gpu, &mut telem);

                println!("\n━━━ Domain 6: Coupled Kinetic-Fluid ━━━\n");
                validate_coupled(&mut harness, &gpu, &mut telem);

                println!("\n━━━ Domain 7: Nuclear EOS (SEMF Batch) ━━━\n");
                validate_nuclear_eos(&mut harness, &gpu, &mut telem);

                println!("\n━━━ Domain 8: Spectral (GPU SpMV) ━━━\n");
                validate_spectral(&mut harness, &gpu, &mut telem);
            } else {
                println!("  SHADER_F64 not supported — skipping GPU domains\n");
            }
        }
        Err(e) => {
            println!("  GPU unavailable: {e}\n");
        }
    }

    let total = total_start.elapsed();
    telem.log("summary", "total_wall_seconds", total.as_secs_f64());
    println!("\n  Total wall time: {:.1}s", total.as_secs_f64());
    harness.finish();
}

fn validate_lattice_qcd(harness: &mut ValidationHarness, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::lattice::gradient_flow::{FlowIntegrator, run_flow};
    use hotspring_barracuda::lattice::hmc::{HmcConfig, hmc_trajectory};
    use hotspring_barracuda::lattice::wilson::Lattice;

    let start = Instant::now();
    println!("  8⁴ β=6.0 quenched HMC + gradient flow...");

    let mut lat = Lattice::hot_start([8, 8, 8, 8], 6.0, 42);
    let mut cfg = HmcConfig {
        n_md_steps: 20,
        dt: 0.05,
        seed: 12345,
        ..Default::default()
    };

    let mut n_accept = 0;
    for _ in 0..50 {
        if hmc_trajectory(&mut lat, &mut cfg).accepted {
            n_accept += 1;
        }
    }
    let plaq = lat.average_plaquette();
    let acceptance = n_accept as f64 / 50.0;

    println!("    ⟨P⟩ = {plaq:.6}, {:.0}% accept", acceptance * 100.0);
    harness.check_lower("qcd_plaquette", plaq, 0.5);
    harness.check_lower("qcd_acceptance", acceptance, 0.25);

    let flow = run_flow(&mut lat, FlowIntegrator::Lscfrk3w7, 0.01, 2.0, 5);
    let monotonic = flow
        .windows(2)
        .all(|w| w[1].energy_density <= w[0].energy_density + 1e-10);
    harness.check_bool("qcd_flow_monotonic", monotonic);

    telem.log_map(
        "qcd",
        &[
            ("plaquette", plaq),
            ("acceptance", acceptance),
            ("flow_points", flow.len() as f64),
            ("wall_seconds", start.elapsed().as_secs_f64()),
        ],
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn validate_dielectric(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_dielectric::{
        GpuDielectricPipeline, validate_gpu_dielectric,
    };

    let start = Instant::now();
    println!("  Standard + completed Mermin (Γ=10, κ=2)...");

    let pipeline = GpuDielectricPipeline::new(gpu);
    let v = validate_gpu_dielectric(gpu, &pipeline, 10.0, 2.0);

    let l2 = v.l2_loss_rel_error;
    println!("    GPU-CPU L² (loss function): {l2:.4e}");
    println!(
        "    DSF positivity (GPU): {:.0}%",
        v.dsf_pos_fraction_gpu * 100.0
    );
    println!(
        "    GPU {:.2}s, CPU {:.2}s",
        v.gpu_wall_seconds, v.cpu_wall_seconds
    );

    harness.check_upper("dielectric_l2", l2, 0.01);
    harness.check_lower("dielectric_dsf_pos", v.dsf_pos_fraction_gpu, 0.95);

    telem.log_map(
        "dielectric",
        &[
            ("l2_rel", l2),
            ("dsf_pos_frac", v.dsf_pos_fraction_gpu),
            ("gpu_seconds", v.gpu_wall_seconds),
            ("cpu_seconds", v.cpu_wall_seconds),
        ],
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn validate_multicomp(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_dielectric_multicomponent::{
        GpuMulticompPipeline, validate_gpu_multicomponent,
    };

    let start = Instant::now();
    println!("  Electron-ion multi-component Mermin...");

    let pipeline = GpuMulticompPipeline::new(gpu);
    let (gpu_loss, cpu_loss) = validate_gpu_multicomponent(gpu, &pipeline);

    let n_close = gpu_loss
        .iter()
        .zip(cpu_loss.iter())
        .filter(|&(&g, &c)| {
            let denom = c.abs().max(1e-15);
            (g - c).abs() / denom < 0.5
        })
        .count();
    let frac = n_close as f64 / gpu_loss.len().max(1) as f64;

    println!("    CPU-GPU agreement: {:.0}%", frac * 100.0);
    harness.check_lower("multicomp_agreement", frac, 0.90);
    telem.log("multicomp", "agreement_frac", frac);
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn validate_bgk(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_kinetic_fluid::{GpuBgkPipeline, validate_gpu_bgk};

    let start = Instant::now();
    println!("  BGK relaxation (GPU vs CPU)...");

    let pipeline = GpuBgkPipeline::new(gpu);
    let (gpu_r, cpu_r) = validate_gpu_bgk(gpu, &pipeline);

    let mass_diff = (gpu_r.result.mass_err_1 - cpu_r.mass_err_1).abs();
    let energy_diff = (gpu_r.result.energy_err - cpu_r.energy_err).abs();

    println!(
        "    GPU mass err: {:.4e}, CPU: {:.4e}, diff: {:.4e}",
        gpu_r.result.mass_err_1, cpu_r.mass_err_1, mass_diff
    );
    println!(
        "    GPU energy err: {:.4e}, CPU: {:.4e}, diff: {:.4e}",
        gpu_r.result.energy_err, cpu_r.energy_err, energy_diff
    );

    harness.check_upper("bgk_mass_diff", mass_diff, 1e-3);
    harness.check_upper("bgk_energy_diff", energy_diff, 0.1);
    harness.check_bool("bgk_gpu_entropy", gpu_r.result.entropy_monotonic);
    harness.check_bool("bgk_cpu_entropy", cpu_r.entropy_monotonic);

    let speedup = if gpu_r.gpu_wall_seconds > 1e-10 {
        gpu_r.cpu_wall_seconds / gpu_r.gpu_wall_seconds
    } else {
        0.0
    };
    telem.log_map(
        "bgk",
        &[
            ("mass_diff", mass_diff),
            ("energy_diff", energy_diff),
            ("gpu_seconds", gpu_r.gpu_wall_seconds),
            ("cpu_seconds", gpu_r.cpu_wall_seconds),
            ("speedup", speedup),
        ],
    );
    println!("    Speedup: {speedup:.1}×");
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn validate_euler(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_euler::{GpuEulerPipeline, validate_gpu_euler};

    let start = Instant::now();
    println!("  Euler/Sod shock tube (GPU vs CPU)...");

    let pipeline = GpuEulerPipeline::new(gpu);
    let r = validate_gpu_euler(gpu, &pipeline);

    let mass_diff = (r.mass_err - r.cpu.mass_err).abs();
    let energy_diff = (r.energy_err - r.cpu.energy_err).abs();

    println!(
        "    GPU mass err: {:.4e}, CPU: {:.4e}",
        r.mass_err, r.cpu.mass_err
    );
    println!(
        "    GPU energy err: {:.4e}, CPU: {:.4e}",
        r.energy_err, r.cpu.energy_err
    );

    harness.check_upper("euler_mass_diff", mass_diff, 0.01);
    harness.check_upper("euler_energy_diff", energy_diff, 0.01);

    let rho_range = r.rho.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - r.rho.iter().copied().fold(f64::INFINITY, f64::min);
    harness.check_lower("euler_shock_resolved", rho_range, 0.5);

    let speedup = if r.gpu_wall_seconds > 1e-10 {
        r.cpu_wall_seconds / r.gpu_wall_seconds
    } else {
        0.0
    };
    telem.log_map(
        "euler",
        &[
            ("mass_diff", mass_diff),
            ("energy_diff", energy_diff),
            ("rho_range", rho_range),
            ("speedup", speedup),
        ],
    );
    println!("    Speedup: {speedup:.1}×");
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn validate_nuclear_eos(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::semf_binding_energy;
    use hotspring_barracuda::provenance::SLY4_PARAMS;

    let start = Instant::now();
    println!("  GPU SEMF batch vs CPU SEMF (SLy4, 20 nuclei)...");

    let test_nuclei: Vec<(usize, usize)> = vec![
        (8, 8),     // O-16
        (20, 20),   // Ca-40
        (26, 30),   // Fe-56
        (28, 30),   // Ni-58
        (50, 70),   // Sn-120
        (82, 126),  // Pb-208
        (92, 146),  // U-238
        (6, 6),     // C-12
        (1, 0),     // H-1
        (2, 2),     // He-4
        (14, 14),   // Si-28
        (29, 34),   // Cu-63
        (47, 60),   // Ag-107
        (79, 118),  // Au-197
        (3, 4),     // Li-7
        (11, 12),   // Na-23
        (19, 20),   // K-39
        (30, 34),   // Zn-64
        (48, 66),   // Cd-114
        (56, 82),   // Ba-138
    ];

    let cpu_energies: Vec<f64> = test_nuclei
        .iter()
        .map(|&(z, n)| semf_binding_energy(z, n, &SLY4_PARAMS))
        .collect();

    let nuclei_flat: Vec<f64> = test_nuclei
        .iter()
        .flat_map(|&(z, n)| {
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

    let shader_src = include_str!("../physics/shaders/semf_batch_f64.wgsl");
    let pipeline = gpu.create_pipeline(shader_src, "semf_parity");

    let n = test_nuclei.len();

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SemfParams {
        n_nuclei: u32,
        _pad: [u32; 3],
        params: [f64; 8],
    }

    let mut params_arr = [0.0f64; 8];
    for (i, &p) in SLY4_PARAMS.iter().enumerate().take(8) {
        params_arr[i] = p;
    }

    let params = SemfParams {
        n_nuclei: n as u32,
        _pad: [0; 3],
        params: params_arr,
    };

    let params_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "semf_params");
    let nuclei_buf = gpu.create_f64_buffer(&nuclei_flat, "nuclei");
    let output_buf = gpu.create_f64_output_buffer(n, "semf_out");

    let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("semf_parity_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: nuclei_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n as u32 + 63) / 64;
    let gpu_energies = gpu
        .dispatch_and_read(&pipeline, &bind_group, workgroups, &output_buf, n)
        .expect("GPU SEMF dispatch");

    let mut max_rel = 0.0f64;
    let mut sum_rel = 0.0f64;
    for (i, (&cpu_e, &gpu_e)) in cpu_energies.iter().zip(gpu_energies.iter()).enumerate() {
        let rel: f64 = if cpu_e.abs() > 1e-10 {
            (cpu_e - gpu_e).abs() / cpu_e.abs()
        } else {
            (cpu_e - gpu_e).abs()
        };
        if i < 3 {
            let (z, n) = test_nuclei[i];
            println!("    Z={z} N={n}: CPU={cpu_e:.4} GPU={gpu_e:.4} rel={rel:.2e}");
        }
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    let mean_rel = sum_rel / n as f64;

    println!("    Max relative error: {max_rel:.4e}");
    println!("    Mean relative error: {mean_rel:.4e}");

    harness.check_upper("nuclear_eos_max_rel", max_rel, 1e-10);
    harness.check_upper("nuclear_eos_mean_rel", mean_rel, 1e-10);

    telem.log_map(
        "nuclear_eos",
        &[
            ("max_rel", max_rel),
            ("mean_rel", mean_rel),
            ("n_nuclei", n as f64),
            ("wall_seconds", start.elapsed().as_secs_f64()),
        ],
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn validate_spectral(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::spectral::{WGSL_SPMV_CSR_F64, anderson_2d};

    let start = Instant::now();
    println!("  GPU SpMV vs CPU SpMV (Anderson 2D, L=8)...");

    let matrix = anderson_2d(8, 8, 0.0, 42);
    let n = matrix.n;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / n as f64).collect();
    let mut cpu_result = vec![0.0; n];
    matrix.spmv(&x, &mut cpu_result);

    let pipeline = gpu.create_pipeline_f64(WGSL_SPMV_CSR_F64, "spmv_parity");

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct SpMVParams {
        n: u32,
        nnz: u32,
        pad0: u32,
        pad1: u32,
    }

    let nnz = matrix.values.len();
    let params = SpMVParams {
        n: n as u32,
        nnz: nnz as u32,
        pad0: 0,
        pad1: 0,
    };

    let row_ptr_u32: Vec<u32> = matrix.row_ptr.iter().map(|&v| v as u32).collect();
    let col_idx_u32: Vec<u32> = matrix.col_idx.iter().map(|&v| v as u32).collect();

    let params_buf = gpu.create_uniform_buffer(bytemuck::bytes_of(&params), "spmv_params");
    let row_ptr_buf = gpu.create_u32_buffer(&row_ptr_u32, "row_ptr");
    let col_idx_buf = gpu.create_u32_buffer(&col_idx_u32, "col_idx");
    let values_buf = gpu.create_f64_buffer(&matrix.values, "values");
    let x_buf = gpu.create_f64_buffer(&x, "x_vec");
    let output_buf = gpu.create_f64_output_buffer(n, "spmv_out");

    let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("spmv_parity_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: row_ptr_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: col_idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: values_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n as u32 + 63) / 64;
    let gpu_result = gpu
        .dispatch_and_read(&pipeline, &bind_group, workgroups, &output_buf, n)
        .expect("GPU SpMV dispatch");

    let mut max_err = 0.0f64;
    let mut l2_sum = 0.0f64;
    for (&cpu_v, &gpu_v) in cpu_result.iter().zip(gpu_result.iter()) {
        let err = (cpu_v - gpu_v).abs();
        max_err = max_err.max(err);
        l2_sum += err * err;
    }
    let l2 = l2_sum.sqrt();

    println!("    Max absolute error: {max_err:.4e}");
    println!("    L2 norm error: {l2:.4e}");
    println!("    Matrix: {}×{} ({} nnz)", n, n, nnz);

    harness.check_upper("spectral_spmv_max_err", max_err, 1e-12);
    harness.check_upper("spectral_spmv_l2", l2, 1e-10);

    telem.log_map(
        "spectral",
        &[
            ("max_err", max_err),
            ("l2_norm", l2),
            ("matrix_n", n as f64),
            ("nnz", nnz as f64),
            ("wall_seconds", start.elapsed().as_secs_f64()),
        ],
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

fn validate_coupled(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_coupled_kinetic_fluid::{
        GpuCoupledPipeline, validate_gpu_coupled,
    };

    let start = Instant::now();
    println!("  Coupled kinetic-fluid (GPU vs CPU)...");

    let pipeline = GpuCoupledPipeline::new(gpu);
    let r = validate_gpu_coupled(gpu, &pipeline);

    let if_rel = if r.cpu.interface_density_match > 1e-15 {
        (r.interface_density_match - r.cpu.interface_density_match).abs()
            / r.cpu.interface_density_match
    } else {
        r.interface_density_match
    };

    println!(
        "    GPU mass err: {:.4e}, CPU: {:.4e}",
        r.mass_err, r.cpu.mass_err
    );
    println!("    Interface parity: {if_rel:.4e}");
    println!(
        "    {} steps, GPU {:.2}s, CPU {:.2}s",
        r.n_steps, r.gpu_wall_seconds, r.cpu_wall_seconds
    );

    harness.check_upper("coupled_mass_err", r.mass_err, 0.05);
    harness.check_upper("coupled_energy_err", r.energy_err, 0.1);
    harness.check_upper("coupled_interface_parity", if_rel, 0.5);

    let speedup = if r.gpu_wall_seconds > 1e-10 {
        r.cpu_wall_seconds / r.gpu_wall_seconds
    } else {
        0.0
    };
    telem.log_map(
        "coupled",
        &[
            ("mass_err", r.mass_err),
            ("energy_err", r.energy_err),
            ("interface_rel", if_rel),
            ("speedup", speedup),
        ],
    );
    println!("    Speedup: {speedup:.1}×");
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}
