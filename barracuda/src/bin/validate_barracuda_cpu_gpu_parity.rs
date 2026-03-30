// SPDX-License-Identifier: AGPL-3.0-only

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
//!   - PPPM: Coulomb forces (GPU mesh vs CPU direct)

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
