// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 45 overnight validation: GPU kinetic-fluid (BGK, Euler, coupled).

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

pub fn paper_45_gpu_bgk(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::gpu_kinetic_fluid::{GpuBgkPipeline, validate_gpu_bgk};

    println!("  GPU BGK relaxation...");
    let start = Instant::now();

    let pipeline = GpuBgkPipeline::new(gpu);
    let (gpu_r, _cpu_r) = validate_gpu_bgk(gpu, &pipeline);

    harness.check_upper("p45_bgk_mass_err", gpu_r.result.mass_err_1, 1e-4);
    harness.check_upper("p45_bgk_energy_err", gpu_r.result.energy_err, 0.05);
    harness.check_bool("p45_bgk_entropy", gpu_r.result.entropy_monotonic);
    telem.log_map(
        "p45_bgk",
        &[
            ("mass_err", gpu_r.result.mass_err_1),
            ("energy_err", gpu_r.result.energy_err),
            ("temp_relaxed", gpu_r.result.temp_relaxed),
            ("gpu_seconds", gpu_r.gpu_wall_seconds),
            ("cpu_seconds", gpu_r.cpu_wall_seconds),
        ],
    );
    println!(
        "    GPU {:.2}s, CPU {:.2}s",
        gpu_r.gpu_wall_seconds, gpu_r.cpu_wall_seconds
    );
    println!("    ΔT/T = {:.4}", gpu_r.result.temp_relaxed);
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

pub fn paper_45_gpu_euler(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::gpu_euler::{GpuEulerPipeline, validate_gpu_euler};

    println!("  GPU Euler / Sod shock tube...");
    let start = Instant::now();

    let pipeline = GpuEulerPipeline::new(gpu);
    let result = validate_gpu_euler(gpu, &pipeline);

    harness.check_upper("p45_euler_mass_err", result.mass_err, 0.01);
    harness.check_upper("p45_euler_energy_err", result.energy_err, 0.01);
    harness.check_upper("p45_euler_cpu_mass_err", result.cpu.mass_err, 0.01);

    let rho_range = result.rho.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - result.rho.iter().copied().fold(f64::INFINITY, f64::min);
    harness.check_lower("p45_euler_shock_resolved", rho_range, 0.5);

    telem.log_map(
        "p45_euler",
        &[
            ("mass_err", result.mass_err),
            ("energy_err", result.energy_err),
            ("rho_range", rho_range),
            ("gpu_seconds", result.gpu_wall_seconds),
            ("cpu_seconds", result.cpu_wall_seconds),
        ],
    );
    println!(
        "    GPU {:.2}s, CPU {:.2}s",
        result.gpu_wall_seconds, result.cpu_wall_seconds
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

pub fn paper_45_gpu_coupled(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::gpu_coupled_kinetic_fluid::{
        GpuCoupledPipeline, validate_gpu_coupled,
    };

    println!("  GPU coupled kinetic-fluid...");
    let start = Instant::now();

    let pipeline = GpuCoupledPipeline::new(gpu);
    let result = validate_gpu_coupled(gpu, &pipeline);

    harness.check_upper("p45_coupled_mass_err", result.mass_err, 0.05);
    harness.check_upper("p45_coupled_energy_err", result.energy_err, 0.1);
    // Interface density mismatch is inherent to half-space Maxwellian coupling
    // (kinetic cell sees both incoming+outgoing; fluid cell is Euler-updated).
    // Check that GPU mismatch agrees with CPU reference within 50% relative.
    let cpu_if = result.cpu.interface_density_match;
    let gpu_if = result.interface_density_match;
    let if_rel = if cpu_if > 1e-15 {
        (gpu_if - cpu_if).abs() / cpu_if
    } else {
        gpu_if
    };
    println!("    interface: GPU={gpu_if:.4e}, CPU={cpu_if:.4e}, rel={if_rel:.4e}");
    harness.check_upper("p45_coupled_interface_parity", if_rel, 0.5);
    telem.log_map(
        "p45_coupled",
        &[
            ("mass_err", result.mass_err),
            ("energy_err", result.energy_err),
            ("interface_gpu", gpu_if),
            ("interface_cpu", cpu_if),
            ("interface_rel", if_rel),
            ("n_steps", result.n_steps as f64),
            ("gpu_seconds", result.gpu_wall_seconds),
            ("cpu_seconds", result.cpu_wall_seconds),
        ],
    );
    println!(
        "    {} steps, GPU {:.2}s, CPU {:.2}s",
        result.n_steps, result.gpu_wall_seconds, result.cpu_wall_seconds
    );
    println!(
        "    CPU reference: mass_err={:.2e}, energy_err={:.2e}",
        result.cpu.mass_err, result.cpu.energy_err
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}
