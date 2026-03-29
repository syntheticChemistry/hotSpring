// SPDX-License-Identifier: AGPL-3.0-only

//! Paper 44/45 validation helpers for `validate_chuna_overnight`.
//!
//! Extracted to keep the main binary under 1000 LOC.

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

// ─── Paper 44: Conservative BGK Dielectric ─────────────────────

pub fn paper_44_cpu(harness: &mut ValidationHarness, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::dielectric::{
        dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
        f_sum_rule_integral, f_sum_rule_integral_completed, PlasmaParams,
    };

    println!("  Single-species Mermin (CPU)...");
    let start = Instant::now();

    let params = PlasmaParams::from_coupling(10.0, 2.0);
    let k = 1.0;
    let nu = 0.5;

    let expected = -std::f64::consts::PI * params.omega_p * params.omega_p / 2.0;
    let f_25 = f_sum_rule_integral(k, nu, &params, 25.0);
    let f_50 = f_sum_rule_integral(k, nu, &params, 50.0);
    let f_100 = f_sum_rule_integral(k, nu, &params, 100.0);
    let err_25 = (f_25 - expected).abs();
    let err_50 = (f_50 - expected).abs();
    let err_100 = (f_100 - expected).abs();
    let converging = err_100 <= err_50 && err_50 <= err_25;
    let same_sign = f_100.signum() == expected.signum();
    println!(
        "    f-sum convergence: err@25={:.4e}, @50={:.4e}, @100={:.4e}, sign={}",
        err_25 / expected.abs(),
        err_50 / expected.abs(),
        err_100 / expected.abs(),
        if same_sign { "OK" } else { "WRONG" }
    );
    telem.log_map(
        "p44_fsum",
        &[
            ("err_25", err_25 / expected.abs()),
            ("err_50", err_50 / expected.abs()),
            ("err_100", err_100 / expected.abs()),
        ],
    );
    harness.check_abs("p44_fsum_converging", f64::from(converging), 1.0, 0.5);
    harness.check_abs("p44_fsum_sign", f64::from(same_sign), 1.0, 0.5);

    let fc_25 = f_sum_rule_integral_completed(k, nu, &params, 25.0);
    let fc_100 = f_sum_rule_integral_completed(k, nu, &params, 100.0);
    let fc_converging = (fc_100 - expected).abs() <= (fc_25 - expected).abs();
    harness.check_abs(
        "p44_fsum_completed_conv",
        f64::from(fc_converging),
        1.0,
        0.5,
    );

    let omegas: Vec<f64> = (1..200).map(|i| 0.1 * i as f64).collect();
    let dsf = dynamic_structure_factor_completed(k, &omegas, nu, &params);
    let n_pos = dsf.iter().filter(|&&s| s >= -1e-15).count();
    let frac = n_pos as f64 / dsf.len() as f64;
    telem.log("p44_dsf", "positive_fraction", frac);
    harness.check_lower("p44_dsf_positive", frac, 0.99);

    let eps_hf = epsilon_completed_mermin(k, 100.0, nu, &params);
    harness.check_upper("p44_hf_limit", (eps_hf.re - 1.0).abs(), 0.01);

    let eps_std = epsilon_mermin(k, 1.5, 1e-10, &params);
    let eps_cmp = epsilon_completed_mermin(k, 1.5, 1e-10, &params);
    let rel = (eps_std.re - eps_cmp.re).abs() / eps_std.abs().max(1e-15);
    harness.check_upper("p44_nu0_agreement", rel, 0.01);

    telem.log("p44_cpu", "wall_seconds", start.elapsed().as_secs_f64());
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

pub fn paper_44_multicomponent_cpu(harness: &mut ValidationHarness, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::dielectric_multicomponent::{
        epsilon_multicomponent_mermin, multicomponent_dsf, multicomponent_f_sum_integral,
        MultiComponentPlasma, SpeciesParams,
    };

    println!("  Multi-component Mermin (CPU)...");
    let start = Instant::now();
    let plasma = MultiComponentPlasma {
        species: vec![
            SpeciesParams {
                mass: 1.0 / 1836.0,
                charge: 1.0,
                density: 1.0,
                temperature: 1.0,
                nu: 0.1,
            },
            SpeciesParams {
                mass: 1.0,
                charge: 1.0,
                density: 1.0,
                temperature: 1.0,
                nu: 0.01,
            },
        ],
    };
    let k = 1.0;

    let eps_static = epsilon_multicomponent_mermin(k, 0.0, &plasma, true);
    let k_d_sq = plasma.total_k_debye_sq();
    let expected = 1.0 + k_d_sq / (k * k);
    let rel = (eps_static.re - expected).abs() / expected;
    println!("    Debye screening: rel = {rel:.4e}");
    harness.check_upper("p44_mc_debye", rel, 0.01);

    let eps_hf = epsilon_multicomponent_mermin(k, 10_000.0, &plasma, true);
    harness.check_upper("p44_mc_hf_limit", (eps_hf.re - 1.0).abs(), 0.01);

    let omegas: Vec<f64> = (1..100).map(|i| 0.1 * i as f64).collect();
    let dsf = multicomponent_dsf(k, &omegas, &plasma);
    let n_pos = dsf.iter().filter(|&&s| s >= 0.0).count();
    let frac = n_pos as f64 / dsf.len() as f64;
    harness.check_lower("p44_mc_dsf_positive", frac, 0.95);

    let total_wp2 = plasma.total_omega_p_sq();
    let expected_fsum = -std::f64::consts::PI * total_wp2 / 2.0;
    let f_50 = multicomponent_f_sum_integral(k, &plasma, 50.0);
    let f_200 = multicomponent_f_sum_integral(k, &plasma, 200.0);
    let converging = (f_200 - expected_fsum).abs() <= (f_50 - expected_fsum).abs();
    let same_sign = f_200.signum() == expected_fsum.signum();
    println!("    f-sum: @50={f_50:.4e}, @200={f_200:.4e}, expected={expected_fsum:.4e}");
    harness.check_abs("p44_mc_fsum_conv", f64::from(converging), 1.0, 0.5);
    harness.check_abs("p44_mc_fsum_sign", f64::from(same_sign), 1.0, 0.5);

    for omega in [0.1, 0.5, 1.0, 5.0, 10.0] {
        let eps = epsilon_multicomponent_mermin(k, omega, &plasma, true);
        harness.check_lower(&format!("p44_mc_passive_w{omega}"), eps.im, -0.01);
    }
    telem.log("p44_mc_cpu", "wall_seconds", start.elapsed().as_secs_f64());
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

pub fn paper_44_gpu(harness: &mut ValidationHarness, gpu: &GpuF64, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::gpu_dielectric::{
        validate_gpu_dielectric, GpuDielectricPipeline,
    };

    println!("  GPU Mermin (standard + completed)...");
    let start = Instant::now();
    let pipeline = GpuDielectricPipeline::new(gpu);
    let validation = validate_gpu_dielectric(gpu, &pipeline, 10.0, 2.0);

    let expected_fsum = -std::f64::consts::PI
        * hotspring_barracuda::physics::dielectric::PlasmaParams::from_coupling(10.0, 2.0)
            .omega_p
            .powi(2)
        / 2.0;
    let gpu_sign_ok = validation.f_sum_gpu.signum() == expected_fsum.signum();
    harness.check_abs("p44_gpu_fsum_sign", f64::from(gpu_sign_ok), 1.0, 0.5);
    harness.check_lower("p44_gpu_dsf_pos", validation.dsf_pos_fraction_gpu, 0.95);
    harness.check_upper("p44_gpu_loss_l2", validation.l2_loss_rel_error, 0.01);

    telem.log_map(
        "p44_gpu",
        &[
            ("gpu_seconds", validation.gpu_wall_seconds),
            ("cpu_seconds", validation.cpu_wall_seconds),
            ("l2_loss_rel", validation.l2_loss_rel_error),
        ],
    );
    println!(
        "    GPU {:.2}s, CPU {:.2}s, L² = {:.4e}",
        validation.gpu_wall_seconds, validation.cpu_wall_seconds, validation.l2_loss_rel_error
    );
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

pub fn paper_44_multicomponent_gpu(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::gpu_dielectric_multicomponent::{
        validate_gpu_multicomponent, GpuMulticompPipeline,
    };

    println!("  GPU Multi-component Mermin...");
    let start = Instant::now();
    let pipeline = GpuMulticompPipeline::new(gpu);
    let (gpu_loss, cpu_loss) = validate_gpu_multicomponent(gpu, &pipeline);

    let n_close = gpu_loss
        .iter()
        .zip(cpu_loss.iter())
        .filter(|(&g, &c)| (g - c).abs() / c.abs().max(1e-15) < 0.5)
        .count();
    let frac = n_close as f64 / gpu_loss.len().max(1) as f64;
    println!("    CPU-GPU agreement: {:.0}%", frac * 100.0);
    telem.log("p44_mc_gpu", "cpu_gpu_agreement", frac);
    harness.check_lower("p44_mc_gpu_agreement", frac, 0.90);
    println!("    {:.1}s", start.elapsed().as_secs_f64());
}

// ─── Paper 45: Multi-Species Kinetic-Fluid Coupling ────────────

pub fn paper_45_gpu_bgk(
    harness: &mut ValidationHarness,
    gpu: &GpuF64,
    telem: &mut TelemetryWriter,
) {
    use hotspring_barracuda::physics::gpu_kinetic_fluid::{validate_gpu_bgk, GpuBgkPipeline};

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
    use hotspring_barracuda::physics::gpu_euler::{validate_gpu_euler, GpuEulerPipeline};

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
        validate_gpu_coupled, GpuCoupledPipeline,
    };

    println!("  GPU coupled kinetic-fluid...");
    let start = Instant::now();
    let pipeline = GpuCoupledPipeline::new(gpu);
    let result = validate_gpu_coupled(gpu, &pipeline);

    harness.check_upper("p45_coupled_mass_err", result.mass_err, 0.05);
    harness.check_upper("p45_coupled_energy_err", result.energy_err, 0.1);
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
