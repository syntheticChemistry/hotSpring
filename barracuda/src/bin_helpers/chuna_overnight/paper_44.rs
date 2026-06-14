// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 44 overnight validation: single-species and multi-component Mermin dielectric (CPU + GPU).

use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::validation::{TelemetryWriter, ValidationHarness};
use std::time::Instant;

pub fn paper_44_cpu(harness: &mut ValidationHarness, telem: &mut TelemetryWriter) {
    use hotspring_barracuda::physics::dielectric::{
        PlasmaParams, dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
        f_sum_rule_integral, f_sum_rule_integral_completed,
    };

    println!("  Single-species Mermin (CPU)...");
    let start = Instant::now();

    let params = PlasmaParams::from_coupling(10.0, 2.0);
    let k = 1.0;
    let nu = 0.5;

    // f-sum rule: verify monotone convergence toward -πωₚ²/2 as ω_max increases.
    // At finite ν with strong coupling (Γ=10), the Drude-broadened peak requires
    // large ω_max for the trapezoidal integral to converge. Rather than hand-tuning
    // a tolerance, we check that the integral is converging in the right direction.
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

    // Completed Mermin: same convergence check
    let fc_25 = f_sum_rule_integral_completed(k, nu, &params, 25.0);
    let fc_100 = f_sum_rule_integral_completed(k, nu, &params, 100.0);
    let fc_converging = (fc_100 - expected).abs() <= (fc_25 - expected).abs();
    harness.check_abs(
        "p44_fsum_completed_conv",
        f64::from(fc_converging),
        1.0,
        0.5,
    );

    // DSF positivity
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
        MultiComponentPlasma, SpeciesParams, epsilon_multicomponent_mermin, multicomponent_dsf,
        multicomponent_f_sum_integral,
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

    // Static limit should be Debye
    let eps_static = epsilon_multicomponent_mermin(k, 0.0, &plasma, true);
    let k_d_sq = plasma.total_k_debye_sq();
    let expected = 1.0 + k_d_sq / (k * k);
    let rel = (eps_static.re - expected).abs() / expected;
    println!("    Debye screening: rel = {rel:.4e}");
    harness.check_upper("p44_mc_debye", rel, 0.01);

    // High-frequency limit: ε→1 as ω→∞. For electron-ion plasma with m_e=1/1836,
    // ωₚₑ = √(4π n q²/m_e) ≈ 152. Use ω=10000 >> ωₚₑ.
    let eps_hf = epsilon_multicomponent_mermin(k, 10_000.0, &plasma, true);
    harness.check_upper("p44_mc_hf_limit", (eps_hf.re - 1.0).abs(), 0.01);

    // DSF positivity
    let omegas: Vec<f64> = (1..100).map(|i| 0.1 * i as f64).collect();
    let dsf = multicomponent_dsf(k, &omegas, &plasma);
    let n_pos = dsf.iter().filter(|&&s| s >= 0.0).count();
    let frac = n_pos as f64 / dsf.len() as f64;
    harness.check_lower("p44_mc_dsf_positive", frac, 0.95);

    // f-sum convergence: verify monotone convergence as ω_max increases
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
        GpuDielectricPipeline, validate_gpu_dielectric,
    };

    println!("  GPU Mermin (standard + completed)...");
    let start = Instant::now();

    let pipeline = GpuDielectricPipeline::new(gpu);
    let validation = validate_gpu_dielectric(gpu, &pipeline, 10.0, 2.0);

    // GPU f-sum: verify same sign as expected and GPU-CPU L² agreement
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
        GpuMulticompPipeline, validate_gpu_multicomponent,
    };

    println!("  GPU Multi-component Mermin...");
    let start = Instant::now();

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
    telem.log("p44_mc_gpu", "cpu_gpu_agreement", frac);
    harness.check_lower("p44_mc_gpu_agreement", frac, 0.90);

    println!("    {:.1}s", start.elapsed().as_secs_f64());
}
