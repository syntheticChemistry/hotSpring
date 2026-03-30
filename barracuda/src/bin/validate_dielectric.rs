// SPDX-License-Identifier: AGPL-3.0-only

//! Validation binary for BGK dielectric functions (Paper 44).
//!
//! Chuna & Murillo, Phys. Rev. E 111, 035206 (2024), arXiv:2405.07871

use hotspring_barracuda::physics::dielectric::{
    Complex, PlasmaParams, conductivity_dc, debye_screening, dynamic_structure_factor,
    dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
    f_sum_rule_integral, f_sum_rule_integral_completed, plasma_dispersion_w, validate_dielectric,
};
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    let mut harness = ValidationHarness::new("validate_dielectric");

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Paper 44: BGK Dielectric (Chuna & Murillo 2024)       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // ── Plasma dispersion function ──
    let w0 = plasma_dispersion_w(Complex::ZERO);
    harness.check_abs("W(0) real part = 1", w0.re, 1.0, 1e-14);
    harness.check_upper("W(0) imaginary part ≈ 0", w0.im.abs(), 1e-14);

    let w_large = plasma_dispersion_w(Complex::new(20.0, 0.0));
    harness.check_upper("|W(20)| → 0", w_large.abs(), 0.01);

    let test_cases: &[(f64, f64, &str)] = &[
        (1.0, 1.0, "weak"),
        (10.0, 1.0, "moderate"),
        (10.0, 2.0, "strong_screen"),
    ];

    for &(gamma, kappa, label) in test_cases {
        let params = PlasmaParams::from_coupling(gamma, kappa);
        let nu = 0.1 * params.omega_p;

        println!("  ── Γ={gamma}, κ={kappa} ({label}) ──");

        // Debye screening
        let (eps_s, eps_d) = debye_screening(1.0, &params);
        harness.check_rel(&format!("Debye_ε_{label}"), eps_s, eps_d, 1e-12);

        // DC conductivity
        let dc = conductivity_dc(nu, &params);
        let dc_exp = params.omega_p.powi(2) / (4.0 * std::f64::consts::PI * nu);
        harness.check_rel(&format!("Drude_σ_{label}"), dc, dc_exp, 1e-13);

        // High-frequency limit
        let eps_high = epsilon_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        harness.check_upper(
            &format!("ε→1_{label}"),
            (eps_high - Complex::ONE).abs(),
            0.01,
        );

        // f-sum rule sign
        let f_sum = f_sum_rule_integral(1.0, nu, &params, 200.0);
        harness.check_upper(&format!("f_sum_sign_{label}"), f_sum, 0.0);
        println!(
            "    f-sum = {f_sum:.4} (expected -πωₚ²/2 ≈ {:.4})",
            -std::f64::consts::PI * params.omega_p.powi(2) / 2.0
        );

        // DSF positivity
        let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();
        let s_kw = dynamic_structure_factor(1.0, &omegas, nu, &params);
        let s_max = s_kw.iter().copied().fold(0.0_f64, f64::max);
        let n_pos = s_kw
            .iter()
            .filter(|&&s| s >= -1e-6 * s_max.max(1e-10))
            .count();
        let frac = n_pos as f64 / s_kw.len() as f64;
        harness.check_lower(&format!("S(k,ω)≥0_{label}"), frac, 0.98);
        println!(
            "    S(k,ω) positive: {:.1}%, max = {s_max:.6}",
            frac * 100.0
        );
    }

    // ── Completed Mermin (Chuna & Murillo 2024, Eq. 26) ──
    println!("\n  ══ Completed Mermin (number + momentum conservation) ══");
    for &(gamma, kappa, label) in test_cases {
        let params = PlasmaParams::from_coupling(gamma, kappa);
        let nu = 0.1 * params.omega_p;
        let f_sum_expected = -std::f64::consts::PI * params.omega_p.powi(2) / 2.0;

        // Completed Mermin high-frequency limit
        let eps_cm = epsilon_completed_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        harness.check_upper(
            &format!("ε_CM→1_{label}"),
            (eps_cm - Complex::ONE).abs(),
            0.01,
        );

        // Completed f-sum
        let f_sum_cm = f_sum_rule_integral_completed(1.0, nu, &params, 200.0);
        let f_sum_std = f_sum_rule_integral(1.0, nu, &params, 200.0);
        let err_cm = (f_sum_cm - f_sum_expected).abs() / f_sum_expected.abs();
        let err_std = (f_sum_std - f_sum_expected).abs() / f_sum_expected.abs();
        harness.check_upper(&format!("f_sum_sign_CM_{label}"), f_sum_cm, 0.0);
        println!(
            "    {label}: f-sum err standard={err_std:.4e} completed={err_cm:.4e} (improvement: {:.1}×)",
            err_std / err_cm.max(1e-30)
        );

        // Completed DSF positivity
        let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();
        let s_cm = dynamic_structure_factor_completed(1.0, &omegas, nu, &params);
        let s_max_cm = s_cm.iter().copied().fold(0.0_f64, f64::max);
        let n_pos_cm = s_cm
            .iter()
            .filter(|&&s| s >= -1e-6 * s_max_cm.max(1e-10))
            .count();
        let frac_cm = n_pos_cm as f64 / s_cm.len() as f64;
        harness.check_lower(&format!("S_CM≥0_{label}"), frac_cm, 0.99);
        println!(
            "    S_CM(k,ω) positive: {:.1}%, max = {s_max_cm:.6}",
            frac_cm * 100.0
        );
    }

    // ── Full validation helper ──
    for &(gamma, kappa) in &[(1.0, 1.0), (10.0, 1.0), (10.0, 2.0)] {
        let r = validate_dielectric(gamma, kappa);
        let passed =
            r.debye_error < 1e-12 && r.f_sum_computed < 0.0 && r.high_freq_deviation < 0.01;
        harness.check_bool(&format!("full_Γ{gamma}_κ{kappa}"), passed);
        println!(
            "  Γ={gamma}, κ={kappa}: Debye err={:.2e}, f-sum std={:.4} cm={:.4}, ε∞ dev={:.4}",
            r.debye_error, r.f_sum_computed, r.f_sum_completed, r.high_freq_deviation,
        );
        println!(
            "    S≥0 std={:.1}% cm={:.1}%, f-sum err std={:.2e} cm={:.2e}",
            r.dsf_fraction_positive * 100.0,
            r.dsf_fraction_positive_completed * 100.0,
            r.f_sum_error_standard,
            r.f_sum_error_completed,
        );
    }

    harness.finish();
}
