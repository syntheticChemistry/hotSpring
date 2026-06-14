// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper 44: BGK dielectric validation.

use hotspring_barracuda::physics::dielectric::{
    Complex, PlasmaParams, conductivity_dc, debye_screening, dynamic_structure_factor,
    dynamic_structure_factor_completed, epsilon_completed_mermin, epsilon_mermin,
    f_sum_rule_integral, f_sum_rule_integral_completed, plasma_dispersion_w, validate_dielectric,
};
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;
use std::time::Instant;

pub fn paper_44_dielectric(harness: &mut ValidationHarness) {
    const PAPER: &str = "Chuna & Murillo, Phys. Rev. E 111, 035206 (2024)";
    const DOMAIN: &str = "plasma_dielectric";

    println!("━━━ Paper 44: BGK Dielectric ━━━");
    let t0 = Instant::now();

    // Plasma dispersion function
    let w0 = plasma_dispersion_w(Complex::ZERO);
    harness.check_abs(
        "dielectric_W0_real",
        w0.re,
        1.0,
        tolerances::DIELECTRIC_PLASMA_DISPERSION_W0_ABS,
    );
    harness.annotate(
        DOMAIN,
        PAPER,
        "dimensionless",
        "W(0)=1 exact from series definition",
    );
    harness.check_upper(
        "dielectric_W0_imag",
        w0.im.abs(),
        tolerances::DIELECTRIC_PLASMA_DISPERSION_W0_ABS,
    );
    harness.annotate(DOMAIN, PAPER, "dimensionless", "Im[W(0)]=0 exact");

    let w_large = plasma_dispersion_w(Complex::new(20.0, 0.0));
    harness.check_upper(
        "dielectric_W_large_arg",
        w_large.abs(),
        tolerances::DIELECTRIC_HIGH_FREQ_LIMIT_ABS,
    );
    harness.annotate(DOMAIN, PAPER, "dimensionless", "W(z)→0 for |z|→∞");

    let test_cases: &[(f64, f64, &str)] = &[
        (1.0, 1.0, "weak"),
        (10.0, 1.0, "moderate"),
        (10.0, 2.0, "strong_screen"),
    ];

    for &(gamma, kappa, label) in test_cases {
        let params = PlasmaParams::from_coupling(gamma, kappa);
        let nu = 0.1 * params.omega_p;

        // Debye screening
        let (eps_s, eps_d) = debye_screening(1.0, &params);
        harness.check_rel(
            &format!("dielectric_debye_{label}"),
            eps_s,
            eps_d,
            tolerances::DIELECTRIC_DEBYE_SCREENING_REL,
        );
        harness.annotate(
            DOMAIN,
            PAPER,
            "relative_error",
            "static limit matches Debye screening",
        );

        // DC conductivity (Drude)
        let dc = conductivity_dc(nu, &params);
        let dc_exp = params.omega_p.powi(2) / (4.0 * std::f64::consts::PI * nu);
        harness.check_rel(
            &format!("dielectric_drude_{label}"),
            dc,
            dc_exp,
            tolerances::DIELECTRIC_DRUDE_CONDUCTIVITY_REL,
        );
        harness.annotate(DOMAIN, PAPER, "conductivity", "Drude σ = ωₚ²/(4πν) exact");

        // High-frequency limit
        let eps_high = epsilon_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        harness.check_upper(
            &format!("dielectric_eps_inf_{label}"),
            (eps_high - Complex::ONE).abs(),
            tolerances::DIELECTRIC_HIGH_FREQ_LIMIT_ABS,
        );
        harness.annotate(DOMAIN, PAPER, "dielectric_function", "ε(ω→∞)→1");

        // f-sum rule sign
        let f_sum = f_sum_rule_integral(1.0, nu, &params, 200.0);
        harness.check_upper(&format!("dielectric_fsum_{label}"), f_sum, 0.0);
        harness.annotate(
            DOMAIN,
            PAPER,
            "sum_rule",
            "f-sum integral is negative (converging to -πωₚ²/2)",
        );

        // DSF positivity
        let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();
        let s_kw = dynamic_structure_factor(1.0, &omegas, nu, &params);
        let s_max = s_kw.iter().copied().fold(0.0_f64, f64::max);
        let n_pos = s_kw
            .iter()
            .filter(|&&s| {
                s >= -tolerances::DIELECTRIC_DSF_RELATIVE_NOISE_FLOOR
                    * s_max.max(tolerances::DIELECTRIC_DSF_MAGNITUDE_FLOOR)
            })
            .count();
        let frac = n_pos as f64 / s_kw.len() as f64;
        harness.check_lower(
            &format!("dielectric_dsf_pos_{label}"),
            frac,
            tolerances::DIELECTRIC_DSF_POSITIVE_FRACTION_MIN,
        );
        harness.annotate(
            DOMAIN,
            PAPER,
            "fraction",
            "S(k,ω)≥0 for ≥98% of ω (physical positivity)",
        );
    }

    // Completed Mermin checks
    for &(gamma, kappa, label) in test_cases {
        let params = PlasmaParams::from_coupling(gamma, kappa);
        let nu = 0.1 * params.omega_p;

        let eps_cm = epsilon_completed_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        harness.check_upper(
            &format!("dielectric_cm_inf_{label}"),
            (eps_cm - Complex::ONE).abs(),
            tolerances::DIELECTRIC_HIGH_FREQ_LIMIT_ABS,
        );
        harness.annotate(
            DOMAIN,
            PAPER,
            "dielectric_function",
            "completed Mermin ε(ω→∞)→1",
        );

        let f_sum_cm = f_sum_rule_integral_completed(1.0, nu, &params, 200.0);
        harness.check_upper(&format!("dielectric_cm_fsum_{label}"), f_sum_cm, 0.0);
        harness.annotate(DOMAIN, PAPER, "sum_rule", "completed Mermin f-sum negative");

        let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();
        let s_cm = dynamic_structure_factor_completed(1.0, &omegas, nu, &params);
        let s_max_cm = s_cm.iter().copied().fold(0.0_f64, f64::max);
        let n_pos_cm = s_cm
            .iter()
            .filter(|&&s| {
                s >= -tolerances::DIELECTRIC_DSF_RELATIVE_NOISE_FLOOR
                    * s_max_cm.max(tolerances::DIELECTRIC_DSF_MAGNITUDE_FLOOR)
            })
            .count();
        let frac_cm = n_pos_cm as f64 / s_cm.len() as f64;
        harness.check_lower(
            &format!("dielectric_cm_dsf_{label}"),
            frac_cm,
            tolerances::DIELECTRIC_COMPLETED_DSF_POSITIVE_FRACTION_MIN,
        );
        harness.annotate(
            DOMAIN,
            PAPER,
            "fraction",
            "completed Mermin S(k,ω)≥0 for ≥99%",
        );
    }

    // Full validation helper
    for &(gamma, kappa) in &[(1.0, 1.0), (10.0, 1.0), (10.0, 2.0)] {
        let r = validate_dielectric(gamma, kappa);
        let passed = r.debye_error < tolerances::DIELECTRIC_DEBYE_SCREENING_REL
            && r.f_sum_computed < 0.0
            && r.high_freq_deviation < tolerances::DIELECTRIC_HIGH_FREQ_LIMIT_ABS;
        harness.check_bool(&format!("dielectric_full_G{gamma}_k{kappa}"), passed);
        harness.annotate(
            DOMAIN,
            PAPER,
            "composite",
            "full validation (Debye + f-sum + ε∞)",
        );
    }

    let dur = t0.elapsed().as_millis() as u64;
    println!("  Paper 44: checks complete ({dur}ms)\n");
}
