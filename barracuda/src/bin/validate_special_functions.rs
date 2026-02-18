// SPDX-License-Identifier: AGPL-3.0-only

//! Validate `barracuda::special` functions against known reference values
//!
//! Tests: gamma, factorial, erf, erfc, bessel (j0, j1, i0, k0),
//!        laguerre, hermite, legendre, digamma, beta, lgamma,
//!        incomplete gamma, chi-squared
//!
//! Reference: Abramowitz & Stegun, DLMF, scipy.special

use std::f64::consts::PI;

use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  BarraCUDA Special Functions Validation");
    println!("  Reference: A&S / DLMF / scipy.special");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut harness = ValidationHarness::new("special_functions");

    // ─── Gamma function ────────────────────────────────────────────
    println!("── Gamma Function Γ(x) ──");
    let gamma_tests: Vec<(f64, f64, &str)> = vec![
        (1.0, 1.0, "Γ(1) = 0! = 1"),
        (2.0, 1.0, "Γ(2) = 1! = 1"),
        (3.0, 2.0, "Γ(3) = 2! = 2"),
        (4.0, 6.0, "Γ(4) = 3! = 6"),
        (5.0, 24.0, "Γ(5) = 4! = 24"),
        (0.5, PI.sqrt(), "Γ(1/2) = √π"),
        (1.5, PI.sqrt() / 2.0, "Γ(3/2) = √π/2"),
        (10.0, 362_880.0, "Γ(10) = 9!"),
    ];
    for (x, expected, desc) in &gamma_tests {
        let got = barracuda::special::gamma(*x).unwrap_or(f64::NAN);
        check(got, *expected, tolerances::GAMMA_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::GAMMA_TOLERANCE);
    }
    println!();

    // ─── Factorial ────────────────────────────────────────────────
    println!("── Factorial n! ──");
    let fact_tests: Vec<(usize, f64, &str)> = vec![
        (0, 1.0, "0! = 1"),
        (1, 1.0, "1! = 1"),
        (5, 120.0, "5! = 120"),
        (10, 3_628_800.0, "10! = 3628800"),
        (20, 2_432_902_008_176_640_000.0, "20!"),
    ];
    for (n, expected, desc) in &fact_tests {
        let got = barracuda::special::factorial(*n);
        check(got, *expected, tolerances::FACTORIAL_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::FACTORIAL_TOLERANCE);
    }
    println!();

    // ─── Error function ────────────────────────────────────────────
    println!("── Error Function erf(x) ──");
    let erf_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 0.0, "erf(0) = 0"),
        (0.5, 0.520_499_877_8, "erf(0.5)"),
        (1.0, 0.842_700_792_9, "erf(1.0)"),
        (2.0, 0.995_322_265_0, "erf(2.0)"),
        (-1.0, -0.842_700_792_9, "erf(-1) = -erf(1)"),
        (3.0, 0.999_977_909_5, "erf(3.0)"),
    ];
    for (x, expected, desc) in &erf_tests {
        let got = barracuda::special::erf(*x);
        check(got, *expected, tolerances::ERF_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::ERF_TOLERANCE);
    }

    println!("\n── Complementary Error Function erfc(x) ──");
    let erfc_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 1.0, "erfc(0) = 1"),
        (1.0, 0.157_299_207_0, "erfc(1.0)"),
        (2.0, 0.004_677_735_0, "erfc(2.0)"),
    ];
    for (x, expected, desc) in &erfc_tests {
        let got = barracuda::special::erfc(*x);
        check(got, *expected, tolerances::ERF_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::ERF_TOLERANCE);
    }
    println!();

    // ─── Bessel functions ──────────────────────────────────────────
    println!("── Bessel J₀(x) ──");
    let j0_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 1.0, "J₀(0) = 1"),
        (1.0, 0.765_197_686_6, "J₀(1)"),
        (2.4048, 0.0, "J₀(2.4048) ≈ 0 (first zero)"),
        (5.0, -0.177_596_771_3, "J₀(5)"),
    ];
    for (x, expected, desc) in &j0_tests {
        let got = barracuda::special::bessel_j0(*x);
        let tol = if desc.contains("first zero") {
            tolerances::BESSEL_NEAR_ZERO_ABS
        } else {
            tolerances::BESSEL_TOLERANCE
        };
        check(got, *expected, tol, desc);
        harness.check_abs_or_rel(desc, got, *expected, tol);
    }

    println!("\n── Bessel J₁(x) ──");
    let j1_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 0.0, "J₁(0) = 0"),
        (1.0, 0.440_050_585_7, "J₁(1)"),
        (2.0, 0.576_724_807_8, "J₁(2)"),
    ];
    for (x, expected, desc) in &j1_tests {
        let got = barracuda::special::bessel_j1(*x);
        check(got, *expected, tolerances::BESSEL_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::BESSEL_TOLERANCE);
    }

    println!("\n── Bessel I₀(x) (Modified, 1st kind) ──");
    let i0_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 1.0, "I₀(0) = 1"),
        (1.0, 1.266_065_877_8, "I₀(1)"),
        (2.0, 2.279_585_302_4, "I₀(2)"),
    ];
    for (x, expected, desc) in &i0_tests {
        let got = barracuda::special::bessel_i0(*x);
        check(got, *expected, tolerances::BESSEL_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::BESSEL_TOLERANCE);
    }

    println!("\n── Bessel K₀(x) (Modified, 2nd kind) ──");
    let k0_tests: Vec<(f64, f64, &str)> = vec![
        (0.5, 0.924_419_071_3, "K₀(0.5)"),
        (1.0, 0.421_024_438_2, "K₀(1)"),
        (2.0, 0.113_893_872_7, "K₀(2)"),
    ];
    for (x, expected, desc) in &k0_tests {
        let got = barracuda::special::bessel_k0(*x);
        check(got, *expected, tolerances::BESSEL_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::BESSEL_TOLERANCE);
    }
    println!();

    // ─── Laguerre polynomials ──────────────────────────────────────
    println!("── Laguerre Polynomials L_n^α(x) ──");
    let lag_tests: Vec<(usize, f64, f64, f64, &str)> = vec![
        (0, 0.0, 2.0, 1.0, "L₀⁰(2) = 1"),
        (1, 0.0, 2.0, -1.0, "L₁⁰(2) = 1-2 = -1"),
        (2, 0.0, 2.0, -1.0, "L₂⁰(2) = 1-2·2+4/2 = -1"),
        (0, 0.5, 1.0, 1.0, "L₀^(1/2)(1) = 1"),
        (1, 0.5, 1.0, 0.5, "L₁^(1/2)(1) = 3/2 - 1 = 0.5"),
    ];
    for (n, alpha, x, expected, desc) in &lag_tests {
        let got = barracuda::special::laguerre(*n, *alpha, *x);
        check(got, *expected, tolerances::LAGUERRE_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::LAGUERRE_TOLERANCE);
    }
    println!();

    // ─── Hermite polynomials ───────────────────────────────────────
    println!("── Hermite Polynomials Hₙ(x) ──");
    let herm_tests: Vec<(usize, f64, f64, &str)> = vec![
        (0, 2.0, 1.0, "H₀(2) = 1"),
        (1, 2.0, 4.0, "H₁(2) = 2·2 = 4"),
        (2, 2.0, 14.0, "H₂(2) = 4·4-2 = 14"),
        (3, 2.0, 40.0, "H₃(2) = 8·8-12·2 = 40"),
        (4, 2.0, 76.0, "H₄(2) = 16·16-48·4+12 = 76"),
        (0, 0.0, 1.0, "H₀(0) = 1"),
        (2, 0.0, -2.0, "H₂(0) = -2"),
    ];
    for (n, x, expected, desc) in &herm_tests {
        let got = barracuda::special::hermite(*n, *x);
        check(got, *expected, tolerances::EXACT_F64, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::EXACT_F64);
    }
    println!();

    // ─── Legendre polynomials ──────────────────────────────────────
    println!("── Legendre Polynomials Pₙ(x) ──");
    let leg_tests: Vec<(usize, f64, f64, &str)> = vec![
        (0, 0.5, 1.0, "P₀(0.5) = 1"),
        (1, 0.5, 0.5, "P₁(0.5) = 0.5"),
        (2, 0.5, -0.125, "P₂(0.5) = (3·0.25-1)/2 = -0.125"),
        (3, 0.5, -0.4375, "P₃(0.5)"),
        (10, 1.0, 1.0, "Pₙ(1) = 1 for all n"),
        (10, -1.0, 1.0, "P₁₀(-1) = 1 (even n)"),
    ];
    for (n, x, expected, desc) in &leg_tests {
        let got = barracuda::special::legendre(*n, *x);
        check(got, *expected, tolerances::EXACT_F64, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::EXACT_F64);
    }

    println!("\n── Associated Legendre Pₙᵐ(x) ──");
    let assoc_tests: Vec<(usize, i32, f64, f64, &str)> = vec![
        (1, 0, 0.5, 0.5, "P₁⁰(0.5) = P₁(0.5)"),
        (1, 1, 0.5, -0.866_025_403_8, "P₁¹(0.5) = -√(1-0.25)"),
        (2, 0, 0.5, -0.125, "P₂⁰(0.5) = P₂(0.5)"),
        (2, 1, 0.5, -1.299_038_105_676_658, "P₂¹(0.5)"),
        (2, 2, 0.5, 2.25, "P₂²(0.5) = 3(1-0.25)"),
    ];
    for (n, m, x, expected, desc) in &assoc_tests {
        let got = barracuda::special::assoc_legendre(*n, *m, *x);
        check(got, *expected, tolerances::ASSOC_LEGENDRE_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::ASSOC_LEGENDRE_TOLERANCE);
    }
    println!();

    // ─── Digamma ψ(x) — computed via ln_gamma numerical derivative ──
    println!("── Digamma ψ(x) ──");
    let digamma_tests: Vec<(f64, f64, &str)> = vec![
        (1.0, -0.577_215_664_9, "ψ(1) = -γ (Euler-Mascheroni)"),
        (2.0, 0.422_784_335_1, "ψ(2) = 1 - γ"),
        (0.5, -1.963_510_026_0, "ψ(1/2) = -γ - 2ln2"),
    ];
    for (x, expected, desc) in &digamma_tests {
        // ψ(x) = d/dx ln Γ(x), approximate via central difference
        let h = 1e-7;
        let lg_plus = barracuda::special::ln_gamma(*x + h).unwrap_or(f64::NAN);
        let lg_minus = barracuda::special::ln_gamma(*x - h).unwrap_or(f64::NAN);
        let got = (lg_plus - lg_minus) / (2.0 * h);
        check(got, *expected, tolerances::DIGAMMA_FD_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::DIGAMMA_FD_TOLERANCE);
    }
    println!();

    // ─── Beta B(a,b) = Γ(a)Γ(b)/Γ(a+b) ──────────────────────────
    println!("── Beta Function B(a,b) ──");
    let beta_tests: Vec<(f64, f64, f64, &str)> = vec![
        (1.0, 1.0, 1.0, "B(1,1) = 1"),
        (2.0, 3.0, 1.0 / 12.0, "B(2,3) = Γ(2)Γ(3)/Γ(5) = 1/12"),
        (0.5, 0.5, PI, "B(1/2,1/2) = π"),
    ];
    for (a, b, expected, desc) in &beta_tests {
        // B(a,b) = exp(lnΓ(a) + lnΓ(b) - lnΓ(a+b))
        let lg_a = barracuda::special::ln_gamma(*a).unwrap_or(f64::NAN);
        let lg_b = barracuda::special::ln_gamma(*b).unwrap_or(f64::NAN);
        let lg_ab = barracuda::special::ln_gamma(*a + *b).unwrap_or(f64::NAN);
        let got = (lg_a + lg_b - lg_ab).exp();
        check(got, *expected, tolerances::BETA_VIA_LNGAMMA_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::BETA_VIA_LNGAMMA_TOLERANCE);
    }
    println!();

    // ─── Log-gamma ln_gamma(x) ──────────────────────────────────────
    println!("── Log-Gamma ln_gamma(x) ──");
    let lgamma_tests: Vec<(f64, f64, &str)> = vec![
        (1.0, 0.0, "ln_gamma(1) = ln(0!) = 0"),
        (2.0, 0.0, "ln_gamma(2) = ln(1!) = 0"),
        (5.0, 24.0_f64.ln(), "ln_gamma(5) = ln(24)"),
        (10.0, 362_880.0_f64.ln(), "ln_gamma(10) = ln(9!)"),
        (0.5, (PI.sqrt()).ln(), "ln_gamma(1/2) = ln(√π)"),
    ];
    for (x, expected, desc) in &lgamma_tests {
        let got = barracuda::special::ln_gamma(*x).unwrap_or(f64::NAN);
        check(got, *expected, tolerances::GAMMA_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::GAMMA_TOLERANCE);
    }
    println!();

    // ─── Incomplete Gamma (NEW) ─────────────────────────────────────
    println!("── Regularized Incomplete Gamma P(a,x) ──");
    let inc_gamma_tests: Vec<(f64, f64, f64, &str)> = vec![
        // P(1,x) = 1 - e^(-x)
        (1.0, 1.0, 1.0 - (-1.0_f64).exp(), "P(1,1) = 1-e⁻¹"),
        (1.0, 2.0, 1.0 - (-2.0_f64).exp(), "P(1,2) = 1-e⁻²"),
        // P(a,0) = 0
        (2.0, 0.0, 0.0, "P(2,0) = 0"),
        // P(a,∞) → 1
        (2.0, 100.0, 1.0, "P(2,100) ≈ 1"),
    ];
    for (a, x, expected, desc) in &inc_gamma_tests {
        let got = barracuda::special::regularized_gamma_p(*a, *x).unwrap_or(f64::NAN);
        check(got, *expected, tolerances::INCOMPLETE_GAMMA_TOLERANCE, desc);
        harness.check_abs_or_rel(desc, got, *expected, tolerances::INCOMPLETE_GAMMA_TOLERANCE);
    }
    println!();

    // ─── Chi-squared Distribution (NEW) ──────────────────────────────
    println!("── Chi-Squared Distribution ──");
    // χ²_CDF(x, k) = P(k/2, x/2)
    {
        // CDF at critical values: scipy.stats.chi2.cdf(3.841, 1) ≈ 0.95
        let got = barracuda::special::chi_squared_cdf(3.841, 1.0).unwrap_or(f64::NAN);
        check(
            got,
            0.95,
            tolerances::CHI2_CDF_TOLERANCE,
            "χ²_CDF(3.841, k=1) ≈ 0.95",
        );
        harness.check_abs_or_rel(
            "χ²_CDF(3.841, k=1) ≈ 0.95",
            got,
            0.95,
            tolerances::CHI2_CDF_TOLERANCE,
        );

        // CDF at critical values: scipy.stats.chi2.cdf(5.991, 2) ≈ 0.95
        let got = barracuda::special::chi_squared_cdf(5.991, 2.0).unwrap_or(f64::NAN);
        check(
            got,
            0.95,
            tolerances::CHI2_CDF_TOLERANCE,
            "χ²_CDF(5.991, k=2) ≈ 0.95",
        );
        harness.check_abs_or_rel(
            "χ²_CDF(5.991, k=2) ≈ 0.95",
            got,
            0.95,
            tolerances::CHI2_CDF_TOLERANCE,
        );

        // Quantile (inverse CDF): scipy.stats.chi2.ppf(0.95, 1) ≈ 3.8415
        let got = barracuda::special::chi_squared_quantile(0.95, 1.0).unwrap_or(f64::NAN);
        check(
            got,
            3.8415,
            tolerances::CHI2_CDF_TOLERANCE,
            "χ²_quantile(0.95, k=1) ≈ 3.8415",
        );
        harness.check_abs_or_rel(
            "χ²_quantile(0.95, k=1) ≈ 3.8415",
            got,
            3.8415,
            tolerances::CHI2_CDF_TOLERANCE,
        );

        // PDF at x=2, k=3: scipy.stats.chi2.pdf(2, 3) ≈ 0.2076
        let got = barracuda::special::chi_squared_pdf(2.0, 3.0).unwrap_or(f64::NAN);
        check(
            got,
            0.2076,
            tolerances::CHI2_CDF_TOLERANCE,
            "χ²_PDF(2, k=3) ≈ 0.2076",
        );
        harness.check_abs_or_rel(
            "χ²_PDF(2, k=3) ≈ 0.2076",
            got,
            0.2076,
            tolerances::CHI2_CDF_TOLERANCE,
        );
    }
    println!();

    // ─── Summary (with exit code) ──────────────────────────────────
    println!("\n  Reference: {}", provenance::SPECIAL_FUNCTION_REFS);
    harness.finish();
}

fn check(got: f64, expected: f64, tol: f64, desc: &str) -> bool {
    let err = (got - expected).abs();
    let rel_err = if expected.abs() > 1e-14 {
        err / expected.abs()
    } else {
        err
    };

    let pass = err < tol || rel_err < tol;

    if pass {
        println!("  ✅ {desc} | got={got:.10} expected={expected:.10}");
    } else {
        println!("  ❌ {desc} | got={got:.10} expected={expected:.10} err={err:.2e}");
    }
    pass
}
