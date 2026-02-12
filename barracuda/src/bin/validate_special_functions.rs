//! Validate barracuda::special functions against known reference values
//!
//! Tests: gamma, factorial, erf, erfc, bessel (j0, j1, i0, k0),
//!        laguerre, hermite, legendre, digamma, beta, lgamma
//!
//! Reference: Abramowitz & Stegun, DLMF, scipy.special

use std::f64::consts::PI;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  BarraCUDA Special Functions Validation");
    println!("  Reference: A&S / DLMF / scipy.special");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut total = 0;
    let mut passed = 0;

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
        (10.0, 362880.0, "Γ(10) = 9!"),
    ];
    for (x, expected, desc) in &gamma_tests {
        let got = barracuda::special::gamma(*x);
        let ok = check(got, *expected, 1e-10, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Factorial ────────────────────────────────────────────────
    println!("── Factorial n! ──");
    let fact_tests: Vec<(usize, f64, &str)> = vec![
        (0, 1.0, "0! = 1"),
        (1, 1.0, "1! = 1"),
        (5, 120.0, "5! = 120"),
        (10, 3628800.0, "10! = 3628800"),
        (20, 2432902008176640000.0, "20!"),
    ];
    for (n, expected, desc) in &fact_tests {
        let got = barracuda::special::factorial(*n);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Error function ────────────────────────────────────────────
    println!("── Error Function erf(x) ──");
    let erf_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 0.0, "erf(0) = 0"),
        (0.5, 0.5204998778, "erf(0.5)"),
        (1.0, 0.8427007929, "erf(1.0)"),
        (2.0, 0.9953222650, "erf(2.0)"),
        (-1.0, -0.8427007929, "erf(-1) = -erf(1)"),
        (3.0, 0.9999779095, "erf(3.0)"),
    ];
    for (x, expected, desc) in &erf_tests {
        let got = barracuda::special::erf(*x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }

    println!("\n── Complementary Error Function erfc(x) ──");
    let erfc_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 1.0, "erfc(0) = 1"),
        (1.0, 0.1572992070, "erfc(1.0)"),
        (2.0, 0.0046777350, "erfc(2.0)"),
    ];
    for (x, expected, desc) in &erfc_tests {
        let got = barracuda::special::erfc(*x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Bessel functions ──────────────────────────────────────────
    println!("── Bessel J₀(x) ──");
    let j0_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 1.0, "J₀(0) = 1"),
        (1.0, 0.7651976866, "J₀(1)"),
        (2.4048, 0.0, "J₀(2.4048) ≈ 0 (first zero)"),
        (5.0, -0.1775967713, "J₀(5)"),
    ];
    for (x, expected, desc) in &j0_tests {
        let got = barracuda::special::bessel_j0(*x);
        let tol = if desc.contains("first zero") { 0.001 } else { 1e-6 };
        let ok = check(got, *expected, tol, desc);
        total += 1;
        if ok { passed += 1; }
    }

    println!("\n── Bessel J₁(x) ──");
    let j1_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 0.0, "J₁(0) = 0"),
        (1.0, 0.4400505857, "J₁(1)"),
        (2.0, 0.5767248078, "J₁(2)"),
    ];
    for (x, expected, desc) in &j1_tests {
        let got = barracuda::special::bessel_j1(*x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }

    println!("\n── Bessel I₀(x) (Modified, 1st kind) ──");
    let i0_tests: Vec<(f64, f64, &str)> = vec![
        (0.0, 1.0, "I₀(0) = 1"),
        (1.0, 1.2660658778, "I₀(1)"),
        (2.0, 2.2795853024, "I₀(2)"),
    ];
    for (x, expected, desc) in &i0_tests {
        let got = barracuda::special::bessel_i0(*x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }

    println!("\n── Bessel K₀(x) (Modified, 2nd kind) ──");
    let k0_tests: Vec<(f64, f64, &str)> = vec![
        (0.5, 0.9244190713, "K₀(0.5)"),
        (1.0, 0.4210244382, "K₀(1)"),
        (2.0, 0.1138938727, "K₀(2)"),
    ];
    for (x, expected, desc) in &k0_tests {
        let got = barracuda::special::bessel_k0(*x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Laguerre polynomials ──────────────────────────────────────
    println!("── Laguerre Polynomials L_n^α(x) ──");
    // L_0^0(x) = 1, L_1^0(x) = 1-x, L_2^0(x) = 1-2x+x²/2
    let lag_tests: Vec<(usize, f64, f64, f64, &str)> = vec![
        (0, 0.0, 2.0, 1.0, "L₀⁰(2) = 1"),
        (1, 0.0, 2.0, -1.0, "L₁⁰(2) = 1-2 = -1"),
        (2, 0.0, 2.0, -1.0, "L₂⁰(2) = 1-2·2+4/2 = -1"),
        (0, 0.5, 1.0, 1.0, "L₀^(1/2)(1) = 1"),
        (1, 0.5, 1.0, 0.5, "L₁^(1/2)(1) = 3/2 - 1 = 0.5"),
    ];
    for (n, alpha, x, expected, desc) in &lag_tests {
        let got = barracuda::special::laguerre(*n, *alpha, *x);
        let ok = check(got, *expected, 1e-10, desc);
        total += 1;
        if ok { passed += 1; }
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
        let ok = check(got, *expected, 1e-10, desc);
        total += 1;
        if ok { passed += 1; }
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
        let ok = check(got, *expected, 1e-10, desc);
        total += 1;
        if ok { passed += 1; }
    }

    println!("\n── Associated Legendre Pₙᵐ(x) ──");
    let assoc_tests: Vec<(usize, i32, f64, f64, &str)> = vec![
        (1, 0, 0.5, 0.5, "P₁⁰(0.5) = P₁(0.5)"),
        (1, 1, 0.5, -0.8660254038, "P₁¹(0.5) = -√(1-0.25)"),
        (2, 0, 0.5, -0.125, "P₂⁰(0.5) = P₂(0.5)"),
        (2, 1, 0.5, -1.299038106, "P₂¹(0.5)"),
        (2, 2, 0.5, 2.25, "P₂²(0.5) = 3(1-0.25)"),
    ];
    for (n, m, x, expected, desc) in &assoc_tests {
        let got = barracuda::special::assoc_legendre(*n, *m, *x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Digamma ψ(x) ─────────────────────────────────────────────
    println!("── Digamma ψ(x) ──");
    let digamma_tests: Vec<(f64, f64, &str)> = vec![
        (1.0, -0.5772156649, "ψ(1) = -γ (Euler-Mascheroni)"),
        (2.0, 0.4227843351, "ψ(2) = 1 - γ"),
        (0.5, -1.9635100260, "ψ(1/2) = -γ - 2ln2"),
    ];
    for (x, expected, desc) in &digamma_tests {
        let got = barracuda::special::digamma(*x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Beta B(a,b) ──────────────────────────────────────────────
    println!("── Beta Function B(a,b) ──");
    let beta_tests: Vec<(f64, f64, f64, &str)> = vec![
        (1.0, 1.0, 1.0, "B(1,1) = 1"),
        (2.0, 3.0, 1.0 / 12.0, "B(2,3) = Γ(2)Γ(3)/Γ(5) = 1/12"),
        (0.5, 0.5, PI, "B(1/2,1/2) = π"),
    ];
    for (a, b, expected, desc) in &beta_tests {
        let got = barracuda::special::beta(*a, *b);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Log-gamma lgamma(x) ──────────────────────────────────────
    println!("── Log-Gamma lgamma(x) ──");
    let lgamma_tests: Vec<(f64, f64, &str)> = vec![
        (1.0, 0.0, "lgamma(1) = ln(0!) = 0"),
        (2.0, 0.0, "lgamma(2) = ln(1!) = 0"),
        (5.0, 24.0_f64.ln(), "lgamma(5) = ln(24)"),
        (10.0, 362880.0_f64.ln(), "lgamma(10) = ln(9!)"),
        (0.5, (PI.sqrt()).ln(), "lgamma(1/2) = ln(√π)"),
    ];
    for (x, expected, desc) in &lgamma_tests {
        let got = barracuda::special::lgamma(*x);
        let ok = check(got, *expected, 1e-6, desc);
        total += 1;
        if ok { passed += 1; }
    }
    println!();

    // ─── Summary ──────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  Special Functions: {}/{} passed", passed, total);
    if passed == total {
        println!("  ✅ ALL TESTS PASSED");
    } else {
        println!("  ❌ {} FAILURES", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");
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
        println!("  ✅ {} | got={:.10} expected={:.10}", desc, got, expected);
    } else {
        println!(
            "  ❌ {} | got={:.10} expected={:.10} err={:.2e}",
            desc, got, expected, err
        );
    }
    pass
}

