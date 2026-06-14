// SPDX-License-Identifier: AGPL-3.0-or-later

use std::time::Instant;

pub fn bench_special_functions_cpu() {
    println!("═══ Phase 1: Special Functions (CPU) ═══");
    println!("  Provenance: hotSpring → toadStool S25–S68 (gamma, erf, bessel, hermite, laguerre)");
    println!();

    let n_evals = 100_000;

    // Gamma function
    let t = Instant::now();
    let mut gamma_sum = 0.0f64;
    for i in 1..=n_evals {
        let x = (f64::from(i) / f64::from(n_evals)).mul_add(10.0, 0.5);
        gamma_sum += barracuda::special::gamma(x).unwrap_or(0.0);
    }
    let gamma_us = t.elapsed().as_micros();
    println!(
        "  gamma(x)     : {n_evals} evals in {gamma_us} µs ({:.1} ns/eval) — checksum={gamma_sum:.6e}",
        gamma_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Error function
    let t = Instant::now();
    let mut erf_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)).mul_add(6.0, -3.0);
        erf_sum += barracuda::special::erf(x);
    }
    let erf_us = t.elapsed().as_micros();
    println!(
        "  erf(x)       : {n_evals} evals in {erf_us} µs ({:.1} ns/eval) — checksum={erf_sum:.6e}",
        erf_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Bessel J0
    let t = Instant::now();
    let mut j0_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)) * 20.0;
        j0_sum += barracuda::special::bessel_j0(x);
    }
    let j0_us = t.elapsed().as_micros();
    println!(
        "  bessel_j0(x) : {n_evals} evals in {j0_us} µs ({:.1} ns/eval) — checksum={j0_sum:.6e}",
        j0_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Hermite polynomial H_10(x) — nuclear HFB wavefunctions
    let t = Instant::now();
    let mut herm_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)).mul_add(8.0, -4.0);
        herm_sum += barracuda::special::hermite(10, x);
    }
    let herm_us = t.elapsed().as_micros();
    println!(
        "  hermite(10,x): {n_evals} evals in {herm_us} µs ({:.1} ns/eval) — checksum={herm_sum:.6e}",
        herm_us as f64 * 1000.0 / f64::from(n_evals)
    );

    // Laguerre polynomial L_5^(0.5)(x) — nuclear deformed basis
    let t = Instant::now();
    let mut lag_sum = 0.0f64;
    for i in 0..n_evals {
        let x = (f64::from(i) / f64::from(n_evals)) * 10.0;
        lag_sum += barracuda::special::laguerre(5, 0.5, x);
    }
    let lag_us = t.elapsed().as_micros();
    println!(
        "  laguerre(5,x): {n_evals} evals in {lag_us} µs ({:.1} ns/eval) — checksum={lag_sum:.6e}",
        lag_us as f64 * 1000.0 / f64::from(n_evals)
    );

    println!();
}

pub fn bench_stable_specials_cpu() {
    println!("═══ Phase 1e: Stable GPU Special Functions (CPU reference) ═══");
    println!("  Provenance: wetSpring+hotSpring → barraCuda Sprint 2");
    println!("  Cross-spring: log1p/expm1/erfc/J₀-1 avoid catastrophic cancellation");
    println!("  hotSpring uses: screened Coulomb (erfc), dielectric (log1p), BCS (expm1)");
    println!("  wetSpring uses: HMM log-domain (log1p), diversity (erfc)");
    println!();

    use barracuda::special::stable_gpu::{bessel_j0_minus1_f64, erfc_f64, expm1_f64, log1p_f64};

    let n = 100_000;

    let t = Instant::now();
    let mut sum = 0.0_f64;
    for i in 0..n {
        sum += log1p_f64(f64::from(i) * 1e-10);
    }
    let us = t.elapsed().as_micros();
    println!("  log1p(x)      : {n} evals in {us}µs — checksum={sum:.6e}");

    let t = Instant::now();
    sum = 0.0;
    for i in 0..n {
        sum += expm1_f64(f64::from(i) * 1e-10);
    }
    let us = t.elapsed().as_micros();
    println!("  expm1(x)      : {n} evals in {us}µs — checksum={sum:.6e}");

    let t = Instant::now();
    sum = 0.0;
    for i in 0..n {
        let x = f64::from(i) / f64::from(n) * 6.0;
        sum += erfc_f64(x);
    }
    let us = t.elapsed().as_micros();
    println!("  erfc(x)       : {n} evals in {us}µs — checksum={sum:.6e}");

    let t = Instant::now();
    sum = 0.0;
    for i in 0..n {
        let x = f64::from(i) / f64::from(n) * 0.1;
        sum += bessel_j0_minus1_f64(x);
    }
    let us = t.elapsed().as_micros();
    println!("  J₀(x)-1       : {n} evals in {us}µs — checksum={sum:.6e}");

    // Validate stable vs naive near cancellation
    let x_small = 1e-14;
    let stable = log1p_f64(x_small);
    let naive = (1.0_f64 + x_small).ln();
    let rel_err = if stable.abs() > 0.0 {
        ((stable - naive) / stable).abs()
    } else {
        0.0
    };
    println!();
    println!(
        "  Cancellation test: log1p(1e-14)={stable:.6e}, ln(1+1e-14)={naive:.6e}, rel_err={rel_err:.2e}"
    );
    println!(
        "  → stable wins: {}",
        if rel_err < 1e-10 {
            "both accurate at this level"
        } else {
            "stable avoids cancellation"
        }
    );
    println!();
}
