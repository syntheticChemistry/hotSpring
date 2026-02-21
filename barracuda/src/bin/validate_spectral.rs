// SPDX-License-Identifier: AGPL-3.0-only

//! Spectral Theory Validation — Kachkovskiy Extension
//!
//! Validates the spectral theory primitives for 1D discrete Schrödinger operators:
//!
//! **Anderson model** (random potential):
//! - All states localized in 1D for any disorder (Anderson 1958)
//! - Lyapunov exponent γ(0) ≈ W²/96 for small disorder (Kappus-Wegner 1981)
//! - Poisson level statistics for strong disorder
//!
//! **Almost-Mathieu operator** (quasiperiodic potential):
//! - Metal-insulator transition at λ = 1 (Aubry-André 1980, Jitomirskaya 1999)
//! - γ(E) = max(0, ln|λ|) for irrational α (Herman 1983, Avila 2015)
//! - Spectrum bounded by [-2-2λ, 2+2λ]
//!
//! These validate the foundational primitives needed for Kachkovskiy's spectral
//! theory / transport work: Sturm eigensolve, transfer matrices, and spectral
//! statistics — all without FFT.
//!
//! # Provenance
//!
//! Anderson (1958) Phys. Rev. 109, 1492
//! Aubry & André (1980) Ann. Israel Phys. Soc. 3, 133
//! Jitomirskaya (1999) Ann. Math. 150, 1159
//! Herman (1983) Comment. Math. Helv. 58, 453
//! Avila (2015) Acta Math. 215, 1

use hotspring_barracuda::spectral;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Spectral Theory Validation — Kachkovskiy Extension        ║");
    println!("║  Anderson localization + almost-Mathieu transition          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("spectral_theory");

    check_anderson_spectrum(&mut harness);
    check_anderson_lyapunov(&mut harness);
    check_anderson_localization(&mut harness);
    check_anderson_level_stats(&mut harness);
    check_almost_mathieu_bounds(&mut harness);
    check_herman_formula(&mut harness);
    check_aubry_andre_transition(&mut harness);
    check_disorder_scaling(&mut harness);

    println!();
    harness.finish();
}

/// Anderson model: spectrum lies within Gershgorin bounds [-2-W/2, 2+W/2].
fn check_anderson_spectrum(harness: &mut ValidationHarness) {
    println!("[1] Anderson Model — Spectrum Bounds");

    let n = 1000;
    let w = 4.0;
    let (d, e) = spectral::anderson_hamiltonian(n, w, 42);
    let evals = spectral::find_all_eigenvalues(&d, &e);

    let lo = -2.0 - w / 2.0;
    let hi = 2.0 + w / 2.0;
    let all_in_bounds = evals.iter().all(|&ev| ev >= lo - 0.01 && ev <= hi + 0.01);
    let min_ev = evals.first().copied().unwrap_or(0.0);
    let max_ev = evals.last().copied().unwrap_or(0.0);

    println!("  N={n}, W={w}");
    println!("  Spectrum: [{min_ev:.4}, {max_ev:.4}]");
    println!("  Gershgorin: [{lo:.4}, {hi:.4}]");

    harness.check_bool("Anderson spectrum in Gershgorin bounds", all_in_bounds);
    println!();
}

/// Anderson model: Lyapunov exponent γ(0) ≈ W²/96 for weak disorder.
fn check_anderson_lyapunov(harness: &mut ValidationHarness) {
    println!("[2] Anderson Model — Lyapunov Exponent (Weak Disorder)");
    println!("    Theory: γ(0) ≈ W²/96 (Kappus & Wegner 1981)\n");

    let n_sites = 100_000;
    let n_realizations = 20;

    for &w in &[0.5, 1.0, 2.0] {
        let gamma = spectral::lyapunov_averaged(n_sites, w, 0.0, n_realizations, 42);
        let theory = w * w / 96.0;
        let rel_err = if theory > 1e-10 {
            ((gamma - theory) / theory).abs()
        } else {
            gamma.abs()
        };

        println!("  W={w:.1}: γ(0) = {gamma:.6}, theory = {theory:.6}, rel error = {rel_err:.1}%",
            rel_err = rel_err * 100.0
        );
    }

    // Use W=1.0 for the quantitative check (best convergence)
    let gamma_1 = spectral::lyapunov_averaged(n_sites, 1.0, 0.0, n_realizations, 42);
    let theory_1 = 1.0 / 96.0;
    let rel_err = ((gamma_1 - theory_1) / theory_1).abs();

    harness.check_upper("γ(0) at W=1 within 30% of W²/96", rel_err, 0.30);
    println!();
}

/// Anderson model: γ(E) > 0 for all energies → all states localized.
fn check_anderson_localization(harness: &mut ValidationHarness) {
    println!("[3] Anderson Model — All States Localized (1D)");
    println!("    Theory: γ(E) > 0 for all E when W > 0 (Goldsheid-Molchanov-Pastur 1977)\n");

    let n_sites = 50_000;
    let w = 2.0;

    let energies = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
    let mut all_positive = true;

    for &e in &energies {
        let gamma = spectral::lyapunov_averaged(n_sites, w, e, 10, 42);
        let localized = gamma > 0.001;
        if !localized {
            all_positive = false;
        }
        println!("  E={e:>5.1}: γ = {gamma:.6} {}", if localized { "✓" } else { "✗" });
    }

    harness.check_bool("all energies have γ > 0 (localized)", all_positive);
    println!();
}

/// Anderson model: strong disorder gives Poisson level statistics.
fn check_anderson_level_stats(harness: &mut ValidationHarness) {
    println!("[4] Anderson Model — Poisson Level Statistics");
    println!(
        "    Theory: ⟨r⟩ ≈ {:.4} for localized states (Oganesyan-Huse 2007)\n",
        spectral::POISSON_R
    );

    let n = 1000;
    let w = 8.0;

    // Average over several realizations for better statistics
    let mut r_sum = 0.0;
    let n_real = 10;
    for seed in 0..n_real {
        let (d, e) = spectral::anderson_hamiltonian(n, w, seed * 100 + 42);
        let evals = spectral::find_all_eigenvalues(&d, &e);
        r_sum += spectral::level_spacing_ratio(&evals);
    }
    let r_mean = r_sum / n_real as f64;

    println!("  N={n}, W={w}, ⟨r⟩ = {r_mean:.4} (Poisson = {:.4})", spectral::POISSON_R);

    let deviation = (r_mean - spectral::POISSON_R).abs();
    harness.check_upper("⟨r⟩ within 0.03 of Poisson", deviation, 0.03);
    println!();
}

/// Almost-Mathieu: spectrum bounded by [-2-2λ, 2+2λ].
fn check_almost_mathieu_bounds(harness: &mut ValidationHarness) {
    println!("[5] Almost-Mathieu — Spectrum Bounds");

    let n = 1000;
    let lambda = 2.0;
    let (d, e) = spectral::almost_mathieu_hamiltonian(n, lambda, spectral::GOLDEN_RATIO, 0.0);
    let evals = spectral::find_all_eigenvalues(&d, &e);

    let bound = 2.0 + 2.0 * lambda;
    let all_bounded = evals.iter().all(|&ev| ev.abs() <= bound + 0.01);
    let min_ev = evals.first().copied().unwrap_or(0.0);
    let max_ev = evals.last().copied().unwrap_or(0.0);

    println!("  λ={lambda}, α=golden, N={n}");
    println!("  Spectrum: [{min_ev:.4}, {max_ev:.4}]");
    println!("  Bound: [-{bound:.1}, {bound:.1}]");

    harness.check_bool("almost-Mathieu spectrum in bounds", all_bounded);
    println!();
}

/// Almost-Mathieu: Herman's formula γ = ln|λ| for λ > 1.
fn check_herman_formula(harness: &mut ValidationHarness) {
    println!("[6] Almost-Mathieu — Herman's Formula");
    println!("    Theory: γ(E) = ln|λ| for λ > 1, irrational α (Herman 1983, Avila 2015)\n");

    let n = 100_000;

    for &lambda in &[1.5, 2.0, 3.0, 5.0] {
        let pot: Vec<f64> = (0..n)
            .map(|i| 2.0 * lambda * (std::f64::consts::TAU * spectral::GOLDEN_RATIO * i as f64).cos())
            .collect();
        let gamma = spectral::lyapunov_exponent(&pot, 0.0);
        let theory = lambda.ln();
        let error = (gamma - theory).abs();

        println!("  λ={lambda:.1}: γ = {gamma:.4}, ln(λ) = {theory:.4}, error = {error:.4}");
    }

    // Quantitative check at λ=2
    let pot_2: Vec<f64> = (0..n)
        .map(|i| 2.0 * 2.0 * (std::f64::consts::TAU * spectral::GOLDEN_RATIO * i as f64).cos())
        .collect();
    let gamma_2 = spectral::lyapunov_exponent(&pot_2, 0.0);
    let theory_2 = (2.0f64).ln();
    let error = (gamma_2 - theory_2).abs();

    harness.check_upper("γ at λ=2 within 0.02 of ln(2)", error, 0.02);
    println!();
}

/// Almost-Mathieu: Aubry-André metal-insulator transition at λ = 1.
fn check_aubry_andre_transition(harness: &mut ValidationHarness) {
    println!("[7] Almost-Mathieu — Aubry-André Transition (λ = 1)");
    println!("    Theory: extended (γ=0) for λ<1, localized (γ>0) for λ>1\n");

    let n = 100_000;
    let lambdas = [0.3, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, 2.0];
    let mut extended_ok = true;
    let mut localized_ok = true;

    for &lambda in &lambdas {
        let pot: Vec<f64> = (0..n)
            .map(|i| {
                2.0 * lambda * (std::f64::consts::TAU * spectral::GOLDEN_RATIO * i as f64).cos()
            })
            .collect();
        let gamma = spectral::lyapunov_exponent(&pot, 0.0);
        let theory = if lambda > 1.0 { lambda.ln() } else { 0.0 };
        let phase = if lambda < 1.0 { "extended" } else { "localized" };

        println!("  λ={lambda:.2}: γ = {gamma:.5}, theory = {theory:.5}  [{phase}]");

        if lambda < 0.9 && gamma > 0.05 {
            extended_ok = false;
        }
        if lambda > 1.1 && gamma < 0.05 {
            localized_ok = false;
        }
    }

    harness.check_bool("extended phase (λ<1): γ ≈ 0", extended_ok);
    harness.check_bool("localized phase (λ>1): γ > 0", localized_ok);
    println!();
}

/// Anderson disorder scaling: γ(0) grows as W² for small W.
fn check_disorder_scaling(harness: &mut ValidationHarness) {
    println!("[8] Disorder Scaling — γ ∝ W²");

    let n_sites = 100_000;
    let n_real = 20;

    let w_values = [0.5, 1.0, 2.0, 4.0];
    let mut gammas = Vec::new();

    for &w in &w_values {
        let gamma = spectral::lyapunov_averaged(n_sites, w, 0.0, n_real, 42);
        gammas.push(gamma);
        println!("  W={w:.1}: γ(0) = {gamma:.6}");
    }

    // Check that doubling W roughly quadruples γ (W² scaling)
    let ratio_1 = gammas[1] / gammas[0]; // W=1 / W=0.5 → should be ~4
    let ratio_2 = gammas[2] / gammas[1]; // W=2 / W=1 → should be ~4

    println!("  γ(W=1)/γ(W=0.5) = {ratio_1:.2} (expect ~4)");
    println!("  γ(W=2)/γ(W=1) = {ratio_2:.2} (expect ~4)");

    // Allow generous tolerance since perturbation theory is approximate at W=2-4
    let scaling_ok = ratio_1 > 2.0 && ratio_1 < 8.0 && ratio_2 > 2.0 && ratio_2 < 8.0;
    harness.check_bool("W² scaling: ratios between 2 and 8", scaling_ok);

    // Monotonicity: γ strictly increases with W
    let monotonic = gammas.windows(2).all(|w| w[1] > w[0]);
    harness.check_bool("γ monotonically increases with disorder", monotonic);
    println!();
}
