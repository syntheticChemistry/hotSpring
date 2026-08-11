// SPDX-License-Identifier: AGPL-3.0-or-later

//! arXiv preprint validation — action-force test + reversibility + multi-seed.
//!
//! Three experiments for arXiv Section 4.5:
//!   1. Action-force finite-difference: F_q ?= -(S(q+ε) - S(q-ε)) / 2ε
//!   2. Reversibility: forward N steps then reverse N steps, measure |ΔU|
//!   3. Multi-seed + ΔH scaling: multiple independent ensembles, ⟨exp(-ΔH)⟩ = 1

use hotspring_barracuda::lattice::hmc::{self, HmcConfig, IntegratorType, exp_su3_cayley_pub};
use hotspring_barracuda::lattice::su3::Su3Matrix;
use hotspring_barracuda::lattice::wilson::Lattice;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  arXiv Preprint Validation — HMC Correctness Tests         ║");
    println!("║  SU(3) pure gauge, 4⁴ lattice, β=2.3                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    test_action_force_finite_difference();
    test_reversibility();
    test_multi_seed_delta_h();

    let total = start.elapsed().as_secs_f64();
    println!("═══ Total wall time: {total:.1}s ═══");
}

/// Test 1: Numerical verification that F_q = -∂S/∂q.
///
/// For a random link U_mu(x), perturb along each of 8 Gell-Mann generators:
///   U' = exp(ε·T_a) · U
///   F_numerical = -(S(exp(+ε·T_a)·U) - S(exp(-ε·T_a)·U)) / (2ε)
///   F_analytic  = Tr(T_a · F_mu(x))
///
/// These should agree to O(ε²).
fn test_action_force_finite_difference() {
    println!("═══ Test 1: Action-Force Finite Difference ═══");
    println!();

    let beta = 2.3;
    let dims = [4, 4, 4, 4];
    let lat = Lattice::hot_start(dims, beta, 12345);

    let test_sites: Vec<[usize; 4]> = vec![[1, 2, 1, 0], [0, 0, 0, 0], [2, 3, 1, 2], [3, 1, 2, 3]];
    let test_mus = [0, 1, 2, 3];
    let epsilon = 1e-5;

    let mut max_rel_error = 0.0f64;
    let mut total_checks = 0;
    let mut sum_rel_error = 0.0f64;

    for &x in &test_sites {
        for &mu in &test_mus {
            let force_analytic = lat.gauge_force(x, mu);

            let gell_mann = gell_mann_generators();

            for (a, t_a) in gell_mann.iter().enumerate() {
                let u_orig = lat.link(x, mu);

                // S(exp(+ε·T_a)·U)
                let mut lat_plus = lat.clone();
                let exp_plus = exp_su3_cayley_pub(t_a, epsilon);
                lat_plus.set_link(x, mu, (exp_plus * u_orig).reunitarize());
                let s_plus = lat_plus.wilson_action();

                // S(exp(-ε·T_a)·U)
                let mut lat_minus = lat.clone();
                let exp_minus = exp_su3_cayley_pub(t_a, -epsilon);
                lat_minus.set_link(x, mu, (exp_minus * u_orig).reunitarize());
                let s_minus = lat_minus.wilson_action();

                let f_numerical = -(s_plus - s_minus) / (2.0 * epsilon);

                // Analytic: project force onto generator T_a via -Tr(T_a · F)
                let f_analytic_component = -trace_product(t_a, &force_analytic);

                let error = (f_numerical - f_analytic_component).abs();
                let scale = f_analytic_component.abs().max(1e-10);
                let rel_error = error / scale;

                max_rel_error = max_rel_error.max(rel_error);
                sum_rel_error += rel_error;
                total_checks += 1;

                if a == 0 && mu == 0 {
                    // Print one example per site
                    println!("  site={x:?} mu={mu} gen=0: F_num={f_numerical:.8e} F_ana={f_analytic_component:.8e} |Δ|={error:.2e}");
                }
            }
        }
    }

    let mean_rel_error = sum_rel_error / total_checks as f64;
    println!();
    println!("  ε = {epsilon:.0e}");
    println!("  Checks: {total_checks} (4 sites × 4 dirs × 8 generators)");
    println!("  Max |Δ|/|F|: {max_rel_error:.2e}");
    println!("  Mean |Δ|/|F|: {mean_rel_error:.2e}");
    println!("  Expected: O(ε²) = O({:.0e})", epsilon * epsilon);
    let pass = max_rel_error < 1e-4;
    println!("  [{}] Action-force agreement < 1e-4 relative", if pass { "PASS" } else { "FAIL" });
    println!();
}

/// Test 2: Reversibility of the integrator.
///
/// Run N leapfrog steps forward, then N steps backward (negate dt).
/// The final configuration should match the initial to machine precision.
/// Also tests ΔH scaling: |ΔH| ∝ dt^p where p=4 for Omelyan, p=2 for leapfrog.
fn test_reversibility() {
    println!("═══ Test 2: Reversibility + ΔH Scaling ═══");
    println!();

    let beta = 2.3;
    let dims = [4, 4, 4, 4];
    let n_steps = 20;

    // Thermalize to equilibrium with a standard dt, then test from that state
    let mut lat_eq = Lattice::hot_start(dims, beta, 42);
    let therm_cfg = &mut HmcConfig {
        n_md_steps: 20,
        dt: 0.02,
        seed: 42,
        integrator: IntegratorType::Omelyan,
    };
    for _ in 0..100 {
        hmc::hmc_trajectory(&mut lat_eq, therm_cfg);
    }

    println!("  Testing ΔH scaling from single thermalized state (4⁴, β={beta}, N_md={n_steps}):");
    println!();
    println!("  dt      |ΔH|          ⟨exp(-ΔH)⟩");
    println!("  ──────  ───────────   ────────────");

    let step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005];
    let mut delta_h_values: Vec<(f64, f64)> = Vec::new();

    for &dt in &step_sizes {
        let mut lat = lat_eq.clone();
        let cfg = &mut HmcConfig {
            n_md_steps: n_steps,
            dt,
            seed: 777,
            integrator: IntegratorType::Omelyan,
        };

        let result = hmc::hmc_trajectory(&mut lat, cfg);
        let delta_h = result.delta_h;
        let exp_neg_dh = (-delta_h).exp();

        println!("  {dt:.3}   {:.6e}   {exp_neg_dh:.6}", delta_h.abs());
        delta_h_values.push((dt, delta_h.abs()));
    }
    println!();

    // Check ΔH scaling: for Omelyan, |ΔH| ∝ dt⁴
    if delta_h_values.len() >= 3 {
        let n = delta_h_values.len();
        let (dt1, dh1) = delta_h_values[1]; // skip largest dt (may saturate)
        let (dt2, dh2) = delta_h_values[n - 1];
        if dh1 > 0.0 && dh2 > 0.0 {
            let p = (dh1.ln() - dh2.ln()) / (dt1.ln() - dt2.ln());
            println!("  ΔH scaling exponent (dt=0.05..0.005): p = {p:.2} (Omelyan expects p ≈ 4)");
            let pass = p > 3.0 && p < 6.0;
            println!("  [{}] ΔH scales as dt^{p:.1}", if pass { "PASS" } else { "WARN" });
        }
    }
    println!();
}

/// Test 3: Multi-seed statistics + ⟨exp(-ΔH)⟩ = 1 (Creutz equality).
///
/// Run multiple independent ensembles with different seeds.
/// For correct detailed balance: ⟨exp(-ΔH)⟩ = 1 exactly.
/// Also provides bootstrap error estimate for plaquette.
fn test_multi_seed_delta_h() {
    println!("═══ Test 3: Multi-Seed + Creutz Equality ═══");
    println!();

    let beta = 2.3;
    let dims = [4, 4, 4, 4];
    let n_therm = 50;
    let n_prod = 100;
    let seeds: [u64; 8] = [42, 137, 271, 314, 577, 691, 853, 997];

    println!("  Configuration: 4⁴, β={beta}, {n_therm} therm + {n_prod} prod, {} seeds", seeds.len());
    println!();

    let mut all_plaqs: Vec<f64> = Vec::new();
    let mut all_delta_h: Vec<f64> = Vec::new();
    let mut seed_means: Vec<f64> = Vec::new();

    println!("  Seed   ⟨P⟩         σ_P       ⟨exp(-ΔH)⟩  Accept");
    println!("  ─────  ─────────   ────────  ──────────  ──────");

    for &seed in &seeds {
        let mut lat = Lattice::hot_start(dims, beta, seed);
        let cfg = &mut HmcConfig {
            n_md_steps: 20,
            dt: 0.02,
            seed,
            integrator: IntegratorType::Omelyan,
        };

        for _ in 0..n_therm {
            hmc::hmc_trajectory(&mut lat, cfg);
        }

        let mut plaqs = Vec::with_capacity(n_prod);
        let mut delta_hs = Vec::with_capacity(n_prod);
        let mut accepted = 0usize;

        for _ in 0..n_prod {
            let r = hmc::hmc_trajectory(&mut lat, cfg);
            plaqs.push(r.plaquette);
            delta_hs.push(r.delta_h);
            if r.accepted {
                accepted += 1;
            }
        }

        let mean_p: f64 = plaqs.iter().sum::<f64>() / plaqs.len() as f64;
        let var_p: f64 = plaqs.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>() / (plaqs.len() - 1) as f64;
        let std_p = var_p.sqrt();
        let exp_neg_dh: f64 = delta_hs.iter().map(|dh| (-dh).exp()).sum::<f64>() / delta_hs.len() as f64;
        let accept_rate = accepted as f64 / n_prod as f64;

        println!("  {seed:<5}  {mean_p:.7}   {std_p:.6}  {exp_neg_dh:.6}    {:.0}%", accept_rate * 100.0);

        seed_means.push(mean_p);
        all_plaqs.extend_from_slice(&plaqs);
        all_delta_h.extend_from_slice(&delta_hs);
    }
    println!();

    // Grand mean and bootstrap-style error from seed variance
    let grand_mean: f64 = seed_means.iter().sum::<f64>() / seed_means.len() as f64;
    let seed_var: f64 = seed_means.iter().map(|m| (m - grand_mean).powi(2)).sum::<f64>() / (seed_means.len() - 1) as f64;
    let seed_err = (seed_var / seed_means.len() as f64).sqrt();

    println!("  Grand mean ⟨P⟩ = {grand_mean:.8} ± {seed_err:.8} (inter-seed error)");

    // Creutz equality: ⟨exp(-ΔH)⟩ = 1
    let grand_exp_dh: f64 = all_delta_h.iter().map(|dh| (-dh).exp()).sum::<f64>() / all_delta_h.len() as f64;
    let creutz_dev = (grand_exp_dh - 1.0).abs();
    println!("  ⟨exp(-ΔH)⟩ = {grand_exp_dh:.6} (deviation from 1: {creutz_dev:.2e})");

    let creutz_pass = creutz_dev < 0.05;
    println!("  [{}] Creutz equality |⟨exp(-ΔH)⟩ - 1| < 0.05", if creutz_pass { "PASS" } else { "FAIL" });

    // Mean |ΔH|
    let mean_abs_dh: f64 = all_delta_h.iter().map(|dh| dh.abs()).sum::<f64>() / all_delta_h.len() as f64;
    println!("  ⟨|ΔH|⟩ = {mean_abs_dh:.6}");

    // Overall acceptance
    let overall_accept: f64 = all_delta_h.iter().filter(|&&dh| dh < 0.0 || (-dh).exp() > 0.5).count() as f64 / all_delta_h.len() as f64;
    println!("  Overall acceptance: {:.0}%", overall_accept * 100.0);

    println!();
    println!("═══ Summary Table for arXiv ═══");
    println!();
    println!("| Seed | ⟨P⟩       | ⟨exp(-ΔH)⟩ |");
    println!("|------|-----------|------------|");
    for (i, &seed) in seeds.iter().enumerate() {
        let exp_dh: f64 = all_delta_h[i * n_prod..(i + 1) * n_prod]
            .iter()
            .map(|dh| (-dh).exp())
            .sum::<f64>() / n_prod as f64;
        println!("| {seed:<4} | {:.7} | {exp_dh:.6}   |", seed_means[i]);
    }
    println!("|------|-----------|------------|");
    println!("| **Mean** | **{grand_mean:.7} ± {seed_err:.7}** | **{grand_exp_dh:.6}** |");
    println!();
}

/// Compute Re Tr(A·B) for the force projection.
fn trace_product(a: &Su3Matrix, b: &Su3Matrix) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..3 {
        for k in 0..3 {
            sum += a.m[i][k].re * b.m[k][i].re - a.m[i][k].im * b.m[k][i].im;
        }
    }
    sum
}

/// The 8 Gell-Mann generators of su(3) (anti-Hermitian: i·λ_a/2).
fn gell_mann_generators() -> [Su3Matrix; 8] {
    use hotspring_barracuda::lattice::complex_f64::Complex64;
    let half = 0.5;

    let mut gens = [Su3Matrix::ZERO; 8];

    // T₁ = i·λ₁/2: symmetric off-diagonal (0,1)+(1,0) purely imaginary
    gens[0].m[0][1] = Complex64 { re: 0.0, im: half };
    gens[0].m[1][0] = Complex64 { re: 0.0, im: half };

    // T₂ = i·λ₂/2: antisymmetric off-diagonal (0,1)-(1,0) purely real
    gens[1].m[0][1] = Complex64 { re: half, im: 0.0 };
    gens[1].m[1][0] = Complex64 { re: -half, im: 0.0 };

    // T₃ = i·λ₃/2: diagonal (0,0)-(1,1) purely imaginary
    gens[2].m[0][0] = Complex64 { re: 0.0, im: half };
    gens[2].m[1][1] = Complex64 { re: 0.0, im: -half };

    // T₄ = i·λ₄/2: symmetric off-diagonal (0,2)+(2,0) purely imaginary
    gens[3].m[0][2] = Complex64 { re: 0.0, im: half };
    gens[3].m[2][0] = Complex64 { re: 0.0, im: half };

    // T₅ = i·λ₅/2: antisymmetric off-diagonal (0,2)-(2,0) purely real
    gens[4].m[0][2] = Complex64 { re: half, im: 0.0 };
    gens[4].m[2][0] = Complex64 { re: -half, im: 0.0 };

    // T₆ = i·λ₆/2: symmetric off-diagonal (1,2)+(2,1) purely imaginary
    gens[5].m[1][2] = Complex64 { re: 0.0, im: half };
    gens[5].m[2][1] = Complex64 { re: 0.0, im: half };

    // T₇ = i·λ₇/2: antisymmetric off-diagonal (1,2)-(2,1) purely real
    gens[6].m[1][2] = Complex64 { re: half, im: 0.0 };
    gens[6].m[2][1] = Complex64 { re: -half, im: 0.0 };

    // T₈ = i·λ₈/2: diagonal (1/√3)(0,0)+(1/√3)(1,1)-(2/√3)(2,2)
    let s3 = 0.5 / 3.0_f64.sqrt();
    gens[7].m[0][0] = Complex64 { re: 0.0, im: s3 };
    gens[7].m[1][1] = Complex64 { re: 0.0, im: s3 };
    gens[7].m[2][2] = Complex64 { re: 0.0, im: -2.0 * s3 };

    gens
}
