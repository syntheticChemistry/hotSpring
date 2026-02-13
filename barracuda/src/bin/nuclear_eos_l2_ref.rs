//! Nuclear EOS Level 2 — BarraCUDA Native Revalidation
//!
//! Now uses BarraCUDA's native implementations evolved from our hotSpring specs:
//!   1. barracuda::sample::direct::direct_sampler — round-based NM with warm-start
//!   2. barracuda::surrogate::loo_cv_optimal_smoothing — LOO-CV for monitoring
//!   3. barracuda::stats::chi2_decomposed_weighted — per-nucleus chi² analysis
//!   4. barracuda::stats::bootstrap_ci — confidence intervals
//!   5. barracuda::optimize::convergence_diagnostics — stagnation detection
//!
//! KEY INSIGHT: L2 (HFB) landscape is extremely rugged in 10D.
//! SOLUTION: L1-seeded DirectSampler with warm_start API.
//!
//! Run: cargo run --release --bin nuclear_eos_l2_ref [--l1-samples=N] [--nm-starts=K] [--evals=M]

use hotspring_barracuda::data;
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::physics::hfb::binding_energy_l2;

// ALL from barracuda native — no hotspring_barracuda::surrogate or ::stats
use barracuda::sample::direct::{direct_sampler, DirectSamplerConfig};
use barracuda::sample::latin_hypercube;
use barracuda::stats::{bootstrap_ci, chi2_decomposed_weighted};
use barracuda::optimize::convergence_diagnostics;
use rayon::prelude::*;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — BarraCUDA Native Revalidation            ║");
    println!("║  L1-Seeded DirectSampler with warm_start API               ║");
    println!("║  ALL math from barracuda:: (evolved from hotSpring specs)   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let args: Vec<String> = std::env::args().collect();
    let n_l1 = args.iter()
        .find(|a| a.starts_with("--l1-samples="))
        .and_then(|a| a.strip_prefix("--l1-samples=")?.parse().ok())
        .unwrap_or(5000);
    let nm_starts = args.iter()
        .find(|a| a.starts_with("--nm-starts="))
        .and_then(|a| a.strip_prefix("--nm-starts=")?.parse().ok())
        .unwrap_or(20);
    let evals_per_start = args.iter()
        .find(|a| a.starts_with("--evals="))
        .and_then(|a| a.strip_prefix("--evals=")?.parse().ok())
        .unwrap_or(200);
    let n_rounds = args.iter()
        .find(|a| a.starts_with("--rounds="))
        .and_then(|a| a.strip_prefix("--rounds=")?.parse().ok())
        .unwrap_or(3);

    // ── Load data ──────────────────────────────────────────────────
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("control/surrogate/nuclear-eos");

    let exp_data = Arc::new(
        data::load_experimental_data(&base.join("exp_data/ame2020_selected.json"))
            .expect("Failed to load experimental data"),
    );
    let bounds = data::load_bounds(&base.join("wrapper/skyrme_bounds.json"))
        .expect("Failed to load parameter bounds");

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    println!("  Experimental nuclei: {}", nuclei.len());
    println!("  Parameters:          {} dimensions", bounds.len());
    println!("  L1 screening:        {} samples", n_l1);
    println!("  NM starts:           {} (from top-K L1 solutions)", nm_starts);
    println!("  Evals per start:     {}", evals_per_start);
    println!("  DirectSampler rounds: {}", n_rounds);
    println!("  Rayon threads:       {}", rayon::current_num_threads());
    println!();

    let t_total = Instant::now();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 1: L1 (SEMF) screening — find promising starting regions
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PHASE 1: L1 Screening ({} SEMF evaluations)", n_l1);
    println!("═══════════════════════════════════════════════════════════════");

    let t1 = Instant::now();
    let l1_samples = latin_hypercube(n_l1, &bounds, 42).expect("LHS failed");

    let l1_scores: Vec<f64> = l1_samples.iter()
        .map(|x| l1_objective(x, &exp_data))
        .collect();

    // Sort by L1 score
    let mut indices: Vec<usize> = (0..l1_scores.len()).collect();
    indices.sort_by(|&a, &b| l1_scores[a].partial_cmp(&l1_scores[b]).unwrap());

    let n_nmp_valid = l1_scores.iter().filter(|&&s| s < 9.0).count();
    let n_good = l1_scores.iter().filter(|&&s| s < 5.0).count();

    println!("  L1 screening: {:.2}s", t1.elapsed().as_secs_f64());
    println!("  NMP-valid:    {}/{} ({:.1}%)", n_nmp_valid, n_l1, 100.0 * n_nmp_valid as f64 / n_l1 as f64);
    println!("  Good (<5):    {}/{} ({:.1}%)", n_good, n_l1, 100.0 * n_good as f64 / n_l1 as f64);
    println!("  Best L1:      log(1+χ²) = {:.4}", l1_scores[indices[0]]);
    println!("  Rank-{} L1:  log(1+χ²) = {:.4}", nm_starts, l1_scores[indices[nm_starts.min(l1_scores.len()) - 1]]);
    println!();

    // Top-K seeds for L2 — use barracuda::sample::direct with warm_start
    let seeds: Vec<Vec<f64>> = indices[..nm_starts.min(indices.len())]
        .iter()
        .map(|&i| l1_samples[i].clone())
        .collect();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: Native DirectSampler with L1 warm-start
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PHASE 2: Native DirectSampler (L1 warm-start → L2 NM)");
    println!("═══════════════════════════════════════════════════════════════");

    let t2 = Instant::now();

    let direct_config = DirectSamplerConfig::new(42)
        .with_rounds(n_rounds)
        .with_solvers(nm_starts)
        .with_eval_budget(evals_per_start)
        .with_patience(2)
        .with_warm_start(seeds.clone());

    println!("  Config: {} rounds, {} solvers, {} evals/solver, {} warm-start seeds",
        direct_config.n_rounds, direct_config.n_solvers,
        direct_config.max_eval_per_solver, direct_config.warm_start_seeds.len());
    println!("  Auto-smoothing: {} (monitoring)", direct_config.auto_smoothing);
    println!();

    // NOTE: DirectSampler runs NM on the objective. Since L2 is expensive,
    // each call to f(x) triggers parallel HFB across nuclei.
    let nuclei_clone = nuclei.clone();
    let l2_obj = move |x: &[f64]| -> f64 {
        l2_objective(x, &nuclei_clone)
    };

    let result = direct_sampler(l2_obj, &bounds, &direct_config)
        .expect("DirectSampler failed");

    let l2_time = t2.elapsed().as_secs_f64();
    let chi2 = result.f_best.exp() - 1.0;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  L2 Native DirectSampler Results                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  χ²/datum:       {:12.4}                              ║", chi2);
    println!("║  log(1+χ²):      {:12.6}                            ║", result.f_best);
    println!("║  L2 evals:       {:6}                                    ║", result.cache.len());
    println!("║  L2 time:        {:6.1}s                                   ║", l2_time);
    println!("║  HFB throughput: {:6.1} evals/s                            ║",
        result.cache.len() as f64 / l2_time);
    println!("║  Early stopped:  {}                                         ║", result.early_stopped);
    println!("║  Total time:     {:6.1}s (L1 screen + L2 NM)              ║", t_total.elapsed().as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Round-by-round diagnostics
    if !result.rounds.is_empty() {
        println!();
        println!("  Round-by-round:");
        for r in &result.rounds {
            let rmse_str = r.surrogate_rmse
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "n/a".to_string());
            println!("    Round {}: best_f={:.6}, evals={}, surrogate_rmse={}, Δ={:.2e}",
                r.round, r.best_f, r.n_evals, rmse_str, r.improvement);
        }
    }

    // Convergence diagnostics (native)
    let history: Vec<f64> = result.rounds.iter().map(|r| r.best_f).collect();
    if history.len() >= 2 {
        match convergence_diagnostics(&history, 5, 0.01, 2) {
            Ok(diag) => {
                println!();
                println!("  {}", diag.summary());
            }
            Err(e) => println!("  Convergence analysis failed: {}", e),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: Statistical analysis at best point (native barracuda::stats)
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STATISTICAL ANALYSIS (barracuda::stats native)");
    println!("═══════════════════════════════════════════════════════════════");

    let (observed, expected, sigma) = compute_l2_binding_energies(&result.x_best, &nuclei);

    if !observed.is_empty() {
        match chi2_decomposed_weighted(&observed, &expected, &sigma, bounds.len()) {
            Ok(chi2_result) => {
                println!();
                println!("{}", chi2_result.summary());

                // Top 5 worst nuclei
                let worst = chi2_result.worst_n(5);
                println!();
                println!("  Top 5 worst-fitting nuclei (by pull):");
                for &idx in &worst {
                    println!("    [{}] pull={:.2}σ, χ²_i={:.2}, residual={:.2} MeV",
                        idx, chi2_result.pulls[idx], chi2_result.contributions[idx],
                        chi2_result.residuals[idx]);
                }

                // Bootstrap CI
                let per_datum: Vec<f64> = chi2_result.contributions.clone();
                if per_datum.len() >= 5 {
                    match bootstrap_ci(
                        &per_datum,
                        |d| d.iter().sum::<f64>() / d.len() as f64,
                        5000, 0.95, 42,
                    ) {
                        Ok(ci) => {
                            println!();
                            println!("  Bootstrap 95% CI on χ²/datum: {}", ci.summary());
                        }
                        Err(e) => println!("  Bootstrap failed: {}", e),
                    }
                }
            }
            Err(e) => println!("  chi2_decomposed_weighted failed: {}", e),
        }
    }

    // Nuclear matter properties
    if let Some(nmp) = nuclear_matter_properties(&result.x_best) {
        println!();
        println!("  Nuclear matter properties:");
        println!("    ρ₀   = {:.4} fm⁻³  (exp: 0.16 ± 0.01)", nmp.rho0_fm3);
        println!("    E/A  = {:.2} MeV    (exp: -15.97 ± 0.2)", nmp.e_a_mev);
        println!("    K∞   = {:.1} MeV    (exp: 230 ± 30)", nmp.k_inf_mev);
        println!("    m*/m = {:.3}        (exp: 0.69 ± 0.1)", nmp.m_eff_ratio);
        println!("    J    = {:.1} MeV    (exp: 32 ± 2)", nmp.j_mev);
    }

    // Evaluation distribution
    let (_, y_all) = result.cache.training_data();
    let mut sorted_y = y_all.clone();
    sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_total = sorted_y.len();
    let n_penalty = sorted_y.iter().filter(|&&v| v > 9.0).count();
    let n_good_l2 = sorted_y.iter().filter(|&&v| v < 5.0).count();

    println!();
    println!("  Evaluation distribution (log(1+χ²)):");
    println!("    Total:        {}", n_total);
    if n_total > 0 {
        println!("    Best:         {:.4}", sorted_y[0]);
        println!("    p10:          {:.4}", sorted_y[n_total / 10]);
        println!("    Median:       {:.4}", sorted_y[n_total / 2]);
        println!("    Penalty (>9): {} ({:.1}%)", n_penalty, 100.0 * n_penalty as f64 / n_total as f64);
        println!("    Good (<5):    {} ({:.1}%)", n_good_l2, 100.0 * n_good_l2 as f64 / n_total as f64);
    }

    // ═══════════════════════════════════════════════════════════════
    // COMPARISON
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  {:40} {:>10} {:>8} {:>8}", "Method", "χ²/datum", "Evals", "Time");
    println!("  {:40} {:>10} {:>8} {:>8}", "─".repeat(40), "─".repeat(10), "─".repeat(8), "─".repeat(8));
    println!("  {:40} {:>10.2} {:>8} {:>6.0}s",
        "Native DirectSampler (L1 warm-start)", chi2, result.cache.len(), l2_time);
    println!("  {:40} {:>10.2} {:>8} {:>8}",
        "Prev BarraCUDA L2 ref (manual NM)", 28450.0, 4022, "743s");
    println!("  {:40} {:>10.2} {:>8} {:>8}",
        "Python/scipy control", 61.87, 96, "~600s");
    println!("  {:40} {:>10.2} {:>8} {:>8}",
        "Prev BarraCUDA (toadstool-orchestrated)", 25.43, 1009, "2091s");

    if chi2 < 61.87 {
        println!();
        println!("  ✅ BarraCUDA BEATS Python L2 by {:.1}%", 100.0 * (61.87 - chi2) / 61.87);
    }

    // Save results
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let result_json = serde_json::json!({
        "level": 2,
        "engine": "barracuda::native_direct_sampler_l2",
        "barracuda_version": "phase5_evolved",
        "chi2_per_datum": chi2,
        "log_chi2": result.f_best,
        "l1_screening_samples": n_l1,
        "nm_starts": nm_starts,
        "direct_sampler_rounds": n_rounds,
        "evals_per_start": evals_per_start,
        "total_l2_evals": result.cache.len(),
        "l2_time_seconds": l2_time,
        "total_time_seconds": t_total.elapsed().as_secs_f64(),
        "early_stopped": result.early_stopped,
        "best_params": result.x_best,
        "references": {
            "prev_barracuda_l2_ref": { "chi2_per_datum": 28450.0, "evals": 4022 },
            "python_scipy": { "chi2_per_datum": 61.87, "evals": 96 },
            "toadstool_orchestrated": { "chi2_per_datum": 25.43, "evals": 1009 },
        },
    });
    let path = results_dir.join("barracuda_l2_native_revalidation.json");
    std::fs::write(&path, serde_json::to_string_pretty(&result_json).unwrap()).ok();
    println!("\n  Results saved to: {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════
// L1 objective (SEMF) — cheap proxy for screening
// ═══════════════════════════════════════════════════════════════════

fn l1_objective(x: &[f64], exp_data: &HashMap<(usize, usize), (f64, f64)>) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 { return (1e4_f64).ln_1p(); }

    let nmp = match nuclear_matter_properties(x) {
        Some(n) => n,
        None => return (1e4_f64).ln_1p(),
    };

    let mut penalty = 0.0;
    if nmp.rho0_fm3 < 0.08 { penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08; }
    if nmp.rho0_fm3 > 0.25 { penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25; }
    if nmp.e_a_mev > -5.0 { penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0); }

    let mut chi2 = 0.0;
    let mut n = 0;
    for (&(z, nn), &(b_exp, _)) in exp_data.iter() {
        let b_calc = semf_binding_energy(z, nn, x);
        if b_calc > 0.0 {
            let sigma_theo = (0.01 * b_exp).max(2.0);
            chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
            n += 1;
        }
    }
    if n == 0 { return (1e4_f64).ln_1p(); }
    (chi2 / n as f64 + penalty).ln_1p()
}

// ═══════════════════════════════════════════════════════════════════
// L2 objective (HFB) — expensive, uses rayon for parallel nuclei
// Penalty is set VERY HIGH (1e10) so NM avoids unphysical regions
// ═══════════════════════════════════════════════════════════════════

fn l2_objective(params: &[f64], nuclei: &[(usize, usize, f64)]) -> f64 {
    if params[8] <= 0.01 || params[8] > 1.0 {
        return (1e10_f64).ln_1p();
    }

    let nmp = match nuclear_matter_properties(params) {
        Some(n) => n,
        None => return (1e10_f64).ln_1p(),
    };

    let mut penalty = 0.0;
    if nmp.rho0_fm3 < 0.08 { penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08; }
    if nmp.rho0_fm3 > 0.25 { penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25; }
    if nmp.e_a_mev > -5.0 { penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0); }

    let results: Vec<(f64, f64)> = nuclei.par_iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, _conv) = binding_energy_l2(z, n, params);
            (b_calc, b_exp)
        })
        .collect();

    let mut chi2 = 0.0;
    let mut n_valid = 0;
    for (b_calc, b_exp) in &results {
        if *b_calc > 0.0 {
            let sigma_theo = (0.01 * b_exp).max(2.0);
            chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
            n_valid += 1;
        }
    }

    if n_valid == 0 { return (1e10_f64).ln_1p(); }
    (chi2 / n_valid as f64 + penalty).ln_1p()
}

/// Compute per-nucleus L2 binding energies at best parameters.
fn compute_l2_binding_energies(
    params: &[f64],
    nuclei: &[(usize, usize, f64)],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let results: Vec<(f64, f64)> = nuclei.par_iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, _conv) = binding_energy_l2(z, n, params);
            (b_calc, b_exp)
        })
        .collect();

    let mut observed = Vec::new();
    let mut expected = Vec::new();
    let mut sigma = Vec::new();

    for (b_calc, b_exp) in results {
        if b_calc > 0.0 {
            observed.push(b_calc);
            expected.push(b_exp);
            sigma.push((0.01 * b_exp).max(2.0));
        }
    }

    (observed, expected, sigma)
}
