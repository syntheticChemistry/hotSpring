//! Nuclear EOS Level 1 — Reference Implementation with auto-smoothing
//!
//! Demonstrates the COMPLETE corrected workflow:
//!   1. LOO-CV auto-smoothing (vs hardcoded 1e-12)
//!   2. Penalty-filtered surrogate training
//!   3. Chi-squared decomposition with per-nucleus analysis
//!   4. Bootstrap confidence intervals on χ²/datum
//!   5. Convergence diagnostics with early stopping
//!
//! This is the SPECIFICATION for BarraCUDA's SparsitySampler evolution.
//! Run: cargo run --release --bin nuclear_eos_l1_ref

use hotspring_barracuda::data;
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::surrogate::{
    loo_cv_optimal_smoothing, filter_training_data, round_based_direct_optimization,
    DirectSamplerConfig, PenaltyFilter, adaptive_penalty,
};
use hotspring_barracuda::stats::{chi2_decomposed, bootstrap_ci, convergence_diagnostics};

use barracuda::surrogate::{RBFKernel, RBFSurrogate};
use barracuda::sample::sparsity::{sparsity_sampler, SparsitySamplerConfig};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L1 — Reference Implementation                 ║");
    println!("║  Auto-Smoothing + Penalty Filtering + Chi² Analysis        ║");
    println!("║  SPECIFICATION for BarraCUDA evolution                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

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

    println!("  Experimental nuclei: {}", exp_data.len());
    println!("  Parameters:          {} dimensions", bounds.len());
    println!();

    // ── Define L1 objective ─────────────────────────────────────────
    let exp_data_obj = exp_data.clone();
    let objective = move |x: &[f64]| -> f64 {
        l1_objective(x, &exp_data_obj)
    };

    // ═══════════════════════════════════════════════════════════════
    // APPROACH 1: SparsitySampler with auto-smoothing
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  APPROACH 1: SparsitySampler + LOO-CV Auto-Smoothing");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let t0 = Instant::now();

    // Step 1: Quick run with default smoothing to get initial data
    let config = SparsitySamplerConfig::new(bounds.len(), 42)
        .with_initial_samples(100)
        .with_solvers(4)
        .with_eval_budget(25)
        .with_iterations(3);

    let quick_result = sparsity_sampler(&objective, &bounds, &config)
        .expect("SparsitySampler failed");

    println!("  Quick scan: {} evals, best f = {:.4}", quick_result.cache.len(), quick_result.f_best);

    // Step 2: LOO-CV auto-smoothing on collected data
    let (x_all, y_all) = quick_result.cache.training_data();

    // Filter penalties before LOO-CV
    let (x_filt, y_filt) = filter_training_data(&x_all, &y_all, PenaltyFilter::AdaptiveMAD(5.0));
    println!("  Filtered: {}/{} points for LOO-CV (removed {} penalties)",
        x_filt.len(), x_all.len(), x_all.len() - x_filt.len());

    println!();
    println!("  Running LOO-CV auto-smoothing grid search...");
    let (opt_smoothing, opt_rmse, cv_results) = loo_cv_optimal_smoothing(
        &x_filt, &y_filt, RBFKernel::ThinPlateSpline, None,
    );
    println!("  LOO-CV results:");
    for (s, rmse) in &cv_results {
        let marker = if (*s - opt_smoothing).abs() < 1e-15 { " ← OPTIMAL" } else { "" };
        println!("    s={:.2e} → RMSE={:.4}{}", s, rmse, marker);
    }
    println!("  Optimal smoothing: {:.2e} (RMSE={:.4})", opt_smoothing, opt_rmse);
    println!();

    // Step 3: Re-run with optimal smoothing
    let config_optimal = SparsitySamplerConfig::new(bounds.len(), 42)
        .with_initial_samples(100)
        .with_solvers(8)
        .with_eval_budget(50)
        .with_iterations(5);
    // Note: we manually set smoothing since with_smoothing may not be available
    let mut config_manual = config_optimal;
    config_manual.smoothing = opt_smoothing;

    let final_result = sparsity_sampler(&objective, &bounds, &config_manual)
        .expect("SparsitySampler failed");

    let approach1_time = t0.elapsed().as_secs_f64();
    let approach1_f = final_result.f_best;
    let approach1_x = final_result.x_best.clone();
    let approach1_evals = final_result.cache.len();

    println!("  SparsitySampler (auto-smoothed): {} evals in {:.2}s",
        approach1_evals, approach1_time);
    let chi2_1 = approach1_f.exp() - 1.0;
    println!("  χ²/datum = {:.4}, log(1+χ²) = {:.4}", chi2_1, approach1_f);

    // ═══════════════════════════════════════════════════════════════
    // APPROACH 2: Round-based direct NM (reference implementation)
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  APPROACH 2: Round-Based Direct NM + Auto-Smoothing Monitor");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let exp_data_obj2 = exp_data.clone();
    let objective2 = move |x: &[f64]| -> f64 {
        l1_objective(x, &exp_data_obj2)
    };

    let direct_config = DirectSamplerConfig::new(bounds.len(), 42)
        .with_rounds(8)
        .with_evals_per_round(120)
        .with_solvers(8)
        .with_patience(3)
        .with_auto_smoothing(true)
        .with_filter(PenaltyFilter::Threshold(8.0));

    let direct_result = round_based_direct_optimization(objective2, &bounds, &direct_config);

    let approach2_f = direct_result.f_best;
    let approach2_x = direct_result.x_best.clone();
    let approach2_evals = direct_result.cache.len();
    let approach2_time = direct_result.total_time;
    let chi2_2 = approach2_f.exp() - 1.0;

    println!();
    println!("  Round-Based NM: {} evals in {:.2}s", approach2_evals, approach2_time);
    println!("  χ²/datum = {:.4}, log(1+χ²) = {:.4}", chi2_2, approach2_f);

    // Convergence diagnostics
    let history: Vec<f64> = direct_result.round_results.iter().map(|r| r.best_f).collect();
    let diag = convergence_diagnostics(&history);
    println!();
    println!("  Convergence: converged={}, improving={}/{}, stagnant={}",
        diag.is_converged, diag.n_improving, history.len().saturating_sub(1), diag.n_stagnant);

    // ═══════════════════════════════════════════════════════════════
    // STATISTICAL ANALYSIS of best result
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STATISTICAL ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Pick the better result
    let (best_x, best_label) = if approach1_f <= approach2_f {
        (&approach1_x, "SparsitySampler+AutoSmooth")
    } else {
        (&approach2_x, "Round-Based NM")
    };
    println!("  Best approach: {}", best_label);

    // Compute per-nucleus chi-squared decomposition
    let (observed, expected, sigma) = compute_binding_energies(best_x, &exp_data);
    let chi2_result = chi2_decomposed(&observed, &expected, &sigma);
    chi2_result.print_summary("L1 SEMF");

    // Bootstrap CI on chi-squared per datum
    let per_datum_chi2s: Vec<f64> = chi2_result.contributions.iter()
        .map(|c| c.chi2)
        .collect();
    println!();
    let ci = bootstrap_ci(
        &per_datum_chi2s,
        |d| d.iter().sum::<f64>() / d.len() as f64,
        5000, 0.95, 42,
    );
    ci.print_summary("χ²/datum");

    // Nuclear matter properties
    if let Some(nmp) = nuclear_matter_properties(best_x) {
        println!();
        println!("  Nuclear matter properties:");
        println!("    ρ₀   = {:.4} fm⁻³  (exp: 0.16 ± 0.01)", nmp.rho0_fm3);
        println!("    E/A  = {:.2} MeV    (exp: -15.97 ± 0.2)", nmp.e_a_mev);
        println!("    K∞   = {:.1} MeV    (exp: 230 ± 30)", nmp.k_inf_mev);
        println!("    m*/m = {:.3}        (exp: 0.69 ± 0.1)", nmp.m_eff_ratio);
        println!("    J    = {:.1} MeV    (exp: 32 ± 2)", nmp.j_mev);
    }

    // ═══════════════════════════════════════════════════════════════
    // COMPARISON SUMMARY
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  {:40} {:>10} {:>8} {:>8}", "Method", "χ²/datum", "Evals", "Time");
    println!("  {:40} {:>10} {:>8} {:>8}", "─".repeat(40), "─".repeat(10), "─".repeat(8), "─".repeat(8));
    println!("  {:40} {:>10.4} {:>8} {:>7.1}s",
        "SparsitySampler + AutoSmooth", chi2_1, approach1_evals, approach1_time);
    println!("  {:40} {:>10.4} {:>8} {:>7.1}s",
        "Round-Based Direct NM", chi2_2, approach2_evals, approach2_time);
    println!("  {:40} {:>10.4} {:>8} {:>8}",
        "Python/scipy control (reference)", 6.62, 1008, "~180s");
    println!();

    let better = chi2_1.min(chi2_2);
    if better < 6.62 {
        println!("  ✅ BarraCUDA BEATS Python by {:.1}% with {}× fewer evals",
            100.0 * (6.62 - better) / 6.62,
            1008 / approach1_evals.min(approach2_evals).max(1));
    } else {
        println!("  ⚠ BarraCUDA behind Python by {:.1}% — needs more tuning",
            100.0 * (better - 6.62) / 6.62);
    }

    // Save results
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let result_json = serde_json::json!({
        "level": 1,
        "engine": "barracuda::reference_l1",
        "approach1_sparsity_autosmooth": {
            "chi2_per_datum": chi2_1,
            "log_chi2": approach1_f,
            "total_evals": approach1_evals,
            "time_seconds": approach1_time,
            "optimal_smoothing": opt_smoothing,
            "loo_cv_rmse": opt_rmse,
        },
        "approach2_round_based_nm": {
            "chi2_per_datum": chi2_2,
            "log_chi2": approach2_f,
            "total_evals": approach2_evals,
            "time_seconds": approach2_time,
            "convergence": {
                "converged": diag.is_converged,
                "n_improving": diag.n_improving,
                "n_stagnant": diag.n_stagnant,
            },
        },
        "statistical_analysis": {
            "chi2_total": chi2_result.total_chi2,
            "chi2_per_datum": chi2_result.chi2_per_datum,
            "reduced_chi2": chi2_result.reduced_chi2(),
            "p_value": chi2_result.p_value(),
            "bootstrap_ci_95": [ci.lower, ci.upper],
            "bootstrap_se": ci.std_error,
        },
        "python_reference": {
            "chi2_per_datum": 6.62,
            "total_evals": 1008,
        },
    });
    let path = results_dir.join("barracuda_l1_reference.json");
    std::fs::write(&path, serde_json::to_string_pretty(&result_json).unwrap()).ok();
    println!("\n  Results saved to: {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════
// L1 objective (SEMF)
// ═══════════════════════════════════════════════════════════════════

fn l1_objective(x: &[f64], exp_data: &std::collections::HashMap<(usize, usize), (f64, f64)>) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    // NMP check
    let nmp = match nuclear_matter_properties(x) {
        Some(n) => n,
        None => return (1e4_f64).ln_1p(),
    };

    // Soft penalties for unphysical ranges
    let mut penalty = 0.0;
    if nmp.rho0_fm3 < 0.08 { penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08; }
    if nmp.rho0_fm3 > 0.25 { penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25; }
    if nmp.e_a_mev > -5.0 { penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0); }

    let mut chi2 = 0.0;
    let mut n = 0;
    for (&(z, nn), &(b_exp, _sigma)) in exp_data.iter() {
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

/// Compute per-nucleus binding energies at given parameters.
fn compute_binding_energies(
    params: &[f64],
    exp_data: &std::collections::HashMap<(usize, usize), (f64, f64)>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut observed = Vec::new();
    let mut expected = Vec::new();
    let mut sigma = Vec::new();

    for (&(z, n), &(b_exp, _)) in exp_data.iter() {
        let b_calc = semf_binding_energy(z, n, params);
        if b_calc > 0.0 {
            observed.push(b_calc);
            expected.push(b_exp);
            sigma.push((0.01 * b_exp).max(2.0));
        }
    }

    (observed, expected, sigma)
}


