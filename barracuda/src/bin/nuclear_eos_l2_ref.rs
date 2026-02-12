//! Nuclear EOS Level 2 — Reference Implementation
//!
//! KEY INSIGHT: L2 (HFB) landscape is extremely rugged in 10D.
//! Random starting points almost never reach physically meaningful regions.
//! The penalty for NMP-invalid regions (9.21) is LOWER than actual HFB
//! values for bad parameters (~11), creating a "penalty attractor" that
//! traps NM in unphysical space.
//!
//! SOLUTION: L1-seeded multi-start NM
//!   1. Generate cheap L1 (SEMF) evaluations to find physically promising regions
//!   2. Take top-K L1 solutions as NM starting points for L2
//!   3. Run NM on the TRUE L2 objective with HIGH penalty (1e10)
//!   4. L1 seeds ensure NM starts in NMP-valid regions
//!   5. High penalty prevents NM from wandering into unphysical space
//!
//! This is the SPECIFICATION for BarraCUDA's L2 pipeline evolution:
//! `barracuda::sample::sparsity` needs a `with_warm_start(seeds)` API
//! that accepts pre-computed starting points from cheaper proxy models.
//!
//! Run: cargo run --release --bin nuclear_eos_l2_ref [--l1-samples=N] [--nm-starts=K] [--evals=M]

use hotspring_barracuda::data;
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::physics::hfb::binding_energy_l2;
use hotspring_barracuda::surrogate::{
    loo_cv_optimal_smoothing, filter_training_data, PenaltyFilter,
};
use hotspring_barracuda::stats::{chi2_decomposed, bootstrap_ci, convergence_diagnostics};

use barracuda::surrogate::{RBFKernel, RBFSurrogate};
use barracuda::sample::latin_hypercube;
use barracuda::optimize::EvaluationCache;
use rayon::prelude::*;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — Reference Implementation                 ║");
    println!("║  L1-Seeded Multi-Start NM on TRUE HFB Objective            ║");
    println!("║  SPECIFICATION for BarraCUDA with_warm_start() API         ║");
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

    // Top-K seeds for L2 NM
    let seeds: Vec<Vec<f64>> = indices[..nm_starts.min(indices.len())]
        .iter()
        .map(|&i| l1_samples[i].clone())
        .collect();

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2: L2 multi-start NM from L1 seeds
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PHASE 2: L2 Multi-Start NM ({} starts × {} evals)", nm_starts, evals_per_start);
    println!("═══════════════════════════════════════════════════════════════");

    let t2 = Instant::now();
    let mut global_best_f = f64::INFINITY;
    let mut global_best_x = vec![0.0; bounds.len()];
    let mut all_l2_evals = 0usize;
    let mut round_history: Vec<f64> = Vec::new();

    // Cache for surrogate monitoring
    let mut cache = EvaluationCache::with_capacity(nm_starts * evals_per_start);

    for (i, x0) in seeds.iter().enumerate() {
        let (x_best, f_best, n_evals) = barracuda::optimize::nelder_mead(
            |x: &[f64]| l2_objective(x, &nuclei),
            x0,
            &bounds,
            evals_per_start,
            1e-8,
        ).expect("Nelder-Mead failed");

        cache.record(x_best.clone(), f_best);
        all_l2_evals += n_evals;

        if f_best < global_best_f {
            global_best_f = f_best;
            global_best_x = x_best;
        }

        round_history.push(global_best_f);

        // Progress reporting every 5 starts
        if (i + 1) % 5 == 0 || i + 1 == seeds.len() {
            let elapsed = t2.elapsed().as_secs_f64();
            println!("  [{:3}/{}] best log(1+χ²) = {:.4}, L2 evals = {}, elapsed = {:.0}s",
                i + 1, seeds.len(), global_best_f, all_l2_evals, elapsed);
        }
    }

    let l2_time = t2.elapsed().as_secs_f64();
    let chi2 = global_best_f.exp() - 1.0;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  L2 Reference Results                                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  χ²/datum:       {:12.4}                              ║", chi2);
    println!("║  log(1+χ²):      {:12.4}                              ║", global_best_f);
    println!("║  L2 evals:       {:6}                                    ║", all_l2_evals);
    println!("║  L2 time:        {:6.1}s                                   ║", l2_time);
    println!("║  HFB throughput: {:6.1} evals/s                            ║",
        all_l2_evals as f64 / l2_time);
    println!("║  Total time:     {:6.1}s  (L1 screen + L2 NM)             ║", t_total.elapsed().as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════════╝");

    // ═══════════════════════════════════════════════════════════════
    // PHASE 3: Surrogate quality monitoring on L2 data
    // ═══════════════════════════════════════════════════════════════
    let (x_all, y_all) = cache.training_data();
    let (x_filt, y_filt) = filter_training_data(&x_all, &y_all, PenaltyFilter::Threshold(12.0));

    println!();
    println!("  Surrogate monitoring:");
    println!("    Total cached:  {}", x_all.len());
    println!("    After filter:  {} (threshold=12.0)", x_filt.len());

    if x_filt.len() >= 15 {
        let (opt_s, opt_rmse, _) = loo_cv_optimal_smoothing(
            &x_filt, &y_filt, RBFKernel::ThinPlateSpline, None,
        );
        println!("    LOO-CV RMSE:   {:.4} (smoothing={:.1e})", opt_rmse, opt_s);
    }

    // Convergence diagnostics
    let diag = convergence_diagnostics(&round_history);
    println!();
    println!("  Convergence: improving={}/{}, rate={:.4}",
        diag.n_improving, round_history.len().saturating_sub(1), diag.rate);

    // ═══════════════════════════════════════════════════════════════
    // PHASE 4: Statistical analysis at best point
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STATISTICAL ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════");

    let (observed, expected, sigma) = compute_l2_binding_energies(&global_best_x, &nuclei);

    if !observed.is_empty() {
        let chi2_result = chi2_decomposed(&observed, &expected, &sigma);
        println!();
        chi2_result.print_summary("L2 HFB");

        let per_datum: Vec<f64> = chi2_result.contributions.iter()
            .map(|c| c.chi2).collect();
        if per_datum.len() >= 5 {
            println!();
            let ci = bootstrap_ci(
                &per_datum,
                |d| d.iter().sum::<f64>() / d.len() as f64,
                5000, 0.95, 42,
            );
            ci.print_summary("χ²/datum");
        }
    }

    // Nuclear matter properties
    if let Some(nmp) = nuclear_matter_properties(&global_best_x) {
        println!();
        println!("  Nuclear matter properties:");
        println!("    ρ₀   = {:.4} fm⁻³  (exp: 0.16 ± 0.01)", nmp.rho0_fm3);
        println!("    E/A  = {:.2} MeV    (exp: -15.97 ± 0.2)", nmp.e_a_mev);
        println!("    K∞   = {:.1} MeV    (exp: 230 ± 30)", nmp.k_inf_mev);
        println!("    m*/m = {:.3}        (exp: 0.69 ± 0.1)", nmp.m_eff_ratio);
        println!("    J    = {:.1} MeV    (exp: 32 ± 2)", nmp.j_mev);
    }

    // Evaluation distribution
    let mut sorted_y: Vec<f64> = y_all.clone();
    sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_total = sorted_y.len();
    let n_penalty = sorted_y.iter().filter(|&&v| v > 9.0).count();
    let n_good_l2 = sorted_y.iter().filter(|&&v| v < 5.0).count();

    println!();
    println!("  Evaluation distribution (log(1+χ²)):");
    println!("    Total:      {}", n_total);
    if n_total > 0 {
        println!("    Best:       {:.4}", sorted_y[0]);
        println!("    p10:        {:.4}", sorted_y[n_total / 10]);
        println!("    Median:     {:.4}", sorted_y[n_total / 2]);
        println!("    Penalty (>9): {} ({:.1}%)", n_penalty, 100.0 * n_penalty as f64 / n_total as f64);
        println!("    Good (<5):  {} ({:.1}%)", n_good_l2, 100.0 * n_good_l2 as f64 / n_total as f64);
    }

    // ═══════════════════════════════════════════════════════════════
    // COMPARISON
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("  {:40} {:>10} {:>8} {:>8}", "Method", "χ²/datum", "Evals", "Time");
    println!("  {:40} {:>10} {:>8} {:>8}", "─".repeat(40), "─".repeat(10), "─".repeat(8), "─".repeat(8));
    println!("  {:40} {:>10.2} {:>8} {:>6.0}s",
        "Reference (L1-seeded NM)", chi2, all_l2_evals, l2_time);
    println!("  {:40} {:>10.2} {:>8} {:>8}",
        "Python/scipy control", 61.87, 96, "~600s");
    println!("  {:40} {:>10.2} {:>8} {:>8}",
        "Prev BarraCUDA (old binary)", 25.43, 1009, "2091s");

    // Save results
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let result_json = serde_json::json!({
        "level": 2,
        "engine": "barracuda::reference_l2_l1_seeded",
        "chi2_per_datum": chi2,
        "log_chi2": global_best_f,
        "l1_screening_samples": n_l1,
        "nm_starts": nm_starts,
        "evals_per_start": evals_per_start,
        "total_l2_evals": all_l2_evals,
        "l2_time_seconds": l2_time,
        "total_time_seconds": t_total.elapsed().as_secs_f64(),
        "best_params": global_best_x,
        "convergence": {
            "improving": diag.n_improving,
            "stagnant": diag.n_stagnant,
        },
        "python_reference": { "chi2_per_datum": 61.87, "evals": 96 },
        "prev_barracuda": { "chi2_per_datum": 25.43, "evals": 1009 },
    });
    let path = results_dir.join("barracuda_l2_reference.json");
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
// rather than converging to the penalty boundary.
// ═══════════════════════════════════════════════════════════════════

fn l2_objective(params: &[f64], nuclei: &[(usize, usize, f64)]) -> f64 {
    // Hard boundary: alpha must be positive
    if params[8] <= 0.01 || params[8] > 1.0 {
        return (1e10_f64).ln_1p(); // ~23 — way above any HFB value
    }

    // NMP check — very high penalty to REPEL NM from unphysical regions
    let nmp = match nuclear_matter_properties(params) {
        Some(n) => n,
        None => return (1e10_f64).ln_1p(), // ~23
    };

    // Soft penalties for NMP ranges (keep as gradient signals)
    let mut penalty = 0.0;
    if nmp.rho0_fm3 < 0.08 { penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08; }
    if nmp.rho0_fm3 > 0.25 { penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25; }
    if nmp.e_a_mev > -5.0 { penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0); }

    // Parallel HFB evaluation across nuclei
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
