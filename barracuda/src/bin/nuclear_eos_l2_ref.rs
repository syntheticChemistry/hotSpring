// SPDX-License-Identifier: AGPL-3.0-only

//! Nuclear EOS Level 2 — NMP-Constrained HFB Pipeline (Evolved)
//!
//! Physics: p/n HFB + BCS + Coulomb(Poisson+Slater) + `T_eff` + CM correction
//! Math: 100% `BarraCUDA` native (`gradient_1d`, `eigh_f64`, brent, trapz)
//!
//! Uses:
//!   1. `barracuda::sample::direct::direct_sampler` — round-based NM with warm-start
//!   2. `barracuda::sample::sparsity` — surrogate-guided with `auto_smoothing`
//!   3. `barracuda::sample::latin_hypercube` — L1 screening
//!   4. `barracuda::stats::{chi2_decomposed_weighted`, `bootstrap_ci`}
//!   5. `barracuda::optimize::convergence_diagnostics`
//!
//! Run: cargo run --release --bin `nuclear_eos_l2_ref` [--lambda=0.1] [--seed=42]
//!      [--rounds=5] [--nm-starts=10] [--evals=100] [--patience=3] [--multi=3]

use hotspring_barracuda::data;
use hotspring_barracuda::physics::hfb::binding_energy_l2;
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

use barracuda::optimize::convergence_diagnostics;
use barracuda::sample::direct::{direct_sampler, DirectSamplerConfig};
use barracuda::sample::latin_hypercube;
use barracuda::sample::sparsity::{sparsity_sampler, PenaltyFilter, SparsitySamplerConfig};
use barracuda::stats::{bootstrap_ci, chi2_decomposed_weighted};
use rayon::prelude::*;

use std::collections::HashMap;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// CLI args
// ═══════════════════════════════════════════════════════════════════

struct CliArgs {
    seed: u64,
    lambda: f64,
    lambda_l1: f64,
    n_l1: usize,
    nm_starts: usize,
    evals_per_start: usize,
    n_rounds: usize,
    patience: usize,
    multi: usize,
    sparsity: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let get = |prefix: &str| -> Option<String> {
        args.iter()
            .find(|a| a.starts_with(prefix))
            .map(|a| a[prefix.len()..].to_string())
    };
    let has = |flag: &str| -> bool { args.iter().any(|a| a == flag) };

    let lambda = get("--lambda=").and_then(|s| s.parse().ok()).unwrap_or(0.1);
    CliArgs {
        seed: get("--seed=").and_then(|s| s.parse().ok()).unwrap_or(42),
        lambda,
        lambda_l1: get("--lambda-l1=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10.0),
        n_l1: get("--l1-samples=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000),
        nm_starts: get("--nm-starts=")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10),
        evals_per_start: get("--evals=").and_then(|s| s.parse().ok()).unwrap_or(100),
        n_rounds: get("--rounds=").and_then(|s| s.parse().ok()).unwrap_or(5),
        patience: get("--patience=").and_then(|s| s.parse().ok()).unwrap_or(3),
        multi: get("--multi=").and_then(|s| s.parse().ok()).unwrap_or(1),
        sparsity: has("--sparsity"),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    let cli = parse_args();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — NMP-Constrained HFB Pipeline (Evolved)   ║");
    println!("║  Physics: p/n HFB + BCS + Coulomb + T_eff + CM             ║");
    println!("║  Math: 100% BarraCUDA (gradient_1d, eigh_f64, brent)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  lambda(L2):      {}", cli.lambda);
    println!("  lambda(L1 seed): {}", cli.lambda_l1);
    println!("  seed:            {}", cli.seed);
    println!("  L1 screening:    {} samples", cli.n_l1);
    println!("  NM starts:       {} (from top-K L1)", cli.nm_starts);
    println!("  Evals per start: {}", cli.evals_per_start);
    println!("  Rounds:          {}", cli.n_rounds);
    println!("  Patience:        {}", cli.patience);
    println!("  Multi-seed:      {}", cli.multi);
    println!(
        "  SparsitySampler: {}",
        if cli.sparsity { "YES" } else { "no" }
    );
    println!("  Rayon threads:   {}", rayon::current_num_threads());
    println!();

    // ── Load data ──────────────────────────────────────────────────
    let ctx = data::load_eos_context().expect("Failed to load EOS context");
    let base = &ctx.base;
    let exp_data = ctx.exp_data.clone();
    let bounds = &ctx.bounds;

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    println!("  Experimental nuclei: {}", nuclei.len());
    println!("  Parameters:          {} dimensions", bounds.len());
    println!();

    let _t_total = Instant::now();

    // Track results across all seeds
    let mut all_results: Vec<SeedResult> = Vec::new();

    for seed_idx in 0..cli.multi {
        let current_seed = cli.seed + seed_idx as u64 * 1000;

        if cli.multi > 1 {
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!(
                "║  SEED RUN {}/{} (seed={})                                  ║",
                seed_idx + 1,
                cli.multi,
                current_seed
            );
            println!("╚══════════════════════════════════════════════════════════════╝");
            println!();
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: L1 screening with NMP bias
        // ═══════════════════════════════════════════════════════════════
        println!("═══════════════════════════════════════════════════════════════");
        println!(
            "  PHASE 1: L1 NMP-Constrained Screening ({} SEMF, lambda_l1={})",
            cli.n_l1, cli.lambda_l1
        );
        println!("═══════════════════════════════════════════════════════════════");

        let t1 = Instant::now();
        let l1_samples = latin_hypercube(cli.n_l1, bounds, current_seed).expect("LHS failed");

        let exp_data_l1 = exp_data.clone();
        let lambda_l1 = cli.lambda_l1;
        let l1_scores: Vec<f64> = l1_samples
            .iter()
            .map(|x| l1_objective_nmp(x, &exp_data_l1, lambda_l1))
            .collect();

        let mut indices: Vec<usize> = (0..l1_scores.len()).collect();
        indices.sort_by(|&a, &b| l1_scores[a].total_cmp(&l1_scores[b]));

        let seeds: Vec<Vec<f64>> = indices
            .iter()
            .take(cli.nm_starts)
            .map(|&i| l1_samples[i].clone())
            .collect();

        println!("  L1 screening:    {:.2}s", t1.elapsed().as_secs_f64());
        println!(
            "  Best L1:         log(1+chi2) = {:.4}",
            l1_scores[indices[0]]
        );
        println!("  L1 seeds:        {}", seeds.len());

        if let Some(nmp) = nuclear_matter_properties(&l1_samples[indices[0]]) {
            println!(
                "  Best L1 NMP:     J={:.1}, rho0={:.4}, E/A={:.2}",
                nmp.j_mev, nmp.rho0_fm3, nmp.e_a_mev
            );
        }
        println!();

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2: L2 DirectSampler
        // ═══════════════════════════════════════════════════════════════
        println!("═══════════════════════════════════════════════════════════════");
        println!("  PHASE 2: L2 NMP-Constrained DirectSampler");
        println!("═══════════════════════════════════════════════════════════════");

        let t2 = Instant::now();

        let direct_config = DirectSamplerConfig::new(current_seed)
            .with_rounds(cli.n_rounds)
            .with_solvers(seeds.len().max(1))
            .with_eval_budget(cli.evals_per_start)
            .with_patience(cli.patience)
            .with_warm_start(seeds.clone());

        println!(
            "  Config: {} rounds, {} solvers, {} evals/solver, patience={}",
            direct_config.n_rounds,
            direct_config.n_solvers,
            direct_config.max_eval_per_solver,
            cli.patience
        );
        println!();

        let nuclei_ds = nuclei.clone();
        let lambda_l2 = cli.lambda;
        let l2_obj_ds = move |x: &[f64]| -> f64 { l2_objective_nmp(x, &nuclei_ds, lambda_l2) };

        let result_direct =
            direct_sampler(l2_obj_ds, bounds, &direct_config).expect("DirectSampler failed");

        let direct_time = t2.elapsed().as_secs_f64();

        let (ds_chi2_be, ds_chi2_nmp, ds_chi2_total) =
            decompose_chi2(&result_direct.x_best, &nuclei, cli.lambda);

        println!();
        println!(
            "  DirectSampler: {} evals in {:.1}s",
            result_direct.cache.len(),
            direct_time
        );
        println!("    chi2_BE/datum:  {ds_chi2_be:.4}");
        println!("    chi2_NMP/datum: {ds_chi2_nmp:.4}");
        println!(
            "    chi2_total:     {:.4} (BE + {} * NMP)",
            ds_chi2_total, cli.lambda
        );

        if !result_direct.rounds.is_empty() {
            println!("\n  Round-by-round:");
            for r in &result_direct.rounds {
                let rmse_str = r
                    .surrogate_rmse
                    .map_or_else(|| "n/a".to_string(), |v| format!("{v:.4}"));
                println!(
                    "    Round {}: best_f={:.6}, evals={}, rmse={}, delta={:.2e}",
                    r.round, r.best_f, r.n_evals, rmse_str, r.improvement
                );
            }
        }

        let history: Vec<f64> = result_direct.rounds.iter().map(|r| r.best_f).collect();
        if history.len() >= 2 {
            if let Ok(diag) = convergence_diagnostics(&history, 5, 0.01, 2) {
                println!("\n  {}", diag.summary());
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2b (optional): SparsitySampler comparison
        // ═══════════════════════════════════════════════════════════════
        let mut sparsity_result: Option<(f64, f64, f64, usize, f64)> = None;

        if cli.sparsity {
            println!();
            println!("═══════════════════════════════════════════════════════════════");
            println!("  PHASE 2b: L2 SparsitySampler (auto_smoothing + exploration)");
            println!("═══════════════════════════════════════════════════════════════");

            let t_sp = Instant::now();

            let mut sp_config = SparsitySamplerConfig::new(bounds.len(), current_seed + 7777);
            sp_config.n_initial = 30;
            sp_config.n_solvers = 6;
            sp_config.max_eval_per_solver = 40;
            sp_config.n_iterations = 4;
            sp_config.auto_smoothing = true;
            sp_config.penalty_filter = PenaltyFilter::AdaptiveMAD(5.0);
            sp_config.warm_start_seeds.clone_from(&seeds);

            let nuclei_sp = nuclei.clone();
            let lambda_sp = cli.lambda;
            let l2_obj_sp = move |x: &[f64]| -> f64 { l2_objective_nmp(x, &nuclei_sp, lambda_sp) };

            match sparsity_sampler(l2_obj_sp, bounds, &sp_config) {
                Ok(result_sp) => {
                    let sp_time = t_sp.elapsed().as_secs_f64();
                    let (sp_be, sp_nmp, sp_total) =
                        decompose_chi2(&result_sp.x_best, &nuclei, cli.lambda);
                    println!(
                        "  SparsitySampler: {} evals in {:.1}s",
                        result_sp.cache.len(),
                        sp_time
                    );
                    println!("    chi2_BE/datum:  {sp_be:.4}");
                    println!("    chi2_NMP/datum: {sp_nmp:.4}");
                    println!("    chi2_total:     {sp_total:.4}");
                    sparsity_result =
                        Some((sp_be, sp_nmp, sp_total, result_sp.cache.len(), sp_time));
                }
                Err(e) => {
                    println!("  SparsitySampler failed: {e}");
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 3: Statistical analysis on best result
        // ═══════════════════════════════════════════════════════════════
        println!();
        println!("═══════════════════════════════════════════════════════════════");
        println!("  STATISTICAL ANALYSIS");
        println!("═══════════════════════════════════════════════════════════════");

        let best_params = &result_direct.x_best;

        let (observed, expected, sigma) = compute_l2_binding_energies(best_params, &nuclei);

        if !observed.is_empty() {
            if let Ok(chi2_result) =
                chi2_decomposed_weighted(&observed, &expected, &sigma, bounds.len())
            {
                println!("\n{}", chi2_result.summary());

                let worst = chi2_result.worst_n(5);
                println!("\n  Top 5 worst-fitting nuclei (by pull):");
                for &idx in &worst {
                    println!(
                        "    [{}] pull={:.2}sigma, chi2_i={:.2}, residual={:.2} MeV",
                        idx,
                        chi2_result.pulls[idx],
                        chi2_result.contributions[idx],
                        chi2_result.residuals[idx]
                    );
                }

                let per_datum: Vec<f64> = chi2_result.contributions.clone();
                if per_datum.len() >= 5 {
                    if let Ok(ci) = bootstrap_ci(
                        &per_datum,
                        |d| d.iter().sum::<f64>() / d.len() as f64,
                        5000,
                        0.95,
                        current_seed,
                    ) {
                        println!("\n  Bootstrap 95% CI on chi2/datum: {}", ci.summary());
                    }
                }

                // Per-region analysis
                println!("\n  Accuracy by mass region:");
                println!(
                    "  {:>15} {:>6} {:>10} {:>10} {:>12} {:>10}",
                    "Region", "Count", "RMS(MeV)", "MAE(MeV)", "Mean|dB/B|", "chi2/dat"
                );
                for (label, lo, hi) in &[
                    ("Light A<56", 0, 56),
                    ("Medium 56-100", 56, 100),
                    ("Heavy 100-200", 100, 200),
                    ("V.Heavy 200+", 200, 999),
                ] {
                    let region: Vec<usize> = nuclei
                        .iter()
                        .enumerate()
                        .filter(|(_, (z, n, _))| {
                            let a = z + n;
                            a >= *lo && a < *hi
                        })
                        .map(|(i, _)| i)
                        .filter(|&i| i < observed.len())
                        .collect();
                    if region.is_empty() {
                        continue;
                    }
                    let mut sum_sq = 0.0;
                    let mut sum_abs = 0.0;
                    let mut sum_rel = 0.0;
                    let mut sum_chi2 = 0.0;
                    for &i in &region {
                        let resid = observed[i] - expected[i];
                        sum_sq += resid * resid;
                        sum_abs += resid.abs();
                        sum_rel += (resid / expected[i]).abs();
                        sum_chi2 += chi2_result.contributions[i];
                    }
                    let cnt = region.len() as f64;
                    println!(
                        "  {:>15} {:>6} {:>10.3} {:>10.3} {:>12.6e} {:>10.4}",
                        label,
                        region.len(),
                        (sum_sq / cnt).sqrt(),
                        sum_abs / cnt,
                        sum_rel / cnt,
                        sum_chi2 / cnt
                    );
                }
            }
        }

        if let Some(nmp) = nuclear_matter_properties(best_params) {
            println!();
            provenance::print_nmp_analysis(&nmp);
        }

        println!("\n  Best parameters:");
        let names = [
            "t0", "t1", "t2", "t3", "x0", "x1", "x2", "x3", "alpha", "W0",
        ];
        for (i, &v) in best_params.iter().enumerate() {
            println!("    {:6} = {:>12.4}", names.get(i).unwrap_or(&"?"), v);
        }

        // Track for multi-seed summary
        all_results.push(SeedResult {
            seed: current_seed,
            chi2_be: ds_chi2_be,
            chi2_nmp: ds_chi2_nmp,
            chi2_total: ds_chi2_total,
            n_evals: result_direct.cache.len(),
            time: direct_time,
            params: best_params.clone(),
            sparsity: sparsity_result,
        });
    }

    // ═══════════════════════════════════════════════════════════════
    // MULTI-SEED SUMMARY
    // ═══════════════════════════════════════════════════════════════
    if cli.multi > 1 {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!(
            "║  MULTI-SEED SUMMARY ({} seeds)                              ║",
            cli.multi
        );
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();
        println!(
            "  {:>8} {:>12} {:>12} {:>12} {:>8} {:>8}",
            "Seed", "chi2_BE", "chi2_NMP", "chi2_total", "Evals", "Time(s)"
        );
        for r in &all_results {
            println!(
                "  {:>8} {:>12.4} {:>12.4} {:>12.4} {:>8} {:>8.1}",
                r.seed, r.chi2_be, r.chi2_nmp, r.chi2_total, r.n_evals, r.time
            );
        }

        let best = all_results
            .iter()
            .min_by(|a, b| a.chi2_be.total_cmp(&b.chi2_be))
            .expect("at least one multi-seed result");
        let mean_be: f64 = all_results.iter().map(|r| r.chi2_be).sum::<f64>() / cli.multi as f64;
        let std_be = if cli.multi > 1 {
            let var: f64 = all_results
                .iter()
                .map(|r| (r.chi2_be - mean_be).powi(2))
                .sum::<f64>()
                / (cli.multi - 1) as f64;
            var.sqrt()
        } else {
            0.0
        };

        println!();
        println!(
            "  Best chi2_BE/datum:  {:.4} (seed={})",
            best.chi2_be, best.seed
        );
        println!("  Mean chi2_BE/datum:  {mean_be:.4} +/- {std_be:.4}");
    }

    // ═══════════════════════════════════════════════════════════════
    // COMPARISON
    // ═══════════════════════════════════════════════════════════════
    let best = all_results
        .iter()
        .min_by(|a, b| a.chi2_be.total_cmp(&b.chi2_be))
        .expect("at least one multi-seed result");

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  {:42} {:>10} {:>8} {:>8}",
        "Method", "chi2/dat", "Evals", "Time"
    );
    println!(
        "  {:42} {:>10} {:>8} {:>8}",
        "-".repeat(42),
        "-".repeat(10),
        "-".repeat(8),
        "-".repeat(8)
    );
    println!(
        "  {:42} {:>10.2} {:>8} {:>6.0}s",
        "DirectSampler (this run)", best.chi2_be, best.n_evals, best.time
    );

    if let Some((sp_be, _, _, sp_evals, sp_time)) = best.sparsity {
        println!(
            "  {:42} {:>10.2} {:>8} {:>6.0}s",
            "SparsitySampler (this run)", sp_be, sp_evals, sp_time
        );
    }

    println!(
        "  {:42} {:>10.2} {:>8} {:>8}",
        "Prev BarraCUDA L2 (pre-fix)",
        provenance::L2_PYTHON_TOTAL_CHI2.value,
        4022,
        "743s"
    );
    println!(
        "  {:42} {:>10.2} {:>8} {:>8}",
        "Prev BarraCUDA L2 (post-fix, small budget)", 16.11, 40, "1559s"
    );
    println!(
        "  {:42} {:>10.2} {:>8} {:>8}",
        "Python/scipy control",
        provenance::L2_PYTHON_CHI2.value,
        provenance::L2_PYTHON_CANDIDATES.value as i32,
        "~600s"
    );

    // Python parity check
    if best.chi2_be < provenance::L2_PYTHON_CHI2.value {
        let improvement = provenance::L2_PYTHON_CHI2.value / best.chi2_be;
        println!(
            "\n  PYTHON PARITY: EXCEEDED by {:.1}x (BarraCUDA {:.2} vs Python {})",
            improvement,
            best.chi2_be,
            provenance::L2_PYTHON_CHI2.value
        );
    } else {
        let gap = best.chi2_be / provenance::L2_PYTHON_CHI2.value;
        println!("\n  PYTHON PARITY: {gap:.1}x gap remaining");
    }

    // Paper parity analysis
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PAPER PARITY ANALYSIS (target ~10^-6 relative accuracy)");
    println!("═══════════════════════════════════════════════════════════════");

    let (observed, expected, _sigma) = compute_l2_binding_energies(&best.params, &nuclei);
    if !observed.is_empty() {
        let mut sum_rel_sq = 0.0;
        let mut sum_abs_sq = 0.0;
        for i in 0..observed.len() {
            let rel = ((observed[i] - expected[i]) / expected[i]).abs();
            sum_rel_sq += rel * rel;
            sum_abs_sq += (observed[i] - expected[i]).powi(2);
        }
        let rms_rel = (sum_rel_sq / observed.len() as f64).sqrt();
        let rms_abs = (sum_abs_sq / observed.len() as f64).sqrt();

        println!();
        println!("  RMS |dB/B|:      {rms_rel:.6e}");
        println!("  RMS |dB| (MeV):  {rms_abs:.3}");
        println!("  Paper target:    ~1.0e-06 (relative)");
        println!(
            "  Gap to paper:    {:.1} orders of magnitude",
            (rms_rel / 1e-6).log10()
        );
        println!();
        println!("  Physics model capabilities:");
        println!("    L1 SEMF floor:     ~5e-3 relative (~2-3 MeV RMS)");
        println!("    L2 HFB floor:      ~3e-4 relative (~0.5 MeV RMS)");
        println!("    L3 (deformed HFB): ~1e-5 relative (~0.1 MeV RMS)");
        println!("    Paper (beyond-MF): ~1e-6 relative (~keV RMS)");
        println!();
        if rms_rel < 1e-3 {
            println!("  L2 HFB is performing well. Next step: L3 deformation.");
        } else if rms_rel < 1e-2 {
            println!("  L2 HFB needs more optimization budget or constraint tuning.");
        } else {
            println!("  Significant gap remains. Need more L2 budget and lambda tuning.");
        }
    }

    // Save results
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let result_json = serde_json::json!({
        "level": 2,
        "engine": "barracuda::l2_evolved_hfb",
        "physics": "p/n_HFB + BCS + Coulomb_Poisson_Slater + T_eff + CM",
        "math": "100pct_barracuda (gradient_1d_2ndorder, eigh_f64, brent)",
        "lambda": cli.lambda,
        "lambda_l1": cli.lambda_l1,
        "seed": cli.seed,
        "multi": cli.multi,
        "chi2_be_per_datum": best.chi2_be,
        "chi2_nmp_per_datum": best.chi2_nmp,
        "chi2_total_per_datum": best.chi2_total,
        "n_evals": best.n_evals,
        "time_seconds": best.time,
        "best_params": best.params,
        "nmp": nuclear_matter_properties(&best.params).map(|n| serde_json::json!({
            "rho0": n.rho0_fm3,
            "E_A": n.e_a_mev,
            "K_inf": n.k_inf_mev,
            "m_eff": n.m_eff_ratio,
            "J": n.j_mev,
        })),
        "all_seeds": all_results.iter().map(|r| serde_json::json!({
            "seed": r.seed,
            "chi2_be": r.chi2_be,
            "chi2_nmp": r.chi2_nmp,
            "n_evals": r.n_evals,
        })).collect::<Vec<_>>(),
        "references": {
            "prev_barracuda_l2_prefix": { "chi2_per_datum": provenance::L2_PYTHON_TOTAL_CHI2.value },
            "prev_barracuda_l2_postfix": { "chi2_per_datum": 16.11 },
            "python_scipy": { "chi2_per_datum": provenance::L2_PYTHON_CHI2.value },
        },
    });
    let path = results_dir.join("barracuda_l2_evolved.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&result_json).expect("JSON serialize"),
    )
    .ok();
    println!("\n  Results saved to: {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════
// Data structures
// ═══════════════════════════════════════════════════════════════════

struct SeedResult {
    seed: u64,
    chi2_be: f64,
    chi2_nmp: f64,
    chi2_total: f64,
    n_evals: usize,
    time: f64,
    params: Vec<f64>,
    sparsity: Option<(f64, f64, f64, usize, f64)>, // (be, nmp, total, evals, time)
}

// ═══════════════════════════════════════════════════════════════════
// Objective functions
// ═══════════════════════════════════════════════════════════════════

fn l1_objective_nmp(x: &[f64], exp_data: &HashMap<(usize, usize), (f64, f64)>, lambda: f64) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    let Some(nmp) = nuclear_matter_properties(x) else {
        return (1e4_f64).ln_1p();
    };

    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 {
        return (1e4_f64).ln_1p();
    }
    if nmp.e_a_mev > 0.0 {
        return (1e4_f64).ln_1p();
    }

    let mut chi2_be = 0.0;
    let mut n = 0;
    for (&(z, nn), &(b_exp, _)) in exp_data {
        let b_calc = semf_binding_energy(z, nn, x);
        if b_calc > 0.0 {
            let sigma_theo = tolerances::sigma_theo(b_exp);
            chi2_be += ((b_calc - b_exp) / sigma_theo).powi(2);
            n += 1;
        }
    }
    if n == 0 {
        return (1e4_f64).ln_1p();
    }

    let chi2_be_datum = chi2_be / f64::from(n);
    let chi2_nmp_datum = provenance::nmp_chi2_from_props(&nmp) / 5.0;
    lambda.mul_add(chi2_nmp_datum, chi2_be_datum).ln_1p()
}

fn l2_objective_nmp(params: &[f64], nuclei: &[(usize, usize, f64)], lambda: f64) -> f64 {
    if params[8] <= 0.01 || params[8] > 1.0 {
        return (1e10_f64).ln_1p();
    }

    let Some(nmp) = nuclear_matter_properties(params) else {
        return (1e10_f64).ln_1p();
    };

    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 {
        return (1e10_f64).ln_1p();
    }
    if nmp.e_a_mev > 0.0 {
        return (1e10_f64).ln_1p();
    }

    let results: Vec<(f64, f64)> = nuclei
        .par_iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, _conv) = binding_energy_l2(z, n, params).expect("HFB solve");
            (b_calc, b_exp)
        })
        .collect();

    let mut chi2_be = 0.0;
    let mut n_valid = 0;
    for &(b_calc, b_exp) in &results {
        if b_calc > 0.0 {
            let sigma_theo = tolerances::sigma_theo(b_exp);
            chi2_be += ((b_calc - b_exp) / sigma_theo).powi(2);
            n_valid += 1;
        }
    }
    if n_valid == 0 {
        return (1e10_f64).ln_1p();
    }

    let chi2_be_datum = chi2_be / f64::from(n_valid);
    let chi2_nmp_datum = provenance::nmp_chi2_from_props(&nmp) / 5.0;
    lambda.mul_add(chi2_nmp_datum, chi2_be_datum).ln_1p()
}

fn decompose_chi2(params: &[f64], nuclei: &[(usize, usize, f64)], lambda: f64) -> (f64, f64, f64) {
    let results: Vec<(f64, f64)> = nuclei
        .par_iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, _) = binding_energy_l2(z, n, params).expect("HFB solve");
            (b_calc, b_exp)
        })
        .collect();

    let mut chi2_be = 0.0;
    let mut n_valid = 0;
    for &(b_calc, b_exp) in &results {
        if b_calc > 0.0 {
            let sigma_theo = tolerances::sigma_theo(b_exp);
            chi2_be += ((b_calc - b_exp) / sigma_theo).powi(2);
            n_valid += 1;
        }
    }

    let chi2_be_datum = if n_valid > 0 {
        chi2_be / f64::from(n_valid)
    } else {
        1e10
    };
    let chi2_nmp_datum = nuclear_matter_properties(params)
        .map_or(1e4, |n| provenance::nmp_chi2_from_props(&n) / 5.0);
    let chi2_total = lambda.mul_add(chi2_nmp_datum, chi2_be_datum);

    (chi2_be_datum, chi2_nmp_datum, chi2_total)
}

fn compute_l2_binding_energies(
    params: &[f64],
    nuclei: &[(usize, usize, f64)],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let results: Vec<(f64, f64)> = nuclei
        .par_iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, _) = binding_energy_l2(z, n, params).expect("HFB solve");
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
            sigma.push(tolerances::sigma_theo(b_exp));
        }
    }

    (observed, expected, sigma)
}
