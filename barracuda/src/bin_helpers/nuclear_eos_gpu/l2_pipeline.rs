// SPDX-License-Identifier: AGPL-3.0-or-later

//! L2 NMP-constrained HFB pipeline stages for `nuclear_eos_l2_ref`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use barracuda::device::WgpuDevice;
use barracuda::optimize::convergence_diagnostics;
use barracuda::sample::direct::{direct_sampler, DirectSamplerConfig};
use barracuda::sample::latin_hypercube;
use barracuda::sample::sparsity::{sparsity_sampler, PenaltyFilter, SparsitySamplerConfig};
use barracuda::stats::{bootstrap_ci, chi2_decomposed_weighted};
use hotspring_barracuda::physics::hfb::binding_energy_l2;
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;
use rayon::prelude::*;

/// CLI configuration for the L2 evolved HFB pipeline.
pub struct L2PipelineCli {
    pub seed: u64,
    pub lambda: f64,
    pub lambda_l1: f64,
    pub n_l1: usize,
    pub nm_starts: usize,
    pub evals_per_start: usize,
    pub n_rounds: usize,
    pub patience: usize,
    pub multi: usize,
    pub sparsity: bool,
}

pub struct SeedResult {
    pub seed: u64,
    pub chi2_be: f64,
    pub chi2_nmp: f64,
    pub chi2_total: f64,
    pub n_evals: usize,
    pub time: f64,
    pub params: Vec<f64>,
    pub sparsity: Option<(f64, f64, f64, usize, f64)>,
}

/// Returns `Some(nmp)` when parameters satisfy NMP saturation constraints; otherwise `None`.
#[must_use]
pub fn validate_nmp_constraints(params: &[f64]) -> Option<hotspring_barracuda::physics::NuclearMatterProps> {
    if params[8] <= 0.01 || params[8] > 1.0 {
        return None;
    }
    let nmp = nuclear_matter_properties(params)?;
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 || nmp.e_a_mev > 0.0 {
        return None;
    }
    Some(nmp)
}

#[must_use]
pub fn l1_objective_nmp(x: &[f64], exp_data: &HashMap<(usize, usize), (f64, f64)>, lambda: f64) -> f64 {
    let Some(nmp) = validate_nmp_constraints(x) else {
        return (1e4_f64).ln_1p();
    };

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

#[must_use]
pub fn l2_objective_nmp(params: &[f64], nuclei: &[(usize, usize, f64)], lambda: f64) -> f64 {
    let Some(nmp) = validate_nmp_constraints(params) else {
        return (1e10_f64).ln_1p();
    };

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

#[must_use]
pub fn decompose_chi2(params: &[f64], nuclei: &[(usize, usize, f64)], lambda: f64) -> (f64, f64, f64) {
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

#[must_use]
pub fn compute_l2_binding_energies(
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

/// Run L1 screening → L2 DirectSampler → optional SparsitySampler → statistical analysis.
pub fn run_l2_seed_pipeline(
    device: Arc<WgpuDevice>,
    cli: &L2PipelineCli,
    current_seed: u64,
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    nuclei: &[(usize, usize, f64)],
    seed_idx: usize,
) -> SeedResult {
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

    // PHASE 1: L1 screening
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

    // PHASE 2: L2 DirectSampler
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

    let nuclei_ds = nuclei.to_vec();
    let lambda_l2 = cli.lambda;
    let l2_obj_ds = move |x: &[f64]| -> f64 { l2_objective_nmp(x, &nuclei_ds, lambda_l2) };

    let result_direct =
        direct_sampler(device.clone(), l2_obj_ds, bounds, &direct_config).expect("DirectSampler failed");

    let direct_time = t2.elapsed().as_secs_f64();

    let (ds_chi2_be, ds_chi2_nmp, ds_chi2_total) =
        decompose_chi2(&result_direct.x_best, nuclei, cli.lambda);

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
    if history.len() >= 2
        && let Ok(diag) = convergence_diagnostics(&history, 5, 0.01, 2)
    {
        println!("\n  {}", diag.summary());
    }

    // PHASE 2b (optional): SparsitySampler
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

        let nuclei_sp = nuclei.to_vec();
        let lambda_sp = cli.lambda;
        let l2_obj_sp = move |x: &[f64]| -> f64 { l2_objective_nmp(x, &nuclei_sp, lambda_sp) };

        match sparsity_sampler(device, l2_obj_sp, bounds, &sp_config) {
            Ok(result_sp) => {
                let sp_time = t_sp.elapsed().as_secs_f64();
                let (sp_be, sp_nmp, sp_total) =
                    decompose_chi2(&result_sp.x_best, nuclei, cli.lambda);
                println!(
                    "  SparsitySampler: {} evals in {:.1}s",
                    result_sp.cache.len(),
                    sp_time
                );
                println!("    chi2_BE/datum:  {sp_be:.4}");
                println!("    chi2_NMP/datum: {sp_nmp:.4}");
                println!("    chi2_total:     {sp_total:.4}");
                sparsity_result = Some((sp_be, sp_nmp, sp_total, result_sp.cache.len(), sp_time));
            }
            Err(e) => {
                println!("  SparsitySampler failed: {e}");
            }
        }
    }

    // PHASE 3: Statistical analysis
    run_statistical_analysis(&result_direct.x_best, nuclei, current_seed);

    SeedResult {
        seed: current_seed,
        chi2_be: ds_chi2_be,
        chi2_nmp: ds_chi2_nmp,
        chi2_total: ds_chi2_total,
        n_evals: result_direct.cache.len(),
        time: direct_time,
        params: result_direct.x_best,
        sparsity: sparsity_result,
    }
}

pub fn run_statistical_analysis(best_params: &[f64], nuclei: &[(usize, usize, f64)], current_seed: u64) {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STATISTICAL ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════");

    let (observed, expected, sigma) = compute_l2_binding_energies(best_params, nuclei);

    if !observed.is_empty()
        && let Ok(chi2_result) =
            chi2_decomposed_weighted(&observed, &expected, &sigma, best_params.len())
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
        if per_datum.len() >= 5
            && let Ok(ci) = bootstrap_ci(
                &per_datum,
                |d| d.iter().sum::<f64>() / d.len() as f64,
                5000,
                0.95,
                current_seed,
            )
        {
            println!("\n  Bootstrap 95% CI on chi2/datum: {}", ci.summary());
        }

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
}
