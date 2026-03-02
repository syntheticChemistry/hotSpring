// SPDX-License-Identifier: AGPL-3.0-only

//! Nuclear EOS Level 1 — Revalidation with `BarraCuda` Native APIs
//!
//! Now uses `BarraCuda`'s own implementations (evolved from our reference specs):
//!   1. `barracuda::sample::direct::direct_sampler` — round-based NM
//!   2. `barracuda::sample::sparsity` — with `auto_smoothing` + `penalty_filter`
//!   3. `barracuda::surrogate::loo_cv_optimal_smoothing` — LOO-CV grid search
//!   4. `barracuda::stats::chi2_decomposed_weighted` — per-nucleus chi² analysis
//!   5. `barracuda::stats::bootstrap_ci` — confidence intervals
//!   6. `barracuda::optimize::convergence_diagnostics` — stagnation detection
//!
//! Run: cargo run --release --bin `nuclear_eos_l1_ref`

use hotspring_barracuda::data;
use hotspring_barracuda::gpu::GpuF64;
use hotspring_barracuda::nuclear_eos_helpers::{
    compute_be_chi2_only, compute_binding_energies, compute_mean_std, make_l1_objective_nmp,
    print_comparison_summary, print_reference_baselines, run_deep_residual_analysis,
};
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::provenance;

// ALL from barracuda native — no hotspring_barracuda::surrogate or ::stats
use barracuda::sample::direct::{direct_sampler, DirectSamplerConfig};
use barracuda::sample::sparsity::{sparsity_sampler, PenaltyFilter, SparsitySamplerConfig};
use barracuda::stats::{bootstrap_ci, chi2_decomposed_weighted};
// loo_cv_optimal_smoothing and RBFKernel now used internally by SparsitySampler
use barracuda::optimize::convergence_diagnostics;

use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// NMP targets — use provenance::NMP_TARGETS, provenance::print_nmp_analysis
// ═══════════════════════════════════════════════════════════════════

/// UNEDF0 — alternate parametrization: differs from `provenance::UNEDF0_PARAMS`.
/// Local: -1883.69, 277.50, -189.08, 14603.6, 0.0047, -1.116, -1.635, 0.390, 0.3222, 78.66
/// Provenance: -1883.68, 277.50, -207.20, 14263.6, 0.0085, -1.532, -1.0, 0.397, 1/6, 79.53
const UNEDF0_PARAMS: [f64; 10] = [
    -1883.69, 277.50, -189.08, 14603.6, 0.0047, -1.116, -1.635, 0.390, 0.3222, 78.66,
];

struct CliArgs {
    seed: u64,
    n_seeds: usize,
    lambda: f64,
    pareto: bool,
}

// EosContext and load_eos_context are now in data module
use hotspring_barracuda::data::EosContext;

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut seed: u64 = 42;
    let mut n_seeds: usize = 1;
    let mut lambda: f64 = 0.0; // default: unconstrained (backward compat)
    let mut pareto = false;
    for arg in &args[1..] {
        if let Some(s) = arg.strip_prefix("--seed=") {
            seed = s.parse().unwrap_or(42);
        } else if let Some(s) = arg.strip_prefix("--multi=") {
            n_seeds = s.parse().unwrap_or(1);
        } else if let Some(s) = arg.strip_prefix("--lambda=") {
            lambda = s.parse().unwrap_or(0.0);
        } else if arg == "--pareto" {
            pareto = true;
        }
    }
    CliArgs {
        seed,
        n_seeds,
        lambda,
        pareto,
    }
}

fn main() {
    let cli = parse_args();
    let base_seed = cli.seed;
    let lambda = cli.lambda;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L1 — BarraCuda Native Revalidation            ║");
    println!("║  NMP-Constrained Objective (UNEDF-style)                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if cli.pareto {
        println!("  Pareto sweep mode: lambda = [0, 1, 5, 10, 25, 50, 100]");
        println!("  Seeds per lambda: 5");
        println!();
        run_pareto_sweep(base_seed);
        return;
    }

    if cli.n_seeds > 1 {
        println!(
            "  Multi-seed mode: {} independent runs (seeds {}-{}), lambda={}",
            cli.n_seeds,
            base_seed,
            base_seed + cli.n_seeds as u64 - 1,
            lambda
        );
        println!();
        run_multi_seed(base_seed, cli.n_seeds, lambda);
        return;
    }

    println!(
        "  Seed: {base_seed}  Lambda(NMP): {lambda}  (--seed=N --lambda=N --multi=N --pareto)"
    );
    println!();

    let EosContext {
        base,
        exp_data,
        bounds,
    } = data::load_eos_context().expect("Failed to load EOS context");

    println!("  Experimental nuclei: {}", exp_data.len());
    println!("  Parameters:          {} dimensions", bounds.len());
    println!();

    let objective = make_l1_objective_nmp(&exp_data, lambda);

    // Initialize GPU device (required for RBF surrogate training inside samplers)
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime creation");
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("Failed to create GPU device (required for surrogate training)");
    let device = gpu.to_wgpu_device();
    println!(
        "  Device: {} (SHADER_F64: {})",
        gpu.adapter_name, gpu.has_f64
    );
    if !gpu.has_f64 {
        println!("  WARNING: SHADER_F64 not supported — surrogate training may fail");
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // APPROACH 1: SparsitySampler with native auto-smoothing
    // ═══════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("  APPROACH 1: SparsitySampler + Native Auto-Smoothing");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let t0 = Instant::now();

    let config = SparsitySamplerConfig::new(bounds.len(), base_seed)
        .with_initial_samples(100)
        .with_solvers(8)
        .with_eval_budget(50)
        .with_iterations(5);

    // Enable native auto-smoothing & penalty filter
    let mut config = config;
    config.auto_smoothing = true;
    config.penalty_filter = PenaltyFilter::AdaptiveMAD(5.0);

    println!(
        "  Config: {} initial, {}×{} solvers×budget, {} iters",
        config.n_initial, config.n_solvers, config.max_eval_per_solver, config.n_iterations
    );
    println!("  Auto-smoothing: ENABLED (LOO-CV)");
    println!("  Penalty filter: AdaptiveMAD(5.0)");
    println!();

    let result1 = sparsity_sampler(device.clone(), objective, &bounds, &config)
        .expect("SparsitySampler failed");

    let approach1_time = t0.elapsed().as_secs_f64();
    let approach1_f = result1.f_best;
    let approach1_x = result1.x_best.clone();
    let approach1_evals = result1.cache.len();

    println!("  SparsitySampler: {approach1_evals} evals in {approach1_time:.2}s");
    let chi2_1 = approach1_f.exp_m1();
    println!("  log(1+χ²) = {approach1_f:.6}, χ²/datum = {chi2_1:.4}");

    // ═══════════════════════════════════════════════════════════════
    // APPROACH 2: Native DirectSampler (round-based NM)
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  APPROACH 2: Native DirectSampler (Round-Based NM)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let objective2 = make_l1_objective_nmp(&exp_data, lambda);

    let direct_config = DirectSamplerConfig::new(base_seed)
        .with_rounds(8)
        .with_solvers(8)
        .with_eval_budget(120)
        .with_patience(3);

    println!(
        "  Config: {} rounds, {} solvers, {} evals/solver, patience={}",
        direct_config.n_rounds,
        direct_config.n_solvers,
        direct_config.max_eval_per_solver,
        direct_config.patience
    );
    println!(
        "  Auto-smoothing: {} (monitoring)",
        direct_config.auto_smoothing
    );
    println!();

    let result2 =
        direct_sampler(device, objective2, &bounds, &direct_config).expect("DirectSampler failed");

    let approach2_f = result2.f_best;
    let approach2_x = result2.x_best.clone();
    let approach2_evals = result2.cache.len();
    let approach2_time = t0.elapsed().as_secs_f64() - approach1_time;

    let chi2_2 = approach2_f.exp_m1();
    println!("  DirectSampler: {approach2_evals} evals");
    println!("  log(1+χ²) = {approach2_f:.6}, χ²/datum = {chi2_2:.4}");

    // Round-by-round diagnostics
    if !result2.rounds.is_empty() {
        println!();
        println!("  Round-by-round:");
        for r in &result2.rounds {
            let rmse_str = r
                .surrogate_rmse
                .map_or_else(|| "n/a".to_string(), |v| format!("{v:.4}"));
            println!(
                "    Round {}: best_f={:.6}, evals={}, surrogate_rmse={}, Δ={:.2e}",
                r.round, r.best_f, r.n_evals, rmse_str, r.improvement
            );
        }
    }
    if result2.early_stopped {
        println!(
            "  ⏹ Early stopped (no improvement for {} rounds)",
            direct_config.patience
        );
    }

    // Native convergence diagnostics
    let history: Vec<f64> = result2.rounds.iter().map(|r| r.best_f).collect();
    if history.len() >= 2 {
        let diag =
            convergence_diagnostics(&history, 5, 0.01, 3).expect("convergence_diagnostics failed");
        println!();
        println!("  {}", diag.summary());
    }

    // ═══════════════════════════════════════════════════════════════
    // STATISTICAL ANALYSIS (all native barracuda::stats)
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  STATISTICAL ANALYSIS (barracuda::stats native)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Pick the better result
    let (best_x, best_f, best_label) = if approach1_f <= approach2_f {
        (&approach1_x, approach1_f, "SparsitySampler+AutoSmooth")
    } else {
        (&approach2_x, approach2_f, "DirectSampler")
    };
    let best_chi2 = best_f.exp_m1();
    println!("  Best approach: {best_label} (χ²/datum = {best_chi2:.4})");

    // Chi-squared decomposition (weighted, with sigma)
    let (observed, expected, sigma) = compute_binding_energies(best_x, &exp_data);
    match chi2_decomposed_weighted(&observed, &expected, &sigma, bounds.len()) {
        Ok(chi2_result) => {
            println!();
            println!("{}", chi2_result.summary());

            // Top 5 worst nuclei
            let worst = chi2_result.worst_n(5);
            println!();
            println!("  Top 5 worst-fitting nuclei (by pull):");
            for &idx in &worst {
                println!(
                    "    [{}] pull={:.2}σ, χ²_i={:.2}, residual={:.2} MeV",
                    idx,
                    chi2_result.pulls[idx],
                    chi2_result.contributions[idx],
                    chi2_result.residuals[idx]
                );
            }

            // Bootstrap CI on per-datum chi²
            let per_datum_contribs: Vec<f64> = chi2_result.contributions;
            match bootstrap_ci(
                &per_datum_contribs,
                |d| d.iter().sum::<f64>() / d.len() as f64,
                5000,
                0.95,
                42,
            ) {
                Ok(ci) => {
                    println!();
                    println!("  Bootstrap 95% CI on χ²/datum: {}", ci.summary());
                }
                Err(e) => println!("  Bootstrap failed: {e}"),
            }
        }
        Err(e) => println!("  chi2_decomposed_weighted failed: {e}"),
    }

    // Nuclear matter properties — detailed analysis
    if let Some(nmp) = nuclear_matter_properties(best_x) {
        println!();
        provenance::print_nmp_analysis(&nmp);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  COMPARISON SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    let better = chi2_1.min(chi2_2);
    print_comparison_summary(
        chi2_1,
        approach1_evals,
        approach1_time,
        chi2_2,
        approach2_evals,
        approach2_time,
        better,
    );

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  REFERENCE BASELINES — Published Parametrizations");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    print_reference_baselines(
        &exp_data,
        lambda,
        &[
            ("SLy4", provenance::SLY4_PARAMS.as_slice()),
            ("UNEDF0", UNEDF0_PARAMS.as_slice()),
        ],
    );

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  DEEP RESIDUAL ANALYSIS — Paper Accuracy Comparison");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    run_deep_residual_analysis(
        &base,
        best_x,
        best_chi2,
        chi2_1,
        approach1_evals,
        approach1_time,
        approach1_f,
        chi2_2,
        approach2_evals,
        &result2,
        base_seed,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Multi-seed variance study
// ═══════════════════════════════════════════════════════════════════

fn run_multi_seed(base_seed: u64, n_seeds: usize, lambda: f64) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime creation");
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("Failed to create GPU device (required for DirectSampler)");
    let device = gpu.to_wgpu_device();

    struct SeedResult {
        seed: u64,
        direct_chi2_total: f64,
        direct_chi2_be: f64,
        direct_chi2_nmp: f64,
        direct_evals: usize,
        direct_time_ms: u128,
        direct_j: f64,
    }
    let EosContext {
        base: base_path,
        exp_data,
        bounds,
    } = data::load_eos_context().expect("Failed to load EOS context");

    println!(
        "  Nuclei: {}, Dimensions: {}, Lambda(NMP): {}",
        exp_data.len(),
        bounds.len(),
        lambda
    );
    println!();

    let mut results: Vec<SeedResult> = Vec::new();

    println!(
        "  {:>6} │ {:>10} {:>10} {:>10} {:>6} {:>8} │ {:>8}",
        "Seed", "chi2_total", "chi2_BE", "chi2_NMP", "Evals", "Time", "J(MeV)"
    );
    println!(
        "  {:─>6}─┼─{:─>10}─{:─>10}─{:─>10}─{:─>6}─{:─>8}─┼─{:─>8}",
        "", "", "", "", "", "", ""
    );

    for i in 0..n_seeds {
        let seed = base_seed + i as u64;

        let obj_d = make_l1_objective_nmp(&exp_data, lambda);
        let dc = DirectSamplerConfig::new(seed)
            .with_rounds(8)
            .with_solvers(8)
            .with_eval_budget(120)
            .with_patience(3);
        let t0 = Instant::now();
        let r_d =
            direct_sampler(device.clone(), obj_d, &bounds, &dc).expect("DirectSampler failed");
        let d_time = t0.elapsed().as_millis();
        let d_evals = r_d.cache.len();

        // Decompose the result into BE and NMP components
        let d_chi2_be = compute_be_chi2_only(&r_d.x_best, &exp_data);
        let (d_chi2_nmp, d_j) = if let Some(nmp) = nuclear_matter_properties(&r_d.x_best) {
            (provenance::nmp_chi2_from_props(&nmp) / 5.0, nmp.j_mev)
        } else {
            (1e4, 0.0)
        };
        let d_chi2_total = lambda.mul_add(d_chi2_nmp, d_chi2_be);

        println!(
            "  {seed:>6} │ {d_chi2_total:>10.4} {d_chi2_be:>10.4} {d_chi2_nmp:>10.4} {d_evals:>6} {d_time:>6}ms │ {d_j:>8.1}"
        );

        results.push(SeedResult {
            seed,
            direct_chi2_total: d_chi2_total,
            direct_chi2_be: d_chi2_be,
            direct_chi2_nmp: d_chi2_nmp,
            direct_evals: d_evals,
            direct_time_ms: d_time,
            direct_j: d_j,
        });
    }

    // Statistics across seeds
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  VARIANCE ANALYSIS ({n_seeds} seeds, lambda={lambda})");
    println!("═══════════════════════════════════════════════════════════════");

    let be_vals: Vec<f64> = results.iter().map(|r| r.direct_chi2_be).collect();
    let nmp_vals: Vec<f64> = results.iter().map(|r| r.direct_chi2_nmp).collect();
    let j_vals: Vec<f64> = results.iter().map(|r| r.direct_j).collect();

    let (be_mean, be_std) = compute_mean_std(&be_vals);
    let (nmp_mean, nmp_std) = compute_mean_std(&nmp_vals);
    let (j_mean, j_std) = compute_mean_std(&j_vals);

    println!();
    println!("  chi2_BE/datum:  {be_mean:.4} +/- {be_std:.4}");
    println!("  chi2_NMP/datum: {nmp_mean:.4} +/- {nmp_std:.4}");
    println!("  J symmetry:     {j_mean:.1} +/- {j_std:.1} MeV (target: 32 +/- 2)");

    let best_idx = results
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.direct_chi2_total.total_cmp(&b.1.direct_chi2_total))
        .expect("at least one multi-seed result")
        .0;
    println!(
        "  Best seed: {} (chi2_BE={:.4}, chi2_NMP={:.4}, J={:.1})",
        results[best_idx].seed,
        results[best_idx].direct_chi2_be,
        results[best_idx].direct_chi2_nmp,
        results[best_idx].direct_j
    );

    // Save results
    let results_dir = base_path.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let multi_json = serde_json::json!({
        "type": "multi_seed_nmp_constrained",
        "lambda": lambda,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "summary": {
            "chi2_be_mean": be_mean, "chi2_be_std": be_std,
            "chi2_nmp_mean": nmp_mean, "chi2_nmp_std": nmp_std,
            "j_mean": j_mean, "j_std": j_std,
        },
        "per_seed": results.iter().map(|r| serde_json::json!({
            "seed": r.seed,
            "chi2_total": r.direct_chi2_total,
            "chi2_be": r.direct_chi2_be,
            "chi2_nmp": r.direct_chi2_nmp,
            "evals": r.direct_evals,
            "time_ms": r.direct_time_ms,
            "j_mev": r.direct_j,
        })).collect::<Vec<_>>(),
    });
    let path = results_dir.join(format!("barracuda_l1_multi_seed_lambda{lambda}.json"));
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&multi_json).expect("JSON serialize"),
    )
    .ok();
    println!("\n  Results saved to: {}", path.display());
}

// ═══════════════════════════════════════════════════════════════════
// Pareto sweep across lambda values
// ═══════════════════════════════════════════════════════════════════

fn run_pareto_sweep(base_seed: u64) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime creation");
    let gpu = rt
        .block_on(GpuF64::new())
        .expect("Failed to create GPU device (required for DirectSampler)");
    let device = gpu.to_wgpu_device();

    struct ParetoPoint {
        lambda: f64,
        chi2_be_mean: f64,
        chi2_be_std: f64,
        chi2_nmp_mean: f64,
        chi2_nmp_std: f64,
        j_mean: f64,
        j_std: f64,
        rms_mev_mean: f64,
        all_nmp_within_2sigma: usize,
        n_seeds: usize,
        best_params: Vec<f64>,
    }
    let EosContext {
        base: base_path,
        exp_data,
        bounds,
    } = data::load_eos_context().expect("Failed to load EOS context");

    let lambdas: [f64; 7] = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0];
    let n_seeds_per_lambda = 5;

    println!("  Nuclei: {}, Dimensions: {}", exp_data.len(), bounds.len());
    println!("  Lambda values: {lambdas:?}");
    println!("  Seeds per lambda: {n_seeds_per_lambda}");
    println!();

    // SLy4 and UNEDF0 baselines
    let sly4_be = compute_be_chi2_only(provenance::SLY4_PARAMS.as_slice(), &exp_data);
    let sly4_nmp = nuclear_matter_properties(provenance::SLY4_PARAMS.as_slice())
        .map_or(1e4, |n| provenance::nmp_chi2_from_props(&n) / 5.0);
    let sly4_j =
        nuclear_matter_properties(provenance::SLY4_PARAMS.as_slice()).map_or(0.0, |n| n.j_mev);
    let unedf0_be = compute_be_chi2_only(UNEDF0_PARAMS.as_slice(), &exp_data);
    let unedf0_nmp = nuclear_matter_properties(UNEDF0_PARAMS.as_slice())
        .map_or(1e4, |n| provenance::nmp_chi2_from_props(&n) / 5.0);
    let unedf0_j = nuclear_matter_properties(UNEDF0_PARAMS.as_slice()).map_or(0.0, |n| n.j_mev);

    let mut pareto: Vec<ParetoPoint> = Vec::new();

    for &lam in &lambdas {
        let t0 = Instant::now();
        let mut be_vals = Vec::new();
        let mut nmp_vals = Vec::new();
        let mut j_vals = Vec::new();
        let mut rms_vals = Vec::new();
        let mut within_2s = 0_usize;
        let mut best_total = f64::INFINITY;
        let mut best_params = vec![0.0; 10];

        for i in 0..n_seeds_per_lambda {
            let seed = base_seed + i as u64;
            let obj = make_l1_objective_nmp(&exp_data, lam);
            let dc = DirectSamplerConfig::new(seed)
                .with_rounds(8)
                .with_solvers(8)
                .with_eval_budget(120)
                .with_patience(3);
            let r =
                direct_sampler(device.clone(), obj, &bounds, &dc).expect("DirectSampler failed");

            let be = compute_be_chi2_only(&r.x_best, &exp_data);
            be_vals.push(be);

            if let Some(nmp) = nuclear_matter_properties(&r.x_best) {
                let nc = provenance::nmp_chi2_from_props(&nmp) / 5.0;
                nmp_vals.push(nc);
                j_vals.push(nmp.j_mev);

                // Check all within 2sigma
                let vals = [
                    nmp.rho0_fm3,
                    nmp.e_a_mev,
                    nmp.k_inf_mev,
                    nmp.m_eff_ratio,
                    nmp.j_mev,
                ];
                let trg = provenance::NMP_TARGETS.values();
                let sig = provenance::NMP_TARGETS.sigmas();
                let ok = vals
                    .iter()
                    .enumerate()
                    .all(|(k, &v)| ((v - trg[k]) / sig[k]).abs() <= 2.0);
                if ok {
                    within_2s += 1;
                }

                let total = lam.mul_add(nc, be);
                if total < best_total {
                    best_total = total;
                    best_params.clone_from(&r.x_best);
                }
            }

            // RMS
            let mut sq_sum = 0.0;
            let mut nn = 0;
            for (&(z, n), &(b_exp, _)) in exp_data.iter() {
                let b_calc = semf_binding_energy(z, n, &r.x_best);
                if b_calc > 0.0 {
                    sq_sum += (b_calc - b_exp).powi(2);
                    nn += 1;
                }
            }
            rms_vals.push((sq_sum / f64::from(nn.max(1))).sqrt());
        }

        let elapsed = t0.elapsed().as_secs_f64();
        let (be_mean, be_std) = compute_mean_std(&be_vals);
        let (nmp_mean, nmp_std) = compute_mean_std(&nmp_vals);
        let (j_mean, j_std) = compute_mean_std(&j_vals);
        let rms_mean = rms_vals.iter().sum::<f64>() / rms_vals.len() as f64;

        println!("  lambda={lam:>5.0}: chi2_BE={be_mean:.4}+/-{be_std:.4}, chi2_NMP={nmp_mean:.4}+/-{nmp_std:.4}, J={j_mean:.1}+/-{j_std:.1}, RMS={rms_mean:.2}MeV, 2sigma={within_2s}/{n_seeds_per_lambda} [{elapsed:.1}s]");

        pareto.push(ParetoPoint {
            lambda: lam,
            chi2_be_mean: be_mean,
            chi2_be_std: be_std,
            chi2_nmp_mean: nmp_mean,
            chi2_nmp_std: nmp_std,
            j_mean,
            j_std,
            rms_mev_mean: rms_mean,
            all_nmp_within_2sigma: within_2s,
            n_seeds: n_seeds_per_lambda,
            best_params,
        });
    }

    // ═══════════════════════════════════════════════════════════════
    // PARETO FRONTIER TABLE
    // ═══════════════════════════════════════════════════════════════
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PARETO FRONTIER: Binding Energy vs NMP Physical Accuracy");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!(
        "  {:>6} │ {:>10} {:>10} {:>8} {:>8} {:>5}",
        "lambda", "chi2_BE", "chi2_NMP", "J(MeV)", "RMS(MeV)", "2sig"
    );
    println!(
        "  {:─>6}─┼─{:─>10}─{:─>10}─{:─>8}─{:─>8}─{:─>5}",
        "", "", "", "", "", ""
    );

    for p in &pareto {
        let marker = if p.all_nmp_within_2sigma == p.n_seeds {
            "<-"
        } else {
            ""
        };
        println!(
            "  {:>6.0} │ {:>10.4} {:>10.4} {:>8.1} {:>8.2} {:>3}/{} {}",
            p.lambda,
            p.chi2_be_mean,
            p.chi2_nmp_mean,
            p.j_mean,
            p.rms_mev_mean,
            p.all_nmp_within_2sigma,
            p.n_seeds,
            marker
        );
    }

    println!();
    println!("  Reference baselines:");
    println!("    SLy4:   chi2_BE={sly4_be:.4}, chi2_NMP={sly4_nmp:.4}, J={sly4_j:.1}");
    println!("    UNEDF0: chi2_BE={unedf0_be:.4}, chi2_NMP={unedf0_nmp:.4}, J={unedf0_j:.1}");

    // Find optimal lambda (best that has all NMP within 2sigma)
    let physical_results: Vec<&ParetoPoint> = pareto
        .iter()
        .filter(|p| p.all_nmp_within_2sigma == p.n_seeds)
        .collect();
    if let Some(best) = physical_results
        .iter()
        .min_by(|a, b| a.chi2_be_mean.total_cmp(&b.chi2_be_mean))
    {
        println!();
        println!(
            "  OPTIMAL lambda = {:.0} (best BE accuracy with all NMP within 2sigma)",
            best.lambda
        );
        println!("    chi2_BE/datum:  {:.4}", best.chi2_be_mean);
        println!("    chi2_NMP/datum: {:.4}", best.chi2_nmp_mean);
        println!("    J:              {:.1} MeV", best.j_mean);
        println!("    RMS:            {:.2} MeV", best.rms_mev_mean);

        // Print NMP for best params
        if let Some(nmp) = nuclear_matter_properties(&best.best_params) {
            println!();
            provenance::print_nmp_analysis(&nmp);
        }
    } else {
        println!();
        println!("  No lambda value achieved all NMP within 2sigma across all seeds.");
        println!("  Best compromise:");
        if let Some(best) = pareto.iter().max_by_key(|p| p.all_nmp_within_2sigma) {
            println!(
                "    lambda={:.0}: {}/{} seeds within 2sigma, chi2_BE={:.4}, J={:.1}",
                best.lambda,
                best.all_nmp_within_2sigma,
                best.n_seeds,
                best.chi2_be_mean,
                best.j_mean
            );
        }
    }

    // Save pareto results
    let results_dir = base_path.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let pareto_json = serde_json::json!({
        "type": "pareto_sweep_nmp_constrained",
        "base_seed": base_seed,
        "n_seeds_per_lambda": n_seeds_per_lambda,
        "reference_baselines": {
            "sly4": { "chi2_be": sly4_be, "chi2_nmp": sly4_nmp, "j_mev": sly4_j },
            "unedf0": { "chi2_be": unedf0_be, "chi2_nmp": unedf0_nmp, "j_mev": unedf0_j },
        },
        "pareto_points": pareto.iter().map(|p| serde_json::json!({
            "lambda": p.lambda,
            "chi2_be_mean": p.chi2_be_mean, "chi2_be_std": p.chi2_be_std,
            "chi2_nmp_mean": p.chi2_nmp_mean, "chi2_nmp_std": p.chi2_nmp_std,
            "j_mean": p.j_mean, "j_std": p.j_std,
            "rms_mev_mean": p.rms_mev_mean,
            "all_nmp_within_2sigma": p.all_nmp_within_2sigma,
            "n_seeds": p.n_seeds,
        })).collect::<Vec<_>>(),
    });
    let path = results_dir.join("barracuda_l1_pareto_sweep.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&pareto_json).expect("JSON serialize"),
    )
    .ok();
    println!("\n  Full results saved to: {}", path.display());
}
