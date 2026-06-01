// SPDX-License-Identifier: AGPL-3.0-or-later

//! L2 heterogeneous pipeline: cascade filtering, HFB evaluation, and surrogate refinement.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use barracuda::device::WgpuDevice;
use barracuda::optimize::{EvaluationCache, multi_start_nelder_mead};
use barracuda::sample::latin_hypercube;
use barracuda::sample::sparsity::{SparsitySamplerConfig, sparsity_sampler};
use barracuda::surrogate::{RBFKernel, RBFSurrogate};
use hotspring_barracuda::data;
use hotspring_barracuda::physics::nuclear_matter_properties;
use hotspring_barracuda::pipeline::{
    ClassifierResult, generate_l1_training_data, l2_objective, train_classifier,
};
use hotspring_barracuda::prescreen::{
    CascadeStats, NMPConstraints, NMPScreenResult, PreScreenClassifier, cascade_filter,
    l1_proxy_prescreen, nmp_prescreen, perturb_params,
};
use hotspring_barracuda::provenance;
use rayon::prelude::*;

pub struct CliConfig {
    pub mode: String,
    pub n_rounds: usize,
    pub l1_samples: usize,
    pub candidates_per_round: usize,
}

pub fn parse_cli() -> CliConfig {
    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .iter()
        .find(|a| a.starts_with("--mode="))
        .and_then(|a| a.strip_prefix("--mode="))
        .unwrap_or("hetero")
        .to_string();
    CliConfig {
        mode,
        n_rounds: data::parse_cli_usize(&args, "--rounds", 15),
        l1_samples: data::parse_cli_usize(&args, "--l1-samples", 5000),
        candidates_per_round: data::parse_cli_usize(&args, "--candidates", 200),
    }
}

pub struct HeteroResult {
    pub best_params: Vec<f64>,
    pub best_f: f64,
    pub cache: EvaluationCache,
    pub stats: CascadeStats,
}

pub fn run_hetero_mode(
    device: Arc<WgpuDevice>,
    cli: &CliConfig,
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
) -> HeteroResult {
    let (l1_xs, l1_ys) = generate_l1_training_data(bounds, exp_data, cli.l1_samples)
        .expect("L1 training data generation failed");
    let clf_result = train_classifier(&l1_xs, &l1_ys);
    let (best_params, best_f, cache, stats) = run_heterogeneous_l2(
        device,
        bounds,
        exp_data,
        &l1_xs,
        &l1_ys,
        &clf_result,
        cli.n_rounds,
        cli.candidates_per_round,
    );
    HeteroResult {
        best_params,
        best_f,
        cache,
        stats,
    }
}

fn seed_and_cascade_filter(
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    l1_xs: &[Vec<f64>],
    l1_ys: &[f64],
    use_classifier: bool,
    classifier: &PreScreenClassifier,
    cascade_stats: &Arc<Mutex<CascadeStats>>,
) -> (Vec<Vec<f64>>, u64) {
    println!("    Seeding from best L1 regions...");

    let mut indices: Vec<usize> = (0..l1_ys.len()).collect();
    indices.sort_by(|&a, &b| l1_ys[a].total_cmp(&l1_ys[b]));
    let top_k = indices.len().min(50);
    let seed_params: Vec<&Vec<f64>> = indices[..top_k].iter().map(|&i| &l1_xs[i]).collect();

    let mut initial_candidates = Vec::new();
    let mut rng_state = 12345_u64;

    for seed in &seed_params {
        for _ in 0..20 {
            initial_candidates.push(perturb_params(seed, bounds, &mut rng_state, 0.2));
        }
    }

    let lhs_samples = latin_hypercube(500, bounds, 777).expect("LHS sampling failed");
    for sample in &lhs_samples {
        initial_candidates.push(sample.clone());
    }

    println!(
        "    Initial pool: {} candidates ({} seeded + {} LHS)",
        initial_candidates.len(),
        seed_params.len() * 20,
        lhs_samples.len()
    );

    let surviving_candidates = cascade_filter(
        &initial_candidates,
        &NMPConstraints::default(),
        exp_data,
        use_classifier,
        classifier,
        cascade_stats,
    );

    println!(
        "    Survived cascade: {} / {} ({:.1}%)",
        surviving_candidates.len(),
        initial_candidates.len(),
        100.0 * surviving_candidates.len() as f64 / initial_candidates.len() as f64
    );

    (surviving_candidates, rng_state)
}

fn evaluate_hfb_parallel(
    surviving_candidates: &[Vec<f64>],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    hfb_cache: &Arc<Mutex<EvaluationCache>>,
) -> (usize, f64) {
    println!(
        "    Evaluating {} survivors with HFB (parallel)...",
        surviving_candidates.len()
    );

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    let eval_results: Vec<(Vec<f64>, f64)> = surviving_candidates
        .par_iter()
        .map(|params| {
            let obj = l2_objective(params, &nuclei);
            (params.clone(), obj)
        })
        .collect();

    for (x, f) in &eval_results {
        hfb_cache
            .lock()
            .expect("hfb_cache lock")
            .record(x.clone(), *f);
    }

    let n_initial = eval_results.len();
    let init_best = eval_results
        .iter()
        .map(|(_, f)| *f)
        .fold(f64::INFINITY, f64::min);
    println!("    Initial evaluations: {n_initial}, best log(1+χ²) = {init_best:.4}");

    (n_initial, init_best)
}

#[expect(clippy::too_many_arguments, reason = "surrogate refinement requires all physics parameters")]
fn surrogate_refinement_round(
    round: usize,
    device: Arc<WgpuDevice>,
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    constraints: &NMPConstraints,
    use_classifier: bool,
    classifier: &PreScreenClassifier,
    cascade_stats: &Arc<Mutex<CascadeStats>>,
    hfb_cache: &Arc<Mutex<EvaluationCache>>,
    nuclei: &[(usize, usize, f64)],
    candidates_per_round: usize,
    rng_state: &mut u64,
) -> bool {
    let round_t0 = Instant::now();

    let (train_x, train_y) = hfb_cache.lock().expect("hfb_cache lock").training_data();
    let n_before = train_x.len();

    if train_x.len() < 5 {
        println!(
            "      Round {}: too few data points ({}), skipping",
            round + 1,
            train_x.len()
        );
        return false;
    }

    let surrogate = match RBFSurrogate::train(
        device,
        &train_x,
        &train_y,
        RBFKernel::ThinPlateSpline,
        0.0,
    ) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "      Round {}: surrogate training failed: {:?}",
                round + 1,
                e
            );
            return false;
        }
    };

    let surrogate_fn = |x: &[f64]| -> f64 {
        if x.iter()
            .zip(bounds.iter())
            .any(|(&v, &(lo, hi))| v < lo || v > hi)
        {
            return 1e10;
        }
        match nmp_prescreen(x, constraints) {
            NMPScreenResult::Fail(_) => return 1e6,
            NMPScreenResult::Pass(_) => {}
        }
        if l1_proxy_prescreen(x, exp_data, 200.0).is_none() {
            return 1e5;
        }
        let x_vec = vec![x.to_vec()];
        surrogate.predict(&x_vec).map_or(1e6, |v| v[0])
    };
    let (nm_result, _, _) = multi_start_nelder_mead(
        surrogate_fn,
        bounds,
        8,
        100,
        1e-8,
        42 + round as u64,
    )
    .expect("multi_start_nelder_mead failed");

    let mut round_candidates = Vec::new();

    for _ in 0..candidates_per_round / 4 {
        round_candidates.push(perturb_params(
            &nm_result.x_best,
            bounds,
            rng_state,
            0.1,
        ));
    }

    if let Some(best_rec) = hfb_cache.lock().expect("hfb_cache lock").best() {
        let best_x = best_rec.x.clone();
        for _ in 0..candidates_per_round / 4 {
            round_candidates.push(perturb_params(&best_x, bounds, rng_state, 0.1));
        }
    }

    let lhs_budget = candidates_per_round.saturating_sub(round_candidates.len());
    if lhs_budget > 0 {
        let lhs = latin_hypercube(lhs_budget, bounds, 1000 + round as u64)
            .expect("LHS sampling failed");
        for sample in &lhs {
            round_candidates.push(sample.clone());
        }
    }

    let round_survivors = cascade_filter(
        &round_candidates,
        constraints,
        exp_data,
        use_classifier,
        classifier,
        cascade_stats,
    );

    let results: Vec<(Vec<f64>, f64)> = round_survivors
        .par_iter()
        .map(|params| {
            let obj = l2_objective(params, nuclei);
            (params.clone(), obj)
        })
        .collect();

    for (x, f) in &results {
        hfb_cache
            .lock()
            .expect("hfb_cache lock")
            .record(x.clone(), *f);
    }

    let cache = hfb_cache.lock().expect("hfb_cache lock");
    let n_new = cache.len() - n_before;
    let best_f = cache.best_f().unwrap_or(f64::INFINITY);

    println!(
        "      Round {:2}: +{} HFB evals (total {}), best={:.4}, cascade survived {}/{}, {:.1}s",
        round + 1,
        n_new,
        cache.len(),
        best_f,
        round_survivors.len(),
        round_candidates.len(),
        round_t0.elapsed().as_secs_f64()
    );

    true
}

pub fn run_heterogeneous_l2(
    device: Arc<WgpuDevice>,
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    l1_xs: &[Vec<f64>],
    l1_ys: &[f64],
    clf_result: &ClassifierResult,
    n_rounds: usize,
    candidates_per_round: usize,
) -> (Vec<f64>, f64, EvaluationCache, CascadeStats) {
    let classifier = &clf_result.classifier;
    let use_classifier = clf_result.usable;
    println!("  Phase 3: Heterogeneous L2 with cascade filtering...");
    let constraints = NMPConstraints::default();
    let cascade_stats = Arc::new(Mutex::new(CascadeStats::default()));
    let hfb_cache = Arc::new(Mutex::new(EvaluationCache::new()));

    let init_t0 = Instant::now();

    let (surviving_candidates, mut rng_state) = seed_and_cascade_filter(
        bounds,
        exp_data,
        l1_xs,
        l1_ys,
        use_classifier,
        classifier,
        &cascade_stats,
    );

    let _ = evaluate_hfb_parallel(&surviving_candidates, exp_data, &hfb_cache);

    let elapsed_init = init_t0.elapsed();
    println!("    Initial phase: {:.1}s", elapsed_init.as_secs_f64());
    println!();

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    println!("    Surrogate refinement ({n_rounds} rounds):");

    for round in 0..n_rounds {
        surrogate_refinement_round(
            round,
            device.clone(),
            bounds,
            exp_data,
            &constraints,
            use_classifier,
            classifier,
            &cascade_stats,
            &hfb_cache,
            &nuclei,
            candidates_per_round,
            &mut rng_state,
        );
    }

    let cache = hfb_cache.lock().expect("hfb_cache lock");
    let final_stats = cascade_stats.lock().expect("cascade_stats lock").clone();

    if cache.is_empty() {
        println!("    ⚠ No HFB evaluations completed. Cascade too aggressive.");
        println!("    Cascade stats:");
        final_stats.print_summary();
        let mut best_idx = 0;
        for i in 1..l1_ys.len() {
            if l1_ys[i] < l1_ys[best_idx] {
                best_idx = i;
            }
        }
        return (
            l1_xs[best_idx].clone(),
            l1_ys[best_idx],
            cache.clone(),
            final_stats,
        );
    }

    let best_record = cache.best().expect("cache non-empty, best exists");
    let best_x = best_record.x.clone();
    let best_f = best_record.f;

    (best_x, best_f, cache.clone(), final_stats)
}

pub fn run_plain_l2(
    device: Arc<WgpuDevice>,
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    n_rounds: usize,
) -> (f64, f64, usize) {
    println!("  Plain SparsitySampler L2 (comparison baseline)...");

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    let objective = move |x: &[f64]| -> f64 {
        if x[8] <= 0.01 || x[8] > 1.0 {
            return (1e4_f64).ln_1p();
        }
        l2_objective(x, &nuclei)
    };

    let config = SparsitySamplerConfig::new(bounds.len(), 42)
        .with_initial_samples(100)
        .with_solvers(8)
        .with_eval_budget(50)
        .with_iterations(n_rounds)
        .with_kernel(RBFKernel::ThinPlateSpline);

    let t0 = Instant::now();
    let result =
        sparsity_sampler(device, &objective, bounds, &config).expect("SparsitySampler failed");
    let elapsed = t0.elapsed();

    let chi2 = result.f_best.exp_m1();
    println!(
        "    χ²/datum: {:.2}, time: {:.1}s, evals: {}",
        chi2,
        elapsed.as_secs_f64(),
        result.cache.len()
    );

    (chi2, elapsed.as_secs_f64(), result.cache.len())
}

pub fn run_screen_l2(
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    n_l1_samples: usize,
    n_l2_evals: usize,
) -> (Vec<f64>, f64, f64, usize) {
    println!("  L1-screen → L2-evaluate approach (no optimization)...");
    println!("    L1 samples: {n_l1_samples}, L2 evaluations: {n_l2_evals}");
    println!();

    let (l1_xs, l1_ys) = generate_l1_training_data(bounds, exp_data, n_l1_samples)
        .expect("L1 training data generation failed");

    let mut indices: Vec<usize> = (0..l1_ys.len()).collect();
    indices.sort_by(|&a, &b| l1_ys[a].total_cmp(&l1_ys[b]));
    let n_eval = n_l2_evals.min(indices.len());
    let candidates: Vec<Vec<f64>> = indices[..n_eval]
        .iter()
        .map(|&i| l1_xs[i].clone())
        .collect();

    println!(
        "    Best L1 score: log(1+χ²) = {:.4} (top candidate)",
        l1_ys[indices[0]]
    );
    println!(
        "    Worst selected L1 score: {:.4} (cutoff at rank {})",
        l1_ys[indices[n_eval - 1]],
        n_eval
    );
    println!();

    println!("  Evaluating {n_eval} candidates with L2 HFB (rayon parallel)...");
    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    let t0 = Instant::now();
    let results: Vec<(Vec<f64>, f64)> = candidates
        .par_iter()
        .map(|params| {
            let f = l2_objective(params, &nuclei);
            (params.clone(), f)
        })
        .collect();
    let elapsed = t0.elapsed();

    let (best_x, best_f) = results
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(x, f)| (x.clone(), *f))
        .expect("at least one L2 evaluation result");

    let chi2 = best_f.exp_m1();
    let time_s = elapsed.as_secs_f64();
    data::print_l2_result_box(
        "L1-Screen → L2-Eval Results",
        chi2,
        best_f,
        n_eval,
        time_s,
        n_eval as f64 / time_s,
    );

    let mut scores: Vec<f64> = results.iter().map(|(_, f)| *f).collect();
    scores.sort_by(f64::total_cmp);
    println!();
    println!("  L2 score distribution (log(1+χ²)):");
    println!("    Best:   {:.4}", scores[0]);
    println!("    p10:    {:.4}", scores[n_eval / 10]);
    println!("    Median: {:.4}", scores[n_eval / 2]);
    println!("    p90:    {:.4}", scores[n_eval * 9 / 10]);
    println!("    Worst:  {:.4}", scores[n_eval - 1]);

    let n_penalty = scores.iter().filter(|&&s| s > 9.0).count();
    let n_good = scores.iter().filter(|&&s| s < 5.0).count();
    println!(
        "    Penalty (>9.0): {}/{} ({:.1}%)",
        n_penalty,
        n_eval,
        100.0 * n_penalty as f64 / n_eval as f64
    );
    println!(
        "    Good (<5.0):    {}/{} ({:.1}%)",
        n_good,
        n_eval,
        100.0 * n_good as f64 / n_eval as f64
    );

    if let Some(nmp) = nuclear_matter_properties(&best_x) {
        println!();
        provenance::print_nmp_analysis(&nmp);
    }

    (best_x, best_f, elapsed.as_secs_f64(), n_eval)
}

pub fn run_direct_l2(
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    n_starts: usize,
    max_evals: usize,
) -> (Vec<f64>, f64, f64, usize) {
    use barracuda::optimize::nelder_mead;

    println!("  Direct L1-seeded NM on L2 HFB objective (no surrogate)...");
    println!("    n_starts: {n_starts} (from best L1 solutions)");
    println!("    max_evals/start: {max_evals}");
    println!();

    let (l1_xs, l1_ys) = generate_l1_training_data(bounds, exp_data, 5000)
        .expect("L1 training data generation failed");

    let mut indices: Vec<usize> = (0..l1_ys.len()).collect();
    indices.sort_by(|&a, &b| l1_ys[a].total_cmp(&l1_ys[b]));
    let n_seeds = n_starts.min(indices.len());
    let seeds: Vec<Vec<f64>> = indices[..n_seeds]
        .iter()
        .map(|&i| l1_xs[i].clone())
        .collect();

    println!(
        "    Best L1 log(1+χ²) = {:.4} (seed quality)",
        l1_ys[indices[0]]
    );
    println!();

    let nuclei: Vec<(usize, usize, f64)> = exp_data
        .iter()
        .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
        .collect();

    let eval_count = std::sync::atomic::AtomicUsize::new(0);
    let hfb_count = std::sync::atomic::AtomicUsize::new(0);

    let objective = |x: &[f64]| -> f64 {
        eval_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if x[8] <= 0.01 || x[8] > 1.0 {
            return (1e4_f64).ln_1p();
        }
        hfb_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        l2_objective(x, &nuclei)
    };

    println!("  Running {n_seeds} seeded NM starts on true L2 objective...");
    let t0 = Instant::now();

    let mut best_x = seeds[0].clone();
    let mut best_f = f64::INFINITY;
    let mut total_hfb = 0;

    for (i, seed) in seeds.iter().enumerate() {
        let (x_star, f_star, _) = nelder_mead(objective, seed, bounds, max_evals, 1e-8)
            .unwrap_or_else(|_| (seed.clone(), f64::INFINITY, 0));

        if f_star < best_f {
            best_f = f_star;
            best_x = x_star;
        }

        let hfb_so_far = hfb_count.load(std::sync::atomic::Ordering::Relaxed);
        if (i + 1) % 5 == 0 || i == n_seeds - 1 {
            println!(
                "    [{}/{}] best log(1+χ²) = {:.4}, HFB evals = {}",
                i + 1,
                n_seeds,
                best_f,
                hfb_so_far
            );
        }
        total_hfb = hfb_so_far;
    }

    let elapsed = t0.elapsed();
    let time_s = elapsed.as_secs_f64();
    let chi2 = best_f.exp_m1();
    data::print_l2_result_box(
        "Direct L2 Results (L1-seeded NM)",
        chi2,
        best_f,
        total_hfb,
        time_s,
        total_hfb as f64 / time_s,
    );

    if let Some(nmp) = nuclear_matter_properties(&best_x) {
        println!();
        provenance::print_nmp_analysis(&nmp);
    }

    (best_x, best_f, elapsed.as_secs_f64(), total_hfb)
}
