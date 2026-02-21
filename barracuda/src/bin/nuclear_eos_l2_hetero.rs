// SPDX-License-Identifier: AGPL-3.0-only

//! Nuclear EOS Level 2 — Heterogeneous Pipeline
//!
//! Solves the L2 `SparsitySampler` quality problem using:
//!   1. L1 data to warm-start the search (previous work informs later)
//!   2. Three-tier pre-screening cascade to filter before expensive HFB
//!   3. NPU/CPU classifier trained on accumulated data
//!   4. Modified sampling loop where surrogate sees ONLY real HFB values
//!
//! **Heterogeneous compute architecture:**
//!   - CPU: NMP pre-screening (Tier 1, ~1μs), L1 proxy (Tier 2, ~100μs)
//!   - NPU/CPU: Learned classifier (Tier 3, ~1μs)
//!   - CPU parallel (rayon): HFB evaluation (Tier 4, ~200ms) — only for survivors
//!   - GPU (barracuda WGSL): RBF surrogate training (cdist shader)
//!
//! **Comparison modes:**
//!   - `--mode=plain`  — standard `SparsitySampler` (baseline, for comparison)
//!   - `--mode=hetero` — heterogeneous pipeline (default)

use hotspring_barracuda::data;
use hotspring_barracuda::physics::hfb::binding_energy_l2;
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::prescreen::{
    l1_proxy_prescreen, nmp_prescreen, CascadeStats, NMPConstraints, NMPScreenResult,
    PreScreenClassifier,
};
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

use barracuda::optimize::{multi_start_nelder_mead, EvaluationCache};
use barracuda::sample::latin_hypercube;
use barracuda::sample::sparsity::{sparsity_sampler, SparsitySamplerConfig};
use barracuda::surrogate::{RBFKernel, RBFSurrogate};

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════
// Local helpers for parameter perturbation and cascade filtering
// ═══════════════════════════════════════════════════════════════════

/// Perturb base parameters by random amounts scaled by `scale` (fraction of
/// parameter range, e.g. 0.1 = ±5%, 0.2 = ±10%), then clamp to bounds.
fn perturb_params(base: &[f64], bounds: &[(f64, f64)], rng: &mut u64, scale: f64) -> Vec<f64> {
    let mut candidate = base.to_vec();
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        let range = hi - lo;
        *rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (*rng >> 33) as f64 / (1u64 << 31) as f64;
        let perturbation = (u - 0.5) * scale * range;
        candidate[i] = (candidate[i] + perturbation).clamp(lo, hi);
    }
    candidate
}

/// Run candidates through NMP (Tier 1), L1 proxy (Tier 2), and classifier (Tier 3).
/// Returns surviving candidates and updates `cascade_stats`.
fn cascade_filter(
    candidates: &[Vec<f64>],
    constraints: &NMPConstraints,
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    use_classifier: bool,
    classifier: &PreScreenClassifier,
    cascade_stats: &Arc<Mutex<CascadeStats>>,
) -> Vec<Vec<f64>> {
    let mut survivors = Vec::new();
    for params in candidates {
        let mut stats = cascade_stats.lock().expect("cascade_stats lock");
        stats.total_candidates += 1;

        match nmp_prescreen(params, constraints) {
            NMPScreenResult::Fail(_) => {
                stats.tier1_rejected += 1;
                continue;
            }
            NMPScreenResult::Pass(_) => {}
        }
        if l1_proxy_prescreen(params, exp_data, 200.0).is_none() {
            stats.tier2_rejected += 1;
            continue;
        }
        if use_classifier && !classifier.predict(params) {
            stats.tier3_rejected += 1;
            continue;
        }
        stats.tier4_evaluated += 1;
        drop(stats);

        survivors.push(params.clone());
    }
    survivors
}

// ═══════════════════════════════════════════════════════════════════
// Phase 1: Generate L1 training data (fast, previous work)
// ═══════════════════════════════════════════════════════════════════

fn generate_l1_training_data(
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    n_samples: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    println!("  Phase 1: Generating L1 training data ({n_samples} samples)...");
    let t0 = Instant::now();

    let samples = latin_hypercube(n_samples, bounds, 42).expect("LHS sampling failed");
    let mut xs = Vec::with_capacity(n_samples);
    let mut ys = Vec::with_capacity(n_samples);

    for sample in &samples {
        let params = sample.clone();

        // Compute L1 objective
        if params[8] <= 0.01 || params[8] > 1.0 {
            xs.push(params);
            ys.push(9.2); // ln(1+10000) penalty
            continue;
        }

        let Some(nmp) = nuclear_matter_properties(&params) else {
            xs.push(params);
            ys.push(9.2);
            continue;
        };

        let mut penalty = 0.0;
        if nmp.rho0_fm3 < 0.08 {
            penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08;
        } else if nmp.rho0_fm3 > 0.25 {
            penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25;
        }
        if nmp.e_a_mev > -5.0 {
            penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0);
        }

        let mut chi2 = 0.0;
        let mut n_valid = 0;
        for (&(z, n), &(b_exp, _sigma)) in exp_data {
            let b_calc = semf_binding_energy(z, n, &params);
            if b_calc > 0.0 {
                let sigma_theo = tolerances::sigma_theo(b_exp);
                chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
                n_valid += 1;
            }
        }

        let obj = if n_valid > 0 {
            (chi2 / f64::from(n_valid) + penalty).ln_1p()
        } else {
            9.2
        };

        xs.push(params);
        ys.push(obj);
    }

    let n_good = ys.iter().filter(|&&y| y < 5.0).count();
    println!(
        "    Generated {} samples in {:.2}s",
        n_samples,
        t0.elapsed().as_secs_f64()
    );
    println!(
        "    Good (log(1+χ²) < 5): {} ({:.1}%)",
        n_good,
        100.0 * n_good as f64 / n_samples as f64
    );
    println!();

    (xs, ys)
}

// ═══════════════════════════════════════════════════════════════════
// Phase 2: Train pre-screening classifier
// ═══════════════════════════════════════════════════════════════════

/// Result of classifier training, including quality metrics
struct ClassifierResult {
    classifier: PreScreenClassifier,
    _recall: f64,
    usable: bool, // true if recall is high enough to trust the classifier
}

fn train_classifier(xs: &[Vec<f64>], ys: &[f64]) -> ClassifierResult {
    println!("  Phase 2: Training pre-screening classifier...");
    let t0 = Instant::now();

    // Use a generous label threshold: log(1+χ²) < 7.0 (~χ² < 1096)
    // This captures a wider band of "possibly reasonable" parameter sets
    let label_threshold = 7.0;
    let mut classifier = PreScreenClassifier::train(xs, ys, label_threshold);

    // Test accuracy on training data
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_ = 0;

    for (x, &y) in xs.iter().zip(ys.iter()) {
        let pred = classifier.predict(x);
        let actual = y < label_threshold;
        if pred && actual {
            tp += 1;
        }
        if pred && !actual {
            fp += 1;
        }
        if !pred && actual {
            fn_ += 1;
        }
        if !pred && !actual {
            tn += 1;
        }
    }

    let n_positive = tp + fn_;
    let recall = if n_positive > 0 {
        f64::from(tp) / f64::from(n_positive)
    } else {
        0.0
    };
    let precision = if tp + fp > 0 {
        f64::from(tp) / f64::from(tp + fp)
    } else {
        0.0
    };
    let accuracy = f64::from(tp + tn) / xs.len().max(1) as f64;

    println!(
        "    Trained in {:.3}s on {} samples ({} positive at threshold={:.1})",
        t0.elapsed().as_secs_f64(),
        xs.len(),
        n_positive,
        label_threshold
    );
    println!("    Accuracy:  {:.1}%", accuracy * 100.0);
    println!("    Precision: {:.1}%", precision * 100.0);
    println!("    Recall:    {:.1}%", recall * 100.0);
    println!("    [TP={tp}, FP={fp}, FN={fn_}, TN={tn}]");

    // If recall is poor, lower the decision threshold to be more permissive
    let usable = if recall < 0.5 && n_positive > 5 {
        // Try lowering threshold until recall ≥ 50%
        let mut best_threshold = 0.5;
        let mut best_recall = recall;
        for t in (5..50).map(|i| f64::from(i) / 100.0) {
            classifier.threshold = t;
            let mut tp2 = 0;
            let mut fn2 = 0;
            for (x, &y) in xs.iter().zip(ys.iter()) {
                let pred = classifier.predict(x);
                let actual = y < label_threshold;
                if pred && actual {
                    tp2 += 1;
                }
                if !pred && actual {
                    fn2 += 1;
                }
            }
            let r = if tp2 + fn2 > 0 {
                f64::from(tp2) / f64::from(tp2 + fn2)
            } else {
                0.0
            };
            if r > best_recall {
                best_recall = r;
                best_threshold = t;
            }
            if r >= 0.5 {
                break;
            }
        }
        classifier.threshold = best_threshold;
        println!(
            "    ⚠ Low recall — adjusted threshold to {:.2} (recall now {:.1}%)",
            best_threshold,
            best_recall * 100.0
        );
        best_recall >= 0.1 // usable if at least 10% recall after adjustment
    } else if n_positive <= 5 {
        println!("    ⚠ Too few positives ({n_positive}) — classifier DISABLED (bypass Tier 3)");
        false
    } else {
        true
    };

    if !usable {
        println!("    → Cascade will skip Tier 3 (classifier), using Tiers 1+2 only");
    }
    println!();

    ClassifierResult {
        classifier,
        _recall: recall,
        usable,
    }
}

// ═══════════════════════════════════════════════════════════════════
// Phase 3: Heterogeneous L2 with cascade filtering
// ═══════════════════════════════════════════════════════════════════

fn run_heterogeneous_l2(
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

    // ── Warm-start: seed from best L1 regions ──
    println!("    Seeding from best L1 regions...");
    let t0 = Instant::now();

    // Find best L1 solutions
    let mut indices: Vec<usize> = (0..l1_ys.len()).collect();
    indices.sort_by(|&a, &b| l1_ys[a].total_cmp(&l1_ys[b]));
    let top_k = indices.len().min(50); // top 50 L1 solutions
    let seed_params: Vec<&Vec<f64>> = indices[..top_k].iter().map(|&i| &l1_xs[i]).collect();

    // Generate candidates near seed points
    let mut initial_candidates = Vec::new();
    let mut rng_state = 12345_u64;

    for seed in &seed_params {
        for _ in 0..20 {
            initial_candidates.push(perturb_params(seed, bounds, &mut rng_state, 0.2));
            // ±10% of range
        }
    }

    // Also add some LHS samples for exploration
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

    // ── Filter initial candidates through cascade ──
    let mut surviving_candidates: Vec<Vec<f64>> = Vec::new();

    for params in &initial_candidates {
        let mut stats = cascade_stats.lock().expect("cascade_stats lock");
        stats.total_candidates += 1;

        // Tier 1: NMP pre-screen (algebraic, ~0 cost)
        match nmp_prescreen(params, &constraints) {
            NMPScreenResult::Fail(_) => {
                stats.tier1_rejected += 1;
                continue;
            }
            NMPScreenResult::Pass(_) => {}
        }

        // Tier 2: L1 proxy pre-screen (~0.1ms)
        if l1_proxy_prescreen(params, exp_data, 200.0).is_none() {
            stats.tier2_rejected += 1;
            continue;
        }

        // Tier 3: Classifier pre-screen (only if usable)
        if use_classifier && !classifier.predict(params) {
            stats.tier3_rejected += 1;
            continue;
        }

        stats.tier4_evaluated += 1;
        drop(stats);

        surviving_candidates.push(params.clone());
    }

    println!(
        "    Survived cascade: {} / {} ({:.1}%)",
        surviving_candidates.len(),
        initial_candidates.len(),
        100.0 * surviving_candidates.len() as f64 / initial_candidates.len() as f64
    );

    // ── Evaluate survivors with HFB (expensive, parallel) ──
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

    let elapsed_init = t0.elapsed();
    println!("    Initial phase: {:.1}s", elapsed_init.as_secs_f64());
    println!();

    // ── Iterative surrogate refinement with cascade filtering ──
    println!("    Surrogate refinement ({n_rounds} rounds):");

    for round in 0..n_rounds {
        let round_t0 = Instant::now();

        // Get current cache data
        let (train_x, train_y) = hfb_cache.lock().expect("hfb_cache lock").training_data();
        let n_before = train_x.len();

        if train_x.len() < 5 {
            println!(
                "      Round {}: too few data points ({}), skipping",
                round + 1,
                train_x.len()
            );
            continue;
        }

        // Build RBF surrogate on real HFB values (no penalties!)
        let surrogate =
            match RBFSurrogate::train(&train_x, &train_y, RBFKernel::ThinPlateSpline, 0.0) {
                Ok(s) => s,
                Err(e) => {
                    println!(
                        "      Round {}: surrogate training failed: {:?}",
                        round + 1,
                        e
                    );
                    continue;
                }
            };

        // Multi-start Nelder-Mead on NMP-aware surrogate
        // The surrogate prediction is penalized for NMP-invalid regions,
        // steering the optimizer towards physically meaningful solutions
        let surrogate_fn = |x: &[f64]| -> f64 {
            // Bounds check
            if x.iter()
                .zip(bounds.iter())
                .any(|(&v, &(lo, hi))| v < lo || v > hi)
            {
                return 1e10;
            }
            // NMP pre-screen: penalize unphysical regions heavily
            match nmp_prescreen(x, &constraints) {
                NMPScreenResult::Fail(_) => return 1e6,
                NMPScreenResult::Pass(_) => {}
            }
            // L1 proxy: penalize if SEMF is terrible
            if l1_proxy_prescreen(x, exp_data, 200.0).is_none() {
                return 1e5;
            }
            // If physical, use surrogate prediction
            let x_vec = vec![x.to_vec()];
            surrogate.predict(&x_vec).map(|v| v[0]).unwrap_or(1e6)
        };
        let (nm_result, _, _) = multi_start_nelder_mead(
            surrogate_fn,
            bounds,
            8,    // n_starts
            100,  // max_evals per start
            1e-8, // tol
            42 + round as u64,
        )
        .expect("multi_start_nelder_mead failed");

        // Generate candidates: NM best + perturbations + best-so-far perturbations + LHS
        let mut round_candidates = Vec::new();

        // NM best point + perturbations (tighter: ±5% of range for exploitation)
        for _ in 0..candidates_per_round / 4 {
            round_candidates.push(perturb_params(
                &nm_result.x_best,
                bounds,
                &mut rng_state,
                0.1,
            ));
        }

        // Best-so-far point + perturbations (exploit known good region)
        if let Some(best_rec) = hfb_cache.lock().expect("hfb_cache lock").best() {
            let best_x = best_rec.x.clone();
            for _ in 0..candidates_per_round / 4 {
                round_candidates.push(perturb_params(&best_x, bounds, &mut rng_state, 0.1));
            }
        }

        // LHS exploration (remaining budget)
        let lhs_budget = candidates_per_round.saturating_sub(round_candidates.len());
        if lhs_budget > 0 {
            let lhs = latin_hypercube(lhs_budget, bounds, 1000 + round as u64)
                .expect("LHS sampling failed");
            for sample in &lhs {
                round_candidates.push(sample.clone());
            }
        }

        // Filter through cascade (NMP-aware surrogate already steered NM candidates,
        // but random LHS still needs filtering)
        let round_survivors = cascade_filter(
            &round_candidates,
            &constraints,
            exp_data,
            use_classifier,
            classifier,
            &cascade_stats,
        );

        // Evaluate survivors with HFB
        let results: Vec<(Vec<f64>, f64)> = round_survivors
            .par_iter()
            .map(|params| {
                let obj = l2_objective(params, &nuclei);
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

        println!("      Round {:2}: +{} HFB evals (total {}), best={:.4}, cascade survived {}/{}, {:.1}s",
            round + 1, n_new, cache.len(), best_f,
            round_survivors.len(), round_candidates.len(),
            round_t0.elapsed().as_secs_f64());
    }

    // ── Final results ──
    let cache = hfb_cache.lock().expect("hfb_cache lock");
    let final_stats = cascade_stats.lock().expect("cascade_stats lock").clone();

    if cache.is_empty() {
        println!("    ⚠ No HFB evaluations completed. Cascade too aggressive.");
        println!("    Cascade stats:");
        final_stats.print_summary();
        // Return the best L1 solution as fallback
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

/// L2 objective function (HFB evaluation)
fn l2_objective(params: &[f64], nuclei: &[(usize, usize, f64)]) -> f64 {
    let Some(nmp) = nuclear_matter_properties(params) else {
        return (1e4_f64).ln_1p();
    };

    let mut penalty = 0.0;
    if nmp.rho0_fm3 < 0.08 {
        penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08;
    } else if nmp.rho0_fm3 > 0.25 {
        penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25;
    }
    if nmp.e_a_mev > -5.0 {
        penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0);
    }

    let results: Vec<(f64, f64)> = nuclei
        .iter()
        .map(|&(z, n, b_exp)| {
            let (b_calc, _conv) = binding_energy_l2(z, n, params).expect("HFB solve");
            (b_calc, b_exp)
        })
        .collect();

    let mut chi2 = 0.0;
    let mut n_valid = 0;
    for (b_calc, b_exp) in results {
        if b_calc > 0.0 {
            let sigma_theo = tolerances::sigma_theo(b_exp);
            chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
            n_valid += 1;
        }
    }

    if n_valid == 0 {
        return (1e4_f64).ln_1p();
    }

    (chi2 / f64::from(n_valid) + penalty).ln_1p()
}

// ═══════════════════════════════════════════════════════════════════
// Phase 4 (comparison): Plain SparsitySampler L2
// ═══════════════════════════════════════════════════════════════════

fn run_plain_l2(
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
    let result = sparsity_sampler(&objective, bounds, &config).expect("SparsitySampler failed");
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

// ═══════════════════════════════════════════════════════════════════
// Direct multi-start NM on true L2 objective (no surrogate)
// ═══════════════════════════════════════════════════════════════════

fn run_screen_l2(
    bounds: &[(f64, f64)],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    n_l1_samples: usize,
    n_l2_evals: usize,
) -> (Vec<f64>, f64, f64, usize) {
    println!("  L1-screen → L2-evaluate approach (no optimization)...");
    println!("    L1 samples: {n_l1_samples}, L2 evaluations: {n_l2_evals}");
    println!();

    // Phase 1: Generate and score with L1
    let (l1_xs, l1_ys) = generate_l1_training_data(bounds, exp_data, n_l1_samples);

    // Sort by L1 score, take top-k for L2 evaluation
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

    // Phase 2: Evaluate ALL candidates with L2 HFB (parallel via rayon)
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

    // Find best
    let (best_x, best_f) = results
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(x, f)| (x.clone(), *f))
        .expect("at least one L2 evaluation result");

    let chi2 = best_f.exp_m1();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  L1-Screen → L2-Eval Results                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  χ²/datum:       {chi2:12.4}                              ║");
    println!("║  log(1+χ²):      {best_f:12.4}                              ║");
    println!("║  L2 evals:       {n_eval:6}                                    ║");
    println!(
        "║  Time (L2):      {:6.1}s                                   ║",
        elapsed.as_secs_f64()
    );
    println!(
        "║  HFB throughput: {:6.1} evals/s                            ║",
        n_eval as f64 / elapsed.as_secs_f64()
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Distribution of L2 scores
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

fn run_direct_l2(
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

    // Phase 1: Generate L1 data to find good seed regions
    let (l1_xs, l1_ys) = generate_l1_training_data(bounds, exp_data, 5000);

    // Sort by L1 score, take top-k as NM starting points
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

    // L2 objective with NMP fast-reject (penalty = 1e4 to match Python control)
    let objective = |x: &[f64]| -> f64 {
        eval_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if x[8] <= 0.01 || x[8] > 1.0 {
            return (1e4_f64).ln_1p();
        }
        hfb_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        l2_objective(x, &nuclei)
    };

    // Phase 2: Run NM from each L1 seed on the TRUE L2 objective
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
    let total_evals = eval_count.load(std::sync::atomic::Ordering::Relaxed);
    let chi2 = best_f.exp_m1();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Direct L2 Results (L1-seeded NM)                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  χ²/datum:       {chi2:12.4}                              ║");
    println!("║  log(1+χ²):      {best_f:12.4}                              ║");
    println!("║  Total evals:    {total_evals:6}  (obj calls)                      ║");
    println!("║  HFB evals:      {total_hfb:6}  (expensive)                      ║");
    println!(
        "║  Time:           {:6.1}s  (NM only)                       ║",
        elapsed.as_secs_f64()
    );
    println!(
        "║  HFB throughput: {:6.1} evals/s                            ║",
        total_hfb as f64 / elapsed.as_secs_f64()
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Nuclear matter at best
    if let Some(nmp) = nuclear_matter_properties(&best_x) {
        println!();
        provenance::print_nmp_analysis(&nmp);
    }

    (best_x, best_f, elapsed.as_secs_f64(), total_hfb)
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — Heterogeneous Pipeline                   ║");
    println!("║  Architecture: L1 warm-start → cascade filter → HFB       ║");
    println!("║  Compute: CPU(NMP) → NPU/CPU(clf) → CPU∥(HFB) → GPU(RBF) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Parse args ──
    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .iter()
        .find(|a| a.starts_with("--mode="))
        .and_then(|a| a.strip_prefix("--mode="))
        .unwrap_or("hetero");
    let n_rounds = args
        .iter()
        .find(|a| a.starts_with("--rounds="))
        .and_then(|a| a.strip_prefix("--rounds=")?.parse().ok())
        .unwrap_or(15);
    let l1_samples = args
        .iter()
        .find(|a| a.starts_with("--l1-samples="))
        .and_then(|a| a.strip_prefix("--l1-samples=")?.parse().ok())
        .unwrap_or(5000);
    let candidates_per_round = args
        .iter()
        .find(|a| a.starts_with("--candidates="))
        .and_then(|a| a.strip_prefix("--candidates=")?.parse().ok())
        .unwrap_or(200);

    // ── Load data ──
    let ctx = data::load_eos_context().expect("Failed to load EOS context");
    let base = &ctx.base;
    let exp_data = &*ctx.exp_data;
    let bounds = &ctx.bounds;

    println!("  Experimental nuclei: {}", exp_data.len());
    println!("  Mode: {mode}");
    println!("  Rayon threads: {}", rayon::current_num_threads());
    println!();

    let total_t0 = Instant::now();

    match mode {
        "both" | "compare" => {
            // Run both for comparison
            println!("═══════════════════════════════════════════════════════════");
            println!("  HETEROGENEOUS PIPELINE");
            println!("═══════════════════════════════════════════════════════════");

            let (l1_xs, l1_ys) = generate_l1_training_data(bounds, exp_data, l1_samples);
            let clf_result = train_classifier(&l1_xs, &l1_ys);
            let (best_params, best_f, cache, stats) = run_heterogeneous_l2(
                bounds,
                exp_data,
                &l1_xs,
                &l1_ys,
                &clf_result,
                n_rounds,
                candidates_per_round,
            );
            let hetero_time = total_t0.elapsed().as_secs_f64();
            let hetero_chi2 = best_f.exp_m1();

            println!();
            stats.print_summary();
            println!();

            println!("═══════════════════════════════════════════════════════════");
            println!("  PLAIN SPARSITYSAMPLER (COMPARISON)");
            println!("═══════════════════════════════════════════════════════════");

            let _plain_t0 = Instant::now();
            let (plain_chi2, plain_time, plain_evals) = run_plain_l2(bounds, exp_data, n_rounds);

            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  COMPARISON                                                ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  Heterogeneous:                                            ║");
            println!("║    χ²/datum = {hetero_chi2:12.2}                                  ║");
            println!("║    Time:      {hetero_time:6.1}s                                      ║");
            println!(
                "║    HFB evals: {:6}                                        ║",
                cache.len()
            );
            println!(
                "║    Cascade pass rate: {:.1}%                                ║",
                stats.pass_rate() * 100.0
            );
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  Plain SparsitySampler:                                    ║");
            println!("║    χ²/datum = {plain_chi2:12.2}                                  ║");
            println!("║    Time:      {plain_time:6.1}s                                      ║");
            println!("║    HFB evals: {plain_evals:6}                                        ║");
            println!("╚══════════════════════════════════════════════════════════════╝");

            // Save comparison results
            save_results(
                base,
                &best_params,
                best_f,
                &cache,
                &stats,
                hetero_time,
                Some(plain_chi2),
                Some(plain_time),
                Some(plain_evals),
            );
        }

        "screen" => {
            // Pure L1 screening: evaluate best L1 solutions with L2 HFB, no NM
            let n_l1 = args
                .iter()
                .find(|a| a.starts_with("--l1-samples="))
                .and_then(|a| a.strip_prefix("--l1-samples=")?.parse().ok())
                .unwrap_or(10_000);
            let n_eval = args
                .iter()
                .find(|a| a.starts_with("--l2-evals="))
                .and_then(|a| a.strip_prefix("--l2-evals=")?.parse().ok())
                .unwrap_or(200);

            let (best_params, best_f, total_time, total_evals) =
                run_screen_l2(bounds, exp_data, n_l1, n_eval);

            let chi2 = best_f.exp_m1();
            let results_dir = base.join("results");
            std::fs::create_dir_all(&results_dir).ok();
            let result_json = serde_json::json!({
                "level": 2,
                "engine": "barracuda::l1_screen_l2_eval",
                "chi2_per_datum": chi2,
                "log_chi2": best_f,
                "total_l2_evals": total_evals,
                "l1_samples": n_l1,
                "time_seconds": total_time,
                "best_params": best_params,
            });
            let path = results_dir.join("barracuda_screen_l2.json");
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&result_json).expect("JSON serialize"),
            )
            .ok();
            println!("\n  Results saved to: {}", path.display());
        }

        "direct" => {
            // Direct multi-start NM on true L2 objective (no surrogate)
            let n_starts = args
                .iter()
                .find(|a| a.starts_with("--starts="))
                .and_then(|a| a.strip_prefix("--starts=")?.parse().ok())
                .unwrap_or(10);
            let max_evals_per = args
                .iter()
                .find(|a| a.starts_with("--evals="))
                .and_then(|a| a.strip_prefix("--evals=")?.parse().ok())
                .unwrap_or(200);

            let (best_params, best_f, total_time, total_evals) =
                run_direct_l2(bounds, exp_data, n_starts, max_evals_per);

            let chi2 = best_f.exp_m1();

            // Save results
            let results_dir = base.join("results");
            std::fs::create_dir_all(&results_dir).ok();
            let result_json = serde_json::json!({
                "level": 2,
                "engine": "barracuda::direct_multi_start_nm",
                "chi2_per_datum": chi2,
                "log_chi2": best_f,
                "total_evals": total_evals,
                "n_starts": n_starts,
                "max_evals_per_start": max_evals_per,
                "time_seconds": total_time,
                "best_params": best_params,
            });
            let path = results_dir.join("barracuda_direct_l2.json");
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&result_json).expect("JSON serialize"),
            )
            .ok();
            println!("\n  Results saved to: {}", path.display());
        }

        _ => {
            // Heterogeneous only (default)
            let (l1_xs, l1_ys) = generate_l1_training_data(bounds, exp_data, l1_samples);
            let clf_result = train_classifier(&l1_xs, &l1_ys);
            let (best_params, best_f, cache, stats) = run_heterogeneous_l2(
                bounds,
                exp_data,
                &l1_xs,
                &l1_ys,
                &clf_result,
                n_rounds,
                candidates_per_round,
            );
            let total_time = total_t0.elapsed().as_secs_f64();

            let chi2 = best_f.exp_m1();

            println!();
            println!("╔══════════════════════════════════════════════════════════════╗");
            println!("║  Heterogeneous L2 Results                                  ║");
            println!("╠══════════════════════════════════════════════════════════════╣");
            println!("║  χ²/datum:       {chi2:12.4}                              ║");
            println!("║  log(1+χ²):      {best_f:12.4}                              ║");
            println!(
                "║  HFB evals:      {:6}                                    ║",
                cache.len()
            );
            println!("║  Total time:     {total_time:6.1}s                                   ║");
            println!(
                "║  HFB throughput: {:6.1} evals/s                            ║",
                cache.len() as f64 / total_time
            );
            println!("╚══════════════════════════════════════════════════════════════╝");
            println!();

            stats.print_summary();

            // Nuclear matter at best
            if let Some(nmp) = nuclear_matter_properties(&best_params) {
                println!();
                provenance::print_nmp_analysis(&nmp);
            }

            save_results(
                base,
                &best_params,
                best_f,
                &cache,
                &stats,
                total_time,
                None,
                None,
                None,
            );
        }
    }
}

fn save_results(
    base: &std::path::Path,
    best_params: &[f64],
    best_f: f64,
    cache: &EvaluationCache,
    stats: &CascadeStats,
    total_time: f64,
    plain_chi2: Option<f64>,
    plain_time: Option<f64>,
    plain_evals: Option<usize>,
) {
    let chi2 = best_f.exp_m1();
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();

    let mut result_json = serde_json::json!({
        "level": 2,
        "engine": "hotspring::heterogeneous_pipeline",
        "architecture": {
            "tier1": "NMP algebraic pre-screen (CPU)",
            "tier2": "L1 SEMF proxy (CPU)",
            "tier3": "logistic regression classifier (NPU-ready, CPU fallback)",
            "tier4": "spherical HFB (CPU parallel, rayon)",
            "surrogate": "barracuda::RBFSurrogate (GPU WGSL cdist)",
        },
        "chi2_per_datum": chi2,
        "log_chi2": best_f,
        "total_hfb_evals": cache.len(),
        "time_seconds": total_time,
        "cascade_stats": {
            "total_candidates": stats.total_candidates,
            "tier1_rejected": stats.tier1_rejected,
            "tier2_rejected": stats.tier2_rejected,
            "tier3_rejected": stats.tier3_rejected,
            "tier4_evaluated": stats.tier4_evaluated,
            "pass_rate": stats.pass_rate(),
        },
        "best_params": best_params,
    });

    if let (Some(pc), Some(pt), Some(pe)) = (plain_chi2, plain_time, plain_evals) {
        result_json["comparison"] = serde_json::json!({
            "plain_chi2_per_datum": pc,
            "plain_time_s": pt,
            "plain_evals": pe,
            "hetero_speedup": pt / total_time,
            "hetero_chi2_improvement": (pc - chi2) / pc,
        });
    }

    let path = results_dir.join("barracuda_l2_hetero.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&result_json).expect("JSON serialize"),
    )
    .ok();
    println!();
    println!("  Results saved to: {}", path.display());
}
