// SPDX-License-Identifier: AGPL-3.0-only

//! **DEPRECATED** — Superseded by `bin/nuclear_eos_l1_ref.rs`.
//!
//! This was the first L1 pipeline using barracuda's SparsitySampler.
//! It has been replaced by the more complete `nuclear_eos_l1_ref` binary
//! which adds NMP prescreening, cascade statistics, and heterogeneous
//! GPU/CPU routing. Retained as fossil record only.
//!
//! # Original purpose (historical)
//!
//! Nuclear EOS Level 1 — Surrogate Learning via BarraCUDA.
//! Full SparsitySampler workflow using barracuda library modules.

use hotspring_barracuda::data;
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};

use barracuda::sample::sparsity::{sparsity_sampler, SparsitySamplerConfig};
use barracuda::surrogate::RBFKernel;

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L1 — BarraCUDA SparsitySampler Validation     ║");
    println!("║  Using: barracuda::sample::sparsity_sampler (library)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Load data ──────────────────────────────────────────────────
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("control/surrogate/nuclear-eos");

    let exp_data = data::load_experimental_data(&base.join("exp_data/ame2020_selected.json"))
        .expect("Failed to load experimental data");
    let bounds = data::load_bounds(&base.join("wrapper/skyrme_bounds.json"))
        .expect("Failed to load parameter bounds");

    println!("  Experimental nuclei: {}", exp_data.len());
    println!("  Parameters:          {} dimensions", bounds.len());
    println!("  Bounds:");
    for (i, name) in data::PARAM_NAMES.iter().enumerate() {
        println!("    {:6} ∈ [{:10.1}, {:10.1}]", name, bounds[i].0, bounds[i].1);
    }
    println!();

    // ── Define objective function (physics) ────────────────────────
    // This is the ONLY application-specific code. Everything else is barracuda.
    let exp_data_ref = &exp_data;
    let objective = move |x: &[f64]| -> f64 {
        // Alpha sanity
        if x[8] <= 0.01 || x[8] > 1.0 {
            return (1e4_f64).ln_1p();
        }

        // Nuclear matter properties
        let nmp = match nuclear_matter_properties(x) {
            Some(nmp) => nmp,
            None => return (1e4_f64).ln_1p(),
        };

        // Physical penalties
        let mut penalty = 0.0;
        if nmp.rho0_fm3 < 0.08 {
            penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08;
        } else if nmp.rho0_fm3 > 0.25 {
            penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25;
        }
        if nmp.e_a_mev > -5.0 {
            penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0);
        }

        // χ²/datum
        let mut chi2 = 0.0;
        let mut n_valid = 0;
        for (&(z, nn), &(b_exp, _sigma)) in exp_data_ref.iter() {
            let b_calc = semf_binding_energy(z, nn, x);
            if b_calc > 0.0 {
                let sigma_theo = (0.01 * b_exp).max(2.0);
                chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
                n_valid += 1;
            }
        }

        if n_valid == 0 {
            return (1e4_f64).ln_1p();
        }

        let chi2_per_datum = chi2 / n_valid as f64 + penalty;
        chi2_per_datum.ln_1p() // log-transform for smooth surrogate learning
    };

    // ── Configure SparsitySampler ──────────────────────────────────
    // Parse CLI args for configurability
    let args: Vec<String> = std::env::args().collect();
    let n_iters = args.iter()
        .find(|a| a.starts_with("--rounds="))
        .and_then(|a| a.strip_prefix("--rounds=")?.parse().ok())
        .unwrap_or(10);
    let n_solvers = args.iter()
        .find(|a| a.starts_with("--solvers="))
        .and_then(|a| a.strip_prefix("--solvers=")?.parse().ok())
        .unwrap_or(16);
    let eval_budget = args.iter()
        .find(|a| a.starts_with("--evals="))
        .and_then(|a| a.strip_prefix("--evals=")?.parse().ok())
        .unwrap_or(100);
    let smoothing: f64 = args.iter()
        .find(|a| a.starts_with("--smoothing="))
        .and_then(|a| a.strip_prefix("--smoothing=")?.parse().ok())
        .unwrap_or(1e-12);

    let mut config = SparsitySamplerConfig::new(bounds.len(), 42)
        .with_initial_samples(100)      // LHS initial exploration
        .with_solvers(n_solvers)        // parallel NM starts per iteration
        .with_eval_budget(eval_budget)  // max evals per solver per iteration
        .with_iterations(n_iters)       // surrogate refinement rounds
        .with_kernel(RBFKernel::ThinPlateSpline);
    config.smoothing = smoothing;

    println!("  SparsitySampler config:");
    println!("    Initial LHS:     {}", config.n_initial);
    println!("    Solvers/iter:    {}", config.n_solvers);
    println!("    Evals/solver:    {}", config.max_eval_per_solver);
    println!("    Iterations:      {}", config.n_iterations);
    println!("    Smoothing:       {:.2e}", config.smoothing);
    println!("    Total budget:    ~{}", config.total_budget());
    println!();

    // ── Run ────────────────────────────────────────────────────────
    println!("  Running SparsitySampler...");
    let t0 = Instant::now();

    let result = sparsity_sampler(&objective, &bounds, &config)
        .expect("SparsitySampler failed");

    let elapsed = t0.elapsed();

    // ── Results ────────────────────────────────────────────────────
    let log_chi2 = result.f_best;
    let chi2 = log_chi2.exp() - 1.0; // undo log1p transform

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Results                                                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  χ²/datum:       {:8.4}                                  ║", chi2);
    println!("║  log(1+χ²):      {:8.4}                                  ║", log_chi2);
    println!("║  Total evals:    {:6}                                    ║", result.cache.len());
    println!("║  Time:           {:6.1}s                                   ║", elapsed.as_secs_f64());
    println!("║  Throughput:     {:6.0} evals/s                            ║",
        result.cache.len() as f64 / elapsed.as_secs_f64());
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Per-iteration diagnostics
    println!("  Per-iteration breakdown:");
    for ir in &result.iteration_results {
        let rmse_str = ir.surrogate_error
            .map(|e| format!("{:.4}", e))
            .unwrap_or_else(|| "N/A".to_string());
        println!(
            "    Round {:2}: best_f={:.4}, +{} evals (total {}), surrogate RMSE={}",
            ir.iteration + 1,
            ir.best_f,
            ir.n_new_evals,
            ir.total_evals,
            rmse_str,
        );
    }

    // Best parameters
    println!();
    println!("  Best Skyrme parameters:");
    for (i, name) in data::PARAM_NAMES.iter().enumerate() {
        println!("    {:6} = {:12.4}", name, result.x_best[i]);
    }

    // Nuclear matter properties at best
    if let Some(nmp) = nuclear_matter_properties(&result.x_best) {
        println!();
        println!("  Nuclear matter at best:");
        println!("    ρ₀   = {:.4} fm⁻³  (exp: 0.16)", nmp.rho0_fm3);
        println!("    E/A  = {:.2} MeV    (exp: -15.97)", nmp.e_a_mev);
        println!("    K∞   = {:.1} MeV    (exp: 230)", nmp.k_inf_mev);
        println!("    m*/m = {:.3}        (exp: 0.69)", nmp.m_eff_ratio);
        println!("    J    = {:.1} MeV    (exp: 32)", nmp.j_mev);
    }

    // Save results
    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();
    let result_json = serde_json::json!({
        "level": 1,
        "engine": "barracuda::sparsity_sampler",
        "chi2_per_datum": chi2,
        "log_chi2": log_chi2,
        "total_evals": result.cache.len(),
        "time_seconds": elapsed.as_secs_f64(),
        "throughput_evals_per_sec": result.cache.len() as f64 / elapsed.as_secs_f64(),
        "best_params": result.x_best,
        "config": {
            "n_initial": config.n_initial,
            "n_solvers": config.n_solvers,
            "max_eval_per_solver": config.max_eval_per_solver,
            "n_iterations": config.n_iterations,
            "smoothing": config.smoothing,
        },
    });
    let path = results_dir.join("barracuda_sparsity_l1.json");
    std::fs::write(&path, serde_json::to_string_pretty(&result_json).unwrap()).ok();
    println!();
    println!("  Results saved to: {}", path.display());
}

