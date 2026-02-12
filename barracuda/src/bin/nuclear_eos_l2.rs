//! Nuclear EOS Level 2 — Surrogate Learning via BarraCUDA (hybrid HFB)
//!
//! Full SparsitySampler workflow using barracuda library modules:
//!   - `barracuda::sample::sparsity::sparsity_sampler` — iterative surrogate learning
//!   - `barracuda::special::{gamma, laguerre}` — HO basis wavefunctions
//!   - `barracuda::numerical::{trapz, gradient_1d}` — numerical integration
//!   - `barracuda::optimize::bisect` — root-finding
//!   - `nalgebra::SymmetricEigen` — eigenvalue decomposition (barracuda gap)
//!
//! Validates against Python control: `control/surrogate/nuclear-eos/scripts/run_surrogate.py --level=2`

use hotspring_barracuda::data;
use hotspring_barracuda::physics::{nuclear_matter_properties, hfb::binding_energy_l2};

use barracuda::sample::sparsity::{sparsity_sampler, SparsitySamplerConfig};
use barracuda::surrogate::RBFKernel;

use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Nuclear EOS L2 — BarraCUDA SparsitySampler Validation     ║");
    println!("║  Using: barracuda::sample::sparsity_sampler (library)      ║");
    println!("║  Physics: Hybrid HFB (barracuda::special + nalgebra)       ║");
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

    // ── Define L2 objective function (with parallel HFB) ──────────
    let exp_data_obj = exp_data.clone();
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

        // Build nuclei list
        let nuclei: Vec<(usize, usize, f64)> = exp_data_obj
            .iter()
            .map(|(&(z, n), &(b_exp, _))| (z, n, b_exp))
            .collect();

        // Parallel HFB evaluation via rayon
        let x_owned: Vec<f64> = x.to_vec();
        let results: Vec<(f64, f64)> = nuclei
            .par_iter()
            .map(|&(z, n, b_exp)| {
                let (b_calc, _conv) = binding_energy_l2(z, n, &x_owned);
                (b_calc, b_exp)
            })
            .collect();

        // χ²/datum
        let mut chi2 = 0.0;
        let mut n_valid = 0;
        for (b_calc, b_exp) in results {
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
        chi2_per_datum.ln_1p()
    };

    // ── Configure SparsitySampler ──────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let n_iters = args.iter()
        .find(|a| a.starts_with("--rounds="))
        .and_then(|a| a.strip_prefix("--rounds=")?.parse().ok())
        .unwrap_or(5);
    let n_solvers = args.iter()
        .find(|a| a.starts_with("--solvers="))
        .and_then(|a| a.strip_prefix("--solvers=")?.parse().ok())
        .unwrap_or(8);
    let eval_budget = args.iter()
        .find(|a| a.starts_with("--evals="))
        .and_then(|a| a.strip_prefix("--evals=")?.parse().ok())
        .unwrap_or(50);

    let config = SparsitySamplerConfig::new(bounds.len(), 42)
        .with_initial_samples(100)
        .with_solvers(n_solvers)
        .with_eval_budget(eval_budget)
        .with_iterations(n_iters)
        .with_kernel(RBFKernel::ThinPlateSpline);

    println!("  SparsitySampler config:");
    println!("    Initial LHS:     {}", config.n_initial);
    println!("    Solvers/iter:    {}", config.n_solvers);
    println!("    Evals/solver:    {}", config.max_eval_per_solver);
    println!("    Iterations:      {}", config.n_iterations);
    println!("    Total budget:    ~{}", config.total_budget());
    println!("    Rayon threads:   {}", rayon::current_num_threads());
    println!();

    // ── Run ────────────────────────────────────────────────────────
    println!("  Running SparsitySampler (L2 — this takes longer due to HFB)...");
    let t0 = Instant::now();

    let result = sparsity_sampler(&objective, &bounds, &config)
        .expect("SparsitySampler failed");

    let elapsed = t0.elapsed();

    // ── Results ────────────────────────────────────────────────────
    let log_chi2 = result.f_best;
    let chi2 = log_chi2.exp() - 1.0;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  L2 Results                                                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  χ²/datum:       {:8.4}                                  ║", chi2);
    println!("║  log(1+χ²):      {:8.4}                                  ║", log_chi2);
    println!("║  Total evals:    {:6}                                    ║", result.cache.len());
    println!("║  Time:           {:6.1}s                                   ║", elapsed.as_secs_f64());
    println!("║  Throughput:     {:6.1} evals/s                            ║",
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
        "level": 2,
        "engine": "barracuda::sparsity_sampler + nalgebra::SymmetricEigen",
        "chi2_per_datum": chi2,
        "log_chi2": log_chi2,
        "total_evals": result.cache.len(),
        "time_seconds": elapsed.as_secs_f64(),
        "throughput_evals_per_sec": result.cache.len() as f64 / elapsed.as_secs_f64(),
        "best_params": result.x_best,
        "barracuda_modules_used": [
            "sample::sparsity::sparsity_sampler",
            "surrogate::RBFSurrogate",
            "surrogate::RBFKernel::ThinPlateSpline",
            "optimize::nelder_mead",
            "optimize::multi_start_nelder_mead",
            "optimize::bisect",
            "sample::latin_hypercube",
            "special::gamma",
            "special::laguerre",
            "numerical::trapz",
            "numerical::gradient_1d",
            "linalg::solve_f64",
        ],
        "external_deps": [
            "nalgebra::SymmetricEigen (gap: barracuda::linalg needs symmetric_eigen)",
        ],
    });
    let path = results_dir.join("barracuda_sparsity_l2.json");
    std::fs::write(&path, serde_json::to_string_pretty(&result_json).unwrap()).ok();
    println!();
    println!("  Results saved to: {}", path.display());
}

