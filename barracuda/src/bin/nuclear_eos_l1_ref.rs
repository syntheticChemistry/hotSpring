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
use hotspring_barracuda::physics::{nuclear_matter_properties, semf_binding_energy};
use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;

// ALL from barracuda native — no hotspring_barracuda::surrogate or ::stats
use barracuda::sample::direct::{direct_sampler, DirectSamplerConfig};
use barracuda::sample::sparsity::{sparsity_sampler, PenaltyFilter, SparsitySamplerConfig};
use barracuda::stats::{bootstrap_ci, chi2_decomposed_weighted};
// loo_cv_optimal_smoothing and RBFKernel now used internally by SparsitySampler
use barracuda::optimize::convergence_diagnostics;

use std::sync::Arc;
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

const PARAM_NAMES: [&str; 10] = [
    "t0", "t1", "t2", "t3", "x0", "x1", "x2", "x3", "alpha", "W0",
];

struct NucleusResidual {
    z: usize,
    n: usize,
    a: usize,
    element: String,
    b_exp: f64,
    b_calc: f64,
    delta_b: f64,
    abs_delta: f64,
    rel_delta: f64,
    chi2_i: f64,
    _sigma: f64,
}

struct CliArgs {
    seed: u64,
    n_seeds: usize,
    lambda: f64,
    pareto: bool,
}

// EosContext and load_eos_context are now in data module
use hotspring_barracuda::data::EosContext;

fn make_l1_objective_nmp(
    exp_data: &Arc<std::collections::HashMap<(usize, usize), (f64, f64)>>,
    lambda: f64,
) -> impl Fn(&[f64]) -> f64 {
    let exp_data = exp_data.clone();
    move |x: &[f64]| l1_objective_nmp(x, &exp_data, lambda)
}

fn compute_mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

fn print_comparison_summary(
    chi2_1: f64,
    approach1_evals: usize,
    approach1_time: f64,
    chi2_2: f64,
    approach2_evals: usize,
    approach2_time: f64,
    better: f64,
) {
    println!(
        "  {:40} {:>10} {:>8} {:>8}",
        "Method", "χ²/datum", "Evals", "Time"
    );
    println!(
        "  {:40} {:>10} {:>8} {:>8}",
        "─".repeat(40),
        "─".repeat(10),
        "─".repeat(8),
        "─".repeat(8)
    );
    println!(
        "  {:40} {:>10.4} {:>8} {:>7.2}s",
        "SparsitySampler + NativeAutoSmooth", chi2_1, approach1_evals, approach1_time
    );
    println!(
        "  {:40} {:>10.4} {:>8} {:>7.2}s",
        "Native DirectSampler", chi2_2, approach2_evals, approach2_time
    );
    println!(
        "  {:40} {:>10.4} {:>8} {:>8}",
        "Python/scipy control (reference)",
        provenance::L1_PYTHON_CHI2.value,
        provenance::L1_PYTHON_CANDIDATES.value,
        "~180s"
    );
    println!(
        "  {:40} {:>10.4} {:>8} {:>8}",
        "Previous BarraCuda best (manual smooth)", 1.19, 164, "0.25s"
    );
    println!();

    if better < provenance::L1_PYTHON_CHI2.value {
        let fewer = provenance::L1_PYTHON_CANDIDATES.value
            / (approach1_evals.min(approach2_evals).max(1) as f64);
        println!(
            "  ✅ BarraCuda BEATS Python by {:.1}%",
            100.0 * (provenance::L1_PYTHON_CHI2.value - better) / provenance::L1_PYTHON_CHI2.value
        );
        if fewer > 1.0 {
            println!("     with {fewer:.0}× fewer evaluations");
        }
    } else {
        println!(
            "  ⚠ BarraCuda behind Python by {:.1}% — needs more tuning",
            100.0 * (better - provenance::L1_PYTHON_CHI2.value) / provenance::L1_PYTHON_CHI2.value
        );
    }
}

fn print_reference_baselines(
    exp_data: &std::collections::HashMap<(usize, usize), (f64, f64)>,
    lambda: f64,
) {
    for (name, params) in &[
        ("SLy4", provenance::SLY4_PARAMS.as_slice()),
        ("UNEDF0", UNEDF0_PARAMS.as_slice()),
    ] {
        let be_chi2 = compute_be_chi2_only(params, exp_data);
        if let Some(nmp) = nuclear_matter_properties(params) {
            let nmp_c2 = provenance::nmp_chi2_from_props(&nmp) / 5.0;
            println!("  {name} (published):");
            println!("    chi2_BE/datum:  {be_chi2:.4}");
            println!("    chi2_NMP/datum: {nmp_c2:.4}");
            println!(
                "    chi2_total (lambda={}): {:.4}",
                lambda,
                lambda.mul_add(nmp_c2, be_chi2)
            );
            provenance::print_nmp_analysis(&nmp);
            println!();
        } else {
            println!("  {name} — NMP computation failed (params outside bisection range)");
            println!();
        }
    }
}

fn print_best_parameters(best_x: &[f64]) {
    println!("  Best Skyrme parameters:");
    for (i, name) in PARAM_NAMES.iter().enumerate() {
        if i < best_x.len() {
            println!("    {:>6} = {:>14.6}", name, best_x[i]);
        }
    }
}

fn print_semf_capability_analysis(rms: f64, best_chi2: f64, mean_rel: f64) {
    println!("  Our SEMF results:");
    println!("    RMS     = {rms:.4} MeV");
    println!("    χ²/datum = {best_chi2:.4} (with σ_theo = max(1%·B, 2 MeV))");
    println!();

    if rms < 3.0 {
        println!("  VERDICT: SEMF optimized to NEAR-THEORETICAL-LIMIT.");
        println!("    - Our RMS {rms:.2} MeV is competitive with published SEMF fits.");
        println!("    - χ²/datum < 1.0 means we fit WITHIN assumed uncertainties.");
    } else if rms < 5.0 {
        println!("  VERDICT: Good SEMF fit, room for minor coefficient improvement.");
    } else {
        println!("  VERDICT: SEMF fit sub-optimal, optimizer may need more budget.");
    }
    println!();
    println!("  To reach paper-level ~10^-6 accuracy:");
    println!(
        "    - Current gap: {:.1} orders of magnitude",
        (mean_rel / 1.0e-6).log10()
    );
    println!("    - Requires: L2 (HFB) physics solver, not SEMF");
    println!(
        "    - SEMF theoretical floor: ~{:.1} MeV RMS (~{:.1e} relative for A=200)",
        rms.max(2.0),
        rms.max(2.0) / 1700.0
    );
    println!("    - HFB theoretical floor: ~0.5 MeV RMS (~3e-4 relative)");
    println!("    - Paper target of 10^-6: requires beyond-HFB corrections");
    println!("      (e.g., Wigner, rotational, shape coexistence)");
}

fn run_deep_residual_analysis(
    base: &std::path::Path,
    best_x: &[f64],
    best_chi2: f64,
    chi2_1: f64,
    approach1_evals: usize,
    approach1_time: f64,
    approach1_f: f64,
    chi2_2: f64,
    approach2_evals: usize,
    result2: &barracuda::sample::direct::DirectSamplerResult,
    base_seed: u64,
) {
    let nuclei_set = data::parse_nuclei_set_from_args();
    let nuclei_path = data::nuclei_data_path(base, nuclei_set);
    let nuclei_reader = std::io::BufReader::new(
        std::fs::File::open(&nuclei_path).expect("Failed to open nuclei JSON"),
    );
    let nuclei_file: serde_json::Value =
        serde_json::from_reader(nuclei_reader).expect("nuclei JSON parse failed");
    let nuclei_list = nuclei_file["nuclei"]
        .as_array()
        .expect("nuclei JSON has nuclei array");

    let mut residuals: Vec<NucleusResidual> = Vec::new();
    for nuc in nuclei_list {
        let z = nuc["Z"].as_u64().expect("nucleus Z") as usize;
        let n = nuc["N"].as_u64().expect("nucleus N") as usize;
        let a = nuc["A"].as_u64().expect("nucleus A") as usize;
        let element = nuc["element"]
            .as_str()
            .expect("nucleus element")
            .to_string();
        let b_exp = nuc["binding_energy_MeV"]
            .as_f64()
            .expect("nucleus binding_energy_MeV");

        let b_calc = semf_binding_energy(z, n, best_x);
        if b_calc > 0.0 {
            let delta_b = b_calc - b_exp;
            let sigma = tolerances::sigma_theo(b_exp);
            residuals.push(NucleusResidual {
                z,
                n,
                a,
                element,
                b_exp,
                b_calc,
                delta_b,
                abs_delta: delta_b.abs(),
                rel_delta: (delta_b / b_exp).abs(),
                chi2_i: (delta_b / sigma).powi(2),
                _sigma: sigma,
            });
        }
    }

    let n_nuclei = residuals.len();
    println!("  Nuclei fitted: {n_nuclei}");
    println!();

    let rms = (residuals.iter().map(|r| r.delta_b.powi(2)).sum::<f64>() / n_nuclei as f64).sqrt();
    let mae = residuals.iter().map(|r| r.abs_delta).sum::<f64>() / n_nuclei as f64;
    let max_err = residuals
        .iter()
        .map(|r| r.abs_delta)
        .fold(0.0_f64, f64::max);
    let mean_rel = residuals.iter().map(|r| r.rel_delta).sum::<f64>() / n_nuclei as f64;
    let max_rel = residuals
        .iter()
        .map(|r| r.rel_delta)
        .fold(0.0_f64, f64::max);
    let median_rel = {
        let mut rels: Vec<f64> = residuals.iter().map(|r| r.rel_delta).collect();
        rels.sort_by(f64::total_cmp);
        if rels.len().is_multiple_of(2) {
            f64::midpoint(rels[rels.len() / 2 - 1], rels[rels.len() / 2])
        } else {
            rels[rels.len() / 2]
        }
    };
    let mean_signed = residuals.iter().map(|r| r.delta_b).sum::<f64>() / n_nuclei as f64;

    println!("  ┌─────────────────────────────────────────────────┐");
    println!("  │  GLOBAL ACCURACY METRICS                        │");
    println!("  ├─────────────────────────────────────────────────┤");
    println!("  │  RMS deviation:        {rms:>10.4} MeV             │");
    println!("  │  Mean absolute error:  {mae:>10.4} MeV             │");
    println!("  │  Max absolute error:   {max_err:>10.4} MeV             │");
    println!("  │  Mean signed error:    {mean_signed:>10.4} MeV (bias)      │");
    println!("  │                                                 │");
    println!("  │  Mean |ΔB/B|:          {mean_rel:>12.6e}             │");
    println!("  │  Median |ΔB/B|:        {median_rel:>12.6e}             │");
    println!("  │  Max |ΔB/B|:           {max_rel:>12.6e}             │");
    println!("  │                                                 │");
    println!("  │  Paper target:         ~1.0e-06 (relative)      │");
    println!("  │  Our mean relative:    {mean_rel:>12.6e}             │");
    println!(
        "  │  Gap to paper:         {:.1}× (orders of mag)   │",
        (mean_rel / 1.0e-6).log10()
    );
    println!("  └─────────────────────────────────────────────────┘");

    let thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1];
    println!();
    println!("  Relative accuracy |ΔB/B| distribution:");
    for &thresh in &thresholds {
        let count = residuals.iter().filter(|r| r.rel_delta < thresh).count();
        let pct = 100.0 * count as f64 / n_nuclei as f64;
        let bar = "█".repeat((pct / 2.0) as usize);
        println!("    < {thresh:.0e}: {count:>4}/{n_nuclei} ({pct:>5.1}%) {bar}");
    }
    let abs_thresholds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
    println!();
    println!("  Absolute accuracy |ΔB| distribution:");
    for &thresh in &abs_thresholds {
        let count = residuals.iter().filter(|r| r.abs_delta < thresh).count();
        let pct = 100.0 * count as f64 / n_nuclei as f64;
        let bar = "█".repeat((pct / 2.0) as usize);
        println!("    < {thresh:>5.1} MeV: {count:>4}/{n_nuclei} ({pct:>5.1}%) {bar}");
    }

    let regions: Vec<(&str, Box<dyn Fn(&NucleusResidual) -> bool>)> = vec![
        ("Light A<50", Box::new(|r: &NucleusResidual| r.a < 50)),
        (
            "Medium 50-100",
            Box::new(|r: &NucleusResidual| r.a >= 50 && r.a < 100),
        ),
        (
            "Heavy 100-200",
            Box::new(|r: &NucleusResidual| r.a >= 100 && r.a < 200),
        ),
        (
            "Very heavy 200+",
            Box::new(|r: &NucleusResidual| r.a >= 200),
        ),
    ];
    println!();
    println!("  Accuracy by mass region:");
    println!(
        "  {:>12} {:>6} {:>10} {:>10} {:>12} {:>10}",
        "Region", "Count", "RMS(MeV)", "MAE(MeV)", "Mean|ΔB/B|", "χ²/datum"
    );
    for (label, pred) in &regions {
        let group: Vec<&NucleusResidual> = residuals.iter().filter(|r| pred(r)).collect();
        if group.is_empty() {
            continue;
        }
        let ng = group.len() as f64;
        let g_rms = (group.iter().map(|r| r.delta_b.powi(2)).sum::<f64>() / ng).sqrt();
        let g_mae = group.iter().map(|r| r.abs_delta).sum::<f64>() / ng;
        let g_rel = group.iter().map(|r| r.rel_delta).sum::<f64>() / ng;
        let g_chi2 = group.iter().map(|r| r.chi2_i).sum::<f64>() / ng;
        println!(
            "  {:>12} {:>6} {:>10.3} {:>10.3} {:>12.6e} {:>10.4}",
            label,
            group.len(),
            g_rms,
            g_mae,
            g_rel,
            g_chi2
        );
    }

    let mut by_rel: Vec<&NucleusResidual> = residuals.iter().collect();
    by_rel.sort_by(|a, b| a.rel_delta.total_cmp(&b.rel_delta));
    println!();
    println!("  Top 10 BEST-fitted nuclei (lowest |ΔB/B|):");
    println!(
        "  {:>6} {:>4} {:>4} {:>10} {:>10} {:>10} {:>12}",
        "Nuclide", "Z", "N", "B_exp", "B_calc", "ΔB(MeV)", "|ΔB/B|"
    );
    for r in by_rel.iter().take(10) {
        println!(
            "  {:>3}-{:<3} {:>4} {:>4} {:>10.3} {:>10.3} {:>+10.3} {:>12.6e}",
            r.element, r.a, r.z, r.n, r.b_exp, r.b_calc, r.delta_b, r.rel_delta
        );
    }

    by_rel.reverse();
    println!();
    println!("  Top 10 WORST-fitted nuclei (highest |ΔB/B|):");
    println!(
        "  {:>6} {:>4} {:>4} {:>10} {:>10} {:>10} {:>12}",
        "Nuclide", "Z", "N", "B_exp", "B_calc", "ΔB(MeV)", "|ΔB/B|"
    );
    for r in by_rel.iter().take(10) {
        println!(
            "  {:>3}-{:<3} {:>4} {:>4} {:>10.3} {:>10.3} {:>+10.3} {:>12.6e}",
            r.element, r.a, r.z, r.n, r.b_exp, r.b_calc, r.delta_b, r.rel_delta
        );
    }

    let mut by_abs: Vec<&NucleusResidual> = residuals.iter().collect();
    by_abs.sort_by(|a, b| b.abs_delta.total_cmp(&a.abs_delta));
    println!();
    println!("  Top 10 largest |ΔB| (MeV):");
    println!(
        "  {:>6} {:>4} {:>4} {:>10} {:>10} {:>10} {:>10}",
        "Nuclide", "Z", "N", "B_exp", "B_calc", "ΔB(MeV)", "χ²_i"
    );
    for r in by_abs.iter().take(10) {
        println!(
            "  {:>3}-{:<3} {:>4} {:>4} {:>10.3} {:>10.3} {:>+10.3} {:>10.4}",
            r.element, r.a, r.z, r.n, r.b_exp, r.b_calc, r.delta_b, r.chi2_i
        );
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SEMF MODEL CAPABILITY vs PAPER TARGET");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  This analysis distinguishes OPTIMIZATION accuracy from MODEL accuracy.");
    println!("  The SEMF (Bethe-Weizsäcker) is a 5-term mass formula.");
    println!("  Published SEMF fits achieve RMS ~2-3 MeV.");
    println!("  Paper-level (~10^-6 relative) requires HFB-level physics:");
    println!("    - HFB mass tables (Goriely et al.): RMS ~0.5-0.7 MeV");
    println!("    - DFT/Fayans: RMS ~0.3 MeV for select nuclei");
    println!("    - For B~1000 MeV, 10^-6 = 0.001 MeV = 1 keV");
    println!();
    print_semf_capability_analysis(rms, best_chi2, mean_rel);

    print_best_parameters(best_x);

    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();

    let per_nucleus_json: Vec<serde_json::Value> = residuals
        .iter()
        .map(|r| {
            serde_json::json!({
                "element": r.element,
                "Z": r.z, "N": r.n, "A": r.a,
                "B_exp_MeV": r.b_exp,
                "B_calc_MeV": r.b_calc,
                "delta_B_MeV": r.delta_b,
                "abs_delta_MeV": r.abs_delta,
                "relative_accuracy": r.rel_delta,
                "chi2_contribution": r.chi2_i,
            })
        })
        .collect();

    let result_json = serde_json::json!({
        "level": 1,
        "engine": "barracuda::native_revalidation_v2",
        "barracuda_version": "phase5_evolved",
        "run_date": "2026-02-13",
        "seed": base_seed,
        "approach1_sparsity_native_autosmooth": {
            "chi2_per_datum": chi2_1,
            "log_chi2": approach1_f,
            "total_evals": approach1_evals,
            "time_seconds": approach1_time,
            "auto_smoothing": true,
            "penalty_filter": "AdaptiveMAD(5.0)",
        },
        "approach2_native_direct_sampler": {
            "chi2_per_datum": chi2_2,
            "log_chi2": result2.f_best,
            "total_evals": approach2_evals,
            "early_stopped": result2.early_stopped,
            "n_rounds": result2.rounds.len(),
        },
        "accuracy_metrics": {
            "rms_MeV": rms,
            "mae_MeV": mae,
            "max_error_MeV": max_err,
            "mean_signed_error_MeV": mean_signed,
            "mean_relative_accuracy": mean_rel,
            "median_relative_accuracy": median_rel,
            "max_relative_accuracy": max_rel,
            "n_nuclei": n_nuclei,
            "chi2_per_datum": best_chi2,
        },
        "paper_comparison": {
            "paper_target_relative": 1.0e-6,
            "our_mean_relative": mean_rel,
            "gap_orders_of_magnitude": (mean_rel / 1.0e-6).log10(),
            "semf_theoretical_limit_MeV": 2.0,
            "hfb_theoretical_limit_MeV": 0.5,
            "notes": "SEMF is a 5-term model; 10^-6 requires HFB+ physics"
        },
        "best_parameters": {
            "names": PARAM_NAMES.to_vec(),
            "values": best_x.to_vec(),
        },
        "per_nucleus": per_nucleus_json,
        "references": {
            "python_scipy": {
                "chi2_per_datum": provenance::L1_PYTHON_CHI2.value,
                "evals": provenance::L1_PYTHON_CANDIDATES.value,
            },
            "previous_barracuda_best": { "chi2_per_datum": 0.7971, "evals": 64 },
        },
    });
    let path = results_dir.join("barracuda_l1_deep_analysis.json");
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&result_json).expect("JSON serialize"),
    )
    .ok();
    println!("\n  Full results saved to: {}", path.display());
}

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

    let result1 = sparsity_sampler(&objective, &bounds, &config).expect("SparsitySampler failed");

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
        direct_sampler(objective2, &bounds, &direct_config).expect("DirectSampler failed");

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
    print_reference_baselines(&exp_data, lambda);

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
        let r_d = direct_sampler(obj_d, &bounds, &dc).expect("DirectSampler failed");
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

    let lambdas = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0];
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
            let r = direct_sampler(obj, &bounds, &dc).expect("DirectSampler failed");

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

// ═══════════════════════════════════════════════════════════════════
// L1 objective with NMP chi-squared constraint (UNEDF-style)
//
// chi2_total = chi2_BE/datum + lambda * chi2_NMP/datum
//
// lambda=0: binding energy only (original, overfits NMP)
// lambda>0: includes NMP chi-squared (published targets)
// ═══════════════════════════════════════════════════════════════════

fn l1_objective_nmp(
    x: &[f64],
    exp_data: &std::collections::HashMap<(usize, usize), (f64, f64)>,
    lambda: f64,
) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    // NMP check
    let Some(nmp) = nuclear_matter_properties(x) else {
        return (1e4_f64).ln_1p();
    };

    // Hard rejection for grossly unphysical (keeps optimizer from wasting time)
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 {
        return (1e4_f64).ln_1p();
    }
    if nmp.e_a_mev > 0.0 {
        return (1e4_f64).ln_1p();
    }

    // Binding energy chi-squared
    let mut chi2_be = 0.0;
    let mut n = 0;
    for (&(z, nn), &(b_exp, _sigma)) in exp_data {
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

    // NMP chi-squared (5 published targets with uncertainties)
    let chi2_nmp_datum = provenance::nmp_chi2_from_props(&nmp) / 5.0;

    // Combined: chi2_BE/datum + lambda * chi2_NMP/datum
    let chi2_total = lambda.mul_add(chi2_nmp_datum, chi2_be_datum);

    chi2_total.ln_1p()
}

/// Compute binding energy chi2/datum only (for reference baselines)
fn compute_be_chi2_only(
    x: &[f64],
    exp_data: &std::collections::HashMap<(usize, usize), (f64, f64)>,
) -> f64 {
    let mut chi2 = 0.0;
    let mut n = 0;
    for (&(z, nn), &(b_exp, _sigma)) in exp_data {
        let b_calc = semf_binding_energy(z, nn, x);
        if b_calc > 0.0 {
            let sigma_theo = tolerances::sigma_theo(b_exp);
            chi2 += ((b_calc - b_exp) / sigma_theo).powi(2);
            n += 1;
        }
    }
    if n == 0 {
        return 1e4;
    }
    chi2 / f64::from(n)
}

/// Compute per-nucleus binding energies at given parameters.
fn compute_binding_energies(
    params: &[f64],
    exp_data: &std::collections::HashMap<(usize, usize), (f64, f64)>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut observed = Vec::new();
    let mut expected = Vec::new();
    let mut sigma = Vec::new();

    for (&(z, n), &(b_exp, _)) in exp_data {
        let b_calc = semf_binding_energy(z, n, params);
        if b_calc > 0.0 {
            observed.push(b_calc);
            expected.push(b_exp);
            sigma.push(tolerances::sigma_theo(b_exp));
        }
    }

    (observed, expected, sigma)
}
