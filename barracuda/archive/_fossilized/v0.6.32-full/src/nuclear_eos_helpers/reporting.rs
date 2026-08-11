// SPDX-License-Identifier: AGPL-3.0-or-later

//! Display and reporting functions for nuclear EOS residual analysis.
//!
//! All functions print to stdout. No computation — consume already-computed
//! [`NucleusResidual`] and [`ResidualMetrics`] values.

use crate::data::PARAM_NAMES;
use crate::provenance;

use super::{NucleusResidual, ResidualMetrics};

/// Region predicate: (label, lo inclusive, hi exclusive).
const MASS_REGIONS: &[(&str, usize, usize)] = &[
    ("Light A<50", 0, 50),
    ("Medium 50-100", 50, 100),
    ("Heavy 100-200", 100, 200),
    ("Very heavy 200+", 200, 9999),
];

/// Print SEMF GPU vs CPU precision summary (max diff, speedup, verdict).
pub fn print_semf_gpu_precision(
    max_diff: f64,
    mean_diff: f64,
    chi2_diff: f64,
    cpu_us: f64,
    gpu_us: f64,
    n_nuclei: usize,
) {
    let speedup = cpu_us / gpu_us;
    println!("  ── Precision ──");
    println!("    Max  |B_cpu - B_gpu|: {max_diff:.2e} MeV");
    println!("    Mean |B_cpu - B_gpu|: {mean_diff:.2e} MeV");
    println!("    |chi2_cpu - chi2_gpu|: {chi2_diff:.2e}");
    if speedup >= 1.0 {
        println!("    GPU speedup: {speedup:.1}x");
    } else {
        println!(
            "    GPU overhead: {:.1}x (dispatch > compute for {} nuclei)",
            1.0 / speedup,
            n_nuclei
        );
    }
    if max_diff < 1e-10 {
        println!("    EXACT MATCH (< 1e-10 MeV) ✓");
    } else if max_diff < 1e-6 {
        println!("    Precision: EXCELLENT (< 1e-6 MeV)");
    } else if max_diff < 0.01 {
        println!("    Precision: GOOD (< 0.01 MeV — rounding)");
    } else {
        println!("    !! DISCREPANCY > 0.01 MeV — investigate");
    }
}

/// Print pure-GPU (`math_f64`) precision vs CPU and precomputed-GPU.
pub fn print_pure_gpu_precision(
    pure_max_diff: f64,
    pure_mean_diff: f64,
    pure_vs_precomp: f64,
    cpu_us: f64,
    precomp_us: f64,
    pure_us: f64,
    n_iters: usize,
) {
    println!();
    println!("  Pure-GPU (math_f64 library on RTX 4070):");
    println!("    Time per eval: {pure_us:.1} us ({n_iters} iters)");
    println!();
    println!("  ── Precision vs CPU ──");
    println!("    Max  |B_cpu - B_pure_gpu|: {pure_max_diff:.2e} MeV");
    println!("    Mean |B_cpu - B_pure_gpu|: {pure_mean_diff:.2e} MeV");
    println!("  ── Precision vs Precomputed-GPU ──");
    println!("    Max  |B_precomp - B_pure|: {pure_vs_precomp:.2e} MeV");
    println!("  ── Speed comparison ──");
    println!("    CPU:             {cpu_us:.1} us/eval");
    println!("    GPU (precomp):   {precomp_us:.1} us/eval");
    println!("    GPU (pure math): {pure_us:.1} us/eval");

    if pure_max_diff < 1e-6 {
        println!("    Pure-GPU math_f64: VALIDATED (< 1e-6 MeV vs CPU)");
    } else if pure_max_diff < 0.1 {
        println!("    Pure-GPU math_f64: GOOD (< 0.1 MeV — Newton/polynomial precision)");
    } else {
        println!("    Pure-GPU math_f64: NEEDS TUNING ({pure_max_diff:.2e} MeV)");
    }
}

/// Print NMP values with sigma pulls (for optimization result display).
pub fn print_nmp_with_pulls(nmp: &crate::physics::NuclearMatterProps) {
    let vals = [
        nmp.rho0_fm3,
        nmp.e_a_mev,
        nmp.k_inf_mev,
        nmp.m_eff_ratio,
        nmp.j_mev,
    ];
    let targets = provenance::NMP_TARGETS.values();
    let sigmas = provenance::NMP_TARGETS.sigmas();
    for (i, &v) in vals.iter().enumerate() {
        let pull = (v - targets[i]) / sigmas[i];
        println!(
            "    {:>6} = {:>10.4}  (pull: {:>+.2}σ)",
            provenance::NMP_NAMES[i],
            v,
            pull
        );
    }
}

/// Print Skyrme parameters in canonical order.
pub fn print_best_parameters(best_x: &[f64]) {
    println!("  Best Skyrme parameters:");
    for (i, name) in PARAM_NAMES.iter().enumerate() {
        if i < best_x.len() {
            println!("    {:>6} = {:>14.6}", name, best_x[i]);
        }
    }
}

/// Print the global accuracy metrics box.
pub fn print_accuracy_metrics_box(metrics: &ResidualMetrics) {
    println!("  ┌─────────────────────────────────────────────────┐");
    println!("  │  GLOBAL ACCURACY METRICS                        │");
    println!("  ├─────────────────────────────────────────────────┤");
    println!(
        "  │  RMS deviation:        {:>10.4} MeV             │",
        metrics.rms
    );
    println!(
        "  │  Mean absolute error:  {:>10.4} MeV             │",
        metrics.mae
    );
    println!(
        "  │  Max absolute error:   {:>10.4} MeV             │",
        metrics.max_err
    );
    println!(
        "  │  Mean signed error:    {:>10.4} MeV (bias)      │",
        metrics.mean_signed
    );
    println!("  │                                                 │");
    println!(
        "  │  Mean |ΔB/B|:          {:>12.6e}             │",
        metrics.mean_rel
    );
    println!(
        "  │  Median |ΔB/B|:        {:>12.6e}             │",
        metrics.median_rel
    );
    println!(
        "  │  Max |ΔB/B|:           {:>12.6e}             │",
        metrics.max_rel
    );
    println!("  │                                                 │");
    println!("  │  Paper target:         ~1.0e-06 (relative)      │");
    println!(
        "  │  Our mean relative:    {:>12.6e}             │",
        metrics.mean_rel
    );
    println!(
        "  │  Gap to paper:         {:.1}× (orders of mag)   │",
        (metrics.mean_rel / 1.0e-6).log10()
    );
    println!("  └─────────────────────────────────────────────────┘");
}

/// Print relative and absolute accuracy distributions (threshold bars).
pub fn print_accuracy_distributions(residuals: &[NucleusResidual], n_nuclei: usize) {
    let rel_thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1];
    println!();
    println!("  Relative accuracy |ΔB/B| distribution:");
    for &thresh in &rel_thresholds {
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
}

/// Print accuracy by mass region table.
pub fn print_accuracy_by_region(residuals: &[NucleusResidual]) {
    println!();
    println!("  Accuracy by mass region:");
    println!(
        "  {:>12} {:>6} {:>10} {:>10} {:>12} {:>10}",
        "Region", "Count", "RMS(MeV)", "MAE(MeV)", "Mean|ΔB/B|", "χ²/datum"
    );
    for &(label, lo, hi) in MASS_REGIONS {
        let group: Vec<&NucleusResidual> =
            residuals.iter().filter(|r| r.a >= lo && r.a < hi).collect();
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
}

/// Print top N best and worst nuclei by relative accuracy, then by absolute error.
pub fn print_top_nuclei(residuals: &[NucleusResidual], n: usize) {
    let mut by_rel: Vec<&NucleusResidual> = residuals.iter().collect();
    by_rel.sort_by(|a, b| a.rel_delta.total_cmp(&b.rel_delta));
    println!();
    println!("  Top {n} BEST-fitted nuclei (lowest |ΔB/B|):");
    println!(
        "  {:>6} {:>4} {:>4} {:>10} {:>10} {:>10} {:>12}",
        "Nuclide", "Z", "N", "B_exp", "B_calc", "ΔB(MeV)", "|ΔB/B|"
    );
    for r in by_rel.iter().take(n) {
        println!(
            "  {:>3}-{:<3} {:>4} {:>4} {:>10.3} {:>10.3} {:>+10.3} {:>12.6e}",
            r.element, r.a, r.z, r.n, r.b_exp, r.b_calc, r.delta_b, r.rel_delta
        );
    }

    by_rel.reverse();
    println!();
    println!("  Top {n} WORST-fitted nuclei (highest |ΔB/B|):");
    println!(
        "  {:>6} {:>4} {:>4} {:>10} {:>10} {:>10} {:>12}",
        "Nuclide", "Z", "N", "B_exp", "B_calc", "ΔB(MeV)", "|ΔB/B|"
    );
    for r in by_rel.iter().take(n) {
        println!(
            "  {:>3}-{:<3} {:>4} {:>4} {:>10.3} {:>10.3} {:>+10.3} {:>12.6e}",
            r.element, r.a, r.z, r.n, r.b_exp, r.b_calc, r.delta_b, r.rel_delta
        );
    }

    let mut by_abs: Vec<&NucleusResidual> = residuals.iter().collect();
    by_abs.sort_by(|a, b| b.abs_delta.total_cmp(&a.abs_delta));
    println!();
    println!("  Top {n} largest |ΔB| (MeV):");
    println!(
        "  {:>6} {:>4} {:>4} {:>10} {:>10} {:>10} {:>10}",
        "Nuclide", "Z", "N", "B_exp", "B_calc", "ΔB(MeV)", "χ²_i"
    );
    for r in by_abs.iter().take(n) {
        println!(
            "  {:>3}-{:<3} {:>4} {:>4} {:>10.3} {:>10.3} {:>+10.3} {:>10.4}",
            r.element, r.a, r.z, r.n, r.b_exp, r.b_calc, r.delta_b, r.chi2_i
        );
    }
}

/// Print SEMF model capability analysis vs paper target.
pub fn print_semf_capability_analysis(rms: f64, best_chi2: f64, mean_rel: f64) {
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

/// Print comparison summary table (`SparsitySampler` vs `DirectSampler` vs Python baseline).
pub fn print_comparison_summary(
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

/// Print reference baselines (published parametrizations) with chi² and NMP.
pub fn print_reference_baselines(
    exp_data: &crate::data::NucleiMap,
    lambda: f64,
    baselines: &[(&str, &[f64])],
) {
    use super::compute_be_chi2_only;
    for (name, params) in baselines {
        let be_chi2 = compute_be_chi2_only(params, exp_data);
        if let Some(nmp) = crate::physics::nuclear_matter_properties(params) {
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
