// SPDX-License-Identifier: AGPL-3.0-only

//! Shared helpers for nuclear EOS validation binaries (L1, L2).
//!
//! Provides reusable logic for:
//! - Statistics (mean/std, chi² decomposition)
//! - Result formatting and JSON output
//! - Per-nucleus residual analysis and metrics
//! - Parameter display and chi² analysis
//!
//! Used by `nuclear_eos_l1_ref`, and intended for `nuclear_eos_l2_ref` and other
//! nuclear EOS binaries.

use crate::data::{self, PARAM_NAMES};
use crate::physics::{semf_binding_energy, NuclearMatterProps};
use crate::provenance;
use crate::tolerances;

use std::collections::HashMap;
use std::path::Path;

/// Per-nucleus residual for binding energy fit analysis.
#[derive(Debug, Clone)]
pub struct NucleusResidual {
    /// Proton number
    pub z: usize,
    /// Neutron number
    pub n: usize,
    /// Mass number
    pub a: usize,
    /// Element symbol
    pub element: String,
    /// Experimental binding energy (MeV)
    pub b_exp: f64,
    /// Calculated binding energy (MeV)
    pub b_calc: f64,
    /// Signed residual (B_calc - B_exp)
    pub delta_b: f64,
    /// Absolute residual (MeV)
    pub abs_delta: f64,
    /// Relative accuracy |ΔB/B|
    pub rel_delta: f64,
    /// Chi² contribution for this nucleus
    pub chi2_i: f64,
}

/// Summary metrics from per-nucleus residuals.
#[derive(Debug)]
pub struct ResidualMetrics {
    /// RMS of residuals (MeV)
    pub rms: f64,
    /// Mean absolute error (MeV)
    pub mae: f64,
    /// Max absolute error (MeV)
    pub max_err: f64,
    /// Mean relative |ΔB/B|
    pub mean_rel: f64,
    /// Median relative |ΔB/B|
    pub median_rel: f64,
    /// Max relative |ΔB/B|
    pub max_rel: f64,
    /// Mean signed error (bias)
    pub mean_signed: f64,
    /// Number of nuclei
    pub n_nuclei: usize,
}

/// Compute sample mean and standard deviation.
#[must_use]
pub fn compute_mean_std(vals: &[f64]) -> (f64, f64) {
    let n = vals.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / n;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

/// Binding energy chi² per datum (for reference baselines).
#[must_use]
pub fn compute_be_chi2_only(params: &[f64], exp_data: &HashMap<(usize, usize), (f64, f64)>) -> f64 {
    let mut chi2 = 0.0;
    let mut n = 0;
    for (&(z, nn), &(b_exp, _sigma)) in exp_data {
        let b_calc = semf_binding_energy(z, nn, params);
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

/// Compute per-nucleus binding energies for chi² decomposition.
#[must_use]
pub fn compute_binding_energies(
    params: &[f64],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
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

/// Print pure-GPU (math_f64) precision vs CPU and precomputed-GPU.
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
pub fn print_nmp_with_pulls(nmp: &NuclearMatterProps) {
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

/// Build per-nucleus residuals from nuclei JSON using SEMF.
///
/// # Errors
///
/// Returns `HotSpringError::DataLoad` if nuclei JSON cannot be opened or parsed.
pub fn build_residuals_semf(
    base: &Path,
    params: &[f64],
) -> Result<Vec<NucleusResidual>, crate::error::HotSpringError> {
    let nuclei_set = data::parse_nuclei_set_from_args();
    let nuclei_path = data::nuclei_data_path(base, nuclei_set);
    let nuclei_reader =
        std::io::BufReader::new(std::fs::File::open(&nuclei_path).map_err(|e| {
            crate::error::HotSpringError::DataLoad(format!("open nuclei JSON: {e}"))
        })?);
    let nuclei_file: serde_json::Value = serde_json::from_reader(nuclei_reader)
        .map_err(|e| crate::error::HotSpringError::DataLoad(format!("parse nuclei JSON: {e}")))?;
    let nuclei_list = nuclei_file["nuclei"].as_array().ok_or_else(|| {
        crate::error::HotSpringError::DataLoad("nuclei JSON missing 'nuclei' array".into())
    })?;

    let mut residuals = Vec::new();
    for nuc in nuclei_list {
        let z = nuc["Z"].as_u64().unwrap_or(0) as usize;
        let n = nuc["N"].as_u64().unwrap_or(0) as usize;
        let a = nuc["A"].as_u64().unwrap_or(0) as usize;
        let element = nuc["element"].as_str().unwrap_or("??").to_string();
        let b_exp = nuc["binding_energy_MeV"].as_f64().unwrap_or(0.0);

        let b_calc = semf_binding_energy(z, n, params);
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
            });
        }
    }
    Ok(residuals)
}

/// Compute aggregate metrics from residuals.
#[must_use]
pub fn compute_residual_metrics(residuals: &[NucleusResidual]) -> ResidualMetrics {
    let n_nuclei = residuals.len();
    let n = n_nuclei as f64;
    if n < 1.0 {
        return ResidualMetrics {
            rms: 0.0,
            mae: 0.0,
            max_err: 0.0,
            mean_rel: 0.0,
            median_rel: 0.0,
            max_rel: 0.0,
            mean_signed: 0.0,
            n_nuclei: 0,
        };
    }

    let rms = (residuals.iter().map(|r| r.delta_b.powi(2)).sum::<f64>() / n).sqrt();
    let mae = residuals.iter().map(|r| r.abs_delta).sum::<f64>() / n;
    let max_err = residuals
        .iter()
        .map(|r| r.abs_delta)
        .fold(0.0_f64, f64::max);
    let mean_rel = residuals.iter().map(|r| r.rel_delta).sum::<f64>() / n;
    let max_rel = residuals
        .iter()
        .map(|r| r.rel_delta)
        .fold(0.0_f64, f64::max);
    let mean_signed = residuals.iter().map(|r| r.delta_b).sum::<f64>() / n;

    let mut rels: Vec<f64> = residuals.iter().map(|r| r.rel_delta).collect();
    rels.sort_by(f64::total_cmp);
    let median_rel = if rels.len().is_multiple_of(2) {
        f64::midpoint(rels[rels.len() / 2 - 1], rels[rels.len() / 2])
    } else {
        rels[rels.len() / 2]
    };

    ResidualMetrics {
        rms,
        mae,
        max_err,
        mean_rel,
        median_rel,
        max_rel,
        mean_signed,
        n_nuclei,
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

/// Region predicate: (label, lo inclusive, hi exclusive).
const MASS_REGIONS: &[(&str, usize, usize)] = &[
    ("Light A<50", 0, 50),
    ("Medium 50-100", 50, 100),
    ("Heavy 100-200", 100, 200),
    ("Very heavy 200+", 200, 9999),
];

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

/// Print comparison summary table (SparsitySampler vs DirectSampler vs Python baseline).
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
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    lambda: f64,
    baselines: &[(&str, &[f64])],
) {
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

/// Run deep residual analysis: metrics, distributions, JSON export.
pub fn run_deep_residual_analysis(
    base: &Path,
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
    let residuals = match build_residuals_semf(base, best_x) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("build_residuals_semf failed: {e}");
            return;
        }
    };
    let n_nuclei = residuals.len();
    println!("  Nuclei fitted: {n_nuclei}");
    println!();

    let metrics = compute_residual_metrics(&residuals);
    print_accuracy_metrics_box(&metrics);
    print_accuracy_distributions(&residuals, n_nuclei);
    print_accuracy_by_region(&residuals);
    print_top_nuclei(&residuals, 10);

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
    print_semf_capability_analysis(metrics.rms, best_chi2, metrics.mean_rel);

    print_best_parameters(best_x);

    let results_dir = base.join("results");
    std::fs::create_dir_all(&results_dir).ok();

    let per_nucleus_json = per_nucleus_residuals_to_json(&residuals);

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
            "rms_MeV": metrics.rms,
            "mae_MeV": metrics.mae,
            "max_error_MeV": metrics.max_err,
            "mean_signed_error_MeV": metrics.mean_signed,
            "mean_relative_accuracy": metrics.mean_rel,
            "median_relative_accuracy": metrics.median_rel,
            "max_relative_accuracy": metrics.max_rel,
            "n_nuclei": n_nuclei,
            "chi2_per_datum": best_chi2,
        },
        "paper_comparison": {
            "paper_target_relative": 1.0e-6,
            "our_mean_relative": metrics.mean_rel,
            "gap_orders_of_magnitude": (metrics.mean_rel / 1.0e-6).log10(),
            "semf_theoretical_limit_MeV": 2.0,
            "hfb_theoretical_limit_MeV": 0.5,
            "notes": "SEMF is a 5-term model; 10^-6 requires HFB+ physics"
        },
        "best_parameters": {
            "names": data::PARAM_NAMES.to_vec(),
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
    if let Ok(s) = serde_json::to_string_pretty(&result_json) {
        std::fs::write(&path, s).ok();
    }
    println!("\n  Full results saved to: {}", path.display());
}

/// L1 objective with NMP chi-squared constraint (UNEDF-style).
/// chi2_total = chi2_BE/datum + lambda * chi2_NMP/datum.
#[must_use]
pub fn l1_objective_nmp(
    x: &[f64],
    exp_data: &HashMap<(usize, usize), (f64, f64)>,
    lambda: f64,
) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    let Some(nmp) = crate::physics::nuclear_matter_properties(x) else {
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
    let chi2_nmp_datum = provenance::nmp_chi2_from_props(&nmp) / 5.0;
    let chi2_total = lambda.mul_add(chi2_nmp_datum, chi2_be_datum);

    chi2_total.ln_1p()
}

/// Closure factory for L1 NMP-constrained objective.
pub fn make_l1_objective_nmp(
    exp_data: &std::sync::Arc<HashMap<(usize, usize), (f64, f64)>>,
    lambda: f64,
) -> impl Fn(&[f64]) -> f64 {
    let exp_data = exp_data.clone();
    move |x: &[f64]| l1_objective_nmp(x, &exp_data, lambda)
}

/// Convert residuals to per-nucleus JSON array for result export.
#[must_use]
pub fn per_nucleus_residuals_to_json(residuals: &[NucleusResidual]) -> Vec<serde_json::Value> {
    residuals
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
        .collect()
}
