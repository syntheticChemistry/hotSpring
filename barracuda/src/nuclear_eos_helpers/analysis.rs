// SPDX-License-Identifier: AGPL-3.0-or-later

//! Residual analysis: build per-nucleus residuals, compute aggregate metrics,
//! export to JSON, and orchestrate deep analysis reports.

use std::path::Path;

use crate::data::{self, NucleiMap};
use crate::physics::semf_binding_energy;
use crate::tolerances;

use super::{NucleusResidual, ResidualMetrics};

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
    use super::reporting::{
        print_accuracy_by_region, print_accuracy_distributions, print_accuracy_metrics_box,
        print_best_parameters, print_semf_capability_analysis, print_top_nuclei,
    };
    use crate::data;
    use crate::provenance;

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

/// Quick chi² helper forwarded from `NucleiMap` for use in objectives.
#[must_use]
pub fn compute_be_chi2_from_map(params: &[f64], exp_data: &NucleiMap) -> f64 {
    super::compute_be_chi2_only(params, exp_data)
}
