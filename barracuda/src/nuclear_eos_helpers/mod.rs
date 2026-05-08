// SPDX-License-Identifier: AGPL-3.0-or-later

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

pub mod objectives;
pub use objectives::{
    l1_chi2_cpu_nuclei, l1_objective_nmp, l1_objective_nmp_nuclei, l2_objective_nmp_exp_data,
    make_l1_objective_nmp,
};

pub mod display;
pub use display::*;

use crate::data::{self, NucleiMap};
use crate::physics::semf_binding_energy;
use crate::tolerances;

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
    /// Experimental binding energy (`MeV`)
    pub b_exp: f64,
    /// Calculated binding energy (`MeV`)
    pub b_calc: f64,
    /// Signed residual (`B_calc` - `B_exp`)
    pub delta_b: f64,
    /// Absolute residual (`MeV`)
    pub abs_delta: f64,
    /// Relative accuracy |ΔB/B|
    pub rel_delta: f64,
    /// Chi² contribution for this nucleus
    pub chi2_i: f64,
}

/// Summary metrics from per-nucleus residuals.
#[derive(Debug)]
pub struct ResidualMetrics {
    /// RMS of residuals (`MeV`)
    pub rms: f64,
    /// Mean absolute error (`MeV`)
    pub mae: f64,
    /// Max absolute error (`MeV`)
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
pub fn compute_be_chi2_only(params: &[f64], exp_data: &NucleiMap) -> f64 {
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
    exp_data: &NucleiMap,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_mean_std_basic() {
        let (mean, std) = compute_mean_std(&[1.0, 2.0, 3.0]);
        assert!((mean - 2.0).abs() < 1e-14, "mean should be 2, got {mean}");
        assert!((std - 1.0).abs() < 1e-14, "std should be 1, got {std}");
    }

    #[test]
    fn compute_mean_std_constant() {
        let (mean, std) = compute_mean_std(&[5.0, 5.0, 5.0]);
        assert!((mean - 5.0).abs() < 1e-14);
        assert!(std.abs() < 1e-14);
    }

    #[test]
    fn compute_mean_std_empty() {
        let (mean, std) = compute_mean_std(&[]);
        assert!((mean).abs() < 1e-14);
        assert!((std).abs() < 1e-14);
    }

    #[test]
    fn compute_mean_std_single() {
        let (mean, std) = compute_mean_std(&[42.0]);
        assert!((mean - 42.0).abs() < 1e-14);
        assert!(std.abs() < 1e-14);
    }

    #[test]
    fn compute_residual_metrics_basic() {
        let residuals = vec![
            NucleusResidual {
                z: 8,
                n: 8,
                a: 16,
                element: "O".to_string(),
                b_exp: 127.619,
                b_calc: 127.0,
                delta_b: -0.619,
                abs_delta: 0.619,
                rel_delta: 0.619 / 127.619,
                chi2_i: 0.619_f64.powi(2) / 0.5_f64.powi(2),
            },
            NucleusResidual {
                z: 20,
                n: 20,
                a: 40,
                element: "Ca".to_string(),
                b_exp: 342.052,
                b_calc: 342.5,
                delta_b: 0.448,
                abs_delta: 0.448,
                rel_delta: 0.448 / 342.052,
                chi2_i: 0.448_f64.powi(2) / 0.5_f64.powi(2),
            },
        ];
        let m = compute_residual_metrics(&residuals);
        assert_eq!(m.n_nuclei, 2);
        assert!(m.rms > 0.0, "RMS should be positive");
        assert!(m.mae > 0.0, "MAE should be positive");
        assert!(m.max_err > 0.0, "max error should be positive");
        assert!(m.mean_rel > 0.0, "mean relative should be positive");
    }

    #[test]
    fn compute_residual_metrics_empty() {
        let m = compute_residual_metrics(&[]);
        assert_eq!(m.n_nuclei, 0);
        assert!(m.rms.abs() < 1e-15);
    }

    #[test]
    fn compute_be_chi2_sly4_is_finite() {
        use crate::data::load_experimental_data;
        use crate::discovery::try_discover_data_root;

        let Ok(root) = try_discover_data_root() else {
            return;
        };
        let Ok(exp_data) = load_experimental_data(&root) else {
            return;
        };
        let sly4_params = crate::provenance::SLY4_PARAMS;
        let chi2 = compute_be_chi2_only(&sly4_params, &exp_data);
        assert!(chi2.is_finite(), "chi2 should be finite");
        assert!(chi2 > 0.0, "chi2 should be positive");
    }
}
