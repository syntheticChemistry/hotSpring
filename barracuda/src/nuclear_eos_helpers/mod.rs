// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shared helpers for nuclear EOS validation binaries (L1, L2).
//!
//! Provides reusable logic for:
//! - Statistics (mean/std, chi² decomposition)
//! - Result formatting and JSON output (`reporting`)
//! - Per-nucleus residual analysis and metrics (`analysis`)
//! - Parameter display and chi² analysis
//!
//! Used by `nuclear_eos_l1_ref`, and intended for `nuclear_eos_l2_ref` and other
//! nuclear EOS binaries.

pub mod analysis;
pub mod objectives;
pub mod persistence;
pub mod reporting;

pub use analysis::{
    build_residuals_semf, compute_residual_metrics, per_nucleus_residuals_to_json,
    run_deep_residual_analysis,
};
pub use objectives::{
    l1_chi2_cpu_nuclei, l1_objective_nmp, l1_objective_nmp_nuclei, l2_objective_nmp_exp_data,
    make_l1_objective_nmp,
};
pub use persistence::save_results;
pub use reporting::{
    print_accuracy_by_region, print_accuracy_distributions, print_accuracy_metrics_box,
    print_best_parameters, print_comparison_summary, print_nmp_with_pulls,
    print_pure_gpu_precision, print_reference_baselines, print_semf_capability_analysis,
    print_semf_gpu_precision, print_top_nuclei,
};

use crate::data::NucleiMap;
use crate::physics::semf_binding_energy;
use crate::tolerances;

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
