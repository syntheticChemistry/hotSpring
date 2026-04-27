// SPDX-License-Identifier: AGPL-3.0-or-later

//! L1/L2 optimization objective functions for nuclear EOS parameter fitting.
//!
//! Each objective combines binding-energy chi² with NMP (nuclear matter properties)
//! constraints via a Lagrange multiplier `lambda`. Invalid parameters (unphysical
//! saturation density, positive E/A) return a large penalty via `ln(1 + 10⁴)`.

use crate::data::{NucleiEntry, NucleiMap};
use crate::physics::{binding_energy_l2, nuclear_matter_properties, semf_binding_energy};
use crate::provenance;
use crate::tolerances;

use rayon::prelude::*;

/// L1 objective with NMP chi-squared constraint (UNEDF-style).
/// `chi2_total` = `chi2_BE/datum` + lambda * `chi2_NMP/datum`.
#[must_use]
pub fn l1_objective_nmp(x: &[f64], exp_data: &NucleiMap, lambda: f64) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    let Some(nmp) = nuclear_matter_properties(x) else {
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
    exp_data: &std::sync::Arc<NucleiMap>,
    lambda: f64,
) -> impl Fn(&[f64]) -> f64 {
    let exp_data = exp_data.clone();
    move |x: &[f64]| l1_objective_nmp(x, &exp_data, lambda)
}

/// L1 chi²/datum for a parameter set (CPU), nuclei as sorted slice.
///
/// Used by `nuclear_eos_gpu` for LHS sweep. Includes NMP validation.
#[must_use]
pub fn l1_chi2_cpu_nuclei(params: &[f64], nuclei: &[NucleiEntry]) -> f64 {
    let Some(nmp) = nuclear_matter_properties(params) else {
        return 1e10;
    };
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 || nmp.e_a_mev > 0.0 {
        return 1e10;
    }

    let mut chi2 = 0.0;
    let mut count = 0;
    for &((z, n), (b_exp, _)) in nuclei {
        let b_calc = semf_binding_energy(z, n, params);
        if b_calc > 0.0 {
            let sigma = tolerances::sigma_theo(b_exp);
            chi2 += ((b_calc - b_exp) / sigma).powi(2);
            count += 1;
        }
    }
    if count == 0 {
        return 1e10;
    }
    chi2 / f64::from(count)
}

/// L1 objective with NMP constraint, nuclei as sorted slice.
///
/// Same as `l1_objective_nmp` but accepts `&[NucleiEntry]`.
#[must_use]
pub fn l1_objective_nmp_nuclei(x: &[f64], nuclei: &[NucleiEntry], lambda: f64) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    let Some(nmp) = nuclear_matter_properties(x) else {
        return (1e4_f64).ln_1p();
    };
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 {
        return (1e4_f64).ln_1p();
    }
    if nmp.e_a_mev > 0.0 {
        return (1e4_f64).ln_1p();
    }

    let mut chi2_be = 0.0;
    let mut count = 0;
    for &((z, n), (b_exp, _)) in nuclei {
        let b_calc = semf_binding_energy(z, n, x);
        if b_calc > 0.0 {
            let sigma = tolerances::sigma_theo(b_exp);
            chi2_be += ((b_calc - b_exp) / sigma).powi(2);
            count += 1;
        }
    }
    if count == 0 {
        return (1e4_f64).ln_1p();
    }

    let chi2_be_datum = chi2_be / f64::from(count);
    let chi2_nmp = provenance::nmp_chi2_from_props(&nmp) / 5.0;
    let total = lambda.mul_add(chi2_nmp, chi2_be_datum);
    total.ln_1p()
}

/// L2 objective (HFB + NMP constraint), exp_data as HashMap, rayon-parallel.
///
/// Used by `nuclear_eos_gpu` for L2 DirectSampler. Uses `lambda * chi2_NMP/5`.
#[must_use]
pub fn l2_objective_nmp_exp_data(x: &[f64], exp_data: &NucleiMap, lambda: f64) -> f64 {
    if x[8] <= 0.01 || x[8] > 1.0 {
        return (1e4_f64).ln_1p();
    }

    let Some(nmp) = nuclear_matter_properties(x) else {
        return (1e4_f64).ln_1p();
    };
    if nmp.rho0_fm3 < 0.05 || nmp.rho0_fm3 > 0.30 {
        return (1e4_f64).ln_1p();
    }
    if nmp.e_a_mev > 0.0 {
        return (1e4_f64).ln_1p();
    }

    let results: Vec<f64> = exp_data
        .par_iter()
        .filter_map(|(&(z, n), &(b_exp, _))| {
            let (b_calc, _) = binding_energy_l2(z, n, x).unwrap_or((0.0, false));
            if b_calc > 0.0 {
                let sigma = tolerances::sigma_theo(b_exp);
                Some(((b_calc - b_exp) / sigma).powi(2))
            } else {
                None
            }
        })
        .collect();

    let count = results.len();
    if count == 0 {
        return (1e4_f64).ln_1p();
    }

    let chi2_be_datum = results.iter().sum::<f64>() / count as f64;
    let chi2_nmp = provenance::nmp_chi2_from_props(&nmp) / 5.0;
    let total = lambda.mul_add(chi2_nmp, chi2_be_datum);
    total.ln_1p()
}
