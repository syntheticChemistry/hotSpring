// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::physics::semf_binding_energy;
use std::collections::HashMap;

/// Tier 2: Quick L1 SEMF χ²/datum check.
/// If a parameterization can't fit nuclei at the simple SEMF level,
/// it won't work with HFB either. Cost: ~0.1ms
pub fn l1_proxy_prescreen<S: std::hash::BuildHasher>(
    params: &[f64],
    exp_data: &HashMap<(usize, usize), (f64, f64), S>,
    chi2_threshold: f64,
) -> Option<f64> {
    let mut chi2 = 0.0;
    let mut n_valid = 0;

    for (&(z, n), &(b_exp, _sigma)) in exp_data {
        let b_calc = semf_binding_energy(z, n, params);
        if b_calc > 0.0 {
            let sigma = crate::tolerances::sigma_theo(b_exp);
            chi2 += ((b_calc - b_exp) / sigma).powi(2);
            n_valid += 1;
        }
    }

    if n_valid == 0 {
        return None;
    }

    let chi2_per_datum = chi2 / f64::from(n_valid);
    if chi2_per_datum < chi2_threshold {
        Some(chi2_per_datum)
    } else {
        None
    }
}
