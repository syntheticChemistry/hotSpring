// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::Path;

use crate::constants::KS_CRITICAL_COEFF;
use crate::parse::parse_colvar_theta;
use crate::types::KsTestResult;

/// Two-sample Kolmogorov-Smirnov test.
///
/// Returns the KS statistic D and whether the null hypothesis (same distribution)
/// cannot be rejected at alpha=0.05.
pub fn ks_two_sample(a: &[f64], b: &[f64]) -> KsTestResult {
    let na = a.len();
    let nb = b.len();

    let mut sorted_a = a.to_vec();
    sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut sorted_b = b.to_vec();
    sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let mut ia = 0;
    let mut ib = 0;
    let mut d_max = 0.0_f64;

    while ia < na && ib < nb {
        let va = sorted_a[ia];
        let vb = sorted_b[ib];

        if va <= vb {
            ia += 1;
        }
        if vb <= va {
            ib += 1;
        }

        let fa = ia as f64 / na as f64;
        let fb = ib as f64 / nb as f64;
        d_max = d_max.max((fa - fb).abs());
    }

    let critical = KS_CRITICAL_COEFF * ((na + nb) as f64 / (na as f64 * nb as f64)).sqrt();

    KsTestResult {
        statistic: d_max,
        n_free: na,
        n_bound: nb,
        critical_value_05: critical,
        distributions_same: d_max < critical,
    }
}

/// Compare theta distributions between free and enzyme-bound COLVAR files.
///
/// If distributions are indistinguishable (KS p > 0.05), the substrate may
/// have dissociated.
pub fn compare_colvar_distributions(
    free_colvar: &Path,
    bound_colvar: &Path,
) -> Result<KsTestResult, String> {
    let free_theta = parse_colvar_theta(free_colvar)?;
    let bound_theta = parse_colvar_theta(bound_colvar)?;
    Ok(ks_two_sample(&free_theta, &bound_theta))
}
