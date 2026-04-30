// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::physics::NuclearMatterProps;

/// Soft penalty for NMP outside preferred ranges in L1/L2 objective functions.
///
/// Uses looser bounds than [`crate::prescreen::NMPConstraints`]: ρ₀ ∈ [0.08, 0.25], E/A ≤ -5 `MeV`.
/// Returns a positive penalty to add to χ² before ln(1+χ²).
#[must_use]
pub fn nmp_objective_penalty(nmp: &NuclearMatterProps) -> f64 {
    let mut penalty = 0.0;
    if nmp.rho0_fm3 < 0.08 {
        penalty += 50.0 * (0.08 - nmp.rho0_fm3) / 0.08;
    } else if nmp.rho0_fm3 > 0.25 {
        penalty += 50.0 * (nmp.rho0_fm3 - 0.25) / 0.25;
    }
    if nmp.e_a_mev > -5.0 {
        penalty += 20.0 * (nmp.e_a_mev + 5.0).max(0.0);
    }
    penalty
}

/// Perturb base parameters by random amounts scaled by `scale` (fraction of
/// parameter range, e.g. 0.1 = ±5%, 0.2 = ±10%), then clamp to bounds.
///
/// Uses a simple LCG for reproducibility. Updates `rng` in place.
#[must_use]
pub fn perturb_params(base: &[f64], bounds: &[(f64, f64)], rng: &mut u64, scale: f64) -> Vec<f64> {
    let mut candidate = base.to_vec();
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        let range = hi - lo;
        *rng = rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (*rng >> 33) as f64 / (1u64 << 31) as f64;
        let perturbation = (u - 0.5) * scale * range;
        candidate[i] = (candidate[i] + perturbation).clamp(lo, hi);
    }
    candidate
}
