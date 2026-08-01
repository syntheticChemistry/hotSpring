// SPDX-License-Identifier: AGPL-3.0-or-later

// Anchored constants — Cremer & Pople JACS 97, 1354 (1975)
pub(crate) const BASIN_4C1_MAX_DEG: f64 = 40.0;
pub(crate) const BASIN_1C4_MIN_DEG: f64 = 140.0;
// Gaussian truncation — captures 99.7% of kernel (exp(-4.5) residual)
pub(crate) const GRID_MARGIN_SIGMA: f64 = 3.0;
// Parity RMSD tolerance — √(tier_1d² + tier_1d²) rounded; see threshold_calibration.toml
pub(crate) const PARITY_TOLERANCE_KJMOL: f64 = 2.0;
// KS test coefficient — Marsaglia et al. J. Stat. Software 8(18), 2003
pub(crate) const KS_CRITICAL_COEFF: f64 = 1.36;
// Wall bias detection threshold — corresponds to 0.002 nm beyond AT with KAPPA=500
pub(crate) const WALL_ACTIVE_THRESHOLD: f64 = 0.001;
// Cross-landscape verdict midpoint factor (equal-width SUSPICIOUS band)
pub(crate) const VERDICT_MIDPOINT: f64 = 0.5;
