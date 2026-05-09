// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rational Hybrid Monte Carlo (RHMC) infrastructure for fractional powers
//! of the fermion determinant.
//!
//! Enables simulation of non-multiples-of-4 flavor counts (Nf=2, 2+1)
//! via the "rooting trick": det(D†D)^{Nf/8} ≈ product of rational functions.
//!
//! # Architecture
//!
//! - [`rational`]: Partial-fraction approximation to x^{p/q}
//! - [`multishift_cg`]: Simultaneous multi-shift CG solver
//! - [`remez`]: Remez exchange for optimal residue fitting
//!
//! # References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — RHMC algorithm
//! - Clark, "The Rational Hybrid Monte Carlo Algorithm" (2006)
//! - Remez algorithm for optimal rational approximation

mod multishift_cg;
pub mod rational;
mod remez;

pub use multishift_cg::{MultiShiftCgResult, multi_shift_cg_solve};
pub use rational::RationalApproximation;

use super::complex_f64::Complex64;
use super::dirac::FermionField;
use super::wilson::Lattice;

/// RHMC configuration for a single fermion taste/flavor sector.
#[derive(Clone, Debug)]
pub struct RhmcFermionConfig {
    /// Mass for this sector.
    pub mass: f64,
    /// Power of the determinant (e.g. 1/4 for one rooted staggered taste).
    pub det_power: f64,
    /// Rational approximation for the action: `x^{det_power`}.
    pub action_approx: RationalApproximation,
    /// Rational approximation for the heatbath: x^{-det_power/2}.
    pub heatbath_approx: RationalApproximation,
    /// Rational approximation for the force: `x^{det_power`} (derivative form).
    pub force_approx: RationalApproximation,
}

impl RhmcFermionConfig {
    /// Build a fermion sector from mass, det_power, and spectral parameters.
    #[must_use]
    fn from_spectral(
        mass: f64,
        det_power: f64,
        n_poles: usize,
        range_min: f64,
        range_max: f64,
    ) -> Self {
        let action_force =
            RationalApproximation::generate(-det_power, n_poles, range_min, range_max);
        Self {
            mass,
            det_power,
            action_approx: action_force.clone(),
            heatbath_approx: RationalApproximation::generate(
                det_power / 2.0,
                n_poles,
                range_min,
                range_max,
            ),
            force_approx: action_force,
        }
    }
}

/// RHMC HMC configuration.
#[derive(Clone, Debug)]
pub struct RhmcConfig {
    /// Fermion sectors (one per flavor group).
    pub sectors: Vec<RhmcFermionConfig>,
    /// Gauge coupling.
    pub beta: f64,
    /// MD step size.
    pub dt: f64,
    /// Number of MD steps.
    pub n_md_steps: usize,
    /// CG tolerance.
    pub cg_tol: f64,
    /// CG max iterations.
    pub cg_max_iter: usize,
}

/// Default MD/CG parameters for standard RHMC configurations.
const DEFAULT_DT: f64 = 0.01;
const DEFAULT_MD_STEPS: usize = 50;
const DEFAULT_CG_TOL: f64 = 1e-8;
const DEFAULT_CG_MAX_ITER: usize = 5000;
const DEFAULT_RANGE: (f64, f64) = (0.01, 64.0);
const DEFAULT_POLES: usize = 8;

impl RhmcConfig {
    /// Calibrated Nf=2 configuration using discovered spectral range.
    ///
    /// All parameters derived from measurements — no hardcoded spectral
    /// bounds. The spectral range comes from `SpectralInfo` (measured via
    /// GPU power iteration), and n_poles/dt/n_md are set by the calibrator.
    #[must_use]
    pub fn calibrated_nf2(
        mass: f64,
        beta: f64,
        n_poles: usize,
        range_min: f64,
        range_max: f64,
        dt: f64,
        n_md_steps: usize,
        cg_tol: f64,
        cg_max_iter: usize,
    ) -> Self {
        Self {
            sectors: vec![RhmcFermionConfig::from_spectral(
                mass, 0.25, n_poles, range_min, range_max,
            )],
            beta,
            dt,
            n_md_steps,
            cg_tol,
            cg_max_iter,
        }
    }

    /// Single-sector helper with default MD/CG parameters.
    fn single_sector(mass: f64, det_power: f64, n_poles: usize, beta: f64) -> Self {
        let (lo, hi) = DEFAULT_RANGE;
        Self {
            sectors: vec![RhmcFermionConfig::from_spectral(
                mass, det_power, n_poles, lo, hi,
            )],
            beta,
            dt: DEFAULT_DT,
            n_md_steps: DEFAULT_MD_STEPS,
            cg_tol: DEFAULT_CG_TOL,
            cg_max_iter: DEFAULT_CG_MAX_ITER,
        }
    }

    /// Nf=1 configuration: det(D†D)^{1/8}.
    #[must_use]
    pub fn nf1(mass: f64, beta: f64) -> Self {
        Self::single_sector(mass, 0.125, DEFAULT_POLES, beta)
    }

    /// Nf=2 configuration: det(D†D)^{1/4} (one rooted staggered field).
    #[must_use]
    pub fn nf2(mass: f64, beta: f64) -> Self {
        Self::single_sector(mass, 0.25, DEFAULT_POLES, beta)
    }

    /// Nf=3 configuration: det(D†D)^{3/8} (more poles for accuracy).
    #[must_use]
    pub fn nf3(mass: f64, beta: f64) -> Self {
        Self::single_sector(mass, 0.375, 12, beta)
    }

    /// Nf=4 configuration: det(D†D)^{1/2} (no rooting needed).
    #[must_use]
    pub fn nf4(mass: f64, beta: f64) -> Self {
        Self::single_sector(mass, 0.5, DEFAULT_POLES, beta)
    }

    /// Nf=2+1 configuration: light (u,d) det^{1/4} + strange det^{1/8}.
    #[must_use]
    pub fn nf2p1(light_mass: f64, strange_mass: f64, beta: f64) -> Self {
        let (lo, hi) = DEFAULT_RANGE;
        Self {
            sectors: vec![
                RhmcFermionConfig::from_spectral(light_mass, 0.25, DEFAULT_POLES, lo, hi),
                RhmcFermionConfig::from_spectral(strange_mass, 0.125, DEFAULT_POLES, lo, hi),
            ],
            beta,
            dt: DEFAULT_DT,
            n_md_steps: DEFAULT_MD_STEPS,
            cg_tol: DEFAULT_CG_TOL,
            cg_max_iter: DEFAULT_CG_MAX_ITER,
        }
    }

    /// Nf=2+1+1 configuration: light (u,d) + strange + charm.
    #[must_use]
    pub fn nf2p1p1(light_mass: f64, strange_mass: f64, charm_mass: f64, beta: f64) -> Self {
        let (lo, hi) = DEFAULT_RANGE;
        Self {
            sectors: vec![
                RhmcFermionConfig::from_spectral(light_mass, 0.25, DEFAULT_POLES, lo, hi),
                RhmcFermionConfig::from_spectral(strange_mass, 0.125, DEFAULT_POLES, lo, hi),
                RhmcFermionConfig::from_spectral(charm_mass, 0.125, DEFAULT_POLES, lo, hi),
            ],
            beta,
            dt: 0.008,
            n_md_steps: 60,
            cg_tol: DEFAULT_CG_TOL,
            cg_max_iter: DEFAULT_CG_MAX_ITER,
        }
    }
}

/// RHMC pseudofermion heatbath: generate φ = (D†D)^{-p/2} η where η ~ N(0,1).
///
/// Uses the rational approximation for x^{-p/2} and multi-shift CG.
pub fn rhmc_heatbath(
    lattice: &Lattice,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: &mut u64,
) -> (FermionField, usize) {
    let vol = lattice.volume();
    let approx = &config.heatbath_approx;

    let mut eta = FermionField::zeros(vol);
    for site in &mut eta.data {
        for c in site.iter_mut() {
            let re = super::constants::lcg_gaussian(seed);
            let im = super::constants::lcg_gaussian(seed);
            *c = Complex64::new(re, im);
        }
    }

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        &eta,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut phi = FermionField::zeros(vol);
    for i in 0..vol {
        for c in 0..3 {
            let mut val = eta.data[i][c].scale(approx.alpha_0);
            for (s, x_s) in solutions.iter().enumerate() {
                val += x_s.data[i][c].scale(approx.alpha[s]);
            }
            phi.data[i][c] = val;
        }
    }

    (phi, result.iterations)
}

/// RHMC fermion action: `S_f` = φ† r(p)(D†D) φ.
///
/// Uses the rational approximation for x^p and multi-shift CG.
#[must_use]
pub fn rhmc_fermion_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (f64, usize) {
    let approx = &config.action_approx;

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        phi,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut action = approx.alpha_0 * phi.dot(phi).re;
    for (s, x_s) in solutions.iter().enumerate() {
        action += approx.alpha[s] * phi.dot(x_s).re;
    }

    (action, result.iterations)
}

/// RHMC fermion force: `dS_f/dU` = Σ αᵢ · d/dU [φ† (D†D + σᵢ)⁻¹ φ].
///
/// Each term in the sum is a standard pseudofermion force evaluated at the
/// shifted CG solution. Returns the total force summed over all poles.
#[must_use]
pub fn rhmc_fermion_force(
    lattice: &Lattice,
    phi: &FermionField,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (Vec<super::su3::Su3Matrix>, usize) {
    let vol = lattice.volume();
    let approx = &config.force_approx;

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        phi,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut total_force = vec![super::su3::Su3Matrix::ZERO; vol * 4];

    for (s, x_s) in solutions.iter().enumerate() {
        let f_s = super::pseudofermion::pseudofermion_force(lattice, x_s, config.mass);
        for (tf, fs) in total_force.iter_mut().zip(f_s.iter()) {
            *tf = *tf + fs.scale(approx.alpha[s]);
        }
    }

    (total_force, result.iterations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_approx_evaluates_correctly() {
        let r = RationalApproximation::fourth_root_8pole();
        let x = 1.0;
        let approx_val = r.eval(x);
        let exact_val = 1.0_f64;
        assert!(
            (approx_val - exact_val).abs() < 0.01,
            "fourth_root(1.0) = {approx_val}, expected ~1.0"
        );

        let x = 16.0;
        let approx_val = r.eval(x);
        let exact_val = 2.0;
        assert!(
            (approx_val - exact_val).abs() < 0.01,
            "fourth_root(16.0) = {approx_val}, expected ~2.0"
        );
    }

    #[test]
    fn inv_fourth_root_evaluates_correctly() {
        let r = RationalApproximation::inv_fourth_root_8pole();
        let x = 1.0;
        let approx_val = r.eval(x);
        let exact_val = 1.0;
        assert!(
            (approx_val - exact_val).abs() < 0.01,
            "inv_fourth_root(1.0) = {approx_val}, expected ~1.0"
        );
    }

    #[test]
    fn sqrt_evaluates_correctly() {
        let r = RationalApproximation::sqrt_8pole();
        let x = 4.0;
        let approx_val = r.eval(x);
        let exact_val = 2.0;
        let rel_err = (approx_val - exact_val).abs() / exact_val;
        assert!(
            rel_err < 0.02,
            "sqrt(4.0) = {approx_val}, expected ~2.0, rel_err={rel_err:.4}"
        );
        assert!(
            r.max_relative_error < 0.05,
            "max relative error {} too large for 8-pole sqrt",
            r.max_relative_error
        );
    }
}
