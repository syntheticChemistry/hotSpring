// SPDX-License-Identifier: AGPL-3.0-or-later

//! Rational Hybrid Monte Carlo (RHMC) infrastructure for fractional powers
//! of the fermion determinant.
//!
//! Enables simulation of non-multiples-of-4 flavor counts (Nf=2, 2+1)
//! via the "rooting trick": det(D†D)^{Nf/8} ≈ product of rational functions.
//!
//! # Components
//!
//! - [`RationalApproximation`]: Partial fraction coefficients for x^{p/q}
//! - [`multi_shift_cg_solve`]: Solves (D†D + σ_i)x_i = b for all shifts
//!   simultaneously with a single Krylov space
//! - [`RhmcConfig`]: Configuration for RHMC trajectories
//!
//! # References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — RHMC algorithm
//! - Clark, "The Rational Hybrid Monte Carlo Algorithm" (2006)
//! - Remez algorithm for optimal rational approximation

mod multi_shift;
mod rational_approx;
mod remez;

use super::complex_f64::Complex64;
use super::dirac::FermionField;
use super::wilson::Lattice;
use crate::tolerances::lattice::DYNAMICAL_CG_MAX_ITER;

pub use multi_shift::{MultiShiftCgResult, multi_shift_cg_solve};
pub use rational_approx::RationalApproximation;

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
        let det_power = 0.25; // Nf/8 = 2/8
        let action_force =
            RationalApproximation::generate(-det_power, n_poles, range_min, range_max);
        Self {
            sectors: vec![RhmcFermionConfig {
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
            }],
            beta,
            dt,
            n_md_steps,
            cg_tol,
            cg_max_iter,
        }
    }

    /// Nf=2 configuration: one rooted staggered field with det(D†D)^{1/4}.
    ///
    /// Staggered fermions produce 4 tastes, so Nf physical flavors needs
    /// det(D†D)^{Nf/8}. For Nf=2: α = 1/4.
    ///
    /// Pseudofermion setup (Clark & Kennedy, NPB 552):
    /// - Action:   S_f = φ† (D†D)^{-α} φ  →  r_act(x) ≈ x^{-1/4}
    /// - Heatbath: φ = (D†D)^{α/2} η       →  r_hb(x)  ≈ x^{+1/8}
    /// - Force:    dS_f/dU uses r_act
    ///
    /// Consistency: r_hb(x)² · r_act(x) = x^{1/4} · x^{-1/4} = 1.
    #[must_use]
    pub fn nf2(mass: f64, beta: f64) -> Self {
        let det_power = 0.25; // Nf/8 = 2/8
        let action_force = RationalApproximation::generate(-det_power, 8, 0.01, 64.0);
        Self {
            sectors: vec![RhmcFermionConfig {
                mass,
                det_power,
                action_approx: action_force.clone(),
                heatbath_approx: RationalApproximation::generate(det_power / 2.0, 8, 0.01, 64.0),
                force_approx: action_force,
            }],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
        }
    }

    /// Nf=1 configuration: one physical flavor from 4-taste staggered.
    ///
    /// det(D†D)^{1/8}. Useful for exploring the Nf dependence of the
    /// deconfinement transition and as the simplest dynamical benchmark.
    #[must_use]
    pub fn nf1(mass: f64, beta: f64) -> Self {
        let det_power = 0.125; // 1/8
        let af = RationalApproximation::generate(-det_power, 8, 0.01, 64.0);
        Self {
            sectors: vec![RhmcFermionConfig {
                mass,
                det_power,
                action_approx: af.clone(),
                heatbath_approx: RationalApproximation::generate(det_power / 2.0, 8, 0.01, 64.0),
                force_approx: af,
            }],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
        }
    }

    /// Nf=2+1 configuration: light (u,d) det^{1/4} + strange det^{1/8}.
    ///
    /// Light sector: α = 1/4 (two degenerate flavors from 4-taste staggered)
    /// Strange sector: α = 1/8 (one flavor from 4-taste staggered)
    #[must_use]
    pub fn nf2p1(light_mass: f64, strange_mass: f64, beta: f64) -> Self {
        let light_power = 0.25; // 2/8
        let strange_power = 0.125; // 1/8
        let light_af = RationalApproximation::generate(-light_power, 8, 0.01, 64.0);
        let strange_af = RationalApproximation::generate(-strange_power, 8, 0.01, 64.0);
        Self {
            sectors: vec![
                RhmcFermionConfig {
                    mass: light_mass,
                    det_power: light_power,
                    action_approx: light_af.clone(),
                    heatbath_approx: RationalApproximation::generate(
                        light_power / 2.0,
                        8,
                        0.01,
                        64.0,
                    ),
                    force_approx: light_af,
                },
                RhmcFermionConfig {
                    mass: strange_mass,
                    det_power: strange_power,
                    action_approx: strange_af.clone(),
                    heatbath_approx: RationalApproximation::generate(
                        strange_power / 2.0,
                        8,
                        0.01,
                        64.0,
                    ),
                    force_approx: strange_af,
                },
            ],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
        }
    }

    /// Nf=3 configuration: three degenerate flavors → det(D†D)^{3/8}.
    ///
    /// Single sector with α = 3/8. More poles needed for higher fractional
    /// powers to maintain rational approximation accuracy.
    #[must_use]
    pub fn nf3(mass: f64, beta: f64) -> Self {
        let det_power = 0.375; // 3/8
        let af = RationalApproximation::generate(-det_power, 12, 0.01, 64.0);
        Self {
            sectors: vec![RhmcFermionConfig {
                mass,
                det_power,
                action_approx: af.clone(),
                heatbath_approx: RationalApproximation::generate(det_power / 2.0, 12, 0.01, 64.0),
                force_approx: af,
            }],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
        }
    }

    /// Nf=4 configuration: four degenerate flavors → det(D†D)^{1/2}.
    ///
    /// α = 4/8 = 1/2. The simplest even case — no rooting needed for
    /// 4-taste staggered. Equivalent to HMC with one unrooted field.
    #[must_use]
    pub fn nf4(mass: f64, beta: f64) -> Self {
        let det_power = 0.5; // 4/8
        let af = RationalApproximation::generate(-det_power, 8, 0.01, 64.0);
        Self {
            sectors: vec![RhmcFermionConfig {
                mass,
                det_power,
                action_approx: af.clone(),
                heatbath_approx: RationalApproximation::generate(det_power / 2.0, 8, 0.01, 64.0),
                force_approx: af,
            }],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
        }
    }

    /// Nf=2+1+1 configuration: light (u,d) + strange + charm.
    ///
    /// Physical QCD: u,d at light mass, s at intermediate, c at heavy.
    /// Three fermion sectors with appropriate rooting powers.
    #[must_use]
    pub fn nf2p1p1(light_mass: f64, strange_mass: f64, charm_mass: f64, beta: f64) -> Self {
        let light_power = 0.25; // 2/8
        let single_power = 0.125; // 1/8
        let light_af = RationalApproximation::generate(-light_power, 8, 0.01, 64.0);
        let strange_af = RationalApproximation::generate(-single_power, 8, 0.01, 64.0);
        let charm_af = RationalApproximation::generate(-single_power, 8, 0.01, 64.0);
        Self {
            sectors: vec![
                RhmcFermionConfig {
                    mass: light_mass,
                    det_power: light_power,
                    action_approx: light_af.clone(),
                    heatbath_approx: RationalApproximation::generate(
                        light_power / 2.0,
                        8,
                        0.01,
                        64.0,
                    ),
                    force_approx: light_af,
                },
                RhmcFermionConfig {
                    mass: strange_mass,
                    det_power: single_power,
                    action_approx: strange_af.clone(),
                    heatbath_approx: RationalApproximation::generate(
                        single_power / 2.0,
                        8,
                        0.01,
                        64.0,
                    ),
                    force_approx: strange_af,
                },
                RhmcFermionConfig {
                    mass: charm_mass,
                    det_power: single_power,
                    action_approx: charm_af.clone(),
                    heatbath_approx: RationalApproximation::generate(
                        single_power / 2.0,
                        8,
                        0.01,
                        64.0,
                    ),
                    force_approx: charm_af,
                },
            ],
            beta,
            dt: 0.008,
            n_md_steps: 60,
            cg_tol: 1e-8,
            cg_max_iter: DYNAMICAL_CG_MAX_ITER,
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

    // φ = r(-p/2)(D†D) η = α₀η + Σ αᵢ (D†D + σᵢ)⁻¹ η
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
        let exact_val = 1.0_f64; // x^{1/4} at x=1 = 1
        assert!(
            (approx_val - exact_val).abs() < 0.01,
            "fourth_root(1.0) = {approx_val}, expected ~1.0"
        );

        let x = 16.0;
        let approx_val = r.eval(x);
        let exact_val = 2.0; // 16^{1/4} = 2
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
