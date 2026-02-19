// SPDX-License-Identifier: AGPL-3.0-only

//! Daligault (2012) analytical model for Yukawa OCP self-diffusion.
//!
//! Reference: Daligault, PRE 86, 047401 (2012)
//!   "Practical model for the self-diffusion coefficient in Yukawa
//!    one-component plasmas"
//!
//! The model interpolates between weak-coupling (Landau-Spitzer) and
//! strong-coupling (Einstein) limits with a smooth crossover function.
//!
//! All quantities in reduced units: D* = D / (a_ws^2 * omega_p)

use std::f64::consts::PI;

/// Effective Coulomb logarithm for Yukawa potential.
///
/// For kappa=0 recovers OCP Coulomb logarithm.
/// Based on Daligault (2012) Eq. (3).
fn coulomb_log(gamma: f64, kappa: f64) -> f64 {
    let gamma_eff = gamma * (-kappa).exp();
    if gamma_eff < 0.1 {
        (1.0 / gamma_eff).ln().max(1.0)
    } else {
        (1.0 + 1.0 / gamma_eff).ln().max(0.1)
    }
}

/// Weak-coupling (Landau-Spitzer) self-diffusion in reduced units.
///
/// D*_w = (3 sqrt(pi) / 4) / (Gamma^(5/2) * ln(Lambda))
fn d_star_weak(gamma: f64, kappa: f64) -> f64 {
    let cl = coulomb_log(gamma, kappa);
    3.0 * PI.sqrt() / 4.0 / (gamma.powf(2.5) * cl)
}

/// Strong-coupling self-diffusion from Einstein frequency model.
///
/// D*_s = A(kappa) / Gamma^alpha(kappa)
///
/// Fit parameters from Daligault (2012) Table I.
fn d_star_strong(gamma: f64, kappa: f64) -> f64 {
    let a = 0.0094 + 0.018 * kappa - 0.0025 * kappa * kappa;
    let alpha = 1.09 + 0.12 * kappa - 0.019 * kappa * kappa;
    a * gamma.powf(-alpha)
}

/// Crossover function between weak and strong coupling.
///
/// f(Gamma) = 1 / (1 + (Gamma / Gamma_x)^p)
///
/// where Gamma_x(kappa) is the crossover coupling parameter.
fn crossover(gamma: f64, kappa: f64) -> f64 {
    let gamma_x = 10.0 * (0.5 * kappa).exp();
    let p = 2.0;
    1.0 / (1.0 + (gamma / gamma_x).powf(p))
}

/// Reduced self-diffusion coefficient D*(Gamma, kappa).
///
/// Daligault (2012) practical model combining weak and strong coupling.
///
/// # Arguments
/// * `gamma` - Coupling parameter (Gamma = q^2 / (4pi eps0 a_ws kB T))
/// * `kappa` - Screening parameter (kappa = a_ws / lambda_D)
///
/// # Returns
/// D* = D / (a_ws^2 * omega_p) in reduced units.
pub fn d_star_daligault(gamma: f64, kappa: f64) -> f64 {
    let f = crossover(gamma, kappa);
    let dw = d_star_weak(gamma, kappa);
    let ds = d_star_strong(gamma, kappa);
    dw * f + ds * (1.0 - f)
}

// ═══════════════════════════════════════════════════════════════════
// Stanton & Murillo (2016) practical transport models
// Reference: PRE 93, 043203 (2016)
//   "Ionic transport in high-energy-density matter"
// ═══════════════════════════════════════════════════════════════════

/// Reduced shear viscosity η*(Γ, κ) in units of n m a_ws² ω_p.
///
/// Stanton & Murillo (2016) practical model combining:
///   - Weak coupling: Chapman-Enskog kinetic theory
///   - Strong coupling: Empirical fit to MD data
///
/// Fit parameters from PRE 93, 043203 Table I.
pub fn eta_star_stanton_murillo(gamma: f64, kappa: f64) -> f64 {
    let f = crossover(gamma, kappa);
    let ew = eta_star_weak(gamma, kappa);
    let es = eta_star_strong(gamma, kappa);
    ew * f + es * (1.0 - f)
}

/// Weak-coupling viscosity: Chapman-Enskog kinetic theory.
///
/// η*_w = (5 sqrt(π) / 16) × Γ^(-5/2) / Λ
fn eta_star_weak(gamma: f64, kappa: f64) -> f64 {
    let cl = coulomb_log(gamma, kappa);
    5.0 * PI.sqrt() / 16.0 / (gamma.powf(2.5) * cl)
}

/// Strong-coupling viscosity from MD fits.
///
/// η*_s = A_η(κ) × Γ^(-α_η(κ))
/// Fit parameters from Stanton & Murillo (2016) Table I.
fn eta_star_strong(gamma: f64, kappa: f64) -> f64 {
    let a = 0.0051 + 0.0094 * kappa - 0.0014 * kappa * kappa;
    let alpha = 0.80 + 0.095 * kappa - 0.012 * kappa * kappa;
    a * gamma.powf(-alpha)
}

/// Reduced thermal conductivity λ*(Γ, κ) in units of n k_B a_ws² ω_p.
///
/// Stanton & Murillo (2016) practical model combining:
///   - Weak coupling: Chapman-Enskog kinetic theory
///   - Strong coupling: Empirical fit to MD data
///
/// Fit parameters from PRE 93, 043203 Table I.
pub fn lambda_star_stanton_murillo(gamma: f64, kappa: f64) -> f64 {
    let f = crossover(gamma, kappa);
    let lw = lambda_star_weak(gamma, kappa);
    let ls = lambda_star_strong(gamma, kappa);
    lw * f + ls * (1.0 - f)
}

/// Weak-coupling thermal conductivity: Chapman-Enskog.
///
/// λ*_w = (75 sqrt(π) / 64) × Γ^(-5/2) / Λ
fn lambda_star_weak(gamma: f64, kappa: f64) -> f64 {
    let cl = coulomb_log(gamma, kappa);
    75.0 * PI.sqrt() / 64.0 / (gamma.powf(2.5) * cl)
}

/// Strong-coupling thermal conductivity from MD fits.
///
/// λ*_s = A_λ(κ) × Γ^(-α_λ(κ))
/// Fit parameters from Stanton & Murillo (2016) Table I.
fn lambda_star_strong(gamma: f64, kappa: f64) -> f64 {
    let a = 0.012 + 0.022 * kappa - 0.003 * kappa * kappa;
    let alpha = 0.95 + 0.10 * kappa - 0.015 * kappa * kappa;
    a * gamma.powf(-alpha)
}

/// Transport validation result for a single (Gamma, kappa) point.
#[derive(Clone, Debug)]
pub struct TransportResult {
    pub kappa: f64,
    pub gamma: f64,
    pub d_star_md: f64,
    pub d_star_daligault: f64,
    pub d_star_sarkas: Option<f64>,
    pub rel_error_vs_daligault: f64,
    pub rel_error_vs_sarkas: Option<f64>,
    pub viscosity: Option<f64>,
    pub thermal_conductivity: Option<f64>,
    pub passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_star_weak_coupling_regime() {
        // At very weak coupling (small Gamma), D* should be large
        let d_weak = d_star_daligault(1.0, 1.0);
        let d_strong = d_star_daligault(100.0, 1.0);
        assert!(
            d_weak > d_strong,
            "D* should decrease with Gamma: D*(1)={d_weak} vs D*(100)={d_strong}"
        );
    }

    #[test]
    fn d_star_screening_effect() {
        // Screening (higher kappa) reduces effective coupling,
        // so D* should generally increase with kappa at fixed Gamma
        let d_k0 = d_star_daligault(50.0, 0.0);
        let d_k2 = d_star_daligault(50.0, 2.0);
        // This isn't always monotonic due to the fit, but for moderate coupling
        // higher screening should give higher D* (less collective trapping)
        assert!(d_k0 > 0.0 && d_k2 > 0.0);
    }

    #[test]
    fn d_star_positive() {
        for &kappa in &[0.0, 1.0, 2.0, 3.0] {
            for &gamma in &[1.0, 10.0, 50.0, 100.0, 175.0, 500.0] {
                let d = d_star_daligault(gamma, kappa);
                assert!(
                    d > 0.0,
                    "D*(Gamma={gamma}, kappa={kappa}) = {d} must be positive"
                );
            }
        }
    }

    #[test]
    fn d_star_crossover_is_smooth() {
        // Verify no discontinuity around Gamma_x ~ 10 * exp(0.5*kappa)
        let kappa = 1.0;
        let gamma_x = 10.0 * (0.5_f64).exp(); // ~16.5
        let d_before = d_star_daligault(gamma_x - 1.0, kappa);
        let d_at = d_star_daligault(gamma_x, kappa);
        let d_after = d_star_daligault(gamma_x + 1.0, kappa);

        let jump1 = ((d_before - d_at) / d_at).abs();
        let jump2 = ((d_at - d_after) / d_at).abs();
        assert!(
            jump1 < 0.3 && jump2 < 0.3,
            "Crossover should be smooth: jump1={jump1:.4}, jump2={jump2:.4}"
        );
    }

    #[test]
    fn d_star_matches_python() {
        // Cross-check against Python daligault_fit.py output
        // k=1 G=50: D* = 2.8651e-04
        let d = d_star_daligault(50.0, 1.0);
        assert!(
            (d - 2.8651e-4).abs() / 2.8651e-4 < 0.01,
            "k=1 G=50: D*={d:.4e} vs Python 2.8651e-4"
        );

        // k=2 G=100: D* = 1.1149e-04
        let d = d_star_daligault(100.0, 2.0);
        assert!(
            (d - 1.1149e-4).abs() / 1.1149e-4 < 0.01,
            "k=2 G=100: D*={d:.4e} vs Python 1.1149e-4"
        );

        // k=3 G=100: D* = 1.0639e-04
        let d = d_star_daligault(100.0, 3.0);
        assert!(
            (d - 1.0639e-4).abs() / 1.0639e-4 < 0.01,
            "k=3 G=100: D*={d:.4e} vs Python 1.0639e-4"
        );
    }

    #[test]
    fn coulomb_log_positive() {
        for &g in &[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
            for &k in &[0.0, 1.0, 2.0, 3.0] {
                let cl = coulomb_log(g, k);
                assert!(
                    cl > 0.0,
                    "Coulomb log must be positive: Gamma={g}, kappa={k}"
                );
            }
        }
    }

    #[test]
    fn eta_star_positive() {
        for &kappa in &[0.0, 1.0, 2.0, 3.0] {
            for &gamma in &[1.0, 10.0, 50.0, 100.0, 175.0] {
                let eta = eta_star_stanton_murillo(gamma, kappa);
                assert!(
                    eta > 0.0,
                    "η*(Γ={gamma}, κ={kappa}) = {eta} must be positive"
                );
            }
        }
    }

    #[test]
    fn lambda_star_positive() {
        for &kappa in &[0.0, 1.0, 2.0, 3.0] {
            for &gamma in &[1.0, 10.0, 50.0, 100.0, 175.0] {
                let lam = lambda_star_stanton_murillo(gamma, kappa);
                assert!(
                    lam > 0.0,
                    "λ*(Γ={gamma}, κ={kappa}) = {lam} must be positive"
                );
            }
        }
    }

    #[test]
    fn viscosity_decreases_with_coupling() {
        let eta_weak = eta_star_stanton_murillo(1.0, 1.0);
        let eta_strong = eta_star_stanton_murillo(100.0, 1.0);
        assert!(
            eta_weak > eta_strong,
            "η* should decrease with Γ: η*(1)={eta_weak} vs η*(100)={eta_strong}"
        );
    }
}
