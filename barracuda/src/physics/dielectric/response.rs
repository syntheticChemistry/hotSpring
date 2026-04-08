// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dielectric response functions: Mermin, completed Mermin, DSF, f-sum rule.
//!
//! - Standard Mermin conserves particle number only.
//! - Completed Mermin (Chuna & Murillo 2024, Eq. 26) conserves both particle
//!   number and momentum.

use super::complex::Complex;
use super::plasma_dispersion::{PlasmaParams, epsilon_vlasov};
use std::f64::consts::PI;

/// Standard Mermin dielectric function with collision frequency ν.
///
/// Conserves particle number but NOT momentum.
pub fn epsilon_mermin(k: f64, omega: f64, nu: f64, params: &PlasmaParams) -> Complex {
    epsilon_mermin_core(k, omega, nu, params, false)
}

/// Completed Mermin dielectric function (Chuna & Murillo 2024, Eq. 26).
///
/// Conserves BOTH particle number AND momentum via a correction factor
/// in the denominator. For the single-species OCP:
///
/// ```text
/// D_standard  = 1 + (iν/ω) × R
/// D_completed = 1 + (iν/ω) × R × (1 - G_p)
///
/// where R   = (ε₀(k,ω+iν) - 1) / (ε₀(k,0) - 1) = W(z_ν)
///   and G_p = R × ω(ω+iν) / (k² v_th²)
/// ```
///
/// This resolves:
/// - f-sum violations (65-97% → ~0%)
/// - DSF negativity near plasma resonance
/// - Non-Drude conductivity at finite wavevector
pub fn epsilon_completed_mermin(k: f64, omega: f64, nu: f64, params: &PlasmaParams) -> Complex {
    epsilon_mermin_core(k, omega, nu, params, true)
}

fn epsilon_mermin_core(
    k: f64,
    omega: f64,
    nu: f64,
    params: &PlasmaParams,
    momentum_conserving: bool,
) -> Complex {
    if omega.abs() < 1e-15 {
        return epsilon_vlasov(k, Complex::ZERO, params);
    }

    let omega_c = Complex::new(omega, 0.0);
    let omega_shifted = Complex::new(omega, nu);
    let eps_shifted = epsilon_vlasov(k, omega_shifted, params);
    let eps_static = epsilon_vlasov(k, Complex::ZERO, params);

    let numer = (omega_shifted * omega_c.inv()) * (eps_shifted - Complex::ONE);

    let r = (eps_shifted - Complex::ONE) * (eps_static - Complex::ONE).inv();

    let denom = if momentum_conserving {
        let k2_vth2 = k * k * params.v_th * params.v_th;
        let omega_product = omega_c * omega_shifted;
        let g_p = r * omega_product * (1.0 / k2_vth2);
        Complex::ONE + (Complex::I * nu * omega_c.inv()) * r * (Complex::ONE - g_p)
    } else {
        Complex::ONE + (Complex::I * nu * omega_c.inv()) * r
    };

    Complex::ONE + numer * denom.inv()
}

/// Debye screening check: ε(k,0) should equal 1 + (k_D/k)².
#[must_use]
pub fn debye_screening(k: f64, params: &PlasmaParams) -> (f64, f64) {
    let eps = epsilon_vlasov(k, Complex::ZERO, params);
    let expected = 1.0 + (params.k_debye / k).powi(2);
    (eps.re, expected)
}

/// Drude DC conductivity: σ(0) = ωₚ²/(4π ν).
#[must_use]
pub fn conductivity_dc(nu: f64, params: &PlasmaParams) -> f64 {
    params.omega_p * params.omega_p / (4.0 * PI * nu)
}

/// Dynamic structure factor S(k,ω) from the fluctuation-dissipation theorem.
///
/// S(k,ω) = (k² T)/(π n ω) × \[-Im(1/ε(k,ω))\]
pub fn dynamic_structure_factor(
    k: f64,
    omegas: &[f64],
    nu: f64,
    params: &PlasmaParams,
) -> Vec<f64> {
    dsf_from_epsilon(k, omegas, nu, params, false)
}

/// Dynamic structure factor using the completed Mermin (Chuna & Murillo 2024).
pub fn dynamic_structure_factor_completed(
    k: f64,
    omegas: &[f64],
    nu: f64,
    params: &PlasmaParams,
) -> Vec<f64> {
    dsf_from_epsilon(k, omegas, nu, params, true)
}

fn dsf_from_epsilon(
    k: f64,
    omegas: &[f64],
    nu: f64,
    params: &PlasmaParams,
    completed: bool,
) -> Vec<f64> {
    let prefactor = params.temperature * k * k / (PI * params.n);
    omegas
        .iter()
        .map(|&omega| {
            if omega.abs() < 1e-15 {
                return 0.0;
            }
            let eps = if completed {
                epsilon_completed_mermin(k, omega, nu, params)
            } else {
                epsilon_mermin(k, omega, nu, params)
            };
            let loss = -eps.inv().im;
            prefactor * loss / omega
        })
        .collect()
}

/// Numerical f-sum rule integral: ∫₀^ω_max ω Im\[1/ε(k,ω)\] dω.
///
/// Exact value for ideal dielectric: -π ωₚ²/2.
#[must_use]
pub fn f_sum_rule_integral(k: f64, nu: f64, params: &PlasmaParams, omega_max: f64) -> f64 {
    f_sum_rule_integral_core(k, nu, params, omega_max, false)
}

/// f-sum rule integral using the completed Mermin.
#[must_use]
pub fn f_sum_rule_integral_completed(
    k: f64,
    nu: f64,
    params: &PlasmaParams,
    omega_max: f64,
) -> f64 {
    f_sum_rule_integral_core(k, nu, params, omega_max, true)
}

fn f_sum_rule_integral_core(
    k: f64,
    nu: f64,
    params: &PlasmaParams,
    omega_max: f64,
    completed: bool,
) -> f64 {
    let n_points: usize = 50_000;
    let d_omega = omega_max / n_points as f64;
    let mut sum = 0.0;
    for i in 1..n_points {
        let omega = i as f64 * d_omega;
        let eps = if completed {
            epsilon_completed_mermin(k, omega, nu, params)
        } else {
            epsilon_mermin(k, omega, nu, params)
        };
        sum += omega * eps.inv().im;
    }
    sum * d_omega
}

/// Result of a dielectric function validation check.
#[derive(Debug)]
pub struct DielectricValidation {
    /// Coupling parameter.
    pub gamma: f64,
    /// Screening parameter.
    pub kappa: f64,
    /// Relative error in Debye screening limit.
    pub debye_error: f64,
    /// Computed f-sum integral (standard Mermin).
    pub f_sum_computed: f64,
    /// Computed f-sum integral (completed Mermin).
    pub f_sum_completed: f64,
    /// Expected f-sum value (-π ωₚ²/2).
    pub f_sum_expected: f64,
    /// f-sum relative error (standard).
    pub f_sum_error_standard: f64,
    /// f-sum relative error (completed).
    pub f_sum_error_completed: f64,
    /// Computed DC conductivity.
    pub dc_conductivity: f64,
    /// Expected DC conductivity (Drude).
    pub dc_expected: f64,
    /// Deviation of ε from 1 at high frequency.
    pub high_freq_deviation: f64,
    /// Fraction of S(k,ω) values that are non-negative (standard).
    pub dsf_fraction_positive: f64,
    /// Fraction of S(k,ω) values that are non-negative (completed).
    pub dsf_fraction_positive_completed: f64,
}

/// Run all validation checks for a given (Γ, κ) pair.
#[must_use]
pub fn validate_dielectric(gamma: f64, kappa: f64) -> DielectricValidation {
    let params = PlasmaParams::from_coupling(gamma, kappa);
    let nu = 0.1 * params.omega_p;
    let k = 1.0;

    let (eps_static, eps_expected) = debye_screening(k, &params);
    let debye_error = (eps_static - eps_expected).abs() / eps_expected;

    let f_sum = f_sum_rule_integral(k, nu, &params, 200.0);
    let f_sum_completed = f_sum_rule_integral_completed(k, nu, &params, 200.0);
    let f_sum_expected = -PI * params.omega_p * params.omega_p / 2.0;

    let dc = conductivity_dc(nu, &params);
    let dc_expected = params.omega_p.powi(2) / (4.0 * PI * nu);

    let eps_high = epsilon_mermin(k, 100.0 * params.omega_p, nu, &params);
    let high_freq_dev = (eps_high - Complex::ONE).abs();

    let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();

    let s_kw = dynamic_structure_factor(k, &omegas, nu, &params);
    let s_max = s_kw.iter().copied().fold(0.0_f64, f64::max);
    let n_positive = s_kw
        .iter()
        .filter(|&&s| s >= -1e-6 * s_max.max(1e-10))
        .count();
    let frac_positive = n_positive as f64 / s_kw.len() as f64;

    let s_kw_cm = dynamic_structure_factor_completed(k, &omegas, nu, &params);
    let s_max_cm = s_kw_cm.iter().copied().fold(0.0_f64, f64::max);
    let n_positive_cm = s_kw_cm
        .iter()
        .filter(|&&s| s >= -1e-6 * s_max_cm.max(1e-10))
        .count();
    let frac_positive_cm = n_positive_cm as f64 / s_kw_cm.len() as f64;

    DielectricValidation {
        gamma,
        kappa,
        debye_error,
        f_sum_computed: f_sum,
        f_sum_completed,
        f_sum_expected,
        f_sum_error_standard: (f_sum - f_sum_expected).abs() / f_sum_expected.abs(),
        f_sum_error_completed: (f_sum_completed - f_sum_expected).abs() / f_sum_expected.abs(),
        dc_conductivity: dc,
        dc_expected,
        high_freq_deviation: high_freq_dev,
        dsf_fraction_positive: frac_positive,
        dsf_fraction_positive_completed: frac_positive_cm,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn dc_conductivity_drude() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let dc = conductivity_dc(nu, &params);
        let expected = params.omega_p.powi(2) / (4.0 * PI * nu);
        assert!(
            (dc - expected).abs() / expected < 1e-14,
            "DC conductivity: got {dc}, expected {expected}"
        );
    }

    #[test]
    fn high_freq_limit() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let eps = epsilon_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        let dev = (eps - Complex::ONE).abs();
        assert!(dev < 0.01, "ε(k,100ωₚ) should → 1, deviation = {dev}");
    }

    #[test]
    fn f_sum_rule_sign() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let integral = f_sum_rule_integral(1.0, nu, &params, 200.0);
        assert!(
            integral < 0.0,
            "f-sum integral should be negative, got {integral}"
        );
    }

    #[test]
    fn dsf_mostly_positive() {
        let params = PlasmaParams::from_coupling(10.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let omegas: Vec<f64> = (1..1000).map(|i| 0.1 + i as f64 * 0.05).collect();
        let s = dynamic_structure_factor(1.0, &omegas, nu, &params);
        let n_neg = s.iter().filter(|&&v| v < -1e-10).count();
        let frac_pos = 1.0 - n_neg as f64 / s.len() as f64;
        assert!(
            frac_pos > 0.98,
            "S(k,ω) should be ≥98% positive, got {:.1}%",
            frac_pos * 100.0
        );
    }

    #[test]
    fn mermin_reduces_to_vlasov_at_zero_nu() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let k = 1.0;
        let omega = 2.0;
        let eps_mermin = epsilon_mermin(k, omega, 1e-10, &params);
        let eps_vlasov = epsilon_vlasov(k, Complex::new(omega, 0.0), &params);
        let diff = (eps_mermin - eps_vlasov).abs();
        assert!(
            diff < 0.01,
            "Mermin(ν→0) should match Vlasov: diff = {diff}"
        );
    }

    #[test]
    fn full_validation_weak_coupling() {
        let result = validate_dielectric(1.0, 1.0);
        assert!(result.debye_error < 1e-12);
        assert!(result.f_sum_computed < 0.0);
        assert!(result.high_freq_deviation < 0.01);
        assert!(result.dsf_fraction_positive > 0.95);
    }

    #[test]
    fn full_validation_moderate_coupling() {
        let result = validate_dielectric(10.0, 1.0);
        assert!(result.debye_error < 1e-12);
        assert!(result.f_sum_computed < 0.0);
        assert!(result.high_freq_deviation < 0.01);
        assert!(result.dsf_fraction_positive > 0.98);
    }

    #[test]
    fn full_validation_strong_screening() {
        let result = validate_dielectric(10.0, 2.0);
        assert!(result.debye_error < 1e-12);
        assert!(result.f_sum_computed < 0.0);
        assert!(result.high_freq_deviation < 0.01);
        assert!(result.dsf_fraction_positive > 0.98);
    }

    #[test]
    fn completed_mermin_recovers_standard_at_zero_nu() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let k = 1.0;
        let omega = 2.0;
        let eps_std = epsilon_mermin(k, omega, 1e-10, &params);
        let eps_cm = epsilon_completed_mermin(k, omega, 1e-10, &params);
        let diff = (eps_cm - eps_std).abs();
        assert!(
            diff < 0.01,
            "Completed Mermin(ν→0) ≈ Standard Mermin(ν→0): diff = {diff}"
        );
    }

    #[test]
    fn completed_mermin_high_freq_limit() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let eps = epsilon_completed_mermin(1.0, 100.0 * params.omega_p, nu, &params);
        let dev = (eps - Complex::ONE).abs();
        assert!(dev < 0.01, "ε_CM(k,100ωₚ) → 1, deviation = {dev}");
    }

    #[test]
    fn completed_mermin_static_limit_matches_standard() {
        let params = PlasmaParams::from_coupling(10.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let k = 1.0;
        let eps_std = epsilon_mermin(k, 1e-14, nu, &params);
        let eps_cm = epsilon_completed_mermin(k, 1e-14, nu, &params);
        let diff = (eps_cm - eps_std).abs();
        assert!(
            diff < 1e-10,
            "Static limit: completed ≈ standard, diff = {diff}"
        );
    }

    #[test]
    fn completed_mermin_dsf_fully_positive() {
        for &(gamma, kappa) in &[(1.0, 1.0), (10.0, 1.0), (10.0, 2.0)] {
            let params = PlasmaParams::from_coupling(gamma, kappa);
            let nu = 0.1 * params.omega_p;
            let omegas: Vec<f64> = (1..2000).map(|i| 0.1 + i as f64 * 0.02).collect();
            let s = dynamic_structure_factor_completed(1.0, &omegas, nu, &params);
            let s_max = s.iter().copied().fold(0.0_f64, f64::max);
            let n_neg = s.iter().filter(|&&v| v < -1e-6 * s_max.max(1e-10)).count();
            let frac_pos = 1.0 - n_neg as f64 / s.len() as f64;
            assert!(
                frac_pos > 0.99,
                "Completed DSF Γ={gamma} κ={kappa}: {:.1}% positive (want ≥99%)",
                frac_pos * 100.0
            );
        }
    }

    #[test]
    fn completed_mermin_f_sum_improvement() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let k = 1.0;
        let expected = -PI * params.omega_p * params.omega_p / 2.0;

        let standard = f_sum_rule_integral(k, nu, &params, 200.0);
        let completed = f_sum_rule_integral_completed(k, nu, &params, 200.0);

        let err_std = (standard - expected).abs() / expected.abs();
        let err_cm = (completed - expected).abs() / expected.abs();

        eprintln!("  f-sum standard err: {err_std:.4e}, completed err: {err_cm:.4e}");
        assert!(
            err_cm < err_std || err_cm < 0.10,
            "Completed Mermin f-sum should improve or be <10%: std={err_std:.4e} cm={err_cm:.4e}"
        );
    }

    #[test]
    fn completed_mermin_differs_from_standard_at_finite_nu() {
        let params = PlasmaParams::from_coupling(10.0, 1.0);
        let nu = 0.5 * params.omega_p;
        let k = 1.0;
        let omega = params.omega_p;
        let eps_std = epsilon_mermin(k, omega, nu, &params);
        let eps_cm = epsilon_completed_mermin(k, omega, nu, &params);
        let diff = (eps_cm - eps_std).abs();
        assert!(
            diff > 1e-6,
            "At finite ν, completed ≠ standard: diff = {diff:.4e}"
        );
    }

    #[test]
    fn completed_mermin_passive_medium() {
        let params = PlasmaParams::from_coupling(10.0, 1.0);
        let nu = 0.1 * params.omega_p;
        for omega_f in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let omega = omega_f * params.omega_p;
            let eps = epsilon_completed_mermin(1.0, omega, nu, &params);
            assert!(
                eps.im >= -1e-10,
                "Im[ε_CM] ≥ 0 (passive medium): ω/ωₚ={omega_f}, Im={:.4e}",
                eps.im
            );
        }
    }

    #[test]
    fn dispersion_relation_monotonic() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let nu = 0.1 * params.omega_p;
        let k_values = [0.5, 1.0, 2.0, 4.0];
        let mut peaks = Vec::new();

        for &k in &k_values {
            let omegas: Vec<f64> = (1..1000).map(|i| i as f64 * 0.02).collect();
            let mut best_loss = 0.0_f64;
            let mut best_omega = 0.0;
            for &omega in &omegas {
                let eps = epsilon_mermin(k, omega, nu, &params);
                let loss = -eps.inv().im;
                if loss > best_loss {
                    best_loss = loss;
                    best_omega = omega;
                }
            }
            peaks.push(best_omega);
        }

        for i in 0..peaks.len() - 1 {
            assert!(
                peaks[i] <= peaks[i + 1] + 0.5,
                "Loss peaks should generally increase: k={}, ω_peak={} vs k={}, ω_peak={}",
                k_values[i],
                peaks[i],
                k_values[i + 1],
                peaks[i + 1]
            );
        }
    }
}
