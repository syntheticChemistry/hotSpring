// SPDX-License-Identifier: AGPL-3.0-only

//! Conservative dielectric functions from the BGK equation.
//!
//! Implements the Mermin dielectric function and related plasma response
//! functions for classical one-component plasmas (OCP). This module
//! provides the theoretical foundation for dynamic structure factor S(k,ω)
//! computation, complementing the MD-derived DSF from Papers 1/5.
//!
//! # References
//!
//! - Mermin, Phys. Rev. B 1, 2362 (1970)
//! - Chuna & Murillo, Phys. Rev. E 111, 035206 (2024), arXiv:2405.07871
//! - Stanton & Murillo, Phys. Rev. E 91, 033104 (2015)

use std::f64::consts::PI;

/// Complex number type for plasma response calculations.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Additive identity.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    /// Multiplicative identity.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    /// Imaginary unit.
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    /// Construct from real and imaginary parts.
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Squared modulus |z|².
    #[must_use]
    pub fn abs_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Modulus |z|.
    #[must_use]
    pub fn abs(self) -> f64 {
        self.abs_sq().sqrt()
    }

    /// Complex conjugate z*.
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Multiplicative inverse 1/z.
    #[must_use]
    pub fn inv(self) -> Self {
        let d = self.abs_sq();
        Self::new(self.re / d, -self.im / d)
    }

    /// Complex exponential exp(z).
    #[must_use]
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        Self::new(e * self.im.cos(), e * self.im.sin())
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl std::ops::Div for Complex {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inv()
    }
}

/// Plasma parameters for an OCP in natural units (a=1, ωₚ=1).
#[derive(Debug, Clone)]
pub struct PlasmaParams {
    /// Wigner-Seitz radius.
    pub a: f64,
    /// Plasma frequency ωₚ = √(4π n q²/m).
    pub omega_p: f64,
    /// Temperature T = q²/(a Γ).
    pub temperature: f64,
    /// Thermal velocity v_th = √(T/m).
    pub v_th: f64,
    /// Debye wave number k_D = κ/a.
    pub k_debye: f64,
    /// Number density.
    pub n: f64,
    /// Coupling parameter Γ.
    pub gamma: f64,
    /// Screening parameter κ.
    pub kappa: f64,
}

impl PlasmaParams {
    /// Construct plasma parameters from (Γ, κ) in natural units.
    #[must_use]
    pub fn from_coupling(gamma: f64, kappa: f64) -> Self {
        let n = 1.0;
        let m = 1.0;
        let q = 1.0;
        let a = (3.0 / (4.0 * PI * n)).cbrt();
        let q2 = q * q;
        let omega_p = (4.0 * PI * n * q2 / m).sqrt();
        #[allow(clippy::suspicious_operation_groupings)]
        let temperature = q2 / (a * gamma);
        let v_th = (temperature / m).sqrt();
        let k_debye = kappa / a;
        Self {
            a,
            omega_p,
            temperature,
            v_th,
            k_debye,
            n,
            gamma,
            kappa,
        }
    }
}

/// Plasma dispersion function Z(z) for a Maxwellian distribution.
///
/// Z(z) = (1/√π) P.V. ∫_{-∞}^{∞} exp(-t²)/(t-z) dt
///
/// Uses power series for |z| < 6 and asymptotic expansion for |z| ≥ 6.
/// Analytic continuation from Im(z) > 0 (Landau prescription).
#[must_use]
pub fn plasma_dispersion_z(z: Complex) -> Complex {
    let z_abs = z.abs();

    if z_abs < 6.0 {
        // Power series: Z(z) = i√π exp(-z²) - 2z Σ c_n
        // c_0 = 1, c_{n+1} = c_n × (-2z²)/(2n+3)
        let z2 = z * z;
        let neg2z2 = z2 * (-2.0);
        let mut term = Complex::ONE;
        let mut total = term;
        for n in 1..100 {
            term = term * neg2z2 * (1.0 / (2 * n + 1) as f64);
            total = total + term;
            if term.abs() < 1e-16 * (total.abs() + 1e-30) {
                break;
            }
        }
        let exp_neg_z2 = (z2 * (-1.0)).exp();
        let imaginary_part = Complex::I * (PI.sqrt()) * exp_neg_z2;
        imaginary_part - z * total * 2.0
    } else {
        // Asymptotic: Z(z) ≈ iσ√π exp(-z²) - (1/z)(1 + 1/(2z²) + 3/(4z⁴) + ...)
        let sigma = if z.im >= 0.0 { 1.0 } else { 2.0 };

        let z2 = z * z;
        let inv_2z2 = (z2 * 2.0).inv();
        let mut term = Complex::ONE;
        let mut total = term;
        for n in 0..30 {
            term = term * inv_2z2 * ((2 * n + 1) as f64);
            total = total + term;
            if term.abs() < 1e-15 * (total.abs() + 1e-30) {
                break;
            }
        }
        let exp_neg_z2 = (z2 * (-1.0)).exp();
        Complex::I * sigma * PI.sqrt() * exp_neg_z2 - total * z.inv()
    }
}

/// W(z) = 1 + z Z(z): the quantity appearing in the Vlasov susceptibility.
///
/// Uses the naive formula for |z| < 4 (cancellation < 15×, safe for f64).
/// For |z| ≥ 4, uses [`plasma_dispersion_w_stable`] which computes W directly.
#[must_use]
pub fn plasma_dispersion_w(z: Complex) -> Complex {
    if z.abs() < 4.0 {
        Complex::ONE + z * plasma_dispersion_z(z)
    } else {
        plasma_dispersion_w_stable(z)
    }
}

/// Numerically stable W(z) via direct asymptotic expansion.
///
/// Avoids catastrophic cancellation in 1 + z·Z(z) by computing W directly:
///   W(z) = -1/(2z²) × (1 + 3/(2z²) + 15/(4z⁴) + ...) + i·z·σ·√π·exp(-z²)
///
/// Stable across all FP precisions (f32, DF64, f64) because no near-cancellation
/// of large values occurs. See `wateringHole/GPU_F64_NUMERICAL_STABILITY.md`.
#[must_use]
pub fn plasma_dispersion_w_stable(z: Complex) -> Complex {
    let z2 = z * z;
    let inv_2z2 = (z2 * 2.0).inv();

    let mut coeff = Complex::ONE;
    let mut total = coeff;
    for n in 0..30 {
        coeff = coeff * inv_2z2 * ((2 * n + 3) as f64);
        total = total + coeff;
        if coeff.abs() < 1e-15 * (total.abs() + 1e-30) {
            break;
        }
    }
    let w_asymp = inv_2z2 * total * (-1.0);

    let exp_neg_z2 = (z2 * (-1.0)).exp();
    let sigma = if z.im >= 0.0 { 1.0 } else { 2.0 };
    let w_exp = Complex::I * z * PI.sqrt() * sigma * exp_neg_z2;

    w_asymp + w_exp
}

/// Free-particle (Vlasov) susceptibility χ₀(k,ω).
///
/// χ₀(k,ω) = -(k_D²/k²) W(ω/(√2 k v_th))
#[must_use]
pub fn chi0_classical(k: f64, omega: Complex, params: &PlasmaParams) -> Complex {
    let z = omega * (1.0 / (std::f64::consts::SQRT_2 * k * params.v_th));
    let w = plasma_dispersion_w(z);
    w * (-(params.k_debye * params.k_debye) / (k * k))
}

/// Vlasov (collisionless) dielectric function.
#[must_use]
pub fn epsilon_vlasov(k: f64, omega: Complex, params: &PlasmaParams) -> Complex {
    Complex::ONE - chi0_classical(k, omega, params)
}

/// Standard Mermin dielectric function with collision frequency ν.
///
/// Conserves particle number but NOT momentum.
#[must_use]
pub fn epsilon_mermin(k: f64, omega: f64, nu: f64, params: &PlasmaParams) -> Complex {
    if omega.abs() < 1e-15 {
        return epsilon_vlasov(k, Complex::ZERO, params);
    }

    let omega_c = Complex::new(omega, 0.0);
    let omega_shifted = Complex::new(omega, nu);
    let eps_shifted = epsilon_vlasov(k, omega_shifted, params);
    let eps_static = epsilon_vlasov(k, Complex::ZERO, params);

    let ratio = (omega_shifted * omega_c.inv()) * (eps_shifted - Complex::ONE);
    let denom = Complex::ONE
        + (Complex::I * nu * omega_c.inv())
            * (eps_shifted - Complex::ONE)
            * (eps_static - Complex::ONE).inv();

    Complex::ONE + ratio * denom.inv()
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
/// S(k,ω) = (k² T)/(π n ω) × [-Im(1/ε(k,ω))]
pub fn dynamic_structure_factor(
    k: f64,
    omegas: &[f64],
    nu: f64,
    params: &PlasmaParams,
) -> Vec<f64> {
    let prefactor = params.temperature * k * k / (PI * params.n);
    omegas
        .iter()
        .map(|&omega| {
            if omega.abs() < 1e-15 {
                return 0.0;
            }
            let eps = epsilon_mermin(k, omega, nu, params);
            let loss = -eps.inv().im;
            prefactor * loss / omega
        })
        .collect()
}

/// Numerical f-sum rule integral: ∫₀^ω_max ω Im[1/ε(k,ω)] dω.
///
/// Exact value for ideal dielectric: -π ωₚ²/2.
#[must_use]
pub fn f_sum_rule_integral(k: f64, nu: f64, params: &PlasmaParams, omega_max: f64) -> f64 {
    let n_points: usize = 50_000;
    let d_omega = omega_max / n_points as f64;
    let mut sum = 0.0;
    for i in 1..n_points {
        let omega = i as f64 * d_omega;
        let eps = epsilon_mermin(k, omega, nu, params);
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
    /// Computed f-sum integral.
    pub f_sum_computed: f64,
    /// Expected f-sum value (-π ωₚ²/2).
    pub f_sum_expected: f64,
    /// Computed DC conductivity.
    pub dc_conductivity: f64,
    /// Expected DC conductivity (Drude).
    pub dc_expected: f64,
    /// Deviation of ε from 1 at high frequency.
    pub high_freq_deviation: f64,
    /// Fraction of S(k,ω) values that are non-negative.
    pub dsf_fraction_positive: f64,
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

    DielectricValidation {
        gamma,
        kappa,
        debye_error,
        f_sum_computed: f_sum,
        f_sum_expected,
        dc_conductivity: dc,
        dc_expected,
        high_freq_deviation: high_freq_dev,
        dsf_fraction_positive: frac_positive,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn z_function_at_zero() {
        let z = plasma_dispersion_z(Complex::ZERO);
        assert!((z.re).abs() < 1e-14, "Re[Z(0)] should be 0, got {}", z.re);
        assert!(
            (z.im - PI.sqrt()).abs() < 1e-14,
            "Im[Z(0)] should be √π, got {}",
            z.im
        );
    }

    #[test]
    fn w_function_at_zero() {
        let w = plasma_dispersion_w(Complex::ZERO);
        assert!((w.re - 1.0).abs() < 1e-14, "W(0) = 1, got {}", w.re);
        assert!(w.im.abs() < 1e-14, "Im[W(0)] = 0, got {}", w.im);
    }

    #[test]
    fn w_function_large_z_vanishes() {
        let z = Complex::new(20.0, 0.0);
        let w = plasma_dispersion_w(z);
        assert!(w.abs() < 0.01, "W(20) should → 0, got {}", w.abs());
    }

    #[test]
    fn debye_screening_exact() {
        for &(gamma, kappa) in &[(1.0, 1.0), (10.0, 1.0), (10.0, 2.0)] {
            let params = PlasmaParams::from_coupling(gamma, kappa);
            let (computed, expected) = debye_screening(1.0, &params);
            let err = (computed - expected).abs() / expected;
            assert!(
                err < 1e-12,
                "Debye screening Γ={gamma}, κ={kappa}: err={err}"
            );
        }
    }

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

    /// Simulate f32 complex arithmetic for W(z) to test cancellation stability.
    pub(super) mod f32_stability {
        #[derive(Clone, Copy)]
        pub(super) struct C32 {
            pub re: f32,
            pub im: f32,
        }
        impl C32 {
            pub fn new(re: f32, im: f32) -> Self {
                Self { re, im }
            }
            fn abs(self) -> f32 {
                (self.re * self.re + self.im * self.im).sqrt()
            }
            fn mul(self, b: Self) -> Self {
                Self::new(
                    self.re * b.re - self.im * b.im,
                    self.re * b.im + self.im * b.re,
                )
            }
            fn add(self, b: Self) -> Self {
                Self::new(self.re + b.re, self.im + b.im)
            }
            fn scale(self, s: f32) -> Self {
                Self::new(self.re * s, self.im * s)
            }
            fn inv(self) -> Self {
                let d = self.re * self.re + self.im * self.im;
                Self::new(self.re / d, -self.im / d)
            }
        }

        fn w_naive_f32(z: C32) -> C32 {
            let z2 = z.mul(z);
            let neg2z2 = z2.scale(-2.0);
            let mut term = C32::new(1.0, 0.0);
            let mut total = term;
            for n in 1..60u32 {
                let denom = (2 * n + 1) as f32;
                term = term.mul(neg2z2).scale(1.0 / denom);
                total = total.add(term);
                if term.abs() < 1e-7 * (total.abs() + 1e-20) {
                    break;
                }
            }
            let exp_re = (-z2.re).exp() * z2.im.cos();
            let exp_im = (-z2.re).exp() * (-z2.im.sin());
            let exp_neg_z2 = C32::new(exp_re, exp_im);
            let sqrt_pi: f32 = std::f32::consts::PI.sqrt();
            let imag = C32::new(-exp_neg_z2.im * sqrt_pi, exp_neg_z2.re * sqrt_pi);
            let z_val = imag.add(z.mul(total).scale(-2.0));
            C32::new(1.0, 0.0).add(z.mul(z_val))
        }

        pub(super) fn w_stable_f32(z: C32) -> C32 {
            let z2 = z.mul(z);
            let inv_2z2 = z2.scale(2.0).inv();
            let mut coeff = C32::new(1.0, 0.0);
            let mut total = coeff;
            for n in 0..30u32 {
                let factor = (2 * n + 3) as f32;
                coeff = coeff.mul(inv_2z2).scale(factor);
                total = total.add(coeff);
                if coeff.abs() < 1e-6 * (total.abs() + 1e-20) {
                    break;
                }
            }
            let w_asymp = inv_2z2.mul(total).scale(-1.0);
            let exp_re = (-z2.re).exp() * z2.im.cos();
            let exp_im = (-z2.re).exp() * (-z2.im.sin());
            let zexp = z.mul(C32::new(exp_re, exp_im));
            let sqrt_pi: f32 = std::f32::consts::PI.sqrt();
            let sigma: f32 = if z.im < 0.0 { 2.0 } else { 1.0 };
            let w_exp = C32::new(-zexp.im * sqrt_pi * sigma, zexp.re * sqrt_pi * sigma);
            w_asymp.add(w_exp)
        }

        #[test]
        fn w_stable_avoids_cancellation_f32() {
            let z = C32::new(5.57, 0.197);
            let stable = w_stable_f32(z);
            assert!(
                stable.abs() < 0.1,
                "f32 stable W(5.57): |W|={}",
                stable.abs()
            );
            assert!(
                stable.re < 0.0,
                "f32 stable W(5.57): Re[W]={} (should be <0)",
                stable.re
            );

            let naive = w_naive_f32(z);
            let diff_re = (naive.re - stable.re).abs();
            let ratio = diff_re / stable.re.abs().max(1e-10);
            eprintln!(
                "  f32 z=5.57: naive Re={:.6e}, stable Re={:.6e}, ratio={ratio:.2e}",
                naive.re, stable.re
            );
            assert!(
                ratio > 0.01,
                "f32 naive should show cancellation error at z=5.57"
            );
        }

        #[test]
        fn w_stable_vs_naive_convergence_region_f32() {
            for &z_re in &[4.5_f32, 5.0, 6.0, 8.0] {
                let z = C32::new(z_re, 0.1);
                let stable = w_stable_f32(z);
                assert!(
                    stable.abs() < 0.5,
                    "f32 stable W({z_re}): |W|={} should be small",
                    stable.abs()
                );
                assert!(
                    stable.re < 0.0,
                    "f32 stable W({z_re}): Re[W]={} should be <0",
                    stable.re
                );
            }
        }
    }

    #[test]
    fn naive_has_cancellation_error_stable_does_not() {
        use super::*;
        // At z=5.5, naive 1+z·Z(z) has >1% cancellation error vs the
        // stable asymptotic expansion (which avoids the near-cancellation).
        let z = Complex::new(5.5, 0.1);
        let naive = Complex::ONE + z * plasma_dispersion_z(z);
        let stable = plasma_dispersion_w_stable(z);
        let rel = (naive - stable).abs() / stable.abs().max(1e-30);
        eprintln!(
            "  z=5.5: naive Re={:.6e} stable Re={:.6e} divergence={rel:.2e}",
            naive.re, stable.re
        );
        assert!(
            rel > 0.01,
            "z=5.5: naive should show >1% cancellation error, got {rel:.2e}"
        );

        // Both agree well at z=2 where cancellation is modest
        let z2 = Complex::new(2.0, 0.1);
        let naive2 = Complex::ONE + z2 * plasma_dispersion_z(z2);
        let stable2 = plasma_dispersion_w_stable(z2);
        let rel2 = (naive2 - stable2).abs() / naive2.abs().max(1e-30);
        eprintln!(
            "  z=2.0: naive Re={:.6e} stable Re={:.6e} divergence={rel2:.2e}",
            naive2.re, stable2.re
        );
    }

    #[test]
    fn w_stable_correct_at_cancellation_region() {
        use super::*;
        for &z_re in &[4.0, 5.0, 5.57, 6.0, 8.0, 10.0] {
            let z = Complex::new(z_re, 0.197);
            let stable = plasma_dispersion_w_stable(z);
            assert!(
                stable.abs() < 0.1,
                "f64 W({z_re}): |W|={:.4e}",
                stable.abs()
            );
            assert!(
                stable.re < 0.0,
                "f64 W({z_re}): Re[W]={:.4e} should be <0",
                stable.re
            );
        }
    }

    #[test]
    fn w_stable_f32_vs_f64_parity() {
        use super::*;
        for &z_re in &[5.0_f64, 5.57, 6.0, 8.0, 10.0] {
            let z_f64 = Complex::new(z_re, 0.197);
            let w_f64 = plasma_dispersion_w_stable(z_f64);

            let z_f32 = f32_stability::C32::new(z_re as f32, 0.197);
            let w_f32 = f32_stability::w_stable_f32(z_f32);

            let rel_re = (w_f32.re as f64 - w_f64.re).abs() / w_f64.re.abs().max(1e-30);
            let rel_im = (w_f32.im as f64 - w_f64.im).abs() / w_f64.im.abs().max(1e-30);
            eprintln!(
                "  |z|={z_re}: f32 Re={:.6e} f64 Re={:.6e} rel={rel_re:.2e}",
                w_f32.re, w_f64.re
            );
            assert!(
                rel_re < 1e-4,
                "f32-vs-f64 Re[W] at z={z_re}: rel={rel_re:.2e}"
            );
            assert!(
                rel_im < 1e-3,
                "f32-vs-f64 Im[W] at z={z_re}: rel={rel_im:.2e}"
            );
        }
    }
}
