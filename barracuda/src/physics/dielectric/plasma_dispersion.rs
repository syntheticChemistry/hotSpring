// SPDX-License-Identifier: AGPL-3.0-or-later

//! Plasma parameters and dispersion functions Z(z), W(z).
//!
//! The core dispersion functions are delegated to barraCuda's
//! `special::plasma_dispersion` module (absorbed from hotSpring in S89).
//! This module adds `PlasmaParams`, `chi0_classical`, and `epsilon_vlasov`
//! which are hotSpring-specific physics.

use super::complex::Complex;
use std::f64::consts::PI;

pub use barracuda::special::{
    plasma_dispersion_w, plasma_dispersion_w_stable, plasma_dispersion_z,
};

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

/// Free-particle (Vlasov) susceptibility χ₀(k,ω).
///
/// χ₀(k,ω) = -(k_D²/k²) W(ω/(√2 k v_th))
pub fn chi0_classical(k: f64, omega: Complex, params: &PlasmaParams) -> Complex {
    let z = omega * (1.0 / (std::f64::consts::SQRT_2 * k * params.v_th));
    let w = plasma_dispersion_w(z);
    w * (-(params.k_debye * params.k_debye) / (k * k))
}

/// Vlasov (collisionless) dielectric function.
pub fn epsilon_vlasov(k: f64, omega: Complex, params: &PlasmaParams) -> Complex {
    Complex::ONE - chi0_classical(k, omega, params)
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
            let eps = epsilon_vlasov(1.0, Complex::ZERO, &params);
            let expected = 1.0 + (params.k_debye / 1.0).powi(2);
            let err = (eps.re - expected).abs() / expected;
            assert!(
                err < 1e-12,
                "Debye screening Γ={gamma}, κ={kappa}: err={err}"
            );
        }
    }

    #[test]
    fn chi0_classical_finite() {
        let params = PlasmaParams::from_coupling(1.0, 1.0);
        let chi = chi0_classical(1.0, Complex::new(1.0, 0.0), &params);
        assert!(chi.abs().is_finite(), "χ₀ should be finite");
    }

    #[test]
    fn plasma_params_from_coupling_physical() {
        let p = PlasmaParams::from_coupling(1.0, 1.0);
        assert!(p.omega_p > 0.0, "ωₚ should be positive");
        assert!(p.v_th > 0.0, "v_th should be positive");
        assert!(p.temperature > 0.0, "T should be positive");
        assert!(p.a > 0.0, "a should be positive");
    }

    /// f32-precision W(z) stability tests (hotSpring-specific).
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
