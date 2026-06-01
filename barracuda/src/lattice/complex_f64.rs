// SPDX-License-Identifier: AGPL-3.0-or-later

//! Complex f64 arithmetic for lattice field theory.
//!
//! Re-exports barraCuda's `Complex64` as the single source of truth,
//! adding hotSpring-specific extensions (`from_polar`) and the WGSL
//! shader constant for GPU lattice QCD shaders.
//!
//! # Provenance
//!
//! Original: barraCuda `ops/lattice/cpu_complex.rs`.
//! hotSpring extensions: `from_polar`.

#[cfg(feature = "barracuda-local")]
pub use barracuda::ops::lattice::complex_f64::WGSL_COMPLEX64;
#[cfg(not(feature = "barracuda-local"))]
pub const WGSL_COMPLEX64: &str = include_str!("shaders/complex_f64.wgsl");
#[cfg(feature = "barracuda-local")]
pub use barracuda::ops::lattice::cpu_complex::Complex64;

#[cfg(not(feature = "barracuda-local"))]
mod ipc_complex {
    use std::ops::{Add, Sub, Mul, Neg, AddAssign, SubAssign, MulAssign, Div};

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Complex64 {
        pub re: f64,
        pub im: f64,
    }
    impl Complex64 {
        pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
        pub const ONE: Self = Self { re: 1.0, im: 0.0 };
        pub const I: Self = Self { re: 0.0, im: 1.0 };
        #[inline] pub fn new(re: f64, im: f64) -> Self { Self { re, im } }
        #[inline] pub fn conj(self) -> Self { Self { re: self.re, im: -self.im } }
        #[inline] pub fn abs_sq(self) -> f64 { self.re * self.re + self.im * self.im }
        #[inline] pub fn abs(self) -> f64 { self.abs_sq().sqrt() }
        #[inline] pub fn scale(self, s: f64) -> Self { Self { re: self.re * s, im: self.im * s } }
        #[inline]
        pub fn exp(self) -> Self {
            let e = self.re.exp();
            Self { re: e * self.im.cos(), im: e * self.im.sin() }
        }
        #[inline]
        pub fn inv(self) -> Self {
            let d = self.abs_sq();
            Self { re: self.re / d, im: -self.im / d }
        }
    }
    impl Add for Complex64 { type Output = Self; fn add(self, r: Self) -> Self { Self { re: self.re + r.re, im: self.im + r.im } } }
    impl Sub for Complex64 { type Output = Self; fn sub(self, r: Self) -> Self { Self { re: self.re - r.re, im: self.im - r.im } } }
    impl Mul for Complex64 { type Output = Self; fn mul(self, r: Self) -> Self { Self { re: self.re * r.re - self.im * r.im, im: self.re * r.im + self.im * r.re } } }
    impl Div for Complex64 { type Output = Self; fn div(self, r: Self) -> Self { self * r.conj().scale(1.0 / r.abs_sq()) } }
    impl Neg for Complex64 { type Output = Self; fn neg(self) -> Self { Self { re: -self.re, im: -self.im } } }
    impl Mul<f64> for Complex64 { type Output = Self; fn mul(self, s: f64) -> Self { self.scale(s) } }
    impl AddAssign for Complex64 { fn add_assign(&mut self, r: Self) { self.re += r.re; self.im += r.im; } }
    impl SubAssign for Complex64 { fn sub_assign(&mut self, r: Self) { self.re -= r.re; self.im -= r.im; } }
    impl MulAssign for Complex64 { fn mul_assign(&mut self, r: Self) { *self = *self * r; } }
    impl std::fmt::Display for Complex64 { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "({} + {}i)", self.re, self.im) } }
}
#[cfg(not(feature = "barracuda-local"))]
pub use ipc_complex::Complex64;

/// hotSpring extension: construct e^{iθ} from polar angle.
#[inline]
pub fn from_polar(theta: f64) -> Complex64 {
    Complex64::new(theta.cos(), theta.sin())
}

#[cfg(all(test, feature = "barracuda-local"))]
mod tests {
    use super::*;

    #[test]
    fn complex_add_sub() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, -1.0);
        let c = a + b;
        assert!((c.re - 4.0).abs() < 1e-15);
        assert!((c.im - 1.0).abs() < 1e-15);
        let d = a - b;
        assert!((d.re - (-2.0)).abs() < 1e-15);
        assert!((d.im - 3.0).abs() < 1e-15);
    }

    #[test]
    fn complex_mul() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, 4.0);
        let c = a * b;
        assert!((c.re - (-5.0)).abs() < 1e-15);
        assert!((c.im - 10.0).abs() < 1e-15);
    }

    #[test]
    fn complex_conj() {
        let a = Complex64::new(3.0, 4.0);
        let c = a.conj();
        assert!((c.re - 3.0).abs() < 1e-15);
        assert!((c.im - (-4.0)).abs() < 1e-15);
    }

    #[test]
    fn complex_abs() {
        let a = Complex64::new(3.0, 4.0);
        assert!((a.abs() - 5.0).abs() < 1e-15);
    }

    #[test]
    fn complex_exp_euler() {
        let z = Complex64::new(0.0, std::f64::consts::PI);
        let e = z.exp();
        assert!((e.re - (-1.0)).abs() < 1e-14, "e^(iπ) = -1");
        assert!(e.im.abs() < 1e-14, "e^(iπ) imag = 0");
    }

    #[test]
    fn complex_from_polar() {
        let z = from_polar(std::f64::consts::FRAC_PI_4);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((z.re - s2).abs() < 1e-15);
        assert!((z.im - s2).abs() < 1e-15);
    }

    #[test]
    fn complex_inv() {
        let a = Complex64::new(1.0, 1.0);
        let inv = a.inv();
        let product = a * inv;
        assert!((product.re - 1.0).abs() < 1e-14);
        assert!(product.im.abs() < 1e-14);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Exact equality for Complex64 literal components (ZERO, ONE, I)."
    )]
    fn complex_constants() {
        assert_eq!(Complex64::ZERO.re, 0.0);
        assert_eq!(Complex64::ZERO.im, 0.0);
        assert_eq!(Complex64::ONE.re, 1.0);
        assert_eq!(Complex64::ONE.im, 0.0);
        assert_eq!(Complex64::I.re, 0.0);
        assert_eq!(Complex64::I.im, 1.0);
    }
}
