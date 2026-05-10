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
//! hotSpring extensions: `from_polar`, `WGSL_COMPLEX64`.

#[cfg(feature = "barracuda-local")]
pub use barracuda::ops::lattice::cpu_complex::Complex64;

#[cfg(not(feature = "barracuda-local"))]
pub use self::local_complex::Complex64;

#[cfg(not(feature = "barracuda-local"))]
mod local_complex {
    /// Minimal Complex64 for IPC-only builds (mirrors barraCuda cpu_complex).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Complex64 {
        pub re: f64,
        pub im: f64,
    }

    impl Complex64 {
        pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
        pub const ONE: Self = Self { re: 1.0, im: 0.0 };
        pub const I: Self = Self { re: 0.0, im: 1.0 };

        #[inline]
        pub const fn new(re: f64, im: f64) -> Self {
            Self { re, im }
        }

        #[inline]
        pub fn conj(self) -> Self {
            Self::new(self.re, -self.im)
        }

        #[inline]
        pub fn abs(self) -> f64 {
            (self.re * self.re + self.im * self.im).sqrt()
        }

        #[inline]
        pub fn norm_sq(self) -> f64 {
            self.re * self.re + self.im * self.im
        }

        #[inline]
        pub fn abs_sq(self) -> f64 {
            self.re * self.re + self.im * self.im
        }

        #[inline]
        pub fn scale(self, s: f64) -> Self {
            Self::new(self.re * s, self.im * s)
        }

        #[inline]
        pub fn inv(self) -> Self {
            let n = self.norm_sq();
            Self::new(self.re / n, -self.im / n)
        }

        #[inline]
        pub fn exp(self) -> Self {
            let r = self.re.exp();
            Self::new(r * self.im.cos(), r * self.im.sin())
        }
    }

    impl std::ops::Add for Complex64 {
        type Output = Self;
        #[inline]
        fn add(self, rhs: Self) -> Self {
            Self::new(self.re + rhs.re, self.im + rhs.im)
        }
    }

    impl std::ops::Sub for Complex64 {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.re - rhs.re, self.im - rhs.im)
        }
    }

    impl std::ops::Mul for Complex64 {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: Self) -> Self {
            Self::new(
                self.re * rhs.re - self.im * rhs.im,
                self.re * rhs.im + self.im * rhs.re,
            )
        }
    }

    impl std::ops::Mul<f64> for Complex64 {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: f64) -> Self {
            Self::new(self.re * rhs, self.im * rhs)
        }
    }

    impl std::ops::Neg for Complex64 {
        type Output = Self;
        #[inline]
        fn neg(self) -> Self {
            Self::new(-self.re, -self.im)
        }
    }

    impl std::ops::AddAssign for Complex64 {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            self.re += rhs.re;
            self.im += rhs.im;
        }
    }

    impl std::ops::SubAssign for Complex64 {
        #[inline]
        fn sub_assign(&mut self, rhs: Self) {
            self.re -= rhs.re;
            self.im -= rhs.im;
        }
    }

    impl std::ops::MulAssign for Complex64 {
        #[inline]
        fn mul_assign(&mut self, rhs: Self) {
            let re = self.re * rhs.re - self.im * rhs.im;
            let im = self.re * rhs.im + self.im * rhs.re;
            self.re = re;
            self.im = im;
        }
    }

    impl std::ops::MulAssign<f64> for Complex64 {
        #[inline]
        fn mul_assign(&mut self, rhs: f64) {
            self.re *= rhs;
            self.im *= rhs;
        }
    }

    impl std::ops::Div for Complex64 {
        type Output = Self;
        #[inline]
        fn div(self, rhs: Self) -> Self {
            self * rhs.inv()
        }
    }
}

/// WGSL shader source for Complex64 operations.
///
/// Matches the Rust-side implementation exactly. Can be prepended to any
/// WGSL shader that needs complex arithmetic.
pub const WGSL_COMPLEX64: &str = include_str!("shaders/complex_f64.wgsl");

/// hotSpring extension: construct e^{iθ} from polar angle.
#[inline]
pub fn from_polar(theta: f64) -> Complex64 {
    Complex64::new(theta.cos(), theta.sin())
}

#[cfg(test)]
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
