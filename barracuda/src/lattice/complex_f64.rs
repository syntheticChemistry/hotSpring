// SPDX-License-Identifier: AGPL-3.0-only

//! Complex f64 arithmetic for lattice field theory.
//!
//! Extracted from barracuda's `fft_1d_f64.wgsl` Complex64 struct and promoted
//! to a standalone module. Both Rust-side reference implementation and WGSL
//! shader source for GPU acceleration.
//!
//! # Provenance
//!
//! Original: `phase1/toadstool/crates/barracuda/src/ops/fft/fft_1d_f64.wgsl`
//! lines 27-59. Extracted Feb 2026 for lattice QCD SU(3) matrix operations.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Complex number with f64 real and imaginary parts.
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
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn abs_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn abs(self) -> f64 {
        self.abs_sq().sqrt()
    }

    #[inline]
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// e^{i theta}
    #[inline]
    pub fn from_polar(theta: f64) -> Self {
        Self {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    #[inline]
    pub fn inv(self) -> Self {
        let d = self.abs_sq();
        Self {
            re: self.re / d,
            im: -self.im / d,
        }
    }
}

impl Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl AddAssign for Complex64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl SubAssign for Complex64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl MulAssign for Complex64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Complex64 {
    type Output = Self;
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let d = rhs.abs_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

impl Neg for Complex64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6}+{:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6}{:.6}i", self.re, self.im)
        }
    }
}

/// WGSL shader source for Complex64 operations.
///
/// Matches the Rust-side implementation exactly. Can be prepended to any
/// WGSL shader that needs complex arithmetic.
pub const WGSL_COMPLEX64: &str = r"
struct Complex64 {
    re: f64,
    im: f64,
}

fn c64_new(re: f64, im: f64) -> Complex64 {
    return Complex64(re, im);
}

fn c64_zero() -> Complex64 {
    return Complex64(f64(0.0), f64(0.0));
}

fn c64_one() -> Complex64 {
    return Complex64(f64(1.0), f64(0.0));
}

fn c64_add(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re + b.re, a.im + b.im);
}

fn c64_sub(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re - b.re, a.im - b.im);
}

fn c64_mul(a: Complex64, b: Complex64) -> Complex64 {
    return Complex64(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

fn c64_conj(a: Complex64) -> Complex64 {
    return Complex64(a.re, -a.im);
}

fn c64_scale(a: Complex64, s: f64) -> Complex64 {
    return Complex64(a.re * s, a.im * s);
}

fn c64_abs_sq(a: Complex64) -> f64 {
    return a.re * a.re + a.im * a.im;
}

fn c64_abs(a: Complex64) -> f64 {
    return sqrt(c64_abs_sq(a));
}

fn c64_inv(a: Complex64) -> Complex64 {
    let d = c64_abs_sq(a);
    return Complex64(a.re / d, -a.im / d);
}

fn c64_div(a: Complex64, b: Complex64) -> Complex64 {
    return c64_mul(a, c64_inv(b));
}

fn c64_exp(a: Complex64) -> Complex64 {
    let r = exp(a.re);
    return Complex64(r * cos(a.im), r * sin(a.im));
}
";

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
    fn complex_mul_conj_gives_abs_sq() {
        let a = Complex64::new(3.0, 4.0);
        let p = a * a.conj();
        assert!((p.re - 25.0).abs() < 1e-14);
        assert!(p.im.abs() < 1e-14);
    }

    #[test]
    fn complex_div_inverse() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, 4.0);
        let c = a / b;
        let d = c * b;
        assert!((d.re - a.re).abs() < 1e-14);
        assert!((d.im - a.im).abs() < 1e-14);
    }

    #[test]
    fn complex_from_polar() {
        let z = Complex64::from_polar(std::f64::consts::FRAC_PI_4);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((z.re - s2).abs() < 1e-15);
        assert!((z.im - s2).abs() < 1e-15);
    }
}
