// SPDX-License-Identifier: AGPL-3.0-only

//! SU(3) matrix operations for lattice gauge theory.
//!
//! An SU(3) matrix is a 3×3 unitary matrix with determinant 1.
//! In lattice QCD, each link variable `U_μ`(x) is an `SU(3)` matrix
//! representing the parallel transporter along direction μ from site x.
//!
//! Storage: row-major, 9 Complex64 values (18 f64).
//!
//! # References
//!
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 2
//! - Creutz, "Quarks, Gluons and Lattices" (1983), Ch. 8

use std::ops::{Add, Mul, Sub};

use super::complex_f64::Complex64;

/// 3×3 complex matrix — SU(3) link variable.
///
/// Row-major storage: `m[row][col]`.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct Su3Matrix {
    /// Matrix elements m[row][col].
    pub m: [[Complex64; 3]; 3],
}

impl Mul for Su3Matrix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut s = Complex64::ZERO;
                for k in 0..3 {
                    s += self.m[i][k] * rhs.m[k][j];
                }
                r.m[i][j] = s;
            }
        }
        r
    }
}

impl Add for Su3Matrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j] + rhs.m[i][j];
            }
        }
        r
    }
}

impl Sub for Su3Matrix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j] - rhs.m[i][j];
            }
        }
        r
    }
}

impl Su3Matrix {
    /// 3×3 identity matrix.
    pub const IDENTITY: Self = Self {
        m: [
            [Complex64::ONE, Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ZERO, Complex64::ONE],
        ],
    };

    /// Zero matrix (all elements 0).
    pub const ZERO: Self = Self {
        m: [[Complex64::ZERO; 3]; 3],
    };

    /// Conjugate transpose (adjoint / dagger).
    pub fn adjoint(self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[j][i].conj();
            }
        }
        r
    }

    /// Trace: Tr(U) = sum\_i `U_ii`
    pub fn trace(self) -> Complex64 {
        self.m[0][0] + self.m[1][1] + self.m[2][2]
    }

    /// Real part of trace.
    #[must_use]
    pub fn re_trace(self) -> f64 {
        self.m[0][0].re + self.m[1][1].re + self.m[2][2].re
    }

    /// Determinant of a 3×3 complex matrix.
    pub fn det(self) -> Complex64 {
        let m = &self.m;
        let a = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
        let b = m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]);
        let c = m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        a - b + c
    }

    /// Scale by a real number.
    pub fn scale(self, s: f64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j].scale(s);
            }
        }
        r
    }

    /// Scale by a complex number.
    pub fn scale_complex(self, s: Complex64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j] * s;
            }
        }
        r
    }

    /// Frobenius norm squared: sum |`m_ij`|²
    #[must_use]
    pub fn norm_sq(self) -> f64 {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += self.m[i][j].abs_sq();
            }
        }
        s
    }

    /// Project back onto SU(3) via modified Gram-Schmidt reunitarization.
    ///
    /// After numerical integration, link matrices drift off the SU(3) manifold.
    /// This projects back by orthonormalizing rows and fixing det = 1.
    pub fn reunitarize(self) -> Self {
        let mut u = self;

        // Normalize row 0
        let n0 = row_norm(&u, 0);
        if n0 > super::constants::LATTICE_DIVISION_GUARD {
            let inv = 1.0 / n0;
            for j in 0..3 {
                u.m[0][j] = u.m[0][j].scale(inv);
            }
        }

        // Orthogonalize row 1 against row 0, then normalize
        let dot01 = row_dot(&u, 0, 1);
        for j in 0..3 {
            u.m[1][j] -= u.m[0][j] * dot01;
        }
        let n1 = row_norm(&u, 1);
        if n1 > super::constants::LATTICE_DIVISION_GUARD {
            let inv = 1.0 / n1;
            for j in 0..3 {
                u.m[1][j] = u.m[1][j].scale(inv);
            }
        }

        // Row 2 = conj(row 0 × row 1) to ensure det = 1
        u.m[2][0] = (u.m[0][1] * u.m[1][2] - u.m[0][2] * u.m[1][1]).conj();
        u.m[2][1] = (u.m[0][2] * u.m[1][0] - u.m[0][0] * u.m[1][2]).conj();
        u.m[2][2] = (u.m[0][0] * u.m[1][1] - u.m[0][1] * u.m[1][0]).conj();

        u
    }

    /// Generate a random SU(3) matrix near identity (for HMC momentum refresh).
    ///
    /// Returns exp(i ε H) where H is a random traceless Hermitian matrix
    /// with components drawn from a simple LCG.
    pub fn random_near_identity(seed: &mut u64, epsilon: f64) -> Self {
        use super::constants::lcg_gaussian;

        let mut h = [[Complex64::ZERO; 3]; 3];
        let mut rand_gauss = || -> f64 { lcg_gaussian(seed) };

        // Diagonal (traceless): a3 * λ3 + a8 * λ8
        let a3 = rand_gauss() * epsilon;
        let a8 = rand_gauss() * epsilon;
        h[0][0] = Complex64::new(a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h[1][1] = Complex64::new(-a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h[2][2] = Complex64::new(-2.0 * a8 / 3.0_f64.sqrt(), 0.0);

        // Off-diagonal
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let re = rand_gauss() * epsilon;
            let im = rand_gauss() * epsilon;
            h[i][j] = Complex64::new(re, im);
            h[j][i] = Complex64::new(re, -im); // Hermitian
        }

        // exp(iH) ≈ I + iH - H²/2 (second-order Cayley for small ε)
        let mut result = Self::IDENTITY;
        for (i, row) in result.m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += Complex64::I * h[i][j];
            }
        }

        // H² contribution
        for (i, row) in result.m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let h2_ij = (0..3).fold(Complex64::ZERO, |acc, k| acc + h[i][k] * h[k][j]);
                *cell -= h2_ij.scale(0.5);
            }
        }

        result.reunitarize()
    }

    /// Generate a random su(3) Lie algebra element (traceless anti-Hermitian).
    ///
    /// Used for HMC momentum initialization: P = i H where H is traceless Hermitian.
    /// Generator coefficients `c_a` ~ N(0, 1/√2) so that T(P) = Σ\_a `c_a`² gives
    /// the correct canonical distribution exp(-T). The 1/√2 factor comes from
    /// the fundamental representation normalization Tr(`T_a` `T_b`) = `δ_ab`/2.
    pub fn random_algebra(seed: &mut u64) -> Self {
        use super::constants::lcg_gaussian;

        let scale = std::f64::consts::FRAC_1_SQRT_2;
        let mut rand_gauss = || -> f64 { scale * lcg_gaussian(seed) };

        let mut h = Self::ZERO;

        let a3 = rand_gauss();
        let a8 = rand_gauss();
        h.m[0][0] = Complex64::new(a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h.m[1][1] = Complex64::new(-a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h.m[2][2] = Complex64::new(-2.0 * a8 / 3.0_f64.sqrt(), 0.0);

        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let re = rand_gauss();
            let im = rand_gauss();
            h.m[i][j] = Complex64::new(re, im);
            h.m[j][i] = Complex64::new(re, -im);
        }

        // Return iH (anti-Hermitian, traceless)
        h.scale_complex(Complex64::I)
    }
}

fn row_norm(u: &Su3Matrix, row: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..3 {
        s += u.m[row][j].abs_sq();
    }
    s.sqrt()
}

fn row_dot(u: &Su3Matrix, r1: usize, r2: usize) -> Complex64 {
    let mut s = Complex64::ZERO;
    for j in 0..3 {
        s += u.m[r1][j].conj() * u.m[r2][j];
    }
    s
}

/// WGSL shader for SU(3) 3×3 complex matrix operations.
///
/// Depends on Complex64 WGSL (prepend `WGSL_COMPLEX64` before this).
pub const WGSL_SU3: &str = include_str!("shaders/su3_f64.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_properties() {
        let i = Su3Matrix::IDENTITY;
        assert!((i.det().re - 1.0).abs() < 1e-14);
        assert!(i.det().im.abs() < 1e-14);
        assert!((i.re_trace() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn adjoint_of_identity() {
        let i = Su3Matrix::IDENTITY;
        let d = i.adjoint();
        for r in 0..3 {
            for c in 0..3 {
                assert!((d.m[r][c].re - i.m[r][c].re).abs() < 1e-14);
                assert!((d.m[r][c].im - i.m[r][c].im).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn mul_identity() {
        let mut seed = 42u64;
        let u = Su3Matrix::random_near_identity(&mut seed, 0.3);
        let v = u * Su3Matrix::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                assert!((v.m[i][j].re - u.m[i][j].re).abs() < 1e-12);
                assert!((v.m[i][j].im - u.m[i][j].im).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn unitarity_check() {
        let mut seed = 123u64;
        let u = Su3Matrix::random_near_identity(&mut seed, 0.2);
        let ud = u.adjoint();
        let prod = u * ud;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.m[i][j].re - expected).abs() < 1e-6,
                    "U U† not identity at ({i},{j}): {:.6e}",
                    prod.m[i][j].re - expected
                );
                assert!(
                    prod.m[i][j].im.abs() < 1e-6,
                    "U U† imaginary at ({i},{j}): {:.6e}",
                    prod.m[i][j].im
                );
            }
        }
    }

    #[test]
    fn det_near_one() {
        let mut seed = 777u64;
        let u = Su3Matrix::random_near_identity(&mut seed, 0.1);
        let d = u.det();
        assert!(
            (d.abs() - 1.0).abs() < 0.01,
            "det should be near 1: |det| = {}",
            d.abs()
        );
    }

    #[test]
    fn reunitarize_fixes_drift() {
        let mut seed = 999u64;
        let mut u = Su3Matrix::random_near_identity(&mut seed, 0.5);
        // Introduce drift
        u.m[0][0].re += 0.1;
        u.m[1][2].im -= 0.05;

        let fixed = u.reunitarize();
        let prod = fixed * fixed.adjoint();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.m[i][j].re - expected).abs() < 1e-10,
                    "reunitarized U U† not identity at ({i},{j})"
                );
            }
        }
    }
}
