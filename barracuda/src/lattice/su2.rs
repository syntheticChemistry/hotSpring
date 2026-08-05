// SPDX-License-Identifier: AGPL-3.0-or-later

//! SU(2) matrix operations for lattice gauge theory.
//!
//! An SU(2) matrix is a 2×2 unitary matrix with determinant 1,
//! parameterized as U = a₀I + i(a₁σ₁ + a₂σ₂ + a₃σ₃) where
//! a₀² + a₁² + a₂² + a₃² = 1 and σᵢ are the Pauli matrices.
//!
//! Storage: row-major, 4 Complex64 values (8 f64).
//! Memory per link: 64 bytes (vs 144 for SU(3) — 2.25× smaller).
//!
//! # References
//!
//! - Creutz, "Quarks, Gluons and Lattices" (1983), Ch. 8
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 4

use super::complex_f64::Complex64;
use super::gauge_group::GaugeGroup;

/// 2×2 complex matrix — SU(2) link variable.
#[derive(Clone, Copy, Debug)]
pub struct Su2Matrix {
    pub m: [[Complex64; 2]; 2],
}

impl Su2Matrix {
    pub const IDENTITY: Self = Self {
        m: [
            [Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE],
        ],
    };

    pub const ZERO: Self = Self {
        m: [[Complex64::ZERO; 2]; 2],
    };
}

impl GaugeGroup for Su2Matrix {
    const NC: usize = 2;
    const LINK_REALS: usize = 8;
    const N_GENERATORS: usize = 3;

    fn gauge_group_tag() -> &'static str {
        "su2"
    }

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn zero() -> Self {
        Self::ZERO
    }

    fn mul(&self, rhs: &Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..2 {
            for j in 0..2 {
                let mut s = Complex64::ZERO;
                for k in 0..2 {
                    s += self.m[i][k] * rhs.m[k][j];
                }
                r.m[i][j] = s;
            }
        }
        r
    }

    fn adjoint(&self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..2 {
            for j in 0..2 {
                r.m[i][j] = self.m[j][i].conj();
            }
        }
        r
    }

    fn re_trace(&self) -> f64 {
        self.m[0][0].re + self.m[1][1].re
    }

    fn trace(&self) -> Complex64 {
        self.m[0][0] + self.m[1][1]
    }

    fn det(&self) -> Complex64 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }

    fn scale(&self, s: f64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..2 {
            for j in 0..2 {
                r.m[i][j] = self.m[i][j].scale(s);
            }
        }
        r
    }

    fn scale_complex(&self, s: Complex64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..2 {
            for j in 0..2 {
                r.m[i][j] = self.m[i][j] * s;
            }
        }
        r
    }

    fn norm_sq(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..2 {
            for j in 0..2 {
                s += self.m[i][j].abs_sq();
            }
        }
        s
    }

    fn reunitarize(&self) -> Self {
        let mut u = *self;

        let n0 = (u.m[0][0].abs_sq() + u.m[0][1].abs_sq()).sqrt();
        if n0 > 1e-15 {
            let inv = 1.0 / n0;
            u.m[0][0] = u.m[0][0].scale(inv);
            u.m[0][1] = u.m[0][1].scale(inv);
        }

        // For SU(2), row 1 = [-conj(m[0][1]), conj(m[0][0])] ensures det = 1
        u.m[1][0] = Complex64::new(-u.m[0][1].re, u.m[0][1].im);
        u.m[1][1] = Complex64::new(u.m[0][0].re, -u.m[0][0].im);

        u
    }

    fn add(&self, rhs: &Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..2 {
            for j in 0..2 {
                r.m[i][j] = self.m[i][j] + rhs.m[i][j];
            }
        }
        r
    }

    fn sub(&self, rhs: &Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..2 {
            for j in 0..2 {
                r.m[i][j] = self.m[i][j] - rhs.m[i][j];
            }
        }
        r
    }

    fn random_near_identity(seed: &mut u64, epsilon: f64) -> Self {
        use super::constants::lcg_gaussian;

        // SU(2) near identity: I + iε(a₁σ₁ + a₂σ₂ + a₃σ₃), then reunitarize
        let a1 = lcg_gaussian(seed) * epsilon;
        let a2 = lcg_gaussian(seed) * epsilon;
        let a3 = lcg_gaussian(seed) * epsilon;

        let mut u = Self {
            m: [
                [
                    Complex64::new(1.0 + 0.0, a3),
                    Complex64::new(a2, a1),
                ],
                [
                    Complex64::new(-a2, a1),
                    Complex64::new(1.0 + 0.0, -a3),
                ],
            ],
        };

        // Proper SU(2) parameterization: normalize quaternion (a0, a1, a2, a3)
        // where U = a0*I + i*(a1*σ1 + a2*σ2 + a3*σ3)
        // => m = [[a0+i*a3, a2+i*a1], [-a2+i*a1, a0-i*a3]]
        u = u.reunitarize();
        u
    }

    fn random_algebra(seed: &mut u64) -> Self {
        use super::constants::lcg_gaussian;

        // su(2) algebra: 3 Pauli generators, traceless anti-Hermitian
        // P = i(c₁σ₁ + c₂σ₂ + c₃σ₃) with c_a ~ N(0, 1/√2)
        let scale = std::f64::consts::FRAC_1_SQRT_2;
        let c1 = lcg_gaussian(seed) * scale;
        let c2 = lcg_gaussian(seed) * scale;
        let c3 = lcg_gaussian(seed) * scale;

        // iσ₁ = [[0, i], [i, 0]], iσ₂ = [[0, 1], [-1, 0]], iσ₃ = [[i, 0], [0, -i]]
        Self {
            m: [
                [
                    Complex64::new(0.0, c3),
                    Complex64::new(c2, c1),
                ],
                [
                    Complex64::new(-c2, c1),
                    Complex64::new(0.0, -c3),
                ],
            ],
        }
    }

    fn inverse(&self) -> Self {
        // 2x2 inverse: A⁻¹ = adj(A)/det(A)
        let d = self.det();
        let inv_det = d.inv();
        Self {
            m: [
                [self.m[1][1] * inv_det, (Complex64::ZERO - self.m[0][1]) * inv_det],
                [(Complex64::ZERO - self.m[1][0]) * inv_det, self.m[0][0] * inv_det],
            ],
        }
    }

    fn sub_diagonal(&mut self, val: Complex64) {
        self.m[0][0] -= val;
        self.m[1][1] -= val;
    }

    fn write_to_buf(&self, buf: &mut Vec<u8>) {
        for row in 0..2 {
            for col in 0..2 {
                buf.extend_from_slice(&self.m[row][col].re.to_le_bytes());
                buf.extend_from_slice(&self.m[row][col].im.to_le_bytes());
            }
        }
    }

    fn read_from_buf(data: &[u8], offset: usize) -> Self {
        let mut m = [[Complex64::ZERO; 2]; 2];
        for row in 0..2 {
            for col in 0..2 {
                let off = offset + (row * 4 + col * 2) * 8;
                let re = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                let im = f64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap());
                m[row][col] = Complex64::new(re, im);
            }
        }
        Su2Matrix { m }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_properties() {
        let i = Su2Matrix::identity();
        assert!((i.det().re - 1.0).abs() < 1e-14);
        assert!(i.det().im.abs() < 1e-14);
        assert!((i.re_trace() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn unitarity_check() {
        let mut seed = 123u64;
        let u = Su2Matrix::random_near_identity(&mut seed, 0.5);
        let ud = u.adjoint();
        let prod = u.mul(&ud);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.m[i][j].re - expected).abs() < 1e-10,
                    "U U† not identity at ({i},{j}): {:.6e}",
                    prod.m[i][j].re - expected
                );
                assert!(
                    prod.m[i][j].im.abs() < 1e-10,
                    "U U† imaginary at ({i},{j}): {:.6e}",
                    prod.m[i][j].im
                );
            }
        }
    }

    #[test]
    fn det_near_one() {
        let mut seed = 777u64;
        let u = Su2Matrix::random_near_identity(&mut seed, 0.3);
        let d = u.det();
        assert!(
            (d.abs() - 1.0).abs() < 0.01,
            "det should be near 1: |det| = {}",
            d.abs()
        );
    }

    #[test]
    fn algebra_is_traceless() {
        let mut seed = 42u64;
        let p = Su2Matrix::random_algebra(&mut seed);
        let tr = p.trace();
        assert!(
            tr.abs() < 1e-14,
            "su(2) algebra element should be traceless: Tr = {:e}",
            tr.abs()
        );
    }

    #[test]
    fn algebra_is_anti_hermitian() {
        let mut seed = 99u64;
        let p = Su2Matrix::random_algebra(&mut seed);
        let pd = p.adjoint();
        let sum = p.add(&pd);
        assert!(
            sum.norm_sq() < 1e-20,
            "su(2) algebra: P + P† should be zero, norm² = {:e}",
            sum.norm_sq()
        );
    }

    #[test]
    fn reunitarize_fixes_drift() {
        let mut seed = 999u64;
        let mut u = Su2Matrix::random_near_identity(&mut seed, 0.5);
        u.m[0][0].re += 0.1;
        u.m[1][0].im -= 0.05;

        let fixed = u.reunitarize();
        let prod = fixed.mul(&fixed.adjoint());

        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.m[i][j].re - expected).abs() < 1e-10,
                    "reunitarized SU(2) U U† not identity at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn serialize_roundtrip() {
        let mut seed = 42u64;
        let u = Su2Matrix::random_near_identity(&mut seed, 0.3);
        let mut buf = Vec::new();
        u.write_to_buf(&mut buf);
        let u2 = Su2Matrix::read_from_buf(&buf, 0);
        for i in 0..2 {
            for j in 0..2 {
                assert!((u.m[i][j].re - u2.m[i][j].re).abs() < 1e-15);
                assert!((u.m[i][j].im - u2.m[i][j].im).abs() < 1e-15);
            }
        }
    }
}
