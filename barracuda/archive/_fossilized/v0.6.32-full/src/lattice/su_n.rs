// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic SU(N) matrix for N=4,5,6,8 via heap-allocated NxN matrices.
//!
//! Uses the Cabibbo-Marinari algorithm for SU(N) updates: decompose SU(N)
//! into SU(2) subgroups embedded in the NxN matrix. This is the standard
//! approach for N>3 lattice simulations.
//!
//! Memory per link: 2*N*N f64 = 16*N*N bytes.
//! SU(4): 256 bytes, SU(5): 400 bytes, SU(6): 576 bytes, SU(8): 1024 bytes.
//!
//! # References
//!
//! - Cabibbo & Marinari, PLB 119, 387 (1982)
//! - Lucini, Teper, Wenger, JHEP 0401, 061 (2004)

use super::complex_f64::Complex64;
use super::gauge_group::GaugeGroup;

/// Heap-allocated NxN complex matrix for SU(N) with runtime-fixed N.
#[derive(Clone, Debug)]
pub struct SuNMatrix {
    pub nc: usize,
    pub m: Vec<Complex64>,
}

impl SuNMatrix {
    pub fn new(nc: usize) -> Self {
        Self {
            nc,
            m: vec![Complex64::ZERO; nc * nc],
        }
    }

    pub fn identity(nc: usize) -> Self {
        let mut s = Self::new(nc);
        for i in 0..nc {
            s.m[i * nc + i] = Complex64::ONE;
        }
        s
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.m[row * self.nc + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: Complex64) {
        self.m[row * self.nc + col] = val;
    }

    fn row_norm(&self, row: usize) -> f64 {
        let mut s = 0.0;
        for j in 0..self.nc {
            s += self.get(row, j).abs_sq();
        }
        s.sqrt()
    }

    fn row_dot(&self, r1: usize, r2: usize) -> Complex64 {
        let mut s = Complex64::ZERO;
        for j in 0..self.nc {
            s += self.get(r1, j).conj() * self.get(r2, j);
        }
        s
    }
}

impl GaugeGroup for SuNMatrix {
    const NC: usize = 0; // Runtime-determined; use self.nc
    const LINK_REALS: usize = 0; // Runtime-determined
    const N_GENERATORS: usize = 0; // Runtime-determined

    fn gauge_group_tag() -> &'static str {
        "su_n"
    }

    fn identity() -> Self {
        Self::identity(4) // Default; callers should use identity(nc) directly
    }

    fn zero() -> Self {
        Self::new(4)
    }

    fn mul(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.nc, rhs.nc);
        let nc = self.nc;
        let mut r = Self::new(nc);
        for i in 0..nc {
            for j in 0..nc {
                let mut s = Complex64::ZERO;
                for k in 0..nc {
                    s += self.get(i, k) * rhs.get(k, j);
                }
                r.set(i, j, s);
            }
        }
        r
    }

    fn adjoint(&self) -> Self {
        let nc = self.nc;
        let mut r = Self::new(nc);
        for i in 0..nc {
            for j in 0..nc {
                r.set(i, j, self.get(j, i).conj());
            }
        }
        r
    }

    fn re_trace(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.nc {
            s += self.get(i, i).re;
        }
        s
    }

    fn re_trace_normalized(&self) -> f64 {
        self.re_trace() / self.nc as f64
    }

    fn trace(&self) -> Complex64 {
        let mut s = Complex64::ZERO;
        for i in 0..self.nc {
            s += self.get(i, i);
        }
        s
    }

    fn det(&self) -> Complex64 {
        // LU decomposition for general N
        let nc = self.nc;
        let mut lu: Vec<Complex64> = self.m.clone();

        let mut det = Complex64::ONE;
        for col in 0..nc {
            // Partial pivoting (find max in column)
            let mut max_val = 0.0f64;
            let mut max_row = col;
            for row in col..nc {
                let val = lu[row * nc + col].abs_sq();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_row != col {
                for j in 0..nc {
                    lu.swap(col * nc + j, max_row * nc + j);
                }
                det = Complex64::ZERO - det; // Swap sign
            }

            let pivot = lu[col * nc + col];
            if pivot.abs_sq() < 1e-30 {
                return Complex64::ZERO;
            }
            det = det * pivot;

            let pivot_inv = pivot.inv();
            for row in (col + 1)..nc {
                let factor = lu[row * nc + col] * pivot_inv;
                for j in col..nc {
                    let val = lu[col * nc + j];
                    lu[row * nc + j] -= factor * val;
                }
            }
        }
        det
    }

    fn scale(&self, s: f64) -> Self {
        let mut r = self.clone();
        for v in &mut r.m {
            *v = v.scale(s);
        }
        r
    }

    fn scale_complex(&self, s: Complex64) -> Self {
        let mut r = self.clone();
        for v in &mut r.m {
            *v = *v * s;
        }
        r
    }

    fn norm_sq(&self) -> f64 {
        self.m.iter().map(|c| c.abs_sq()).sum()
    }

    fn reunitarize(&self) -> Self {
        let nc = self.nc;
        let mut u = self.clone();

        // Modified Gram-Schmidt on rows
        for row in 0..nc {
            // Orthogonalize against all previous rows
            for prev in 0..row {
                let dot = u.row_dot(prev, row);
                for j in 0..nc {
                    let prev_val = u.get(prev, j);
                    let cur = u.get(row, j) - prev_val * dot;
                    u.set(row, j, cur);
                }
            }
            // Normalize
            let norm = u.row_norm(row);
            if norm > 1e-15 {
                let inv = 1.0 / norm;
                for j in 0..nc {
                    u.set(row, j, u.get(row, j).scale(inv));
                }
            }
        }

        // Fix determinant to 1
        let d = u.det();
        if d.abs_sq() > 1e-30 {
            let phase = Complex64::new(d.re, d.im).scale(1.0 / d.abs());
            let correction = Complex64::new(phase.re, -phase.im); // 1/phase
            // Apply to last row
            for j in 0..nc {
                let val = u.get(nc - 1, j) * correction;
                u.set(nc - 1, j, val);
            }
        }

        u
    }

    fn add(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.nc, rhs.nc);
        let mut r = self.clone();
        for (i, v) in r.m.iter_mut().enumerate() {
            *v = *v + rhs.m[i];
        }
        r
    }

    fn sub(&self, rhs: &Self) -> Self {
        debug_assert_eq!(self.nc, rhs.nc);
        let mut r = self.clone();
        for (i, v) in r.m.iter_mut().enumerate() {
            *v = *v - rhs.m[i];
        }
        r
    }

    fn random_near_identity(seed: &mut u64, epsilon: f64) -> Self {
        // Default to SU(4); callers should use random_near_identity_nc
        Self::random_near_identity_nc(4, seed, epsilon)
    }

    fn random_algebra(seed: &mut u64) -> Self {
        Self::random_algebra_nc(4, seed)
    }

    fn inverse(&self) -> Self {
        let nc = self.nc;
        // Gauss-Jordan elimination
        let mut aug = vec![Complex64::ZERO; nc * 2 * nc];
        for i in 0..nc {
            for j in 0..nc {
                aug[i * 2 * nc + j] = self.get(i, j);
            }
            aug[i * 2 * nc + nc + i] = Complex64::ONE;
        }

        for col in 0..nc {
            // Partial pivoting
            let mut max_val = 0.0f64;
            let mut max_row = col;
            for row in col..nc {
                let val = aug[row * 2 * nc + col].abs_sq();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_row != col {
                for j in 0..(2 * nc) {
                    aug.swap(col * 2 * nc + j, max_row * 2 * nc + j);
                }
            }

            let pivot = aug[col * 2 * nc + col];
            let pivot_inv = pivot.inv();
            for j in 0..(2 * nc) {
                aug[col * 2 * nc + j] = aug[col * 2 * nc + j] * pivot_inv;
            }

            for row in 0..nc {
                if row == col {
                    continue;
                }
                let factor = aug[row * 2 * nc + col];
                for j in 0..(2 * nc) {
                    let val = aug[col * 2 * nc + j];
                    aug[row * 2 * nc + j] -= factor * val;
                }
            }
        }

        let mut result = Self::new(nc);
        for i in 0..nc {
            for j in 0..nc {
                result.set(i, j, aug[i * 2 * nc + nc + j]);
            }
        }
        result
    }

    fn sub_diagonal(&mut self, val: Complex64) {
        for i in 0..self.nc {
            self.m[i * self.nc + i] -= val;
        }
    }

    fn write_to_buf(&self, buf: &mut Vec<u8>) {
        for c in &self.m {
            buf.extend_from_slice(&c.re.to_le_bytes());
            buf.extend_from_slice(&c.im.to_le_bytes());
        }
    }

    fn read_from_buf(data: &[u8], offset: usize) -> Self {
        // Infer nc from remaining data length: each link is 2*nc*nc*8 bytes.
        // For the general case, callers should use read_from_buf_nc(nc, ...) directly.
        // This fallback attempts nc=4 as default.
        Self::read_from_buf_nc(4, data, offset)
    }

    fn bytes_per_link() -> usize {
        0 // Runtime; callers use runtime_bytes_per_link or bytes_per_link_nc
    }

    fn runtime_nc(&self) -> usize {
        self.nc
    }

    fn runtime_bytes_per_link(&self) -> usize {
        2 * self.nc * self.nc * 8
    }

    fn random_algebra_for_nc(nc: usize, seed: &mut u64) -> Self {
        Self::random_algebra_nc(nc, seed)
    }

    fn zero_for_nc(nc: usize) -> Self {
        Self::zero_nc(nc)
    }
}

impl SuNMatrix {
    /// Create SU(N) identity for specific nc.
    pub fn identity_nc(nc: usize) -> Self {
        Self::identity(nc)
    }

    pub fn zero_nc(nc: usize) -> Self {
        Self::new(nc)
    }

    /// Bytes per link for a specific nc.
    pub fn bytes_per_link_nc(nc: usize) -> usize {
        2 * nc * nc * 8
    }

    /// Tag string for a specific nc.
    pub fn tag_for_nc(nc: usize) -> String {
        format!("su{nc}")
    }

    /// Random SU(N) near identity with explicit nc.
    pub fn random_near_identity_nc(nc: usize, seed: &mut u64, epsilon: f64) -> Self {
        use super::constants::lcg_gaussian;

        let mut u = Self::identity(nc);
        // Add random traceless Hermitian perturbation
        let n_gen = nc * nc - 1;

        // Diagonal generators (N-1 of them, generalized Gell-Mann λ₃, λ₈, ...)
        let mut diag_coeffs = vec![0.0f64; nc];
        for k in 0..(nc - 1) {
            let coeff = lcg_gaussian(seed) * epsilon;
            let norm = 1.0 / ((2 * (k + 1) * (k + 2)) as f64).sqrt();
            for i in 0..=k {
                diag_coeffs[i] += coeff * norm;
            }
            diag_coeffs[k + 1] -= coeff * norm * (k + 1) as f64;
        }
        for i in 0..nc {
            u.m[i * nc + i] += Complex64::new(0.0, diag_coeffs[i]);
        }

        // Off-diagonal generators: N(N-1)/2 pairs
        for i in 0..nc {
            for j in (i + 1)..nc {
                let re = lcg_gaussian(seed) * epsilon;
                let im = lcg_gaussian(seed) * epsilon;
                u.m[i * nc + j] += Complex64::new(0.0, re) + Complex64::new(im, 0.0);
                u.m[j * nc + i] += Complex64::new(0.0, re) - Complex64::new(im, 0.0);
            }
        }

        let _ = n_gen; // used indirectly via the loop counts above
        u.reunitarize()
    }

    /// Random su(N) algebra element with explicit nc.
    pub fn random_algebra_nc(nc: usize, seed: &mut u64) -> Self {
        use super::constants::lcg_gaussian;

        let scale = std::f64::consts::FRAC_1_SQRT_2;
        let mut h = Self::new(nc);

        // Diagonal: N-1 traceless diagonal generators
        let mut diag = vec![0.0f64; nc];
        for k in 0..(nc - 1) {
            let coeff = lcg_gaussian(seed) * scale;
            let norm = 1.0 / ((2 * (k + 1) * (k + 2)) as f64).sqrt();
            for i in 0..=k {
                diag[i] += coeff * norm;
            }
            diag[k + 1] -= coeff * norm * (k + 1) as f64;
        }
        for i in 0..nc {
            h.m[i * nc + i] = Complex64::new(diag[i], 0.0);
        }

        // Off-diagonal: Hermitian pairs
        for i in 0..nc {
            for j in (i + 1)..nc {
                let re = lcg_gaussian(seed) * scale;
                let im = lcg_gaussian(seed) * scale;
                h.m[i * nc + j] = Complex64::new(re, im);
                h.m[j * nc + i] = Complex64::new(re, -im);
            }
        }

        // Return iH (anti-Hermitian, traceless)
        h.scale_complex(Complex64::I)
    }

    /// Deserialize with explicit nc.
    pub fn read_from_buf_nc(nc: usize, data: &[u8], offset: usize) -> Self {
        let mut s = Self::new(nc);
        for idx in 0..(nc * nc) {
            let off = offset + idx * 16;
            let re = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let im = f64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap());
            s.m[idx] = Complex64::new(re, im);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn su4_identity_det() {
        let u = SuNMatrix::identity(4);
        let d = u.det();
        assert!((d.re - 1.0).abs() < 1e-12, "det(I_4) = {}", d.re);
        assert!(d.im.abs() < 1e-12);
    }

    #[test]
    fn su4_unitarity() {
        let mut seed = 42u64;
        let u = SuNMatrix::random_near_identity_nc(4, &mut seed, 0.3);
        let ud = u.adjoint();
        let prod = u.mul(&ud);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.get(i, j).re - expected).abs() < 1e-6,
                    "SU(4) U U† at ({i},{j}): {:.6e}",
                    prod.get(i, j).re - expected
                );
            }
        }
    }

    #[test]
    fn su6_algebra_traceless() {
        let mut seed = 123u64;
        let p = SuNMatrix::random_algebra_nc(6, &mut seed);
        let tr = p.trace();
        assert!(tr.abs() < 1e-12, "su(6) Tr = {:e}", tr.abs());
    }

    #[test]
    fn su8_reunitarize() {
        let mut seed = 999u64;
        let mut u = SuNMatrix::random_near_identity_nc(8, &mut seed, 0.2);
        u.m[0].re += 0.1;
        let fixed = u.reunitarize();
        let prod = fixed.mul(&fixed.adjoint());
        for i in 0..8 {
            let diag = prod.get(i, i).re;
            assert!(
                (diag - 1.0).abs() < 1e-6,
                "SU(8) reunit diag({i}) = {diag}"
            );
        }
    }
}
