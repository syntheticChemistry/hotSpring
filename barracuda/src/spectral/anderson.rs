// SPDX-License-Identifier: AGPL-3.0-only

//! Anderson localization models: 1D, 2D, 3D discrete Schrödinger operators
//! with random potential, plus transfer-matrix Lyapunov exponent.
//!
//! 1D/2D: all states localized (Abrahams et al. 1979).
//! 3D: genuine metal-insulator transition at W_c ≈ 16.5.

use super::csr::CsrMatrix;

/// Construct the 1D Anderson Hamiltonian on N sites with periodic boundary
/// conditions: H = -Δ + V, where V_i ~ Uniform[-W/2, W/2].
///
/// Returns (diagonal, off_diagonal) for the tridiagonal representation.
/// The hopping is t = 1, so the clean bandwidth is [-2, 2].
///
/// # Provenance
/// Anderson (1958), Phys. Rev. 109, 1492
pub fn anderson_hamiltonian(n: usize, disorder: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);

    let diagonal: Vec<f64> = (0..n).map(|_| disorder * (rng.uniform() - 0.5)).collect();
    let off_diag = vec![-1.0; n - 1];

    (diagonal, off_diag)
}

/// Generate the random potential for the Anderson model (for use with
/// transfer matrix methods that need the raw potential, not the tridiag form).
pub fn anderson_potential(n: usize, disorder: f64, seed: u64) -> Vec<f64> {
    let mut rng = LcgRng::new(seed);
    (0..n).map(|_| disorder * (rng.uniform() - 0.5)).collect()
}

/// Compute the Lyapunov exponent γ(E) via the transfer matrix method.
///
/// For the 1D Schrödinger equation ψ_{n+1} + ψ_{n-1} + V_n ψ_n = E ψ_n,
/// the transfer matrix at site n is:
///   T_n = [[E − V_n, −1], [1, 0]]
///
/// The Lyapunov exponent γ = lim_{N→∞} (1/N) ln ‖T_N ⋯ T_1‖ measures
/// the exponential growth rate. γ > 0 implies localization.
///
/// Uses iterative renormalization to prevent overflow.
///
/// # Known results
/// - Anderson model, E=0, small W: γ(0) ≈ W²/96 (Kappus-Wegner 1981)
/// - Almost-Mathieu, irrational α: γ(E) = max(0, ln|λ|) a.e. (Herman 1983, Avila 2015)
pub fn lyapunov_exponent(potential: &[f64], energy: f64) -> f64 {
    let n = potential.len();
    if n == 0 {
        return 0.0;
    }

    let mut v_prev = 0.0f64;
    let mut v_curr = 1.0f64;
    let mut log_growth = 0.0f64;

    for &v_i in potential {
        let v_next = (energy - v_i) * v_curr - v_prev;
        v_prev = v_curr;
        v_curr = v_next;

        let norm = v_curr.hypot(v_prev);
        if norm > 0.0 {
            log_growth += norm.ln();
            v_curr /= norm;
            v_prev /= norm;
        }
    }

    log_growth / n as f64
}

/// Compute Lyapunov exponent averaged over multiple disorder realizations.
pub fn lyapunov_averaged(
    n_sites: usize,
    disorder: f64,
    energy: f64,
    n_realizations: usize,
    base_seed: u64,
) -> f64 {
    let mut sum = 0.0;
    for r in 0..n_realizations {
        let pot = anderson_potential(n_sites, disorder, base_seed + r as u64 * 1000);
        sum += lyapunov_exponent(&pot, energy);
    }
    sum / n_realizations as f64
}

/// Construct the 2D Anderson Hamiltonian on an Lx × Ly square lattice
/// with open boundary conditions.
///
/// H = -Δ₂D + V, where Δ₂D is the 2D discrete Laplacian (4 nearest
/// neighbors) and V_i ~ Uniform[-W/2, W/2].
///
/// The clean bandwidth is [-4, 4] (hopping t=1, coordination number z=4).
/// With disorder W, spectrum lies in [-4-W/2, 4+W/2].
///
/// Returns a CsrMatrix of dimension N = Lx × Ly.
///
/// # Provenance
/// Abrahams, Anderson, Licciardello, Ramakrishnan (1979) "Scaling Theory
/// of Localization in an Open and Topologically Disordered System"
/// Phys. Rev. Lett. 42, 673
pub fn anderson_2d(lx: usize, ly: usize, disorder: f64, seed: u64) -> CsrMatrix {
    let n = lx * ly;
    let mut rng = LcgRng::new(seed);

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let idx = |ix: usize, iy: usize| -> usize { ix * ly + iy };

    row_ptr.push(0);

    for ix in 0..lx {
        for iy in 0..ly {
            let i = idx(ix, iy);
            let v_i = disorder * (rng.uniform() - 0.5);

            // Collect (column, value) pairs, then sort by column
            let mut entries: Vec<(usize, f64)> = Vec::new();

            // Left neighbor
            if ix > 0 {
                entries.push((idx(ix - 1, iy), -1.0));
            }
            // Down neighbor
            if iy > 0 {
                entries.push((idx(ix, iy - 1), -1.0));
            }
            // Diagonal
            entries.push((i, v_i));
            // Up neighbor
            if iy + 1 < ly {
                entries.push((idx(ix, iy + 1), -1.0));
            }
            // Right neighbor
            if ix + 1 < lx {
                entries.push((idx(ix + 1, iy), -1.0));
            }

            entries.sort_by_key(|&(c, _)| c);
            for (c, v) in entries {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }
    }

    CsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

/// Construct the clean 2D tight-binding Hamiltonian (no disorder).
pub fn clean_2d_lattice(lx: usize, ly: usize) -> CsrMatrix {
    anderson_2d(lx, ly, 0.0, 0)
}

/// Construct the 3D Anderson Hamiltonian on an Lx × Ly × Lz cubic lattice
/// with open boundary conditions.
///
/// H = -Δ₃D + V, where Δ₃D is the 3D discrete Laplacian (6 nearest
/// neighbors) and V_i ~ Uniform[-W/2, W/2].
///
/// The clean bandwidth is [-6, 6] (hopping t=1, coordination number z=6).
/// With disorder W, spectrum lies in [-6-W/2, 6+W/2].
///
/// In 3D, a genuine **Anderson metal-insulator transition** exists at
/// critical disorder W_c ≈ 16.5 (band center, orthogonal class):
/// - W < W_c: extended states near band center, localized at band edges
///   → **mobility edge** separates extended from localized states
/// - W > W_c: all states localized
///
/// This is qualitatively different from 1D/2D where all states are
/// localized for any nonzero disorder.
///
/// Returns a CsrMatrix of dimension N = Lx × Ly × Lz.
///
/// # Provenance
/// Anderson (1958) Phys. Rev. 109, 1492
/// Abrahams, Anderson, Licciardello, Ramakrishnan (1979) Phys. Rev. Lett. 42, 673
/// Slevin & Ohtsuki (1999) Phys. Rev. Lett. 82, 382 [W_c ≈ 16.5]
pub fn anderson_3d(lx: usize, ly: usize, lz: usize, disorder: f64, seed: u64) -> CsrMatrix {
    let n = lx * ly * lz;
    let mut rng = LcgRng::new(seed);

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * ly + iy) * lz + iz };

    row_ptr.push(0);

    for ix in 0..lx {
        for iy in 0..ly {
            for iz in 0..lz {
                let v_i = disorder * (rng.uniform() - 0.5);

                let mut entries: Vec<(usize, f64)> = Vec::new();

                if ix > 0 {
                    entries.push((idx(ix - 1, iy, iz), -1.0));
                }
                if iy > 0 {
                    entries.push((idx(ix, iy - 1, iz), -1.0));
                }
                if iz > 0 {
                    entries.push((idx(ix, iy, iz - 1), -1.0));
                }
                entries.push((idx(ix, iy, iz), v_i));
                if iz + 1 < lz {
                    entries.push((idx(ix, iy, iz + 1), -1.0));
                }
                if iy + 1 < ly {
                    entries.push((idx(ix, iy + 1, iz), -1.0));
                }
                if ix + 1 < lx {
                    entries.push((idx(ix + 1, iy, iz), -1.0));
                }

                entries.sort_by_key(|&(c, _)| c);
                for (c, v) in entries {
                    col_idx.push(c);
                    values.push(v);
                }
                row_ptr.push(col_idx.len());
            }
        }
    }

    CsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

/// Construct the clean 3D tight-binding Hamiltonian (no disorder).
pub fn clean_3d_lattice(l: usize) -> CsrMatrix {
    anderson_3d(l, l, l, 0.0, 0)
}

// ═══════════════════════════════════════════════════════════════════
//  PRNG (LCG for reproducible disorder)
// ═══════════════════════════════════════════════════════════════════

/// LCG RNG for reproducible disorder; used by Anderson models and Lanczos.
pub(crate) struct LcgRng(u64);

impl LcgRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::{lanczos, lanczos_eigenvalues, GOLDEN_RATIO};

    #[test]
    fn lyapunov_positive_anderson() {
        let pot = anderson_potential(10_000, 2.0, 42);
        let gamma = lyapunov_exponent(&pot, 0.0);
        assert!(
            gamma > 0.0,
            "γ should be positive for Anderson, got {gamma}"
        );
    }

    #[test]
    fn lyapunov_herman_formula() {
        // Almost-Mathieu with λ=2, α=golden ratio: γ = ln(2) ≈ 0.693
        let n = 50_000;
        let pot: Vec<f64> = (0..n)
            .map(|i| 2.0 * 2.0 * (std::f64::consts::TAU * GOLDEN_RATIO * i as f64).cos())
            .collect();
        let gamma = lyapunov_exponent(&pot, 0.0);
        let expected = (2.0f64).ln();
        assert!(
            (gamma - expected).abs() < 0.05,
            "Herman: γ={gamma:.4}, expected ln(2)={expected:.4}"
        );
    }

    #[test]
    fn lyapunov_extended_phase() {
        // Almost-Mathieu with λ=0.5: γ should be ~0 (extended)
        let n = 50_000;
        let pot: Vec<f64> = (0..n)
            .map(|i| 2.0 * 0.5 * (std::f64::consts::TAU * GOLDEN_RATIO * i as f64).cos())
            .collect();
        let gamma = lyapunov_exponent(&pot, 0.0);
        assert!(
            gamma.abs() < 0.05,
            "Extended phase: γ={gamma:.4}, expected ~0"
        );
    }

    #[test]
    fn anderson_2d_spectrum_bounds() {
        let l = 15;
        let w = 3.0;
        let mat = anderson_2d(l, l, w, 42);
        let result = lanczos(&mat, l * l, 42);
        let evals = lanczos_eigenvalues(&result);

        let bound = 4.0 + w / 2.0 + 0.1;
        for &ev in &evals {
            assert!(
                ev.abs() <= bound,
                "2D eigenvalue {ev:.4} outside [-{bound:.2}, {bound:.2}]"
            );
        }
    }

    #[test]
    fn clean_2d_bandwidth() {
        let l = 20;
        let mat = clean_2d_lattice(l, l);
        let result = lanczos(&mat, l * l, 42);
        let evals = lanczos_eigenvalues(&result);

        let min_ev = evals[0];
        let max_ev = *evals.last().expect("collection verified non-empty");
        let bandwidth = max_ev - min_ev;

        // Clean 2D: bandwidth should be 8 (from -4 to 4)
        assert!(
            (bandwidth - 8.0).abs() < 0.1,
            "2D bandwidth={bandwidth:.4}, expected ~8.0"
        );
        assert!(min_ev > -4.1, "min eigenvalue {min_ev:.4} should be > -4.1");
        assert!(max_ev < 4.1, "max eigenvalue {max_ev:.4} should be < 4.1");
    }

    #[test]
    fn anderson_2d_nnz_correct() {
        let l = 10;
        let mat = anderson_2d(l, l, 1.0, 42);
        assert_eq!(mat.n, l * l);
        // Interior: 5 entries, edge: 4, corner: 3
        // Total nnz = N + 2*(lx-1)*ly + 2*lx*(ly-1)  (diag + horiz + vert)
        let expected_nnz = l * l + 2 * (l - 1) * l + 2 * l * (l - 1);
        assert_eq!(
            mat.nnz(),
            expected_nnz,
            "nnz={}, expected={expected_nnz}",
            mat.nnz()
        );
    }

    #[test]
    fn anderson_3d_nnz_correct() {
        let l = 6;
        let mat = anderson_3d(l, l, l, 1.0, 42);
        let n = l * l * l;
        assert_eq!(mat.n, n);
        // nnz = N (diag) + 2 * [(l-1)*l*l (x-bonds) + l*(l-1)*l (y-bonds) + l*l*(l-1) (z-bonds)]
        let expected_nnz = n + 2 * ((l - 1) * l * l + l * (l - 1) * l + l * l * (l - 1));
        assert_eq!(
            mat.nnz(),
            expected_nnz,
            "3D nnz={}, expected={expected_nnz}",
            mat.nnz()
        );
    }

    #[test]
    fn clean_3d_bandwidth() {
        let l = 8;
        let mat = clean_3d_lattice(l);
        let result = lanczos(&mat, l * l * l, 42);
        let evals = lanczos_eigenvalues(&result);

        let min_ev = evals[0];
        let max_ev = *evals.last().expect("collection verified non-empty");
        let bandwidth = max_ev - min_ev;

        // Clean 3D with open BCs: exact BW = 12·cos(π/(L+1))
        // L=8 → 12·cos(π/9) ≈ 11.276. Approaches 12 as L→∞.
        let exact_bw = 12.0 * (std::f64::consts::PI / (l as f64 + 1.0)).cos();
        assert!(
            (bandwidth - exact_bw).abs() < 0.1,
            "3D bandwidth={bandwidth:.4}, expected ~{exact_bw:.4}"
        );
    }

    #[test]
    fn anderson_3d_spectrum_bounds() {
        let l = 6;
        let w = 4.0;
        let mat = anderson_3d(l, l, l, w, 42);
        let result = lanczos(&mat, l * l * l, 42);
        let evals = lanczos_eigenvalues(&result);

        let bound = 6.0 + w / 2.0 + 0.1;
        for &ev in &evals {
            assert!(
                ev.abs() <= bound,
                "3D eigenvalue {ev:.4} outside [-{bound:.2}, {bound:.2}]"
            );
        }
    }
}
