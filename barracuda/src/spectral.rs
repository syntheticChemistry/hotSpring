// SPDX-License-Identifier: AGPL-3.0-only

//! Spectral theory for discrete Schrödinger operators.
//!
//! Implements lattice Hamiltonians and spectral analysis tools for the
//! Kachkovskiy extension (spectral theory / transport):
//!
//! - **CsrMatrix + SpMV**: sparse matrix-vector product (P1 GPU primitive)
//! - **Lanczos eigensolve**: Krylov tridiagonalization with full reorthogonalization
//! - **Anderson model**: random potential in 1D, 2D, and 3D
//!   - 1D/2D: all states localized (Abrahams et al. 1979)
//!   - 3D: genuine metal-insulator transition with mobility edge (W_c ≈ 16.5)
//! - **Almost-Mathieu operator**: quasiperiodic potential, Aubry-André transition
//! - **Transfer matrix**: Lyapunov exponent computation
//! - **Tridiagonal eigensolve**: Sturm bisection for all eigenvalues
//! - **Level statistics**: spacing ratio for localization diagnostics
//!
//! # Physics
//!
//! The 1D discrete Schrödinger equation on ℤ:
//!   ψ_{n+1} + ψ_{n-1} + V_n ψ_n = E ψ_n
//!
//! is equivalent to the eigenvalue problem for the tridiagonal matrix
//! H with diagonal V_i and off-diagonal −1. The spectral properties of H
//! (eigenvalues, eigenvectors, Lyapunov exponent) determine transport:
//! extended states → metallic, localized states → insulating.
//!
//! # Provenance
//!
//! - Anderson (1958) "Absence of diffusion in certain random lattices"
//! - Aubry & André (1980) "Analyticity breaking and Anderson localization"
//! - Jitomirskaya (1999) "Metal-insulator transition for the almost Mathieu operator"
//! - Herman (1983) "Une méthode pour minorer les exposants de Lyapunov"
//! - Avila (2015) "Global theory of one-frequency Schrödinger operators" (Fields Medal)
//! - Kappus & Wegner (1981) "Anomaly in the band centre of the 1D Anderson model"

// ═══════════════════════════════════════════════════════════════════
//  Tridiagonal eigenvalue solver (Sturm bisection)
// ═══════════════════════════════════════════════════════════════════

/// Count eigenvalues of a symmetric tridiagonal matrix strictly less than λ.
///
/// Uses the LDLT factorization (Sturm sequence): the number of negative
/// pivots equals the number of eigenvalues below λ.
///
/// - `diagonal`: main diagonal d[0..n]
/// - `off_diag`: sub/super-diagonal e[0..n-1]
pub fn sturm_count(diagonal: &[f64], off_diag: &[f64], lambda: f64) -> usize {
    let n = diagonal.len();
    if n == 0 {
        return 0;
    }

    let mut count = 0;
    let mut q = diagonal[0] - lambda;
    if q < 0.0 {
        count += 1;
    }

    for i in 1..n {
        let q_safe = if q.abs() < 1e-300 {
            if q >= 0.0 { 1e-300 } else { -1e-300 }
        } else {
            q
        };
        q = (diagonal[i] - lambda) - off_diag[i - 1] * off_diag[i - 1] / q_safe;
        if q < 0.0 {
            count += 1;
        }
    }
    count
}

/// Find all eigenvalues of a symmetric tridiagonal matrix via Sturm bisection.
///
/// Returns eigenvalues sorted in ascending order. Complexity: O(N² log(1/ε)).
/// Exact to machine precision for well-separated eigenvalues.
pub fn find_all_eigenvalues(diagonal: &[f64], off_diag: &[f64]) -> Vec<f64> {
    let n = diagonal.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![diagonal[0]];
    }

    // Gershgorin bounds
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for i in 0..n {
        let e_left = if i > 0 { off_diag[i - 1].abs() } else { 0.0 };
        let e_right = if i < n - 1 { off_diag[i].abs() } else { 0.0 };
        lo = lo.min(diagonal[i] - e_left - e_right);
        hi = hi.max(diagonal[i] + e_left + e_right);
    }
    lo -= 1.0;
    hi += 1.0;

    let mut eigenvalues = Vec::with_capacity(n);
    for k in 0..n {
        let mut a = lo;
        let mut b = hi;
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if (b - a) < 2.0 * f64::EPSILON * mid.abs().max(1.0) {
                break;
            }
            if sturm_count(diagonal, off_diag, mid) <= k {
                a = mid;
            } else {
                b = mid;
            }
        }
        eigenvalues.push(0.5 * (a + b));
    }
    eigenvalues
}

// ═══════════════════════════════════════════════════════════════════
//  1D Anderson model
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
//  Almost-Mathieu (Harper) operator
// ═══════════════════════════════════════════════════════════════════

/// Construct the almost-Mathieu (Harper) operator on N sites:
///   H ψ_n = ψ_{n+1} + ψ_{n-1} + 2λ cos(2παn + θ) ψ_n
///
/// Returns (diagonal, off_diagonal) for the tridiagonal representation.
///
/// At the Aubry-André self-dual point λ = 1, the operator undergoes a
/// metal-insulator transition:
/// - λ < 1: absolutely continuous spectrum (extended states)
/// - λ > 1: pure point spectrum (localized states, Anderson-like)
/// - λ = 1: singular continuous spectrum (critical)
///
/// # Provenance
/// - Aubry & André (1980)
/// - Harper (1955), Proc. Phys. Soc. London A 68, 874
/// - Hofstadter (1976), Phys. Rev. B 14, 2239 (butterfly)
pub fn almost_mathieu_hamiltonian(
    n: usize,
    lambda: f64,
    alpha: f64,
    theta: f64,
) -> (Vec<f64>, Vec<f64>) {
    let diagonal: Vec<f64> = (0..n)
        .map(|i| 2.0 * lambda * (std::f64::consts::TAU * alpha * i as f64 + theta).cos())
        .collect();
    let off_diag = vec![1.0; n - 1];

    (diagonal, off_diag)
}

/// The golden ratio (√5 − 1)/2 — the canonical choice of irrational
/// frequency for the almost-Mathieu operator.
pub const GOLDEN_RATIO: f64 = 0.618_033_988_749_894_9;

// ═══════════════════════════════════════════════════════════════════
//  Transfer matrix Lyapunov exponent
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
//  Level spacing statistics
// ═══════════════════════════════════════════════════════════════════

/// Compute the mean level spacing ratio ⟨r⟩ from sorted eigenvalues.
///
/// r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
/// where s_i = λ_{i+1} − λ_i.
///
/// Known values:
/// - Poisson (localized): ⟨r⟩ = 2 ln 2 − 1 ≈ 0.3863
/// - GOE (extended + time-reversal): ⟨r⟩ ≈ 0.5307
///
/// # Provenance
/// Oganesyan & Huse (2007), Phys. Rev. B 75, 155111
/// Atas et al. (2013), Phys. Rev. Lett. 110, 084101
pub fn level_spacing_ratio(eigenvalues: &[f64]) -> f64 {
    let n = eigenvalues.len();
    if n < 3 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n - 2 {
        let s1 = eigenvalues[i + 1] - eigenvalues[i];
        let s2 = eigenvalues[i + 2] - eigenvalues[i + 1];
        if s1 > 0.0 && s2 > 0.0 {
            let r = s1.min(s2) / s1.max(s2);
            sum += r;
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Poisson level spacing ratio (localized states).
pub const POISSON_R: f64 = 0.386_294_361_119_890_6; // 2 ln 2 - 1

/// GOE level spacing ratio (extended states with time-reversal symmetry).
pub const GOE_R: f64 = 0.5307;

// ═══════════════════════════════════════════════════════════════════
//  Compressed Sparse Row (CSR) matrix + SpMV
// ═══════════════════════════════════════════════════════════════════

/// Sparse symmetric matrix in Compressed Sparse Row format.
///
/// This is the standard format for GPU SpMV kernels (P1 for Kachkovskiy GPU
/// promotion). The CSR layout maps directly to a WGSL compute shader.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Sparse matrix-vector product: y = A * x.
    ///
    /// This is the P1 primitive for GPU promotion — the inner loop of Lanczos.
    /// CPU version here; GPU version will be a WGSL compute shader.
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let mut sum = 0.0;
            for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[j] * x[self.col_idx[j]];
            }
            y[i] = sum;
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

// ═══════════════════════════════════════════════════════════════════
//  GPU WGSL shader: CSR SpMV (absorption-ready for toadstool/barracuda)
// ═══════════════════════════════════════════════════════════════════

/// WGSL compute shader for CSR sparse matrix-vector product y = A*x (f64).
///
/// Direct GPU port of [`CsrMatrix::spmv()`]: one thread per matrix row,
/// f64 multiply-accumulate in the inner loop. The binding layout is
/// documented for direct absorption by toadstool/barracuda.
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | uniform | `{ n: u32, nnz: u32, pad: u32, pad: u32 }` |
/// | 1 | storage, read | `row_ptr: array<u32>` (n+1 entries) |
/// | 2 | storage, read | `col_idx: array<u32>` (nnz entries) |
/// | 3 | storage, read | `values: array<f64>` (nnz entries) |
/// | 4 | storage, read | `x: array<f64>` (n entries, input) |
/// | 5 | storage, read_write | `y: array<f64>` (n entries, output) |
///
/// ## Dispatch
///
/// `ceil(n / 64)` workgroups of 64 threads.
///
/// ## Provenance
///
/// GPU promotion of CPU [`CsrMatrix::spmv()`] for Kachkovskiy spectral
/// theory GPU Lanczos and Bazavov lattice QCD GPU Dirac.
pub const WGSL_SPMV_CSR_F64: &str = r"
struct Params {
    n: u32,
    nnz: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx: array<u32>;
@group(0) @binding(3) var<storage, read> vals: array<f64>;
@group(0) @binding(4) var<storage, read> x_vec: array<f64>;
@group(0) @binding(5) var<storage, read_write> y_vec: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.n {
        return;
    }

    let start = row_ptr[row];
    let end = row_ptr[row + 1u];

    var sum: f64 = f64(0.0);
    for (var j = start; j < end; j = j + 1u) {
        sum = sum + vals[j] * x_vec[col_idx[j]];
    }

    y_vec[row] = sum;
}
";

// ═══════════════════════════════════════════════════════════════════
//  Lanczos eigensolve
// ═══════════════════════════════════════════════════════════════════

/// Result of the Lanczos algorithm: a tridiagonal representation of the
/// original matrix restricted to the Krylov subspace.
pub struct LanczosTridiag {
    /// Diagonal elements α_j = ⟨v_j, A v_j⟩
    pub alpha: Vec<f64>,
    /// Off-diagonal elements β_j = ‖w_j‖
    pub beta: Vec<f64>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
}

/// Lanczos tridiagonalization with full reorthogonalization.
///
/// Builds an m-step Krylov subspace for the sparse symmetric matrix A.
/// The eigenvalues of the resulting tridiagonal matrix approximate the
/// eigenvalues of A, with extremal eigenvalues converging first.
///
/// With full reorthogonalization and m = n, the tridiagonal eigenvalues
/// are the exact eigenvalues of A (up to machine precision).
///
/// # Arguments
/// - `matrix`: symmetric sparse matrix in CSR format
/// - `max_iter`: maximum Lanczos iterations (cap at matrix dimension)
/// - `seed`: PRNG seed for initial random vector
///
/// # Provenance
/// Lanczos (1950), J. Res. Nat. Bur. Standards 45, 255
pub fn lanczos(matrix: &CsrMatrix, max_iter: usize, seed: u64) -> LanczosTridiag {
    let n = matrix.n;
    let m = max_iter.min(n);

    let mut rng = LcgRng::new(seed);

    // Random starting vector, normalized
    let mut v: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let norm = dot(&v, &v).sqrt();
    for x in &mut v {
        *x /= norm;
    }

    let mut alpha = Vec::with_capacity(m);
    let mut beta = Vec::with_capacity(m);

    // Store all Lanczos vectors for reorthogonalization
    let mut vecs: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    vecs.push(v.clone());

    let mut v_prev = vec![0.0; n];
    let mut beta_prev = 0.0;
    let mut w = vec![0.0; n];

    for j in 0..m {
        // w = A * v_j
        matrix.spmv(&v, &mut w);

        // w = w - β_j * v_{j-1}
        if j > 0 {
            for i in 0..n {
                w[i] -= beta_prev * v_prev[i];
            }
        }

        // α_j = ⟨w, v_j⟩
        let a_j = dot(&w, &v);
        alpha.push(a_j);

        // w = w - α_j * v_j
        for i in 0..n {
            w[i] -= a_j * v[i];
        }

        // Full reorthogonalization (Gram-Schmidt against all previous vectors)
        for prev in &vecs {
            let proj = dot(&w, prev);
            for i in 0..n {
                w[i] -= proj * prev[i];
            }
        }

        // β_{j+1} = ‖w‖
        let b_next = dot(&w, &w).sqrt();

        if b_next < 1e-14 {
            // Invariant subspace found — Lanczos has converged
            beta.push(0.0);
            break;
        }

        beta.push(b_next);

        // v_{j+1} = w / β_{j+1}
        v_prev.copy_from_slice(&v);
        beta_prev = b_next;
        for i in 0..n {
            v[i] = w[i] / b_next;
        }
        vecs.push(v.clone());
    }

    LanczosTridiag {
        iterations: alpha.len(),
        alpha,
        beta,
    }
}

/// Extract eigenvalues from a Lanczos tridiagonal via Sturm bisection.
pub fn lanczos_eigenvalues(result: &LanczosTridiag) -> Vec<f64> {
    let m = result.iterations;
    if m == 0 {
        return Vec::new();
    }

    let off_diag: Vec<f64> = result.beta[..m.saturating_sub(1)].to_vec();
    find_all_eigenvalues(&result.alpha, &off_diag)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ═══════════════════════════════════════════════════════════════════
//  2D Anderson model
// ═══════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════
//  3D Anderson model
// ═══════════════════════════════════════════════════════════════════

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
//  Hofstadter butterfly (spectral topology)
// ═══════════════════════════════════════════════════════════════════

/// Detect spectral bands from sorted eigenvalues.
///
/// Groups eigenvalues into bands separated by gaps. A "gap" is defined as a
/// spacing exceeding `gap_factor` times the median spacing. Returns a vector
/// of (band_min, band_max) pairs.
pub fn detect_bands(eigenvalues: &[f64], gap_factor: f64) -> Vec<(f64, f64)> {
    if eigenvalues.len() < 2 {
        if eigenvalues.len() == 1 {
            return vec![(eigenvalues[0], eigenvalues[0])];
        }
        return Vec::new();
    }

    let mut spacings: Vec<f64> = eigenvalues.windows(2).map(|w| w[1] - w[0]).collect();
    spacings.sort_by(f64::total_cmp);
    let median = spacings[spacings.len() / 2];

    let threshold = median * gap_factor;
    let mut bands = Vec::new();
    let mut band_start = eigenvalues[0];

    for w in eigenvalues.windows(2) {
        if w[1] - w[0] > threshold {
            bands.push((band_start, w[0]));
            band_start = w[1];
        }
    }
    bands.push((band_start, *eigenvalues.last().unwrap()));

    bands
}

/// Compute the Hofstadter butterfly: spectrum of the almost-Mathieu operator
/// as a function of magnetic flux α.
///
/// For each rational flux α = p/q with q ≤ q_max and gcd(p,q) = 1,
/// computes all eigenvalues of a large almost-Mathieu operator. The
/// resulting (α, E) point cloud forms the Hofstadter butterfly — a fractal
/// spectrum that is the canonical example of spectral topology.
///
/// At λ = 1 (critical coupling), the spectrum at irrational α is a Cantor
/// set of measure zero (the "Ten Martini Problem", proved by Avila &
/// Jitomirskaya 2009).
///
/// Returns: Vec of (alpha, eigenvalues) pairs.
///
/// # Provenance
/// Hofstadter (1976) Phys. Rev. B 14, 2239
/// Avila & Jitomirskaya (2009) Ann. Math. 170, 303
pub fn hofstadter_butterfly(
    q_max: usize,
    lambda: f64,
    n_sites: usize,
) -> Vec<(f64, Vec<f64>)> {
    let mut results = Vec::new();

    for q in 1..=q_max {
        for p in 1..q {
            if gcd(p, q) != 1 {
                continue;
            }
            let alpha = p as f64 / q as f64;
            let (d, e) = almost_mathieu_hamiltonian(n_sites, lambda, alpha, 0.0);
            let evals = find_all_eigenvalues(&d, &e);
            results.push((alpha, evals));
        }
    }

    results
}

/// Greatest common divisor (Euclid's algorithm).
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ═══════════════════════════════════════════════════════════════════
//  PRNG (LCG for reproducible disorder)
// ═══════════════════════════════════════════════════════════════════

struct LcgRng(u64);

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sturm_count_identity_2x2() {
        // Matrix: [[1, -1], [-1, 3]] → eigenvalues ≈ 0.382, 3.618
        let d = [1.0, 3.0];
        let e = [-1.0];
        assert_eq!(sturm_count(&d, &e, 0.0), 0);
        assert_eq!(sturm_count(&d, &e, 1.0), 1);
        assert_eq!(sturm_count(&d, &e, 4.0), 2);
    }

    #[test]
    fn eigenvalues_clean_chain() {
        // Clean tight-binding chain: d_i = 0, e_i = -1
        // Eigenvalues: 2 cos(kπ/(N+1)) for k = 1..N
        let n = 50;
        let d = vec![0.0; n];
        let e = vec![-1.0; n - 1];
        let evals = find_all_eigenvalues(&d, &e);

        assert_eq!(evals.len(), n);

        for k in 1..=n {
            let exact = 2.0 * (k as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            let closest = evals
                .iter()
                .map(|&ev| (ev - exact).abs())
                .fold(f64::MAX, f64::min);
            assert!(
                closest < 1e-10,
                "k={k}, exact={exact:.6}, closest error={closest:.2e}"
            );
        }
    }

    #[test]
    fn eigenvalues_sorted() {
        let (d, e) = anderson_hamiltonian(200, 2.0, 42);
        let evals = find_all_eigenvalues(&d, &e);
        for i in 1..evals.len() {
            assert!(
                evals[i] >= evals[i - 1] - 1e-12,
                "eigenvalues not sorted at index {i}"
            );
        }
    }

    #[test]
    fn anderson_spectrum_in_gershgorin() {
        let w = 3.0;
        let (d, e) = anderson_hamiltonian(500, w, 99);
        let evals = find_all_eigenvalues(&d, &e);
        let lo = -2.0 - w / 2.0 - 0.01;
        let hi = 2.0 + w / 2.0 + 0.01;
        for &ev in &evals {
            assert!(
                ev >= lo && ev <= hi,
                "eigenvalue {ev:.4} outside [{lo:.4}, {hi:.4}]"
            );
        }
    }

    #[test]
    fn lyapunov_positive_anderson() {
        let pot = anderson_potential(10_000, 2.0, 42);
        let gamma = lyapunov_exponent(&pot, 0.0);
        assert!(gamma > 0.0, "γ should be positive for Anderson, got {gamma}");
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
    fn level_spacing_poisson() {
        // Strong Anderson disorder → Poisson statistics
        let (d, e) = anderson_hamiltonian(1000, 8.0, 42);
        let evals = find_all_eigenvalues(&d, &e);
        let r = level_spacing_ratio(&evals);
        assert!(
            (r - POISSON_R).abs() < 0.05,
            "Strong disorder: r={r:.4}, expected Poisson={POISSON_R:.4}"
        );
    }

    #[test]
    fn almost_mathieu_spectrum_bounds() {
        let lambda = 1.5;
        let n = 500;
        let (d, e) = almost_mathieu_hamiltonian(n, lambda, GOLDEN_RATIO, 0.0);
        let evals = find_all_eigenvalues(&d, &e);
        let bound = 2.0 + 2.0 * lambda + 0.01;
        for &ev in &evals {
            assert!(
                ev.abs() <= bound,
                "eigenvalue {ev:.4} outside [-{bound:.2}, {bound:.2}]"
            );
        }
    }

    #[test]
    fn eigensolve_count_consistency() {
        let (d, e) = anderson_hamiltonian(300, 1.0, 77);
        let evals = find_all_eigenvalues(&d, &e);
        for (k, &ev) in evals.iter().enumerate() {
            let count_below = sturm_count(&d, &e, ev + 1e-8);
            assert!(
                count_below >= k + 1,
                "Sturm count at λ={ev:.6}+ε is {count_below}, expected ≥ {k_plus}",
                k_plus = k + 1
            );
        }
    }

    #[test]
    fn csr_spmv_identity() {
        // 3×3 identity in CSR
        let mat = CsrMatrix {
            n: 3,
            row_ptr: vec![0, 1, 2, 3],
            col_idx: vec![0, 1, 2],
            values: vec![1.0, 1.0, 1.0],
        };
        let x = vec![3.0, 5.0, 7.0];
        let mut y = vec![0.0; 3];
        mat.spmv(&x, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!((y[1] - 5.0).abs() < 1e-14);
        assert!((y[2] - 7.0).abs() < 1e-14);
    }

    #[test]
    fn csr_spmv_tridiag() {
        // 4×4 tridiagonal: d=2, e=-1 → same as 1D tight-binding with V=2
        let mat = CsrMatrix {
            n: 4,
            row_ptr: vec![0, 2, 5, 8, 10],
            col_idx: vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            values: vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        };
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let mut y = vec![0.0; 4];
        mat.spmv(&x, &mut y);
        assert!((y[0] - 2.0).abs() < 1e-14);
        assert!((y[1] - -1.0).abs() < 1e-14);
        assert!((y[2] - 0.0).abs() < 1e-14);
        assert!((y[3] - 0.0).abs() < 1e-14);
    }

    #[test]
    fn lanczos_vs_sturm_1d() {
        // 1D Anderson: Lanczos should recover the same eigenvalues as Sturm
        let n = 100;
        let (d, e) = anderson_hamiltonian(n, 2.0, 42);

        // Build 1D Anderson as CSR for Lanczos
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_idx.push(i - 1);
                values.push(e[i - 1]);
            }
            col_idx.push(i);
            values.push(d[i]);
            if i < n - 1 {
                col_idx.push(i + 1);
                values.push(e[i]);
            }
            row_ptr.push(col_idx.len());
        }
        let csr = CsrMatrix {
            n,
            row_ptr,
            col_idx,
            values,
        };

        let sturm_evals = find_all_eigenvalues(&d, &e);
        let lanczos_result = lanczos(&csr, n, 42);
        let lanczos_evals = lanczos_eigenvalues(&lanczos_result);

        // Compare extremal eigenvalues
        let sturm_min = sturm_evals[0];
        let sturm_max = sturm_evals[n - 1];
        let lanczos_min = lanczos_evals[0];
        let lanczos_max = *lanczos_evals.last().unwrap();

        assert!(
            (sturm_min - lanczos_min).abs() < 1e-8,
            "min: Sturm={sturm_min:.8}, Lanczos={lanczos_min:.8}"
        );
        assert!(
            (sturm_max - lanczos_max).abs() < 1e-8,
            "max: Sturm={sturm_max:.8}, Lanczos={lanczos_max:.8}"
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
        let max_ev = *evals.last().unwrap();
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
        let expected_nnz =
            n + 2 * ((l - 1) * l * l + l * (l - 1) * l + l * l * (l - 1));
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
        let max_ev = *evals.last().unwrap();
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
