// SPDX-License-Identifier: AGPL-3.0-only

//! Sturm bisection eigensolve for symmetric tridiagonal matrices.
//!
//! Counts eigenvalues below a given value using LDLT factorization (Sturm
//! sequence) and finds all eigenvalues via bisection.

/// Count eigenvalues of a symmetric tridiagonal matrix strictly less than λ.
///
/// Uses the LDLT factorization (Sturm sequence): the number of negative
/// pivots equals the number of eigenvalues below λ.
///
/// - `diagonal`: main diagonal d[0..n]
/// - `off_diag`: sub/super-diagonal e[0..n-1]
#[must_use]
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

    let pivot_guard = crate::tolerances::TRIDIAG_STURM_PIVOT_GUARD;
    for i in 1..n {
        let q_safe = if q.abs() < pivot_guard {
            if q >= 0.0 {
                pivot_guard
            } else {
                -pivot_guard
            }
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
#[must_use]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral::anderson_hamiltonian;

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
    fn eigensolve_count_consistency() {
        let (d, e) = anderson_hamiltonian(300, 1.0, 77);
        let evals = find_all_eigenvalues(&d, &e);
        for (k, &ev) in evals.iter().enumerate() {
            let count_below = sturm_count(&d, &e, ev + 1e-8);
            assert!(
                count_below > k,
                "Sturm count at λ={ev:.6}+ε is {count_below}, expected > {k}"
            );
        }
    }
}
