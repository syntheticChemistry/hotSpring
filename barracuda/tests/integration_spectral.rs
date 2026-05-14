// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration tests: spectral theory (Lanczos, CSR SpMV, DOS), CPU only.

use hotspring_barracuda::spectral::{
    SpectralCsrMatrix, anderson_hamiltonian, find_all_eigenvalues, lanczos, lanczos_eigenvalues,
};

fn symmetric_tridiag_to_csr(diag: &[f64], off: &[f64]) -> SpectralCsrMatrix {
    let n = diag.len();
    let mut row_ptr = vec![0usize];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        if i > 0 {
            col_idx.push(i - 1);
            values.push(off[i - 1]);
        }
        col_idx.push(i);
        values.push(diag[i]);
        if i < n - 1 {
            col_idx.push(i + 1);
            values.push(off[i]);
        }
        row_ptr.push(col_idx.len());
    }
    SpectralCsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

#[test]
fn lanczos_recovers_known_eigenvalues_on_two_by_two() {
    let csr = SpectralCsrMatrix {
        n: 2,
        row_ptr: vec![0, 2, 4],
        col_idx: vec![0, 1, 0, 1],
        values: vec![2.0, 1.0, 1.0, 3.0],
    };
    let tri = lanczos(&csr, 2, 7);
    let mut got = lanczos_eigenvalues(&tri);
    got.sort_by(f64::total_cmp);

    let trace = 5.0_f64;
    let det = 5.0_f64;
    let disc = (trace * trace - 4.0 * det).sqrt();
    let want_min = 0.5 * (trace - disc);
    let want_max = 0.5 * (trace + disc);

    assert_eq!(got.len(), 2);
    assert!((got[0] - want_min).abs() < 1e-10, "λ_min mismatch");
    assert!((got[1] - want_max).abs() < 1e-10, "λ_max mismatch");
}

#[test]
fn csr_spmv_matches_dense_symmetric_multiply() {
    let csr = symmetric_tridiag_to_csr(&[4.0, 4.0, 4.0], &[-1.0, -1.0]);
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![0.0; 3];
    csr.spmv(&x, &mut y);

    let dense_ax = |i: usize| -> f64 {
        let mut s = 4.0 * x[i];
        if i > 0 {
            s += -x[i - 1];
        }
        if i + 1 < 3 {
            s += -x[i + 1];
        }
        s
    };
    for (i, &yi) in y.iter().enumerate().take(3) {
        assert!(
            (yi - dense_ax(i)).abs() < 1e-14,
            "row {i}: SpMV {yi} vs dense {}",
            dense_ax(i)
        );
    }
}

/// Normalized histogram masses sum to 1; ∫ρ(E)dE with ρᵢ = massᵢ/w over equal bins equals Σᵢ ρᵢ w = 1.
fn assert_dos_histogram_normalizes(evals: &[f64], n_bins: usize) {
    let n_tot = evals.len() as f64;
    let min = evals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).abs().max(1e-15);
    let w = range / n_bins as f64;
    let mut mass = vec![0.0_f64; n_bins];
    for &e in evals {
        let idx = (((e - min) / w).floor() as isize).clamp(0, n_bins as isize - 1) as usize;
        mass[idx] += 1.0 / n_tot;
    }
    let sum_mass: f64 = mass.iter().sum();
    let integral: f64 = mass.iter().map(|m| (m / w) * w).sum();
    assert!((sum_mass - 1.0).abs() < 1e-10);
    assert!((integral - 1.0).abs() < 1e-10);
}

#[test]
fn density_of_states_histogram_normalizes_and_integrates() {
    let n = 24usize;
    let (d, e) = anderson_hamiltonian(n, 0.0, 99);
    let exact = find_all_eigenvalues(&d, &e);
    let csr = symmetric_tridiag_to_csr(&d, &e);
    let tri = lanczos(&csr, n, 42);
    let mut l_vals = lanczos_eigenvalues(&tri);
    l_vals.sort_by(f64::total_cmp);

    for i in 0..n {
        assert!(
            (l_vals[i] - exact[i]).abs() < 1e-6,
            "Lanczos eigenvalue {i}: {} vs exact {}",
            l_vals[i],
            exact[i]
        );
    }

    assert_dos_histogram_normalizes(&exact, 12);
}
