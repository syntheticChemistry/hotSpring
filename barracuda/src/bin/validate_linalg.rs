// SPDX-License-Identifier: AGPL-3.0-only

//! Validate `barracuda::ops::linalg` functions against known reference values
//!
//! Tests: LU decomposition, QR decomposition, SVD, tridiagonal solver
//! Reference: NumPy/SciPy linear algebra

use hotspring_barracuda::provenance;
use hotspring_barracuda::tolerances;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  BarraCUDA Linear Algebra Validation");
    println!("  Reference: numpy.linalg / scipy.linalg");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut harness = ValidationHarness::new("linear_algebra");

    // ─── LU Decomposition ─────────────────────────────────────────
    println!("── LU Decomposition ──");
    {
        // Test matrix:
        // A = [[2, 1, 1],
        //      [4, 3, 3],
        //      [8, 7, 9]]
        let a = vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0];
        match barracuda::ops::linalg::lu_decompose(&a, 3) {
            Ok(lu) => {
                // Verify PA = LU
                let l = lu.l();
                let u = lu.u();
                let p = lu.p();

                // Compute PA
                let pa = mat_mul(&p, &a, 3);
                // Compute LU
                let lu_product = mat_mul(&l, &u, 3);

                let mut lu_ok = true;
                for i in 0..9 {
                    if (pa[i] - lu_product[i]).abs() > tolerances::EXACT_F64 {
                        lu_ok = false;
                    }
                }

                if lu_ok {
                    println!("  ✅ PA = LU reconstruction");
                } else {
                    println!("  ❌ PA ≠ LU reconstruction");
                }
                harness.check_bool("PA = LU reconstruction", lu_ok);

                // Test determinant: det = 2*3*9 + 1*3*8 + 1*4*7 - 1*3*8 - 2*3*7 - 1*4*9
                // = 54 + 24 + 28 - 24 - 42 - 36 = 4
                let det = lu.det();
                let det_ok = (det - 4.0).abs() < tolerances::EXACT_F64;
                if det_ok {
                    println!("  ✅ det(A) = {det:.6} (expected 4.0)");
                } else {
                    println!("  ❌ det(A) = {det:.6} (expected 4.0)");
                }
                harness.check_abs("det(A) = 4.0", det, 4.0, tolerances::EXACT_F64);
            }
            Err(e) => {
                println!("  ❌ LU decomposition failed: {e}");
                harness.check_bool("LU PA=LU reconstruction", false);
                harness.check_bool("LU det(A)=4.0", false);
            }
        }

        // LU solve: Ax = b
        let b = vec![1.0, 1.0, 1.0];
        match barracuda::ops::linalg::lu_solve(
            &[2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
            3,
            &b,
        ) {
            Ok(x) => {
                // Verify Ax ≈ b
                let ax = mat_vec(&[2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0], &x, 3);
                let err: f64 = ax.iter().zip(b.iter()).map(|(a, b)| (a - b).abs()).sum();
                if err < tolerances::EXACT_F64 {
                    println!("  ✅ LU solve: ||Ax - b|| = {err:.2e}");
                } else {
                    println!("  ❌ LU solve: ||Ax - b|| = {err:.2e}");
                }
                harness.check_upper("LU solve ||Ax-b||", err, tolerances::EXACT_F64);
            }
            Err(e) => {
                println!("  ❌ LU solve failed: {e}");
                harness.check_bool("LU solve", false);
            }
        }
    }
    println!();

    // ─── QR Decomposition ─────────────────────────────────────────
    println!("── QR Decomposition ──");
    {
        // A = [[1, 1], [1, -1], [0, 1]]  (3×2)
        // Q is m×m = 3×3, R is m×n = 3×2
        let a = vec![1.0, 1.0, 1.0, -1.0, 0.0, 1.0];
        let m = 3;
        let n = 2;
        match barracuda::ops::linalg::qr_decompose(&a, m, n) {
            Ok(qr) => {
                // Verify Q^T Q ≈ I (Q is m×m = 3×3, orthogonal)
                let qt = transpose(&qr.q, m, m);
                let qtq = mat_mul_mn(&qt, &qr.q, m, m, m);

                let mut ortho_ok = true;
                for i in 0..m {
                    for j in 0..m {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        if (qtq[i * m + j] - expected).abs() > tolerances::EXACT_F64 {
                            ortho_ok = false;
                        }
                    }
                }
                if ortho_ok {
                    println!("  ✅ Q^T Q = I (orthonormality, {m}×{m})");
                } else {
                    println!("  ❌ Q^T Q ≠ I");
                }
                harness.check_bool("QR: Q^T Q = I", ortho_ok);

                // Verify QR ≈ A (Q is m×m, R is m×n → product is m×n)
                let qr_product = mat_mul_mn(&qr.q, &qr.r, m, m, n);
                let err: f64 = qr_product
                    .iter()
                    .zip(a.iter())
                    .map(|(q, a)| (q - a).abs())
                    .sum();
                if err < tolerances::EXACT_F64 {
                    println!("  ✅ QR = A reconstruction, err = {err:.2e}");
                } else {
                    println!("  ❌ QR ≠ A, err = {err:.2e}");
                }
                harness.check_upper("QR = A reconstruction", err, tolerances::EXACT_F64);
            }
            Err(e) => {
                println!("  ❌ QR decomposition failed: {e}");
                harness.check_bool("QR: Q^T Q = I", false);
                harness.check_bool("QR = A reconstruction", false);
            }
        }
    }
    println!();

    // ─── SVD ──────────────────────────────────────────────────────
    println!("── SVD ──");
    {
        // A = [[3, 0], [0, 2]] → σ = [3, 2]
        let a = vec![3.0, 0.0, 0.0, 2.0];
        match barracuda::ops::linalg::svd_decompose(&a, 2, 2) {
            Ok(svd) => {
                // Singular values should be [3, 2]
                let sv = &svd.s;
                let sv_ok = sv.len() >= 2
                    && (sv[0] - 3.0).abs() < tolerances::SVD_TOLERANCE
                    && (sv[1] - 2.0).abs() < tolerances::SVD_TOLERANCE;
                if sv_ok {
                    println!(
                        "  ✅ SVD singular values: [{:.4}, {:.4}] (expected [3, 2])",
                        sv[0], sv[1]
                    );
                } else {
                    println!("  ❌ SVD singular values: {sv:?} (expected [3, 2])");
                }
                harness.check_bool("SVD singular values [3,2]", sv_ok);

                // Test reconstruction: A = U Σ V^T
                let mut reconstructed = [0.0; 4];
                for i in 0..2 {
                    for j in 0..2 {
                        for (k, &sv_k) in sv.iter().enumerate() {
                            reconstructed[i * 2 + j] += svd.u[i * 2 + k] * sv_k * svd.vt[k * 2 + j];
                        }
                    }
                }
                let err: f64 = reconstructed
                    .iter()
                    .zip(a.iter())
                    .map(|(r, a)| (r - a).abs())
                    .sum();
                if err < tolerances::SVD_TOLERANCE {
                    println!("  ✅ UΣVᵀ = A reconstruction, err = {err:.2e}");
                } else {
                    println!("  ❌ UΣVᵀ ≠ A, err = {err:.2e}");
                }
                harness.check_upper("SVD UΣVᵀ=A reconstruction", err, tolerances::SVD_TOLERANCE);
            }
            Err(e) => {
                println!("  ❌ SVD failed: {e}");
                harness.check_bool("SVD singular values", false);
                harness.check_bool("SVD reconstruction", false);
            }
        }

        // Pseudoinverse test
        let a2 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3×2
        match barracuda::ops::linalg::svd_pinv(&a2, 3, 2, 1e-10) {
            Ok(pinv) => {
                // A⁺ A should ≈ I (2×2)
                let apa = mat_mul_mn(&pinv, &a2, 2, 3, 2);
                let mut ok = true;
                for i in 0..2 {
                    for j in 0..2 {
                        let expected = if i == j { 1.0 } else { 0.0 };
                        if (apa[i * 2 + j] - expected).abs() > tolerances::SVD_TOLERANCE {
                            ok = false;
                        }
                    }
                }
                if ok {
                    println!("  ✅ Pseudoinverse: A⁺A ≈ I");
                } else {
                    println!("  ❌ Pseudoinverse: A⁺A ≠ I");
                    println!("      A⁺A = {apa:?}");
                }
                harness.check_bool("SVD pseudoinverse A⁺A≈I", ok);
            }
            Err(e) => {
                println!("  ❌ Pseudoinverse failed: {e}");
                harness.check_bool("SVD pseudoinverse", false);
            }
        }
    }
    println!();

    // ─── Tridiagonal solver ───────────────────────────────────────
    println!("── Tridiagonal Solver (Thomas Algorithm) ──");
    {
        // System: [4 1 0; 1 4 1; 0 1 4] x = [1; 2; 1]
        let a_sub = vec![1.0, 1.0];
        let b_diag = vec![4.0, 4.0, 4.0];
        let c_sup = vec![1.0, 1.0];
        let d = vec![1.0, 2.0, 1.0];

        match barracuda::ops::linalg::tridiagonal::tridiagonal_solve(&a_sub, &b_diag, &c_sup, &d) {
            Ok(x) => {
                // Verify Ax ≈ d
                let ax0 = 4.0f64.mul_add(x[0], 1.0 * x[1]);
                let ax1 = 1.0f64.mul_add(x[2], 1.0f64.mul_add(x[0], 4.0 * x[1]));
                let ax2 = 1.0f64.mul_add(x[1], 4.0 * x[2]);
                let err = (ax0 - 1.0).abs() + (ax1 - 2.0).abs() + (ax2 - 1.0).abs();

                if err < tolerances::EXACT_F64 {
                    println!("  ✅ Thomas algorithm: ||Ax - d|| = {err:.2e}");
                } else {
                    println!("  ❌ Thomas algorithm: ||Ax - d|| = {err:.2e}");
                }
                harness.check_upper("Thomas algorithm ||Ax-d||", err, tolerances::EXACT_F64);
            }
            Err(e) => {
                println!("  ❌ Thomas algorithm failed: {e}");
                harness.check_bool("Thomas algorithm", false);
            }
        }

        // Larger system (100 points, Laplacian-like)
        let n = 100;
        let a_sub = vec![1.0; n - 1];
        let b_diag = vec![-2.0; n];
        let c_sup = vec![1.0; n - 1];
        let d: Vec<f64> = (0..n)
            .map(|i| if i == n / 2 { -1.0 } else { 0.0 })
            .collect();

        match barracuda::ops::linalg::tridiagonal::tridiagonal_solve(&a_sub, &b_diag, &c_sup, &d) {
            Ok(x) => {
                // Verify Ax ≈ d
                let mut err = 0.0;
                for i in 0..n {
                    let ax_i = b_diag[i]
                        .mul_add(x[i], if i > 0 { a_sub[i - 1] * x[i - 1] } else { 0.0 })
                        + if i < n - 1 { c_sup[i] * x[i + 1] } else { 0.0 };
                    err += (ax_i - d[i]).abs();
                }
                if err < tolerances::ITERATIVE_F64 {
                    println!("  ✅ Tridiag 100pt: ||Ax - d|| = {err:.2e}");
                } else {
                    println!("  ❌ Tridiag 100pt: ||Ax - d|| = {err:.2e}");
                }
                harness.check_upper("Tridiag 100pt ||Ax-d||", err, tolerances::ITERATIVE_F64);
            }
            Err(e) => {
                println!("  ❌ Tridiag 100pt failed: {e}");
                harness.check_bool("Tridiag 100pt", false);
            }
        }
    }
    println!();

    // ─── Summary (with exit code) ──────────────────────────────────
    println!("\n  Reference: {}", provenance::LINALG_REFS);
    harness.finish();
}

// Helper: n×n matrix multiply
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    mat_mul_mn(a, b, n, n, n)
}

// Helper: general matrix multiply (m×k) × (k×n) → (m×n)
fn mat_mul_mn(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
    c
}

// Helper: matrix-vector multiply (n×n) × (n) → (n)
fn mat_vec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            y[i] += a[i * n + j] * x[j];
        }
    }
    y
}

// Helper: transpose m×n → n×m
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}
