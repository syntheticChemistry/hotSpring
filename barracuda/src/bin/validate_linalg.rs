//! Validate barracuda::ops::linalg functions against known reference values
//!
//! Tests: LU decomposition, QR decomposition, SVD, tridiagonal solver
//! Reference: NumPy/SciPy linear algebra

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  BarraCUDA Linear Algebra Validation");
    println!("  Reference: numpy.linalg / scipy.linalg");
    println!("═══════════════════════════════════════════════════════════\n");

    let mut total = 0;
    let mut passed = 0;

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
                    if (pa[i] - lu_product[i]).abs() > 1e-10 {
                        lu_ok = false;
                    }
                }

                total += 1;
                if lu_ok {
                    passed += 1;
                    println!("  ✅ PA = LU reconstruction");
                } else {
                    println!("  ❌ PA ≠ LU reconstruction");
                }

                // Test determinant: det = 2*3*9 + 1*3*8 + 1*4*7 - 1*3*8 - 2*3*7 - 1*4*9
                // = 54 + 24 + 28 - 24 - 42 - 36 = 4
                let det = lu.det();
                total += 1;
                if (det - 4.0).abs() < 1e-10 {
                    passed += 1;
                    println!("  ✅ det(A) = {:.6} (expected 4.0)", det);
                } else {
                    println!("  ❌ det(A) = {:.6} (expected 4.0)", det);
                }
            }
            Err(e) => {
                println!("  ❌ LU decomposition failed: {}", e);
                total += 2;
            }
        }

        // LU solve: Ax = b
        let b = vec![1.0, 1.0, 1.0];
        match barracuda::ops::linalg::lu_solve(
            &vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
            3,
            &b,
        ) {
            Ok(x) => {
                // Verify Ax ≈ b
                let ax = mat_vec(&vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0], &x, 3);
                let err: f64 = ax.iter().zip(b.iter()).map(|(a, b)| (a - b).abs()).sum();
                total += 1;
                if err < 1e-10 {
                    passed += 1;
                    println!("  ✅ LU solve: ||Ax - b|| = {:.2e}", err);
                } else {
                    println!("  ❌ LU solve: ||Ax - b|| = {:.2e}", err);
                }
            }
            Err(e) => {
                total += 1;
                println!("  ❌ LU solve failed: {}", e);
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
                        if (qtq[i * m + j] - expected).abs() > 1e-10 {
                            ortho_ok = false;
                        }
                    }
                }
                total += 1;
                if ortho_ok {
                    passed += 1;
                    println!("  ✅ Q^T Q = I (orthonormality, {}×{})", m, m);
                } else {
                    println!("  ❌ Q^T Q ≠ I");
                }

                // Verify QR ≈ A (Q is m×m, R is m×n → product is m×n)
                let qr_product = mat_mul_mn(&qr.q, &qr.r, m, m, n);
                let err: f64 = qr_product
                    .iter()
                    .zip(a.iter())
                    .map(|(q, a)| (q - a).abs())
                    .sum();
                total += 1;
                if err < 1e-10 {
                    passed += 1;
                    println!("  ✅ QR = A reconstruction, err = {:.2e}", err);
                } else {
                    println!("  ❌ QR ≠ A, err = {:.2e}", err);
                }
            }
            Err(e) => {
                total += 2;
                println!("  ❌ QR decomposition failed: {}", e);
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
                total += 1;
                if sv.len() >= 2 && (sv[0] - 3.0).abs() < 1e-6 && (sv[1] - 2.0).abs() < 1e-6 {
                    passed += 1;
                    println!("  ✅ SVD singular values: [{:.4}, {:.4}] (expected [3, 2])", sv[0], sv[1]);
                } else {
                    println!("  ❌ SVD singular values: {:?} (expected [3, 2])", sv);
                }

                // Test reconstruction: A = U Σ V^T
                let mut reconstructed = vec![0.0; 4];
                for i in 0..2 {
                    for j in 0..2 {
                        for k in 0..sv.len() {
                            reconstructed[i * 2 + j] +=
                                svd.u[i * 2 + k] * sv[k] * svd.vt[k * 2 + j];
                        }
                    }
                }
                let err: f64 = reconstructed
                    .iter()
                    .zip(a.iter())
                    .map(|(r, a)| (r - a).abs())
                    .sum();
                total += 1;
                if err < 1e-6 {
                    passed += 1;
                    println!("  ✅ UΣVᵀ = A reconstruction, err = {:.2e}", err);
                } else {
                    println!("  ❌ UΣVᵀ ≠ A, err = {:.2e}", err);
                }
            }
            Err(e) => {
                total += 2;
                println!("  ❌ SVD failed: {}", e);
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
                        if (apa[i * 2 + j] - expected).abs() > 1e-6 {
                            ok = false;
                        }
                    }
                }
                total += 1;
                if ok {
                    passed += 1;
                    println!("  ✅ Pseudoinverse: A⁺A ≈ I");
                } else {
                    println!("  ❌ Pseudoinverse: A⁺A ≠ I");
                    println!("      A⁺A = {:?}", apa);
                }
            }
            Err(e) => {
                total += 1;
                println!("  ❌ Pseudoinverse failed: {}", e);
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
                let ax0 = 4.0 * x[0] + 1.0 * x[1];
                let ax1 = 1.0 * x[0] + 4.0 * x[1] + 1.0 * x[2];
                let ax2 = 1.0 * x[1] + 4.0 * x[2];
                let err = (ax0 - 1.0).abs() + (ax1 - 2.0).abs() + (ax2 - 1.0).abs();

                total += 1;
                if err < 1e-10 {
                    passed += 1;
                    println!("  ✅ Thomas algorithm: ||Ax - d|| = {:.2e}", err);
                } else {
                    println!("  ❌ Thomas algorithm: ||Ax - d|| = {:.2e}", err);
                }
            }
            Err(e) => {
                total += 1;
                println!("  ❌ Thomas algorithm failed: {}", e);
            }
        }

        // Larger system (100 points, Laplacian-like)
        let n = 100;
        let a_sub = vec![1.0; n - 1];
        let b_diag = vec![-2.0; n];
        let c_sup = vec![1.0; n - 1];
        let d: Vec<f64> = (0..n).map(|i| if i == n / 2 { -1.0 } else { 0.0 }).collect();

        match barracuda::ops::linalg::tridiagonal::tridiagonal_solve(&a_sub, &b_diag, &c_sup, &d) {
            Ok(x) => {
                // Verify Ax ≈ d
                let mut err = 0.0;
                for i in 0..n {
                    let ax_i = b_diag[i] * x[i]
                        + if i > 0 { a_sub[i - 1] * x[i - 1] } else { 0.0 }
                        + if i < n - 1 { c_sup[i] * x[i + 1] } else { 0.0 };
                    err += (ax_i - d[i]).abs();
                }
                total += 1;
                if err < 1e-8 {
                    passed += 1;
                    println!("  ✅ Tridiag 100pt: ||Ax - d|| = {:.2e}", err);
                } else {
                    println!("  ❌ Tridiag 100pt: ||Ax - d|| = {:.2e}", err);
                }
            }
            Err(e) => {
                total += 1;
                println!("  ❌ Tridiag 100pt failed: {}", e);
            }
        }
    }
    println!();

    // ─── Summary ──────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  Linear Algebra: {}/{} passed", passed, total);
    if passed == total {
        println!("  ✅ ALL TESTS PASSED");
    } else {
        println!("  ❌ {} FAILURES", total - passed);
    }
    println!("═══════════════════════════════════════════════════════════");
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

