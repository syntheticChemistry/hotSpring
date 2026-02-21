// SPDX-License-Identifier: AGPL-3.0-only

//! Lanczos + SpMV + 2D Anderson Validation — Kachkovskiy Extension (Tier 2)
//!
//! Validates the P1 primitives for GPU promotion of spectral theory:
//!
//! **SpMV (Sparse Matrix-Vector Product)**:
//! - CSR format correctness against dense reference
//! - Foundation for all iterative eigensolvers on GPU
//!
//! **Lanczos eigensolve**:
//! - Cross-validates against Sturm bisection on 1D Anderson
//! - Full spectrum recovery with reorthogonalization
//! - Eigenvalue convergence with iteration count
//!
//! **2D Anderson model**:
//! - Spectrum bounds [-4-W/2, 4+W/2] (coordination z=4)
//! - Clean bandwidth = 8 (tight-binding on square lattice)
//! - Level statistics: GOE (weak disorder) → Poisson (strong disorder)
//!
//! These extend the Kachkovskiy foundation from 1D tridiagonal operators to
//! general sparse Hamiltonians, enabling 2D/3D Anderson models and ultimately
//! Hofstadter-butterfly spectral topology.
//!
//! # Provenance
//!
//! Lanczos (1950) J. Res. Nat. Bur. Standards 45, 255
//! Abrahams, Anderson, Licciardello, Ramakrishnan (1979) Phys. Rev. Lett. 42, 673
//! Oganesyan & Huse (2007) Phys. Rev. B 75, 155111

use hotspring_barracuda::spectral;
use hotspring_barracuda::spectral::CsrMatrix;
use hotspring_barracuda::validation::ValidationHarness;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Lanczos + SpMV + 2D Anderson — Kachkovskiy Tier 2         ║");
    println!("║  P1 primitives for GPU promotion of spectral theory        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut harness = ValidationHarness::new("lanczos_2d_anderson");

    check_spmv_correctness(&mut harness);
    check_lanczos_sturm_parity(&mut harness);
    check_lanczos_full_spectrum(&mut harness);
    check_lanczos_convergence(&mut harness);
    check_clean_2d_bandwidth(&mut harness);
    check_2d_spectrum_bounds(&mut harness);
    check_2d_goe_statistics(&mut harness);
    check_2d_poisson_statistics(&mut harness);
    check_2d_statistics_transition(&mut harness);
    check_2d_vs_1d_bandwidth(&mut harness);

    println!();
    harness.finish();
}

/// \[1\] SpMV: CSR product matches dense reference on a known matrix.
fn check_spmv_correctness(harness: &mut ValidationHarness) {
    println!("[1] SpMV Correctness — CSR vs Dense Reference");

    // Build 5×5 tridiagonal: d=[1,2,3,4,5], e=[-1,-1,-1,-1]
    let n = 5;
    let d = [1.0, 2.0, 3.0, 4.0, 5.0];
    let e = [-1.0; 4];

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

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y = vec![0.0; n];
    csr.spmv(&x, &mut y);

    // Dense reference: y[i] = d[i]*x[i] + e[i-1]*x[i-1] + e[i]*x[i+1]
    let y_ref = [
        1.0 * 1.0 + -2.0,
        -1.0 + 2.0 * 2.0 + -3.0,
        -2.0 + 3.0 * 3.0 + -4.0,
        -3.0 + 4.0 * 4.0 + -5.0,
        -4.0 + 5.0 * 5.0,
    ];

    let max_err: f64 = y
        .iter()
        .zip(y_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("  CSR × x = {y:?}");
    println!("  Dense ref = {y_ref:?}");
    println!("  Max error = {max_err:.2e}");

    harness.check_upper("SpMV matches dense reference", max_err, 1e-14);
    println!();
}

/// \[2\] Lanczos vs Sturm: 1D Anderson extremal eigenvalues match.
fn check_lanczos_sturm_parity(harness: &mut ValidationHarness) {
    println!("[2] Lanczos vs Sturm — 1D Anderson Parity");

    let n = 200;
    let w = 3.0;
    let (d, e) = spectral::anderson_hamiltonian(n, w, 42);

    let sturm_evals = spectral::find_all_eigenvalues(&d, &e);

    let csr = tridiag_to_csr(&d, &e);
    let lanczos_result = spectral::lanczos(&csr, n, 42);
    let lanczos_evals = spectral::lanczos_eigenvalues(&lanczos_result);

    let sturm_min = sturm_evals[0];
    let sturm_max = *sturm_evals.last().expect("collection verified non-empty");
    let lanczos_min = lanczos_evals[0];
    let lanczos_max = *lanczos_evals.last().expect("collection verified non-empty");

    println!("  N={n}, W={w}");
    println!("  Sturm:   [{sturm_min:.6}, {sturm_max:.6}]");
    println!("  Lanczos: [{lanczos_min:.6}, {lanczos_max:.6}]");

    let err_min = (sturm_min - lanczos_min).abs();
    let err_max = (sturm_max - lanczos_max).abs();
    println!("  Δ(min) = {err_min:.2e}, Δ(max) = {err_max:.2e}");

    harness.check_upper("Lanczos min eigenvalue matches Sturm", err_min, 1e-6);
    harness.check_upper("Lanczos max eigenvalue matches Sturm", err_max, 1e-6);
    println!();
}

/// \[3\] Lanczos full spectrum: m=N iterations recovers all eigenvalues.
fn check_lanczos_full_spectrum(harness: &mut ValidationHarness) {
    println!("[3] Lanczos Full Spectrum — m = N Recovers All Eigenvalues");

    let n = 100;
    let w = 2.0;
    let (d, e) = spectral::anderson_hamiltonian(n, w, 99);

    let sturm_evals = spectral::find_all_eigenvalues(&d, &e);
    let csr = tridiag_to_csr(&d, &e);
    let lanczos_result = spectral::lanczos(&csr, n, 77);
    let lanczos_evals = spectral::lanczos_eigenvalues(&lanczos_result);

    assert_eq!(
        lanczos_result.iterations, n,
        "Lanczos should run n iterations"
    );

    let max_err: f64 = sturm_evals
        .iter()
        .zip(lanczos_evals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    println!("  N={n}, W={w}");
    println!("  Lanczos iterations: {}", lanczos_result.iterations);
    println!("  Max |λ_Sturm - λ_Lanczos| = {max_err:.2e}");

    harness.check_upper("full spectrum max error < 1e-8", max_err, 1e-8);
    println!();
}

/// \[4\] Lanczos convergence: extremal eigenvalues converge with iteration count.
fn check_lanczos_convergence(harness: &mut ValidationHarness) {
    println!("[4] Lanczos Convergence — Eigenvalues Improve with Iterations");

    let n = 300;
    let w = 2.0;
    let (d, e) = spectral::anderson_hamiltonian(n, w, 55);
    let csr = tridiag_to_csr(&d, &e);

    let sturm_evals = spectral::find_all_eigenvalues(&d, &e);
    let exact_min = sturm_evals[0];

    let ms = [20, 50, 100, 200, n];
    let mut errors = Vec::new();

    for &m in &ms {
        let result = spectral::lanczos(&csr, m, 55);
        let evals = spectral::lanczos_eigenvalues(&result);
        let err = (evals[0] - exact_min).abs();
        errors.push(err);
        println!("  m={m:>4}: λ_min = {:.8}, error = {err:.2e}", evals[0]);
    }

    // Error should decrease monotonically (or at least the last should be < first)
    let converging = errors.last().expect("collection verified non-empty")
        < errors.first().expect("collection verified non-empty");
    harness.check_bool("extremal eigenvalue converges with iterations", converging);
    println!();
}

/// \[5\] Clean 2D lattice: bandwidth = 8 (from -4 to 4).
fn check_clean_2d_bandwidth(harness: &mut ValidationHarness) {
    println!("[5] Clean 2D Lattice — Bandwidth = 8");

    let l = 20;
    let mat = spectral::clean_2d_lattice(l, l);
    let result = spectral::lanczos(&mat, l * l, 42);
    let evals = spectral::lanczos_eigenvalues(&result);

    let e_min = evals[0];
    let e_max = *evals.last().expect("collection verified non-empty");
    let bw = e_max - e_min;

    println!("  L×L = {l}×{l}, N = {}", l * l);
    println!("  E_min = {e_min:.6}, E_max = {e_max:.6}");
    println!("  Bandwidth = {bw:.4} (theory: 8.0)");

    harness.check_upper("bandwidth within 0.1 of 8.0", (bw - 8.0).abs(), 0.1);
    println!();
}

/// \[6\] 2D Anderson: spectrum bounded by [-4 - W/2, 4 + W/2].
fn check_2d_spectrum_bounds(harness: &mut ValidationHarness) {
    println!("[6] 2D Anderson — Spectrum Bounds");

    let l = 16;
    let w = 5.0;
    let mat = spectral::anderson_2d(l, l, w, 42);
    let result = spectral::lanczos(&mat, l * l, 42);
    let evals = spectral::lanczos_eigenvalues(&result);

    let bound = 4.0 + w / 2.0;
    let e_min = evals[0];
    let e_max = *evals.last().expect("collection verified non-empty");
    let in_bounds = e_min >= -(bound + 0.1) && e_max <= bound + 0.1;

    println!("  L={l}, W={w}, N={}", l * l);
    println!("  Spectrum: [{e_min:.4}, {e_max:.4}]");
    println!("  Bound: [-{bound:.1}, {bound:.1}]");

    harness.check_bool("2D spectrum within Gershgorin bounds", in_bounds);
    println!();
}

/// \[7\] 2D weak disorder: GOE level statistics (⟨r⟩ ≈ 0.531).
///
/// W=2.0 on L=16 places the system in the metallic regime (ξ >> L)
/// where random-matrix universality applies. Very weak disorder (W<1)
/// on small lattices retains clean-lattice symmetries that suppress
/// level repulsion — a finite-size effect, not localization.
fn check_2d_goe_statistics(harness: &mut ValidationHarness) {
    println!("[7] 2D Metallic Regime — GOE Level Statistics");
    println!("    Theory: ⟨r⟩ ≈ 0.531 when ξ >> L (Wigner-Dyson)");
    println!("    W=2 on L=16 → metallic regime, avoids clean-lattice symmetry artifacts\n");

    let l = 16;
    let w = 2.0;
    let n_real = 8;
    let goe_r = 0.531;

    let mut r_sum = 0.0;
    for seed in 0..n_real {
        let mat = spectral::anderson_2d(l, l, w, seed * 137 + 42);
        let result = spectral::lanczos(&mat, l * l, seed * 37 + 1);
        let evals = spectral::lanczos_eigenvalues(&result);
        let mid = evals.len() / 4;
        let end = 3 * evals.len() / 4;
        let bulk = &evals[mid..end];
        r_sum += spectral::level_spacing_ratio(bulk);
    }
    let r_mean = r_sum / n_real as f64;

    println!("  L={l}, W={w}, realizations={n_real}");
    println!("  ⟨r⟩ = {r_mean:.4} (GOE = {goe_r:.4})");

    harness.check_bool(
        "⟨r⟩ > 0.48 (GOE-like for metallic 2D regime)",
        r_mean > 0.48,
    );
    println!();
}

/// \[8\] 2D strong disorder: Poisson level statistics (⟨r⟩ ≈ 0.386).
fn check_2d_poisson_statistics(harness: &mut ValidationHarness) {
    println!("[8] 2D Strong Disorder — Poisson Level Statistics");
    println!(
        "    Theory: ⟨r⟩ ≈ {:.4} when ξ << L (localized)\n",
        spectral::POISSON_R
    );

    let l = 16;
    let w = 20.0;
    let n_real = 5;

    let mut r_sum = 0.0;
    for seed in 0..n_real {
        let mat = spectral::anderson_2d(l, l, w, seed * 137 + 42);
        let result = spectral::lanczos(&mat, l * l, seed * 37 + 1);
        let evals = spectral::lanczos_eigenvalues(&result);
        let mid = evals.len() / 4;
        let end = 3 * evals.len() / 4;
        let bulk = &evals[mid..end];
        r_sum += spectral::level_spacing_ratio(bulk);
    }
    let r_mean = r_sum / n_real as f64;

    println!("  L={l}, W={w}, realizations={n_real}");
    println!("  ⟨r⟩ = {r_mean:.4} (Poisson = {:.4})", spectral::POISSON_R);

    let deviation = (r_mean - spectral::POISSON_R).abs();
    harness.check_upper("⟨r⟩ within 0.04 of Poisson", deviation, 0.04);
    println!();
}

/// \[9\] 2D statistics transition: ⟨r⟩ decreases monotonically with disorder.
fn check_2d_statistics_transition(harness: &mut ValidationHarness) {
    println!("[9] 2D Statistics Transition — GOE → Poisson with Disorder");

    let l = 14;
    let w_values = [0.5, 2.0, 6.0, 15.0, 30.0];
    let n_real = 3;

    let mut r_values = Vec::new();

    for &w in &w_values {
        let mut r_sum = 0.0;
        for seed in 0..n_real {
            let mat = spectral::anderson_2d(l, l, w, seed * 137 + 42);
            let result = spectral::lanczos(&mat, l * l, seed * 37 + 1);
            let evals = spectral::lanczos_eigenvalues(&result);
            let mid = evals.len() / 4;
            let end = 3 * evals.len() / 4;
            let bulk = &evals[mid..end];
            r_sum += spectral::level_spacing_ratio(bulk);
        }
        let r_mean = r_sum / n_real as f64;
        r_values.push(r_mean);
        println!("  W={w:>5.1}: ⟨r⟩ = {r_mean:.4}");
    }

    // r should decrease monotonically (GOE→Poisson) — check first > last
    let transition = r_values.first().expect("collection verified non-empty")
        > r_values.last().expect("collection verified non-empty");
    // At least ~0.05 drop
    let delta = r_values.first().expect("collection verified non-empty")
        - r_values.last().expect("collection verified non-empty");
    println!("  Δ⟨r⟩ = {delta:.4} (weak→strong)");

    harness.check_bool(
        "⟨r⟩ decreases from weak to strong disorder",
        transition && delta > 0.05,
    );
    println!();
}

/// \[10\] 2D bandwidth > 1D bandwidth for same disorder.
fn check_2d_vs_1d_bandwidth(harness: &mut ValidationHarness) {
    println!("[10] 2D vs 1D Bandwidth Comparison");

    let w = 2.0;

    // 1D: N=400, bandwidth ≈ 2+W = 4+W = 2+2+W (clean BW=4, -2..2)
    let n_1d = 400;
    let (d, e) = spectral::anderson_hamiltonian(n_1d, w, 42);
    let evals_1d = spectral::find_all_eigenvalues(&d, &e);
    let bw_1d = evals_1d.last().expect("collection verified non-empty")
        - evals_1d.first().expect("collection verified non-empty");

    // 2D: 20×20=400, bandwidth ≈ 8+W (clean BW=8, -4..4)
    let l = 20;
    let mat = spectral::anderson_2d(l, l, w, 42);
    let result = spectral::lanczos(&mat, l * l, 42);
    let evals_2d = spectral::lanczos_eigenvalues(&result);
    let bw_2d = evals_2d.last().expect("collection verified non-empty")
        - evals_2d.first().expect("collection verified non-empty");

    println!("  W={w}");
    println!("  1D (N={n_1d}): bandwidth = {bw_1d:.4}");
    println!("  2D ({l}×{l}={n}):  bandwidth = {bw_2d:.4}", n = l * l);
    println!("  Ratio: {:.2}×", bw_2d / bw_1d);

    harness.check_bool("2D bandwidth > 1D bandwidth", bw_2d > bw_1d);
    println!();
}

// ── helpers ──────────────────────────────────────────────────────────

fn tridiag_to_csr(d: &[f64], e: &[f64]) -> CsrMatrix {
    let n = d.len();
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
    CsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}
