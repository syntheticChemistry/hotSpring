// SPDX-License-Identifier: AGPL-3.0-only

//! Conjugate Gradient solver for D†D x = b on the lattice.
//!
//! The standard CG algorithm for positive-definite Hermitian systems.
//! In lattice QCD, CG is the dominant cost (>95% of runtime) because
//! each HMC trajectory requires solving the Dirac equation.
//!
//! # Algorithm
//!
//! Standard CG with relative residual convergence criterion:
//!   ||r||² / ||b||² < tol²
//!
//! # References
//!
//! - Hestenes & Stiefel (1952) — original CG
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 8.4

use super::complex_f64::Complex64;
use super::dirac::{apply_dirac_sq, FermionField};
use super::wilson::Lattice;

// ═══════════════════════════════════════════════════════════════════
//  GPU WGSL shaders: CG vector operations (absorption-ready)
// ═══════════════════════════════════════════════════════════════════

/// WGSL shader: real part of complex dot product.
///
/// Computes `out[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]` for each complex pair,
/// producing a real array that can be summed via `ReduceScalarPipeline`.
///
/// Combined with reduction: `Re(<a|b>) = Σ out[i]`.
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | uniform | `{ n_pairs: u32, pad: u32×3 }` |
/// | 1 | storage, read | `a: array<f64>` (2×n_pairs) |
/// | 2 | storage, read | `b: array<f64>` (2×n_pairs) |
/// | 3 | storage, read_write | `out: array<f64>` (n_pairs) |
pub const WGSL_COMPLEX_DOT_RE_F64: &str = include_str!("shaders/complex_dot_re_f64.wgsl");

/// WGSL shader: real-scalar axpy on f64 arrays.
///
/// `y[i] += alpha * x[i]` for all i. Works on flat f64 arrays representing
/// complex fermion fields — alpha is real, so re/im scale identically.
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | uniform | `{ n: u32, pad: u32, alpha: f64 }` |
/// | 1 | storage, read | `x: array<f64>` |
/// | 2 | storage, read_write | `y: array<f64>` |
pub const WGSL_AXPY_F64: &str = include_str!("shaders/axpy_f64.wgsl");

/// WGSL shader: xpay operation `p[i] = x[i] + beta * p[i]`.
///
/// Used in CG to update the search direction: `p = r + beta * p`.
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | uniform | `{ n: u32, pad: u32, beta: f64 }` |
/// | 1 | storage, read | `x: array<f64>` |
/// | 2 | storage, read_write | `p: array<f64>` |
pub const WGSL_XPAY_F64: &str = include_str!("shaders/xpay_f64.wgsl");

/// WGSL shader: tree sum reduction for f64 arrays.
///
/// Reduces N values to ceil(N/256) partial sums per dispatch. Chain two
/// dispatches for full scalar output. Workgroup size = 256, shared memory.
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | storage, read | `input: array<f64>` (N) |
/// | 1 | storage, read_write | `output: array<f64>` (ceil(N/256)) |
/// | 2 | uniform | `{ size: u32, pad: u32×3 }` |
pub const WGSL_SUM_REDUCE_F64: &str = include_str!("shaders/sum_reduce_f64.wgsl");

/// WGSL shader: CG scalar alpha = rz / pAp (1-thread kernel).
pub const WGSL_CG_COMPUTE_ALPHA_F64: &str = include_str!("shaders/cg_compute_alpha_f64.wgsl");

/// WGSL shader: CG scalar beta = rz_new / rz_old + copy (1-thread kernel).
pub const WGSL_CG_COMPUTE_BETA_F64: &str = include_str!("shaders/cg_compute_beta_f64.wgsl");

/// WGSL shader: CG vector update x += alpha*p, r -= alpha*ap (reads alpha from GPU buffer).
pub const WGSL_CG_UPDATE_XR_F64: &str = include_str!("shaders/cg_update_xr_f64.wgsl");

/// WGSL shader: CG vector update p = r + beta*p (reads beta from GPU buffer).
pub const WGSL_CG_UPDATE_P_F64: &str = include_str!("shaders/cg_update_p_f64.wgsl");

/// CG solver result.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct CgResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_residual: f64,
    pub initial_residual: f64,
}

/// Solve D†D x = b using Conjugate Gradient.
///
/// `x` is modified in place with the solution.
/// `b` is the right-hand side.
/// `mass` is the fermion mass for the Dirac operator.
/// `tol` is the relative residual tolerance.
/// `max_iter` is the maximum number of iterations.
pub fn cg_solve(
    lattice: &Lattice,
    x: &mut FermionField,
    b: &FermionField,
    mass: f64,
    tol: f64,
    max_iter: usize,
) -> CgResult {
    let vol = lattice.volume();

    // r = b - A x
    let ax = apply_dirac_sq(lattice, x, mass);
    let mut r = FermionField::zeros(vol);
    for i in 0..vol {
        for c in 0..3 {
            r.data[i][c] = b.data[i][c] - ax.data[i][c];
        }
    }

    let b_norm_sq = b.norm_sq();
    if b_norm_sq < super::constants::LATTICE_DIVISION_GUARD {
        return CgResult {
            converged: true,
            iterations: 0,
            final_residual: 0.0,
            initial_residual: 0.0,
        };
    }

    let mut r_norm_sq = r.norm_sq();
    let initial_residual = (r_norm_sq / b_norm_sq).sqrt();
    let tol_sq = tol * tol * b_norm_sq;

    if r_norm_sq < tol_sq {
        return CgResult {
            converged: true,
            iterations: 0,
            final_residual: initial_residual,
            initial_residual,
        };
    }

    // p = r
    let mut p = FermionField::zeros(vol);
    p.copy_from(&r);

    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Ap = D†D p
        let ap = apply_dirac_sq(lattice, &p, mass);

        // alpha = <r|r> / <p|Ap>
        let p_ap = p.dot(&ap).re;
        if p_ap.abs() < super::constants::LATTICE_DIVISION_GUARD {
            break;
        }
        let alpha = r_norm_sq / p_ap;

        // x = x + alpha * p
        x.axpy(Complex64::new(alpha, 0.0), &p);

        // r = r - alpha * Ap
        r.axpy(Complex64::new(-alpha, 0.0), &ap);

        let r_norm_sq_new = r.norm_sq();

        if r_norm_sq_new < tol_sq {
            r_norm_sq = r_norm_sq_new;
            break;
        }

        // beta = <r_new|r_new> / <r_old|r_old>
        let beta = r_norm_sq_new / r_norm_sq;
        r_norm_sq = r_norm_sq_new;

        // p = r + beta * p
        for i in 0..vol {
            for c in 0..3 {
                p.data[i][c] = r.data[i][c] + p.data[i][c].scale(beta);
            }
        }
    }

    let final_residual = (r_norm_sq / b_norm_sq).sqrt();

    CgResult {
        converged: final_residual < tol,
        iterations,
        final_residual,
        initial_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cg_identity_lattice() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let b = FermionField::random(vol, 42);
        let mut x = FermionField::zeros(vol);

        let result = cg_solve(&lat, &mut x, &b, 1.0, 1e-8, 500);

        assert!(
            result.converged,
            "CG should converge on identity lattice: residual={}",
            result.final_residual
        );
        assert!(
            result.final_residual < 1e-8,
            "residual should be < 1e-8: {}",
            result.final_residual
        );
    }

    #[test]
    fn cg_zero_rhs() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let b = FermionField::zeros(vol);
        let mut x = FermionField::zeros(vol);

        let result = cg_solve(&lat, &mut x, &b, 0.1, 1e-10, 100);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn cg_verify_solution() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let b = FermionField::random(vol, 99);
        let mut x = FermionField::zeros(vol);

        let result = cg_solve(&lat, &mut x, &b, 0.5, 1e-6, 1000);
        assert!(result.converged, "CG should converge");

        // Verify: A x ≈ b
        let ax = apply_dirac_sq(&lat, &x, 0.5);
        let mut residual_sq = 0.0;
        for i in 0..vol {
            for c in 0..3 {
                let diff = ax.data[i][c] - b.data[i][c];
                residual_sq += diff.abs_sq();
            }
        }
        let rel_residual = (residual_sq / b.norm_sq()).sqrt();
        assert!(
            rel_residual < 1e-6,
            "Ax should ≈ b: relative residual = {rel_residual}"
        );
    }

    #[test]
    fn cg_hot_lattice() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let vol = lat.volume();
        let b = FermionField::random(vol, 123);
        let mut x = FermionField::zeros(vol);

        let result = cg_solve(&lat, &mut x, &b, 0.5, 1e-4, 2000);
        assert!(
            result.converged,
            "CG should converge on hot lattice: residual={}, iters={}",
            result.final_residual, result.iterations
        );
    }
}
