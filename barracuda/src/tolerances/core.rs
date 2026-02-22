// SPDX-License-Identifier: AGPL-3.0-only

//! Core numerical tolerances: machine precision, linear algebra, special functions,
//! and optimizer/ODE solver validation thresholds.

// ═══════════════════════════════════════════════════════════════════
// Machine-precision tolerances (IEEE 754 f64)
// ═══════════════════════════════════════════════════════════════════

/// Tolerance for operations that should be exact in f64 arithmetic.
///
/// f64 has ~15.9 significant digits; 1e-10 allows 5 digits of accumulated
/// rounding in compositions of exact operations (e.g. LU decompose → solve).
pub const EXACT_F64: f64 = 1e-10;

/// Tolerance for f64 operations with moderate accumulation.
///
/// Used for iterative algorithms (QR, SVD) where O(n) rounding steps
/// accumulate. 1e-8 allows ~7 digits of precision after iteration.
pub const ITERATIVE_F64: f64 = 1e-8;

/// Tolerance for comparing GPU f64 results against CPU f64.
///
/// GPU `SHADER_F64` uses the same IEEE 754 representation but different
/// instruction ordering. Max observed difference: 8e-8 for `exp()` on
/// RTX 4070 (see `experiments/001_N_SCALING_GPU.md` §4.4).
pub const GPU_VS_CPU_F64: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Linear algebra tolerances
// ═══════════════════════════════════════════════════════════════════

/// LU decomposition: determinant and solve accuracy.
///
/// LU is O(n³) with exact pivoting; accumulated rounding is small.
/// Reference: `NumPy` `numpy.linalg.solve` achieves ~1e-14 for well-conditioned
/// systems. We allow 1e-10 for condition numbers up to ~1e5.
pub const LU_TOLERANCE: f64 = EXACT_F64;

/// QR decomposition: orthonormality check (Q^T Q = I).
///
/// Modified Gram-Schmidt has O(n²) rounding; Householder is more stable.
/// barracuda uses Householder. 1e-10 is conservative.
pub const QR_TOLERANCE: f64 = EXACT_F64;

/// SVD: singular value accuracy.
///
/// Golub-Kahan bidiagonalization + QR iteration. Converges to ~machine
/// epsilon for well-conditioned matrices. 1e-6 accounts for accumulated
/// iteration in the QR sweep phase.
pub const SVD_TOLERANCE: f64 = 1e-6;

/// Tridiagonal solver: direct Thomas algorithm.
///
/// O(n) operations, no iteration. Should be near machine precision.
pub const TRIDIAG_TOLERANCE: f64 = EXACT_F64;

/// Eigenvalue decomposition (eigh): relative error on eigenvalues.
///
/// Jacobi rotation converges quadratically. For well-separated eigenvalues,
/// relative accuracy is ~machine epsilon. 1e-8 accounts for degenerate cases.
pub const EIGH_TOLERANCE: f64 = ITERATIVE_F64;

// ═══════════════════════════════════════════════════════════════════
// Special function tolerances
// ═══════════════════════════════════════════════════════════════════

/// Gamma function: Lanczos approximation.
///
/// Max relative error ~2e-10 (Lanczos g=7, n=9 coefficients).
/// Reference: A&S 6.1, DLMF 5.2.
pub const GAMMA_TOLERANCE: f64 = EXACT_F64;

/// Error function (erf/erfc): rational approximation.
///
/// Max absolute error ~1.5e-7 for Horner-form rational approximation.
/// Reference: A&S 7.1.26.
pub const ERF_TOLERANCE: f64 = 1e-6;

/// Bessel functions: series/asymptotic.
///
/// Accuracy depends on argument range. For small arguments (|x| < 10),
/// series converges to ~1e-15. For large arguments, asymptotic expansion
/// is less precise. 1e-6 is conservative across the full range.
pub const BESSEL_TOLERANCE: f64 = 1e-6;

/// Factorial: exact integer multiplication in f64.
///
/// For n ≤ 20, n! fits within f64 mantissa (2^53 > 20!). The computation
/// is a chain of exact integer multiplications, so the result should match
/// the reference to machine precision. Uses `EXACT_F64`.
pub const FACTORIAL_TOLERANCE: f64 = EXACT_F64;

/// Associated Legendre `P_n^m(x)`: three-term recurrence.
///
/// For small n ≤ 5 and |m| ≤ n, the recurrence involves O(n) multiplications
/// and subtractions. Accumulated rounding is negligible for small n.
/// Uses `EXACT_F64` (tested up to n=2, m=2).
pub const ASSOC_LEGENDRE_TOLERANCE: f64 = EXACT_F64;

/// Digamma ψ(x) via finite-difference of `ln_gamma`.
///
/// Central difference (f(x+h) − f(x−h))/(2h) with h=1e-7.
/// Truncation error: O(h²) = O(1e-14).
/// Cancellation error: `O(ε_mach/h)` = O(1e-16/1e-7) = O(1e-9).
/// Total error dominated by cancellation: ~1e-9.
/// 1e-5 is conservative (4 orders of margin).
pub const DIGAMMA_FD_TOLERANCE: f64 = 1e-5;

/// Beta function B(a,b) via exp(lnΓ(a) + lnΓ(b) − lnΓ(a+b)).
///
/// Three `ln_gamma` evaluations (each ~1e-10 Lanczos error) plus `exp()`.
/// Error propagation through exp amplifies by the exponent magnitude.
/// For small arguments (a,b ≤ 3), the exponent is O(1) so amplification
/// is modest. 1e-6 accounts for accumulated composition error.
pub const BETA_VIA_LNGAMMA_TOLERANCE: f64 = 1e-6;

/// Regularized incomplete gamma P(a,x): series or continued-fraction.
///
/// For small a and x, the series converges rapidly with ~1e-10 per-term
/// accuracy. For larger arguments, the continued-fraction representation
/// converges more slowly. 1e-6 is conservative across the tested range
/// (a ≤ 2, x ≤ 100).
pub const INCOMPLETE_GAMMA_TOLERANCE: f64 = 1e-6;

/// Bessel near-zero absolute tolerance.
///
/// Near zero crossings (e.g. J₀(2.4048) ≈ 0), the expected value is zero
/// and relative error is meaningless. The function changes sign with slope
/// |J₀'(x)| ≈ 0.5, so an absolute error of 1e-3 corresponds to an
/// argument error of ~0.002, well within series truncation bounds.
pub const BESSEL_NEAR_ZERO_ABS: f64 = 1e-3;

/// Chi-squared CDF: regularized incomplete gamma.
///
/// Numerical integration introduces ~0.1% relative error for large df.
/// 0.005 absolute tolerance is conservative.
pub const CHI2_CDF_TOLERANCE: f64 = 0.005;

/// Laguerre polynomial: three-term recurrence.
///
/// Exact for small n; accumulated rounding grows with n.
/// 1e-10 is conservative for n ≤ 20.
pub const LAGUERRE_TOLERANCE: f64 = EXACT_F64;

// ═══════════════════════════════════════════════════════════════════
// Optimizer tolerances
// ═══════════════════════════════════════════════════════════════════

/// Nelder-Mead: convergence on Rosenbrock.
///
/// NM is a derivative-free method with linear convergence. On the
/// Rosenbrock function, it typically reaches ~1e-4 accuracy in the
/// minimum location with default settings.
pub const NELDER_MEAD_TOLERANCE: f64 = 1e-4;

/// BFGS: position convergence on smooth test functions.
///
/// Quasi-Newton with superlinear convergence. The algorithm's internal gradient
/// tolerance (`gtol`) is typically 1e-8, but the position of the minimum is
/// validated to 1e-4 because numerical gradient (finite differences) introduces
/// O(sqrt(eps)) ≈ 1e-8 error per step, accumulating over iterations.
/// Rosenbrock 2D: converges to (1,1) within 1e-4.
/// Quadratic N-D: converges to origin within 1e-4.
pub const BFGS_TOLERANCE: f64 = 1e-4;

/// Bisection/Brent: root-finding convergence.
///
/// Brent's method converges superlinearly. With 100 iterations and
/// initial bracket of width ~1, reaches ~1e-12.
pub const BRENT_TOLERANCE: f64 = EXACT_F64;

/// ODE integrator (RK45): step accuracy.
///
/// Adaptive step-size control targets local truncation error ~1e-6.
/// Accumulated global error over long integrations may reach ~1e-4.
pub const RK45_TOLERANCE: f64 = 1e-4;

/// Sobol sequence: uniformity check.
///
/// Low-discrepancy sequences have known star-discrepancy bounds.
/// For N=1024, D* ≈ O(log(N)^d / N). 0.05 absolute tolerance on
/// mean deviation from uniform is conservative.
pub const SOBOL_TOLERANCE: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════
// Optimizer and ODE solver tolerances (validation binaries)
// ═══════════════════════════════════════════════════════════════════

/// BFGS: gradient-norm convergence tolerance.
///
/// BFGS converges when ||∇f|| < gtol. For smooth test functions
/// (Rosenbrock, quadratic), 1e-8 achieves near-optimal minima.
pub const BFGS_GTOL: f64 = 1e-8;

/// Nelder-Mead: function-value convergence tolerance.
///
/// NM converges when the simplex spans < ftol in function value.
/// 1e-10 is tight but achievable for smooth 2D test functions.
pub const NELDER_MEAD_FTOL: f64 = 1e-10;

/// Bisection: root-finding convergence tolerance.
///
/// Bisection narrows the bracket to width < tol. For f64, 1e-12
/// gives ~40 bits of accuracy in the root location.
pub const BISECT_CONVERGENCE_TOL: f64 = 1e-12;

/// RK45: default absolute tolerance for ODE integration.
///
/// Controls the local error estimator. For exponential decay and
/// harmonic oscillator test problems, 1e-8 atol with 1e-10 rtol
/// gives global errors well below 1e-6.
pub const RK45_ATOL: f64 = 1e-8;

/// RK45: default relative tolerance for ODE integration.
pub const RK45_RTOL: f64 = 1e-10;

/// Normal CDF: maximum absolute error vs reference table.
///
/// The erf-based CDF implementation matches published tables to ~1e-4.
/// Limited by the 4-term rational approximation in `barracuda::special::erf`.
pub const NORMAL_CDF_TOLERANCE: f64 = 1e-4;

/// Normal PPF (inverse CDF): maximum absolute error vs reference table.
///
/// The rational approximation for the quantile function has ~1e-3
/// accuracy in the tails (|z| > 2).
pub const NORMAL_PPF_TOLERANCE: f64 = 1e-3;
