// SPDX-License-Identifier: AGPL-3.0-only

//! Centralized validation tolerances with physical justification.
//!
//! Every tolerance threshold used in validation binaries is defined here
//! with documentation of its origin and rationale. No ad-hoc magic numbers.
//!
//! # Tolerance categories
//!
//! | Category | Basis | Example |
//! |----------|-------|---------|
//! | Machine precision | IEEE 754 f64 | 1e-10 for exact arithmetic |
//! | Numerical method | Algorithm convergence | 1e-6 for iterative solvers |
//! | Physical model | Model limitations | 5% for energy drift |
//! | Literature | Published uncertainty | NMP 2σ bounds |
//!
//! See `whitePaper/METHODOLOGY.md` for the full acceptance protocol.

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
/// GPU SHADER_F64 uses the same IEEE 754 representation but different
/// instruction ordering. Max observed difference: 8e-8 for exp() on
/// RTX 4070 (see `experiments/001_N_SCALING_GPU.md` §4.4).
pub const GPU_VS_CPU_F64: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════
// Linear algebra tolerances
// ═══════════════════════════════════════════════════════════════════

/// LU decomposition: determinant and solve accuracy.
///
/// LU is O(n³) with exact pivoting; accumulated rounding is small.
/// Reference: NumPy `numpy.linalg.solve` achieves ~1e-14 for well-conditioned
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
// MD force tolerances
// ═══════════════════════════════════════════════════════════════════

/// MD force comparison: GPU f32 vs analytical CPU f64.
///
/// GPU forces are computed in f32 (WGSL `vec3<f32>`) while the reference
/// is f64. After exp(), sqrt(), and force accumulation, GPU-vs-CPU
/// relative error of up to 1% is expected for f32 arithmetic.
/// Measured: ~0.1% for well-separated pairs, ~0.5% near cutoff.
pub const MD_FORCE_TOLERANCE: f64 = 0.01;

/// MD absolute error floor for near-zero expected values.
///
/// When the expected force component is near zero (e.g. transverse
/// components at equilibrium), relative error is meaningless.
/// This absolute floor (1e-4) is consistent with f32 mantissa precision
/// at force magnitudes of O(1).
pub const MD_ABSOLUTE_FLOOR: f64 = 1e-4;

/// Newton's 3rd law: absolute residual tolerance.
///
/// For two-body forces, F_i + F_j should be exactly zero.
/// f32 rounding produces residuals of O(eps_f32 * F_mag) ≈ 1e-7 * 10 = 1e-6.
/// 1e-3 is conservative for multi-particle sums.
pub const NEWTON_3RD_LAW_ABS: f64 = 1e-3;

/// MD equilibrium force: absolute magnitude at zero-force point.
///
/// At the LJ minimum (r = 2^(1/6) * sigma), the net force is zero.
/// GPU f32 rounding produces residual force of O(eps_f32 * F_scale).
/// 0.05 absolute threshold is conservative.
pub const MD_EQUILIBRIUM_FORCE_ABS: f64 = 0.05;

/// Cell-list vs all-pairs: relative PE difference.
///
/// Both compute the same pairwise sum; differences arise from cutoff
/// boundary effects and floating-point summation order. 1e-6 relative
/// error is expected for well-converged cell lists.
pub const CELLLIST_PE_TOLERANCE: f64 = 1e-6;

/// Cell-list vs all-pairs: net force magnitude.
///
/// Net force on the system should be zero (Newton's 3rd law). Cell-list
/// summation order differs from all-pairs, so floating-point residual
/// is O(N * eps * F_avg).
pub const CELLLIST_NETFORCE_TOLERANCE: f64 = 1e-4;

// ═══════════════════════════════════════════════════════════════════
// MD observable tolerances (METHODOLOGY.md acceptance criteria)
// ═══════════════════════════════════════════════════════════════════

/// Energy drift acceptance: |ΔE/E| over production run.
///
/// Symplectic integrators (Velocity-Verlet) conserve a shadow Hamiltonian
/// with drift O(dt²). For dt=0.01 and 80k steps, measured drift is
/// 0.000–0.002% on RTX 4070. 5% is the generous acceptance threshold
/// from METHODOLOGY.md.
///
/// Source: METHODOLOGY.md §Phase C acceptance criteria.
pub const ENERGY_DRIFT_PCT: f64 = 5.0;

/// RDF tail convergence: |g(r→∞) − 1|.
///
/// For a well-equilibrated fluid, g(r) → 1 at large r. Finite-size
/// effects and poor statistics cause deviations. 0.15 is conservative
/// for N=10,000 with ~80 snapshots.
///
/// Source: METHODOLOGY.md §Phase C acceptance criteria.
pub const RDF_TAIL_TOLERANCE: f64 = 0.15;

/// GPU eigensolve: relative eigenvalue error tolerance.
///
/// Jacobi rotation on GPU converges to ~1e-4 relative error for matrices
/// up to 32×32. CPU fallback achieves 1e-8. The tolerance reflects the
/// GPU path (single-precision intermediate, f64 accumulation).
pub const GPU_EIGENSOLVE_REL: f64 = 1e-4;

/// GPU eigensolve: eigenvector orthogonality tolerance.
///
/// Max |Q^T Q - I|_ij for GPU-computed eigenvectors. Jacobi rotation
/// maintains orthogonality to ~1e-4 on GPU (vs ~1e-10 on CPU).
pub const GPU_EIGENVECTOR_ORTHO: f64 = 1e-4;

/// BCS particle number: absolute error tolerance.
///
/// BCS bisection on GPU converges particle number (sum of occupation
/// probabilities) to within 0.01 of the target. Higher precision requires
/// more bisection iterations (currently 100).
pub const BCS_PARTICLE_NUMBER_ABS: f64 = 0.01;

/// BCS chemical potential: GPU vs CPU relative error.
///
/// GPU and CPU bisection use different arithmetic paths; the chemical
/// potential μ should agree to within GPU f64 precision.
pub const BCS_CHEMICAL_POTENTIAL_REL: f64 = 1e-6;

/// PPPM Newton's 3rd law: absolute net-force tolerance.
///
/// Long-range Coulomb via PPPM satisfies Newton's 3rd law to machine
/// precision for the real-space part, but the reciprocal-space contribution
/// has O(N * eps) residual from FFT rounding. 1e-6 absolute on net force
/// per particle is consistent with f64 FFT at N ~ 1000.
pub const PPPM_NEWTON_3RD_ABS: f64 = 1e-6;

/// PPPM Madelung constant: relative error tolerance.
///
/// The NaCl Madelung constant (1.747565...) is analytically known.
/// PPPM converges to it as mesh density increases. 0.01 (1%) relative
/// error is achievable with modest mesh parameters.
pub const PPPM_MADELUNG_REL: f64 = 0.01;

/// HFB Rust-vs-Python: relative binding energy tolerance.
///
/// Rust and Python L2 solvers use different numerical methods (bisection
/// vs Brent, density mixing). Light nuclei (A < 60) are most sensitive.
/// Measured: Ni-56 shows ~11%, medium nuclei 1-5%. 12% accounts for all.
pub const HFB_RUST_VS_PYTHON_REL: f64 = 0.12;

/// HFB Rust-vs-experiment: relative binding energy tolerance.
///
/// The L2 HFB solver with SLy4 parametrization reproduces experimental
/// binding energies to ~5-10% for medium-mass nuclei. Missing physics
/// (deformation, continuum, 3-body) limits accuracy.
pub const HFB_RUST_VS_EXP_REL: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════
// Nuclear EOS acceptance criteria (METHODOLOGY.md)
// ═══════════════════════════════════════════════════════════════════

/// L1 (SEMF) acceptance: χ²/datum threshold.
///
/// A good L1 fit achieves χ²/datum < 10. The Python control best is 6.62.
pub const L1_CHI2_THRESHOLD: f64 = 10.0;

/// L2 (HFB) acceptance: χ²/datum threshold.
///
/// A good L2 fit achieves χ²/datum < 5. The Python control best is ~2
/// for well-tuned parametrizations.
pub const L2_CHI2_THRESHOLD: f64 = 5.0;

/// L1 proxy pre-screen: χ²/datum cutoff.
///
/// Generous threshold to avoid false negatives. If a parameter set
/// can't achieve χ²/datum < 200 at L1, it won't work at L2.
pub const L1_PROXY_THRESHOLD: f64 = 200.0;

/// NMP acceptance: number of standard deviations.
///
/// Parameter sets must produce NMP within 2σ of experimental targets
/// for all five properties. See `provenance::NMP_TARGETS`.
pub const NMP_N_SIGMA: f64 = 2.0;

/// Theoretical uncertainty floor for binding energy comparison.
///
/// σ_theo = max(1% × B_exp, 2.0 MeV). The 1% accounts for missing
/// physics (deformation, pairing fluctuations); the 2 MeV floor prevents
/// division by zero for light nuclei.
pub const SIGMA_THEO_FRACTION: f64 = 0.01;
pub const SIGMA_THEO_FLOOR_MEV: f64 = 2.0;

/// Compute theoretical uncertainty for a given experimental binding energy.
#[must_use]
pub fn sigma_theo(b_exp: f64) -> f64 {
    (SIGMA_THEO_FRACTION * b_exp).max(SIGMA_THEO_FLOOR_MEV)
}

// ═══════════════════════════════════════════════════════════════════
// Physics guard constants (numerical singularity avoidance)
// ═══════════════════════════════════════════════════════════════════

/// Density floor: minimum nuclear density in fm⁻³.
///
/// Prevents division-by-zero in Skyrme potential terms that depend on
/// `ρ^α` or `ρ^(α-1)` where α ≈ 1/6. At 1e-15 fm⁻³, the density is
/// ~14 orders of magnitude below saturation density (0.16 fm⁻³), safely
/// below any physical nuclear density while avoiding floating-point
/// underflow in `ρ.powf(α)`.
pub const DENSITY_FLOOR: f64 = 1e-15;

/// Minimum radius for spin-orbit 1/r singularity guard (fm).
///
/// The spin-orbit potential V_so ∝ (1/r)(dρ/dr) diverges at r=0.
/// The physical nuclear surface starts at r ≈ 1.2·A^(1/3) fm (≈3 fm
/// for A=16), so r < 0.1 fm is deep inside the nuclear core where
/// the spin-orbit integrand should vanish anyway. The 0.1 fm floor
/// (~0.5 × proton charge radius) prevents numerical overflow without
/// affecting physics in the surface region where spin-orbit matters.
pub const SPIN_ORBIT_R_MIN: f64 = 0.1;

/// Coulomb singularity guard for charge-enclosed / r (fm).
///
/// The Coulomb direct potential V_C(r) = e² × Q(r)/r diverges at r=0.
/// At r = 1e-10 fm (well below quark confinement scale ~1 fm), the
/// enclosed charge Q(r) ∝ r³ → 0 faster than 1/r → ∞, so the product
/// is bounded. The guard prevents 0/0 without affecting physics.
pub const COULOMB_R_MIN: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_ordering() {
        assert!(EXACT_F64 < ITERATIVE_F64);
        assert!(ITERATIVE_F64 < GPU_VS_CPU_F64);
        assert!(GPU_VS_CPU_F64 < MD_FORCE_TOLERANCE);
    }

    #[test]
    fn sigma_theo_floor() {
        assert_eq!(sigma_theo(100.0), 2.0);
        assert_eq!(sigma_theo(500.0), 5.0);
        assert_eq!(sigma_theo(0.0), 2.0);
    }

    #[test]
    fn energy_drift_is_generous() {
        // Measured drift is 0.000-0.002%; threshold is 5%.
        // This ensures we don't false-fail on numerical noise.
        assert!(ENERGY_DRIFT_PCT > 1.0);
    }

    #[test]
    fn nmp_sigma_is_two() {
        assert!((NMP_N_SIGMA - 2.0).abs() < f64::EPSILON);
    }
}
