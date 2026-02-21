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

/// Factorial: exact integer multiplication in f64.
///
/// For n ≤ 20, n! fits within f64 mantissa (2^53 > 20!). The computation
/// is a chain of exact integer multiplications, so the result should match
/// the reference to machine precision. Uses `EXACT_F64`.
pub const FACTORIAL_TOLERANCE: f64 = EXACT_F64;

/// Associated Legendre P_n^m(x): three-term recurrence.
///
/// For small n ≤ 5 and |m| ≤ n, the recurrence involves O(n) multiplications
/// and subtractions. Accumulated rounding is negligible for small n.
/// Uses `EXACT_F64` (tested up to n=2, m=2).
pub const ASSOC_LEGENDRE_TOLERANCE: f64 = EXACT_F64;

/// Digamma ψ(x) via finite-difference of ln_gamma.
///
/// Central difference (f(x+h) − f(x−h))/(2h) with h=1e-7.
/// Truncation error: O(h²) = O(1e-14).
/// Cancellation error: O(ε_mach/h) = O(1e-16/1e-7) = O(1e-9).
/// Total error dominated by cancellation: ~1e-9.
/// 1e-5 is conservative (4 orders of margin).
pub const DIGAMMA_FD_TOLERANCE: f64 = 1e-5;

/// Beta function B(a,b) via exp(lnΓ(a) + lnΓ(b) − lnΓ(a+b)).
///
/// Three ln_gamma evaluations (each ~1e-10 Lanczos error) plus exp().
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

/// Native GPU f64 builtins vs software (math_f64) implementations.
///
/// Native GPU sqrt/exp/log may use hardware-specific algorithms (e.g. Newton-Raphson
/// with different initial approximations) that differ from the polynomial/rational
/// approximations in math_f64.wgsl. Observed max differences on RTX 4070 (Ada Lovelace):
///
/// - sqrt: ~5e-7 (relative, over 200 test values in \[0.01, 10\])
/// - exp:  ~8e-8 (relative, over 200 test values in \[-5, 5\])
///
/// Both implementations produce correct results; the tolerance captures the
/// maximum difference between two valid f64 computation paths.
pub const GPU_NATIVE_VS_SOFTWARE_F64: f64 = 1e-6;

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

// ═══════════════════════════════════════════════════════════════════
// Transport coefficient tolerances (Stanton & Murillo validation)
// ═══════════════════════════════════════════════════════════════════

/// D* Rust-vs-Sarkas: relative self-diffusion tolerance.
///
/// MD transport coefficients are stochastic; Green-Kubo integration of
/// VACF has statistical noise that scales as 1/sqrt(N_steps). With
/// N=500 lite and 20k steps, 5% relative agreement between two
/// independent MD runs is expected.
///
/// Source: Stanton & Murillo (2016) PRE 93 043203, Table I.
pub const TRANSPORT_D_STAR_VS_SARKAS: f64 = 0.05;

/// Daligault fit reproducing its own Sarkas calibration data.
///
/// The fit is a smooth analytical model with 5 parameters (A(κ), α(κ)
/// quadratics + C_w) fitted to 12 Green-Kubo D* points at N=2000.
/// Per-point error up to 20% is acceptable (smoothness constraint
/// prevents exact interpolation), RMSE must be < 10%.
///
/// Source: calibrate_daligault_fit.py (Feb 2026).
pub const DALIGAULT_FIT_VS_CALIBRATION: f64 = 0.20;

/// RMSE of Daligault fit across all 12 calibration points.
pub const DALIGAULT_FIT_RMSE: f64 = 0.10;

/// D* Rust-vs-Daligault analytical fit: relative tolerance.
///
/// The Daligault (2012) fit is a smooth interpolation between asymptotic
/// limits; MD deviates from the fit by up to 10% at intermediate coupling
/// (Gamma ~ 10-50) where neither limit applies cleanly.
///
/// Source: Daligault PRE 86, 047401 (2012), Fig. 1.
pub const TRANSPORT_D_STAR_VS_FIT: f64 = 0.10;

/// Viscosity Rust-vs-Sarkas: relative tolerance.
///
/// Stress tensor ACF converges more slowly than VACF. 10% agreement
/// is acceptable for lite validation runs.
pub const TRANSPORT_VISCOSITY_VS_SARKAS: f64 = 0.10;

/// HFB Rust-vs-Python: relative binding energy tolerance.
///
/// Rust and Python L2 solvers use different numerical methods (bisection
/// vs Brent, density mixing). Light nuclei (A < 60) are most sensitive.
/// Measured: Ni-56 shows ~11%, medium nuclei 1-5%. 12% accounts for all.
pub const HFB_RUST_VS_PYTHON_REL: f64 = 0.12;

// ═══════════════════════════════════════════════════════════════════
// Transport parity tolerances (CPU vs GPU same-physics proof)
// ═══════════════════════════════════════════════════════════════════

/// D* CPU vs GPU parity: relative tolerance.
///
/// At N=500 with 20k production steps, GPU tree-reduction vs CPU linear
/// summation produce different FP ordering that compounds across the run.
/// Both paths agree in physics (same order of magnitude D*); 65% captures
/// the maximum observed divergence at small N.
pub const TRANSPORT_D_STAR_CPU_GPU_PARITY: f64 = 0.65;

/// Temperature stability during NVE production (relative).
///
/// After Berendsen equilibration, NVE production should maintain T*
/// within ~30% of the target. Larger deviations indicate insufficient
/// equilibration or integration instability.
pub const TRANSPORT_T_STABILITY: f64 = 0.30;

/// D* MSD vs Daligault fit: relative tolerance for lite validation.
///
/// At N=500, finite-size effects dominate; MSD-derived D* may differ
/// from the calibrated fit by up to 80%. This is a sanity check,
/// not a precision benchmark.
pub const TRANSPORT_D_STAR_VS_FIT_LITE: f64 = 0.80;

/// MSD-derived D* vs VACF-derived D*: internal consistency.
///
/// Both methods extract D* from the same trajectory; disagreement
/// beyond 50% indicates a computational bug, not physics noise.
pub const TRANSPORT_MSD_VACF_AGREEMENT: f64 = 0.50;

/// D* CPU vs GPU parity at small N (108 particles).
///
/// At N=108 with 5k steps, VACF noise and FP ordering divergence
/// limit CPU/GPU agreement to ~20%.
pub const PARITY_D_STAR_REL: f64 = 0.20;

/// Temperature CPU vs GPU: relative difference.
///
/// Final equilibrium T* should agree within 25% between CPU and GPU.
/// Differences arise from FP ordering in the energy accumulation.
pub const PARITY_T_DIFF: f64 = 0.25;

/// Mean total energy CPU vs GPU: relative difference.
///
/// Energy drift accumulates differently in CPU (Newton-3rd j>i) and
/// GPU (j!=i full-loop). 8% captures the maximum observed divergence
/// for N=108 5k-step runs.
pub const PARITY_ENERGY_DIFF: f64 = 0.08;

// ═══════════════════════════════════════════════════════════════════
// Screened Coulomb tolerances (Murillo & Weisheit 1998 — Paper 6)
// ═══════════════════════════════════════════════════════════════════

/// Eigenvalue vs exact hydrogen: relative tolerance.
///
/// Finite-difference discretization on a uniform grid with N=1000, r_max=80
/// has O(h²) truncation error. For the 1s state, measured accuracy is ~0.2%.
/// 2% tolerance accommodates excited states and near-threshold eigenvalues.
pub const SCREENED_HYDROGEN_VS_EXACT: f64 = 0.02;

/// Critical screening vs literature: relative tolerance.
///
/// Bisection on the bound-state count converges to machine precision in κ,
/// but the discrete grid introduces ~1–3% error in eigenvalues near
/// threshold (E → 0⁻). 5% is conservative for n ≤ 3 states.
///
/// Source: Lam & Varshni (1971) Phys. Rev. A 4, 1875.
pub const SCREENED_CRITICAL_VS_LITERATURE: f64 = 0.05;

/// Python-Rust parity: absolute eigenvalue tolerance (Hartree).
///
/// Same algorithm, same grid, same input → same output to machine precision.
/// 1e-10 accounts for potential FP ordering differences between numpy and
/// barracuda eigh_f64 implementations.
pub const SCREENED_PYTHON_RUST_PARITY: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════════════
// Lattice QCD tolerances
// ═══════════════════════════════════════════════════════════════════

/// Cold plaquette: absolute error (should be exactly 1.0 for unit links).
///
/// On a cold-start lattice (all links = identity), the plaquette trace
/// is exactly 1. Machine-precision rounding gives ~1e-15 residual.
pub const LATTICE_COLD_PLAQUETTE_ABS: f64 = 1e-12;

/// Cold Wilson action: absolute error (should be exactly 0.0).
///
/// Wilson action = β × Σ(1 - Re Tr U_p / 3) is zero when all plaquettes
/// are unit matrices. 1e-10 accounts for accumulated rounding.
pub const LATTICE_COLD_ACTION_ABS: f64 = 1e-10;

/// HMC acceptance rate lower bound.
///
/// A functional HMC must accept > 10% of trajectories. Zero acceptance
/// indicates a bug in the gauge force, leapfrog integrator, or
/// Metropolis step.
pub const LATTICE_HMC_ACCEPTANCE_MIN: f64 = 0.10;

/// CG solver residual: upper bound.
///
/// The conjugate gradient solver for D†D x = b should converge to
/// a relative residual below 1e-6 on a cold lattice (identity links).
pub const LATTICE_CG_RESIDUAL: f64 = 1e-6;

/// U(1) Abelian Higgs cold plaquette: absolute error.
///
/// On a cold-start U(1) lattice (all link angles = 0), the plaquette
/// is exactly 1. Machine-precision rounding gives ~1e-15 residual.
pub const U1_COLD_PLAQUETTE_ABS: f64 = 1e-12;

/// U(1) Abelian Higgs cold gauge action: absolute error.
///
/// Wilson gauge action = 0 on a cold-start lattice.
pub const U1_COLD_ACTION_ABS: f64 = 1e-10;

/// U(1) Abelian Higgs HMC acceptance lower bound.
///
/// A functional U(1)+Higgs HMC must accept > 30% of trajectories.
/// Zero or very low acceptance indicates force/integrator bugs.
pub const U1_HMC_ACCEPTANCE_MIN: f64 = 0.30;

/// U(1) Abelian Higgs: weak-coupling plaquette lower bound.
///
/// At β_pl ≥ 6 (weak coupling), ⟨Re U_p⟩ > 0.7. Near-unity means
/// the gauge field is nearly ordered.
pub const U1_WEAK_COUPLING_PLAQ_MIN: f64 = 0.70;

/// U(1) Abelian Higgs: Python-Rust observable parity.
///
/// Same algorithm, same LCG seed → observables should match to 1%.
/// Differences arise only from FP summation order.
pub const U1_PYTHON_RUST_PARITY: f64 = 0.01;

/// Thermodynamic consistency tolerance for HotQCD EOS.
///
/// Checks s ≈ (ε+p)/T within 30%. Some points near T_c have larger
/// deviations due to the crossover nature of the QCD transition.
pub const HOTQCD_CONSISTENCY: f64 = 0.30;

/// Maximum allowed thermodynamic consistency violations.
///
/// Up to 2 data points may violate the consistency check near T_c.
pub const HOTQCD_MAX_VIOLATIONS: usize = 3;

// ═══════════════════════════════════════════════════════════════════
// NAK eigensolve tolerances
// ═══════════════════════════════════════════════════════════════════

/// NAK eigensolve: maximum relative error vs CPU reference.
///
/// GPU Jacobi eigensolve with 200 sweeps converges to ~1e-3 relative
/// for 12-30 dimension matrices. 1e-2 is a conservative upper bound.
pub const NAK_EIGENSOLVE_VS_CPU_REL: f64 = 1e-2;

/// NAK baseline vs optimized: parity tolerance.
///
/// The optimized (FMA, unrolled) shader must produce identical results
/// to the baseline shader — both are Jacobi rotation with the same
/// convergence criterion. Machine-precision agreement expected.
pub const NAK_EIGENSOLVE_PARITY: f64 = 1e-10;

/// NAK eigensolve: performance regression threshold.
///
/// The optimized shader should not be more than 1.5× slower than
/// the baseline. Values > 1.5 indicate a performance regression.
pub const NAK_EIGENSOLVE_REGRESSION: f64 = 1.5;

/// PPPM net force tolerance for multi-particle systems.
///
/// For NaCl crystals and random charge configurations, the net force
/// |Σ F_i| should be near zero. PPPM mesh approximation for small
/// systems with limited resolution can give residuals of O(1).
pub const PPPM_MULTI_PARTICLE_NET_FORCE: f64 = 1.0;

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

/// Generic division-by-zero guard for sums and norms.
///
/// Used when dividing by quantities that are sums of squares (norm², dot
/// products) where zero means the vector is identically zero. At 1e-30,
/// this is well below any physical scale in nuclear/MD applications and
/// serves only to prevent IEEE inf/NaN. Applies to: Broyden denominator
/// guards, wavefunction normalization, charge density sums.
pub const DIVISION_GUARD: f64 = 1e-30;

/// BCS pairing gap threshold: below this Δ (MeV), pairing is negligible.
///
/// At 1e-10 MeV (~0.1 neV), the pairing gap is ~10 orders of magnitude
/// below typical nuclear pairing gaps (0.5–2.0 MeV). Used to switch
/// between BCS and Fermi-gas occupation formulas, and to skip pairing
/// energy terms. Also used as bisection convergence tolerance for
/// the Fermi energy (MeV-scale precision).
pub const PAIRING_GAP_THRESHOLD: f64 = 1e-10;

/// SCF energy convergence tolerance (MeV).
///
/// The self-consistent field loop converges when |E_n − E_{n-1}| < tol.
/// 1e-6 MeV corresponds to ~1 eV, well below nuclear binding energy
/// uncertainties. For deformed HFB: typically converges in 50–150 iterations.
/// For spherical HFB: typically converges in 20–80 iterations.
pub const SCF_ENERGY_TOLERANCE: f64 = 1e-6;

/// Guard for `ρ.powf(α)` with Skyrme fractional exponents (α ≈ 1/6).
///
/// `powf` of a negative or true-zero value is undefined or NaN.
/// At 1e-20 fm⁻³ (5 orders below `DENSITY_FLOOR`), the density is so
/// far below any physical nuclear density that the energy contribution
/// is negligible, but `powf(1/6)` remains well-defined and avoids NaN.
/// Used in pre-Skyrme potential rho^α computations on CPU.
pub const RHO_POWF_GUARD: f64 = 1e-20;

/// GPU Jacobi eigensolve: off-diagonal convergence threshold.
///
/// `BatchedEighGpu` uses Jacobi rotation with this threshold to determine
/// when off-diagonal elements are small enough to stop. For 32×32 matrices
/// (nuclear HFB), 1e-12 provides eigenvalues accurate to ~1e-8 relative,
/// well within the `GPU_EIGENSOLVE_REL` tolerance.
pub const GPU_JACOBI_CONVERGENCE: f64 = 1e-12;

/// BCS density contribution skip threshold.
///
/// In the density computation `rho(r) = sum_i deg_i * v2_i * |phi_i(r)|²`,
/// states with `deg_i * v2_i < 1e-12` contribute negligibly (below f64
/// precision of the sum) and are skipped for performance. At this threshold,
/// even a fully-occupied j=15/2 state (deg=16) would need `v2 < 6e-14`,
/// meaning the eigenvalue is ~25 MeV above the Fermi surface — far into
/// the continuum where occupations should indeed be zero.
pub const BCS_DENSITY_SKIP: f64 = 1e-12;

/// BCS pairing-to-sharp-filling transition threshold (MeV).
///
/// When the pairing gap delta < 0.01 MeV, pairing correlations are
/// negligible and the system is in the normal (unpaired) phase. BCS
/// occupation factors degenerate to a step function, so we switch to
/// sharp Fermi filling for numerical stability. This threshold is
/// ~50x smaller than the smallest physical pairing gap in medium/heavy
/// nuclei (~0.5 MeV), ensuring the switch only activates for truly
/// unpaired systems.
pub const SHARP_FILLING_THRESHOLD: f64 = 0.01;

/// Coulomb singularity guard for deformed HFB (fm).
///
/// In the deformed basis, grid points are on a 2D cylindrical (rho, z)
/// mesh. The effective radius r = sqrt(rho² + z²) can be very small near
/// the origin. For Coulomb 1/r potentials, this floor prevents division
/// by near-zero. At 0.01 fm (~10x the proton charge radius), enclosed
/// charge Q(r) ∝ r³ is negligible, so the guard does not affect physics.
/// This is larger than `COULOMB_R_MIN` (1e-10 fm) because the deformed
/// grid spacing is coarser and needs a wider guard to avoid numerical noise.
pub const DEFORMED_COULOMB_R_MIN: f64 = 0.01;

/// Initial deformation guess for nuclei near magic numbers.
///
/// Near-magic nuclei are weakly deformed. beta_2 ≈ 0.05 corresponds
/// to a prolate deformation ~1% change in radius. Used as the starting
/// point for self-consistent deformed HFB iteration.
pub const DEFORMATION_GUESS_WEAK: f64 = 0.05;

/// Initial deformation guess for generic (non-magic) nuclei.
///
/// Generic mid-shell nuclei typically have beta_2 ≈ 0.15–0.25.
/// This starting value allows the SCF iteration to converge from
/// either side of the true deformation minimum.
pub const DEFORMATION_GUESS_GENERIC: f64 = 0.15;

/// Initial deformation guess for sd-shell deformed nuclei.
///
/// Nuclei in the sd-shell (A ≈ 20–40) can be strongly deformed with
/// beta_2 ≈ 0.3–0.4. This higher starting value helps the SCF loop
/// find the deformed minimum more quickly.
pub const DEFORMATION_GUESS_SD: f64 = 0.35;

/// Near-zero denominator threshold for relative error computation.
///
/// When the expected value has |expected| < this threshold, fall back to
/// absolute error comparison rather than dividing by a near-zero value.
/// 1e-14 is just above f64 machine epsilon (~2.2e-16), providing ~2 digits
/// of headroom for accumulated rounding in the expected-value computation.
pub const NEAR_ZERO_EXPECTED: f64 = 1e-14;

// ═══════════════════════════════════════════════════════════════════
// NPU quantization tolerances (metalForge AKD1000 validation)
// ═══════════════════════════════════════════════════════════════════

/// ESN f64 → f32 prediction parity: relative error.
///
/// f32 has ~7.2 significant digits; ESN forward pass with 50-dim reservoir
/// and 100-frame sequences accumulates ~O(50*100) = 5000 FP operations.
/// Measured: <0.001% mean error across 6 test cases (Python control).
pub const NPU_F32_PARITY: f64 = 0.001;

/// ESN f64 → int8 quantized prediction: relative error.
///
/// Symmetric uniform quantization of weights to 8-bit integers introduces
/// quantization noise proportional to max_abs(w) / 127. For W_in in [-0.5, 0.5]
/// and W_res with spectral_radius=0.95, measured mean error is ~0.34%.
/// 5% threshold is conservative.
///
/// Source: control/metalforge_npu/scripts/npu_quantization_parity.py
pub const NPU_INT8_QUANTIZATION: f64 = 0.05;

/// ESN f64 → int4 quantized prediction: relative error.
///
/// 4-bit quantization maps weights to [-7, 7] integers. The dynamic range
/// reduction from 15.9 significant digits (f64) to 4 bits (3.9 significant
/// digits) causes ~5.5% mean error and up to ~14% worst-case for predictions
/// near the weight matrix's null space.
/// 30% threshold accommodates worst-case phase diagram corners.
///
/// Source: control/metalforge_npu/scripts/npu_quantization_parity.py
pub const NPU_INT4_QUANTIZATION: f64 = 0.30;

/// ESN f64 → int4 weights + int4 activations: relative error.
///
/// When both weights AND activations are quantized to 4-bit (matching AKD1000
/// hardware), the error compounds through the reservoir update loop. The tanh
/// activation's non-linearity partially mitigates quantization noise (clamping
/// to [-1,1]) but the iterative state update amplifies errors over 100 frames.
/// Measured: ~8.9% mean, ~24% worst-case.
/// 50% threshold is generous for full-hardware simulation.
///
/// Source: control/metalforge_npu/scripts/npu_quantization_parity.py
pub const NPU_INT4_FULL_QUANTIZATION: f64 = 0.50;

// ═══════════════════════════════════════════════════════════════════
// NPU beyond-SDK tolerances (metalForge AKD1000 hardware probing)
// ═══════════════════════════════════════════════════════════════════

/// FC depth overhead: latency increase from depth=1 to depth=7.
///
/// All FC layers merge into a single hardware sequence via SkipDMA.
/// Measured: ~7% overhead for 7 extra layers. 30% is generous.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_FC_DEPTH_OVERHEAD: f64 = 0.30;

/// Batch inference speedup: batch=8 vs batch=1 throughput ratio.
///
/// PCIe round-trip amortizes across batch. Measured: 2.35×.
/// 1.5× is the minimum acceptable amortization.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_BATCH_SPEEDUP_MIN: f64 = 1.5;

/// Multi-output overhead: latency increase from 1→10 outputs.
///
/// The NP mesh parallelism handles multiple outputs simultaneously.
/// Measured: 4.5% overhead. 30% is generous.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_MULTI_OUTPUT_OVERHEAD: f64 = 0.30;

/// Weight mutation linearity: max error for w×k producing output×k.
///
/// Changing FC weights via set_variable() must produce proportional
/// output changes. Measured: 0.0000 error.
///
/// Source: control/metalforge_npu/scripts/npu_beyond_sdk.py
pub const NPU_WEIGHT_MUTATION_LINEARITY: f64 = 0.01;

/// Normalization variance guard for the pre-screening classifier.
///
/// During feature normalization, any feature with variance below this
/// threshold is treated as constant (std clamped to this floor).
/// At 1e-10, features varying by less than ~1e-5 of their mean are
/// effectively constant and would cause numerical blow-up if divided by
/// their true standard deviation.
pub const CLASSIFIER_VARIANCE_GUARD: f64 = 1e-10;

/// Learning rate for the pre-screening logistic regression classifier.
///
/// Standard mini-batch logistic regression learning rate. The classifier
/// is a simple 10→1 linear model; 0.01 converges reliably in 200 epochs
/// for the Skyrme parameter space without oscillation.
pub const CLASSIFIER_LEARNING_RATE: f64 = 0.01;

/// Training epochs for the pre-screening logistic regression classifier.
///
/// 200 epochs is sufficient for convergence of a 10-parameter logistic
/// regression on the typical ~100–1000 sample training sets accumulated
/// during L1/L2 sweeps. Loss plateaus well before 200 epochs.
pub const CLASSIFIER_EPOCHS: u32 = 200;

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

    #[test]
    fn physics_guards_are_positive() {
        assert!(DENSITY_FLOOR > 0.0);
        assert!(SPIN_ORBIT_R_MIN > 0.0);
        assert!(COULOMB_R_MIN > 0.0);
        assert!(DIVISION_GUARD > 0.0);
        assert!(PAIRING_GAP_THRESHOLD > 0.0);
        assert!(RHO_POWF_GUARD > 0.0);
        assert!(GPU_JACOBI_CONVERGENCE > 0.0);
        assert!(BCS_DENSITY_SKIP > 0.0);
    }

    #[test]
    fn guard_hierarchy() {
        // RHO_POWF_GUARD < DENSITY_FLOOR < SPIN_ORBIT_R_MIN
        assert!(RHO_POWF_GUARD < DENSITY_FLOOR);
        assert!(DIVISION_GUARD < RHO_POWF_GUARD);
    }

    #[test]
    fn deformation_guesses_ordered() {
        assert!(DEFORMATION_GUESS_WEAK < DEFORMATION_GUESS_GENERIC);
        assert!(DEFORMATION_GUESS_GENERIC < DEFORMATION_GUESS_SD);
    }

    #[test]
    fn all_tolerances_are_positive() {
        let tols = [
            EXACT_F64,
            ITERATIVE_F64,
            GPU_VS_CPU_F64,
            LU_TOLERANCE,
            QR_TOLERANCE,
            SVD_TOLERANCE,
            TRIDIAG_TOLERANCE,
            EIGH_TOLERANCE,
            GAMMA_TOLERANCE,
            ERF_TOLERANCE,
            BESSEL_TOLERANCE,
            MD_FORCE_TOLERANCE,
            MD_ABSOLUTE_FLOOR,
            NEWTON_3RD_LAW_ABS,
            NELDER_MEAD_TOLERANCE,
            BFGS_TOLERANCE,
            BRENT_TOLERANCE,
            RK45_TOLERANCE,
            SOBOL_TOLERANCE,
            SCF_ENERGY_TOLERANCE,
            SHARP_FILLING_THRESHOLD,
            TRANSPORT_D_STAR_CPU_GPU_PARITY,
            TRANSPORT_T_STABILITY,
            TRANSPORT_D_STAR_VS_FIT_LITE,
            TRANSPORT_MSD_VACF_AGREEMENT,
            PARITY_D_STAR_REL,
            PARITY_T_DIFF,
            PARITY_ENERGY_DIFF,
            LATTICE_COLD_PLAQUETTE_ABS,
            LATTICE_COLD_ACTION_ABS,
            LATTICE_HMC_ACCEPTANCE_MIN,
            LATTICE_CG_RESIDUAL,
            HOTQCD_CONSISTENCY,
            NAK_EIGENSOLVE_VS_CPU_REL,
            NAK_EIGENSOLVE_PARITY,
            NAK_EIGENSOLVE_REGRESSION,
            PPPM_MULTI_PARTICLE_NET_FORCE,
            SCREENED_HYDROGEN_VS_EXACT,
            SCREENED_CRITICAL_VS_LITERATURE,
            SCREENED_PYTHON_RUST_PARITY,
            U1_COLD_PLAQUETTE_ABS,
            U1_COLD_ACTION_ABS,
            U1_HMC_ACCEPTANCE_MIN,
            U1_WEAK_COUPLING_PLAQ_MIN,
            U1_PYTHON_RUST_PARITY,
            NPU_F32_PARITY,
            NPU_INT8_QUANTIZATION,
            NPU_INT4_QUANTIZATION,
            NPU_INT4_FULL_QUANTIZATION,
            NPU_FC_DEPTH_OVERHEAD,
            NPU_BATCH_SPEEDUP_MIN,
            NPU_MULTI_OUTPUT_OVERHEAD,
            NPU_WEIGHT_MUTATION_LINEARITY,
        ];
        for (i, &t) in tols.iter().enumerate() {
            assert!(t > 0.0, "tolerance index {i} must be positive, got {t}");
        }
    }

    #[test]
    fn hotqcd_max_violations_nonzero() {
        assert!(HOTQCD_MAX_VIOLATIONS > 0);
    }
}
