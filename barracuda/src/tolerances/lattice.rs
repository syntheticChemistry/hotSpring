// SPDX-License-Identifier: AGPL-3.0-only

//! Lattice QCD, GPU SpMV/Lanczos, and spectral theory tolerances.

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
/// Wilson action = β × Σ(1 - Re Tr `U_p` / 3) is zero when all plaquettes
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
/// At `β_pl` ≥ 6 (weak coupling), ⟨Re `U_p`⟩ > 0.7. Near-unity means
/// the gauge field is nearly ordered.
pub const U1_WEAK_COUPLING_PLAQ_MIN: f64 = 0.70;

/// U(1) Abelian Higgs: Python-Rust observable parity.
///
/// Same algorithm, same LCG seed → observables should match to 1%.
/// Differences arise only from FP summation order.
pub const U1_PYTHON_RUST_PARITY: f64 = 0.01;

/// U(1) Abelian Higgs: strong-coupling plaquette upper bound.
///
/// At `β_pl=0.5` (strong coupling), ⟨Re `U_p`⟩ < 0.5. Near-zero means
/// the gauge field is strongly disordered.
pub const U1_STRONG_COUPLING_PLAQ_MAX: f64 = 0.50;

/// U(1) Abelian Higgs: Higgs condensate phase transition.
///
/// The Higgs condensate ⟨|φ|⟩ undergoes a crossover between the
/// Coulomb (small ⟨|φ|⟩) and Higgs (large ⟨|φ|⟩) phases. 1.5 is
/// the boundary below which the system is in the Coulomb phase at
/// strong gauge coupling.
pub const U1_HIGGS_CONDENSATE_BOUNDARY: f64 = 1.5;

/// U(1) Abelian Higgs: condensate vs coupling monotonicity tolerance.
///
/// The condensate should increase with `κ_higgs` (Higgs self-coupling).
/// 0.15 relative tolerance on the monotonicity check accommodates
/// finite-size fluctuations on small (L=16) lattices.
pub const U1_CONDENSATE_MONOTONICITY: f64 = 0.15;

/// Thermodynamic consistency tolerance for `HotQCD` EOS.
///
/// Checks s ≈ (ε+p)/T within 30%. Some points near `T_c` have larger
/// deviations due to the crossover nature of the QCD transition.
pub const HOTQCD_CONSISTENCY: f64 = 0.30;

/// Maximum allowed thermodynamic consistency violations.
///
/// Up to 3 data points may violate the consistency check near `T_c`,
/// where the QCD crossover produces large ∂s/∂T gradients that amplify
/// the discretization error in s ≈ (ε+p)/T. Measured: 2 violations
/// typical, 3 worst-case on coarse interpolation grids.
pub const HOTQCD_MAX_VIOLATIONS: usize = 3;

// ═══════════════════════════════════════════════════════════════════
// Production QCD β-scan tolerances
// ═══════════════════════════════════════════════════════════════════

/// β-scan plaquette monotonicity: relative margin.
///
/// The average plaquette `<P>` must increase monotonically with β.
/// At each β, the plaquette is averaged over measurement trajectories
/// after thermalization. Zero tolerance — any non-monotonic pair fails.
pub const BETA_SCAN_PLAQUETTE_MONOTONICITY: f64 = 0.0;

/// β-scan Polyakov loop: confined-phase upper bound.
///
/// In the confined phase (β < β_c ≈ 5.69 for SU(3) on N_t=4),
/// the spatial average of |L| should be suppressed. On a 4^4 lattice,
/// finite-size effects give |L| ~ 0.1-0.3 even in confinement.
/// 0.40 allows for fluctuations on small volumes.
pub const BETA_SCAN_CONFINED_POLYAKOV_MAX: f64 = 0.40;

/// β-scan acceptance rate: minimum for all β values.
///
/// Every β point in the scan must have HMC acceptance > 30%.
/// Lower than 30% suggests the step size is too large for that coupling.
pub const BETA_SCAN_ACCEPTANCE_MIN: f64 = 0.30;

/// β-scan plaquette: Python-Rust parity.
///
/// Same algorithm, same LCG seed → same plaquette trajectory. The
/// parity is limited by FP summation order differences between NumPy
/// and Rust iterators. On 4^4 (256 sites), 1% relative is achievable.
pub const BETA_SCAN_PYTHON_RUST_PLAQUETTE_PARITY: f64 = 0.01;

/// β-scan 8^4 scaling: plaquette should approach 4^4 at large β.
///
/// At β ≥ 6.0 (weak coupling), finite-size effects vanish and the
/// 4^4 and 8^4 plaquettes converge. 5% relative tolerance accounts
/// for residual finite-volume corrections.
pub const BETA_SCAN_SCALING_PARITY: f64 = 0.05;

/// Known plaquette at β=6.0 on 4^4: ~0.594.
///
/// Bali et al. (1993) and Necco & Sommer (2002) give the continuum-limit
/// plaquette. On a finite 4^4 lattice, measured value is 0.55-0.61 with
/// O(100) trajectories. 10% relative tolerance.
pub const BETA6_PLAQUETTE_REF: f64 = 0.594;

/// Tolerance for β=6.0 plaquette reference comparison.
pub const BETA6_PLAQUETTE_TOLERANCE: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════
// Dynamical fermion QCD tolerances (Paper 10)
// ═══════════════════════════════════════════════════════════════════

/// Dynamical HMC acceptance rate lower bound.
///
/// Naive staggered fermion HMC without multi-timescale integration or
/// mass preconditioning has low acceptance on coarse lattices due to
/// the stiff fermion force. Any nonzero acceptance proves the Metropolis
/// step is functioning. Production efficiency requires Omelyan integrator
/// and Hasenbusch mass splitting (future optimization).
pub const DYNAMICAL_HMC_ACCEPTANCE_MIN: f64 = 0.01;

/// Dynamical plaquette: must remain physical (0 < P < 1).
///
/// Fermion backreaction modifies the plaquette relative to quenched,
/// but it must remain in (0, 1) for any valid SU(3) configuration.
pub const DYNAMICAL_PLAQUETTE_MAX: f64 = 1.0;

/// Dynamical fermion action: must be positive.
///
/// S_F = φ†(D†D)⁻¹φ ≥ 0 since D†D is positive-definite. A negative
/// value indicates a CG convergence failure or sign error.
pub const DYNAMICAL_FERMION_ACTION_MIN: f64 = 0.0;

/// CG convergence: all solves must converge within max iterations.
///
/// The CG solver for (D†D)x = φ should converge at the requested
/// tolerance within 5000 iterations on 4^4 lattices.
pub const DYNAMICAL_CG_MAX_ITER: usize = 5000;

/// Dynamical plaquette vs quenched: fermion backreaction changes plaquette.
///
/// At the same β, dynamical fermions shift the plaquette relative to
/// quenched. The shift should not exceed 0.15 on 4^4 with light quarks
/// (m=0.1). A larger shift indicates incorrect fermion force.
pub const DYNAMICAL_VS_QUENCHED_SHIFT_MAX: f64 = 0.15;

/// Polyakov loop confined-phase upper bound (dynamical).
///
/// In the confined phase with dynamical fermions, |L| is suppressed
/// but string breaking allows slightly larger values than quenched.
/// 0.5 is generous for 4^4 with light quarks.
pub const DYNAMICAL_CONFINED_POLYAKOV_MAX: f64 = 0.50;

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

// ═══════════════════════════════════════════════════════════════════
// GPU lattice QCD validation (CG, Dirac, SpMV)
// ═══════════════════════════════════════════════════════════════════

/// GPU CG solver: cold-lattice solution parity (GPU vs CPU).
///
/// On a cold (ordered) SU(3) lattice, the CG solver is well-conditioned.
/// GPU and CPU f64 solutions should agree to ~1e-6 relative (limited by
/// FP summation order differences in dot products).
pub const LATTICE_GPU_CG_COLD_PARITY: f64 = 1e-6;

/// GPU CG solver: hot-lattice solution parity (GPU vs CPU).
///
/// On a hot (random) lattice the condition number is higher and FP
/// accumulation differences between GPU parallel reduction and CPU
/// sequential sum become more visible. 1e-4 relative is achievable.
pub const LATTICE_GPU_CG_HOT_PARITY: f64 = 1e-4;

/// GPU CG: D†D x ≈ b residual verification.
///
/// After CG converges, reconstructing b' = D†D x and comparing to the
/// original b gives a residual bounded by the CG tolerance times the
/// condition number. 1e-7 is conservative for nuclear lattice sizes.
pub const LATTICE_CG_VERIFY_RESIDUAL: f64 = 1e-7;

/// GPU staggered Dirac: zero-input absolute error.
///
/// D*0 must equal 0 exactly; any deviation is a shader indexing bug.
/// Machine epsilon for f64 is 2.22e-16; 1e-15 allows a few ULP.
pub const LATTICE_DIRAC_ZERO_INPUT_ABS: f64 = 1e-15;

/// GPU staggered Dirac: cold-lattice parity (GPU vs CPU).
///
/// Cold lattice: ordered gauge field, low condition number. GPU and CPU
/// Dirac apply should agree to ~1e-14 (near machine epsilon, limited
/// only by FMA vs separate multiply-add).
pub const LATTICE_DIRAC_COLD_PARITY: f64 = 1e-14;

/// GPU staggered Dirac: hot-lattice parity (GPU vs CPU).
///
/// Hot lattice: random gauge field, higher condition number. The GPU
/// parallel summation order differs from CPU sequential, giving ~1e-13
/// max component-wise error.
pub const LATTICE_DIRAC_HOT_PARITY: f64 = 1e-13;

// ═══════════════════════════════════════════════════════════════════
// GPU SpMV and Lanczos eigensolve validation
// ═══════════════════════════════════════════════════════════════════

/// GPU `SpMV`: identity matrix absolute error.
///
/// I*x = x must hold to machine precision. Any deviation indicates
/// a CSR indexing or buffer layout bug in the WGSL shader.
pub const SPMV_IDENTITY_ABS: f64 = 1e-15;

/// GPU `SpMV`: general matrix GPU-vs-CPU parity.
///
/// For Anderson model and lattice Hamiltonians, GPU CSR `SpMV` matches
/// CPU reference to ~1e-14 (near machine epsilon). The parallel
/// reduction per row introduces at most 1-2 ULP of rounding difference.
pub const SPMV_GPU_VS_CPU_ABS: f64 = 1e-14;

/// GPU `SpMV`: iterated product (A²x) error accumulation.
///
/// Two successive `SpMV` applications accumulate rounding errors; the
/// tolerance is ~10× the single-pass tolerance.
pub const SPMV_ITERATED_ABS: f64 = 1e-13;

/// GPU Lanczos: β breakdown detection threshold.
///
/// In the Lanczos iteration, β_{k+1} = ||w|| measures the norm of the
/// new Krylov vector. When β < 1e-14, the Krylov subspace is (near-)
/// invariant and the iteration has converged or broken down.
pub const LANCZOS_BREAKDOWN_THRESHOLD: f64 = 1e-14;

/// GPU Lanczos: eigenvalue GPU-vs-CPU parity.
///
/// Full-spectrum Lanczos eigenvalues from GPU `SpMV` inner loop match CPU
/// Lanczos to ~1e-10. The larger tolerance (vs `SpMV`) reflects error
/// accumulation over O(N) Lanczos iterations, each with GPU `SpMV` and
/// reorthogonalization.
pub const LANCZOS_EIGENVALUE_GPU_PARITY: f64 = 1e-10;

// ═══════════════════════════════════════════════════════════════════
// Spectral theory tolerances (Anderson, Lanczos, Hofstadter)
// ═══════════════════════════════════════════════════════════════════

/// Lanczos tridiagonal eigenvalue: absolute error vs Sturm bisection.
///
/// Lanczos tridiagonalization + Sturm eigenvalue finding produces the
/// same tridiagonal matrix as direct construction, so eigenvalues agree
/// to machine precision. 1e-14 allows a few ULP of rounding.
pub const LANCZOS_TRIDIAG_EIGENVALUE_ABS: f64 = 1e-14;

/// Lanczos convergence: relative error for extreme eigenvalues.
///
/// Lanczos converges extremal eigenvalues first; for m=50 Lanczos
/// vectors on an N=100 tridiagonal, the top/bottom eigenvalues converge
/// to ~1e-6 relative error. Interior eigenvalues converge more slowly.
pub const LANCZOS_EXTREMAL_REL: f64 = 1e-6;

/// Anderson localization: IPR relative tolerance.
///
/// The inverse participation ratio IPR = `Σ|ψ_i|⁴` fluctuates between
/// disorder realizations. For W=2 (weak disorder), IPR ~ 1/N (extended);
/// for W=20 (strong disorder), IPR ~ O(1) (localized). 1e-8 absolute
/// tolerance for the tridiagonal→Lanczos eigenvalue comparison.
pub const ANDERSON_EIGENVALUE_ABS: f64 = 1e-8;

/// GOE level-spacing ratio: analytical ⟨r⟩ ≈ 0.5307.
///
/// For extended states in the GOE universality class, the mean adjacent
/// gap ratio is `r_GOE` = 4 - 2√3 ≈ 0.5307 (Atas et al., PRL 110, 2013).
/// Finite-size fluctuations at N=200 give ~0.04 spread.
pub const GOE_MEAN_R: f64 = 0.5307;

/// GOE level-spacing ratio: deviation tolerance.
///
/// At N=200 with 10 disorder realizations, ⟨r⟩ fluctuates by ~0.04
/// around the analytical value. 0.05 accommodates sample variance.
pub const GOE_DEVIATION_TOLERANCE: f64 = 0.05;

/// Poisson level-spacing ratio: analytical ⟨r⟩ ≈ 0.3863.
///
/// For localized states (Poisson level statistics), ⟨r⟩ = 2 ln 2 - 1
/// ≈ 0.3863. This is the strong-disorder limit.
pub const POISSON_MEAN_R: f64 = 0.3863;

/// Poisson level-spacing ratio: deviation tolerance.
///
/// At N=200 with strong disorder (W=20), ⟨r⟩ converges reliably.
/// 0.05 tolerance matches the GOE side.
pub const POISSON_DEVIATION_TOLERANCE: f64 = 0.05;

/// Sturm bisection: LDLT pivot guard to avoid division by zero.
///
/// In the Sturm sequence (LDLT factorization), when q = diagonal - λ
/// is nearly zero, we substitute ±1e-300 to prevent inf/NaN. This is
/// well below any physical eigenvalue scale and serves only to avoid
/// floating-point exceptions in the recurrence.
pub const TRIDIAG_STURM_PIVOT_GUARD: f64 = 1e-300;

/// Anderson 1D localization length: ln(2) analytical value tolerance.
///
/// At W=2 in 1D, the Lyapunov exponent γ ≈ W²/96. The localization
/// length diverges as ξ ~ 96/W². 0.02 absolute tolerance on the
/// normalized localization length ratio.
pub const ANDERSON_1D_LYAPUNOV_TOLERANCE: f64 = 0.02;

/// Hofstadter butterfly: energy band symmetry tolerance.
///
/// The Hofstadter Hamiltonian at rational flux p/q has spectrum
/// symmetric about E=0: |`E_min` + `E_max`| should be near zero.
/// Finite q gives max asymmetry of ~0.5 for edge-of-band states.
pub const HOFSTADTER_SYMMETRY_TOLERANCE: f64 = 0.5;
