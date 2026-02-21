// SPDX-License-Identifier: AGPL-3.0-only

//! Physics-domain tolerances: screened Coulomb, nuclear EOS acceptance criteria,
//! and numerical singularity guards for nuclear/MD computations.

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

/// Screened Coulomb: ion-sphere κ_reduced vs √3 analytical value.
///
/// The ion-sphere model gives κ_reduced = √3 exactly. The numerical
/// computation (eigenvalue + potential) agrees to machine precision.
pub const SCREENED_ION_SPHERE_SQRT3_ABS: f64 = 1e-14;

/// Screened Coulomb: short-range to ion-sphere screening limit.
///
/// As κ → ∞, the screened potential approaches the ion-sphere limit.
/// The numerical transition check verifies monotonic eigenvalue decrease
/// with increasing κ. 0.05 absolute tolerance on the normalized overlap.
pub const SCREENED_SP_TO_IS_LIMIT: f64 = 0.05;

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

/// HFB Rust-vs-experiment: relative binding energy tolerance.
///
/// The L2 HFB solver with SLy4 parametrization reproduces experimental
/// binding energies to ~5-10% for medium-mass nuclei. Missing physics
/// (deformation, continuum, 3-body) limits accuracy. Light nuclei like
/// Ni-56 show up to ~11% deviation from experiment (AME2020).
///
/// Measurement: validate_nuclear_eos Phase 4, Feb 2026.
///   SLy4 on 6 HFB test nuclei. Worst: Ni-56 → 11.2% vs AME2020.
///   15% = max + 3.8% margin for numerical method differences.
pub const HFB_RUST_VS_EXP_REL: f64 = 0.15;

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
