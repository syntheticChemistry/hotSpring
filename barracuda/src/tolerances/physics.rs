// SPDX-License-Identifier: AGPL-3.0-or-later

//! Physics-domain tolerances: screened Coulomb, nuclear EOS acceptance criteria,
//! and numerical singularity guards for nuclear/MD computations.

// ═══════════════════════════════════════════════════════════════════
// Screened Coulomb tolerances (Murillo & Weisheit 1998 — Paper 6)
// ═══════════════════════════════════════════════════════════════════

/// Eigenvalue vs exact hydrogen: relative tolerance.
///
/// Finite-difference discretization on a uniform grid with N=1000, `r_max=80`
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
/// barracuda `eigh_f64` implementations.
pub const SCREENED_PYTHON_RUST_PARITY: f64 = 1e-10;

/// Screened Coulomb: ion-sphere `κ_reduced` vs √3 analytical value.
///
/// The ion-sphere model gives `κ_reduced` = √3 exactly. The numerical
/// computation (eigenvalue + potential) agrees to machine precision.
pub const SCREENED_ION_SPHERE_SQRT3_ABS: f64 = 1e-14;

/// Screened Coulomb: short-range to ion-sphere screening limit.
///
/// As κ → ∞, the screened potential approaches the ion-sphere limit.
/// The numerical transition check verifies monotonic eigenvalue decrease
/// with increasing κ. 0.05 absolute tolerance on the normalized overlap.
pub const SCREENED_SP_TO_IS_LIMIT: f64 = 0.05;

/// Screened Coulomb: minimum |`E_He+/E_H`| for Z² scaling check.
///
/// At κ=0, E ∝ −Z² so He+ (Z=2) is 4× deeper than H (Z=1).
/// Finite grid gives ratio ~3.9–4.0; 3.5 is conservative lower bound.
pub const SCREENED_Z2_SCALING_MIN_RATIO: f64 = 3.5;

// ═══════════════════════════════════════════════════════════════════
// TTM (Two-Temperature Model) Paper 2 — laser-plasma equilibration
// ═══════════════════════════════════════════════════════════════════

/// TTM equilibrium temperature vs Python control: relative tolerance.
///
/// Spitzer-based 0D model vs SMT ( Stanton–Murillo) in Python.
/// Different transport models → ~10–20% difference in `T_eq`. 20% accommodates.
pub const TTM_EQUILIBRIUM_T_REL: f64 = 0.20;

/// TTM Helium: equilibrium temperature vs Python control — relative tolerance.
///
/// 50% tolerance accounts for fundamental model mismatch between Spitzer Z=1
/// collision frequency (gives 15150 K) and SMT local-field model used in Python
/// control (gives 10700 K). The light-species (Z=2, 4 amu) amplifies
/// collision-frequency model sensitivity. See control/ttm/scripts/run_local_model.py
/// for the Python baseline.
pub const TTM_HELIUM_EQUILIBRIUM_T_REL: f64 = 0.50;

/// TTM energy conservation: relative drift allowed.
///
/// RK4 integration; total energy E = Ce·Te + Ci·Ti should be conserved.
/// 1% drift acceptable for long runs.
pub const TTM_ENERGY_DRIFT_REL: f64 = 0.01;

/// TTM equilibration-time detection: |Te − Ti| threshold (Kelvin).
///
/// `find_equilibration_time` marks equilibrium when electron and ion temperatures
/// differ by less than this absolute value (see `validate_ttm`).
pub const TTM_EQUILIBRATION_CLOSE_K: f64 = 500.0;

/// TTM trajectory monotonicity: allowed Te/Ti step inversion (Kelvin-scale noise floor).
///
/// RK4 and print sampling can produce ≤1e-6 K apparent non-monotonicity; the
/// harness treats adjacent steps within this slop as still monotonic.
pub const TTM_TEMPERATURE_TRAJECTORY_SLOP_K: f64 = 1e-6;

/// Militzer FPEOS: grid-point and interpolation relative error on P and E.
///
/// Table lookup and bilinear interpolation should match embedded control values
/// to 1% relative (`validate_fpeos`).
pub const FPEOS_GRID_POINT_REL: f64 = 0.01;

/// Militzer FPEOS: He vs H pressure at identical table indices.
///
/// Helium pressure must not exceed hydrogen by more than this factor (1% margin)
/// at each shared grid point.
pub const FPEOS_HE_TO_H_PRESSURE_RATIO_MAX: f64 = 1.01;

/// Militzer FPEOS: thermodynamic consistency max |ΔP/P| from numerical derivatives.
///
/// s ≈ (ε+P)/T style checks accumulate finite-difference noise; 2.0 is an upper
/// bound on the reported inconsistency metric (`validate_fpeos`).
pub const FPEOS_THERMO_CONSISTENCY_MAX: f64 = 2.0;

/// Kinetic-fluid BGK: species mass conservation relative error.
pub const KINETIC_FLUID_BGK_MASS_REL: f64 = 1e-8;

/// Kinetic-fluid BGK: total momentum conservation relative error.
pub const KINETIC_FLUID_BGK_MOMENTUM_REL: f64 = 1e-10;

/// Kinetic-fluid BGK: temperature relaxation |T₁−T₂|/T scale (dimensionless).
pub const KINETIC_FLUID_TEMP_RELAXATION_REL: f64 = 0.01;

/// Kinetic-fluid BGK: equilibrium temperature absolute check vs analytic midpoint.
pub const KINETIC_FLUID_EQUILIBRIUM_T_ABS: f64 = 0.05;

/// Sod shock tube: mass and energy conservation relative error.
pub const KINETIC_FLUID_SOD_CONSERVATION_REL: f64 = 1e-10;

/// Sod shock tube: minimum physical density (lower bound check).
pub const KINETIC_FLUID_SOD_RHO_MIN: f64 = 0.1;

/// Sod shock tube: maximum physical density (upper bound check).
pub const KINETIC_FLUID_SOD_RHO_MAX: f64 = 1.1;

/// Coupled kinetic–fluid: mass / energy conservation relative error.
pub const KINETIC_FLUID_COUPLED_MASS_ENERGY_REL: f64 = 0.15;

/// Coupled kinetic–fluid: momentum conservation relative error.
pub const KINETIC_FLUID_COUPLED_MOMENTUM_REL: f64 = 0.25;

/// Coupled kinetic–fluid: interface |Δρ|/ρ₀ match metric.
pub const KINETIC_FLUID_INTERFACE_DENSITY_MATCH: f64 = 0.5;

/// Coupled kinetic–fluid: fluid-region density lower bound.
pub const KINETIC_FLUID_REGION_RHO_MIN: f64 = 0.5;

/// Coupled kinetic–fluid: fluid-region density upper bound.
pub const KINETIC_FLUID_REGION_RHO_MAX: f64 = 2.0;

/// GPU batched Mermin dielectric: f-sum quadrature vs CPU reference integral (relative).
///
/// GPU ω-quadrature of ∫ ω Im[1/ε] dω can differ from the CPU reference by up to ~6%
/// for Γ = 10–200, κ = 1–2 plasmas (`validate_gpu_dielectric`).
pub const DIELECTRIC_F_SUM_GPU_CPU_REL: f64 = 0.06;

/// Dielectric high-frequency / large-argument limit: small |W| or small |loss|.
///
/// Plasma dispersion |W(z)| at large |z| and high-ω dielectric loss should vanish.
/// Used for |W(ω→∞)| in `validate_dielectric` and |Im[1/ε]| at high ω in
/// `validate_gpu_dielectric`.
pub const DIELECTRIC_HIGH_FREQ_LIMIT_ABS: f64 = 0.01;

/// Plasma dispersion function at z=0: |Re W(0)−1| and |Im W(0)| absolute gate.
///
/// The series definition gives W(0)=1 and Im W(0)=0 analytically; the implementation
/// should match to ~machine precision. 1e-14 allows a few ULP beyond f64 epsilon
/// (`validate_chuna` Paper 44).
pub const DIELECTRIC_PLASMA_DISPERSION_W0_ABS: f64 = 1e-14;

/// Static Debye limit: relative agreement between Mermin static ε and analytic Debye.
///
/// `debye_screening` compares long-wavelength screening in the collisionless limit;
/// 1e-12 is tight enough to catch implementation errors while allowing quadrature
/// noise in κ-space (`validate_chuna` Paper 44).
pub const DIELECTRIC_DEBYE_SCREENING_REL: f64 = 1e-12;

/// Drude DC conductivity: relative agreement with σ = ωₚ²/(4πν).
///
/// The closed form is exact for the Drude model; 1e-13 bounds floating-point
/// composition of plasma frequency and collision rate (`validate_chuna` Paper 44).
pub const DIELECTRIC_DRUDE_CONDUCTIVITY_REL: f64 = 1e-13;

/// Dynamic structure factor: relative slack for “numerical negativity” near zero.
///
/// Quadrature noise can make S(k,ω) slightly negative when the true value is 0;
/// points with S ≥ −`tol`×max(S, floor) count as physically nonnegative (`validate_chuna`).
pub const DIELECTRIC_DSF_RELATIVE_NOISE_FLOOR: f64 = 1e-6;

/// Dynamic structure factor: floor inside max(S_max, floor) for positivity slack.
///
/// When all S are tiny, comparing to zero in relative terms would be ill-conditioned;
/// 1e-10 provides an absolute scale so the near-positivity test remains stable.
pub const DIELECTRIC_DSF_MAGNITUDE_FLOOR: f64 = 1e-10;

/// Dynamic structure factor: minimum fraction of ω samples required nonnegative.
///
/// After applying [`DIELECTRIC_DSF_RELATIVE_NOISE_FLOOR`] slack, at least 98% of
/// frequency samples should pass — leaving margin for isolated quadrature outliers.
pub const DIELECTRIC_DSF_POSITIVE_FRACTION_MIN: f64 = 0.98;

/// Completed Mermin DSF: stricter minimum nonnegative fraction than [`DIELECTRIC_DSF_POSITIVE_FRACTION_MIN`].
///
/// The completed construction is smoother; 99% is the acceptance bar for
/// `dynamic_structure_factor_completed` in Paper 44 validation.
pub const DIELECTRIC_COMPLETED_DSF_POSITIVE_FRACTION_MIN: f64 = 0.99;

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

/// NMP pass/fail threshold: maximum pull (in σ units) for PASS status.
///
/// When |value - target| / σ &lt; this threshold, the NMP observable passes.
/// 2σ corresponds to ~95% confidence for normally distributed errors.
/// Used in `print_nmp_analysis` and related provenance checks.
pub const NMP_SIGMA_THRESHOLD: f64 = 2.0;

/// Theoretical uncertainty floor for binding energy comparison.
///
/// `σ_theo` = max(1% × `B_exp`, 2.0 `MeV`). The 1% accounts for missing
/// physics (deformation, pairing fluctuations); the 2 `MeV` floor prevents
/// division by zero for light nuclei.
pub const SIGMA_THEO_FRACTION: f64 = 0.01;
/// Minimum `σ_theo` in `MeV` to avoid division by zero for light nuclei.
pub const SIGMA_THEO_FLOOR_MEV: f64 = 2.0;

/// Compute theoretical uncertainty for a given experimental binding energy.
#[must_use]
pub fn sigma_theo(b_exp: f64) -> f64 {
    (SIGMA_THEO_FRACTION * b_exp).max(SIGMA_THEO_FLOOR_MEV)
}

/// HFB Rust-vs-experiment: relative binding energy tolerance.
///
/// The L2 HFB solver with `SLy4` parametrization reproduces experimental
/// binding energies to ~5-10% for medium-mass nuclei. Missing physics
/// (deformation, continuum, 3-body) limits accuracy. Light nuclei like
/// Ni-56 show up to ~11% deviation from experiment (AME2020).
///
/// Measurement: `validate_nuclear_eos` Phase 4, Feb 2026.
///   `SLy4` on 6 HFB test nuclei. Worst: Ni-56 → 11.2% vs AME2020.
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
/// The spin-orbit potential `V_so` ∝ (1/r)(dρ/dr) diverges at r=0.
/// The physical nuclear surface starts at r ≈ 1.2·A^(1/3) fm (≈3 fm
/// for A=16), so r < 0.1 fm is deep inside the nuclear core where
/// the spin-orbit integrand should vanish anyway. The 0.1 fm floor
/// (~0.5 × proton charge radius) prevents numerical overflow without
/// affecting physics in the surface region where spin-orbit matters.
pub const SPIN_ORBIT_R_MIN: f64 = 0.1;

/// Coulomb singularity guard for charge-enclosed / r (fm).
///
/// The Coulomb direct potential `V_C(r)` = e² × Q(r)/r diverges at r=0.
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

/// BCS pairing gap threshold: below this Δ (`MeV`), pairing is negligible.
///
/// At 1e-10 `MeV` (~0.1 neV), the pairing gap is ~10 orders of magnitude
/// below typical nuclear pairing gaps (0.5–2.0 `MeV`). Used to switch
/// between BCS and Fermi-gas occupation formulas, and to skip pairing
/// energy terms. Also used as bisection convergence tolerance for
/// the Fermi energy (MeV-scale precision).
pub const PAIRING_GAP_THRESHOLD: f64 = 1e-10;

/// SCF energy convergence tolerance (`MeV`).
///
/// The self-consistent field loop converges when |`E_n` − E_{n-1}| < tol.
/// 1e-6 `MeV` corresponds to ~1 eV, well below nuclear binding energy
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
/// meaning the eigenvalue is ~25 `MeV` above the Fermi surface — far into
/// the continuum where occupations should indeed be zero.
pub const BCS_DENSITY_SKIP: f64 = 1e-12;

/// BCS pairing-to-sharp-filling transition threshold (`MeV`).
///
/// When the pairing gap delta < 0.01 `MeV`, pairing correlations are
/// negligible and the system is in the normal (unpaired) phase. BCS
/// occupation factors degenerate to a step function, so we switch to
/// sharp Fermi filling for numerical stability. This threshold is
/// ~50x smaller than the smallest physical pairing gap in medium/heavy
/// nuclei (~0.5 `MeV`), ensuring the switch only activates for truly
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
/// Near-magic nuclei are weakly deformed. `beta_2` ≈ 0.05 corresponds
/// to a prolate deformation ~1% change in radius. Used as the starting
/// point for self-consistent deformed HFB iteration.
pub const DEFORMATION_GUESS_WEAK: f64 = 0.05;

/// Initial deformation guess for generic (non-magic) nuclei.
///
/// Generic mid-shell nuclei typically have `beta_2` ≈ 0.15–0.25.
/// This starting value allows the SCF iteration to converge from
/// either side of the true deformation minimum.
pub const DEFORMATION_GUESS_GENERIC: f64 = 0.15;

/// Initial deformation guess for sd-shell deformed nuclei.
///
/// Nuclei in the sd-shell (A ≈ 20–40) can be strongly deformed with
/// `beta_2` ≈ 0.3–0.4. This higher starting value helps the SCF loop
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
// HFB solver configuration
//
// These control the self-consistent field (SCF) iteration loop for
// nuclear HFB solvers. Changing these affects convergence speed and
// reliability for all L2 and L3 binding energy calculations.
// ═══════════════════════════════════════════════════════════════════

/// Maximum SCF iterations for HFB solvers.
///
/// 200 iterations is sufficient for all nuclei in the AME2020 chart
/// (tested Z=8..120, N=8..180). Convergence typically occurs in 40–80
/// iterations for spherical, 80–150 for deformed nuclei. The limit
/// prevents infinite loops on pathological parameter sets.
pub const HFB_MAX_ITER: usize = 200;

/// Broyden mixing warmup iterations.
///
/// During the first `BROYDEN_WARMUP` iterations, simple linear mixing
/// is used (alpha=0.5). After warmup, Broyden's method uses the
/// accumulated history for quasi-Newton acceleration. 50 iterations
/// allows the density to stabilize before switching to the more
/// aggressive Broyden update.
pub const BROYDEN_WARMUP: usize = 50;

/// Maximum Broyden history vectors retained.
///
/// The modified Broyden method stores `BROYDEN_HISTORY` pairs of
/// (`delta_f`, `delta_u`) vectors. 8 vectors balance memory use against
/// convergence acceleration. Older vectors are discarded FIFO.
pub const BROYDEN_HISTORY: usize = 8;

/// Default L2 (spherical HFB) mixing fraction.
///
/// Linear mixing alpha for density updates: `rho_new` = (1-α)*`rho_old` + α*`rho_calc`.
/// 0.3 is conservative; higher values converge faster but may oscillate.
pub const HFB_L2_MIXING: f64 = 0.3;

/// Default L2 (spherical HFB) convergence tolerance (`MeV`).
///
/// SCF is converged when |`E_n` - E_{n-1}| < tol. 0.05 `MeV` is ~0.005%
/// relative accuracy for A~100 nuclei (BE ≈ 800 `MeV`).
pub const HFB_L2_TOLERANCE: f64 = 0.05;

/// Fermi level bisection search range (`MeV`).
///
/// The BCS Fermi level search spans [`E_min` - margin, `E_max` + margin]
/// where margin is this value. 50 `MeV` ensures the Fermi level is
/// always bracketed even for loosely-bound systems.
pub const FERMI_SEARCH_MARGIN: f64 = 50.0;
