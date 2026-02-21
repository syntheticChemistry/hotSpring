// SPDX-License-Identifier: AGPL-3.0-only

//! Molecular dynamics tolerances: forces, observables, transport coefficients,
//! and CPU/GPU parity validation.

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

/// Maximum acceptable energy drift for MD simulations.
///
/// Measured drift is 0.000–0.002% across 9 Sarkas PP cases (N=10k, 80k steps).
/// Tightened from 5% to 0.5% — still 250× above measured worst case, providing
/// ample margin for different system sizes and integration parameters.
///
/// Source: METHODOLOGY.md §Phase C acceptance criteria.
pub const ENERGY_DRIFT_PCT: f64 = 0.5;

/// RDF tail convergence: |g(r→∞) − 1|.
///
/// For a well-equilibrated fluid, g(r) → 1 at large r. Measured tail
/// deviation across 9 Sarkas PP cases: ≤ 0.0017 (METHODOLOGY.md).
/// Tightened from 0.15 to 0.02 — ~12× above measured worst case.
///
/// Source: METHODOLOGY.md §Phase C acceptance criteria.
pub const RDF_TAIL_TOLERANCE: f64 = 0.02;

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

/// C_w(κ) calibration: relative error vs Python analytical fit.
///
/// The weak-coupling coefficient C_w(κ) = exp(a₀ + a₁κ + a₂κ²) from
/// calibrate_daligault_fit.py must match its own analytical values to 2%.
/// The fit was calibrated at κ ∈ {0, 1, 2, 3} with 12 Sarkas data points.
///
/// Source: calibrate_daligault_fit.py, commit 0a6405f, Feb 2026.
pub const TRANSPORT_C_W_CALIBRATION_REL: f64 = 0.02;

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
///
/// Measurement: validate_cpu_gpu_parity on RTX 4070, Feb 19 2026.
///   N=108, seed=42, 5k equil + 5k prod. Worst case: κ=1 Γ=14 → 62%.
///   N=500, seed=42, 5k equil + 20k prod. Worst case: κ=2 Γ=31 → 58%.
///   Threshold set to 65% = max observed + 3% margin.
pub const TRANSPORT_D_STAR_CPU_GPU_PARITY: f64 = 0.65;

/// Temperature stability during NVE production (relative).
///
/// After Berendsen equilibration, NVE production should maintain T*
/// within ~30% of the target. Larger deviations indicate insufficient
/// equilibration or integration instability.
///
/// Measurement: validate_transport lite runs, Feb 19 2026.
///   N=500, 5k equil + 20k prod, seed=42. Max |T_final/T_target - 1|
///   across 9 Sarkas-matched cases: 24% (κ=3 Γ=1510). 30% = max + 6% margin.
pub const TRANSPORT_T_STABILITY: f64 = 0.30;

/// D* MSD vs Daligault fit: relative tolerance for lite validation.
///
/// At N=500, finite-size effects dominate; MSD-derived D* may differ
/// from the calibrated fit by up to 80%. This is a sanity check,
/// not a precision benchmark — the fit was calibrated against N=2000
/// Sarkas data, so N=500 lite runs have systematic finite-size bias.
///
/// Measurement: validate_transport --lite, Feb 19 2026.
///   N=500, seed=42. Worst case: κ=3 Γ=1510 → 74% (strong coupling,
///   longest correlation time, most sensitive to finite-size). 80% = max + 6%.
pub const TRANSPORT_D_STAR_VS_FIT_LITE: f64 = 0.80;

/// MSD-derived D* vs VACF-derived D*: internal consistency.
///
/// Both methods extract D* from the same trajectory; disagreement
/// beyond 50% indicates a computational bug, not physics noise.
/// MSD uses long-time slope of ⟨|r(t)-r(0)|²⟩; VACF uses Green-Kubo
/// integral of ⟨v(t)·v(0)⟩. At small N, limited trajectory length
/// causes both estimators to have large variance.
///
/// Measurement: validate_transport, Feb 19 2026.
///   N=500, seed=42. Worst MSD/VACF disagreement: 43% (κ=3 Γ=1510).
///   50% = max + 7% margin, serving as a bug-detection gate.
pub const TRANSPORT_MSD_VACF_AGREEMENT: f64 = 0.50;

/// D* CPU vs GPU parity at small N (108 particles).
///
/// At N=108 with 5k steps, VACF noise and FP ordering divergence
/// limit CPU/GPU agreement to ~20%.
///
/// Measurement: validate_cpu_gpu_parity, RTX 4070, Feb 19 2026.
///   N=108, seed=42, κ=1 Γ=10, 5k equil + 5k prod. D* relative diff: 17%.
///   20% = max + 3% margin.
pub const PARITY_D_STAR_REL: f64 = 0.20;

/// Temperature CPU vs GPU: relative difference.
///
/// Final equilibrium T* should agree within 25% between CPU and GPU.
/// Differences arise from FP ordering in the energy accumulation.
///
/// Measurement: validate_cpu_gpu_parity, RTX 4070, Feb 19 2026.
///   N=108, seed=42, 6 (κ,Γ) cases. Worst |T_cpu/T_gpu - 1|: 21%.
///   25% = max + 4% margin.
pub const PARITY_T_DIFF: f64 = 0.25;

/// Mean total energy CPU vs GPU: relative difference.
///
/// Energy drift accumulates differently in CPU (Newton-3rd j>i) and
/// GPU (j!=i full-loop). 8% captures the maximum observed divergence
/// for N=108 5k-step runs.
///
/// Measurement: validate_cpu_gpu_parity, RTX 4070, Feb 19 2026.
///   N=108, seed=42, 6 cases. Worst |E_cpu/E_gpu - 1|: 6.8%.
///   8% = max + 1.2% margin.
pub const PARITY_ENERGY_DIFF: f64 = 0.08;

/// PPPM net force tolerance for multi-particle systems.
///
/// For NaCl crystals and random charge configurations, the net force
/// |Σ F_i| should be near zero. PPPM mesh approximation for small
/// systems with limited resolution can give residuals of O(1).
pub const PPPM_MULTI_PARTICLE_NET_FORCE: f64 = 1.0;

// ═══════════════════════════════════════════════════════════════════
// MD solver configuration
//
// These control the cell-list GPU simulation loop behavior.
// Changing these affects performance and thermostat coupling.
// ═══════════════════════════════════════════════════════════════════

/// Cell-list rebuild interval (MD steps).
///
/// GPU cell-list is rebuilt every this many steps during both
/// equilibration and production. 20 steps balances rebuild cost
/// (~0.3ms per rebuild) against force accuracy. Particles moving
/// further than cell_size/2 between rebuilds can miss neighbors.
pub const CELLLIST_REBUILD_INTERVAL: usize = 20;

/// Thermostat application interval (MD steps).
///
/// Berendsen thermostat is applied every this many steps during
/// equilibration. 10 steps provides smooth temperature coupling
/// without overdamping velocity correlations.
pub const THERMOSTAT_INTERVAL: usize = 10;
