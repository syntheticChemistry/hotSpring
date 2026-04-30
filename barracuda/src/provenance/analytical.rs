// SPDX-License-Identifier: AGPL-3.0-or-later

/// Reference source for special function validation.
///
/// Expected values are exact mathematical identities (e.g. Γ(n) = (n-1)!,
/// J₀(0) = 1) or published high-precision tables:
///   - Abramowitz & Stegun, "Handbook of Mathematical Functions" (1964)
///   - NIST Digital Library of Mathematical Functions, <https://dlmf.nist.gov> (2023)
///   - `SciPy` 1.11 `scipy.special` used for cross-validation (not as primary source)
pub const SPECIAL_FUNCTION_REFS: &str =
    "Abramowitz & Stegun (1964), NIST DLMF (2023), scipy.special 1.11";

/// Reference source for linear algebra validation.
///
/// Expected values are mathematical identities: ‖A − LU‖ < ε, `QᵀQ` = I,
/// Ax = `λ`x for eigendecomposition. Cross-validated against `NumPy` 1.26 / `SciPy` 1.11.
pub const LINALG_REFS: &str = "NumPy 1.26 / SciPy 1.11 linear algebra";

/// Reference source for optimizer validation.
///
/// Expected values are analytical optima of standard test functions:
/// Rosenbrock minimum at (1,1), sphere minimum at origin, etc.
/// ODE integrators validated via known solutions (exponential decay, harmonic).
/// Cross-validated against `SciPy` 1.11 `optimize` and `integrate` modules.
pub const OPTIMIZER_REFS: &str = "scipy.optimize 1.11, scipy.integrate 1.11";

/// Reference source for MD validation.
///
/// Expected values are analytical force laws: Lennard-Jones 12-6,
/// Coulomb 1/r², Morse exponential. Velocity-Verlet integration checked
/// via energy conservation and symplectic invariants. No Python baselines:
/// the physics is exact analytical math.
pub const MD_FORCE_REFS: &str = "Analytical LJ/Coulomb/Morse force laws, Allen & Tildesley (1987)";

/// Reference source for GPU kernel validation (Yukawa, PPPM).
///
/// GPU f32 kernels compared against CPU f64 reference with relative
/// tolerance accounting for f32 precision (~7 significant digits).
/// PPPM validated against direct Coulomb summation (exact but O(N²)).
pub const GPU_KERNEL_REFS: &str = "CPU f64 direct summation reference, f32 precision tolerance";
