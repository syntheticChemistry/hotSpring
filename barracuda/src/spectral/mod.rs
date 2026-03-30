// SPDX-License-Identifier: AGPL-3.0-only

//! Spectral theory for discrete Schrödinger operators.
//!
//! **Upstream lean**: This module re-exports from `barracuda::spectral`, which
//! absorbed hotSpring's spectral implementation (v0.6.0, Sessions 25-31h).
//!
//! The full API — Anderson localization, Almost-Mathieu operator, Lanczos
//! eigensolve, Sturm tridiagonal, level statistics, and CSR `SpMV` — now lives
//! in the shared barracuda crate. hotSpring re-exports for backward
//! compatibility of validation binaries and integration tests.
//!
//! # Provenance
//!
//! - Anderson (1958) "Absence of diffusion in certain random lattices"
//! - Aubry & André (1980) "Analyticity breaking and Anderson localization"
//! - Jitomirskaya (1999) "Metal-insulator transition for the almost Mathieu operator"
//! - Herman (1983) "Une méthode pour minorer les exposants de Lyapunov"
//! - Avila (2015) "Global theory of one-frequency Schrödinger operators" (Fields Medal)
//! - Kappus & Wegner (1981) "Anomaly in the band centre of the 1D Anderson model"

pub use barracuda::spectral::{
    BatchIprGpu, GOE_R, GOLDEN_RATIO, LanczosTridiag, POISSON_R, SpectralAnalysis,
    SpectralCsrMatrix, SpectralPhase, WGSL_SPMV_CSR_F64, almost_mathieu_hamiltonian, anderson_2d,
    anderson_3d, anderson_hamiltonian, anderson_potential, classify_spectral_phase,
    clean_2d_lattice, clean_3d_lattice, detect_bands, find_all_eigenvalues, gcd,
    hofstadter_butterfly, lanczos, lanczos_eigenvalues, level_spacing_ratio, spectral_bandwidth,
    spectral_condition_number, sturm_count,
};
pub use barracuda::spectral::{lyapunov_averaged, lyapunov_exponent};

/// Backward-compatible type alias for the upstream `SpectralCsrMatrix`.
///
/// hotSpring validation binaries use `CsrMatrix`; upstream renamed to
/// `SpectralCsrMatrix` to avoid collision with `linalg::sparse::CsrMatrix`.
pub type CsrMatrix = SpectralCsrMatrix;
