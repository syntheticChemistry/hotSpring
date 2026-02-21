// SPDX-License-Identifier: AGPL-3.0-only

//! Spectral theory for discrete Schrödinger operators.
//!
//! Implements lattice Hamiltonians and spectral analysis tools for the
//! Kachkovskiy extension (spectral theory / transport):
//!
//! - **CsrMatrix + SpMV**: sparse matrix-vector product (P1 GPU primitive)
//! - **Lanczos eigensolve**: Krylov tridiagonalization with full reorthogonalization
//! - **Anderson model**: random potential in 1D, 2D, and 3D
//!   - 1D/2D: all states localized (Abrahams et al. 1979)
//!   - 3D: genuine metal-insulator transition with mobility edge (W_c ≈ 16.5)
//! - **Almost-Mathieu operator**: quasiperiodic potential, Aubry-André transition
//! - **Transfer matrix**: Lyapunov exponent computation
//! - **Tridiagonal eigensolve**: Sturm bisection for all eigenvalues
//! - **Level statistics**: spacing ratio for localization diagnostics
//!
//! # Physics
//!
//! The 1D discrete Schrödinger equation on ℤ:
//!   ψ_{n+1} + ψ_{n-1} + V_n ψ_n = E ψ_n
//!
//! is equivalent to the eigenvalue problem for the tridiagonal matrix
//! H with diagonal V_i and off-diagonal −1. The spectral properties of H
//! (eigenvalues, eigenvectors, Lyapunov exponent) determine transport:
//! extended states → metallic, localized states → insulating.
//!
//! # Provenance
//!
//! - Anderson (1958) "Absence of diffusion in certain random lattices"
//! - Aubry & André (1980) "Analyticity breaking and Anderson localization"
//! - Jitomirskaya (1999) "Metal-insulator transition for the almost Mathieu operator"
//! - Herman (1983) "Une méthode pour minorer les exposants de Lyapunov"
//! - Avila (2015) "Global theory of one-frequency Schrödinger operators" (Fields Medal)
//! - Kappus & Wegner (1981) "Anomaly in the band centre of the 1D Anderson model"

mod anderson;
mod csr;
mod hofstadter;
mod lanczos;
mod stats;
mod tridiag;

pub use anderson::{
    anderson_2d, anderson_3d, anderson_hamiltonian, anderson_potential, clean_2d_lattice,
    clean_3d_lattice, lyapunov_averaged, lyapunov_exponent,
};
pub use csr::{CsrMatrix, WGSL_SPMV_CSR_F64};
pub use hofstadter::{almost_mathieu_hamiltonian, gcd, hofstadter_butterfly, GOLDEN_RATIO};
pub use lanczos::{lanczos, lanczos_eigenvalues, LanczosTridiag};
pub use stats::{detect_bands, level_spacing_ratio, GOE_R, POISSON_R};
pub use tridiag::{find_all_eigenvalues, sturm_count};
