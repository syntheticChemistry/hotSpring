# hotSpring → ToadStool: Rewire v4 — Spectral Module Lean

**Date:** 2026-02-22
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Context:** ToadStool Sessions 25-31h (commits `dc540afd`..`77f70b2e`) reviewed
against hotSpring v0.6.3 (33/33 validation suites, 648 unit tests)

---

## Executive Summary

ToadStool's Sessions 25-31h sprint (10 commits) absorbed hotSpring's **entire
spectral module** — Anderson localization (1D/2D/3D), Lanczos eigensolve, CSR
SpMV, Hofstadter butterfly, Sturm tridiagonal, and level statistics — plus a
new `BatchIprGpu` for inverse participation ratio. The absorption was
byte-for-byte faithful to hotSpring's implementation (same doc comments, same
provenance citations), with the CSR struct renamed from `CsrMatrix` to
`SpectralCsrMatrix` to avoid collision with `linalg::sparse::CsrMatrix`.

hotSpring has completed the lean phase: all local spectral source files deleted
(~41 KB across 7 files), replaced with re-exports from `barracuda::spectral`
and a `CsrMatrix` type alias for backward compatibility. Zero code broken.

---

## Part 1: What ToadStool Absorbed (Sessions 25-31h)

### Spectral Module (Full Absorption)

| Component | Upstream Location | Commit Range | Tests |
|-----------|-------------------|-------------|-------|
| `CsrMatrix` → `SpectralCsrMatrix` | `barracuda::spectral::sparse` | S25-31h | 2 |
| `WGSL_SPMV_CSR_F64` shader | `barracuda::spectral::sparse` (inline) | S25-31h | — |
| Anderson 1D/2D/3D | `barracuda::spectral::anderson` | S25-31h | 4 |
| Lanczos + `LanczosTridiag` | `barracuda::spectral::lanczos` | S25-31h | 1 |
| Hofstadter + `GOLDEN_RATIO` | `barracuda::spectral::hofstadter` | S25-31h | 4 |
| Sturm tridiagonal | `barracuda::spectral::tridiag` | S25-31h | 2 |
| Level statistics (GOE_R, POISSON_R) | `barracuda::spectral::stats` | S25-31h | 3 |
| **BatchIprGpu** (NEW) | `barracuda::spectral::batch_ipr` | S25-31h | — |

### Other Evolution (Sessions 25-31h)

| Topic | Impact on hotSpring |
|-------|-------------------|
| 570+ typed shader constants, zero orphans | No direct impact — hotSpring uses upstream ops |
| Executor wiring, MathOp integration | Future benefit: GPU ops may get new dispatch paths |
| Dead code cleanup, PollConfig refactor | Cleaner upstream API |
| RF batch inference shaders | New capability for ESN/ML workloads |
| f64 linalg extensions | `GemmCachedF64`, `LinSolveF64` available |

---

## Part 2: hotSpring Actions Taken

### Deleted Local Spectral Sources (41 KB)

| File | Size | Replaced By |
|------|------|-------------|
| `spectral/anderson.rs` | 15,142 B | `barracuda::spectral::anderson_*` |
| `spectral/csr.rs` | 3,559 B | `barracuda::spectral::SpectralCsrMatrix` |
| `spectral/hofstadter.rs` | 5,325 B | `barracuda::spectral::hofstadter_*` |
| `spectral/lanczos.rs` | 5,500 B | `barracuda::spectral::lanczos*` |
| `spectral/stats.rs` | 6,367 B | `barracuda::spectral::stats` |
| `spectral/tridiag.rs` | 4,987 B | `barracuda::spectral::tridiag` |
| `spectral/shaders/spmv_csr_f64.wgsl` | (dir) | `barracuda::spectral::WGSL_SPMV_CSR_F64` |

### New `spectral/mod.rs` (Re-export Only)

```rust
pub use barracuda::spectral::{
    almost_mathieu_hamiltonian, anderson_2d, anderson_3d, anderson_hamiltonian,
    anderson_potential, clean_2d_lattice, clean_3d_lattice, detect_bands,
    find_all_eigenvalues, gcd, hofstadter_butterfly, lanczos, lanczos_eigenvalues,
    level_spacing_ratio, sturm_count, BatchIprGpu, LanczosTridiag,
    SpectralCsrMatrix, GOE_R, GOLDEN_RATIO, POISSON_R, WGSL_SPMV_CSR_F64,
};
pub use barracuda::spectral::{lyapunov_averaged, lyapunov_exponent};

/// Backward-compatible type alias.
pub type CsrMatrix = SpectralCsrMatrix;
```

### Documentation Updated

| File | Change |
|------|--------|
| `ABSORPTION_MANIFEST.md` | Spectral module moved to "Already Absorbed"; version → v0.6.4 |
| `DEPRECATION_MIGRATION.md` | Spectral files tracked as "Deleted — re-exported from upstream" |
| `EVOLUTION_READINESS.md` | Spectral section → "Fully Leaning on Upstream" |
| `CHANGELOG.md` | v0.6.4 entry with full change list |
| `README.md` | Test counts updated (637, spectral tests upstream), Rewire v4 row added |
| `Cargo.toml` | Version 0.6.3 → 0.6.4 |

---

## Part 3: Current hotSpring State (v0.6.4)

| Metric | Value |
|--------|-------|
| Unit tests | **637** pass (+ 6 GPU-ignored; 44 spectral tests now upstream) |
| Integration tests | **24** (3 suites) |
| Validation suites | **33/33** pass (CPU) |
| Clippy warnings | **0** (pedantic) |
| Doc warnings | **0** |
| Local spectral code | **0** — fully leaning on `barracuda::spectral` |
| Shader files | **41** `.wgsl` files |
| ToadStool dependency | `path = "../../phase1/toadstool/crates/barracuda"` |

---

## Part 4: Remaining Absorption Candidates

### Tier 1 — Ready Now

| Module | Location | Tests | Notes |
|--------|----------|-------|-------|
| Staggered Dirac | `lattice/dirac.rs` | 8/8 | GPU shader validated; WGSL ready |
| CG Solver | `lattice/cg.rs` | 9/9 | 3 GPU shaders validated |
| ESN Reservoir | `md/reservoir.rs` | 16+ | 2 WGSL shaders + NPU path |

### Tier 2 — Medium Priority

| Module | Location | Tests | Notes |
|--------|----------|-------|-------|
| Screened Coulomb | `physics/screened_coulomb.rs` | 23/23 | CPU Sturm eigensolve |
| Wilson action | `lattice/wilson.rs` | 12/12 | GPU shader already absorbed |
| HMC integrator | `lattice/hmc.rs` | Tests | Cayley SU(3), leapfrog |
| Abelian Higgs | `lattice/abelian_higgs.rs` | 17/17 | GPU shader already absorbed |

### P1 — Local GpuCellList Migration

The deprecated local `GpuCellList` in `md/celllist.rs` is still in use by
`run_simulation_celllist` and `sarkas_gpu`. Migration to upstream
`barracuda::ops::md::CellListGpu` requires mapping the indirect force shader
API. This is the next concrete rewire task.

---

## Part 5: New Upstream Primitives Available

ToadStool Sessions 25-31h added primitives hotSpring can use:

| Primitive | Location | Use Case |
|-----------|----------|----------|
| `BatchIprGpu` | `barracuda::spectral::batch_ipr` | Anderson localization — GPU IPR |
| `GemmCachedF64` | `barracuda::ops::linalg` | Cached GEMM for repeated H-builds |
| `GenEighGpu` | `barracuda::ops::linalg` | Generalized eigenvalue (deformed HFB) |
| `GridQuadratureGemm` | `barracuda::ops::linalg` | Hamiltonian construction |
| `NelderMeadGpu` | `barracuda::optimize` | GPU Nelder-Mead for parameter optimization |
| `ResumableNelderMead` | `barracuda::optimize` | Resumable optimization |
| `Fft1DF64` / `Fft3DF64` | `barracuda::ops::fft` | GPU FFT for PPPM and lattice QCD |
| RF batch inference | `shaders/ml/rf_batch_inference.wgsl` | Random forest inference for ESN |

---

*This document supersedes Rewire v3 for toadstool session tracking.
Rewire v3 remains authoritative for lattice QCD and CellListGpu history.*

*License: AGPL-3.0-only. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required.*
