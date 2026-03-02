# Evolution Readiness: Rust → WGSL Shader Promotion → ToadStool Absorption

This document maps each Rust module to its GPU shader readiness tier,
tracks what toadstool has absorbed, and identifies next absorption targets.

## Evolution Path

```
Python baseline → Rust validation → WGSL template → GPU shader → ToadStool absorption → Lean on upstream
```

## Tier Definitions

| Tier | Label | Meaning |
|------|-------|---------|
| **A** | Rewire | Shader exists and is validated; wire into pipeline |
| **B** | Adapt | Shader exists but needs modification (API, precision, layout) |
| **C** | New | No shader exists; must be written from scratch |
| **✅** | Absorbed | ToadStool has absorbed this as a first-class barracuda primitive |

## ToadStool Absorption Status (Mar 2, 2026 — v0.6.15 synced to toadStool S78)

| hotSpring Module | ToadStool Primitive | Commit | Status |
|-----------------|--------------------| -------|--------|
| `lattice/complex_f64.rs` WGSL template | `shaders/math/complex_f64.wgsl` | `8fb5d5a0` | ✅ Absorbed |
| `lattice/su3.rs` WGSL template | `shaders/math/su3.wgsl` | `8fb5d5a0` | ✅ Absorbed |
| Wilson plaquette design | `shaders/lattice/wilson_plaquette_f64.wgsl` | `8fb5d5a0` | ✅ Absorbed |
| HMC force design | `shaders/lattice/su3_hmc_force_f64.wgsl` | `8fb5d5a0` | ✅ Absorbed |
| Abelian Higgs design | `shaders/lattice/higgs_u1_hmc_f64.wgsl` | `8fb5d5a0` | ✅ Absorbed |
| Local `GpuCellList` | `CellListGpu` (BGL fixed) | `8fb5d5a0` | ✅ Absorbed (local deprecated) |
| NAK eigensolve workarounds | `batched_eigh_nak_optimized_f64.wgsl` | `82f953c8` | ✅ Absorbed |
| FFT need documented | `Fft1DF64` + `Fft3DF64` | `1ffe8b1a` | ✅ Absorbed |
| `ReduceScalar` feedback | `ReduceScalarPipeline` (`scalar_buffer`, `max_f64`, `min_f64`) | v0.5.16 | ✅ Absorbed |
| Driver profiling feedback | `GpuDriverProfile` + `WgslOptimizer` | v0.5.15 | ✅ Absorbed |

### Next Absorption Targets

| hotSpring Module | What Needs Writing | Priority | Absorption Value |
|-----------------|-------------------|----------|-----------------|
| `spectral/csr.rs::CsrMatrix::spmv()` | ~~GPU CSR SpMV WGSL shader~~ | ~~**P1**~~ | ✅ **Done** — `WGSL_SPMV_CSR_F64` validated 8/8 checks (machine-epsilon parity on RTX 4070) |
| `spectral/lanczos.rs::lanczos()` | ~~GPU Lanczos eigensolve~~ | ~~**P1**~~ | ✅ **Done** — GPU SpMV inner loop + CPU control, 6/6 checks (eigenvalues match to 1e-15) |
| `lattice/dirac.rs` | ~~GPU staggered Dirac SpMV~~ | ~~**P1**~~ | ✅ **Done** — `WGSL_DIRAC_STAGGERED_F64` validated 8/8 checks (max error 4.44e-16, cold+hot+asymmetric lattices) |
| `md/celllist.rs` → upstream | ~~Migrate `run_simulation_celllist` to upstream API~~ | ~~**P1**~~ | ✅ **Done (v0.6.2)** — `CellListGpu` migrated, 282 lines + 3 shaders deleted |
| `lattice/cg.rs` | ~~GPU CG solver~~ | ~~**P2**~~ | ✅ **Done** — GPU CG (D†D) validated 9/9 checks (machine-epsilon parity, identical iteration counts) |
| `physics/hfb_gpu_resident.rs` energy | ~~Wire `batched_hfb_energy_f64.wgsl`~~ | ~~**P2**~~ | ✅ **Done (v0.6.2)** — GPU energy dispatch wired behind `gpu_energy` feature flag |
| `lattice/gpu_hmc.rs` streaming | ~~GPU streaming HMC dispatch~~ | ~~**P1**~~ | ✅ **Done (v0.6.8)** — single-encoder batched dispatch, 67× CPU at 16⁴, 9/9 checks |
| `lattice/gpu_hmc.rs` resident CG | ~~GPU-resident CG scalars~~ | ~~**P1**~~ | ✅ **Done (v0.6.8)** — α/β/rz on GPU, 10-iter batches, 15,360× readback reduction, 30.7× speedup |
| `lattice/gpu_hmc.rs` bidirectional | ~~Async readback + NPU branch~~ | ~~**P1**~~ | ✅ **Done (v0.6.8)** — 90% to GPU, 10% readback, std::sync::mpsc NPU observation routing |
| `md/reservoir/` CPU solver | ~~GPU-free ESN training~~ | ~~**P2**~~ | ✅ **Done (v0.6.8)** — local gauss_jordan_solve() for small ESN matrices (50-200 dim) |

## Physics Modules

| Rust Module | WGSL Shader(s) | Tier | Status | Blocker |
|-------------|----------------|------|--------|---------|
| `physics/semf.rs` | `SHADER_SEMF_BATCH`, `SHADER_CHI2` (inline in `nuclear_eos_gpu.rs`) | **A** | GPU pipeline exists | None — production-ready |
| `physics/hfb.rs` | `batched_hfb_*.wgsl` (4 shaders) via `hfb_gpu.rs` | **A** | GPU pipeline exists | None — validated against CPU |
| `physics/hfb_gpu.rs` | Uses `BatchedEighGpu::execute_single_dispatch` | **A** | Production GPU — single-dispatch (v0.5.3) | None — all rotations in one shader |
| `physics/bcs_gpu.rs` | `bcs_bisection_f64.wgsl` | **A** | Production GPU — pipeline cached (v0.5.3) | None — ToadStool `target` bug absorbed (`0c477306`) |
| `physics/hfb_gpu_resident/` | `batched_hfb_potentials_f64.wgsl`, `batched_hfb_hamiltonian_f64.wgsl`, `batched_hfb_density_f64.wgsl`, `batched_hfb_energy_f64.wgsl`, `BatchedEighGpu`, `SpinOrbitGpu` | **A** | GPU H-build + eigensolve + spin-orbit + density + mixing + energy (v0.6.2) | BCS Brent on CPU (root-finding not GPU-efficient) |
| `physics/hfb_deformed/` | — | **C** | CPU only (refactored: mod, potentials, basis, tests) | Deformed HFB needs new shaders for 2D grid Hamiltonian build |
| `physics/hfb_deformed_gpu/` | `deformed_*.wgsl` (5 shaders exist, not all wired) | **B** | Partial GPU (refactored: mod, types, physics, gpu_diag, tests) | H-build on CPU; deformed Hamiltonian shaders exist but unwired |
| `physics/nuclear_matter.rs` | — | **C** | CPU only | Uses `barracuda::optimize::bisect` (CPU); no NMP shader. Low priority — fast on CPU |
| `physics/hfb_common.rs` | — | N/A | Shared utilities | Pure CPU helpers (WS radii, deformation estimation) |
| `physics/constants.rs` | — | N/A | Physical constants | Data only |

## MD Modules

| Rust Module | WGSL Shader(s) | Tier | Status | Blocker |
|-------------|----------------|------|--------|---------|
| `md/simulation.rs` | Yukawa (all-pairs + cell-list), VV integrator, Berendsen thermostat, KE per-particle, `ReduceScalarPipeline` (inline in `md/shaders.rs`) | **A** | Full GPU pipeline | None — production-ready |
| `md/celllist.rs` | GPU cell-list via upstream `CellListGpu` + indirect force shader. Zero CPU readback | **✅** | **Migrated** (v0.6.2) — local `GpuCellList` deleted, upstream `barracuda::ops::md::CellListGpu` | None |
| `md/shaders.rs` | 11 WGSL shaders (all `.wgsl` files, zero inline). GPU cell-list shaders added v0.5.13 | **A** | Production | v0.6.3: all inline extracted to `.wgsl` |
| `md/observables/` | Uses `SsfGpu` from BarraCuda | **A** | SSF on GPU; RDF/VACF CPU post-process | VACF now correct (particle identity preserved by indirect indexing) |
| `md/cpu_reference.rs` | — | N/A | Validation reference | Intentionally CPU-only for baseline comparison |
| `md/config.rs` | — | N/A | Configuration | Data structures only |

## WGSL Shader Inventory

### Physics Shaders (`src/physics/shaders/`, 10 files, ~1950 lines)

| Shader | Lines | Pipeline Stage |
|--------|-------|----------------|
| `batched_hfb_density_f64.wgsl` | 150 | Density + BCS + mixing for spherical HFB (batched wf) |
| `batched_hfb_potentials_f64.wgsl` | 170 | Skyrme potentials (U_total, f_q) |
| `batched_hfb_hamiltonian_f64.wgsl` | 123 | HFB Hamiltonian H = T_eff + V |
| `batched_hfb_energy_f64.wgsl` | 147 | HFB energy functional (shared-memory reduce) |
| `bcs_bisection_f64.wgsl` | 141 | BCS chemical-potential bisection |
| `deformed_wavefunction_f64.wgsl` | 241 | Nilsson HO wavefunctions on 2D (ρ,z) grid |
| `deformed_hamiltonian_f64.wgsl` | 214 | Block Hamiltonian for deformed HFB |
| `deformed_density_energy_f64.wgsl` | 293 | Deformed density, energy, Q20, RMS radius |
| `deformed_gradient_f64.wgsl` | 205 | Gradient of deformed densities |
| `deformed_potentials_f64.wgsl` | 268 | Deformed mean-field potentials |

### MD Reference Shaders (absorbed — directory removed)

Toadstool reference shaders were absorbed upstream and the
`src/md/shaders_toadstool_ref/` directory was deleted in v0.6.3.
Production equivalents live in `src/md/shaders/`.

### MD Production Shaders (`src/md/shaders/`)

| Shader | Physics | Location |
|--------|---------|----------|
| `yukawa_force_f64.wgsl` | Yukawa all-pairs (native f64) | `.wgsl` file |
| `yukawa_force_celllist_f64.wgsl` | Cell-list v1 (27-neighbor, sorted positions) | `.wgsl` file |
| `yukawa_force_celllist_v2_f64.wgsl` | Cell-list v2 (flat loop, sorted positions) | `.wgsl` file |
| `yukawa_force_celllist_indirect_f64.wgsl` | Cell-list indirect (unsorted positions + `sorted_indices`) **(v0.5.13)** | `.wgsl` file |
| `vv_kick_drift_f64.wgsl` | Velocity-Verlet kick+drift | `.wgsl` file |
| `vv_half_kick_f64.wgsl` | VV second half-kick | `.wgsl` file **(v0.6.3)** |
| `berendsen_f64.wgsl` | Berendsen thermostat rescale | `.wgsl` file **(v0.6.3)** |
| `kinetic_energy_f64.wgsl` | Kinetic energy reduction | `.wgsl` file **(v0.6.3)** |
| `rdf_histogram_f64.wgsl` | RDF histogram binning | `.wgsl` file |
| `esn_reservoir_update.wgsl` | ESN reservoir state update (f32) | `.wgsl` file |
| `esn_readout.wgsl` | ESN readout layer (f32) | `.wgsl` file |

**Note**: `cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl` were deleted in v0.6.2 (GPU cell-list build migrated to upstream `CellListGpu`).

## BarraCuda Primitives Used

| BarraCuda Module | hotSpring Usage |
|------------------|-----------------|
| `barracuda::linalg::eigh_f64` | Symmetric eigendecomposition (CPU) |
| `barracuda::ops::linalg::BatchedEighGpu` | Batched GPU eigensolve |
| `barracuda::ops::grid::SpinOrbitGpu` | GPU spin-orbit correction in HFB **(v0.5.6)** |
| `barracuda::ops::grid::compute_ls_factor` | Canonical l·s factor for spin-orbit **(v0.5.6)** |
| `barracuda::numerical::{trapz, gradient_1d}` | Radial integration, gradient |
| `barracuda::optimize::*` | Bisection, Brent, Nelder-Mead, multi-start NM |
| `barracuda::sample::*` | Latin hypercube, Sobol, direct |
| `barracuda::surrogate::*` | RBF surrogate, kernels |
| `barracuda::stats::*` | Chi², bootstrap CI, correlation |
| `barracuda::special::*` | Gamma, Laguerre, Bessel, Hermite, Legendre, erf |
| `barracuda::ops::md::*` | Forces, integrators, thermostats, observables |
| `barracuda::pipeline::ReduceScalarPipeline` | GPU f64 sum-reduction (KE, PE, thermostat) **(v0.5.12)** |
| `GpuCellList` (local, **deprecated**) | GPU-resident 3-pass cell-list build — upstream `CellListGpu` fixed (toadstool `8fb5d5a0`) |
| `barracuda::ops::lattice::*` | Complex f64, SU(3), Wilson plaquette, HMC force, Abelian Higgs GPU shaders (toadstool `8fb5d5a0`) |
| `barracuda::ops::fft::Fft1DF64` | GPU FFT f64 for momentum-space (toadstool `1ffe8b1a`) |
| `barracuda::ops::fft::Fft3DF64` | GPU 3D FFT for lattice QCD / PPPM (toadstool `1ffe8b1a`) |
| `barracuda::device::{WgpuDevice, TensorContext}` | GPU device bridge |

No duplicate math — all mathematical operations use BarraCuda primitives.
`hermite_value` now delegates to `barracuda::special::hermite` (v0.5.7).
`factorial_f64` now delegates to `barracuda::special::factorial` (v0.5.10).
`solve_linear_system` in `reservoir/` uses local `gauss_jordan_solve()` CPU fallback (v0.6.8; was `barracuda::linalg::solve_f64` in v0.6.2).
WGSL `abs_f64` and `cbrt_f64` now injected via `ShaderTemplate::with_math_f64_auto()` (v0.5.8).
Force shaders compiled via `GpuF64::create_pipeline_f64()` → barracuda driver-aware path **(v0.5.11)**.
`GpuCellList` migrated to upstream `barracuda::ops::md::CellListGpu` (v0.6.2) — 3 local shaders deleted.

## Completed (v0.6.3, Feb 22 2026)

- ✅ **Inline WGSL extraction**: 5 more inline shader strings extracted to `.wgsl` files:
  - `md/shaders.rs`: `SHADER_VV_HALF_KICK` → `vv_half_kick_f64.wgsl`, `SHADER_BERENDSEN` → `berendsen_f64.wgsl`, `SHADER_KINETIC_ENERGY` → `kinetic_energy_f64.wgsl`
  - `lattice/complex_f64.rs`: `WGSL_COMPLEX64` → `shaders/complex_f64.wgsl`
  - `lattice/su3.rs`: `WGSL_SU3` → `shaders/su3_f64.wgsl`
- ✅ **Deformed HFB coverage**: 13 new tests covering `diagonalize_blocks` (V=0, constant V, sharp Fermi), `potential_matrix_element` (constant V, Hermitian symmetry), `solve()` SCF loop (smoke test, determinism, physical bounds), `binding_energy_l3`, and Hermite/Laguerre norm integrals
- ✅ **648 tests** (was 638), 0 failures, 6 ignored
- ✅ **Stale documentation cleaned**: Deleted shader references (`cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl`) removed from shader inventory; extracted shaders added

## Completed (v0.6.2, Feb 21 2026)

- ✅ **Zero clippy pedantic+nursery warnings**: was ~1500 in v0.6.1, now 0. Systematic resolution of `mul_add` (150+), `doc_markdown` (600+), `must_use` (186+), `imprecise_flops` (30+), `use_self` (14), `const_fn` (4), `option_if_let_else` (5), `HashMap` hasher (2), `significant_drop_tightening` (1).
- ✅ **Duplicate math eliminated**: `reservoir/` Gaussian elimination → `barracuda::linalg::solve_f64`
- ✅ **GPU energy pipeline wired**: `batched_hfb_energy_f64.wgsl` dispatched in SCF loop behind `gpu_energy` feature flag
- ✅ **Large file refactoring**: `bench.rs` (1005→4 files), `hfb_gpu_resident/mod.rs` (7 helpers extracted), `celllist_diag.rs` (1156→951)
- ✅ **Cast safety documentation**: Crate-level `#![allow]` with mantissa/range analysis; per-function annotations on critical GPU casts
- ✅ **MutexGuard tightening**: `PowerMonitor::finish()` clones samples immediately, drops lock before processing
- ✅ **561 tests** (was 505), 0 failures, 67.4% region / 78.8% function coverage
- ✅ **metalForge/forge**: zero pedantic warnings
- ✅ **Version**: 0.6.0 → 0.6.2

## Completed (v0.6.1, Feb 21 2026)

- ✅ **Zero `expect()`/`unwrap()` in library code**: `#![deny(clippy::expect_used, clippy::unwrap_used)]` enforced crate-wide. All 15 production `expect()` calls replaced with `Result` propagation, `bytemuck` zero-copy, or safe pattern matching.
- ✅ **Tolerances module tree**: `tolerances.rs` (1384 lines) refactored into `tolerances/{mod,core,md,physics,lattice,npu}.rs`. Each submodule under 300 lines. Zero API change via `pub use` re-exports.
- ✅ **Solver config centralized**: 8 new constants (`HFB_MAX_ITER`, `BROYDEN_WARMUP`, `BROYDEN_HISTORY`, `HFB_L2_MIXING`, `HFB_L2_TOLERANCE`, `FERMI_SEARCH_MARGIN`, `CELLLIST_REBUILD_INTERVAL`, `THERMOSTAT_INTERVAL`) extracted from 7 files. Zero hardcoded solver params in library code.
- ✅ **Large file refactoring**: 4 monolithic files decomposed into module directories: `hfb/` (3 files), `hfb_deformed/` (4 files), `hfb_deformed_gpu/` (5 files), `hfb_gpu_resident/` (3 files). All new files under 500 LOC except `hfb_gpu_resident/mod.rs` (1456 — monolithic GPU pipeline).
- ✅ **Integration test suites**: 3 new suites: `integration_physics.rs` (11), `integration_data.rs` (8), `integration_transport.rs` (5) — 24 tests covering cross-module interactions.
- ✅ **Provenance completeness**: `SCREENED_COULOMB_PROVENANCE` and `DALIGAULT_CALIBRATION_PROVENANCE` added. Commit verification documentation included.
- ✅ **Capability-based discovery**: `try_discover_data_root()` returns `Result`; `available_capabilities()` probes runtime validation domains.
- ✅ **Tolerances tightened**: `ENERGY_DRIFT_PCT` 5%→0.5%, `RDF_TAIL_TOLERANCE` 0.15→0.02 (both still 10×+ above measured worst case).
- ✅ **Zero-copy GPU buffer reads**: `bytemuck::try_cast_slice` with alignment fallback replaces manual byte conversion.
- ✅ **Control JSON policy documented** in tolerances module.
- ✅ **505 unit tests + 24 integration + 8 forge (505 passing + 5 GPU-ignored), 0 clippy warnings, 0 doc warnings, 154 centralized constants**

## Promotion Priority

1. ~~**GPU energy integrands + SumReduceF64**~~ ✅ **DONE (v0.6.2)** — `batched_hfb_energy_f64.wgsl` wired into SCF loop behind `gpu_energy` feature flag. `compute_energy_integrands` + `compute_pairing_energy` GPU passes, staging buffer readback. CPU fallback preserved.
2. ~~**BCS on GPU**~~ ✅ **DONE (v0.5.10)** — Density + mixing on GPU; BCS Brent remains on CPU (root-finding not GPU-efficient)
3. ~~**SpinOrbitGpu**~~ ✅ **DONE (v0.5.6)** — Wired with CPU fallback
4. ~~**WGSL preamble injection**~~ ✅ **DONE (v0.5.8)** — `ShaderTemplate::with_math_f64_auto()`
5. **hfb_deformed_gpu.rs** → Wire existing deformed_*.wgsl shaders for full GPU H-build
6. **nuclear_matter.rs** → Low priority; CPU bisection is fast enough

## Completed (v0.5.9)

- ✅ **Final tolerance wiring pass**: 6 new constants (`BCS_DENSITY_SKIP`, `SHARP_FILLING_THRESHOLD`,
  `DEFORMED_COULOMB_R_MIN`, `DEFORMATION_GUESS_WEAK/GENERIC/SD`) — 15 remaining inline values
  in `hfb.rs`, `hfb_deformed.rs`, `hfb_deformed_gpu.rs`, `md/observables/` → named constants
- ✅ **Clippy pedantic**: 0 clippy warnings across all targets
- ✅ **Full audit report**: specs, wateringHole compliance, validation fidelity, dependency health,
  evolution readiness, test coverage, code size, licensing, data provenance

## CPU vs GPU Scaling (v0.5.11, Feb 19 2026)

See `experiments/007_CPU_GPU_SCALING_BENCHMARK.md` for full data.

| N | GPU mode | CPU steps/s | GPU steps/s | Speedup |
|------:|:-----------|----------:|----------:|--------:|
| 108 | all-pairs | 10,734 | 4,725 | 0.4× |
| 500 | all-pairs | 651 | 1,167 | **1.8×** |
| 2,000 | all-pairs | 67 | 449 | **6.7×** |
| 5,000 | all-pairs | ~6.5* | 158 | **~24×** |
| 10,000 | cell-list | ~1.6* | 136 | **~84×** |

Paper-parity run (N=10k, 80k steps): 9.8 min, $0.0012. 98 runs/day idle.

### Streaming dispatch (v0.5.11)

- `GpuF64::begin_encoder()` / `submit_encoder()` / `read_staging_f64()`:
  batch multiple VV steps into single GPU submission
- Production MD: `dump_step` iterations batched per encoder
- Cell-list: `rebuild_interval=20` steps between CPU-side rebuilds
- Result: N=500 GPU went from 1.0× to 1.8×; N=10000 cell-list at 136 steps/s

### Next evolution targets

1. ~~**GPU-resident reduction**~~ ✅ **DONE (v0.5.12)** — `ReduceScalarPipeline` from ToadStool
2. ~~**GPU-resident cell-list**~~ ✅ **DONE (v0.5.13)** — 3-pass GPU build (bin→scan→scatter) + indirect force shader. ToadStool's `CellListGpu` had prefix_sum binding mismatch; hotSpring built corrected implementation locally. Ready for ToadStool absorption.
3. ~~**Daligault D* model evolution**~~ ✅ **DONE (v0.5.14)** — κ-dependent weak-coupling correction `C_w(κ)` replaces constant `C_w=5.3`. Crossover-regime error reduced from 44-63% to <10% across all 12 Sarkas calibration points. See Completed (v0.5.14).
4. **`StatefulPipeline` for MD loops**: ToadStool's `run_iterations()` / `run_until_converged()` could replace manual encoder batching. Architectural change — deferred.
5. **GPU HFB energy integrands**: Shader exists (`batched_hfb_energy_f64.wgsl`); wiring requires threading 10+ buffer references through `GroupResources`. Estimated ~200 lines of refactoring.
6. **Titan V proprietary driver**: unlock 6.9 TFLOPS fp64, fair hardware comparison

## Completed (v0.5.14)

- ✅ **Daligault D* model evolution**: κ-dependent weak-coupling correction
  - Replaced constant `C_w=5.3` with `C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²)`
  - Root cause: Yukawa screening suppresses the Coulomb logarithm faster than the classical formula captures; correction grows exponentially with κ (4.2× at κ=0 → 1332× at κ=3)
  - Crossover-regime errors: 44-63% → <10% (12/12 points pass 20% per-point, RMSE<10%)
  - Same `C_w(κ)` applied to η*_w and λ*_w (Chapman-Enskog transport)
  - Calibrated from `calibrate_daligault_fit.py` weak-coupling correction analysis
- ✅ **12 Sarkas D_MKS provenance constants**: all 12 Green-Kubo D_MKS values from
  `all_observables_validation.json` stored in `transport.rs::SARKAS_D_MKS_REFERENCE`
  with `A2_OMEGA_P` conversion constant and `sarkas_d_star_lookup()` function
- ✅ **Transport grid expanded**: `transport_cases()` now includes 9 Sarkas DSF points
  (κ=1: Γ=14,72,217; κ=2: Γ=31,158,476; κ=3: Γ=503,1510) alongside original 12 → 20 total
- ✅ **`sarkas_validated_cases()` added**: returns 9 κ>0 Sarkas-matched transport configs
- ✅ **`validate_stanton_murillo.rs` extended**: 2→6 transport points; Sarkas D* reference
  displayed for matched cases; cross-case D* ordering checks per-κ and cross-κ
- ✅ **Tolerance constants added**: `DALIGAULT_FIT_VS_CALIBRATION` (20% per-point),
  `DALIGAULT_FIT_RMSE` (10% over 12 points) in `tolerances.rs`
- ✅ **454 tests, 0 clippy warnings, 0 failures**

## Completed (audit, Feb 19 2026)

- ✅ **`validate_transport` added to `validate_all.rs`**: was missing from meta-validator SUITES
- ✅ **37 tolerance constants centralized**: 16 new constants in `tolerances.rs` covering transport parity,
  lattice QCD, NAK eigensolve, PPPM; wired into 7 validation binaries (was ~21 inline magic numbers)
- ✅ **`lattice/constants.rs` created**: centralizes LCG PRNG (MMIX multiplier/increment), `lcg_uniform_f64()`,
  `lcg_gaussian()` Box-Muller, `LATTICE_DIVISION_GUARD`, `N_COLORS`, `N_DIM`; wired into `su3.rs`,
  `hmc.rs`, `dirac.rs`, `cg.rs` — 14 magic-number sites eliminated
- ✅ **Provenance expanded**: `HOTQCD_EOS_PROVENANCE` struct + `HOTQCD_DOI`, `PURE_GAUGE_REFS` for
  lattice QCD validation targets
- ✅ **25 new tests**: `hfb_gpu_types` (7), `celllist` (7), `lattice/constants` (7), tolerance (2),
  provenance (2); total 441 (436 passing + 5 GPU-ignored)
- ✅ **Clippy warnings: 0** (was 3 `uninlined_format_args` in `md/transport.rs`)
- ✅ **`validate_pppm.rs` semantic fix**: multi-particle net force checks now use
  `PPPM_MULTI_PARTICLE_NET_FORCE` instead of `PPPM_NEWTON_3RD_ABS`

## Completed (v0.5.13)

- ✅ **GPU-resident cell-list build**: 3-pass GPU pipeline (bin → exclusive prefix sum → scatter)
  replaces CPU cell-list rebuild. Zero CPU readback for neighbor-list updates.
  - 4 new WGSL shaders: `cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl`,
    `yukawa_force_celllist_indirect_f64.wgsl`
  - `GpuCellList` struct: compiles 3 pipelines, manages 5 intermediate buffers, single-encoder dispatch
  - Cell-list rebuild reduced from 7 lines (3 readbacks + CPU sort + 5 uploads) to 1 line (`gpu_cl.build()`)
  - Eliminated 1.4 MB of PCIe round-trip data per rebuild at N=10,000

- ✅ **Indirect force shader**: Positions, velocities, and forces stay in original particle order.
  The force shader uses `sorted_indices[cell_start[c] + jj]` for indirect neighbor access.
  VV integrator, KE shader, and thermostat are unchanged (operate on original-order arrays).

- ✅ **VACF particle-identity fix**: The old sorted-array approach scrambled particle ordering
  every 20 steps when the cell-list was rebuilt. Velocity autocorrelation C(t) = ⟨v(t)·v(0)⟩
  requires consistent particle identity across snapshots. The indirect approach preserves
  original particle ordering — VACF is now correct by construction.

- ✅ **ToadStool `CellListGpu` binding mismatch found**: ToadStool's `prefix_sum.wgsl` has a
  4-binding layout (uniform + 3 storage) but `cell_list_gpu.rs` wires a 3-binding BGL
  (2 storage + 1 uniform) — the pipeline would fail to create. hotSpring's local
  `exclusive_prefix_sum.wgsl` matches the intended 3-binding layout. Ready for ToadStool absorption.

- ✅ **Zero regressions**: 279 tests pass, 0 clippy warnings, 0 doc warnings, 0 linter errors.

## Completed (v0.5.12)

- ✅ **ReduceScalarPipeline rewire**: Local `SHADER_SUM_REDUCE` (inline WGSL copy of barracuda's
  `sum_reduce_f64.wgsl`) removed from `md/shaders.rs`. Both `md/simulation.rs` and `md/celllist.rs`
  now use `barracuda::pipeline::ReduceScalarPipeline::sum_f64()` for KE, PE, and thermostat reductions.
  Eliminated ~50 lines of boilerplate per MD path (4 bind groups, 6 buffers, reduce_pipeline).
- ✅ **Error bridge**: `HotSpringError::Barracuda(BarracudaError)` variant + `From` impl enables
  clean `?` propagation from barracuda primitive calls into hotSpring result types.
- ✅ **Zero regressions**: 454 tests pass (449 + 5 GPU-ignored), 0 clippy warnings, 0 doc warnings.

## Completed (v0.5.11)

- ✅ **Barracuda op integration**: Validation binaries now use `barracuda::ops::md::forces::YukawaForceF64`
  directly instead of the local `yukawa_nvk_safe` workaround module (deleted)
- ✅ **Driver-aware shader compilation**: Added `GpuF64::create_pipeline_f64()` which delegates to
  barracuda's `WgpuDevice::compile_shader_f64()` — auto-patches `exp()` on NVK/nouveau
- ✅ **Production MD uses `create_pipeline_f64()`**: Force shaders in `simulation.rs` now compile
  via driver-aware path; VV/berendsen/KE use raw path (no exp/log, safe on all drivers)
- ✅ **Removed `yukawa_nvk_safe.rs`** and `yukawa_force_f64_nvk_safe.wgsl` — barracuda handles NVK
- ✅ **Clippy pedantic+nursery: 0 warnings** in library code (was 167+ cast_precision_loss, 87 mul_add)
- ✅ **Zero doc warnings**: All rustdoc links resolve correctly
- ✅ **Auto-fixed 28 clippy suggestions** across binaries (redundant clone, From, let-else)

## Completed (v0.5.10)

- ✅ **GPU density pipeline**: `batched_hfb_density_f64.wgsl` wired into `hfb_gpu_resident.rs`
  - Shader updated for batched per-nucleus wavefunctions
  - 14 GPU buffers, 3 compute pipelines (`compute_density`, `mix_density`)
  - Full staging readback + CPU density fallback
  - SCF loop restructured: CPU Brent → GPU density + mixing → CPU energy
  - L2 GPU validation confirmed: chi2/datum=5.42, physics consistent
- ✅ **Energy pipeline stubs**: bind groups, buffers, staging for future GPU energy dispatch
- ✅ **rho buffers upgraded**: `COPY_SRC` added for GPU-to-staging density transfer

## Completed (v0.5.8)

- ✅ **WGSL preamble injection**: `abs_f64` (bcs_bisection) and `cbrt_f64` (potentials) replaced
  with `ShaderTemplate::with_math_f64_auto()` — zero duplicate WGSL math
- ✅ **Exhaustive tolerance wiring**: 8 new constants (`FACTORIAL_TOLERANCE`, `ASSOC_LEGENDRE_TOLERANCE`,
  `DIGAMMA_FD_TOLERANCE`, `BETA_VIA_LNGAMMA_TOLERANCE`, `INCOMPLETE_GAMMA_TOLERANCE`,
  `BESSEL_NEAR_ZERO_ABS`, `RHO_POWF_GUARD`, `GPU_JACOBI_CONVERGENCE`)
- ✅ **Core physics tolerance wiring**: `hfb.rs`, `hfb_gpu.rs`, `hfb_gpu_resident.rs` — all
  inline density floors, powf guards, GPU eigensolve thresholds → named constants
- ✅ **Comprehensive audit**: zero unsafe, zero TODO/FIXME, zero mocks, zero hardcoded paths,
  all AGPL-3.0 licensed, all validation binaries follow hotSpring pattern

## Completed (v0.5.7)

- ✅ `hermite_value` → delegates to `barracuda::special::hermite` (zero duplicate math)
- ✅ Validation binary tolerances fully wired: `validate_linalg`, `validate_special_functions`,
  `validate_md`, `validate_barracuda_pipeline`, `validate_optimizers` (~50 inline → ~12 niche)
- ✅ `HFB_TEST_NUCLEI_PROVENANCE` — machine-readable `BaselineProvenance` struct
- ✅ L1/L2 provenance environments expanded: NumPy 1.24, SciPy 1.11, mystic 0.4.2
- ✅ 7 new determinism + coverage tests (189 total, 44% line / 61% function coverage)
- ✅ `GpuResidentL2Result` fully documented

## Completed (v0.5.6)

- ✅ `SpinOrbitGpu` wired into `hfb_gpu_resident.rs` with CPU fallback
- ✅ `compute_ls_factor` from barracuda replaces manual `(j(j+1)-l(l+1)-0.75)/2` in `hfb.rs`, `hfb_gpu_resident.rs`
- ✅ Physics guard constants centralized: `DENSITY_FLOOR`, `SPIN_ORBIT_R_MIN`, `COULOMB_R_MIN`
  - 20+ inline `1e-15`, `0.1`, `1e-10` guards replaced across 5 physics modules
- ✅ SPDX headers added to all 17 WGSL shaders that were missing them (30/30 total)
- ✅ `panic!()` in library code converted to `expect()` (GPU buffer map failures)
- ✅ WGSL math duplicates resolved via `ShaderTemplate::with_math_f64_auto()` preamble injection

## Completed (v0.5.5)

- ✅ `data::load_eos_context()` → Shared EOS context loading for all nuclear EOS binaries
- ✅ `data::chi2_per_datum()` → Shared χ² computation with `tolerances::sigma_theo`
- ✅ `tolerances::BFGS_TOLERANCE` → Corrected from 0.1 to 1e-4 with proper justification
- ✅ `validate_optimizers` → Wired to use `tolerances::BFGS_TOLERANCE`
- ✅ All inline WGSL extracted from `celllist_diag.rs`
- ✅ 16 new unit tests (176 total)
- ✅ `verify_hfb` added to `validate_all` meta-validator

## Completed (v0.5.4)

- ✅ `hfb_gpu_resident.rs` → GPU eigensolve via `execute_single_dispatch` (was CPU `eigh_f64`)
- ✅ `validate_nuclear_eos` → Formal L1/L2/NMP validation with harness (37 checks)
- ✅ `validate_all` → Meta-validator for all 33 validation suites

## Completed (v0.5.3)

- ✅ `bcs_gpu.rs` → ToadStool `target` keyword fix absorbed (commit `0c477306`)
- ✅ `hfb_gpu.rs` → Single-dispatch eigensolve wired
- ✅ `hfb_deformed_gpu.rs` → Single-dispatch with fallback wired
- ✅ MD shaders → 5 large shaders extracted to `.wgsl` files
- ✅ BCS pipeline → Shader compilation cached at construction

## GPU Reduction via ReduceScalarPipeline (v0.5.12, Feb 19 2026)

**Rewired**: Local `SHADER_SUM_REDUCE` (inline WGSL copy) replaced by
`barracuda::pipeline::ReduceScalarPipeline` from ToadStool (Feb 19 2026
absorption). Both `simulation.rs` and `celllist.rs` now call
`reducer.sum_f64(&buffer)` instead of manually managing 4 bind groups,
6 intermediate buffers, and 4 reduce dispatches per energy readback.

| Before (N=10000) | After | Reduction |
|-------------------|-------|-----------|
| KE readback: 80 KB (N×8) | 8 bytes (1 scalar) | 10,000× |
| PE readback: 80 KB (N×8) | 8 bytes (1 scalar) | 10,000× |
| Equil thermostat: 80 KB | 8 bytes | 10,000× |
| Total per dump: 160 KB | 16 bytes | 10,000× |

**Code eliminated**: ~50 lines of boilerplate per MD path (reduce pipeline,
partial buffers, scalar buffers, param buffers, bind groups, inline dispatches).
`SHADER_SUM_REDUCE` removed from `shaders.rs`.

**Validation**: 454 tests pass, 0 clippy warnings, 0 doc warnings.

**Remaining readback**: position/velocity snapshots for VACF and cell-list
rebuilds. Both can be eliminated with GPU-resident VACF and `CellListGpu`.

See `wateringHole/handoffs/HOTSPRING_UNIDIRECTIONAL_FEEDBACK_FEB19_2026.md`
for full design feedback to ToadStool on the unidirectional pattern,
`StatefulPipeline` proposal, and NAK universal solution.

## Multi-GPU Benchmark Results (v0.5.10, Feb 17 2026)

RTX 4070 (nvidia proprietary) vs Titan V (NVK/nouveau, open-source).
See `experiments/006_GPU_FP64_COMPARISON.md` for full analysis.

| Workload | RTX 4070 | Titan V | Ratio | Bottleneck |
|----------|----------|---------|-------|------------|
| BCS bisection (2048 nuclei) | 1.54 ms | 1.72 ms | 1.11× | Dispatch overhead |
| Eigensolve 128×20 | 4.64 ms | 31.11 ms | 6.7× | NVK shader compiler |
| Eigensolve 128×30 | 12.95 ms | 93.33 ms | 7.2× | NVK shader compiler |
| L2 HFB pipeline (18 nuclei) | 8.1 ms | 8.6 ms | 1.06× | CPU-dominated (Amdahl) |

**Key finding**: NVK is functionally correct (identical physics to 1e-15)
but 4–7× slower on compute-bound shaders. This is a driver maturity gap,
not hardware. Proprietary driver on Titan V would unlock its 6.9 TFLOPS fp64.

### NVK Reinstated (Feb 21, 2026)

NVK rebuilt from Mesa 25.1.5 source (`-Dvulkan-drivers=nouveau`) after Pop!_OS
Mesa package was found to omit NVK. Titan V now visible as GPU1 alongside
RTX 4070 (GPU0). Full validation re-confirmed:

- `validate_cpu_gpu_parity`: **6/6 checks passed** (energy, temperature, D* parity)
- `validate_stanton_murillo`: **40/40 checks passed** (6 (κ,Γ) transport cases, 16.5 min)
- `bench_gpu_fp64`: BCS bisection + batched eigensolve — all correct

ToadStool barracuda auto-detects NVK via adapter info and applies Volta
driver profile (`DriverKind::Nvk`, `CompilerKind::Nak`, `GpuArch::Volta`).

## Lattice QCD Modules (Updated Feb 20, 2026 — Post ToadStool Session 25)

| Rust Module | Lines | WGSL | Tier | Status |
|-------------|-------|------|------|--------|
| `lattice/complex_f64.rs` | 257 | `WGSL_COMPLEX64` → `shaders/complex_f64.wgsl` | **✅** | **Absorbed** — toadstool `8fb5d5a0`; extracted v0.6.3 |
| `lattice/su3.rs` | 393 | `WGSL_SU3` → `shaders/su3_f64.wgsl` | **✅** | **Absorbed** — toadstool `8fb5d5a0`; extracted v0.6.3 |
| `lattice/wilson.rs` | 338 | → `wilson_plaquette_f64.wgsl` | **✅** | **Absorbed** — GPU plaquette shader |
| `lattice/hmc.rs` | 350 | → `su3_hmc_force_f64.wgsl` | **✅** | **Absorbed** — GPU HMC force shader |
| `lattice/abelian_higgs.rs` | ~500 | → `higgs_u1_hmc_f64.wgsl` | **✅** | **Absorbed** — GPU U(1) Higgs HMC |
| `lattice/dirac.rs` | 440+ | `WGSL_DIRAC_STAGGERED_F64` | **A** | ✅ GPU validated (8/8 checks, max error 4.44e-16); ready for toadstool absorption |
| `lattice/cg.rs` | 320+ | `WGSL_COMPLEX_DOT_RE_F64` + `WGSL_AXPY_F64` + `WGSL_XPAY_F64` | **A** | ✅ GPU validated (9/9 checks, CG iterations match CPU exactly) |
| `lattice/eos_tables.rs` | 307 | — | N/A | HotQCD reference data (CPU-only) |
| `lattice/multi_gpu.rs` | 237 | — | **C** | CPU-threaded dispatcher; needs GPU dispatch |

## Spectral Theory Modules (Feb 22, 2026 — Fully Leaning on Upstream)

**All spectral source code deleted from hotSpring.** The `spectral/mod.rs` now
contains only re-exports from `barracuda::spectral` plus a `CsrMatrix` type
alias for backward compatibility. ~41 KB of local source removed.

ToadStool absorbed the entire spectral module in Sessions 25-31h (commit
`dc540afd`..`0bd6a92d`), including Anderson 1D/2D/3D, Lanczos, Hofstadter,
Sturm tridiagonal, level statistics, CSR SpMV, and a new `BatchIprGpu`.

| Upstream Module | hotSpring Status |
|----------------|------------------|
| `barracuda::spectral::anderson` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::SpectralCsrMatrix` | ✅ **Leaning** — aliased as `CsrMatrix` |
| `barracuda::spectral::lanczos` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::hofstadter` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::tridiag` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::stats` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::BatchIprGpu` | ✅ **NEW** — available via re-export |

### HMC Implementation Notes

The HMC leapfrog integrator uses the **Cayley transform** for the SU(3) matrix
exponential: `exp(X) ≈ (I + X/2)(I - X/2)^{-1}`. This is exactly unitary when
X is anti-Hermitian, eliminating unitarity drift that plagues Taylor approximations.
The 3×3 inverse uses cofactor expansion (exact, no iteration).

Gauge force: `dP/dt = -(β/3) Proj_TA(U × V)` where V is the staple sum
(NOT V†). This was debugged from first principles during the Feb 19 audit —
the original sign and adjoint were both wrong, causing 0% HMC acceptance.

### Lattice QCD GPU Promotion Roadmap (Updated Feb 20, 2026)

1. ~~**Complex f64 + SU(3)**~~ ✅ **Absorbed** — toadstool `8fb5d5a0`
2. ~~**Plaquette shader**~~ ✅ **Absorbed** — `wilson_plaquette_f64.wgsl`
3. ~~**HMC force on GPU**~~ ✅ **Absorbed** — `su3_hmc_force_f64.wgsl` + `higgs_u1_hmc_f64.wgsl`
4. ~~**FFT**~~ ✅ **Absorbed** — `Fft1DF64` + `Fft3DF64` (toadstool `1ffe8b1a`)
5. ~~**Dirac apply on GPU**~~ ✅ **Done** — `WGSL_DIRAC_STAGGERED_F64` validated
   8/8 checks (cold/hot/asymmetric lattices, max error 4.44e-16).
6. ~~**GPU CG solver**~~ ✅ **Done** — `WGSL_COMPLEX_DOT_RE_F64` + `WGSL_AXPY_F64`
   + `WGSL_XPAY_F64` validated 9/9 checks. Full CG iteration on GPU:
   D†D + dot + axpy + xpay, only scalar coefficients transfer per iteration.
   **Full GPU lattice QCD pipeline: COMPLETE.**
7. ✅ **Pure GPU workload validated** — `validate_pure_gpu_qcd` (3/3 checks):
   CPU HMC thermalization (10 traj, 100% accepted) → GPU CG on thermalized
   configs (5 solves, 32 iters each, exact iteration match, solution parity
   4.10e-16). Only 24 bytes/iter CPU↔GPU transfer. **Production-like workload: VALIDATED.**
8. ✅ **Python baseline established** — `bench_lattice_cg` + `lattice_cg_control.py`:
   Rust 200× faster than Python on identical CG algorithm. Iterations match exactly
   (5 cold, 37 hot). Dirac apply: 0.023ms (Rust) vs 4.59ms (Python) = 200×.

## Evolution Gaps Identified

| Gap | Impact | Priority | Status |
|-----|--------|----------|--------|
| ~~GPU energy integrands not wired in spherical HFB~~ | ~~CPU bottleneck in SCF energy~~ | ~~High~~ | ✅ Resolved v0.6.2: `batched_hfb_energy_f64.wgsl` wired behind `gpu_energy` feature flag |
| ~~`SumReduceF64` not used for MD energy sums~~ | ~~CPU readback for reduction~~ | ~~High~~ | ✅ Resolved v0.5.12: `ReduceScalarPipeline` (GPU-buffer variant) |
| ~~Lattice QCD GPU shaders~~ | ~~CPU-only lattice modules~~ | ~~Medium~~ | ✅ Absorbed by toadstool S25 (5 GPU shaders) |
| ~~GPU SpMV (CSR)~~ | ~~CPU-only sparse matrix-vector product~~ | ~~**P1**~~ | ✅ **Done** — `WGSL_SPMV_CSR_F64` validated, `validate_gpu_spmv` binary (28th suite) |
| ~~GPU Lanczos~~ | ~~CPU-only iterative eigensolve~~ | ~~**P1**~~ | ✅ **Done** — GPU SpMV Lanczos validated, `validate_gpu_lanczos` (29th suite) |
| ~~GPU Dirac SpMV~~ | ~~CPU-only staggered Dirac operator~~ | ~~**P1**~~ | ✅ **Done** — `WGSL_DIRAC_STAGGERED_F64` validated, `validate_gpu_dirac` binary (30th suite) |
| ~~Pure GPU QCD workload~~ | ~~Thermalized-config CG validation~~ | ~~**P1**~~ | ✅ **Done** — `validate_pure_gpu_qcd` (3/3): HMC → GPU CG, 4.10e-16 parity (31st suite) |
| ~~Python baseline~~ | ~~Interpreted-language benchmark~~ | ~~**P1**~~ | ✅ **Done** — Rust 200× faster: CG iters match exactly, Dirac 0.023ms vs 4.59ms |
| ~~8 files > 1000 lines~~ | ~~Code organization~~ | ~~Medium~~ | ✅ Resolved v0.6.1: 4 monolithic files decomposed into module dirs. Only `hfb_gpu_resident/mod.rs` (1456) remains >1000 (justified — monolithic GPU pipeline) |
| ~~Stanton-Murillo transport normalization~~ | ~~Paper 5 calibration~~ | ~~High~~ | ✅ Resolved: Sarkas-calibrated (12 points, N=2000) |
| ~~BCS + density shader not wired~~ | ~~CPU readback after eigensolve~~ | ~~High~~ | ✅ Resolved v0.5.10 |
| ~~WGSL inline math~~ | ~~Maintenance drift~~ | ~~Medium~~ | ✅ Resolved v0.5.8 |
| ~~Hardcoded tolerances in validation binaries~~ | ~~Not traceable/justified~~ | ~~High~~ | ✅ Resolved: 37 constants in `tolerances.rs` |
| ~~Lattice LCG magic numbers scattered~~ | ~~Maintenance risk~~ | ~~Medium~~ | ✅ Resolved: `lattice/constants.rs` centralizes all |

## Gaps Resolved (v0.5.5)

- ✅ `celllist_diag.rs` inline WGSL → Extracted 8 shaders to `.wgsl` files (1672 → 1124 lines)
- ✅ Dead_code in deformed HFB → 6 field renames, 3 documented GPU-reserved functions
- ✅ Nuclear EOS path duplication → Shared `data::load_eos_context()` replaces 9 inline path constructions
- ✅ Inline tolerances → 30+ magic numbers replaced with `tolerances::` constants
- ✅ Inline `sigma_theo` → 19 instances replaced with `tolerances::sigma_theo()`

## Evolution v0.6.9 → v0.6.13 (Feb 24-25, 2026)

| Version | Key Change | Status |
|---------|-----------|--------|
| v0.6.9 | toadStool S62 sync. Spectral lean (41 KB deleted). CsrMatrix alias | ✅ Fully leaning |
| v0.6.10 | DF64 gauge force on RTX 3090. 9.9× FP32 core throughput | ✅ Production |
| v0.6.11 | t-major site indexing standardization. 119/119 unit tests | ✅ Convention adopted |
| v0.6.12 | toadStool S60 DF64 expansion (plaquette, KE, transcendentals). 60% HMC DF64 | ✅ Absorbed |
| v0.6.13 | GPU Polyakov loop (72× less transfer). NVK guard. su3_math_f64. PRNG fix | ✅ 13/13 checks |

### New Shaders (v0.6.10-v0.6.13)

| Shader | Version | Origin | Absorption Status |
|--------|---------|--------|-------------------|
| `su3_gauge_force_df64.wgsl` | v0.6.10 | hotSpring | Local (DF64 neighbor-buffer variant) |
| `wilson_plaquette_df64.wgsl` | v0.6.12 | toadStool S60 | Bidirectional |
| `su3_kinetic_energy_df64.wgsl` | v0.6.12 | toadStool S60 | Bidirectional |
| `polyakov_loop_f64.wgsl` | v0.6.13 | toadStool → hotSpring | Bidirectional (pending upstream sync) |
| `su3_math_f64.wgsl` | v0.6.13 | hotSpring | Local (pending upstream absorption) |

### Key Discoveries

- **DF64 core streaming** (v0.6.10): FP32 cores deliver 3.24 TFLOPS at 14-digit precision on consumer GPUs. 9.9× native f64 throughput.
- **Naga composition bug** (v0.6.13): `naga` rejects WGSL with unused `ptr<storage>` functions when prepended as preamble. Workaround: split into `su3_math_f64.wgsl`.
- **PRNG type mismatch** (v0.6.13): `ShaderTemplate` patches `cos`→`cos_f64`, but `f32(theta)` broke the call. Fix: keep theta as `f64` throughout.
