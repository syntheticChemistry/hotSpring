# Changelog

All notable changes to the hotSpring BarraCuda validation crate.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.6.14 — Debt Reduction + Cross-Primal Discovery (Feb 25, 2026)

### Cross-primal hardcoding → capability-based discovery

Eliminated all hardcoded cross-primal references from production code.
Code now has self-knowledge only and discovers other primals at runtime.

- Changed: `validate_three_substrate.rs`, `production_mixed_pipeline.rs` — replaced
  hardcoded `/dev/akida0` path with `discovery::probe_npu_available()` (sysfs scan)
- Changed: `discovery.rs` — renamed `control/metalforge_npu` → `control/npu`
  (generic capability, no embedded primal name)
- Changed: `nuclear_eos_gpu.rs` — removed toadstool path from error message
- Changed: `bench_multi_gpu.rs` — replaced metalForge doc reference with env vars
- Added: `discovery::probe_npu_available()` — feature-gated NPU probe
  (`npu-hw` → akida-driver, fallback → `/dev/akida*` sysfs scan)

### Gauss-Jordan → barracuda LU decomposition

Replaced the local 68-line `gauss_jordan_solve` in `reservoir.rs` with
`barracuda::ops::linalg::lu_solve` — the shared primitive with partial pivoting.
Same API surface (dense Ax=b), but uses the validated barracuda implementation.

### WGSL shader deduplication

Created `prng_pcg_f64.wgsl` shared PRNG library (PCG hash + uniform f64).
Lattice PRNG shaders (`su3_random_momenta_f64`, `gaussian_fermion_f64`) now
compose via `include_str!` concatenation, eliminating 30 lines of duplicated code.
Each consumer retains its own `box_muller_cos`/`gaussian` (f64 cos vs f32 cast
difference is intentional — validated separately).

### Clippy pedantic/nursery: zero warnings (lib + bins)

Fixed all 57 clippy warnings across 17 validation binaries:
- `manual_div_ceil` → `.div_ceil()` (7)
- `uninlined_format_args` → inline `{var}` (13)
- `manual_let_else` → `let...else` (1)
- `ref_option` → `Option<&T>` (1)
- `needless_pass_by_ref_mut` → `&T` (2)
- `collection_is_never_read` — removed dead polyakov collectors (3)
- `no_effect_underscore_binding` — removed unused bindings (2)
- Various: `redundant_clone`, `map_unwrap_or`, `unnecessary_hashes`,
  `manual_midpoint`, `cloned_instead_of_copied`, etc.

### Discovery coverage improvements

Added 5 new tests to `discovery.rs` (18 → 23 tests):
- `probe_npu_available_returns_bool` — exercises the NPU probe path
- `nuclear_eos_dir_resolves` — ensures resolution produces non-empty path
- `capability_probes_have_unique_names` — validates probe registry uniqueness
- `capability_probes_paths_are_relative` — no accidental absolute paths
- `is_valid_root_rejects_nonexistent` — hardening

### Validation fidelity hardening

- Added: `provenance::KNOWN_BETA_C_SU3_NT4` (5.6925) with Bali/Engels/Creutz citations
- Changed: 7 binaries migrated from local `const KNOWN_BETA_C` to centralized provenance
- Added: `tolerances::GPU_STREAMING_PLAQUETTE_PARITY` (1e-10) — streaming vs dispatch
- Added: `tolerances::GPU_FERMION_FORCE_PARITY` (1e-12) — CPU vs GPU fermion force
- Added: `tolerances::GPU_CG_ACTION_PARITY` (1e-6) — CPU vs GPU CG action
- Added: `tolerances::GPU_DYN_STREAMING_PLAQUETTE_PARITY` (0.10) — dynamical streaming

### Audit results (clean)

- 0 mocks in production code (NpuSimulator is a legitimate software implementation)
- 0 TODO/FIXME/HACK/unimplemented markers across 167 .rs files
- 0 clippy warnings (lib + bins, pedantic + nursery)
- 664 tests passing (629 lib + 31 integration + 4 doc), 0 failures
- ~150 centralized tolerances with physics justification (~95% coverage)

---

## v0.6.13 — GPU Polyakov Loop + NVK Guard + PRNG Fix (Feb 25, 2026)

### GPU-resident Polyakov loop (eliminates CPU readback)

Absorbed toadStool `GpuPolyakovLoop` pattern: temporal Wilson line computed
entirely on GPU. Returns both magnitude and phase (previously only magnitude
via CPU readback of the full V×4×18 link buffer).

- Added: `polyakov_loop_f64.wgsl` — GPU Polyakov loop shader (t-major indexing)
- Added: `su3_math_f64.wgsl` — naga-safe SU(3) math library for shader composition
- Changed: `GpuHmcPipelines` — added `polyakov_pipeline`
- Changed: `GpuHmcState` — added `poly_out_buf`, `poly_params_buf`, `spatial_vol`
- Changed: `gpu_polyakov_loop()` — GPU-compute, returns `(magnitude, phase)` tuple
- Changed: All 5 binary call sites updated for new signature
- Fixed: `su3_random_momenta_f64.wgsl` — f32/f64 type mismatch in `box_muller_cos`

### NVK allocation guard (toadStool cross-spring evolution)

- Added: `GpuHmcState::from_lattice()` — NVK allocation guard warns when total
  estimated VRAM exceeds nouveau PTE fault limit (~1.2 GB)

### Benchmark (v0.6.13, RTX 3090 with DF64 Hybrid)

| Lattice | CPU ms/traj | GPU ms/traj | Speedup |
|---------|-------------|-------------|---------|
| 4⁴      | 73.4        | 22.6        | 3.2×    |
| 8⁴      | 1157.3      | 30.1        | 38.5×   |
| 8³×16   | 2341.7      | 48.1        | 48.6×   |
| 16⁴     | 18342.1     | 259.5       | 70.7×   |

### Validation (all pass)

- `validate_gpu_streaming`: 7/7 — bit-identical streaming vs dispatch parity
- `validate_gpu_beta_scan`: 6/6 — monotonic plaquette, physical Polyakov loop
- `bench_gpu_hmc`: 4 lattice sizes, 100% acceptance at β=6.0

### Cross-spring evolution

- toadStool `GpuPolyakovLoop` → hotSpring v0.6.13 (GPU-resident observable)
- toadStool `check_allocation_safe()` → hotSpring v0.6.13 (NVK PTE guard)
- hotSpring `su3_math_f64.wgsl` (naga-safe) → candidate for toadStool absorption

---

## v0.6.12 — toadStool S60 DF64 Expansion (Feb 25, 2026)

### Expanded DF64 core streaming to 60% of HMC

Absorbed toadStool S60 improvements: FMA-optimized df64_core.wgsl,
df64_transcendentals.wgsl, and expanded DF64 coverage from gauge force only
(40% of HMC) to gauge force + Wilson plaquette + kinetic energy (60% of HMC).

- Added: `wilson_plaquette_df64.wgsl` — DF64 plaquette with neighbor-buffer indexing
- Added: `su3_kinetic_energy_df64.wgsl` — DF64 kinetic energy (from toadStool)
- Changed: `GpuHmcPipelines::new()` — auto-selects DF64 for plaquette and KE on consumer GPUs
- Changed: Uses `su3_df64_preamble()` with FMA-optimized two_prod and transcendentals
- Added: `production_mixed_pipeline.rs` — three-substrate production binary (3090+NPU+Titan V)
- Added: Experiment 015 write-up (mixed pipeline partial results)

### Benchmark (v0.6.12 vs v0.6.11)

| Lattice | v0.6.11 | v0.6.12 | Improvement |
|---------|---------|---------|-------------|
| 8⁴      | 36.7 ms | 32.2 ms | 12% faster  |
| 16⁴     | 293 ms  | 270 ms  | 8% faster   |

### Cross-spring evolution

- toadStool S60 → hotSpring v0.6.12: FMA df64_core, transcendentals, KE shader
- hotSpring → toadStool: neighbor-buffer DF64 plaquette shader pattern

---

## v0.6.11 — Site-Indexing Standardization (Feb 25, 2026)

### Breaking: t-major site ordering

Migrated `Lattice::site_index` and `site_coords` from hotSpring's original
x-fastest ordering (`idx = x + Nx*(y + Ny*(z + Nz*t))`) to toadStool's
z-fastest ordering (`idx = t*NxNyNz + x*NyNz + y*Nz + z`).

This aligns hotSpring with upstream toadStool lattice ops, enabling direct use
of upstream DF64 shaders and HMC operations without buffer reordering. Existing
serialized lattice snapshots are incompatible with this ordering.

- Changed: `Lattice::site_index()` — z fastest, t slowest (was x fastest)
- Changed: `Lattice::site_coords()` — decomposition matches new convention
- Validated: 119/119 unit tests pass
- Validated: 3/3 pure GPU HMC checks (plaq=0.584339, 100% acceptance)
- Validated: 6/6 GPU beta scan checks (plaquette monotonic, cross-lattice parity)
- Validated: 7/7 streaming HMC checks (dispatch/streaming parity exact to 1e-8)
- Version: 0.6.10 → 0.6.11

## v0.6.10 — DF64 Core Streaming Rewire (Feb 25, 2026)

### DF64 Gauge Force — Live on Consumer GPUs

Rewired `GpuHmcPipelines` to auto-select DF64 (f32-pair) gauge force shader
on consumer GPUs (RTX 3090, 4070 — 1:64 FP64:FP32 hardware). The staple SU(3)
multiplications (18 per link, 40% of HMC wall time) now route through the FP32
core array via `su3_gauge_force_df64.wgsl`.

- New `su3_gauge_force_df64.wgsl`: DF64 staple computation with hotSpring's
  neighbor-buffer indexing, f64 algebra projection + final multiply
- `Fp64Strategy::Hybrid` auto-detected on RTX 3090, selects DF64 force
- `Fp64Strategy::Native` on Titan V/V100/A100, keeps native f64
- DF64 math library imported from upstream: `WGSL_DF64_CORE` + `WGSL_SU3_DF64`
- Validated: all existing checks pass (3/3 pure GPU HMC, plaquette physical range)

### Cross-Spring Evolution

- hotSpring Exp 012 (Feb 2026) → `df64_core.wgsl` → toadStool S58 absorption
- toadStool built `su3_df64.wgsl` + DF64 HMC pipeline (S58-S62)
- hotSpring v0.6.10 imports upstream DF64 math, writes local DF64 force with
  hotSpring's neighbor-buffer indexing (incompatible site-indexing between repos)
- wetSpring f64 transcendental workarounds (Ada Lovelace) contributed via
  `Workaround::NvvmAdaF64Transcendentals` in driver_profile.rs

### Metrics
- **All validation checks pass** with DF64 active on RTX 3090
- **Version**: 0.6.9 → 0.6.10

## v0.6.9 — ToadStool S58–S62 Sync + DF64 Absorption Confirmation (Feb 24, 2026)

### ToadStool Sync (S53 → S62)

Pulled 10 toadStool sessions (172 files changed, +15,847/-7,066 lines).
hotSpring compiles cleanly against the new upstream with zero errors/warnings.

### DF64 Core Streaming — Absorbed and Extended

toadStool absorbed hotSpring's DF64 discovery (Experiment 012) in S58, then
built the full production DF64 HMC pipeline:

- `df64_core.wgsl` → `shaders/math/df64_core.wgsl` (absorbed)
- `su3_df64.wgsl` — DF64 SU(3) matrix algebra (NEW, toadStool-built)
- `su3_hmc_force_df64.wgsl` — DF64 gauge force (NEW, the 6.7× speedup kernel)
- `wilson_plaquette_df64.wgsl`, `wilson_action_df64.wgsl`, `kinetic_energy_df64.wgsl` (NEW)
- `Fp64Strategy` enum — auto-selects Native vs Hybrid per-GPU
- All lattice ops now auto-select f64/DF64 based on hardware

### Local Cleanup

- Deleted local `df64_core.wgsl` (now upstream at `barracuda::ops::lattice::su3::WGSL_DF64_CORE`)
- Updated ABSORPTION_MANIFEST with S58–S62 absorption tracking

### Metrics
- **39/39 validation suites** pass (unchanged)
- **Zero compile errors** against toadStool S62
- **Version**: 0.6.8 → 0.6.9

## v0.6.8 — biomeGate Prep + Streaming CG + Debt Fix + Suite Expansion (Feb 23, 2026)

### biomeGate Node Preparation
Registered biomeGate (Threadripper 3970X, RTX 3090 + Titan V, Akida NPU, 256GB DDR4)
as lab-deployable mini HPC. Node profile system (`metalForge/nodes/`) enables
`source biomegate.env && cargo run` for any node. Multi-GPU bench now reads
`HOTSPRING_GPU_PRIMARY`/`HOTSPRING_GPU_SECONDARY` env vars (was hardcoded 4070/titan).
NVK setup guide at `metalForge/gpu/nvidia/NVK_SETUP.md`.

### API Debt Resolution
- `md/reservoir.rs`: `solve_f64` → local `gauss_jordan_solve()` CPU fallback (ESN matrices 50-200 dim)
- `nuclear_eos_l1_ref.rs`, `nuclear_eos_gpu.rs`, `nuclear_eos_l2_hetero.rs`, `nuclear_eos_l2_ref.rs`:
  `direct_sampler`, `sparsity_sampler`, `RBFSurrogate::train` updated for `Arc<WgpuDevice>` first arg

### Validation Suite Expansion (34 → 39)
`validate_all.rs` now includes 5 additional suites:
- `validate_gpu_streaming` (9/9 — streaming HMC, 4⁴→16⁴)
- `validate_gpu_streaming_dyn` (13/13 — dynamical fermion streaming, GPU-resident CG)
- `validate_reservoir_transport` (10/10 — ESN transport prediction)
- `validate_stanton_murillo` (13/13 — Paper 5 transport)
- `validate_transport` (CPU/GPU transport parity)

### Documentation Cleanup
- Root README, whitePaper/README, whitePaper/STUDY.md updated to Feb 23
- Experiment journal 011: GPU streaming + resident CG + bidirectional pipeline
- New handoff: biomeGate prep + streaming CG + debt fix
- `experiments/data/` added to `.gitignore` (profiling CSVs untracked)
- `verify_results.py` placeholder `n_pass` removed

### Metrics
- **39/39 validation suites** pass (was 34/34)
- **155/155 checks** in latest manual validation session
- **Zero clippy warnings** (all targets)
- **All unit tests pass**
- **Version**: 0.6.7 → 0.6.8

## v0.6.7 — ToadStool S42 Catch-Up + Loop Unroller Fix (Feb 22, 2026)

### Loop Unroller u32 Fix (Applied to ToadStool)
Fixed the documented `substitute_loop_var` bug in toadstool's loop unroller:
`iter.to_string()` → `format!("{iter}u")`. WGSL now emits `0u`, `1u` etc. instead of
bare `i32` literals, resolving `BatchedEighGpu::execute_single_dispatch` WGSL
validation panics.

### catch_unwind Removal
`validate_barracuda_hfb.rs` no longer wraps single-dispatch eigensolve in
`std::panic::catch_unwind`. Direct call with `.expect()`.

### BarraCUDA → BarraCuda Rename
Display name synchronized with toadstool's S42 rename. 20 Rust source files,
28 documentation files, `Cargo.toml`, `clippy.toml`, `CITATION.cff` updated.
Archive handoffs preserved as fossil record.

### ToadStool S40–42 Evolution Tracked
- S40: Richards PDE solver, moving window GPU stats
- S41: 6 f64 shader compile bugs fixed (including GemmCachedF64), API exposure for Springs
- S42: 19 new WGSL shaders (612 total), shader-first unified math
- Jacobi eigenvector rotation fix, ODE f64 builtins, SNP binding mismatch

### Metrics
- **34/34 validation suites** pass (702.7s)
- **Zero clippy warnings** (all targets)
- **All unit tests pass**
- **Version**: 0.6.5 → 0.6.7

## v0.6.5 — Deep Debt Resolution + GPU Transport Pipeline (Feb 22, 2026)

### GPU Module Refactoring
Monolithic `gpu.rs` (895 lines) refactored into `gpu/` module:
`adapter.rs`, `buffers.rs`, `dispatch.rs`, `telemetry.rs`, `mod.rs`.

### Idiomatic Rust Evolution
- `.expect()` → `Result` in GPU transport functions
- `partial_cmp().unwrap()` → `total_cmp()` for NaN-safe sorting
- `Vec::new()+push` → iterator `collect()` patterns
- Magic numbers → named constants (`WORKGROUP_SIZE`, `PLATEAU_DETECTION_TIME`)

### API Evolution
- `PppmGpu::new()` deprecated → `PppmGpu::from_device()`
- Binary targets registered in `Cargo.toml`
- Documentation synchronized

### Metrics
- **34/34 validation suites** pass (688.0s)
- **Zero clippy warnings**
- **Version**: 0.6.4 → 0.6.5

## v0.6.4 — ToadStool Rewire v4: Spectral Lean (Feb 22, 2026)

### Spectral Module → Fully Leaning on Upstream

ToadStool Sessions 25-31h absorbed hotSpring's entire spectral module into
`barracuda::spectral`. Local source files deleted (~41 KB), replaced with
re-exports from upstream:

- `spectral/anderson.rs` → `barracuda::spectral::anderson_*`
- `spectral/csr.rs` → `barracuda::spectral::SpectralCsrMatrix` (+ `CsrMatrix` alias)
- `spectral/hofstadter.rs` → `barracuda::spectral::hofstadter_butterfly`
- `spectral/lanczos.rs` → `barracuda::spectral::lanczos`
- `spectral/stats.rs` → `barracuda::spectral::{level_spacing_ratio, detect_bands}`
- `spectral/tridiag.rs` → `barracuda::spectral::{sturm_count, find_all_eigenvalues}`
- `spectral/shaders/spmv_csr_f64.wgsl` → `barracuda::spectral::WGSL_SPMV_CSR_F64`

New upstream primitive now available: `barracuda::spectral::BatchIprGpu`.

### Documentation Updated

- `ABSORPTION_MANIFEST.md`: Spectral moved to "Already Absorbed" section
- `DEPRECATION_MIGRATION.md`: Spectral module tracked as fully deprecated
- `EVOLUTION_READINESS.md`: Spectral section updated to "Fully Leaning"
- `README.md`: Test counts updated, Rewire v4 status added

### Metrics

- **637 tests** (was 648; 44 spectral tests now run upstream in barracuda), 0 failures, 6 ignored
- **Zero clippy warnings** (all targets, pedantic)
- **Zero doc warnings**
- **Version**: 0.6.3 → 0.6.4

## v0.6.3 — WGSL Extraction & Coverage Push (Feb 22, 2026)

### Inline WGSL Extraction (5 shaders)

- `SHADER_VV_HALF_KICK` → `md/shaders/vv_half_kick_f64.wgsl`
- `SHADER_BERENDSEN` → `md/shaders/berendsen_f64.wgsl`
- `SHADER_KINETIC_ENERGY` → `md/shaders/kinetic_energy_f64.wgsl`
- `WGSL_COMPLEX64` → `lattice/shaders/complex_f64.wgsl`
- `WGSL_SU3` → `lattice/shaders/su3_f64.wgsl`

All switched to `include_str!()`. Zero inline WGSL remaining in production library code.

### Deformed HFB Coverage: 29% → 95%

13 new tests covering previously-untested critical paths:

- **`diagonalize_blocks`**: V=0 (HO eigenvalues), constant-V (shift), sharp Fermi (zero pairing)
- **`potential_matrix_element`**: constant-V diagonal, Hermitian symmetry
- **`solve()` SCF loop**: smoke test (minimal grid), deterministic rerun, physical result bounds
- **`binding_energy_l3`**: public API smoke (marked `#[ignore]` — ~90s)
- **Norm integrals**: Hermite oscillator 1D norm, Laguerre oscillator 2D norm

Module coverage: `hfb_deformed/mod.rs` 29.3% → 94.9%, `basis.rs` 65.8% → 98.7%.

### Metrics

- **648 tests** (was 638), 0 failures, 6 ignored
- **Coverage**: 74.9% region / 83.8% function / 72.0% line (was 73.0 / 82.9 / 70.4)
- **Zero clippy warnings** (all targets, pedantic + nursery)
- **Zero doc warnings**
- **Version**: 0.6.2 → 0.6.3

## v0.6.2 — Deep Debt Resolution & Pedantic Clean (Feb 21, 2026)

### Clippy Pedantic: Zero Warnings
- **0 warnings** on `clippy::pedantic + clippy::nursery` (was ~1500 in v0.6.1)
- 150+ `mul_add` conversions: improved IEEE 754 accuracy in all physics computations
- 600+ `doc_markdown` fixes: backtick-wrapped identifiers in all doc comments
- 30+ `imprecise_flops` fixes: `cbrt()`, `hypot()`, `ln_1p()` replacing less-accurate expressions
- `#[must_use]` on all pure functions and key data types (186+ annotations)
- `Self` replacing type name repetition via `use_self` lint
- `is_multiple_of()` replacing manual modulo checks
- `const fn` on 4 eligible functions (`lcg_step`, `next_u64`, `SpeciesResult::new`, `LcgRng::next_u64`)
- HashMap hasher generalized on `chi2_per_datum` and `l1_proxy_prescreen`
- Crate-level lint configuration: documented `#![allow]` for physics-appropriate lints (cast precision, sign, wrap)

### Duplicate Math Eliminated
- `reservoir.rs` Gaussian elimination (60 lines) → `barracuda::linalg::solve_f64`
- Single upstream call per RHS column; zero local linear algebra code

### Refactoring
- `bench.rs` (1005 lines) decomposed into `bench/` module directory:
  `hardware.rs` (193), `power.rs` (218), `report.rs` (354), `mod.rs` (246)
- `hfb_gpu_resident/mod.rs` refactored: 7 extracted helper functions
  (`create_potential_pipelines`, `create_hamiltonian_pipelines`, `create_density_pipelines`,
  `allocate_group_resources`, `upload_densities`, `dispatch_hbuild_and_pack`, `run_density_mixing_pass`)
- `celllist_diag.rs` reduced below 1000 lines (1156→951); shared `unsort_pe()`, `net_force()`
- `nuclear_eos_l1_ref.rs` and `nuclear_eos_l2_hetero.rs` restructured with extracted helpers

### GPU Energy Pipeline (Feature: `gpu_energy`)
- Wired `batched_hfb_energy_f64.wgsl` shader dispatch (was stub since v0.5.10)
- `compute_energy_integrands` + `compute_pairing_energy` GPU passes
- Staging buffer readback and trapezoidal sum
- CPU fallback preserved when feature disabled

### Magic Number Extraction
- `NMP_SIGMA_THRESHOLD`, `TRIDIAG_STURM_PIVOT_GUARD`, `ESN_SPECTRAL_RADIUS_NEGLIGIBLE`
- Pivot guards wired to `DIVISION_GUARD`

### Cast Safety
- Per-function `#[allow(clippy::cast_possible_truncation)]` with documented bounds
- Crate-level documented allows for physics-safe casts with mantissa/range analysis

### MutexGuard Drop Tightening
- `PowerMonitor::finish()`: GPU samples cloned and mutex released before processing

### Test Coverage
- **638 tests** (was 505 in v0.6.1), 0 failures, 5 GPU-ignored
- +133 new tests across: hfb_deformed/{potentials,basis,mod}, hfb/mod, prescreen,
  hfb_gpu_types, data, bench/{report,hardware}, md/observables/{ssf,summary},
  screened_coulomb, error, discovery, lattice/multi_gpu, spectral/stats, validation
- **72.4% region / 82.5% function** coverage (was 65.7% / 77.6%)

### CellListGpu Migration (P1 Evolution Target)
- Local `GpuCellList` (282 lines) deleted — replaced with upstream `barracuda::ops::md::CellListGpu`
- 3 local WGSL shaders deleted: `cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl`
- Force shader `yukawa_force_celllist_indirect_f64.wgsl` unchanged (buffer-compatible)

### Inline WGSL Extraction
- 5 inline shader strings extracted to dedicated `.wgsl` files:
  `complex_dot_re_f64.wgsl`, `axpy_f64.wgsl`, `xpay_f64.wgsl`, `dirac_staggered_f64.wgsl`, `spmv_csr_f64.wgsl`
- All loaded via `include_str!("shaders/filename.wgsl")`

### metalForge/forge
- Zero pedantic clippy warnings (was 10)
- `#[must_use]` on all probe functions; backticks in all doc comments

## v0.6.1 — Code Quality Evolution (Feb 21, 2026)

### Safety
- `#![deny(clippy::expect_used, clippy::unwrap_used)]` enforced in library code
- All 15 production `expect()` calls eliminated: `bytemuck` zero-copy, safe pattern matching, `Result` propagation
- Zero `unsafe` blocks (unchanged)

### Architecture
- `tolerances.rs` (1384 lines) → `tolerances/{mod,core,md,physics,lattice,npu}.rs` module tree
- `discovery.rs`: `try_discover_data_root()` Result API + `available_capabilities()` runtime probing
- `gpu.rs`: `mapped_bytes_to_f64()` helper — `bytemuck::try_cast_slice` with alignment fallback
- `hfb_common.rs`: `initial_wood_saxon_density()` extracted — shared by `hfb.rs`, `hfb_gpu.rs`, `hfb_gpu_resident.rs`
- `hfb_gpu_resident.rs`: GPU energy pipeline feature-gated behind `gpu_energy` (dead allocation removed from default build)
- `hfb.rs`: Removed unused `a` field from `SphericalHFB`; `hfb_gpu_types.rs`: removed unused `mat_size` field

### Provenance
- `SCREENED_COULOMB_PROVENANCE` + `DALIGAULT_CALIBRATION_PROVENANCE` added to `provenance.rs`
- Commit verification documentation for all 4 baseline commits
- Control JSON policy documented

### Quality
- `ENERGY_DRIFT_PCT` tightened: 5% → 0.5% (250× above measured worst case)
- `RDF_TAIL_TOLERANCE` tightened: 0.15 → 0.02 (12× above measured worst case)
- 9 new tests (3 summary.rs + 4 discovery.rs + 2 Wood-Saxon): 463 total (458 passing + 5 GPU-ignored)
- `partial_cmp().expect()` → `total_cmp()` in all sort operations
- `.to_string()` on literals → `String::from()` in production code
- `bytemuck::cast_slice` for zero-copy u32 buffer creation
- 13 library functions promoted to `const fn` (lattice accessors, spectral constructors)
- 5 redundant `.clone()` calls removed (summary, transport, VACF, reservoir)
- Deprecated `set_var`/`remove_var` suppressed in tests with edition 2024 migration plan

## [0.6.0] — 2026-02-21

### Changed

- **Full Result propagation:** Eliminated all `.expect()` and `.unwrap()` from library
  code. GPU pipelines (HFB spherical, deformed, GPU-batched), BCS solver, and ESN
  `predict()` now return `Result<T, HotSpringError>`. New error variant
  `HotSpringError::InvalidOperation` for state-dependent failures (e.g. ESN not trained).
  Provably unreachable byte-slice `try_into()` calls annotated with `#[allow(clippy::expect_used)]`
  and `// SAFETY:` justifications.
- **Idiomatic Arc usage:** All `.clone()` on `Arc<T>` replaced with `Arc::clone(&...)`
  for explicit reference-count semantics (`gpu.rs`, `bench.rs`, `ssf.rs`, `summary.rs`).
- **Tolerance expansion:** 146 centralized constants in `tolerances.rs` (up from 122).
  Added ESN/heterogeneous pipeline tolerances (`ESN_F32_LATTICE_PARITY`,
  `ESN_F32_CLASSIFICATION_AGREEMENT`, `ESN_INT4_PREDICTION_PARITY`,
  `ESN_PHASE_ACCURACY_MIN`, `ESN_MONITORING_OVERHEAD_PCT`, `PHASE_BOUNDARY_BETA_C_ERROR`)
  and `BCS_DEGENERACY_PARTICLE_NUMBER_ABS`. All validation binaries wired to named
  constants — zero inline magic numbers remain.
- **Semantic tolerance fix:** `validate_barracuda_hfb.rs` now uses
  `BCS_DEGENERACY_PARTICLE_NUMBER_ABS` instead of misused `MD_EQUILIBRIUM_FORCE_ABS`.
- **Capability-based paths:** `nuclear_eos_gpu.rs` and `sarkas_gpu.rs` benchmark result
  paths replaced with `discovery::paths::BENCHMARK_RESULTS`.

### Added

- **16 determinism tests** for all stochastic algorithms: ESN predict, HMC trajectory,
  Anderson 2D/3D, MD FCC lattice velocities, MD force computation — all verify
  reproducibility with fixed seeds.
- **4 SSF CPU-path tests:** empty snapshots, multi-frame averaging, k-spacing, single
  particle identity.
- **Test coverage:** 454 unit tests (449 passing + 5 GPU-ignored), up from 441.
  33 validation suites (33/33 pass). ~63% overall / ~96% unit-testable library coverage
  (measured with `cargo-llvm-cov`).

## [0.5.16] — 2026-02-20

### Changed

- **Code quality and idiomatic Rust:**
  - Zero `.unwrap()` across entire crate — all replaced with `.expect("descriptive message")` or `?` propagation
  - Zero `cargo clippy` warnings across all targets (pedantic + nursery lints enabled)
  - Zero `cargo doc --no-deps` warnings
  - Zero `cargo fmt` diffs
  - `#[allow()]` audit: removed 7 redundant directives, 18 justified remain with comments
  - Streaming JSON I/O in `data.rs` via `BufReader` + `serde_json::from_reader`
- **Module refactoring:** Split `spectral.rs` (1,140 lines) into `spectral/` module:
  `tridiag.rs`, `csr.rs`, `lanczos.rs`, `anderson.rs`, `hofstadter.rs`, `stats.rs`.
  Split `md/observables.rs` (1,174 lines) into `md/observables/`:
  `rdf.rs`, `vacf.rs`, `ssf.rs`, `transport.rs`, `energy.rs`, `summary.rs`.
  Refactored `hfb_gpu_resident.rs`: extracted 5 helper functions from 1,184-line
  monolithic SCF loop. All public APIs preserved — zero breaking changes.
- **Compliance:** SPDX `AGPL-3.0-only` on all 106 `.rs` files and all 34 `.wgsl` shaders (line 1).
  All hardcoded paths migrated to `discovery` module. Runtime hostname detection
  (no hardcoded system names). 122 centralized tolerance constants in `tolerances.rs`.
  Provenance records for all validation baselines.

### Added

- **Test coverage:** 441 unit tests (436 passing + 5 GPU-ignored), up from 345.
  33 validation suites (33/33 pass). Line coverage: 60.4% total, 81.2% non-GPU
  (measured with `cargo-llvm-cov`). Added tests for: spectral, observables, error
  handling, complex arithmetic, cell-list, data loading, validation harness, CPU MD,
  transport.

## [0.5.10] — 2026-02-17

### Added

- **GPU density pipeline** wired into GPU-resident HFB solver:
  - Density shader (`batched_hfb_density_f64.wgsl`) updated for batched
    per-nucleus wavefunctions (`wf[batch_idx * ns * nr + j * nr + k]`)
  - 14 GPU buffers for eigenvectors, BCS occupations, density output, mixing
  - 3 compute pipelines: `compute_density` (proton/neutron), `mix_density`
  - Full staging readback with fallback to CPU density on failure
  - Energy pipeline stubs (bind groups, buffers) for future wiring
- SCF loop restructured into 3 phases:
  1. CPU eigensolve extraction + BCS Brent bisection (parallel via Rayon)
  2. GPU density computation + mixing (batched, single encoder/submit)
  3. CPU energy computation with GPU-mixed densities
- `WorkItem` now tracks group index (`gi`) for correct GPU buffer routing
- `rho_p_buf`/`rho_n_buf` upgraded with `COPY_SRC` for GPU-to-staging transfer
- L2 GPU validation binary confirmed: chi2/datum=5.42, all physics consistent

### Fixed

- `DensityParamsUniform`, `MixParamsUniform`, `EnergyParamsUniform` fully wired
  (were stub structs, now used in GPU dispatch)

## [0.5.9] — 2026-02-17

### Added

- **6 new physics guard constants** with full justifications:
  `BCS_DENSITY_SKIP` (1e-12), `SHARP_FILLING_THRESHOLD` (0.01 MeV),
  `DEFORMED_COULOMB_R_MIN` (0.01 fm), `DEFORMATION_GUESS_WEAK` (0.05),
  `DEFORMATION_GUESS_GENERIC` (0.15), `DEFORMATION_GUESS_SD` (0.35)

### Changed

- **Exhaustive tolerance wiring — final pass**: replaced all remaining inline
  numeric literals in library code with named constants from `tolerances.rs`:
  - `hfb.rs`: 3× `1e-12` BCS density skip → `BCS_DENSITY_SKIP`
  - `hfb.rs`: `0.01` pairing gap → `SHARP_FILLING_THRESHOLD`
  - `hfb_deformed.rs`: 4× `0.01` Coulomb guard → `DEFORMED_COULOMB_R_MIN`
  - `hfb_deformed.rs`: deformation guesses → `DEFORMATION_GUESS_*`
  - `hfb_deformed_gpu.rs`: 4× `0.01` + 3× deformation guesses → named constants
  - `md/observables.rs`: `5.0` → `ENERGY_DRIFT_PCT`, `0.15` → `RDF_TAIL_TOLERANCE`
- **Clippy pedantic compliance**: 76 warnings eliminated (408 → 332):
  - Fixed ~40 `unreadable_literal` warnings (underscore separators in physics constants)
  - Fixed 4 `doc_markdown` warnings (backtick formatting in doc comments)

### Audit Results (v0.5.9)

- **Quality gates**: fmt PASS, clippy PASS (332 pedantic warnings, all justified), test PASS (189/189), doc PASS
- **Zero unsafe**, zero TODO/FIXME/HACK, zero mock code in production
- **SPDX AGPL-3.0-only**: 81/81 files (51 .rs + 30 .wgsl)
- **Remaining inline numerics in library code**: ~10 (all in test blocks or algorithm parameters)
- **BarraCuda compatibility**: upstream v0.2.0 compiles cleanly; 16 primitive families used
- **No duplicate math**: zero hand-rolled implementations of barracuda primitives

## [0.5.8] — 2026-02-17

### Added

- **11 new tolerance constants** with full physical/numerical justifications:
  `FACTORIAL_TOLERANCE`, `ASSOC_LEGENDRE_TOLERANCE`, `DIGAMMA_FD_TOLERANCE`,
  `BETA_VIA_LNGAMMA_TOLERANCE`, `INCOMPLETE_GAMMA_TOLERANCE`, `BESSEL_NEAR_ZERO_ABS`,
  `RHO_POWF_GUARD`, `GPU_JACOBI_CONVERGENCE`, `DIVISION_GUARD`,
  `PAIRING_GAP_THRESHOLD`, `SCF_ENERGY_TOLERANCE`.
- **`HotSpringError::DataLoad`** variant — `load_eos_context()` now returns
  `Result<EosContext, HotSpringError>` instead of panicking on file load failures.

### Changed

- **WGSL preamble injection:** Inline `abs_f64` (bcs_bisection) and `cbrt_f64`
  (potentials) replaced with `ShaderTemplate::with_math_f64_auto()` canonical
  barracuda math library injection. Zero duplicate WGSL math.
- **Exhaustive tolerance wiring across all physics modules:**
  - `validate_special_functions`: factorial, assoc Legendre, digamma, beta,
    incomplete gamma, Bessel near-zero
  - `hfb.rs`: `DENSITY_FLOOR`, `RHO_POWF_GUARD`, `BRENT_TOLERANCE`
  - `hfb_gpu.rs`: `DENSITY_FLOOR`, `GPU_JACOBI_CONVERGENCE`
  - `hfb_gpu_resident.rs`: `DENSITY_FLOOR`, `RHO_POWF_GUARD`, `GPU_JACOBI_CONVERGENCE`
  - `hfb_deformed.rs`: `DENSITY_FLOOR`, `DIVISION_GUARD`, `PAIRING_GAP_THRESHOLD`,
    `SCF_ENERGY_TOLERANCE`
  - `hfb_deformed_gpu.rs`: same deformed constants + `GPU_JACOBI_CONVERGENCE`
  - `md/observables.rs`: `DIVISION_GUARD` for RDF, VACF, energy drift
- **`load_eos_context()` evolved to `Result<>`** — callers now decide how to handle
  data loading failures instead of hard panicking in library code.
- **Removed unused import** `wgpu::util::DeviceExt` from `hfb_deformed_gpu.rs`.
- **Comprehensive audit completed:** zero unsafe, zero TODO/FIXME, zero mocks,
  zero hardcoded paths, all AGPL-3.0 licensed, all validation binaries follow
  hotSpring pattern (ValidationHarness, exit 0/1).

### Quality gate

- `cargo fmt` — clean
- `cargo clippy --all-targets -- -W clippy::pedantic` — clean (warnings only)
- `cargo test` — 189 passed, 5 ignored (GPU), 0 failed
- `cargo doc` — clean (0 warnings)

## [0.5.7] — 2026-02-17

### Added

- **7 new tests** (189 total): determinism tests for `chi2_per_datum`, `binding_energy_l2`,
  `SphericalHFB` construction, `compute_rdf`; CPU-path coverage for `DENSITY_FLOOR`
  and `SPIN_ORBIT_R_MIN` guards, GPU f64 buffer edge case.
- **`HFB_TEST_NUCLEI_PROVENANCE`** — machine-readable `BaselineProvenance` struct for
  HFB test nuclei (previously comment-only provenance).
- **Doc comments** on `GpuResidentL2Result` and all its fields.

### Changed

- **`hermite_value` → `barracuda::special::hermite`** — eliminated last duplicate math.
  `hfb_common::hermite_value` now delegates to the canonical barracuda implementation.
- **Validation binary tolerances fully wired:**
  - `validate_linalg`: `EXACT_F64`, `SVD_TOLERANCE`, `ITERATIVE_F64`
  - `validate_special_functions`: `GAMMA_TOLERANCE`, `ERF_TOLERANCE`, `BESSEL_TOLERANCE`,
    `LAGUERRE_TOLERANCE`, `CHI2_CDF_TOLERANCE`, `EXACT_F64`
  - `validate_md`: `EXACT_F64`, `GPU_VS_CPU_F64`, `MD_FORCE_TOLERANCE`,
    `NEWTON_3RD_LAW_ABS`, `MD_EQUILIBRIUM_FORCE_ABS`
  - `validate_barracuda_pipeline`: `GPU_VS_CPU_F64`, `NEWTON_3RD_LAW_ABS`,
    `MD_ABSOLUTE_FLOOR`, `ENERGY_DRIFT_PCT`
  - `validate_optimizers`: `EXACT_F64`, `GPU_VS_CPU_F64`, `RK45_TOLERANCE`,
    `SOBOL_TOLERANCE`
- **Provenance environments** — L1/L2 `BaselineProvenance` records now include
  NumPy 1.24, SciPy 1.11, mystic 0.4.2 (previously just "Python 3.10, mystic").

### Metrics

| Metric | Before (v0.5.6) | After (v0.5.7) |
|--------|:---:|:---:|
| Unit tests | 182 | **189** (+7) |
| Line coverage | 39% | **44%** |
| Function coverage | 57% | **61%** |
| Inline tolerance magic numbers (validation bins) | ~50 | **~12** (remaining: factorial, digamma, beta, assoc Legendre) |
| Duplicate math functions | 1 (hermite) | **0** |

## [0.5.6] — 2026-02-17

### Added

- **`SpinOrbitGpu` wired into `hfb_gpu_resident.rs`.** Replaces manual CPU spin-orbit
  loop with barracuda's `ops::grid::SpinOrbitGpu` dispatch. Falls back to CPU on GPU
  failure. Eliminates custom l·s factor computation — now uses canonical
  `barracuda::ops::grid::compute_ls_factor`.
- **`compute_ls_factor` wired into `hfb.rs`.** Replaces manual
  `(j*(j+1) - l*(l+1) - 0.75)/2` calculation in both `build_hamiltonian` and
  `binding_energy_l2` with the barracuda canonical implementation.
- **Physics guard constants centralized.** Added `DENSITY_FLOOR` (1e-15 fm⁻³),
  `SPIN_ORBIT_R_MIN` (0.1 fm), `COULOMB_R_MIN` (1e-10 fm) to `tolerances.rs` with
  physical justification. Replaced 20+ inline magic numbers across 5 physics modules
  (`hfb.rs`, `hfb_gpu.rs`, `hfb_gpu_resident.rs`, `hfb_deformed.rs`, `hfb_deformed_gpu.rs`).
- **SPDX headers** on all 17 WGSL shaders that were missing them (now 30/30).

### Changed

- `panic!()` in GPU buffer map failure paths (`hfb_gpu_resident.rs`) converted to
  idiomatic `expect()` with descriptive messages.
- WGSL inline math duplicates (`abs_f64`, `cbrt_f64`) annotated with `TODO(evolution)`
  for future preamble injection from ToadStool canonical `math_f64.wgsl`.

### Fixed

- Clippy `if_not_else` warning in `SpinOrbitGpu` guard.

## [0.5.5] — 2026-02-16

### Added

- **Centralized tolerance constants.** Added `GPU_EIGENSOLVE_REL`, `GPU_EIGENVECTOR_ORTHO`,
  `BCS_PARTICLE_NUMBER_ABS`, `BCS_CHEMICAL_POTENTIAL_REL`, `PPPM_NEWTON_3RD_ABS`,
  `PPPM_MADELUNG_REL`, `HFB_RUST_VS_PYTHON_REL`, `HFB_RUST_VS_EXP_REL`,
  `MD_ABSOLUTE_FLOOR`, `NEWTON_3RD_LAW_ABS`, `MD_EQUILIBRIUM_FORCE_ABS` to
  `tolerances.rs` with physical justification for each.
- **Unit tests** for `hfb_gpu.rs` (5), `hfb_gpu_resident.rs` (5), `bcs_gpu.rs` (6),
  `hfb.rs` (6), `data.rs` (2). Total library tests: 158 → 182.
- **`verify_hfb`** added to `validate_all` meta-validator SUITES list.
- **`data::EosContext` and `data::load_eos_context()`** — shared EOS context loading
  for all nuclear EOS binaries, eliminating 9 duplicated path constructions.
- **`data::chi2_per_datum()`** — shared χ² computation using `tolerances::sigma_theo`.
- **SPDX license headers** added to 4 archive files.

### Changed

- **Extracted 8 inline WGSL shaders** from `celllist_diag.rs` to
  `src/bin/shaders/celllist_diag/*.wgsl` — file reduced from 1672 to 1124 lines.
- **Wired all validation binaries to `tolerances.rs`.** Replaced 30+ inline
  tolerance constants across `validate_md`, `validate_barracuda_hfb`,
  `validate_pppm`, and `verify_hfb` with named constants from `tolerances.rs`.
- **Replaced 19 inline `sigma_theo` expressions** across 7 nuclear EOS binaries
  with `tolerances::sigma_theo(b_exp)`.
- **Fixed `BFGS_TOLERANCE`** — corrected from 0.1 to 1e-4 with proper justification;
  wired `validate_optimizers` to use it.
- **Eliminated 25+ clippy warnings** via auto-fix (`cloned` → `copied`,
  format string interpolation, unnecessary borrows).
- **Nuclear EOS binaries** now use `data::load_eos_context()` instead of inline
  path construction.
- **Fixed `MD_FORCE_TOLERANCE` doc comment** — was incorrectly documenting 1e-10
  for a 0.01 (1%) GPU f32 vs CPU f64 tolerance.
- **Cleaned up dead_code.** Renamed 6 unused struct fields with `_` prefix,
  documented 3 GPU-reserved functions with evolution comments.
- **Improved `HFB_TEST_NUCLEI` provenance.** Added exact Python command, environment,
  and date to the doc comment.
- **Bumped to v0.5.5.**

## [0.5.4] — 2026-02-16

### Added

- **`validate_nuclear_eos` binary.** Formal `ValidationHarness` for the complete
  nuclear EOS pipeline (L1 SEMF, L2 HFB, NMP) — pure Rust replication of all
  Python control work. 6 phases: L1 binding energies vs AME2020, NMP for SLy4
  (within 2σ of literature), NMP physicality check for UNEDF0, L2 HFB vs
  Python `skyrme_hf.py` baselines (12% tolerance for method differences),
  L1 χ²/datum, cross-parametrization consistency. All 37 checks pass.
- **`validate_all` meta-binary.** Runs all 9 validation suites in sequence
  (5 CPU-only + 4 GPU). `--skip-gpu` flag for CI without GPU hardware.
  Exit code 0 only if all suites pass.
- **GPU eigensolve in `hfb_gpu_resident`.** Replaced CPU `eigh_f64` with
  `BatchedEighGpu::execute_single_dispatch` — all proton+neutron Hamiltonians
  across all groups eigendecomposed in ONE shader invocation per SCF iteration.
  Falls back to CPU `eigh_f64` if `global_max_ns > 32` or GPU fails.
  Spin-orbit corrections applied before GPU pack (CPU, full f64).

### Changed

- **Bumped to v0.5.4.**
- **`hfb_gpu_resident` architecture evolved.** SCF loop now: GPU H-build →
  CPU readback → CPU spin-orbit → GPU eigensolve → CPU BCS/density/energy.
  Two GPU trips per iteration (H-build + eigensolve), eliminating the CPU
  eigensolve bottleneck.

## [0.5.3] — 2026-02-16

### Added

- **Single-dispatch eigensolve.** `hfb_gpu.rs` now uses
  `BatchedEighGpu::execute_single_dispatch()` — ALL Jacobi rotations execute
  inside ONE shader invocation with workgroup barriers, eliminating CPU
  readback between rotations. Estimated 3-5× speedup for GPU HFB (dispatch
  overhead was dominant). Falls back to `execute_f64()` if n > 32.
- **Single-dispatch in deformed HFB.** `hfb_deformed_gpu.rs` uses
  `execute_single_dispatch` for block sizes ≤ 32, with automatic fallback
  to multi-dispatch then CPU for larger blocks.
- **Validation binary tests single-dispatch.** `validate_barracuda_hfb` now
  validates BOTH multi-dispatch and single-dispatch eigensolve paths against
  CPU reference (eigenvalue error + orthogonality). 4 new checks.
- **5 MD WGSL shaders extracted to files.** `yukawa_force_f64.wgsl`,
  `vv_kick_drift_f64.wgsl`, `yukawa_force_celllist_f64.wgsl`,
  `yukawa_force_celllist_v2_f64.wgsl`, `rdf_histogram_f64.wgsl` — moved
  from inline `const &str` in `md/shaders.rs` to `src/md/shaders/` directory
  with `include_str!`. Reduced `md/shaders.rs` from 759 → 299 lines (60%).

### Fixed

- **BCS pipeline cached.** `BcsBisectionGpu` now compiles the WGSL shader
  and creates the compute pipeline ONCE at construction, not on every
  `solve_bcs()` / `solve_bcs_with_degeneracy()` call.
- **LHS import path standardized.** `nuclear_eos_gpu.rs` changed from
  `barracuda::sample::lhs::latin_hypercube` to `barracuda::sample::latin_hypercube`
  (the public re-export), matching all other binaries.
- **sarkas_gpu report path.** Replaced relative `"../benchmarks/"` with
  `CARGO_MANIFEST_DIR`-based resolution.

### Changed

- **BCS bisection docs updated.** The `target` → `target_val` WGSL keyword
  fix has been absorbed by ToadStool (commit `0c477306`). Local shader
  retained for domain-specific `use_degeneracy` feature (2j+1 shell model).
- **math_f64 preamble docs updated.** Documented ToadStool's `(zero + literal)`
  precision evolution (Feb 16 2026) — `log_f64` improved from ~1e-3 to ~1e-15.
  Patched output inherits these improvements.
- **Bumped to v0.5.3.**

## [0.5.2] — 2026-02-16

### Fixed

- **Production unwrap/expect cleanup.** `md/simulation.rs`: `read_back_f64`
  now returns `Result` instead of panicking on GPU channel failures; energy
  history guard uses `if let` instead of `.expect()`. `hfb_gpu_resident.rs`:
  non-panicking channel send, improved `eigh_f64` error messages.
  `hfb_deformed_gpu.rs`: empty eigenvalue guard in `find_fermi_bcs`.
  `bench.rs`: mutex uses `unwrap_or_else(PoisonError::into_inner)`;
  `nvidia-smi` stdout uses `let Some(...) else { return }`.
  `observables.rs`: added invariant comments to `first()`/`last()` calls.
- **All Clippy auto-fixes applied.** Resolved remaining `manual_range_contains`,
  `redundant_closure`, `explicit_iter_loop`, `print_literal` warnings.

### Added

- **Full provenance wiring.** All 6 nuclear EOS binaries (`nuclear_eos_gpu`,
  `l1_ref`, `l2_ref`, `l2_gpu`, `l3_ref`, `l3_gpu`) now use
  `provenance::SLY4_PARAMS`, `provenance::NMP_TARGETS`, and
  `provenance::L1_PYTHON_CHI2` / `L2_PYTHON_CHI2` / `L2_PYTHON_TOTAL_CHI2`
  instead of inline duplicates. UNEDF0 in `l1_ref` is a different
  parametrization and retained with a documenting comment.
- **Library test SLy4 deduplication.** All 5 test modules (`semf`,
  `nuclear_matter`, `prescreen`, `data`, `hfb_deformed`) now use
  `provenance::SLY4_PARAMS` instead of local `const SLY4` definitions.
- **`provenance::print_nmp_analysis()`** — shared formatted NMP analysis
  table (observable, value, target, sigma, pull, PASS/FAIL). Replaces
  4 duplicate implementations across binaries.
- **`provenance::NMP_NAMES` and `NMP_UNITS`** — standard display arrays.
- **`ValidationHarness` in 3 more binaries**: `f64_builtin_test` (8 GPU
  op checks), `sarkas_gpu` (per-case MD checks), `celllist_diag`
  (force/energy/pair-count checks). Total with harness: **11 of 18 binaries**
  (remaining 7 are optimization explorers, not validation targets).
- **`nuclear_eos_l2_hetero` uses `provenance::print_nmp_analysis()`**.

- **SPDX license headers.** All 45 `.rs` files (excluding archive/) now have
  `// SPDX-License-Identifier: AGPL-3.0-only` as the first line.
- **22 new unit tests.** Added `#[cfg(test)]` sections to `error.rs` (5 tests),
  `gpu.rs` (6 tests), `hfb.rs` (5 tests), `simulation.rs` (9 tests).
  Total: **107 tests** (was 85). 4 GPU/slow tests marked `#[ignore]`.
- **Data provenance DOIs.** Added DOIs for AME2020 (Wang et al. 2021),
  SLy4 (Chabanat 1998), UNEDF0 (Kortelainen 2010), NMP targets (Bender 2003,
  Lattimer & Prakash 2016), and Sarkas MD (Silvestri 2022) to `provenance.rs`.
  New constant: `AME2020_DOI`.
- **Physics tolerance documentation.** Added inline comments explaining
  convergence thresholds in `hfb_deformed.rs`, `hfb_gpu_resident.rs`,
  `observables.rs`, and `prescreen.rs` (1e-6, 1e-10, 1e-15, 1e-30, etc.).
- **GPU helper methods centralized on `GpuF64`.** `upload_f64`, `read_back_f64`,
  `dispatch`, `create_bind_group`, `create_u32_buffer` moved from duplicate
  free functions in `simulation.rs` / `celllist_diag.rs` to `GpuF64` methods.
- **`EVOLUTION_READINESS.md`** — Rust module → WGSL shader → promotion tier
  mapping with blockers and BarraCuda primitive inventory.
- **Zero library `unwrap()` calls.** All 5 remaining `unwrap()` in library
  code replaced with safe indexing or `expect()` with invariant documentation.
- **`patch_math_f64_preamble` centralized.** Moved from duplicate local
  functions in `celllist_diag.rs` and `f64_builtin_test.rs` to
  `md::shaders::patch_math_f64_preamble()` in the library.
- **`load_eos_context()` extracted in `nuclear_eos_l1_ref`.** Eliminated
  repeated data loading across `main`, `run_multi_seed`, `run_pareto_sweep`.
- **53 new unit tests.** Added `#[cfg(test)]` to `bench.rs` (12 tests),
  `shaders.rs` (6 tests), `validation.rs` (+10), `provenance.rs` (+8),
  `prescreen.rs` (+9), `observables.rs` (+2), `data.rs` (+2).
  Total: **158 tests** (was 85 at v0.5.1). 5 ignored (GPU/slow).
- **Test coverage measured.** `cargo llvm-cov`: 33.2% line, 50.3% function.
  CPU-testable modules average >90%. GPU modules at 0% (require hardware).
- **`EVOLUTION_READINESS.md`** — Rust → WGSL → promotion tier mapping.
- **Shared HFB physics formulas.** Extracted `bcs_v2()`, `coulomb_exchange_slater()`,
  `coulomb_exchange_energy_density()`, `cm_correction()`, `skyrme_central_t0()` into
  `hfb_common.rs`. Wired into `hfb.rs`, `hfb_deformed.rs`, `hfb_deformed_gpu.rs` —
  eliminates duplicated Slater exchange formula (4 sites).
- **WGSL shaders extracted.** 3 inline shaders from `nuclear_eos_gpu.rs` moved to
  `src/physics/shaders/` as `.wgsl` files (`semf_batch_f64`, `semf_pure_gpu_f64`,
  `chi2_batch_f64`). Binary uses `include_str!`. Reduced binary from 1183→1045 lines.
- **`nuclear_eos_l2_hetero` refactored.** Extracted `perturb_params()` and
  `cascade_filter()` helpers, eliminating ~100 lines of duplicated loop logic.
- **`validate_md.rs` cleaned.** All 78 `.unwrap()` calls replaced with descriptive
  `.expect()` messages for GPU operations, tensor readback, and force computation.
- **7 new `hfb_common` tests**: BCS v², Coulomb exchange, CM correction, Skyrme t0.

### Changed

- **Binary file sizes reduced.** Extracted shared NMP analysis, chi2, and
  parameter constants reduced nuclear EOS binaries by ~15–50 lines each.
  `nuclear_eos_l2_ref`: 833 → 785 lines. `nuclear_eos_gpu`: 1182 → 1163.
  `celllist_diag.rs`: 1785 → 1672 (GPU helpers moved to library).
  `nuclear_eos_gpu.rs`: 1183 → 1045 (shaders moved to .wgsl files).

## [0.5.1] — 2026-02-16

### Fixed

- **Clippy pedantic: 1,067 → 0 warnings.** Moved lint configuration from
  `lib.rs` `#![allow]` to workspace-level `[workspace.lints.clippy]` in
  `Cargo.toml` so all targets (lib + 18 bins) share physics-justified allows.
  Ran `cargo clippy --fix` for auto-fixable patterns (324 uninlined format
  args, 55 cast_lossless, 11 explicit_iter_loop, etc.), then fixed remaining
  manually: removed 7 unused `async` markers, fixed 2 `match_same_arms`,
  prefixed 3 unused struct fields, added `clippy.toml` thresholds.
- **Rustdoc: 3 → 0 warnings.** Escaped brackets in physics doc comments
  (`[future]`, `[nr]`, `[10]`).
- **10 compiler warnings eliminated.** Removed unused imports (`PI`,
  `barracuda::numerical::trapz`), fixed unused `mut`, prefixed unused
  variables, moved `#[cfg(test)]`-only density sanity checks inside the
  test block.
- **README version mismatch.** Updated v0.4.0 → v0.5.0 to match Cargo.toml.
  Corrected inaccurate claim "All 18 binaries use ValidationHarness" and
  "Clean with pedantic/nursery" clippy claim.
- **HFB provenance documentation.** Added detailed doc comment to
  `HFB_TEST_NUCLEI` explaining the provenance mismatch between `verify_hfb.rs`
  inline values and `provenance::HFB_TEST_NUCLEI` — different L2 solver
  configurations produce different values within the 10% tolerance.
- **GPU `dispatch_and_read` now returns `Result`.** Replaced `expect()` panics
  in channel send/recv with proper `Result<Vec<f64>, HotSpringError>`.
  Validation binary callers use `.expect()` (correct pattern for exit-on-fail).

### Added

- **Centralized SLy4/UNEDF0 parameters** in `provenance` module
  (`SLY4_PARAMS`, `UNEDF0_PARAMS`, `PARAM_NAMES`). Canonical source for all
  binaries and tests.
- **`nmp_chi2()` and `nmp_chi2_from_props()`** in `provenance` module. Shared
  NMP χ² evaluation previously duplicated across 5+ binaries.
- **`validate_pppm` now uses `ValidationHarness`** with proper exit code 0/1.
  Added Newton 3rd law, energy sign, and net force checks.
- **`verify_hfb` now uses `provenance::HFB_TEST_NUCLEI` and `SLY4_PARAMS`.**
  No more inline duplicate constants.
- **4 new provenance tests**: `sly4_params_have_correct_length`,
  `nmp_chi2_sly4_is_small`, `nmp_chi2_exact_match_is_zero`,
  `hfb_test_nuclei_have_positive_energies`. Total: **85 tests** (was 81).

## [0.5.0] — 2026-02-16

### Added

- **`validate_barracuda_pipeline`** binary — end-to-end Yukawa OCP MD through
  BarraCuda's abstracted ops (YukawaForceF64, VelocityVerletKickDrift/HalfKick,
  BerendsenThermostat, KineticEnergy). 12/12 checks pass; 0.000% energy drift
  over 300 production steps; force magnitude error 1.86e-7.
- **`validate_barracuda_hfb`** binary — GPU BCS bisection + batched eigensolve
  vs CPU reference. 14/14 checks pass; BCS μ error 6.2e-11; eigenvalue error
  2.4e-12; O-16 proton BCS with nuclear degeneracy validated.
- **`bcs_gpu`** module — local GPU BCS bisection solver with corrected WGSL
  shader (fixes `target` reserved keyword in ToadStool's
  `batched_bisection_f64.wgsl`). Full Rust wrapper with f64 buffer management.
- **`bcs_bisection_f64.wgsl`** shader — hotSpring's local copy of the batched
  BCS bisection shader with the WGSL keyword fix applied.
- `HANDOFF_HOTSPRING_TO_TOADSTOOL_FEB_16_2026.md` — pipeline validation
  handoff documenting 26 new GPU op checks, device creation bug, and tier
  status updates.
- Grand total: **195/195 quantitative checks** across all phases + pipeline.

### Changed

- Bumped version to 0.5.0 (pipeline validation milestone).
- README: check count 186→195 (added Phase F 9 checks), binary count 16→18,
  RAM corrected 64→32 GB DDR5.
- CONTROL_EXPERIMENT_STATUS: added Phase F (9) + Pipeline (26) rows, total 195.
- whitePaper/: METHODOLOGY, STUDY, SUMMARY, README all updated to 195.

### Found (ToadStool bugs — documented for handoff)

- `batched_bisection_f64.wgsl` line 154: `let target = ...` — `target` is a
  WGSL reserved keyword. Fix: rename to `target_val`.
- `WgpuDevice::from_adapter_index()` line 333: `required_features: Features::empty()`
  — does not request SHADER_F64 even on f64-capable adapters. Fix: check
  adapter features and request SHADER_F64 when available.

## [0.4.0] — 2026-02-16

### Added

- **`error`** module — typed `HotSpringError` enum replacing `Result<_, String>`
  in `GpuF64::new()`, `run_simulation()`, and `run_simulation_celllist()`.
  Callers can now pattern-match on `NoAdapter`, `DeviceCreation`, `NoShaderF64`.
- **`hfb_common`** module — shared types/utilities for HFB solver family:
  `Mat` (row-major square matrix), `hermite_value`, `factorial_f64`. Eliminates
  duplication across spherical HFB (L2), deformed HFB (L3), and GPU HFB.
- 12 new tests (81 total, up from 70):
  - Bitwise determinism: `semf`, `nuclear_matter`, `prescreen`, `hfb_common`,
    `hfb_deformed` basis construction
  - `hfb_common` unit tests: `Mat`, Hermite polynomials, factorials
  - Integration: JSON round-trip consistency, discovery→load→SEMF pipeline

### Changed

- **GPU: capability-based discovery** — backend, power preference, and fallback
  adapter controlled via `HOTSPRING_WGPU_BACKEND`, `HOTSPRING_GPU_POWER`,
  `HOTSPRING_ALLOW_FALLBACK` environment variables. Buffer limits derived from
  `adapter.limits()` instead of hardcoded values.
- **NaN-safe sorting** — all 28 instances of `partial_cmp().unwrap()` across
  12 files replaced with `f64::total_cmp()` (stable since Rust 1.62).
- **Merged `compute_energy` / `compute_energy_debug`** in `hfb.rs` into a
  single function with `verbose: bool` flag (saved 106 lines).
- **Smart refactoring** of HFB solvers: `hfb.rs` 1176→1070 LOC,
  `hfb_deformed.rs` 1322→1257 LOC, `hfb_deformed_gpu.rs` 1211→1182 LOC.
- **Pure-Rust ISO 8601** — `bench.rs` `now_iso8601()` replaced external `date`
  command with `SystemTime` + Hinnant's `civil_from_days` algorithm.
- Hardened all GPU readback / channel `.unwrap()` calls with descriptive
  `.expect()` messages in `gpu.rs`, `md/simulation.rs`, `hfb_gpu_resident.rs`.
- Added `#[derive(Debug)]` to `GpuF64`, `MdSimulation`, `CylindricalGrid`,
  `PowerMonitor`; `#[derive(Debug, Clone)]` to `Mat`.
- Documented inline shader reductions as intentional performance choices
  (fused force+accumulation avoids separate reduce dispatches).

### Fixed

- `clippy::pedantic` warning in `bench.rs`: `yoe as i64` → `i64::from(yoe)`.

## [0.3.0] — 2026-02-16

### Added

- **`provenance`** module — traces every hardcoded validation value to its
  Python origin (script, commit `fd908c41`, date, exact command).
- **`tolerances`** module — centralizes all validation thresholds with
  physical justification (machine precision, numerical method, model,
  literature). No more ad-hoc magic numbers.
- **`validation`** module — harness for pass/fail binaries with structured
  check tracking and exit code 0 (pass) / 1 (fail).
- **`discovery`** module — capability-based data path resolution via
  `HOTSPRING_DATA_ROOT` env var, `CARGO_MANIFEST_DIR`, or CWD discovery.
  Replaces scattered hardcoded `PathBuf` construction.
- 62 new unit tests across all library modules:
  - `constants`: CODATA 2018 values, derived quantities
  - `semf`: known nuclei, pairing, monotonicity
  - `nuclear_matter`: SLy4 NMP (ρ₀, E/A, K∞, m*/m, J), saturation minimum
  - `prescreen`: NMP cascade, classifier, constraint bounds
  - `data`: param names, nuclei set parsing, file loading
  - `config`: box side, DSF cases, paper parity, reduced units
  - `cpu_reference`: FCC lattice, velocity initialization, Yukawa force
  - `observables`: energy validation, RDF, VACF

### Changed

- All WGSL shader string constants use `r"..."` instead of `r#"..."#`
  (clippy `needless_raw_string_hashes`).
- Heavy GPU test `test_deformed_hfb_runs` marked `#[ignore]` with
  instructions for manual invocation.
- Crate-level documentation expanded with evolution path and architecture.

### Fixed

- `cargo fmt` applied to all 34 source files.
- Pre-existing compile warnings addressed (unused variables, dead code).

## [0.2.0] — 2026-02-12

### Added

- Phase E: Paper-parity Yukawa OCP — 9/9 PP cases at N=10,000, 80k steps.
- Phase D: N-scaling with native f64 builtins and cell-list forces.
- L3 deformed HFB (CPU and GPU paths).
- GPU-resident HFB prototype (Experiment 005b).
- Cell-list `i32 %` bug fix (branch-based wrapping).
- 160/160 quantitative checks passing.

### Baselines

All Python baselines from:
- **Sarkas** v1.0.0 (commit `fd908c41`), environment `envs/sarkas.yaml`
- **Surrogate** from Zenodo DOI 10.5281/zenodo.10908462, environment `envs/surrogate.yaml`
- **TTM** from Two-Temperature-Model repo, environment `envs/ttm.yaml`
- **AME2020** from Wang et al. 2021 (IAEA/AMDC)

### Tolerance changes

None — all thresholds stable since Phase B.

## [0.1.0] — 2026-01-15

### Added

- Initial L1/L2 nuclear EOS validation.
- BarraCuda L1: χ²=2.27, 478× faster than Python.
- BarraCuda L2: χ²=16.11, 1.7× faster than Python.
- Sarkas MD Phase A (60/60 checks).
- Surrogate learning (15/15 checks).
- TTM validation (6/6 checks).
