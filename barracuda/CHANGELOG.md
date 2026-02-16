# Changelog

All notable changes to the hotSpring BarraCUDA validation crate.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  mapping with blockers and BarraCUDA primitive inventory.
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
  BarraCUDA's abstracted ops (YukawaForceF64, VelocityVerletKickDrift/HalfKick,
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
- BarraCUDA L1: χ²=2.27, 478× faster than Python.
- BarraCUDA L2: χ²=16.11, 1.7× faster than Python.
- Sarkas MD Phase A (60/60 checks).
- Surrogate learning (15/15 checks).
- TTM validation (6/6 checks).
