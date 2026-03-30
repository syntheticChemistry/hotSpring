# Experiment 017: Debt Reduction Audit (v0.6.14)

**Date:** February 25, 2026
**Version:** v0.6.14
**Crate:** hotspring-barracuda
**Status:** ✅ Complete

---

## Objective

Systematic audit and debt reduction of the hotspring-barracuda crate across
8 dimensions: completion status, code quality, validation fidelity, dependency
health, evolution readiness, test coverage, semantic guidelines, and hardcoding.

## Scope

All library code (`src/lib.rs` tree) and all 76 binary targets (`src/bin/`).
Archive code and docs explicitly excluded.

---

## Results

### Clippy: 0 Warnings (Library + Binaries)

Before: 28 library warnings + 57 binary warnings = 85 total.
After: 0 warnings across all targets.

| Category | Count | Fix Method |
|----------|:-----:|------------|
| `uninlined_format_args` | ~15 | `cargo clippy --fix` |
| `manual_div_ceil` / `manual_midpoint` | 4 | Auto-fix |
| `needless_pass_by_ref_mut` | 3 | Manual: `&mut` → `&` for `ValidationHarness`, `EchoStateNetwork` |
| `collection_is_never_read` | 3 | Removed unused `Vec::new()` + `push` calls |
| `ref_option` | 1 | `&Option<GpuF64>` → `Option<&GpuF64>` in `run_titan_validation` |
| `needless_update` | 1 | Removed `..Default::default()` |
| `let_else` / `single_match_else` | 2 | Converted `match` to `let...else` |
| `used_underscore_binding` | 1 | Renamed `_chi_l_max` to `chi_l_max` and used it |
| Other (redundant_clone, map_unwrap_or, etc.) | ~55 | Auto-fix + manual |

### Cross-Primal Discovery

Replaced all hardcoded device paths and cross-primal references:

| Before | After |
|--------|-------|
| `Path::new("/dev/akida0").exists()` (3 binaries) | `discovery::probe_npu_available()` |
| `metalForge/nodes/biomegate.env` in docs | `HOTSPRING_GPU_PRIMARY=0` env vars |
| `toadstool/showcase` in error messages | Generic `cargo run --bin f64_builtin_test` |

New `probe_npu_available()` function: uses `NpuHardware::discover()` when
`npu-hw` feature is enabled, falls back to sysfs `/dev/akida*` scan otherwise.

### β_c Provenance

Added `KNOWN_BETA_C_SU3_NT4: f64 = 5.6925` to `provenance.rs` with citations:
- Bali et al. (2000): β_c = 5.6925(2) at N_t = 4
- Engels et al. (1990): β_c ≈ 5.69
- Creutz (1980): strong-to-weak coupling crossover

7 binaries migrated from local `KNOWN_BETA_C` constants to centralized import.

### WGSL PRNG Deduplication

Created `prng_pcg_f64.wgsl` shared library with 3 core functions:
- `pcg_hash(u32) → u32`
- `hash_u32(idx, seq) → u32`
- `uniform_f64(idx, seq) → f64`

Consumer shaders (`su3_random_momenta_f64.wgsl`, `gaussian_fermion_f64.wgsl`)
now include the shared library via Rust `LazyLock<String>` concatenation at
runtime. Box-Muller/gaussian implementations remain per-consumer to preserve
bit-exact validation (f32 vs f64 cos differences).

### Dependency Evolution

| Dependency | Status | Rationale |
|------------|--------|-----------|
| `barracuda` (shared) | Keep | Core compute library |
| `wgpu` 22 | Keep | GPU access, must match barracuda |
| `tokio` 1.35 | Keep | Async GPU operations |
| `serde` / `serde_json` | Keep | Data loading |
| `rayon` 1.8 | Keep | L2 HFB parallelism |
| `bytemuck` 1.14 | Keep | Zero-copy GPU uniform buffers |
| `akida-driver` | Keep (optional) | Pure Rust NPU access |
| `akida-models` | Monitor | Indirect usage, potential evolution |

Local `gauss_jordan_solve` (68 lines) replaced with `barracuda::ops::linalg::lu_solve`.

### gpu_hmc Module Refactor

Refactored `gpu_hmc.rs` (2,919 lines) into `gpu_hmc/` module:

| File | Lines | Responsibility |
|------|:-----:|----------------|
| `mod.rs` | 749 | Shared types, dispatch helpers, pure gauge trajectory |
| `dynamical.rs` | 456 | Dynamical fermion HMC |
| `streaming.rs` | 434 | Streaming variants, GPU PRNG, batched encoders |
| `resident_cg.rs` | 941 | GPU-resident CG solver |
| `observables.rs` | 208 | Stream observables, bidirectional NPU screening |

All files under 1,000 LOC guideline. Public API preserved.

### New GPU Validation Tolerances

4 new constants in `tolerances::lattice`:

| Constant | Value | Source |
|----------|:-----:|--------|
| `GPU_STREAMING_PLAQUETTE_PARITY` | 0.03 | Statistical on 20 trajectories |
| `GPU_FERMION_FORCE_PARITY` | 1e-14 | Analytical (∂S_F/∂U rounding) |
| `GPU_CG_ACTION_PARITY` | 1e-10 | CG convergence tolerance |
| `GPU_DYN_STREAMING_PLAQUETTE_PARITY` | 0.05 | Statistical + fermion variance |

### Discovery Test Coverage

5 new tests added to `discovery.rs`:
- `probe_npu_available_returns_bool`
- `nuclear_eos_dir_resolves`
- `capability_probes_have_unique_names`
- `capability_probes_paths_are_relative`
- `is_valid_root_rejects_nonexistent`

---

## Metrics Summary

| Metric | Before (v0.6.13) | After (v0.6.14) |
|--------|:-----------------:|:----------------:|
| Clippy warnings (lib) | 28 | **0** |
| Clippy warnings (bins) | 57 | **0** |
| Hardcoded device paths | 3 | **0** |
| Local KNOWN_BETA_C copies | 7 | **0** (centralized) |
| Duplicated PRNG functions | 6 (2 shaders × 3 funcs) | **0** (shared library) |
| Max file size (LOC) | 2,919 (gpu_hmc.rs) | **941** (resident_cg.rs) |
| Centralized tolerances | ~145 | **~150** |
| Library tests | 619 | **629** |
| Total tests | — | **664** |
| WGSL shaders | 24 | **25** |

---

## Cross-References

- **CHANGELOG**: `barracuda/CHANGELOG.md` v0.6.14 entry
- **Handoff**: `wateringHole/handoffs/HOTSPRING_V0613_TOADSTOOL_ABSORPTION_HANDOFF_FEB25_2026.md` Part 5
- **Previous**: `016_CROSS_SPRING_EVOLUTION_MAP.md` (v0.6.13 cross-spring shader mapping)
