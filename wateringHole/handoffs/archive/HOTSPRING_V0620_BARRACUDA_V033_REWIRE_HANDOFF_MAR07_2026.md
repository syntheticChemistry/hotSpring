SPDX-License-Identifier: AGPL-3.0-only

# hotSpring v0.6.20 — barraCuda v0.3.3 Rewire Handoff

**Date:** March 7, 2026
**From:** hotSpring v0.6.20 (731 tests, 97 binaries, 84 WGSL shaders)
**To:** toadStool (S128+), barraCuda (v0.3.3), coralReef (Phase 9)

---

## Summary

hotSpring rewired local LSCFRK duplicate code to shared `barracuda::numerical::lscfrk`,
validated the Verlet pipeline against barraCuda HEAD, confirmed sovereign compiler
compatibility across all 47 `compile_shader_f64` call sites, and documented raw
shader compilation decisions in 4 benchmark binaries.

---

## What Changed

### 1. LSCFRK Rewired to barraCuda Imports

**Files:** `barracuda/src/lattice/gradient_flow.rs`, `barracuda/src/lattice/gpu_flow.rs`

Removed 278 lines of duplicate code:
- Local `derive_lscfrk3` → `barracuda::numerical::lscfrk::derive_lscfrk3`
- Local `FlowMeasurement` → `barracuda::numerical::lscfrk::FlowMeasurement`
- Local `Lscfrk` struct → `barracuda::numerical::lscfrk::LscfrkCoefficients`
- Local `LSCFRK3W6/W7/CK4` → `barracuda::numerical::lscfrk::LSCFRK3_W6/W7/LSCFRK4_CK`
- Local `find_t0`, `find_w0`, `compute_w_function` → re-exported from barraCuda
- Local `FlowCoeffs`, `W6`/`W7` const derivations in `gpu_flow.rs` → replaced with `LscfrkCoefficients`

All 14 gradient flow tests pass through imported path. `gradient_flow_production.rs`
binary works without import changes (re-exports transparent to downstream).

### 2. Raw Shader Compilation Audit

5 `create_shader_module` call sites:
- `gpu/mod.rs::build_pipeline` — private helper; receives already-optimized WGSL from
  `create_pipeline()` and `create_pipeline_f64_precise()`. Correctly positioned.
- `bench_fp64_ratio.rs` — micro-benchmarks raw ALU throughput (intentional)
- `validate_nak_eigensolve.rs` — NAK-specific shader validation (intentional)
- `f64_builtin_test.rs` — probes native f64 builtin compilation (intentional)
- `bench_wgsize_nvk.rs` — workgroup scheduling benchmark (intentional)

All 4 binaries now have documentation explaining the intentional bypass.

### 3. Verlet Pipeline Validated

- 136 MD tests pass (CellListGpu compatible with barraCuda HEAD)
- 16 celllist tests pass (CPU neighbor search API stable)
- GPU validation binaries compile against barraCuda v0.3.3
- `VerletListGpu` skin-based rebuild works with upstream `CellListGpu` 3-pass pipeline

### 4. Sovereign Compiler Validated

731 lib tests pass through barraCuda's sovereign compiler (FMA fusion + dead expr
elimination via naga IR). 47 production `create_pipeline_f64` call sites confirmed working.

---

## What hotSpring Now Uses from barraCuda

| API | Usage Count | Purpose |
|-----|:-----------:|---------|
| `compile_shader_f64()` | 47 | Sovereign compilation for f64 shaders |
| `compile_shader_df64()` | via `create_pipeline_df64` | DF64 shaders |
| `compile_shader_universal(Precision::Df64)` | via `compile_full_df64_pipeline` | Full DF64 precision tier |
| `GpuDriverProfile` / `Fp64Strategy` | All GPU modules | Hardware capability routing |
| `ShaderTemplate::for_driver_profile` | `create_pipeline`, `create_pipeline_f64_precise` | Driver-aware optimization |
| `CellListGpu` | MD neighbor search | GPU-resident cell list |
| `WORKGROUP_SIZE_COMPACT` | MD tolerances, transport | Shared workgroup constant |
| `barracuda::numerical::lscfrk::*` | Gradient flow | LSCFRK coefficients, scale utilities |

---

## What hotSpring Does NOT Yet Use (Future)

| API | Reason |
|-----|--------|
| `GpuView<T>` | Requires significant refactor of MD buffer management |
| `PrecisionRoutingAdvice` | Lives in toadStool, not barraCuda |
| `ReduceScalarPipeline::encode_reduce_to_buffer` | Buffer-resident CG (P2) |
| coralReef sovereign dispatch | Blocked on coralDriver availability |

---

## toadStool Pin Recommendation

**Current pin:** hotSpring v0.6.17
**Recommended:** hotSpring v0.6.20

New since v0.6.17:
- **v0.6.18**: Deep debt resolution — zero clippy warnings, pedantic compliance
- **v0.6.19**: Precision stability (Exp 046), Chuna paper extensions (completed Mermin,
  GPU BGK, DSF vs MD validation, production gradient flow), 731 tests, 84 WGSL shaders
- **v0.6.20**: LSCFRK rewired to shared barraCuda, sovereign compiler validated,
  Verlet pipeline confirmed, raw shader audit documented

### New Absorption Candidates for barraCuda

| Item | Files | Impact |
|------|-------|--------|
| Completed Mermin dielectric | `dielectric.rs`, `dielectric_mermin_f64.wgsl` | Momentum-conserving Mermin (Eq. 26) |
| GPU BGK relaxation | `gpu_kinetic_fluid.rs`, `bgk_relaxation_f64.wgsl` | Multi-species kinetic relaxation |
| DSF vs MD validation | `validate_dsf_vs_md.rs` | Analytical vs MD peak comparison |
| Production gradient flow | `gradient_flow_production.rs` | HMC thermalization + scale setting |
| Dielectric GPU pipeline | `gpu_dielectric.rs` | Batched ε(k,ω) with standard + completed modes |

---

## coralReef Readiness Assessment

coralReef (Phase 9) can compile WGSL → NVIDIA SASS and AMD GFX binary.
hotSpring's 84 WGSL shaders are a validation corpus.

**Blocked on:** coralDriver (DRM ioctl dispatch). When available, hotSpring should
route f64-heavy shaders (lattice QCD, dielectric, HFB) through coralReef for
native binary execution without Vulkan overhead.

**Priority shaders for coralReef validation:**
1. `su3_gauge_force_f64.wgsl` (HMC force — hottest shader)
2. `yukawa_force_verlet_f64.wgsl` (MD force — production workload)
3. `dielectric_mermin_f64.wgsl` (plasma physics — Chuna Paper 44)

---

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| Lib tests | 731 | 731 |
| WGSL shaders | 84 | 84 |
| Lines (gradient_flow.rs) | 740 | 491 |
| Lines (gpu_flow.rs) | 265 | 236 |
| Local LSCFRK code | 278 lines | 0 (imported from barraCuda) |
| `compile_shader_f64` call sites | 47 | 47 |
| Raw `create_shader_module` | 5 (undocumented) | 5 (documented) |

*AGPL-3.0-only*
