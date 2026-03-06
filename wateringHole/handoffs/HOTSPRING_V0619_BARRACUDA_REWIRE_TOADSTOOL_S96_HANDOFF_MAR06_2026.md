# hotSpring v0.6.19 → toadStool/barraCuda/coralReef — Rewire Handoff

**Date:** 2026-03-06
**From:** hotSpring v0.6.19 (685 lib tests, 0 clippy warnings)
**To:** toadStool (S96) / barraCuda (v0.3.3) / coralReef (Phase 5)
**License:** AGPL-3.0-only

## Executive Summary

Rewired hotSpring to delegate DF64 precision ownership entirely to barraCuda.
Local DF64 assembly code (text-based `downcast_f64_to_df64` + `strip_f64_from_df64_core`
+ manual `df64_core.wgsl`/`df64_transcendentals.wgsl` concatenation) replaced by
barraCuda's `compile_shader_universal(Precision::Df64)` which uses the two-layer
approach: naga-guided infix rewrite with text-based fallback.

**Math is universal. Precision is silicon. barraCuda owns it.**

## Changes

### 1. DF64 Compilation Delegation

| Before | After |
|--------|-------|
| `compile_full_df64_pipeline()` manually called `downcast_f64_to_df64()`, stripped `df64_core`, concatenated `DF64_PACK_UNPACK` + `DF64_TRANSCENDENTALS` | Delegates to `wgpu_device.compile_shader_universal(source, Precision::Df64, label)` |
| `create_pipeline_df64()` manually imported `WGSL_DF64_CORE` + `WGSL_DF64_TRANSCENDENTALS` from `ops::lattice::su3` and assembled them | Delegates to `wgpu_device.compile_shader_df64(source, label)` |
| Local `strip_f64_from_df64_core()` helper (26 lines) | Removed — barraCuda handles internally |

### 2. Stale Documentation Fixed

- `buffers.rs`: `Maintain::Wait` → `PollType::Wait` (wgpu 28)
- `gpu_hmc/mod.rs`: `downcast_f64_to_df64` references → `compile_shader_universal(Precision::Df64)`

### 3. What Was Already Aligned

| Item | Status |
|------|--------|
| wgpu 28 (`PollType::Wait`) | Already aligned |
| `device_clone()` / `queue_clone()` | Already using new API |
| `ShaderTemplate::for_driver_profile()` | Already using |
| `Fp64Strategy` dispatch | Already using |
| barraCuda path dependency | `../../barraCuda/crates/barracuda` — correct |

## Upstream Absorption Audit

### toadStool S88–S96 (reviewed)

| Evolution | hotSpring Impact |
|-----------|-----------------|
| `HardwareFingerprint` / `SubstrateCapabilityKind` | Future: use for hardware-aware dispatch |
| `GpuAdapterInfo::is_sovereign_capable()` | Future: sovereign pipeline routing |
| `SubstrateType` 4→8 variants | No direct match usage in hotSpring currently |
| `NpuDispatch` / `NpuParameterController` | hotSpring has its own NPU worker; evaluate merge |
| Spring absorption tracker | hotSpring v0.6.17 tracked; v0.6.19 now current |
| barracuda fossilized in toadStool | No impact — already using standalone |

### barraCuda v0.3.3 (reviewed)

| New API | Adoption Status |
|---------|----------------|
| `compile_shader_universal(Precision)` | **Adopted** — DF64 path now delegates |
| `compile_shader_df64()` | **Adopted** — hybrid DF64 path now delegates |
| `GpuView<T>` (zero-copy GPU-resident) | Ready — evaluate for MD VACF, KE/PE reductions |
| `mean_variance_buffer()` | Ready — zero-copy stats for physics |
| `AutocorrelationF64` | Ready — useful for transport validation |
| `CorrelationResult::r_squared()` | Ready — GPU correlation stats |
| `WarmupOp` pipeline warming | Ready — reduce cold-start latency |
| 13-tier tolerance architecture | Already aligned via `tolerances::` |
| Nuclear physics WGSL shaders absorbed | hotSpring's domain shaders now in barraCuda |

### coralReef (Phase 5, reviewed)

| Item | Status |
|------|--------|
| WGSL → naga → SASS compilation | Operational (390 tests) |
| f64 transcendentals (DFMA) | sqrt, rcp, exp2, log2, sin, cos |
| `CoralCompiler` IPC client in barraCuda | Already wired (fire-and-forget) |
| coralDriver (userspace dispatch) | **Blocker** — not yet available |
| hotSpring shader corpus | 81 WGSL shaders available as validation corpus |

## Codebase State

| Metric | Value |
|--------|-------|
| Lib tests | 685 (0 fail, 6 ignored) |
| Integration tests | 19 (0 fail) |
| Clippy (pedantic+nursery) | 0 warnings |
| Unsafe blocks | 0 |
| Max file size | <1000 lines |
| SPDX | 100% AGPL-3.0-only |

## Dependency Alignment

| Dependency | hotSpring | barraCuda | Notes |
|------------|-----------|-----------|-------|
| wgpu | 28 | 28 | Aligned |
| pollster | 0.3 | 0.3 | Aligned |
| bytemuck | 1.25 | 1.25 | Aligned |
| tokio | 1.50 | 1.50 | Aligned |
| Edition | 2021 | 2024 | Migration candidate |

## Next Evolution Targets

1. **`GpuView<T>` adoption** — eliminate per-call buffer upload/download for MD forces, VACF
2. **Buffer-resident fused stats** — `mean_variance_buffer()` for physics observables
3. **Edition 2024 migration** — align with barraCuda
4. **Sovereign pipeline** — when coralDriver lands, route f64-heavy shaders through coralReef
5. **`SubstrateCapabilityKind` dispatch** — use for capability-based hardware routing

---
*hotSpring v0.6.19 — AGPL-3.0-only*
