# Experiment 055: DF64 Naga SPIR-V Poisoning Diagnostic

**Date**: 2026-03-11  
**hotSpring**: v0.6.29  
**barraCuda**: v0.3.5 (8d63c77)  
**Binary**: `test_df64_nvk_poison`

## Objective

Determine whether the DF64 transcendental poisoning (exp_df64/sqrt_df64 producing
all-zero forces) is driver-specific (NVVM) or a naga SPIR-V codegen issue.

## Hypothesis

If NVK (Mesa's NAK compiler) produces correct DF64 forces, the bug is in NVVM.
If NVK also produces zeros, the bug is in naga's WGSL→SPIR-V path.

## Method

Run `test_df64_nvk_poison` on three backends:
1. RTX 3090 (Ampere) — NVIDIA proprietary 580.119.02 (NVVM)
2. Titan V (Volta) — NVK / Mesa 25.1.5 (NAK)
3. llvmpipe — LLVM 15.0.7 (CPU reference)

Each test dispatches the same 500-particle Yukawa force kernel twice:
- Test 1: `SHADER_YUKAWA_FORCE` (native f64) — control
- Test 2: `SHADER_YUKAWA_FORCE_DF64` (DF64 transcendentals) — experiment

Compare non-zero force components, max |force|, and sum |force|.

## Results

### RTX 3090 — NVIDIA Proprietary (NVVM)

```
  has_nvvm_df64_poisoning_risk(): true
  fp64_strategy(): Hybrid

  Test 1 (f64):  PASS — 1500/1500 non-zero, max 2.656e-16
  Test 2 (DF64): FAIL — 0/1500 non-zero, all zero
```

### Titan V — NVK (NAK)

```
  has_nvvm_df64_poisoning_risk(): false
  fp64_strategy(): Native

  Test 1 (f64):  PASS — 1500/1500 non-zero, max 2.584e-16
  Test 2 (DF64): FAIL — 0/1500 non-zero, all zero
```

### llvmpipe — LLVM 15 (CPU)

```
  has_nvvm_df64_poisoning_risk(): false
  fp64_strategy(): Hybrid

  Test 1 (f64):  FAIL — 0/1500 non-zero (LLVM 15 f64 baseline broken)
  Test 2 (DF64): FAIL — 0/1500 non-zero
```

## Analysis

| Backend | f64 | DF64 | DF64 transcendentals included? |
|---------|-----|------|-------------------------------|
| Proprietary (NVVM) | PASS | FAIL | NO (stripped by poisoning guard) |
| NVK (NAK) | PASS | FAIL | YES (guard is false) |
| llvmpipe | FAIL | FAIL | YES (guard is false) |

**Critical finding**: NVK includes the full DF64 transcendentals (exp_df64, sqrt_df64)
in the compiled shader. NAK compiles it without error. But the dispatch still produces
all-zero forces.

Since two completely independent compiler backends (NVVM and NAK) both produce zeros
from the same naga-generated SPIR-V, the root cause is in **naga's WGSL → SPIR-V
codegen**, not in any driver JIT compiler.

## Root Cause: naga SPIR-V Codegen

The DF64 Yukawa shader mixes:
- `array<f64>` storage (positions, forces, params)
- `Df64` struct (`{ hi: f32, lo: f32 }`) computation
- `df64_from_f64()` / `df64_to_f64()` type conversions

This mixed f64/f32 pattern likely triggers a naga codegen bug where:
- `f64` → `(f32, f32)` Dekker split produces incorrect SPIR-V
- Or the `Df64` struct layout is misaligned in SPIR-V
- Or f32 builtin calls (`round`, `ldexp`, `exp2`) are mistyped in the f64 context

## Impact

- DF64 would give ~32× throughput on Ampere f32 cores (1:1 vs 1:32 f64)
- Current native f64 fallback causes 10-12× gap vs Kokkos-CUDA
- DF64 recovery would narrow gap to ~3-4×

## Upstream Response: coralReef Iteration 33

coralReef Iter 33 (865de7a) validated the sovereign fix with 6/6 tests passing:

- Compiled our **exact Yukawa DF64 shader** through sovereign pipeline (WGSL → naga → IR → SASS)
- SM70 (Titan V), SM86 (RTX 3090), RDNA2 (Radeon) all PASS
- Compiled SASS uses `FADD/FMUL/FFMA/MUFU.EX2` for DF64, `DADD/DMUL/DFMA` for f64 PBC
- No NVVM or naga SPIR-V involved at any point
- Expected throughput recovery: 4-8x (gap narrowing from 12x to 1.5-3x)

See: `nvvm_poisoning_validation.rs` in coralReef Iter 33

## Handoff

Written to `ecoPrimals/wateringHole/handoffs/HOTSPRING_DF64_NAGA_POISONING_DIAGNOSTIC_HANDOFF_MAR11_2026.md`
with four resolution paths (sovereign dispatch, naga fix, preamble swap, pure Df64 storage).
Updated with coralReef Iter 33 validation results.

## Next Step

Integrate sovereign compilation into hotSpring MD pipeline. coralReef Iter 33 handoff
provides three integration options (toadStool delegation, direct library call, IPC).
Requires `create_pipeline_from_binary` on barraCuda's WgpuDevice.
