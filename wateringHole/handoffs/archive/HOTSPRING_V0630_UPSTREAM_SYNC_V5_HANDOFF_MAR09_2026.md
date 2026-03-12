# hotSpring v0.6.30 — Upstream Sync v5 Handoff

**Date**: 2026-03-09
**From**: hotSpring
**To**: barraCuda, coralReef, toadStool teams
**Scope**: barraCuda d761c5d + coralReef Iter 35 + toadStool S146 sync

---

## Summary

hotSpring has pulled and absorbed all recent upstream evolution from barraCuda
(2 commits since `875e116`), coralReef (3 commits since `b783217`, Iter 33→35),
and toadStool (stable at S146 with hw-learn). Key rewiring:

1. **ReduceScalarPipeline f64 zeros fix** — the all-zeros bug discovered by
   hotSpring Exp 055 is now fixed upstream. barraCuda always uses DF64 (f32-pair)
   accumulation in `var<workgroup>` memory. No hotSpring code changes needed —
   the fix is transparent.

2. **Df64SpirVPoisoning rename** — `has_nvvm_df64_poisoning_risk()` →
   `has_df64_spir_v_poisoning()`. Correctly reflects that the root cause is
   naga WGSL→SPIR-V codegen, not NVVM-specific. Detection now applies to ALL
   Vulkan backends unconditionally. Updated in 3 lib modules + 1 diagnostic binary.

3. **BatchedComputeDispatch wired** — sovereign engine's inner MD step
   (kick_drift + force + half_kick) now uses `BatchedComputeDispatch<B>` for
   single-submission batching. Reduces per-dispatch host overhead from ~1.6ms
   to ~0.1ms amortized on Vulkan. Dead sequential helpers removed.

4. **coralReef Iter 35 absorption acknowledged** — FirmwareInventory struct
   (absorbed from toadStool hwLearn handoff), 5 deformed HFB shaders absorbed
   from hotSpring (cross-spring evolution), drm_ioctl_named migration, 1616 tests.

---

## Cross-Spring Evolution Notes

### Shaders flowing hotSpring → coralReef (Iter 34)

coralReef absorbed **5 deformed HFB shaders** from hotSpring into
`crates/coral-reef/tests/deformed_hfb_absorption.rs` (493 lines, 9 passing, 1 ignored):
- Deformed WS potential
- Deformed Coulomb potential
- Deformed spin-orbit
- Deformed density
- Deformed BCS gap

These originated in `hotSpring/barracuda/src/physics/hfb_deformed_gpu/` and
were handed off via `wateringHole/handoffs/` for sovereign compilation validation.
coralReef now validates them on SM70/SM86/RDNA2 targets.

### Precision patterns flowing hotSpring → toadStool → coralReef

The `FirmwareInventory` pattern demonstrates the full cross-spring cycle:
1. hotSpring Exp 057 discovered PMU firmware blocking on Volta
2. toadStool hw-learn crate formalized `FirmwareInventory` with `FwStatus` enum
3. coralReef Iter 35 absorbed the pattern into `coral-driver/src/nv/identity.rs`
   with `compute_viable()` and `compute_blockers()` — parallel implementation
   optimized for the compiler's DRM context

### Virtual GSP Progress

coralReef's `FirmwareInventory` now provides:
- `compute_viable()` — combines GR + (PMU || GSP) presence
- `compute_blockers()` — human-readable missing-component diagnostics
- Compile-time UAPI struct size assertions (5 guards prevent ABI drift)

toadStool's `FirmwareInventory` provides:
- Runtime probing via sysfs + PCI device ID → chip name mapping
- `FwStatus` enum (Present/Missing/NotRequired)
- Per-vendor probing (NVIDIA: PMU/GSP/ACR/GR/SEC2, Intel: GuC/HuC, AMD: not-required)

Together these form the firmware awareness layer needed for virtual GSP:
observe what firmware does → distill into recipes → apply without firmware.

---

## barraCuda Team Guidance

### Immediate

- **BatchedComputeDispatch validated** by hotSpring in sovereign engine. Works
  correctly through the `GpuBackend` trait. The wgpu override produces a
  single command encoder for all dispatches in the batch.

- **Df64SpirVPoisoning now universal** — the workaround applies to all Vulkan
  backends, which means `has_df64_spir_v_poisoning()` always returns true.
  Consequence: DF64 force shaders containing transcendentals are never used
  through the wgpu/Vulkan tier. They only produce correct results through
  sovereign dispatch (coralReef). This is correct behavior.

### Medium-term

- **naga SPIR-V codegen bug**: The root cause of DF64 transcendental zeros is
  in naga's SPIR-V emission, not driver JIT. Filing upstream is recommended.
  Until fixed, the DF64 force path is sovereign-only.

- **ReduceScalarPipeline DF64 path**: The fix uses DF64 (~48-bit precision)
  for workgroup reductions. For sum/max/min this is acceptable. For dot products
  or high-precision accumulation, consider GPU-resident CG's local multi-pass
  reduce pattern (hotSpring `resident_cg_buffers.rs`) which avoids workgroup
  shared memory entirely.

---

## coralReef Team Guidance

### Immediate

- **5 compile-time UAPI ABI guards** in Iter 35 are exactly right. These
  prevent the class of bugs hotSpring Exp 057 found (struct size mismatches
  against kernel drm_nouveau.h).

- **FirmwareInventory** pattern is solid. The `compute_blockers()` API is
  particularly useful for diagnostic tooling.

### For virtual GSP

- The hw-learn crate in toadStool provides the observe→distill→apply pipeline
  for learning GPU init patterns from firmware traces. coralReef's
  `FirmwareInventory` tells you WHAT firmware is missing; hw-learn tells you
  HOW to replicate what that firmware does.

- Key registers to target for virtual GSP (from hw-learn AMD GFX10 baseline):
  1. GR engine enable (FECS/GPCCS context switch)
  2. PMU scheduler init (power state, clock gating)
  3. Memory controller setup (compute queue allocation)
  4. Channel credential verification (ACR/SEC2)

---

## toadStool Team Guidance

- hw-learn crate is stable and validated. No upstream changes needed.
- coralReef Iter 35 independently implemented `FirmwareInventory` — this is
  healthy convergent evolution. The two implementations serve different contexts
  (sysmon runtime vs driver compile-time) and don't need to be unified.
- Next step for hw-learn: capture GSP RPC traces on eastgate (RTX 4070) where
  GSP firmware is present and compute works. This provides the "teacher" data
  that can unlock Volta cards missing PMU firmware.

---

## Validation Results

| Check | Result |
|-------|--------|
| `cargo check` | Clean (0 errors, 0 warnings) |
| `cargo clippy --lib` | 0 warnings (pedantic) |
| `cargo test` | 848 tests pass (0 failures) |
| API rename | 4 files updated, all compile |
| BatchedComputeDispatch | Wired into sovereign_engine, compiles clean |

---

## Pin Summary

| Primal | Previous Pin | New Pin | Key Changes |
|--------|-------------|---------|-------------|
| barraCuda | `875e116` | `d761c5d` | ReduceScalarPipeline fix, Df64SpirVPoisoning rename, BatchedComputeDispatch, double-alloc cleanup |
| coralReef | `b783217` (Iter 33) | `1dfbaff` (Iter 35) | FirmwareInventory, drm_ioctl_named, 5 UAPI ABI guards, SM89 DF64, 5 HFB shaders absorbed |
| toadStool | S146 (`edcea15b`) | S146 (unchanged) | hw-learn stable |
