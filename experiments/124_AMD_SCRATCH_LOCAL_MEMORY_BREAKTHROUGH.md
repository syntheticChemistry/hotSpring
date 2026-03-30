# Experiment 124: AMD Scratch/Local Memory Breakthrough

**Date:** 2026-03-30
**Hardware:** AMD Radeon RX 6950 XT (Navi 21, RDNA2, GFX10.3)
**Host:** strandgate
**Stack:** coralReef Iter 70c → coral-driver (AMD DRM) → coral-gpu parity harness

---

## Summary

Three-layer fix enables per-thread scratch (local) memory on RDNA2 via the
sovereign DRM pipeline. The `parity_hw_local_memory_f64` test — an `array<f64, 18>`
written via local `var` and summed back — passes with result = 6.0 (expected: 1+2+3).

This is the first time the sovereign stack has executed scratch-memory-dependent
shaders on AMD hardware without ROCm, HIP, or any vendor runtime.

---

## Problem Statement

QCD kernels (Wilson plaquette, gauge force) require per-thread temporary storage
for SU(3) matrix intermediates. WGSL `var<private>` arrays that exceed register
capacity are spilled to scratch memory. On AMD RDNA2, this requires:

1. The compiler to emit `FLAT_SCRATCH_LOAD/STORE` with `SEG=SCRATCH` (1)
2. The driver to allocate a GEM scratch buffer and configure PM4 registers
3. The GPU hardware to initialize `FLAT_SCRATCH_LO/HI` before shader execution

---

## Root Cause Chain

### Layer 1: Compiler — Wrong Memory Segment

**Before:** `OpLd`/`OpSt` with `MemSpace::Local` emitted `SEG=GLOBAL` (2) via
`encode_flat_load`/`encode_flat_store`, generating `FLAT_LOAD_DWORD` instructions
that dereferenced thread-private pointers as global addresses → GPU hang.

**Fix:** Added `encode_scratch_load` and `encode_scratch_store` methods to
`Rdna2Encoder` in `coral-reef/src/codegen/amd/encoding.rs`. These encode the
instruction word with `SEG=1` (SCRATCH). `OpLd`/`OpSt` for `MemSpace::Local`
now route to these methods.

### Layer 2: Driver — Scratch Buffer + PM4 Registers

**Before:** `COMPUTE_TMPRING_SIZE`, `COMPUTE_PGM_RSRC2.SCRATCH_EN`, and
`COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI` were not configured.

**Fix:** `AmdDevice::dispatch` in `coral-driver/src/amd/mod.rs`:
- Allocates a GEM buffer (GTT domain) sized as `per_wave_scratch * max_waves`
- Sets `COMPUTE_TMPRING_SIZE` = `(per_wave_scratch / 256) | (max_waves << 12)`
- Sets `COMPUTE_PGM_RSRC2` with `SCRATCH_EN` bit
- Sets `COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI` = scratch buffer GPU VA
- Includes scratch handle in BO list for `DRM_AMDGPU_CS`

### Layer 3: FLAT_SCRATCH Initialization — THE KEY DISCOVERY

**Root cause:** The amdgpu DRM Command Processor for compute IB submissions
does **NOT** auto-initialize `FLAT_SCRATCH_LO/HI` from `COMPUTE_DISPATCH_SCRATCH_BASE`.
This differs from the HSA/KFD path (used by ROCm/HIP), which initializes these
registers automatically. Matches open ROCm issue #6030.

**Fix:** Dynamic shader prolog patching. When `scratch_va != 0` and `gfx_major >= 10`,
a 24-byte (6-dword) prolog is prepended to the shader binary before upload:

```
S_MOV_B32     s11, scratch_va_lo        // SOP1: load low 32 bits
S_SETREG_B32  hwreg(FLAT_SCR_LO), s11   // SOPK: write HW_REG 20
S_MOV_B32     s11, scratch_va_hi        // SOP1: load high 32 bits
S_SETREG_B32  hwreg(FLAT_SCR_HI), s11   // SOPK: write HW_REG 21
```

Uses `s11` as temporary (above the 8 user SGPRs + 3 system SGPRs).
`S_MOV_B32` opcode 3 + `S_SETREG_B32` opcode 19, with
`hwreg(id, 0, 32)` = `id | (31 << 11)`.

An initial attempt using `S_SETREG_IMM32_B32` (opcode 20) failed — this
instruction appears unsupported or absent on GFX10.3.

---

## Validation

### parity_hw_local_memory_f64

```
test parity_hw_local_memory_f64 ... ok
  target: Amd(Rdna2)
  shader: array<f64, 18>, stores 1.0/2.0/3.0, loads and sums
  scratch_va: allocated, prolog patched (24 bytes)
  result: 6.0 (expected: 6.0)
```

### Full coral-gpu Hardware Parity Suite

```
test result: ok. 7 passed; 0 failed; 1 ignored
```

| Test | NVIDIA | AMD |
|------|--------|-----|
| parity_hw_simple_store | PASS | PASS |
| parity_hw_f64_arith | PASS | PASS |
| parity_hw_multi_workgroup | PASS | PASS |
| parity_hw_multi_buffer | PASS | PASS |
| parity_hw_global_load | PASS | PASS |
| parity_hw_local_memory_f64 | PASS | PASS |
| parity_hw_dual_card_wilson_plaquette | **NVIDIA: result=0** | **AMD: skip (EXEC mask)** |

### Unit Tests

- coral-driver: 358/358 pass
- coral-reef: 1314/1314 pass

---

## Remaining Frontiers

| Frontier | Card | Blocker |
|----------|------|---------|
| Wilson plaquette QCD | NVIDIA RTX 3090 | GR context allocation for LDL/STL (local memory requires SKEDCHECK05_LOCAL_MEMORY_TOTAL_SIZE) |
| Wilson plaquette QCD | AMD RX 6950 XT | EXEC masking for divergent wavefront control flow (loops/branches in SU(3) matrix multiply) |
| Euler HLL f64 | Both | SSARef LARGE_SIZE panic (SSA allocator limit, separate issue) |

---

## Impact

- **biomeGate GPU cracking team**: The FLAT_SCRATCH prolog pattern is architecture-generic
  for GFX10+ and likely applies to GFX9 (MI50/Vega) with different HW_REG IDs.
  Validates the entire scratch memory pipeline for the broader ROCm community.
- **Sovereign physics**: Unblocks all QCD kernels that require per-thread scratch
  (Wilson plaquette, gauge force, RHMC integrator) on AMD once EXEC masking is added.
- **Cross-vendor parity**: 6/7 dispatch tests now pass on both NVIDIA and AMD,
  demonstrating the sovereign stack's vendor-agnostic design.

---

*hotSpring v0.6.32 — Experiment 124 — 2026-03-30*
