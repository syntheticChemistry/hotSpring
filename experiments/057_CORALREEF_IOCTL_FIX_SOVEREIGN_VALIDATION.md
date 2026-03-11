# Experiment 057 — coralReef DRM Ioctl Struct Fix & Sovereign Validation

**Date:** 2026-03-11
**Author:** hotSpring team
**Depends on:** Exp 056 (Sovereign Dispatch Benchmark), coralReef Iter 33

---

## Objective

Diagnose and fix the persistent `DRM ioctl returned 22` (EINVAL) blocking
sovereign dispatch, then validate the full pipeline.

## Root Cause Analysis

### Bug 1: `NouveauVmInit` struct size mismatch (critical)

coralReef defined `NouveauVmInit` with **4 fields (32 bytes)** including
`unmanaged_addr` and `unmanaged_size`, but the kernel UAPI
(`drm_nouveau_vm_init`) has only **2 fields (16 bytes)**.

The DRM ioctl number encodes the struct size in its upper bits. A 32-byte
struct produces a different ioctl number than a 16-byte struct, so the
kernel rejects the call with EINVAL without even parsing the arguments.

**Fix:** Removed the two extra fields from `NouveauVmInit`.

### Bug 2: `NouveauExec` field order mismatch

coralReef had: `channel, push_count, wait_count, sig_count, push_ptr, wait_ptr, sig_ptr`
Kernel expects: `channel, push_count, wait_count, sig_count, wait_ptr, sig_ptr, push_ptr`

The last three u64 pointer fields were in the wrong order.

**Fix:** Reordered to match kernel `drm_nouveau_exec`.

### Bug 3: `NouveauVmBind` field order mismatch

coralReef had `op_ptr` between `flags` and `wait_count`.
Kernel expects `op_ptr` at the end: `op_count, flags, wait_count, sig_count, wait_ptr, sig_ptr, op_ptr`.

**Fix:** Reordered to match kernel `drm_nouveau_vm_bind`.

### Bug 4: `NouveauChannelAlloc` / `NouveauChannelFree` padding

coralReef added a `pad: u32` to both structs that the kernel doesn't have:
- `NouveauChannelAlloc`: 92 bytes (should be 88)
- `NouveauChannelFree`: 8 bytes (should be 4)

**Fix:** Removed the `pad` fields.

## Validation Results

### After fixes

| Step | Ioctl | Before Fix | After Fix |
|------|-------|-----------|-----------|
| VM_INIT | `DRM_NOUVEAU_VM_INIT` | EINVAL (size 32≠16) | **OK ✓** |
| CHANNEL_ALLOC | `DRM_NOUVEAU_CHANNEL_ALLOC` | EINVAL (size 92≠88) | EINVAL (PMU firmware) |
| GEM_NEW | untested | untested | blocked by CHANNEL_ALLOC |
| VM_BIND | untested | untested | blocked by CHANNEL_ALLOC |
| EXEC | untested | untested | blocked by CHANNEL_ALLOC |

### CHANNEL_ALLOC still fails: PMU firmware

After fixing all struct mismatches, VM_INIT succeeds but CHANNEL_ALLOC
returns EINVAL. Kernel log reveals:

```
nouveau 0000:4b:00.0: pmu: firmware unavailable
```

NVIDIA does not distribute signed PMU firmware blobs for desktop Volta (GV100).
Without PMU firmware, nouveau cannot initialize the GPU's compute/graphics
engine, so channel creation fails regardless of struct correctness.

**Hardware:** Titan V (GV100) on nouveau kernel driver 1.4, kernel 6.17.9.

### Firmware inventory

| Component | Status |
|-----------|--------|
| ACR (Application Context Runtime) | ✓ Present |
| GR (Graphics/Compute) | ✓ Present (FECS, GPCCS, ctx) |
| SEC2 (Security Engine v2) | ✓ Present |
| NVDEC (Video Decode) | ✓ Present |
| **PMU (Power Management Unit)** | **✗ MISSING** |

PMU firmware is distributed for Tegra SoCs (gp10b, gm20b) but NOT for
desktop GPUs (gv100, ga102, tu102, etc.).

### System hardware summary

| GPU | Driver | DRM card | Render node | Compute |
|-----|--------|----------|-------------|---------|
| RTX 3090 (GA102) | nvidia proprietary 580.119.02 | card1 | renderD129 | ✓ wgpu/Vulkan |
| Titan V (GV100) | nouveau 1.4 | card0 | renderD128 | ✗ missing PMU FW |

## Modern Rewiring Completed

Despite DRM dispatch being blocked on this hardware, we completed the
modern integration:

1. **`GenericMdBackend`** — new backend in `bench/md_backend.rs` that
   implements `MdBenchmarkBackend` using the sovereign engine
   (`run_simulation_generic<B>`) with automatic device selection:
   sovereign → wgpu fallback.

2. **Multi-driver benchmark** — `bench_sovereign_dispatch` now tries
   nouveau (SM70), nouveau (SM86), nvidia-drm (SM86), and amdgpu
   before giving up on sovereign.

3. **All 848 tests pass** with both `--features sovereign-dispatch`
   and without.

## wgpu Performance Baseline

| Metric | Value |
|--------|-------|
| Backend | NVIDIA GeForce RTX 3090 via wgpu/Vulkan |
| N | 2000 particles |
| Steps/s | 143.5 |
| Final KE | 14.3381 |
| Final PE | 230.2525 |
| Energy conservation | stable |

## Upstream Action Items

### For coralReef (P0)

- [ ] Accept the 4 struct fixes (VM_INIT size, EXEC order, VM_BIND order,
      CHANNEL_ALLOC/FREE padding)
- [ ] Add `static_assert` / compile-time size checks against kernel UAPI
      to prevent future struct drift
- [ ] Document PMU firmware requirement for desktop Volta compute
- [ ] Investigate whether GSP (GPU System Processor) firmware can
      substitute for PMU on Ampere+ (ga102 has gsp/ directory)

### For toadStool

- [ ] Hardware capability discovery should probe PMU firmware availability
      and report `compute_available: false` for GPUs without it

### Paths to unblock DRM dispatch

1. **AMD hardware** — amdgpu dispatch is E2E verified by coralReef
2. **NVIDIA GSP path** — Ampere+ GPUs have GSP firmware; nouveau may
   not need PMU when GSP is present
3. **PMU firmware extraction** — possible from proprietary driver but
   legally/technically complex
4. **nvidia-drm UVM integration** — coralReef Iter 31 started this;
   would bypass nouveau entirely for RTX 3090
