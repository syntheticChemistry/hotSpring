# coralReef Dispatch Frontier — strandgate Handoff

> **HISTORICAL (2026-03-28):** This handoff document is superseded by the
> Exp 124 AMD scratch/local memory breakthrough (RDNA2 FLAT_STORE solved),
> the RTX 3090 GPFIFO sovereign dispatch progress (Exp 126-128), and the
> firmware-interface strategy for Titan V FECS (Exp 125/127). Retained for
> provenance. The feature flag is `sovereign-dispatch` (not `coral-sovereign`).

**Date:** 2026-03-28
**From:** strandgate (hotSpring) — AMD RDNA2 + RTX 3090 (SM86)
**To:** biomeGate (coralReef/coral-driver) — Titan V (SM70) + K80 (SM35)

## Executive Summary

strandgate validated the **entire sovereign compilation pipeline** — 24/24 QCD
production shaders compile to native GPU binaries through coral-reef (no wgpu, no
naga, no Vulkan). DRM device enumeration works on both AMD and NVIDIA. AMD PM4
nop dispatch completes with fence sync.

**All 4 remaining failures are localized to coral-driver's hardware-specific
initialization**, not the compiler, not pipeline wiring, not shader code.

## What strandgate Proved (LIVE)

| Layer | Status | Evidence |
|-------|--------|----------|
| GPU enumeration (DRM) | **LIVE** | AMD RDNA2 + NVIDIA SM86 both discovered via render nodes |
| WGSL → native compilation | **LIVE** | 24/24 QCD shaders → ~60KB native AMD GFX ISA in 102ms |
| coral-reef auto-prepend | **LIVE** | Complex64 + SU3 + PRNG preambles inject correctly |
| PM4 nop dispatch (AMD) | **LIVE** | Command buffer submits, fence syncs, no GPU hang |
| validate_coral_sovereign | **LIVE** | Full validation binary, `--features sovereign-dispatch` |

## The 4 Failures — Precise Diagnosis

### Failure 1 & 2: AMD RDNA2 — FLAT_STORE writes zeros

**Symptom:** Shader dispatches complete (no hang), but `FLAT_STORE` writes are
not visible in readback. All buffer values remain 0.

**Root cause:** `SH_MEM_CONFIG` and `SH_MEM_BASES` registers are **never set**
in the PM4 command stream. These registers configure the flat/global address
space for compute shaders on GCN/RDNA.

**Location:** `coral-driver/src/amd/pm4.rs`, function `build_compute_dispatch()`

**What is correct:** `COMPUTE_USER_DATA` correctly loads buffer VAs into user
SGPRs. `COMPUTE_PGM_RSRC1/2`, `DISPATCH_DIRECT`, `ACQUIRE_MEM` are all present.

**What is missing:** No `SET_SH_REG` packets for:
- `SH_MEM_CONFIG` (register 0x2210 for GFX10.3) — flat address type, alignment
- `SH_MEM_BASES` (register 0x2214 for GFX10.3) — private/shared aperture bases

**Fix complexity:** LOCALIZED. ~10-15 lines of PM4 packets added before
`DISPATCH_DIRECT`. Reference values available from:
- Mesa radeonsi: `si_emit_compute_shader_pointers()` in `si_compute.c`
- RADV: `radv_emit_compute_shader()` — look for `SH_MEM_CONFIG` writes

**Verification:** Run `handcrafted_store_42_shader` and
`hardcoded_va_store_42_shader` in `hw_amd_e2e.rs` after the fix.

**Secondary concern:** `COMPUTE_PGM_RSRC1` hardcodes `sgpr_count = 16`. For
simple shaders this works, but production QCD shaders with 74+ GPRs may need
dynamic SGPR allocation. Not blocking for the initial store-42 fix.

### Failure 3 & 4: NVIDIA RTX 3090 — GPFIFO channel alloc INVALID_ADDRESS

**Symptom:** `NvUvmComputeDevice::open()` fails during initialization. All
subsequent compute operations return "UVM compute backend not available".

**Root cause:** `RM_ALLOC(AMPERE_CHANNEL_GPFIFO_A)` returns status `0x0000001E`
(`NV_ERR_INVALID_ADDRESS`). The RM rejects the GPFIFO GPU VA or USERD handle.

**Location:** `coral-driver/src/nv/uvm_compute.rs`, line 213-219:
```rust
let h_channel = client.alloc_gpfifo_channel(
    h_changrp,
    h_ctrl_mem,       // ← USERD handle (combined USERD+GPFIFO alloc)
    gpfifo_gpu_va,    // ← from rm_map_memory_dma
    GPFIFO_ENTRIES,
    gpu_gen.channel_class(),  // ← AMPERE_CHANNEL_GPFIFO_A
)?;
```

**Environment:** strandgate has ALL required device nodes and kernel modules:
- `/dev/nvidia0` ✓ (crw-rw-rw-)
- `/dev/nvidiactl` ✓ (crw-rw-rw-)
- `/dev/nvidia-uvm` ✓ (crw-rw-rw-)
- `nvidia`, `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset` all loaded

**Hypothesis:** The Ampere channel class (`AMPERE_CHANNEL_GPFIFO_A`) may require:
1. **Separate USERD allocation** — Volta may accept combined USERD+GPFIFO memory,
   but Ampere may need USERD as its own RM allocation (not a sub-region)
2. **USERD VA mapping** — the current code passes `h_ctrl_mem` as the USERD
   handle but never DMA-maps it into the GPU VA space for Ampere's use
3. **GPFIFO alignment** — Ampere may require 4K or 64K alignment for the GPFIFO
   VA, which `rm_map_memory_dma` may not guarantee at the requested offset
4. **Different `NvChannelAllocParams` fields** — Ampere may need `userd_offset[0]`
   or additional fields set (currently zeroed via `..Default::default()`)

**Fix complexity:** MODERATE. Requires understanding the AMPERE_CHANNEL_GPFIFO_A
allocation contract. Reference sources:
- nvidia-open-gpu-kernel-modules: `NV_CHANNEL_ALLOC_PARAMS` struct usage
- Open-source CUDA driver (nvc0): `nouveau/nvc0/nvc0_compute.c` channel setup
- biomeGate's own Volta path: compare how `VOLTA_CHANNEL_GPFIFO_A` parameters
  differ from what Ampere needs

**Alternative paths if RM stays broken:**
- **`nouveau` backend** (`NvDevice`) — DRM render node + nouveau ioctls (no RM)
- **`cuda` backend** (`CudaComputeDevice`) — via cudarc, if CUDA toolkit installed
- **`vfio` backend** (`NvVfioComputeDevice`) — direct BAR0, requires VFIO binding

## How This Maps to biomeGate's Work

biomeGate is cracking the **Volta RM chain** (Titan V) and the **Kepler PIO
path** (K80). The strandgate failures are the SAME CLASS of work but for
DIFFERENT GPU generations:

| GPU cracking layer | biomeGate (Volta/Kepler) | strandgate (Ampere/RDNA2) |
|-------------------|--------------------------|---------------------------|
| DRM enumeration | ✓ (VFIO path) | ✓ (render nodes) |
| RM object chain | Layers 1-8 solved, L10 blocked (WPR2) | GPFIFO alloc fails (0x1E) |
| PM4 dispatch | N/A (NVIDIA) | nop works, stores fail (SH_MEM_CONFIG) |
| Shader compilation | N/A (hotSpring does this) | **24/24 shaders LIVE** |

**Key insight:** If biomeGate's K80 validates the full compute pipeline
(PFIFO → PBDMA → GR → FECS → GPCCS → shader dispatch) WITHOUT security
layers, the successful patterns directly inform the Ampere GPFIFO fix.

## Actionable Items for coralReef

### Priority 1: AMD SH_MEM_CONFIG (estimated: 1 session)

In `coral-driver/src/amd/pm4.rs`, add to `build_compute_dispatch()`:

```rust
// Before DISPATCH_DIRECT, after COMPUTE_PGM setup:
// GFX10.3 (RDNA2): SH_MEM_CONFIG = 0x2210, SH_MEM_BASES = 0x2214
// Values from Mesa radeonsi for GFX10 compute:
//   SH_MEM_CONFIG: flat address type = 0 (INST), retry disable = 0
//   SH_MEM_BASES: private_base = 0, shared_base = 0
emit_set_sh_reg(&mut pm4, 0x2210, &[0x00000000]); // SH_MEM_CONFIG
emit_set_sh_reg(&mut pm4, 0x2214, &[0x00000000]); // SH_MEM_BASES
```

Exact values should be validated against Mesa/RADV for GFX10.3. The zeros above
are a starting point — Mesa may set specific bits for flat addressing mode.

### Priority 2: NVIDIA Ampere GPFIFO (estimated: 2-3 sessions)

Debug `alloc_gpfifo_channel` for `AMPERE_CHANNEL_GPFIFO_A`:

1. Check if USERD needs its own `alloc_system_memory` + DMA mapping separate
   from the combined ctrl_mem allocation
2. Check if `userd_offset[0]` needs to be set in `NvChannelAllocParams`
3. Try setting `h_userd_memory[0]` to a dedicated USERD handle (not ctrl_mem)
4. Compare parameter layout with nvidia-open-gpu-kernel-modules for Ampere

### Priority 3: Validate with production shaders

Once dispatch works on either platform:
```bash
cd hotSpring/barracuda
cargo run --release --features sovereign-dispatch --bin validate_coral_sovereign
```

All 24 shaders are pre-wired. Compilation is instant (102ms). Only dispatch
needs the driver fix.

## What strandgate Fixed for coralReef (upstream-worthy)

1. **coral-reef auto-prepend guard discovery:** When composing shaders with
   barraCuda's `vec2<f64>` complex library, coral-reef's auto-prepend of its own
   `Complex64` struct library causes redefinition errors. Guard: include
   `"struct Complex64"` in the source (even as a comment) to suppress auto-prepend.

2. **Composite shader library chain:** The 5 composite QCD shaders need:
   - `complex_f64.wgsl` (vec2 version from barraCuda)
   - `su3.wgsl` (vec2 version from barraCuda)
   - `lcg_f64.wgsl` (PRNG — for hmc_leapfrog only)
   - `su3_extended_f64.wgsl` (exp_cayley, reunitarize — for hmc_leapfrog only)

3. **Two complex representations exist:**
   - barraCuda: `vec2<f64>` where `.x`=real, `.y`=imag
   - coral-reef/hotSpring: `struct Complex64 { re: f64, im: f64 }`
   
   These should be unified. The `vec2<f64>` representation is more compact in
   storage buffers but the struct is more readable. coralReef's auto-prepend
   assumes the struct version.

## Files Created/Modified

| File | Role |
|------|------|
| `barracuda/src/bin/validate_coral_sovereign.rs` | Sovereign pipeline validation binary |
| `barracuda/Cargo.toml` | `sovereign-dispatch` feature flag + `coral-gpu` dep |
