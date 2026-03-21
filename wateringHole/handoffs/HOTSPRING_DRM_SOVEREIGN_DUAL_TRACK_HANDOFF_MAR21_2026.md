# hotSpring → Compute Trio: DRM + Sovereign Dual-Track Handoff

**Date:** March 21, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** Exp 072 (DRM Dispatch Evolution Matrix), strategic context from Exp 055-071

---

## Executive Summary

- **Dual-track dispatch strategy**: DRM (kernel-mediated) and sovereign VFIO in parallel
- **AMD DRM dispatch is coded and ready to test** — `AmdDevice` has full `ComputeDevice`
  implementation with PM4 command submission, GEM buffers, and fence sync
- **NVIDIA DRM new UAPI is coded** (`VM_INIT`/`VM_BIND`/`EXEC`) but PMU firmware blocks
  `CHANNEL_ALLOC` on Titan V. K80 (Kepler, no PMU needed) incoming
- **Naga DF64 poisoning bypass**: WGSL → coral-reef → native ISA → DRM dispatch avoids
  the naga codegen bug entirely
- **GCN5 backend COMPLETE**: coral-reef compiles and dispatches GCN5 (GFX906) — 64/64 E2E verified on MI50. See updated Mar 21 GCN5 E2E handoff

---

## Part 1: The Dual-Track Strategy

Sovereign VFIO cracking (Exp 060-071) proved 6 of 10 pipeline layers. The remaining
blocker — MMU page table translation (`0xbad00200`) — is a focused engineering problem
but may take time. Meanwhile, coral-driver already has complete DRM dispatch paths:

| Aspect | Sovereign VFIO | DRM Dispatch |
|--------|----------------|--------------|
| MMU | Must build page tables manually | Kernel handles this |
| Firmware | Must load FECS/GPCCS manually | Kernel handles this |
| Vendor lock | None (any PCIe device) | Requires kernel driver |
| Hardware scope | Any GPU with MMIO access | GPUs with working DRM support |
| Current state | 6/10 layers, MMU blocked | Code complete, needs HW validation |

Both paths feed information back to each other. DRM dispatch gives us a working
reference for page table encoding, channel setup, and command format. Sovereign
cracking gives us vendor independence and deeper hardware understanding.

---

## Part 2: AMD DRM Dispatch (Fastest Path)

### What exists

coral-driver `AmdDevice` (`crates/coral-driver/src/amd/mod.rs`) implements:

```
AmdDevice::open()          → opens amdgpu render node, creates GPU context
AmdDevice::alloc()         → GEM buffer creation + GPU VA mapping
AmdDevice::upload/readback → mmap-based memory transfer
AmdDevice::dispatch()      → PM4 command buffer + DRM_AMDGPU_CS submission
AmdDevice::sync()          → DRM_AMDGPU_WAIT_CS fence synchronization
```

The PM4 builder (`amd/pm4.rs`) constructs full compute dispatch packets:
`SET_SH_REG` for shader program address + resources, `DISPATCH_DIRECT` for
grid launch, buffer VA user data, trailing NOP for alignment.

### What's needed

1. **GCN5 architecture in coral-reef** — add `Gcn5` variant to `AmdArch` in
   `gpu_arch.rs` with `gfx_major() = 9`, `has_native_f64() = true`,
   `f64_rate_divisor() = 4` (MI50 has 1/4 rate f64 — 3.5 TFLOPS, much faster
   than RDNA2's 1/16 rate)

2. **GCN5 instruction encoding** — core VOP3 f64 ops (`v_fma_f64`, `v_add_f64`,
   `v_mul_f64`) use identical encoding on GCN5 and RDNA2. SOPP
   (`s_endpgm`, `s_barrier`, `s_waitcnt`) is identical. Key difference: GCN5
   FLAT instructions have no `offset` field (bits [12:0] must be 0 on GFX9).
   Either parameterize `Rdna2Encoder` by GFX version or create a thin `Gcn5Encoder`.

3. **NOP dispatch test** — minimal binary: `AmdDevice::open()` → alloc →
   upload `s_endpgm` (0xBF810000) → `dispatch()` → `sync()`. This validates
   the entire PM4 + ioctl + fence pipeline independent of codegen.

### MI50 specifics

- PCI BDF: `0000:4d:00.0` (biomeGate)
- GFX version: GFX906 (Vega 20 / GCN 5th gen)
- Wave size: 64 (fixed, no wave32 mode)
- VGPRs: 256 per CU, SGPRs: 102
- f64 rate: 1/4 FP32 (3.5 TFLOPS) — **4× faster than RDNA2 for DF64**
- HBM2: 16 GB, 1 TB/s bandwidth
- Driver: `amdgpu` (kernel module), `RADV` (Vulkan userspace)
- GlowPlug: 1 round-trip/boot (Vega 20 SMU limitation)

---

## Part 3: NVIDIA DRM Dispatch

### What exists

coral-driver `NvDevice` (`crates/coral-driver/src/nv/mod.rs`) implements:

```
NvDevice::open()           → opens nouveau render node, VM_INIT, CHANNEL_ALLOC
  auto-detects new UAPI    → VM_INIT first (kernel 6.6+, required for Volta on 6.17+)
NvDevice::alloc()          → GEM new + VM_BIND map (new UAPI) or GEM new (legacy)
NvDevice::dispatch()       → QMD + push buffer → EXEC (new UAPI) or GEM_PUSHBUF (legacy)
NvDevice::sync()           → syncobj wait (new UAPI) or gem_cpu_prep (legacy)
```

The new UAPI structs are ABI-corrected (Exp 057): `NouveauVmInit` (16 bytes),
`NouveauExec` (correct field order), `NouveauVmBind` (correct field order),
`NouveauChannelAlloc` (no extra padding).

### Current blocker: PMU firmware (Titan V)

`CHANNEL_ALLOC` returns EINVAL because nouveau cannot initialize the GPU's
compute engine without PMU firmware. NVIDIA does not distribute PMU blobs for
desktop Volta (GV100). `VM_INIT` succeeds — the blocker is engine init, not
VA space setup.

### K80 path (incoming hardware)

Tesla K80 (Kepler / GK210 / SM35):
- **No PMU firmware needed** — Kepler uses legacy nouveau initialization
- **No `VM_INIT` needed** — legacy UAPI (`CHANNEL_ALLOC` → `GEM_PUSHBUF`)
- **No firmware signing** — full sovereign pipeline also works
- Expected cost: ~$30 on eBay
- This is the **bridge hardware** that validates both DRM and sovereign

### Titan V PMU workaround options

1. **FECS-only channel**: Exp 068 proved FECS can execute from host-loaded IMEM
   on a clean falcon. Can `CHANNEL_ALLOC` succeed if FECS/GR is already initialized?
2. **Compute-only channel type**: nouveau supports multiple channel types —
   investigate if compute-only bypasses the PMU requirement
3. **K80 reference trace**: observe what `CHANNEL_ALLOC` actually writes to registers
   on K80, then replicate the effect on Titan V via BAR0

---

## Part 4: The Naga Bypass

Exp 055 proved that DF64 transcendentals produce zero forces on ALL Vulkan backends
(proprietary, NVK, llvmpipe). Root cause: naga WGSL → SPIR-V codegen bug, not
driver JIT. The DRM dispatch path bypasses naga entirely:

```
Broken:  WGSL → naga → SPIR-V → Vulkan driver JIT → GPU  (zeros)
Bypass:  WGSL → coral-reef → native ISA → coral-driver DRM → GPU  (correct)
```

coral-reef already compiles WGSL to native GPU binary: SPH header + SASS for NVIDIA,
raw ISA for AMD. The `AmdBackend` uses `Rdna2Encoder` for VOP3 f64 instructions.
Adding GCN5 support means the MI50 can dispatch DF64 Lennard-Jones natively —
the exact kernel that returns zeros through Vulkan.

---

## Part 5: Action Items per Primal

### coralReef (P0 — owns both dispatch paths)

1. **AMD NOP dispatch validation** — create `amd_nop_dispatch` example, test on MI50
2. **GCN5 arch in coral-reef** — `Gcn5` variant in `AmdArch`, `ShaderModelGcn5`,
   validate VOP3 f64 encoding on GFX906
3. **GCN5 FLAT load fix** — zero offset field for GFX9 (differs from RDNA2)
4. **DF64 end-to-end** — compile Lennard-Jones through `AmdBackend` → dispatch via
   `AmdDevice` → verify non-zero forces
5. **K80 validation** (when hardware arrives) — `NvDevice::open()` with legacy UAPI,
   trace `CHANNEL_ALLOC` register writes

### toadStool (P2 — unblocked, infrastructure)

1. **GlowPlug socket client** — connect to coral-glowplug for automated swaps
2. **hw-learn DRM dispatch reporting** — track which DRM dispatch paths work per GPU
3. **MI50 GCN5 profile** — `DeviceCapabilities` for GFX906 (wave64, 256 VGPRs,
   1/4 f64 rate, 16GB HBM2, 1 TB/s BW)

### barraCuda (P3 — stable, evolving)

1. **RegisterMap GCN5** — extend `AmdRegisterMap` with GFX906-specific register
   offsets (if different from default GFX10 map)
2. **DF64 kernel audit** — identify which barraCuda DF64 kernels are candidates for
   first DRM dispatch (Lennard-Jones, Wilson plaquette, transport Green-Kubo)
3. **Prepare test harness** — `validate_cpu_gpu_parity` should support a
   `HOTSPRING_DISPATCH=drm` mode for testing coral-reef → coral-driver path

---

## Part 6: Hardware Matrix

| GPU | Location | Sovereign Status | DRM Status | Next Step |
|-----|----------|-----------------|------------|-----------|
| Titan V (GV100) | biomeGate 03:00.0 | 6/10 (MMU blocked) | EXEC coded, PMU blocked | Sovereign: MMU fix. DRM: PMU workaround |
| Radeon VII (MI50) | biomeGate 4d:00.0 | VFIO lifecycle only | PM4 coded, ready to test | **NOP dispatch test** |
| K80 (GK210) | In transit | Not tested | Legacy UAPI should work | Install, test both paths |
| AKD1000 (NPU) | biomeGate 45:00.0 | N/A (not GPU) | N/A | GlowPlug lifecycle validated |
| RTX 5060 (GB206) | biomeGate display | N/A (display head) | N/A (display head) | — |

---

## References

- Exp 055: DF64 Naga Poisoning
- Exp 057: coralReef Ioctl Fix (ABI corrections, PMU discovery)
- Exp 071: PFIFO Diagnostic Matrix (sovereign 6/10 layers)
- Exp 072: DRM Dispatch Evolution Matrix (this handoff's experiment journal)
- `coralReef/crates/coral-driver/src/amd/` — full AMD DRM backend
- `coralReef/crates/coral-driver/src/nv/ioctl/new_uapi.rs` — NVIDIA new UAPI
- `coralReef/crates/coral-reef/src/codegen/amd/` — AMD ISA encoder
- `coralReef/crates/coral-reef/src/gpu_arch.rs` — architecture enum
