# Experiment 072: DRM Dispatch Evolution Matrix

**Date:** 2026-03-21
**Author:** hotSpring team
**Depends on:** Exp 057 (Ioctl Fix), Exp 071 (PFIFO/MMU), coralReef Iter 47+

---

## Objective

Validate DRM dispatch across AMD (MI50/GCN5) and NVIDIA (Titan V/GV100, K80/Kepler)
hardware using coralReef's existing coral-driver backends. This experiment runs in
parallel with the ongoing sovereign VFIO MMU cracking (Exp 071) and establishes the
**DRM path** as the faster route to working DF64 compute dispatch.

## Strategic Context

The sovereign and DRM paths are complementary:

- **Sovereign VFIO** (Exp 060-071): Direct hardware control, vendor-agnostic. 6/10
  layers proven, blocked at MMU page table translation (`0xbad00200`).
- **DRM dispatch** (this experiment): Kernel-mediated command submission via `nouveau`
  or `amdgpu` drivers. Kernel handles MMU, firmware, and scheduling — we supply the
  shader binary and push buffer.

coralReef already has **fully coded DRM dispatch paths** for both vendors:

| Component | AMD (`coral-driver::amd`) | NVIDIA (`coral-driver::nv`) |
|-----------|--------------------------|----------------------------|
| Device open | `AmdDevice::open()` | `NvDevice::open()` |
| Buffer alloc | GEM create + VA map | GEM new + VM_BIND |
| Cmd submission | PM4 `DRM_AMDGPU_CS` | `DRM_NOUVEAU_EXEC` (new UAPI) |
| Synchronization | `DRM_AMDGPU_WAIT_CS` | DRM syncobj wait |
| Shader format | GCN/RDNA ISA (raw binary) | SASS (SPH header + code) |

### DRM Blockers by Hardware

| GPU | Arch | DRM Path | Blocker | Workaround |
|-----|------|----------|---------|------------|
| MI50 (Radeon VII) | GCN5 / GFX906 | `amdgpu` | coral-reef only targets RDNA2+ | Add GCN5 arch; `s_endpgm` encoding identical across GCN/RDNA |
| Titan V | Volta / SM70 | `nouveau` new UAPI | `CHANNEL_ALLOC` fails: PMU firmware missing | K80 for reference; investigate FECS-only channel |
| K80 (incoming) | Kepler / SM35 | `nouveau` legacy UAPI | None expected (no PMU/GSP needed) | — |

## Experiment Plan

### Phase 1: AMD DRM NOP Dispatch (MI50)

**Goal:** Prove PM4 command submission works end-to-end on GCN5 hardware.

1. Swap MI50 to `amdgpu` via GlowPlug (`coralctl swap 0000:4d:00.0 amdgpu`)
2. Run `amd_nop_dispatch` example: `AmdDevice::open()` → alloc → upload `s_endpgm` → dispatch → sync
3. If NOP dispatch succeeds, extend to a trivial buffer-write compute shader
4. Record: context creation success, PM4 submission ioctl return, fence completion, time

**Key files:**
- `coralReef/crates/coral-driver/src/amd/mod.rs` — `AmdDevice` (`ComputeDevice` impl)
- `coralReef/crates/coral-driver/src/amd/ioctl.rs` — amdgpu DRM ioctls
- `coralReef/crates/coral-driver/src/amd/pm4.rs` — PM4 command buffer builder
- `coralReef/crates/coral-driver/examples/amd_nop_dispatch.rs` — **NEW** test binary

The `s_endpgm` instruction encoding is `0xBF810000` on all AMD architectures (SOPP
format: bits [31:23] = 0b101111111, [22:16] = opcode 0x01, [15:0] = 0x0000). This is
identical on GCN1 through RDNA4. The PM4 dispatch wrapper requires `ShaderInfo` with
valid workgroup dimensions — use `[1,1,1]` and 0 GPRs for a NOP.

### Phase 2: Full GCN5 Backend in coral-reef

**Goal:** Compile real compute shaders (including DF64) for GCN5 via coral-reef.

1. Add `Gcn5` (or `Vega`) variant to `AmdArch` in `coral-reef/src/gpu_arch.rs`
   - `gfx_major() = 9`, `has_native_f64() = true`, `f64_rate_divisor() = 4`
   - MI50 has 1/4 rate f64 (3.5 TFLOPS) — 4× faster f64 than RDNA2
2. Validate `Rdna2Encoder` VOP3 f64 compatibility with GCN5:
   - VOP3 `v_fma_f64`, `v_add_f64`, `v_mul_f64` — same encoding on both
   - SOPP (`s_endpgm`, `s_barrier`, `s_waitcnt`) — identical
   - FLAT loads — GFX9 has no `offset` field (bits [12:0] must be 0)
3. Create `ShaderModelGcn5` or parameterize `ShaderModelRdna2` by GFX version
4. Test with Lennard-Jones force kernel (the one poisoned by Naga on Vulkan)

**Key ISA differences GCN5 vs RDNA2:**
- Wave size: 64 (GCN5) vs 32 (RDNA2 default, 64 in wave64 mode)
- FLAT offset: 0 on GFX9, 12-bit signed on GFX10+
- `s_waitcnt` format: `vm_cnt[3:0]` + `exp_cnt[6:4]` + `lgkm_cnt[13:8]` (same layout)
- VOP3 opcode map: most f64 ops at same opcode numbers

### Phase 3: NVIDIA DRM Validation (K80)

**Goal:** Validate full nouveau DRM dispatch on unlocked Kepler hardware.

When K80 arrives:
1. Install, bind to `nouveau` via GlowPlug
2. `NvDevice::open()` — should succeed (Kepler uses legacy UAPI, no VM_INIT needed)
3. `CHANNEL_ALLOC` → `GEM_NEW` → `GEM_PUSHBUF` pipeline
4. NOP pushbuf (METHOD = 0, DATA = 0), verify fence completion
5. Trace nouveau's MMU setup via BAR0 to inform sovereign page table work

### Phase 4: NVIDIA DRM on Titan V — PMU Investigation

**Goal:** Determine if Titan V DRM dispatch can be unblocked without PMU firmware.

Options:
1. **FECS-only channel** — Exp 068 proved FECS executes from host IMEM. Can we
   satisfy `CHANNEL_ALLOC`'s engine init requirement with just FECS/GR?
2. **Compute-only channel type** — nouveau may support channel types that bypass PMU
3. **GSP firmware** — GV100 predates GSP, but check if kernel 6.17 has any GSP
   paths that could substitute
4. **DRM dispatch on K80 reference** — use K80's working DRM to understand what
   CHANNEL_ALLOC actually needs, then replicate on Titan V

## Expected Outcomes

| Milestone | Status | Dependencies |
|-----------|--------|--------------|
| AMD NOP dispatch (PM4 → fence) | Pending | MI50 on `amdgpu` |
| AMD buffer-write dispatch | Pending | NOP success |
| GCN5 arch in coral-reef | Pending | — |
| GCN5 DF64 Lennard-Jones | Pending | GCN5 arch + NOP success |
| K80 channel creation | Pending | K80 hardware arrival |
| K80 NOP pushbuf dispatch | Pending | K80 channel |
| Titan V PMU workaround | Research | K80 reference data |

## The Naga Bypass

This experiment is the fastest path to resolving the DF64 Naga poisoning (Exp 055):

```
Current (broken):  WGSL → naga → SPIR-V → Vulkan driver → GPU  (all zeros)
Bypass (this exp): WGSL → coral-reef → native ISA → coral-driver DRM → GPU  (correct)
```

If Phase 2 succeeds, we can dispatch the exact same Lennard-Jones kernel that
produces zeros through Vulkan and verify correct forces through DRM. This would
confirm the Naga codegen bug in isolation and provide a working compute path.

## Relationship to Sovereign Path

DRM dispatch does NOT replace the sovereign VFIO work — it complements it:

- **DRM gives us a working reference** for MMU page table encoding, command
  submission format, and firmware loading sequences
- **Sovereign gives us vendor independence** — no kernel driver, no firmware blobs,
  any PCIe device
- **K80 bridges both** — unlocked firmware means sovereign works, and legacy
  nouveau means DRM works, both on the same hardware

## References

- Exp 055: DF64 Naga Poisoning — root cause in naga WGSL→SPIR-V codegen
- Exp 057: coralReef Ioctl Fix — ABI corrections, PMU firmware blocker discovered
- Exp 071: PFIFO Diagnostic Matrix — sovereign pipeline 6/10 layers, MMU blocker
- `coralReef/crates/coral-driver/src/amd/` — full AMD DRM backend
- `coralReef/crates/coral-driver/src/nv/ioctl/new_uapi.rs` — nouveau new UAPI
- `coralReef/crates/coral-reef/src/codegen/amd/` — AMD ISA encoder (RDNA2)
- `coralReef/crates/coral-reef/src/gpu_arch.rs` — architecture enum (needs GCN5)
