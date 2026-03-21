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
| MI50 (Radeon VII) | GCN5 / GFX906 | `amdgpu` | ~~coral-reef only targets RDNA2+~~ **RESOLVED** | GCN5 arch added; E2E PASSED (Phase 2) |
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

## Results

### Phase 1 Results: AMD NOP Dispatch — PASSED (March 21, 2026)

```
╔═══════════════════════════════════════════════╗
║  AMD DRM NOP Dispatch Test (Exp 072)         ║
╚═══════════════════════════════════════════════╝

  Phase 1: AmdDevice::open()... OK ✓       (amdgpu context created on renderD129)
  Phase 2: Alloc shader buffer (256B GTT).. OK ✓  (GEM create + VA map)
  Phase 3: Upload s_endpgm (0xBF810000)... OK ✓   (mmap write)
  Phase 4: Dispatch (1×1×1 workgroups).... OK ✓    (PM4 DISPATCH_DIRECT → DRM_AMDGPU_CS)
  Phase 5: Sync (fence wait).............. OK ✓    (DRM_AMDGPU_WAIT_CS completed)

  ✓ AMD DRM NOP DISPATCH: ALL PHASES PASSED
```

**Hardware:** Radeon VII (Vega 20 / GFX906), BDF `0000:4d:00.0`, `amdgpu` driver,
`renderD129`, kernel 6.17.9.

**What this proves:**
- coral-driver `AmdDevice` `ComputeDevice` trait implementation is functional E2E
- amdgpu DRM ioctl wrappers (context create, GEM alloc, VA map, CS submit, wait) all correct
- PM4 `DISPATCH_DIRECT` packet format accepted by GCN5 hardware
- GPU executed the instruction and signaled fence completion
- The `s_endpgm` encoding (0xBF810000) is confirmed identical on GCN5 and RDNA2

### Phase 2 Results: GCN5 E2E Compute Dispatch — PASSED (March 21, 2026)

Full compiler pipeline WGSL → coral-reef (GCN5/GFX906) → coral-driver (PM4) → MI50 → readback:

```
╔═══════════════════════════════════════════════════╗
║  GCN5 E2E: coral-reef → coral-driver → MI50     ║
╚═══════════════════════════════════════════════════╝

  Phase 1: Compile WGSL → GCN5 (gfx906)... OK ✓ (68 bytes, 26 GPRs, 12 instrs)
  Phase 2: AmdDevice::open()... OK ✓
  Phase 3: Alloc output buffer (256 bytes)... OK ✓
  Phase 4: Dispatch (1 workgroup × 64 threads)... OK ✓
  Phase 5: Sync... OK ✓
  Phase 6: Readback and verify... OK ✓ (64/64 elements = 42.0)

  ✓ GCN5 E2E DISPATCH: ALL PHASES PASSED
```

**What this proves:**
- coral-reef compiles WGSL to correct GCN5 native ISA (VOP3 opcode translation, wave64)
- coral-driver PM4 dispatch handles GCN5 (VGPR granularity 4, wave64, ACQUIRE_MEM L2 flush)
- Per-thread global memory stores work correctly across all 64 lanes
- The Naga bypass path is **validated end-to-end**: WGSL → native ISA → DRM → correct output

**Bugs found and fixed during Phase 2:**

| Bug | Root Cause | Fix |
|-----|------------|-----|
| GPU hang (fence timeout) | PM4 VGPR granularity hardcoded for RDNA2 (8); GCN5 needs 4. `CS_W32_EN` set for wave32; GCN5 is wave64 | Added `wave_size` to `ShaderInfo`, dynamic VGPR granularity and DISPATCH_INITIATOR |
| GPU hang (fence timeout #2) | VOP3 prefix `110101` (RDNA2) causes illegal instruction on GFX9 (`110100`) | `patch_vop3_prefix_for_gfx9()` in encoder |
| 63/64 elements wrong (only out[0]) | Missing `s_waitcnt vmcnt(0)` before `s_endpgm` | Added to `encode_rdna2_shader` |
| 63/64 elements wrong | FLAT addressing (SEG=00) relies on aperture; compute uses GLOBAL (SEG=10) | Changed `encode_flat_load/store/atomic` to set SEG=10 |
| 63/64 elements wrong | `SR_CTAID_X` (workgroup_id) mapped to VGPR instead of SGPR | Refactored to `amd_sys_reg_src`, added `user_sgpr_count` tracking |
| GPU hang (ACQUIRE_MEM) | PM4 packet header declared 6 dwords but only pushed 5 (missing POLL_INTERVAL) | Added 7th push for POLL_INTERVAL |
| VOP3 MAD produces zeros | GFX9 VOP3-only opcodes differ from RDNA2 (V_MAD_U32_U24: 323→451). `patch_vop3_prefix_for_gfx9` only changed prefix, not opcode | Added `vop3_only_opcode_for_gfx9()` translation table (LLVM-validated) |

**Diagnostic methodology:**

7 handcrafted binary tests isolated the root cause systematically:
- Test A (fixed addr): Basic GLOBAL store works ✓
- Test B (per-thread VOP2): `V_LSHLREV_B32` + `V_ADD_CO_U32` → 64/64 ✓
- Test C (dump v0): Confirmed v0 = thread_id_x ✓
- Test D (compiler binary, gpr=5): Ruled out ShaderInfo as cause ✗ (1/64)
- Test E (VOP3 MAD per-thread): Failed → isolated VOP3 MAD as broken ✗
- Test F (VOP3 MAD dump): MAD produced all zeros → opcode wrong ✗
- Test G (VOP2 MUL per-thread): 64/64 ✓ → confirmed VOP2 works, VOP3-only broken

This led to the discovery that GFX9 and RDNA2 VOP3-only opcodes occupy different
ranges despite sharing the same 10-bit OP field layout. Group A (MAD/FMA/BFE/BFI,
RDNA2 320-351) shifts by +128. Group B (F64/MUL_HI) has per-instruction mapping.
Every entry was validated against `llvm-mc --mcpu=gfx906 --show-encoding`.

## Expected Outcomes

| Milestone | Status | Dependencies |
|-----------|--------|--------------|
| AMD NOP dispatch (PM4 → fence) | **PASSED** | MI50 on `amdgpu` |
| AMD buffer-write dispatch | **PASSED** (64/64) | NOP success |
| GCN5 arch in coral-reef | **COMPLETE** | — |
| GCN5 E2E compute dispatch | **PASSED** | GCN5 arch + NOP success |
| GCN5 DF64 Lennard-Jones | Next | GCN5 E2E success |
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
- `coralReef/crates/coral-reef/src/codegen/amd/` — AMD ISA encoder (RDNA2 + GCN5)
- `coralReef/crates/coral-reef/src/gpu_arch.rs` — architecture enum (includes GCN5)
