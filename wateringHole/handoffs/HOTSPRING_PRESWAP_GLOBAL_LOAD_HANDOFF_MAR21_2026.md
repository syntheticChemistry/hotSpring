# hotSpring → Compute Trio: GCN5 Preswap Validation + GLOBAL_LOAD Investigation

**Date:** March 21, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** Exp 072 Phase 3 — Preswap validation, GLOBAL_LOAD exhaustive investigation

---

## Executive Summary

- **Preswap phases A/B/C PASS**: f64 write (42.0), f64 arithmetic (6.0×7.0=42.0), multi-workgroup (4×64 threads) — all verified on MI50 via coral-reef → coral-driver PM4
- **5 additional coral-reef compiler bugs found and fixed** (12 total for GCN5 bring-up)
- **GLOBAL_LOAD is fundamentally broken** in the current PM4 dispatch: every variant tested hangs the GPU. The existing E2E test (64/64 pass) uses only `GLOBAL_STORE`, never `GLOBAL_LOAD`
- **Root cause hypothesis**: missing PM4 register writes that Mesa/RADV `radeonsi` sets for compute dispatch (e.g., `COMPUTE_STATIC_THREAD_MGMT`, shader resource descriptors, SPI config)
- **Not ready for dual Titan runs**: AMD GLOBAL_LOAD blocks, NVIDIA PMU blocks, sovereign MMU blocks

---

## Part 1: Preswap Validation Results

Six-phase preswap test (`amd_gcn5_preswap.rs` in coral-driver examples):

| Phase | Test | Status | Key Fix |
|-------|------|--------|---------|
| A | f64 write (42.0 per thread) | **PASS** | `flat_offset` GFX9 clamping bug |
| B | f64 arithmetic (6.0 × 7.0) | **PASS** | 3 bugs: OpF2F encoding, var→let literals, f64 VGPR pair materialization |
| C | Multi-workgroup (4×64) | **PASS** | Store-only, workgroup ID routing correct |
| D | Multi-buffer read+write | **FAIL** | GLOBAL_LOAD GPU hang |
| E | f64 Lennard-Jones force | **FAIL** | Blocked by GLOBAL_LOAD |
| F | HBM2 bandwidth streaming | **FAIL** | Blocked by GLOBAL_LOAD |

---

## Part 2: Coral-Reef Compiler Fixes (5 new, 12 total)

### Fix 1: `flat_offset` GFX9 Clamping (Phase A)

**File:** `coral-reef/src/codegen/ops/mod.rs`

The `flat_offset` function clamped all offsets to 0 for `gfx_major < 10`, based on the
incorrect assumption that GFX9 FLAT instructions have no offset. GFX9 **FLAT** (SEG=00)
has no offset, but **GLOBAL** (SEG=10) supports a 13-bit signed offset. Since coral-reef
encodes all memory ops as GLOBAL, the offset should always pass through.

**Impact:** GLOBAL_STORE_DWORDX2 emitted two stores at offset 0, causing the high 32-bit
word to overwrite the low word. Result: f64 values had swapped halves (e.g., `42.0`
became `5.327e-315`).

### Fix 2: `OpF2F` Encoding (Phase B)

**File:** `coral-reef/src/codegen/ops/convert.rs`

`OpF2F` (float-to-float conversion) always emitted `V_MOV_B32` regardless of type
widths. Now dispatches to `V_CVT_F64_F32` (f32→f64) or `V_CVT_F32_F64` (f64→f32).

### Fix 3: f64 Literal VGPR Pair Materialization (Phase B)

**Files:** `coral-reef/src/codegen/ops/mod.rs`, `coral-reef/src/codegen/ops/alu_float.rs`

The optimizer collapsed `{r0, literal}` IR pairs into a single scalar. The existing
`materialize_if_literal` emitted one `V_MOV_B32` — insufficient for 64-bit values.
New `materialize_f64_if_literal` emits two `V_MOV_B32` instructions (lo=0, hi=literal)
into an adjacent VGPR pair. New `encode_vop3_f64_from_srcs` calls this for
`OpDMul`, `OpDAdd`, `OpDFma`, `OpDMnMx`.

### Fix 4: S_WAITCNT After GLOBAL_LOAD

**File:** `coral-reef/src/codegen/ops/memory.rs`

GCN5 has no hardware interlock between VMEM loads and ALU. `OpLd::encode` now appends
`S_WAITCNT vmcnt(0)` after every `GLOBAL_LOAD` to prevent WAW hazards.

### Fix 5: L1+L2 Cache Invalidation Before Dispatch

**File:** `coral-driver/src/amd/pm4.rs`

New `emit_cache_invalidate` PM4 packet (`ACQUIRE_MEM` with `TC_ACTION_ENA` + `TCL1_ACTION_ENA`)
invalidates GPU L1 (TCP) and L2 (TCC) caches before dispatch, ensuring `GLOBAL_LOAD`
reads fresh CPU-uploaded data.

---

## Part 3: GLOBAL_LOAD Investigation (Exhaustive)

**Every GLOBAL_LOAD variant causes GPU hang.** Tested:

| Variant | Encoding Details | Result |
|---------|------------------|--------|
| `GLOBAL_LOAD_DWORD` off, GLC=0 | Standard encoding | fence timeout |
| `GLOBAL_LOAD_DWORD` off, GLC=1 | Bit 17 set, L2 bypass | fence timeout |
| `FLAT_LOAD_DWORD` (SEG=00) | FLAT segment instead of GLOBAL | fence timeout |
| `GLOBAL_LOAD_DWORD` SADDR=s0 | Scalar base address mode | fence timeout |
| Load-only + S_WAITCNT + S_ENDPGM | Minimal 3-instruction shader | fence timeout |
| Load-only + S_ENDPGM (no wait) | Minimal 2-instruction shader | fence timeout |
| GTT buffer (system memory) | PCIe-accessible | fence timeout |
| VRAM buffer (device memory) | HBM2 | fence timeout |
| L1+L2 cache invalidation | PM4 ACQUIRE_MEM before dispatch | fence timeout |

**The existing `amd_gcn5_e2e.rs` (64/64 pass) uses only `GLOBAL_STORE`.** This is the
first attempt to execute `GLOBAL_LOAD` through coral-driver's PM4 path.

### Root Cause Hypothesis

Mesa's `radeonsi` compute dispatch sets additional registers that coral-driver doesn't.
Likely candidates from `si_emit_compute_shader()`:

1. `COMPUTE_STATIC_THREAD_MGMT_SE0/1` — thread distribution across shader engines
2. `SH_MEM_BASES` / `SH_MEM_CONFIG` — shared memory / flat scratch aperture config
3. `SPI_TMPRING_SIZE` — may need nonzero even without scratch usage
4. Shader resource descriptors (V# / T# / S#) for buffer access
5. `COMPUTE_DISPATCH_INITIATOR` additional bits (e.g., USE_THREAD_DIMENSIONS)

### Next Step for coralReef

Reference Mesa `radeonsi` compute dispatch path (`si_emit_compute_shader` in
`src/gallium/drivers/radeonsi/si_compute.c`) and compare PM4 register writes against
coral-driver's `build_compute_dispatch`. The delta likely contains the missing register
write that enables VMEM reads.

---

## Part 4: Action Items Per Primal

### coralReef (P0 — owns the fix)

1. **Reference Mesa radeonsi compute dispatch** — compare PM4 register writes against
   `build_compute_dispatch` in `coral-driver/src/amd/pm4.rs`
2. **Add missing registers** — likely `COMPUTE_STATIC_THREAD_MGMT_SE0/1`, `SH_MEM_BASES`,
   or shader resource descriptors
3. **Test GLOBAL_LOAD** — the preswap `amd_gcn5_preswap.rs` has handcrafted binary variants
   ready for re-testing after fixes
4. **Commit preswap fixes** — 7 modified files + 1 new in coralReef (uncommitted)

### toadStool (P2 — unblocked for other work)

1. **GlowPlug socket client** — still the main integration priority
2. **AMD DRM not yet production-ready** — store-only shaders work, but any shader reading
   GPU memory hangs. toadStool should continue using wgpu/Vulkan for AMD dispatch
3. **hw-learn update** — record that MI50 GCN5 via coral-driver supports store-only dispatch
   but not loads. This is a per-hardware capability to track

### barraCuda (P3 — absorb knowledge)

1. **RegisterMap GFX906 validated partially** — store operations confirmed working, register
   map for compute dispatch is a subset of what's needed
2. **DF64 kernel candidates still blocked** — `SHADER_YUKAWA_FORCE` (LJ), `wilson_plaquette_df64.wgsl`,
   `su3_gauge_force_df64.wgsl` all require GLOBAL_LOAD for input data
3. **`HOTSPRING_DISPATCH=drm` mode** — defer implementing until GLOBAL_LOAD is resolved
4. **EVOLUTION_READINESS.md update** — DRM dispatch section should note store-only status
   and GLOBAL_LOAD blocker

---

## Part 5: Uncommitted coralReef Changes

7 modified files + 1 new file in coralReef (need commit + push):

| File | Changes |
|------|---------|
| `crates/coral-driver/src/amd/pm4.rs` | `emit_cache_invalidate` (L1+L2 before dispatch) |
| `crates/coral-reef/src/backend.rs` | `CORAL_DEBUG_IR` env var support |
| `crates/coral-reef/src/codegen/ops/alu_float.rs` | `encode_vop3_f64_from_srcs` for f64 ops |
| `crates/coral-reef/src/codegen/ops/convert.rs` | `V_CVT_F64_F32` / `V_CVT_F32_F64` dispatch |
| `crates/coral-reef/src/codegen/ops/memory.rs` | `S_WAITCNT vmcnt(0)` after `GLOBAL_LOAD` |
| `crates/coral-reef/src/codegen/ops/mod.rs` | `flat_offset` fix, `materialize_f64_if_literal` |
| `crates/coral-reef/src/codegen/pipeline.rs` | IR debug dump at 3 stages |
| `crates/coral-driver/examples/amd_gcn5_preswap.rs` | **NEW** — 6-phase preswap test suite |

---

## Part 6: Hardware Matrix (Updated)

| GPU | Location | DRM Status | Sovereign Status | Next Step |
|-----|----------|------------|-----------------|-----------|
| Radeon VII (MI50) | biomeGate 4d:00.0 | Store-only PASS, GLOBAL_LOAD blocked | VFIO lifecycle only | Fix PM4 register config for VMEM reads |
| Titan V (GV100) | biomeGate 03:00.0 | EXEC coded, PMU blocked | 6/10 (MMU blocked) | K80 reference; sovereign MMU fix |
| K80 (GK210) | In transit | Legacy UAPI expected to work | Unlocked firmware | Install + test both paths |

---

## References

- Exp 072: DRM Dispatch Evolution Matrix
- Exp 055: DF64 Naga Poisoning
- `coralReef/crates/coral-driver/examples/amd_gcn5_preswap.rs` — preswap test suite
- `coralReef/crates/coral-driver/src/amd/pm4.rs` — PM4 command builder
- Mesa `radeonsi` compute dispatch: `src/gallium/drivers/radeonsi/si_compute.c`
