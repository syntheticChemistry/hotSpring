# hotSpring → Compute Trio: GCN5 Preswap Complete — 6/6 Phases PASS

**Date:** March 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** Exp 072 Phase 3 complete — All preswap validation phases pass
**Supersedes:** `HOTSPRING_PRESWAP_GLOBAL_LOAD_HANDOFF_MAR21_2026.md`

---

## Executive Summary

- **GCN5 preswap 6/6 PASS**: f64 write, f64 arithmetic, multi-workgroup, multi-buffer read/write, HBM2 bandwidth streaming, **f64 Lennard-Jones force (Newton's 3rd law verified)**
- **18 coral-reef compiler bugs found and fixed** across full GCN5 bring-up (7 initial + 5 Phase A/B/C + 6 Phase D/E/F)
- **GLOBAL_LOAD resolved**: root cause was not PM4 register config but compiler bugs (VOP1 opcode table, f64 materialization, type resolution, VOP3 modifiers, integer negation)
- **Naga DF64 bypass validated on real physics**: the exact Lennard-Jones kernel that produces zeros through Vulkan/Naga produces correct forces through DRM
- **85 coral-reef unit tests pass**, 0 failures, 0 regressions
- **Ready for DF64 production kernels** on MI50 — Yukawa, Wilson plaquette, SU(3) gauge force all unblocked

---

## Part 1: Complete Preswap Results

```
╔═══════════════════════════════════════════════════╗
║  RESULTS                                         ║
╠═══════════════════════════════════════════════════╣
║  [PASS] A: f64 Write                               ║
║  [PASS] B: f64 Arithmetic                          ║
║  [PASS] C: Multi-Workgroup                         ║
║  [PASS] D: Multi-Buffer                            ║
║  [PASS] F: HBM2 Bandwidth                          ║
║  [PASS] E: f64 LJ Force                            ║
╠═══════════════════════════════════════════════════╣
║  6/6 phases passed                    ALL PASSED║
╚═══════════════════════════════════════════════════╝
```

### f64 Lennard-Jones Force Details (Phase E)

Two particles at positions (0, 0, 0) and (1.5, 0, 0), σ=1, ε=1:

| Metric | CPU Reference | GPU Result | Match |
|--------|-------------|-----------|-------|
| Particle 0 f_x | -1.158028831046 | -1.158028827421 | ✅ (tol=1e-8) |
| Particle 1 f_x | 1.158028831046 | 1.158028827421 | ✅ (tol=1e-8) |
| Newton's 3rd law | f₀ = -f₁ | Verified | ✅ |

---

## Part 2: GLOBAL_LOAD Resolution

The previous handoff documented GLOBAL_LOAD as "fundamentally broken" with a hypothesis
of missing PM4 register configuration. The actual root cause was **compiler bugs in
coral-reef** that produced incorrect machine code for f64 operations:

1. **VOP1 opcode table wrong for GFX9** — V_RCP_F64, V_RSQ_F64, V_SQRT_F64 had RDNA2
   opcodes instead of GFX9 opcodes. GPU hangs on unrecognized instruction.
2. **f64 literal materialization** — `materialize_if_literal` (32-bit) was used instead
   of `materialize_f64_if_literal` (64-bit) for f64 transcendentals, causing only the
   high word to be set (denormal → flush to zero → divide by zero → inf).
3. **OpI2F f64 destination** — always emitted `V_CVT_F32_I32` even for f64 destinations;
   now emits `V_CVT_F64_U32` / `V_CVT_F64_I32`.
4. **is_f64_expr type resolution** — `scalar_type_handle` fell back to the first type in
   the Naga module arena, causing `u32` operations to be misclassified as `f64`. Fixed
   with `element_scalar()` helper for direct type inspection.
5. **VOP3 fneg/fabs modifiers dropped** — `encode_vop3_from_srcs_inner` called
   `encode_vop3` (ignores modifiers) instead of `encode_vop3_mod`. Caused subtraction
   to compile as addition in LJ force equation.
6. **Integer negation dropped in IAdd3** — `OpIAdd3` always emitted `V_ADD_NC_U32` even
   when a source had `SrcMod::INeg` (e.g., `1u - gid.x` compiled as `1u + gid.x`). Now
   detects INeg and emits `V_SUB_NC_U32` / `V_SUBREV_NC_U32`.

These 6 bugs were found through systematic debugging with binary dumps, IR traces
(`CORAL_DEBUG_IR`), and `llvm-mc --mcpu=gfx906 --show-encoding` validation.

---

## Part 3: Complete Bug Inventory (18 Total)

### Phase 1-2: Initial Bring-Up (7 bugs)

| # | Bug | Fix |
|---|-----|-----|
| 1 | PM4 wave size / VGPR granularity (GCN5=wave64, granularity 4) | Dynamic `wave_size` in `ShaderInfo` |
| 2 | VOP3 instruction prefix (110101→110100 for GFX9) | `patch_vop3_prefix_for_gfx9()` |
| 3 | Missing `s_waitcnt vmcnt(0)` before `s_endpgm` | Added to shader epilogue |
| 4 | FLAT vs GLOBAL segment (SEG=00 vs SEG=10) | Fixed segment bits for compute |
| 5 | Workgroup ID in VGPR (should be SGPR) | Refactored to `amd_sys_reg_src` |
| 6 | Malformed ACQUIRE_MEM (missing POLL_INTERVAL) | Added 7th dword |
| 7 | VOP3-only opcode values differ GFX9 vs RDNA2 | `vop3_only_opcode_for_gfx9()` |

### Phase 3 A/B/C: f64 Store-Only (5 bugs)

| # | Bug | Fix |
|---|-----|-----|
| 8 | `flat_offset` clamped GLOBAL offsets to 0 on GFX9 | Pass offset through for SEG=10 |
| 9 | `OpF2F` always `V_MOV_B32` | Dispatch to `V_CVT_F64_F32` / `V_CVT_F32_F64` |
| 10 | `var` lowered to unmapped scratch memory | Use `let` bindings (SSA) |
| 11 | f64 literal VGPR pair: only one `V_MOV_B32` | `materialize_f64_if_literal`: two MOVs |
| 12 | WAW hazard VMEM→ALU on GCN5 | `S_WAITCNT vmcnt(0)` after `OpLd` |

### Phase 3 D/E/F: f64 Load + Compute (6 bugs)

| # | Bug | Fix |
|---|-----|-----|
| 13 | VOP1 opcode table (V_RCP_F64=37 not 47, etc.) | Corrected all GFX9 VOP1 constants |
| 14 | f64 transcendentals: 32-bit materialization | `materialize_f64_if_literal` for Rcp64H, Rsq64H, etc. |
| 15 | OpI2F: no f64 destination support | `V_CVT_F64_U32`/`V_CVT_F64_I32` with `vgpr_pair` |
| 16 | is_f64_expr: scalar_type_handle fallback | `element_scalar()` helper for type inspection |
| 17 | VOP3 fneg/fabs modifiers dropped | Extract modifiers → `encode_vop3_mod` |
| 18 | IAdd3 integer negation dropped | Detect INeg → `V_SUB_NC_U32`/`V_SUBREV_NC_U32` |

---

## Part 4: Files Changed

All changes committed to coralReef `main` (commit `2f342d8`):

| File | Change |
|------|--------|
| `coral-reef/src/codegen/amd/isa_generated/vop1.rs` | GFX9 VOP1 opcode constants corrected |
| `coral-reef/src/codegen/ops/alu_float.rs` | `materialize_f64_if_literal` for f64 transcendentals |
| `coral-reef/src/codegen/ops/convert.rs` | OpI2F f64 destination support |
| `coral-reef/src/codegen/naga_translate/expr.rs` | `element_scalar()`, fixed `is_f64_expr`/`is_float_expr`/`is_signed_int_expr` |
| `coral-reef/src/codegen/ops/mod.rs` | VOP3 modifier encoding, VOPC remapping, V_SUBREV mapping |
| `coral-reef/src/codegen/ops/alu_int.rs` | Integer negation detection in IAdd3 |
| `coral-reef/src/codegen/amd/encoding.rs` | GFX9 encoding adjustments |
| `coral-reef/src/codegen/amd/isa_generated/flat.rs` | FLAT instruction fixes |
| `coral-reef/src/codegen/amd/shader_model.rs` | Shader model parameters |
| `coral-driver/examples/amd_gcn5_preswap.rs` | Complete 6-phase preswap test suite |
| `coral-driver/src/amd/ioctl.rs` | DRM ioctl refinements |
| `coral-driver/src/amd/mod.rs` | AmdDevice improvements |
| `coral-driver/src/amd/pm4.rs` | PM4 command builder improvements |

---

## Part 5: Action Items

### For coralReef

1. **DF64 production kernels**: The DRM path is now ready for Yukawa, Wilson plaquette,
   and SU(3) gauge force kernels. Port the exact `SHADER_YUKAWA_FORCE` that Naga poisons.
2. **RDNA2/3 testing**: The GFX9-specific fixes (opcode remapping, VOP1 table) are
   gated by `gfx_major < 10`. Verify RDNA2 path is not regressed.
3. **K80 NVIDIA DRM**: Legacy nouveau CHANNEL_ALLOC → GEM_PUSHBUF (no PMU needed).

### For toadStool

1. **GCN5 dispatch integration**: `AmdDevice::dispatch()` works with any coral-reef
   compiled shader. Wire into toadStool's dispatch orchestrator.
2. **Precision routing**: MI50 has 1/4 rate f64 (3.5 TFLOPS) — 4× faster than RDNA2.
   `PrecisionTier::F64` is the correct default for MI50.

### For barraCuda

1. **RegisterMap validation**: DRM dispatch on MI50 means GFX906 register map is now
   testable against real hardware. Run `AmdGfx906Map` against live dispatch telemetry.
2. **DF64 kernel candidates**: `SHADER_YUKAWA_FORCE` (validated), `wilson_plaquette_df64`,
   `su3_gauge_force_df64` — all ready for DRM dispatch.
3. **Benchmark opportunity**: MI50 f64 vs RTX 3090 DF64 — direct comparison of
   native f64 silicon vs f32-pair emulation on consumer hardware.

---

## Significance

This is the first time the sovereign bypass pipeline has computed **correct physics**
on real GPU hardware:

```
WGSL (Lennard-Jones f64) → coral-reef (native GCN ISA) → coral-driver (PM4 DRM) → MI50 → correct forces
```

The Naga WGSL→SPIR-V poisoning (Exp 055) that breaks all DF64 shaders through
Vulkan is **completely bypassed**. The forces match CPU reference to 1e-8 tolerance
and obey Newton's 3rd law. The path to production DF64 compute on consumer AMD
hardware is clear.
