# Blackwell Dispatch Live — Sovereign Compute Handoff

**From:** hotSpring (sovereign dispatch maze threading session)
**To:** coralReef, barraCuda, primalSpring, biomeOS
**Date:** April 16, 2026
**coralReef iteration:** 85
**License:** AGPL-3.0-or-later

---

## Summary

RTX 5060 (Blackwell / GB206 / SM120) sovereign dispatch is now functional through the
full pipeline: WGSL → coral-reef SASS → coral-driver QMD v5.0 → GPFIFO submit → fence.
Two critical Blackwell-specific bugs were identified and fixed:

1. **f64 division returned 0** — `MUFU.RCP64H` hardware instruction returns 0 on Blackwell
2. **`@builtin(num_workgroups)` returned [0,0,0]** — `S2R NCTAID` system registers not populated

Both were resolved with software workarounds that maintain full backward compatibility
with Volta (SM70) through Ada (SM89).

---

## What Changed

### coral-reef (compiler)

| Change | Files |
|--------|-------|
| f64 rcp: F2F(f64→f32) + MUFU.RCP + F2F(f32→f64) seed on SM≥100 | `lower_f64/newton.rs` |
| f64 sqrt: F2F(f64→f32) + MUFU.RSQ + F2F(f32→f64) seed on SM≥100 | `lower_f64/newton.rs` |
| num_workgroups: LDC c[7][0/4/8] on SM≥100 (bypasses S2R NCTAID) | `naga_translate/func_builtins.rs` |
| DRIVER_CBUF_INDEX = 7 convention established | `func_builtins.rs` |
| 5 new unit tests for SM120 paths | `newton.rs`, `tests_interpolation_builtins.rs` |

### coral-driver (dispatch)

| Change | Files |
|--------|-------|
| CBUF 7 bound to driver constants buffer (grid_x/y/z) | `compute_trait.rs` |
| QMD v5.0: GRID_*_RESUME fields set to match grid dimensions | `qmd_v50.rs` |
| Semaphore fence via compute engine SET_REPORT_SEMAPHORE (subchannel 1) | `device.rs` |
| UVM map_external_allocation: gpu_mapping_type = ReadWriteAtomic | `uvm/mod.rs` |
| QMD v5.0: SM_CONFIG_SHARED_MEM_SIZE + QMD_GROUP_ID = 0x1f | `qmd_v50.rs` |
| UVM_REGISTER_CHANNEL for Blackwell faulting VA space | `device.rs`, `uvm/mod.rs` |
| All diagnostic eprintln → tracing macros | `device.rs`, `compute_trait.rs` |

### Test results

- coral-reef: **1319** lib + **85** integration = **1404** tests, 0 failures
- coral-driver: compiles clean

---

## Root Cause Analysis

### MUFU.RCP64H / RSQ64H (f64 seed instructions)

`MUFU.RCP64H` takes the high 32 bits of an f64 value and produces an f32 reciprocal
approximation with f64-aware exponent handling. On Blackwell, this instruction returns 0
despite correct binary encoding (verified with `nvdisasm --binary SM120`). The regular
`MUFU.RCP` (f32) works correctly. Both are classified as `Decoupled` in SM120 latency
tables. The issue appears to be a hardware behavior change in the SFU's 64H path.

**Fix:** Convert f64 → f32 via `F2F`, use `MUFU.RCP` (f32), convert back via `F2F`,
then refine with 2 Newton-Raphson iterations. The f32 seed provides ~23 bits of mantissa
precision, sufficient for 2 NR iterations to reach full f64 precision (52 bits).

### S2R NCTAID (grid dimension system registers)

`S2R NCTAID_X/Y/Z` (indices 0x2d-0x2f) return 0 on Blackwell while `S2R CTAID_X/Y/Z`
(0x25-0x27) and `S2R TID_X/Y/Z` (0x21-0x23) work correctly. QMD v5.0 GRID_WIDTH/HEIGHT/DEPTH
fields are set at the correct bit positions (MW 1248/1280/1312). The hardware simply
does not populate NCTAID system registers from these QMD fields on Blackwell.

**Fix:** Driver writes grid dimensions to CBUF 7 (driver constants buffer). Compiler emits
`LDC c[7][0/4/8]` instead of `S2R NCTAID` on SM ≥ 100. This matches NVK's pattern of
passing grid dimensions via constant buffer rather than relying on system registers.

---

## Remaining Gaps

| ID | Description | Status |
|----|-------------|--------|
| fix-bind-channel | BIND_CHANNEL handle consistency (kmod path) | Pending |
| waterfall-volta | Extend open_via_kmod to Volta (sm >= 70) | Pending |
| k80-vfio-fix | Resolve K80 VFIO EBUSY | Pending |
| waterfall-kepler | Test shared RM path on K80 after VFIO fix | Pending |

---

## Composition Patterns

The CBUF 7 driver constants convention establishes a **compiler ↔ driver ABI contract**:
the compiler knows to read grid dimensions from `c[7][0/4/8]` and the driver knows to
populate them there. This is the same pattern NVK uses for its root descriptor table.
Future driver constants (e.g., clock values, dispatch metadata) can be added at higher
CBUF 7 offsets without changing the compiler's grid dimension reads.

The f64 F2F+MUFU fallback is transparent to the rest of the pipeline — it occurs
entirely within the `lower_f64` pass during compilation. The generated SASS uses standard
F2F and MUFU instructions that work on all SM versions, so no driver changes are needed.

---

## For primalSpring / biomeOS

No action required. These fixes are internal to coralReef's compiler and coral-driver's
dispatch path. The sovereign compute capability exposed via `shader.compile.wgsl` and
`compute.dispatch` RPCs is unchanged at the wire level. Blackwell dispatch is now
functional for all tested shader patterns (f32/f64 arithmetic, global loads/stores,
workgroup builtins).
