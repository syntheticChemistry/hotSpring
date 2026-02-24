# hotSpring v0.6.7 → ToadStool Session 42+ Handoff

**Date:** February 22, 2026
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-only
**Context:** ToadStool Sessions 40–42 absorbed remaining Spring shader gaps, fixed critical
eigensolve bugs, and renamed BarraCUDA → BarraCuda. hotSpring applied the loop_unroller fix,
removed its catch_unwind workaround, and synchronized the rename.

---

## Executive Summary

ToadStool completed Sessions 40–42 since the V065 handoff, bringing the shader count to
612 (zero orphans), fixing 3 critical bugs (Jacobi eigenvector rotation, ODE f64 builtins,
SNP binding mismatch), adding 19 new WGSL shaders (MD observables, special functions,
numerical methods), and renaming BarraCUDA → BarraCuda to match ecoPrimals camelCase.

**hotSpring applied the documented loop_unroller u32 fix** to toadstool's
`substitute_loop_var` — changing `iter.to_string()` to `format!("{iter}u")`. This resolved
the single-dispatch `BatchedEighGpu` panic. The `catch_unwind` wrapper in
`validate_barracuda_hfb.rs` has been removed and replaced with a direct `.expect()` call.

**Full validation: 34/34 suites pass (702.7s), 0 clippy warnings, all unit tests green.**

---

## Part 1: ToadStool Sessions 40–42 Changes (Absorbed Upstream)

### Session 40 — Richards PDE + Moving Window Stats
| Item | Impact |
|------|--------|
| Richards 1D unsaturated flow solver | Van Genuchten-Mualem, Picard/CN (airSpring precision agriculture) |
| Moving window statistics WGSL shader | Mean/var/min/max over sliding windows (IoT sensor streams) |
| Dependency audit | Workspace confirmed pure Rust; libc only for akida VFIO ioctls |

### Session 41 — f64 Shader Fixes + API Exposure
| Item | Impact |
|------|--------|
| 6 f64 shader compile bugs fixed | `batched_ode_rk4`, `batch_pair_reduce_f64`, `batch_tolerance_search_f64`, `kmd_grouping_f64`, `hill_f64`, **`GemmCachedF64`** — all now correctly use `compile_shader_f64()` |
| `cpu_conv_pool` promoted to `pub` | `conv2d`, `max_pool2d`, `avg_pool2d` — unblocks neuralSpring LeNet-5 |
| All 25 bio ops re-exported at crate root | Was 10, now complete |
| Zero orphan shaders (593/593 → 593) | rdf_histogram.wgsl + u64_emu.wgsl wired |

### Session 42 — Shader-First Unified Math (19 New WGSL)
| Category | Shaders |
|----------|---------|
| Special functions | `ln_beta`, `regularized_gamma`, `incomplete_gamma`, `factorial`, `erfc_deriv` |
| Statistics | `chi_squared`, `chi2_decomposed`, `bootstrap_mean` |
| Numerical methods | `trapz`, `rk45`, `cubic_spline_eval` |
| Science | `van_genuchten` (soil hydraulics) |
| MD observables | `msd`, `vacf`, `ssf`, `bspline`, `charge_spread`, `force_interpolation` |
| Misc | `iou`, `quantize_params`, `laguerre_generalized` |
| **Total** | **612 WGSL shaders, zero orphans** |

### Session 42+ — BarraCuda Rename
All 255 files updated in toadstool. Display name: **BarraCuda** (Barrier-free Rust-Abstracted
Computationally Unified Dimensionalized Algebra). Crate name `barracuda` unchanged.

### Bug Fix Commit — Critical Corrections
| Bug | Fix |
|-----|-----|
| SNP BGL binding mismatch | BGL had 7 entries; shader expects 6. Binding 2 corrected to `read_only` |
| ODE shader f64 builtins | WGSL `max`/`pow`/`clamp` don't support f64 in naga 22.1.0; replaced with manual polyfills |
| Jacobi eigenvector rotation | V-rotation was inside `k!=p && k!=q` guard (wrong for eigenvectors). Now rotates ALL rows |

---

## Part 2: hotSpring Changes Applied

### Loop Unroller u32 Fix (Applied to ToadStool)
**File:** `crates/barracuda/src/shaders/optimizer/loop_unroller.rs:312`

```rust
// Before (bug):
let replacement = iter.to_string();   // emits "0", "1" — WGSL infers i32
// After (fix):
let replacement = format!("{iter}u"); // emits "0u", "1u" — correct u32
```

**Impact:** Single-dispatch `BatchedEighGpu` now compiles and passes WGSL validation.
Combined with the Jacobi eigenvector fix, single-dispatch eigensolve produces:
- Eigenvalue relative error: 2.38e-12
- Eigenvector orthogonality: 2.89e-15

### catch_unwind Removal (hotSpring)
`validate_barracuda_hfb.rs` no longer wraps `execute_single_dispatch` in
`std::panic::catch_unwind(AssertUnwindSafe(...))`. Direct call with `.expect()`.

### BarraCuda Rename (hotSpring)
All active Rust source (20 files), documentation (28 files), `Cargo.toml`, `clippy.toml`,
and `CITATION.cff` updated. Archive handoffs left as historical fossil record.

---

## Part 3: Absorption Status — Complete Matrix

| Priority | Item | Status | Notes |
|----------|------|--------|-------|
| P1 | Staggered Dirac GPU shader | ✅ **Done** (S31d) | |
| P1 | CG solver (3 shaders) | ✅ **Done** (S31d) | |
| P1 | Pseudofermion HMC | ❌ **Still pending** | CPU module, 477 lines, 4+7 tests |
| P1 | ESN `export_weights()` | ✅ **Done** (S36-37) | |
| P1 | Loop unroller u32 fix | ✅ **FIXED** (hotSpring → toadstool) | One-line change, validated |
| P2 | Screened Coulomb eigensolve | ❌ **Still pending** | Murillo-Weisheit Sturm bisection |
| P2 | HFB shader suite (spherical) | ✅ **Done** (S36-37) | |
| P2 | HFB shader suite (deformed) | ✅ **Done** (S36-37) | |
| P3 | forge substrate discovery | ✅ **Partial** | `SubstrateCapability` absorbed; full NPU probing in hotSpring |
| P1 | GemmCachedF64 f64 compile fix | ✅ **Done** (S41) | Now usable for HFB SCF |

**Remaining absorption queue: 2 items (pseudofermion HMC + screened Coulomb)**

---

## Part 4: What hotSpring Should Adopt from ToadStool

| Primitive | Upstream Module | Benefit | Priority |
|-----------|----------------|---------|----------|
| `GemmCachedF64` | `ops::linalg` | HFB SCF: cached workspace eliminates per-iteration alloc | P1 |
| `execute_to_buffer` | `ops::linalg::GemmCachedF64` | GPU-resident GEMM output feeds next dispatch directly | P1 |
| `BatchIprGpu` | `spectral` | GPU inverse participation ratio for Anderson diagnostics | P2 |
| `NelderMeadGpu` | `optimize` | GPU-parallel L1 parameter search | P2 |
| `UnidirectionalPipeline` | `staging` | Ring-buffer staging for MD streaming dispatch | P2 |
| Upstream `ESN` (v2) | `esn_v2` | Replace local `md/reservoir.rs` — upstream now has weight export | P3 |
| `msd_f64.wgsl` | `ops::md::observables` | GPU MSD with PBC (hotSpring currently CPU-only MSD) | P3 |
| `vacf_f64.wgsl` | `ops::md::observables` | Upload-based VACF (hotSpring keeps streaming VACF for transport) | P3 |

**Note on VACF divergence:** hotSpring's `vacf_batch_f64.wgsl` uses `GpuVelocityRing`
(zero-readback streaming transport pipeline). ToadStool's `vacf_f64.wgsl` uses
upload-then-compute. Both compute the same physics; hotSpring's path is specialized for
streaming MD → D* pipeline. No replacement needed, but upstream should eventually support
a ring-buffer variant.

---

## Part 5: Codebase Health

| Metric | hotSpring (v0.6.7) | ToadStool (Session 42+) |
|--------|-------------------|------------------------|
| Unit tests | 619 (all pass) | 3,847+ |
| Validation suites | 34/34 | 195/195 + 48/48 wetSpring |
| WGSL shaders | 34 (zero inline) | 612 (zero orphans) |
| Clippy warnings | 0 | 0 |
| Unsafe blocks | 0 | 55 (all SAFETY documented) |
| Display name | BarraCuda | BarraCuda |

---

## Part 6: Active Handoff Documents

| Document | Scope | Status |
|----------|-------|--------|
| **This document** (V067) | S40-S42 catch-up, loop_unroller fix, rename | **Latest** |
| `HOTSPRING_V066_GPU_TRANSPORT_HANDOFF_FEB22_2026.md` | GPU-resident transport pipeline | Current |
| `HOTSPRING_V064_DYNAMICAL_QCD_HANDOFF_FEB22_2026.md` | Pseudofermion HMC details | Current |
| `CROSS_SPRING_EVOLUTION_FEB22_2026.md` | Cross-spring shader map | Current |

All prior handoffs archived in `wateringHole/handoffs/archive/`.

---

*This handoff follows the unidirectional pattern: hotSpring → wateringHole → ToadStool.
No inter-Spring imports. All code is AGPL-3.0.*
