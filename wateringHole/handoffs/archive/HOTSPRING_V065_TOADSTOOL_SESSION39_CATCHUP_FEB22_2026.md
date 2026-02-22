# hotSpring v0.6.5 → ToadStool Session 39 Catch-Up Handoff

**Date:** February 22, 2026
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-only
**Context:** ToadStool Sessions 31d–39 absorbed most of hotSpring v0.6.4 handoff items.
This document acknowledges absorptions, updates the remaining gap list, and documents
the loop_unroller bug root cause with a one-line fix.

---

## Executive Summary

ToadStool has done significant work since the v0.6.4 handoff: 30+ commits across
Sessions 31d–39, absorbing Dirac/CG lattice GPU primitives, deformed and spherical
HFB shader suites, ESN weight export/import, SubstrateCapability model, and extensive
code quality hardening (zero clippy warnings, 3,847+ tests, 589+ shaders, zero orphans).

**Of the 8 absorption targets in the v0.6.4 handoff, 6 are now DONE:**

| Priority | Item | Status |
|----------|------|--------|
| P1 | Staggered Dirac GPU shader | ✅ **Absorbed** (Session 31d) |
| P1 | CG solver (3 shaders) | ✅ **Absorbed** (Session 31d) |
| P1 | Pseudofermion HMC | ❌ **Still pending** |
| P1 | ESN `export_weights()` | ✅ **Absorbed** (Session 36-37) |
| P1 | Loop unroller u32 fix | ❌ **Still open** (root cause documented below) |
| P2 | Screened Coulomb eigensolve | ❌ **Still pending** |
| P2 | HFB shader suite (spherical) | ✅ **Absorbed** (Session 36-37, 5 shaders) |
| P2 | HFB shader suite (deformed) | ✅ **Absorbed** (Session 36-37, 5 shaders) |
| P3 | forge substrate discovery | ✅ **Partial** — `SubstrateCapability` enum absorbed (Session 31d), full `/dev/akida*` probing still in hotSpring metalForge |

**Remaining absorption queue: 2 items (pseudofermion HMC + screened Coulomb)**
**Remaining bug fix: 1 item (loop_unroller u32 — one-line fix documented below)**

---

## Part 1: What ToadStool Absorbed (Confirmed)

### Session 31d — Lattice QCD GPU Primitives

| Module | Upstream Path | Tests |
|--------|-------------|-------|
| Staggered Dirac operator | `ops/lattice/dirac.rs` + `shaders/lattice/dirac_staggered_f64.wgsl` | GPU wrapper + hotSpring 8/8 checks |
| CG lattice kernels | `ops/lattice/cg.rs` + `shaders/lattice/cg_kernels_f64.wgsl` | `WGSL_CG_KERNELS_F64`, `WGSL_COMPLEX_DOT_RE_F64`, `WGSL_AXPY_F64`, `WGSL_XPAY_F64` |
| SubstrateCapability model | `device/substrate.rs` | 12-variant enum, runtime-probed from wgpu features + `/dev/akida*` |

These give toadstool a complete GPU lattice QCD pipeline (Dirac + CG + plaquette +
HMC force), matching the full hotSpring validation suite (8/8 Dirac, 9/9 CG, 3/3
pure GPU QCD workload).

### Sessions 36-37 — Nuclear Structure + ESN

| Module | Upstream Path | Notes |
|--------|-------------|-------|
| 5 deformed HFB shaders | `shaders/science/hfb_deformed/` (energy, potential, wavefunction, hamiltonian, bcs) | Nilsson basis, Skyrme+Coulomb, cylindrical Laplacian |
| 5 spherical HFB shaders | `shaders/science/hfb/` (density, potentials, energy, hamiltonian) | Batched f64 GPU-resident SCF |
| ESN export_weights + import_weights | `esn_v2.rs` | GPU-train → NPU-deploy pipeline now possible |
| Trig precision fixes | `sin_simple`/`cos_simple` → 7-term Taylor + Cody-Waite range reduction | TS-003 resolution |
| pow_f64 fix | f64 `exp_f64` extended for |k| up to 1023 | TS-001 resolution |
| Yukawa cell-list GPU dispatch | Full GPU dispatch with sorted particles and result unsorting | Extends beyond CPU-only |

### Sessions 38-39 — Code Quality Hardening

| Improvement | Impact |
|------------|--------|
| Zero clippy warnings workspace-wide | `manual_div_ceil` fix in Yukawa GPU dispatch; `#[allow(clippy::expect_used)]` on infallible `Drop` |
| Blind `unwrap()` elimination | 3 production `.unwrap()` → descriptive `.expect()` |
| Env-test race fix | 3 `std::env::set_var` tests → direct struct construction |
| 11 new distributed tests | `NetworkLoadBalancer` + `NetworkDistributor` behavioral coverage |
| Dead code sweep (Session 39) | 5 broken pending tests removed, stale comments cleaned |
| Total: 3,847+ tests, 589+ shaders, zero orphans | All quality gates PASS |

---

## Part 2: What's Still Pending

### P1 — Pseudofermion HMC (CPU Module)

| Property | Value |
|----------|-------|
| Source | `barracuda/src/lattice/pseudofermion.rs` (477 lines) |
| Tests | 4 unit + 7/7 validation checks |
| Dependencies | `lattice/{su3,wilson,hmc,dirac,cg}.rs` |
| GPU status | CPU, follows WGSL-ready pattern |
| Suggested upstream | `barracuda::ops::lattice::pseudofermion` |

API:
```rust
pub fn pseudofermion_heatbath(lattice, mass, rng) -> (PseudofermionField, PseudofermionField)
pub fn pseudofermion_action(lattice, phi, chi, mass) -> f64
pub fn pseudofermion_force(lattice, phi, chi, mass) -> Vec<Su3>
pub fn dynamical_hmc_trajectory(lattice, momenta, phi, chi, mass, dt, n_steps) -> (Lattice, f64)
```

**Critical implementation detail**: The force must be `F = TA(U_μ(x) × M)`, not `TA(M)`.
The gauge link multiplication before traceless anti-Hermitian projection ensures the force
lives in the correct tangent space. This was a bug found and fixed during v0.6.4 development.

### P1 — Loop Unroller u32 Bug Fix

**File**: `crates/barracuda/src/shaders/optimizer/loop_unroller.rs`
**Line**: 310-312

```rust
fn substitute_loop_var(line: &str, var_name: &str, iter: u32) -> String {
    let result = line.to_string();
    let replacement = iter.to_string();  // BUG: emits "0", not "0u"
```

**Root cause**: `iter.to_string()` produces bare integer literals like `"0"`, `"1"`, etc.
WGSL type-infers bare ints as `i32`. When the substituted variable appears as an argument
to a function expecting `u32` (e.g., `idx2d(k, j)` → `idx2d(0, j)`), the shader validator
rejects the call as a type mismatch.

**Fix** (single line):
```rust
    let replacement = format!("{iter}u");
```

**Impact**: This fixes the `BatchedEighGpu` single-dispatch panic that hotSpring currently
works around via `std::panic::catch_unwind` in `validate_barracuda_hfb.rs`. With the fix,
the `catch_unwind` wrapper can be removed and single-dispatch eigensolve will work correctly.

**Note**: The `let {var_name} = {iter}u;` binding at line 289 correctly emits the `u`
suffix, but `substitute_loop_var` also does a direct text replacement of the variable in
the body, and that replacement path lacks the suffix. Both paths must agree.

### P2 — Screened Coulomb Eigensolve

| Source | `barracuda/src/physics/screened_coulomb.rs` |
|--------|-----|
| Tests | 23/23 checks (`validate_screened_coulomb`) |
| Physics | Murillo-Weisheit Sturm bisection eigenvalues for Yukawa-screened Coulomb |
| GPU status | CPU-only (Sturm bisection is inherently sequential) |
| Absorption value | Enriches barracuda physics coverage; same Yukawa potential as MD forces |

---

## Part 3: What hotSpring Should Adopt from ToadStool

With toadstool now at Session 39, these upstream capabilities are ready for hotSpring adoption:

| Primitive | Upstream Module | Benefit | Priority |
|-----------|----------------|---------|----------|
| `GemmCachedF64` | `ops::linalg` | HFB SCF: cached workspace eliminates per-iteration GEMM alloc | P1 |
| `BatchIprGpu` | `spectral` | GPU inverse participation ratio for Anderson diagnostics | P2 |
| `NelderMeadGpu` | `optimize` | GPU-parallel L1 parameter search | P2 |
| `UnidirectionalPipeline` | `staging` | Ring-buffer staging for MD streaming dispatch | P2 |
| Upstream `ESN` (v2) | `esn_v2` | Replace local `md/reservoir.rs` — upstream now has weight export | P3 |
| Fixed `sin_simple`/`cos_simple` | `math_f64.wgsl` | Better trig precision via 7-term Taylor + Cody-Waite | Automatic via `ShaderTemplate` |

---

## Part 4: Cross-Reference to Existing Handoffs

This catch-up document supersedes the absorption tracking in:
- `HOTSPRING_V064_TOADSTOOL_HANDOFF_FEB22_2026.md` — Part 5 "Concrete Next Steps" is now stale for Dirac, CG, ESN, HFB items
- `HOTSPRING_V064_DYNAMICAL_QCD_HANDOFF_FEB22_2026.md` — Still current for pseudofermion details
- `specs/BARRACUDA_REQUIREMENTS.md` — Updated in parallel with this document

**Active handoff documents** (ordered by currency):
1. **This document** — `HOTSPRING_V065_TOADSTOOL_SESSION39_CATCHUP_FEB22_2026.md` (latest)
2. `HOTSPRING_V064_DYNAMICAL_QCD_HANDOFF_FEB22_2026.md` — pseudofermion HMC details
3. `HOTSPRING_TOADSTOOL_REWIRE_V4_FEB22_2026.md` — spectral lean reference
4. `CROSS_SPRING_EVOLUTION_FEB22_2026.md` — cross-spring shader map

---

## Codebase Health Comparison

| Metric | hotSpring (v0.6.4) | ToadStool (Session 39) |
|--------|-------------------|----------------------|
| Unit tests | 616 (609 pass + 1 flaky + 6 GPU-ignored) | 3,847+ non-GPU + barracuda targeted |
| Validation suites | 34/34 | 195/195 acceptance + 48/48 wetSpring |
| WGSL shaders | 34 (zero inline) | 589+ (zero orphans) |
| Clippy warnings | 0 | 0 |
| Unsafe blocks | 0 | 55 (all SAFETY documented) |
| Papers reproduced | 22 | N/A (toadstool validates; Springs reproduce) |
| Tolerance constants | 172 | N/A (hotSpring-specific) |
| Coverage | 74.9% region / 83.8% function | common 87%, config 89%, core 79%, server 77% |

---

*This handoff follows the unidirectional pattern: hotSpring → wateringHole → ToadStool.
No inter-Spring imports. All code is AGPL-3.0.*
