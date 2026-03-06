# hotSpring v0.6.19 — Precision Stability & Full Debt Resolution Handoff

**Date:** 2026-03-06
**From:** hotSpring v0.6.19 (724 lib tests, 95 binaries, 0 clippy warnings)
**To:** toadStool / barraCuda / coralReef / ALL springs
**License:** AGPL-3.0-only

## Executive Summary

hotSpring completed a full multi-tier precision stability analysis (Experiment
046) covering all cancellation-prone computations across f32, DF64, f64, and
CKKS FHE tiers. Two severe cancellation bugs were found and fixed. The entire
codebase passes clippy (lib + all 95 binaries) with zero warnings, zero unsafe
code, zero TODO/FIXME, and all 71 WGSL shaders are now AGPL-3.0-only.

**Key numbers**: 724 lib tests, 95 binaries, 71 WGSL shaders, 9/9 cancellation
families audited, 10 new stability tests, 3 shader fixes, 59 license fixes.

---

## Part I: Precision Stability Findings (Absorb into barraCuda)

### C1: Plasma W(z) = 1 + z*Z(z) — FIXED (prior session)

Direct asymptotic expansion for |z| >= 4. Avoids catastrophic cancellation.
- f32 naive: GARBAGE (-8e7), stable: correct (-1.7e-2)
- CKKS: 80 mults → 10 mults (8x depth reduction)
- Files: `dielectric.rs`, `dielectric_mermin_f64.wgsl`

### C2: BCS v² = 0.5*(1 - eps/e_qp) — FIXED (this session)

Stable formula: `v² = Δ²/(2·E_qp·(E_qp + |ε|))` when |ε| > |Δ|.
- f32 naive: v²=0 (GARBAGE), stable: correct (2.5e-5)
- Affects nuclear pairing occupations in 2,042-nucleus GPU sweep
- Files: `hfb_common.rs` (CPU), `batched_hfb_density_f64.wgsl`,
  `bcs_bisection_f64.wgsl`, `deformed_density_energy_f64.wgsl` (GPU)
- 5 new stability tests (f32 naive vs stable, symmetry, range)

### C3: Jacobi diff = aqq - app — GUARDED (existing)

1e-14 threshold produces π/4 rotation for degenerate eigenvalues.
Safe at all tiers. No code change needed.

### C4-C9: Documented (no code change needed)

- Flow energy (1-plaq)*6: acceptable at f64 for t₀ physics
- Mermin susceptibility: already mitigated via chi0
- SU(3) Gram-Schmidt: theoretical risk, CGS ok in practice
- Yukawa, ESN leaky, sigmoid: no cancellation concern

### For barraCuda to absorb

1. **Stable BCS v² formula** → ship as `bcs_v2_stable()` in barraCuda's
   physics primitives (same pattern as `plasma_w_stable`)
2. **Cancellation pattern library** → document the 3-step pattern:
   identify cancellation → algebraic rearrangement → physics validation
3. **DF64 enablement** → stable algorithms make DF64 viable for more
   workloads (14 digits with no cancellation amplification)
4. **CKKS FHE** → stable W(z) asymptotic is the ONLY case where stability
   makes FHE infeasible → feasible. Other computations are either already
   FHE-friendly or too complex regardless.

---

## Part II: Codebase Quality

| Metric | Before | After |
|--------|--------|-------|
| Lib tests | 719 | **724** |
| Clippy (lib) | 0 | 0 |
| Clippy (all targets) | 2 warnings | **0** |
| `cargo fmt` | clean | clean |
| `unsafe` blocks | 0 | 0 |
| TODO/FIXME/HACK | 0 | 0 |
| Files > 1000 lines | 0 | 0 |
| Mocks in production | 0 | 0 |
| WGSL license headers | 59 `AGPL-3.0-or-later` | **All 71 `AGPL-3.0-only`** |

### Clippy fixes applied

- `validate_dielectric.rs`: `.cloned()` → `.copied()`
- `validate_streaming_pipeline.rs`: `&mut` → `&` for immutable refs

### Remaining low-priority debt

- ~150 `unwrap`/`expect` in binaries (acceptable for validation pattern)
- 2 `panic!` in lib code (GPU flow RK2 guard, file creation)
- 6 Linux-specific `/proc`/`/sys` paths (standard, not portable)

---

## Part III: Evolution Status

### Paper Queue (specs/PAPER_REVIEW_QUEUE.md)

| Papers | Status |
|--------|--------|
| 43 (Gradient Flow) | GPU-complete, 7/7, 38.5× speedup |
| 44 (BGK Dielectric) | GPU-complete, 12/12 physics checks |
| 45 (Kinetic-Fluid) | CPU-complete, GPU pending (needs BGK moments shader) |
| All others (1-42) | CPU-complete per queue |

### What toadStool/barraCuda should absorb

1. **Stable BCS v² formula** (3 WGSL shaders updated — shader content ready)
2. **Stable W(z) asymptotic** (already in `dielectric_mermin_f64.wgsl`)
3. **Jacobi degenerate guard pattern** (good template for other eigensolvers)
4. **Physics-based GPU validation** (f-sum, DSF positivity, not CPU parity)
5. **Multi-precision test pattern** (f32 naive/stable comparison)

### What coralReef should absorb

1. **FMA control** — `NoContraction` SPIR-V decoration for bit-exact parity
2. **Precision manifest** — detect and report FMA behavior per device
3. **DF64 routing confidence** — stable algorithms make DF64 safe for more
   workloads, expanding the set of computations routable to f32 cores

---

## Part IV: Files Changed This Session

### New files

- `experiments/046_PRECISION_STABILITY_ANALYSIS.md`

### Modified files

| File | Change |
|------|--------|
| `physics/hfb_common.rs` | `bcs_v2_stable()` + routing + 5 tests |
| `physics/shaders/batched_hfb_density_f64.wgsl` | Stable BCS v² branch |
| `physics/shaders/bcs_bisection_f64.wgsl` | Stable BCS v² branch |
| `physics/shaders/deformed_density_energy_f64.wgsl` | Stable BCS v² branch |
| `bin/validate_dielectric.rs` | `.cloned()` → `.copied()` |
| `bin/validate_streaming_pipeline.rs` | `&mut` → `&` immutable refs |
| `specs/PRECISION_STABILITY_SPECIFICATION.md` | Full inventory, BCS, FHE |
| `specs/README.md` | Test count 724 |
| 59 WGSL shaders | License `AGPL-3.0-or-later` → `AGPL-3.0-only` |
| `README.md` | Exp 046 + test counts |
| `CONTROL_EXPERIMENT_STATUS.md` | v0.6.19, 724 tests, S96+ |

### Modified in ecoPrimals/wateringHole/

| File | Change |
|------|--------|
| `GPU_F64_NUMERICAL_STABILITY.md` | BCS case study, inventory ref |
| `NUMERICAL_STABILITY_EVOLUTION_PLAN.md` | Tier 1 COMPLETE, evidence |

---

*hotSpring v0.6.19 — AGPL-3.0-only*
