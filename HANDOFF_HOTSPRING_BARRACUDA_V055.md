# Handoff: hotSpring BarraCUDA v0.5.5 — Code Quality Hardening

**Date:** February 16, 2026 (evening)
**From:** hotSpring validation crate
**To:** ToadStool/BarraCUDA core team
**Builds on:** `HANDOFF_HOTSPRING_TO_TOADSTOOL_FEB_16_2026.md` (195/195 checks)
**License:** AGPL-3.0-only

---

## What Changed Since the Feb 16 Morning Handoff

The morning handoff documented 195/195 checks and two bugs. This handoff
documents the **code quality hardening pass** that followed — no new physics,
no new checks, but significantly cleaner infrastructure for future evolution.

### At a Glance

| Metric | Before (v0.5.4) | After (v0.5.5) |
|--------|:---:|:---:|
| Unit tests | 158 | **182** (+24) |
| `.rs` files | 45 | **51** (+6 new modules) |
| Binaries | 18 | **20** (+2 validators) |
| Line coverage | 33% | **39%** |
| Function coverage | 50% | **57%** |
| Inline tolerance magic numbers | 30+ | **0** |
| Inline `sigma_theo` expressions | 19 | **0** (centralized) |
| Inline WGSL in `celllist_diag.rs` | 640 lines | **0** (8 files extracted) |
| SPDX headers | 45/45 | **51/51** |
| Clippy warnings | 0 | **0** |
| `unsafe` blocks | 0 | **0** |
| `unwrap()` in lib code | 0 | **0** |

---

## Part 1: Tolerance Consolidation

All validation binaries now source thresholds from `tolerances.rs` instead
of inline numeric literals. This makes tolerance provenance auditable and
changes traceable.

### New Constants Added (with physical justification)

| Constant | Value | Justification |
|----------|-------|---------------|
| `GPU_EIGENSOLVE_REL` | 1e-4 | Jacobi rotation on GPU, f64 accumulation |
| `GPU_EIGENVECTOR_ORTHO` | 1e-4 | Max \|Q^T Q - I\|_ij for GPU eigenvectors |
| `BCS_PARTICLE_NUMBER_ABS` | 1e-6 | BCS bisection particle number constraint |
| `BCS_CHEMICAL_POTENTIAL_REL` | 1e-6 | BCS chemical potential vs CPU Brent |
| `PPPM_NEWTON_3RD_ABS` | 1e-6 | Newton's 3rd law residual for PPPM forces |
| `PPPM_MADELUNG_REL` | 1e-4 | Madelung constant vs analytical (NaCl) |
| `HFB_RUST_VS_PYTHON_REL` | 0.12 | Rust L2 vs Python L2 (method differences) |
| `HFB_RUST_VS_EXP_REL` | 0.10 | Rust L2 vs AME2020 experiment |
| `MD_ABSOLUTE_FLOOR` | 1e-10 | Absolute error floor for near-zero forces |
| `NEWTON_3RD_LAW_ABS` | 1e-3 | Newton's 3rd law for CPU MD forces |
| `MD_EQUILIBRIUM_FORCE_ABS` | 1e-6 | Equilibrium force magnitude upper bound |
| `BFGS_TOLERANCE` | 1e-4 | Position convergence (corrected from 0.1) |

### Binaries Wired

- `validate_md.rs` — 11 inline constants → `tolerances::*`
- `validate_barracuda_hfb.rs` — 10 inline constants → `tolerances::*`
- `validate_pppm.rs` — Newton's 3rd law → `tolerances::PPPM_NEWTON_3RD_ABS`
- `verify_hfb.rs` — Rust-vs-exp → `tolerances::HFB_RUST_VS_EXP_REL`
- `validate_optimizers.rs` — BFGS checks → `tolerances::BFGS_TOLERANCE`
- 7 nuclear EOS binaries — 19× `sigma_theo` → `tolerances::sigma_theo(b_exp)`

---

## Part 2: Shared Infrastructure

### `data::EosContext` + `data::load_eos_context()`

All 7 nuclear EOS binaries duplicated this pattern:

```rust
let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .parent().unwrap()
    .join("control/surrogate/nuclear-eos");
let exp_data = data::load_nuclei(&base, ...);
let bounds = data::load_bounds(&base.join("wrapper/skyrme_bounds.json"));
```

Now one call: `let ctx = data::load_eos_context();` → `ctx.base`, `ctx.exp_data`, `ctx.bounds`.

### `data::chi2_per_datum()`

Shared χ² computation using `tolerances::sigma_theo`. Tested with known-value
and perfect-fit unit tests.

---

## Part 3: WGSL Extraction

8 inline WGSL diagnostic shaders extracted from `celllist_diag.rs` into
`src/bin/shaders/celllist_diag/*.wgsl`:

| Shader | Purpose |
|--------|---------|
| `celllist_v4_f64.wgsl` | Cell-list v4 force kernel |
| `celllist_v5_pe_f64.wgsl` | Cell-list v5 potential energy |
| `celllist_v6_debug_f64.wgsl` | Cell-list v6 debug output |
| `allpairs_nocutoff_f64.wgsl` | All-pairs reference (no cutoff) |
| `hybrid_allpairs_f64.wgsl` | Hybrid all-pairs kernel |
| `paircount_allpairs_f64.wgsl` | Pair count (all-pairs) |
| `paircount_celllist_f64.wgsl` | Pair count (cell-list) |
| `verify_u32_f64.wgsl` | u32/f64 conversion verification |

File reduced from 1672 → 1124 lines. All shaders use `include_str!` with
runtime preamble patching via `format!("{}\n\n{}", md_math, include_str!(...))`.

---

## Part 4: Test Coverage Expansion

### New Tests (24 total)

| Module | Tests Added | What They Cover |
|--------|:-----------:|-----------------|
| `hfb.rs` | 6 | Hamiltonian symmetry, BCS particle number, quantum numbers, wavefunction accessors, lj_same flags, adaptive scaling |
| `hfb_gpu.rs` | 5 | BatchedL2Result fields, nucleus partitioning, adaptive basis size |
| `hfb_gpu_resident.rs` | 5 | GpuResidentL2Result, PotentialDimsUniform layout, HamiltonianDimsUniform, nucleus grouping, density init |
| `bcs_gpu.rs` | 6 | BisectionParams tolerance round-trip, degeneracy flag, layout size, param packing |
| `data.rs` | 2 | chi2_per_datum perfect fit, chi2_per_datum known deviation |

### Coverage by Category

| Category | Line Coverage | Notes |
|----------|:---:|-------|
| CPU-testable modules | **90-100%** | error, tolerances, constants, semf, nuclear_matter, hfb_common, provenance, shaders |
| Data, validation, prescreen | **81-91%** | |
| HFB spherical | **53%** | Up from 25%; 6 new tests |
| GPU-dependent modules | **9-25%** | Requires GPU adapter; CPU-side logic tested |

---

## Part 5: Evolution Readiness Update

### GPU Energy Pipeline — Priority #1

The `batched_hfb_energy_f64.wgsl` shader already exists and computes
per-grid-point energy integrands. Currently the CPU reads these back and
sums with `trapz`. The next step is:

1. Create a compute pipeline for `batched_hfb_energy_f64.wgsl`
2. Use `barracuda::ops::reduce::SumReduceF64` to reduce integrands → scalar
3. Read back only the scalar total energy (not the full array)

This eliminates `compute_energy_with_v2` and its CPU `trapz` calls.
Estimated: ~100 lines of pipeline wiring code.

### Updated Promotion Priority

1. **GPU energy integrands + SumReduceF64** — shader exists, needs wiring
2. **BCS on GPU** — move occupations + density to shader
3. **SpinOrbitGpu** — wire ToadStool's existing op into HFB
4. **Deformed HFB GPU** — wire existing 7 deformed shaders
5. **Nuclear matter** — CPU bisection is fast enough (low priority)

---

## Part 6: Dead Code + Clippy Cleanup

- 6 unused struct fields renamed with `_` prefix (HFB deformed)
- 3 GPU-reserved functions documented with evolution comments
- 25+ clippy warnings fixed (`cloned` → `copied`, format string interpolation,
  unnecessary borrows)
- SPDX AGPL-3.0-only headers added to 4 archive files (51/51 complete)
- `HFB_TEST_NUCLEI` provenance enhanced: exact Python command, commit
  `fd908c41`, date 2026-01-15, environment spec

---

## Part 7: What ToadStool Should Know

### barracuda Primitives We Use

No changes to the BarraCUDA API surface. All existing imports from the
morning handoff remain valid. No new barracuda dependencies added.

### barracuda Primitives We Should Use Next

| Primitive | Where | Current CPU Fallback |
|-----------|-------|---------------------|
| `SumReduceF64` | HFB energy integrands | `trapz` on CPU |
| `SpinOrbitGpu` | HFB Hamiltonian | CPU spin-orbit matrix elements |
| `FusedMapReduceF64` | MD post-processing | CPU observable reduction |

### Files > 1000 Lines (Documented Deviation)

8 files exceed the wateringHole 1000-line guideline:

| File | Lines | Rationale |
|------|:-----:|-----------|
| `hfb.rs` | 1286 | Physics-coherent solver; splitting fragments SCF logic |
| `hfb_deformed.rs` | 1285 | Same — deformed SCF loop is one coherent algorithm |
| `nuclear_eos_l1_ref.rs` | 1250 | Self-contained L1 optimization pipeline |
| `nuclear_eos_l2_hetero.rs` | 1207 | Self-contained L2 heterogeneous cascade |
| `hfb_deformed_gpu.rs` | 1198 | GPU+CPU hybrid solver, coherent pipeline |
| `celllist_diag.rs` | 1124 | Reduced from 1672 via WGSL extraction |
| `hfb_gpu_resident.rs` | 1093 | GPU-resident SCF, coherent pipeline |
| `nuclear_eos_gpu.rs` | 1039 | Self-contained GPU validation + profiling |

The 4 HFB library files are physics solvers where splitting would fragment
the SCF iteration loop. This is an accepted trade-off for solver coherence.

---

## Summary

v0.5.5 is a quality-of-life release — no new physics, no new checks, but
the codebase is now significantly cleaner for the next evolution step.
The immediate next target is wiring `SumReduceF64` into the HFB energy
pipeline (the shader already exists), which eliminates the last major CPU
bottleneck in the GPU-resident SCF loop.

```
Quality gates (all pass):
  cargo fmt --check     → 0 diffs
  cargo clippy          → 0 warnings
  cargo test --lib      → 182 passed, 0 failed, 5 ignored
  cargo doc --no-deps   → 0 warnings
  cargo llvm-cov        → 39% line / 57% function
```

---

*February 16, 2026 (evening) — Code quality hardening. 182 tests, zero
inline magic numbers, 8 WGSL shaders extracted, shared EOS infrastructure,
full tolerance provenance. Next: wire SumReduceF64 for GPU-resident energy.*
