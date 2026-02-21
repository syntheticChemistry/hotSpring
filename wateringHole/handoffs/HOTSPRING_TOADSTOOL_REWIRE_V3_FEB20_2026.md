# hotSpring → ToadStool: Rewire v3 — Post Session 25 Absorption Audit

**Date:** 2026-02-20 (evening)
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Context:** ToadStool Sessions 18-25 (commits `81a6fd4b`..`dc540afd`) reviewed
against hotSpring v0.5.16 (33/33 validation suites, 441 unit tests, 18 papers)

---

## Executive Summary

ToadStool's Feb 20 sprint (Sessions 18-25, 7 commits) resolved **every P0/P1/P2
blocker** from the previous rewire document. The CellListGpu BGL mismatch is
fixed, Complex f64 + SU(3) are first-class GPU primitives, Wilson plaquette +
HMC force + Abelian Higgs have GPU shaders, and — most critically — **GPU FFT
f64 is working** (3 bugs fixed, 14 tests pass, roundtrip to 1e-10). This
unblocks full lattice QCD (Papers 9-12) for the first time.

hotSpring has reviewed the absorption, updated all documentation, and deprecated
the local `GpuCellList` workaround. No code was broken by these changes.

---

## Part 1: Absorption Audit — What ToadStool Resolved

| Item | Commit | Rewire v2 Status | New Status |
|------|--------|-----------------|------------|
| `CellListGpu` prefix-sum BGL | `8fb5d5a0` | **BROKEN** (3 vs 4 bindings) | **FIXED** — 4-binding layout, `scan_pass_a` + `scan_pass_b` split |
| `complex_f64.wgsl` | `8fb5d5a0` | Design only | **Absorbed** — ~70 lines, 14 functions (c64_new through c64_exp) |
| `su3.wgsl` | `8fb5d5a0` | Design only | **Absorbed** — ~110 lines, `@unroll_hint 3` on inner loop |
| `wilson_plaquette_f64.wgsl` | `8fb5d5a0` | Design only | **Absorbed** — 4D, 6 orientations, periodic BC, `ceil(V/64)` dispatch |
| `su3_hmc_force_f64.wgsl` | `8fb5d5a0` | Design only | **Absorbed** — staple sum + algebra projection P(F) = (F−F†)/2 − Tr(...)/6·I |
| `higgs_u1_hmc_f64.wgsl` | `8fb5d5a0` | Design only | **Absorbed** — Wirtinger factor-of-2 baked in, `set_dt()` for step-size tuning |
| `Fft1DF64` | `1ffe8b1a` | Not started | **DONE** — 3 bugs fixed (fossil floor, missing deps, inverse conjugation), 14 GPU tests |
| `Fft3DF64` | `1ffe8b1a` | Not started | **DONE** — 3D FFT for PPPM/Ewald and lattice momentum-space |
| wetSpring bio ops | `cce8fe7c` | N/A | Smith-Waterman, Gillespie, Felsenstein, tree inference absorbed |
| neuralSpring ML ops | `fbedd222` | N/A | TensorSession: matmul, relu, gelu, softmax, layer_norm, attention |
| Deep debt + hygiene | `7c302d7b`, `dc540afd` | N/A | Futures eliminated, async fixes, dead_code cleanup |

---

## Part 2: hotSpring Actions Taken

### Deprecated: Local GpuCellList

The local `GpuCellList` in `md/celllist.rs` (~260 lines + 3 WGSL shaders) was
created as a workaround for the broken upstream CellListGpu. Now that toadstool
has fixed the BGL mismatch, this code is **deprecated** with clear migration
instructions in the source.

**Migration path** (next evolution cycle):
1. Replace `GpuCellList::new()` with `barracuda::ops::md::neighbor::CellListGpu::new()`
2. Replace `GpuCellList::build()` with upstream `build()`
3. Delete local WGSL: `cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl`
4. Delete `struct GpuCellList` and refactor `run_simulation_celllist` to use upstream API

**Not deleted yet** because `run_simulation_celllist` is used by `sarkas_gpu`
and `bench_cpu_gpu_scaling` — the API migration requires mapping toadstool's
`CellListGpu` interface to our simulation loop, which is best done as a focused
refactoring session.

### Documentation Updates

| File | Change |
|------|--------|
| `md/mod.rs` | CellListGpu status: "Not used (broken)" → "Fixed, deprecated local" |
| `md/shaders.rs` | Cell-list shader comment: "remains local" → "deprecated, upstream fixed" |
| `md/celllist.rs` | Added deprecation notice + migration path to `GpuCellList` |
| `specs/PAPER_REVIEW_QUEUE.md` | Tier 3 renamed "Unblocked", FFT resolved, 7-item toadstool catch-up table |
| `specs/BARRACUDA_REQUIREMENTS.md` | FFT/plaquette/HMC/Higgs marked Done, evolution timeline updated |
| `README.md` | ToadStool rewire v3 status reflected |
| `whitePaper/README.md` | Updated with unblocked Tier 3 status |

---

## Part 3: What Was Previously Impossible — Now Possible

With toadstool Sessions 18-25, the following are now achievable that were not
before this sprint:

1. **Full lattice QCD on consumer GPU** — GPU FFT f64 was THE blocker for
   momentum-space fermion propagator computation. With `Fft1DF64`/`Fft3DF64`
   validated to 1e-10 roundtrip, the Dirac CG solver can use GPU FFT for
   staggered fermion inversion. Only GPU Dirac SpMV shader remains.

2. **GPU-accelerated Wilson gauge** — plaquette measurement and HMC force
   computation are now GPU shaders. A 16^4 lattice (~65k sites × 4 links × 18
   f64 per link = ~37 MB) fits entirely in GPU memory. HMC trajectory with
   GPU force + GPU plaquette could run 100-1000× faster than CPU.

3. **GPU-accelerated U(1) Abelian Higgs** — the full HMC half-kick is a GPU
   shader. The Wirtinger factor-of-2 (critical for HMC acceptance) is baked
   into the shader, preventing the most common implementation bug.

4. **Zero-readback MD with upstream cell-list** — once the local GpuCellList
   migration is complete, the entire MD pipeline uses upstream barracuda ops
   with no local workarounds.

---

## Part 4: Current hotSpring State

| Metric | Value |
|--------|-------|
| Unit tests | **441** pass, 5 GPU-ignored |
| Validation suites | **33/33** pass (CPU) |
| Clippy warnings | **0** |
| Papers reproduced | **18** |
| Total compute cost | **~$0.20** |
| Python control scripts | **34** |
| Local GPU cell-list | **Deprecated** (upstream fixed) |
| Shader compilation | All via `ShaderTemplate::for_driver_profile()` |
| ToadStool dependency | `path = "../../phase1/toadstool/crates/barracuda"` |

---

## Part 5: Remaining Evolution Roadmap

### P0 — None (all P0 resolved)

### P1 — GPU Dirac SpMV
Last blocker for full lattice QCD (Papers 9-12). CPU reference in
`lattice/dirac.rs` + `lattice/cg.rs`. The staggered Dirac operator is a
sparse matrix-vector product — same pattern as `spectral::CsrMatrix::spmv()`.

### P1 — GPU SpMV + Lanczos (Kachkovskiy)
CSR SpMV on GPU enables large-scale 3D Anderson model (currently limited to
~8^3 = 512 sites on CPU Lanczos). GPU SpMV + GPU reorthogonalization would
allow 32^3+ lattices for mobility edge studies.

### P1 — Local GpuCellList Migration
Replace local workaround with upstream `CellListGpu` (fixed). Removes ~400
lines of local code + 3 WGSL shaders.

### P2 — ESN `export_weights()` on `esn_v2::ESN`
Needed for GPU-train → NPU-deploy path. Currently only available on
hotSpring's local `EchoStateNetwork`.

### P2 — NPU Reservoir Transport Absorption
ESN shaders, weight export, Akida wiring. See metalForge NPU handoff.

---

## Part 6: Archived Handoffs

This document supersedes the Rewire v2 handoff (now in `archive/`).
All 15 prior handoffs are archived. The consolidated handoff
(`HOTSPRING_V0516_CONSOLIDATED_HANDOFF_FEB20_2026.md`) remains active
as the authoritative primitive catalog.

---

*License: AGPL-3.0-only. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required.*
