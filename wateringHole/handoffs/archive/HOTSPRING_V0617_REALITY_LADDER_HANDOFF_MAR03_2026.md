SPDX-License-Identifier: AGPL-3.0-only

# hotSpring → toadStool: Reality Ladder + New Absorption Targets

**Date:** 2026-03-03
**From:** hotSpring v0.6.17+
**To:** toadStool/barracuda team
**Covers:** Exp 032 (8^4 production) + Exp 033 (Reality Ladder Rung 0) + Rungs 1-3 code
**Supersedes:** V0617 S80 Absorption Handoff (Mar 02)
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring completed two major experiments and four code infrastructure rungs since
the last handoff. 663 tests pass, 0 warnings. The GPU path now supports multi-field
pseudofermions (Nf=8, 12, ...), Hasenbusch mass preconditioning, and RHMC
infrastructure for Nf=2 and Nf=2+1 — the path to physical QCD.

**New absorption targets for toadStool** (highest impact first):
1. **RHMC module** — rational approximation with Remez+pole optimization, multi-shift CG
2. **GPU Hasenbusch** — multi-scale leapfrog for light quarks
3. **Multi-field GPU HMC** — `Vec<phi_bufs>` pattern for arbitrary Nf
4. **Omelyan integrator** — already in toadStool, hotSpring should lean on it

---

## Part 1: Experiment Results

### Exp 032: 8^4 Volume Scaling (135 trajectories)

First 8^4 dynamical fermion run on consumer GPU (RTX 3090, NVK).
NPU prioritized beta=5.69 (transition) first, made 127 NPU calls.

| beta | <P> | sigma | acc% | <CG> | phase | time |
|------|-----|-------|------|------|-------|------|
| 5.50 | 0.529 | 0.016 | 80% | 40,110 | confined | 2822s |
| 5.69 | 0.561 | 0.013 | 67% | 40,132 | transition | 3035s |
| 6.00 | 0.589 | 0.012 | 80% | 40,111 | deconfined | 2776s |

**Key learning:** dt=0.008 on 8^4 gives 67-80% acceptance. delta_H ~ +0.3
indicates the integrator step could be smaller. ~80s/trajectory at 8^4.

### Exp 033: Reality Ladder Rung 0 (479 trajectories)

Mass x beta x volume scan at Nf=4. Five masses (0.5, 0.2, 0.1, 0.05, 0.01)
across three volumes (2^4, 4^4, 6^4).

**Key findings:**
- CG cost scales as ~1/m: 13k (m=0.5) → 25k (m=0.05) → broken (m=0.01)
- Mass=0.01 at 4^4: 0% acceptance, delta_H ~ 10^9 (CG can't converge)
- This proves the need for Hasenbusch preconditioning below m=0.05
- 6^4 at m=0.05 hits 40% acceptance — volume amplifies the difficulty
- Full results: `whitePaper/baseCamp/reality_ladder_rung0.md`

---

## Part 2: New Code for toadStool Absorption

### Tier 0 — Highest Impact (new infrastructure)

| Module | Lines | Tests | Description |
|--------|-------|-------|-------------|
| `lattice/rhmc.rs` | ~600 | 3/3 | **RHMC infrastructure**: `RationalApproximation` with Remez exchange + pole optimization via golden section search, `multi_shift_cg_solve` (CPU), `RhmcFermionConfig`/`RhmcConfig` with `nf2()` and `nf2p1()` constructors, `rhmc_heatbath`/`rhmc_fermion_action`/`rhmc_fermion_force` |
| `lattice/gpu_hmc/hasenbusch.rs` | ~350 | — | **GPU Hasenbusch**: `GpuHasenbuschBuffers`, `GpuHasenbuschConfig`, multi-scale leapfrog with `gpu_heavy_action`/`gpu_ratio_action`, `gpu_heavy_force_kick`/`gpu_ratio_force_kick`, bilinear force decomposition |

**RHMC rational approximation details:**
- `RationalApproximation::generate(power, n_poles, lambda_min, lambda_max)`
- Uses proper Remez exchange with equioscillation variable E
- Coordinate descent on pole positions (golden section search in log space)
- 8-pole approximation to x^{1/2} achieves <5% relative error on [0.01, 64]
- Prebuilt: `fourth_root_8pole()`, `inv_fourth_root_8pole()`, `sqrt_8pole()`, `inv_sqrt_8pole()`

### Tier 1 — Ready Now (validated, extends previous handoff)

| Module | Lines | Description |
|--------|-------|-------------|
| `lattice/gpu_hmc/dynamical.rs` | Δ+100 | **Multi-field support**: `phi_bufs: Vec<Buffer>`, `from_lattice_multi()`, `gpu_fermion_action_all()`, `gpu_total_force_dispatch()` loops over fields |
| `lattice/gpu_hmc/resident_cg_brain.rs` | Δ+80 | Brain-mode multi-field: `gpu_fermion_action_brain_all()`, field-loop in force dispatch |
| `lattice/gpu_hmc/resident_cg.rs` | Δ+80 | Resident CG multi-field: `gpu_fermion_action_resident_all()` |
| `lattice/gpu_hmc/streaming.rs` | Δ+50 | Streaming multi-field: GPU PRNG heatbath per field |
| `bin/production_dynamical_mixed.rs` | Δ+30 | CLI: `--n-fields=N`, `--cg-max-iter=N`, Nf in output JSONL |

### Tier 2 — Learnings (patterns, not code)

| Pattern | Description |
|---------|-------------|
| Mass-dependent NPU steering | NPU proxy features already include `mass` — the 033 data teaches mass→CG cost mapping |
| Algorithmic boundary detection | 0% acceptance at m=0.01 is a clear NPU training signal for "needs Hasenbusch" |
| Volume-CG scaling | CG cost doubles from 4^4 → 6^4 at same mass; useful for NPU resource prediction |

---

## Part 3: What hotSpring Should Lean On from toadStool

toadStool S80 has evolved several modules that hotSpring currently duplicates:

| toadStool Module | hotSpring Equivalent | Action |
|------------------|---------------------|--------|
| `ops::lattice::omelyan_integrator` | Manual leapfrog in `gpu_hmc/dynamical.rs` | **Adopt** — toadStool's Omelyan is more sophisticated |
| `ops::lattice::gpu_kinetic_energy` | Inline shader dispatch in `gpu_hmc/` | **Adopt** — cleaner separation |
| `ops::lattice::gpu_polyakov` | `observables.rs` Polyakov calculation | **Adopt** — standardized |
| `ops::lattice::gpu_lattice_init` | Manual buffer init in `from_lattice()` | **Consider** — may simplify state setup |
| `ops::lattice::gpu_cg_resident` | `resident_cg.rs` (hotSpring version) | **Compare** — hotSpring's version has brain interrupt + async readback |

**Note:** hotSpring's `resident_cg_brain.rs` (NPU-interleaved CG) and
`resident_cg_async.rs` (latency-adaptive check intervals) are beyond toadStool's
current `gpu_cg_resident`. These should flow TO toadStool, not be replaced.

---

## Part 4: Recommended toadStool Absorption Priority

1. **RHMC** (lattice/rhmc.rs) — Unlocks Nf=2, 2+1 for ALL springs
2. **Multi-shift CG** (in rhmc.rs) — Essential for RHMC, useful for Hasenbusch
3. **Hasenbusch GPU** (gpu_hmc/hasenbusch.rs) — Unlocks light quarks for all springs
4. **Multi-field pattern** (Vec<phi_bufs> in dynamical.rs) — Simple, high-value
5. **Brain-mode CG** (resident_cg_brain.rs) — NPU-interleaved solving pattern
6. **Async CG readback** (resident_cg_async.rs) — Latency-adaptive GPU→CPU

---

## Files Changed Since Last Handoff

```
barracuda/src/lattice/rhmc.rs                        (NEW — 600 lines)
barracuda/src/lattice/gpu_hmc/hasenbusch.rs          (NEW — 350 lines)
barracuda/src/lattice/gpu_hmc/dynamical.rs           (MODIFIED — multi-field)
barracuda/src/lattice/gpu_hmc/streaming.rs           (MODIFIED — multi-field)
barracuda/src/lattice/gpu_hmc/resident_cg.rs         (MODIFIED — multi-field)
barracuda/src/lattice/gpu_hmc/resident_cg_brain.rs   (MODIFIED — multi-field)
barracuda/src/lattice/gpu_hmc/mod.rs                 (MODIFIED — new exports)
barracuda/src/lattice/mod.rs                         (MODIFIED — pub mod rhmc)
barracuda/src/bin/production_dynamical_mixed.rs       (MODIFIED — CLI flags)
barracuda/src/bin/validate_gpu_dynamical_hmc.rs       (MODIFIED — phi_bufs[0])
experiments/033_reality_ladder_rung0.sh               (NEW)
whitePaper/baseCamp/reality_ladder_rung0.md           (NEW)
results/exp032_8x8_production.jsonl                   (NEW — 135 traj)
results/exp033_*.jsonl                                (NEW — 479 traj)
```
