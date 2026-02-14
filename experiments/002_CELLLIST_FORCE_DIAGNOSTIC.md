# Experiment 002: Cell-List Force Kernel Investigation

**Date**: February 15, 2026  
**Researcher**: Kevin Eastgate (via Cursor AI assistant)  
**Parent**: Experiment 001 (N-scaling) — cell-list bug discovered during N=10k scaling  
**Binary**: `celllist_diag`  
**Hardware**: RTX 4070 (12 GB GDDR6X), i9-12900K, 32 GB DDR5

---

## 1. Why This Experiment

### The problem

During Experiment 001's N-scaling sweep, the GPU MD simulation switched from
all-pairs to cell-list force computation at N=10,000 (where cells_per_dim >= 5).
The result was **catastrophic energy non-conservation**:

```
N=10000 cell-list equilibration:
  Step    0: T* = 0.006327  (target 0.006329) ← correct
  Step 1000: T* = 0.042557  ← 7× too high
  Step 4999: T* = 0.094711  ← 15× too high, thermostat overwhelmed

N=10000 cell-list production:
  Step    0: E = 3,341  ← initial
  Step 5000: E = 7,025  ← doubled
  Step 10000: E = 12,516 ← tripled
```

Energy grows linearly — the hallmark of systematically incorrect forces.

For comparison, all-pairs at N=5000: **E = 623.6235 → 623.6233** over 30,000 steps
(0.000% drift). The physics is perfect with all-pairs; broken with cell-list.

### What we know

From code inspection of `simulation.rs` and `shaders.rs`:

1. **Force formula**: identical between all-pairs and cell-list kernels
   - Same PBC minimum image (`pbc_delta`)
   - Same Yukawa force magnitude (`prefactor * exp(-κr) * (1+κr) / r²`)
   - Same direction convention (`fx -= force_mag * dx / r`)
   - Same PE half-counting (`0.5 * prefactor * screening / r`)

2. **Cell-list infrastructure**: looks correct on paper
   - Cell assignment: correct modular indexing
   - 3×3×3 neighbor search: no double-counting at 5 cells/dim
   - Sort/unsort: consistent permutation across pos/vel/force arrays
   - GPU buffer sync: read_back waits, write_buffer enqueued before dispatch

3. **Cell-list was NEVER tested at a working scale**
   - The threshold `cells_per_dim >= 5` means cell-list only activates at N≥10,000
   - All previous runs used all-pairs (N≤5,000)
   - We have zero positive evidence the cell-list kernel ever produced correct results

### What this experiment tests

**Hypothesis A**: The cell-list FORCE KERNEL computes different forces than all-pairs
for the same particle positions → bug is in the shader.

**Hypothesis B**: The cell-list forces are correct, but the sort/rebuild cycle during
integration corrupts particle-force associations → bug is in the infrastructure.

**Method**: Run BOTH kernels on identical initial FCC lattice positions. Compare forces
element-by-element. No integration, no thermostat, no sort/rebuild — just one force
computation. If forces differ, it's Hypothesis A (shader bug).

### The scaling question

If cell-list forces are correct, the integration loop must be at fault — perhaps
the every-step GPU↔CPU sort roundtrip introduces accumulating errors. If cell-list
forces are wrong, we need to find the specific shader logic error.

Either way, a working cell-list is the **critical unlock** for N > 20,000:

| N | All-pairs pairs | Est. steps/s (GPU) | Cell-list steps/s (est.) |
|:---:|:---:|:---:|:---:|
| 20,000 | 200M | ~4-8 | ~40-80 |
| 50,000 | 1.25B | ~0.5 | ~30-60 |
| 100,000 | 5B | ~0.05 (days) | ~20-40 |

For HPC GPU (A100, H100) reruns at N=100k+, cell-list is mandatory.

---

## 2. Experimental Design

```
Test N values:
  108    — 3³×4 FCC, 3 cells/dim (cell-list degenerates to all-pairs)
  500    — small, 1 cell/dim → forced to 3 (effectively all-pairs)
  2048   — 3 cells/dim → first real test
  4000   — 4 cells/dim → 64 cells
  8788   — 5 cells/dim → matches N=10k threshold
  10976  — actual N≈10k FCC → the exact failure case

For each N:
  1. Initialize FCC lattice (deterministic)
  2. Compute all-pairs forces (GPU, one dispatch)
  3. Build cell list, sort positions
  4. Compute cell-list forces (GPU, one dispatch)
  5. Unsort cell-list forces back to original order
  6. Compare element-by-element:
     - Total PE
     - Per-particle force RMS difference
     - Maximum force difference
     - Net force (Newton's 3rd law check)
```

### What constitutes a "pass"

- PE relative difference < 1e-6
- Force RMS relative difference < 1e-6
- Net force magnitude < 1e-4 × average force × N

### What constitutes useful diagnostic data

If forces DIFFER, we record:
- Which particles have the worst disagreement
- Their positions and cell assignments
- Whether they're near cell boundaries (edge effect)
- Whether the pattern is systematic (e.g., all particles in certain cells)

---

## 3. Results

### 3.1 Pre-Fix Force Comparison (BROKEN cell_idx)

| N | cells/dim | AP PE | CL PE | PE ratio | All mismatched | Net force |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 108 | 3 | 11.357 | 17.119 | 1.51x | 76/108 | 5.83 |
| 500 | 3 | 52.951 | 113.220 | 2.14x | 500/500 | 20.7 |
| 2048 | 3 | 216.889 | 482.688 | 2.23x | 2048/2048 | 53.7 |
| 4000 | 3 | 423.612 | 919.103 | 2.17x | 4000/4000 | 82.0 |
| 8788 | 5 | 930.675 | 1561.204 | 1.68x | 8788/8788 | 116.4 |
| 10976 | 5 | 1162.391 | 1884.548 | 1.62x | 10976/10976 | 131.8 |

All-pairs net force: ~1e-15 (perfect). Cell-list net force: always in (-,-,-) direction.

### 3.2 Isolation Tests

| Test | Description | Result |
|:---|:---|:---|
| Hybrid | All-pairs loop + cell-list bindings (sorted positions) | **PASS** — PE matches all-pairs |
| V2 (flat loop) | Flat 27-iteration loop instead of nested dz/dy/dx | **FAIL** — same wrong PE |
| V3 (no cutoff) | All-pairs loop, no cutoff, sorted positions | **PASS** |
| V4 (f64 cell data) | Cell-list with f64 arrays instead of u32 | **FAIL** — same wrong PE |
| V5 (CL enum, no cutoff) | Cell-list enumeration but no cutoff check | **FAIL** — same wrong PE |
| V6 (j-trace) | Record all j-indices visited by particle 0 | **76 DUPLICATES found** |

### 3.3 The Smoking Gun: j-trace

Particle 0 in cell (0,0,0) with nx=3. The 27 neighbor offsets should map to 27
unique cells, but many mapped to **cell 0**:

```
neigh  0: off=(-1,-1,-1) → cell  0  ← SHOULD BE cell 26 = (2,2,2)!
neigh  1: off=(+0,-1,-1) → cell  0  ← SHOULD BE cell 18 = (0,2,2)!
neigh  3: off=(-1,+0,-1) → cell  0  ← SHOULD BE cell  6 = (2,0,2)!
neigh  4: off=(+0,+0,-1) → cell  0  ← correct (self)
neigh  9: off=(-1,-1,+0) → cell  0  ← SHOULD BE cell  2 = (2,2,0)!
...
27 neighbor visits → only 8 UNIQUE cells. Cell 0 visited 8 times.
```

**76 duplicate j-indices** (108 total visits, only 32 unique).

### 3.4 Post-Fix Force Comparison (branch-based cell_idx)

| N | cells/dim | AP PE | CL PE | PE match? |
|:---:|:---:|:---:|:---:|:---:|
| 108 | 3 | 11.356937 | 11.356937 | **YES** (diff < 1e-15) |
| 500 | 3 | 52.951495 | 52.951495 | **YES** |
| 2048 | 3 | 216.889322 | 216.889322 | **YES** |
| 4000 | 3 | 423.611957 | 423.611957 | **YES** |
| 8788 | 5 | 930.675470 | 930.675470 | **YES** |
| 10976 | 5 | 1162.391210 | 1162.391210 | **YES** |

**All N values pass after the fix.**

---

## 4. Root Cause

### The bug: WGSL `i32 %` operator for negative operands

The `cell_idx` function used modular arithmetic to wrap negative cell coordinates:

```wgsl
fn cell_idx(cx, cy, cz, nx, ny, nz) -> u32 {
    let wx = ((cx % nx) + nx) % nx;  // ← BUG HERE
    ...
}
```

The WGSL spec defines `%` for signed integers as truncated-division remainder
(same sign as dividend). So `(-1) % 3` should return `-1`. The formula
`((-1) + 3) % 3 = 2` would then correctly wrap to cell 2.

**However**, the Naga WGSL-to-SPIR-V compiler and/or the NVIDIA Vulkan driver
produced incorrect results for this expression. The j-trace shows `(-1,-1,-1)` → cell 0
instead of cell 26. This means `(-1 % 3)` was returning `0` on this GPU instead
of `-1`, or there was an intermediate precision/optimization issue.

The result: **most negative offsets mapped back to cell 0**, causing massive
pair duplication. Particles in cell 0 were processed 8 times instead of once,
inflating PE by 1.5-2.2× and creating enormous non-physical net forces.

### Why all-pairs was unaffected

The all-pairs kernel uses `for j in 0..N` — no modular arithmetic, no `%` operator,
no cell indexing. The `pbc_delta` function uses `round_f64` (not `%`), which
worked correctly.

### Why the bug wasn't caught earlier

The cell-list mode only activates when `cells_per_dim >= 5` (i.e., N ≥ ~10,000).
All previous validation runs used N ≤ 5,000, which always used the all-pairs kernel.
The cell-list kernel was never tested with valid data until the N-scaling experiment.

---

## 5. Fix

Replace modular arithmetic with branch-based wrapping in `cell_idx`:

```wgsl
// BEFORE (broken):
let wx = ((cx % nx) + nx) % nx;

// AFTER (fixed):
var wx = cx;
if (wx < 0)  { wx = wx + nx; }
if (wx >= nx) { wx = wx - nx; }
```

This avoids the `%` operator entirely. Since cell offsets are always in [-1, +1]
and cell coordinates are in [0, n_cells-1], a single branch in each direction
suffices.

**Files changed:**
- `barracuda/src/md/shaders.rs` — `cell_idx()` and `wrap_cell()` in both
  `SHADER_YUKAWA_FORCE_CELLLIST` and `SHADER_YUKAWA_FORCE_CELLLIST_V2`
- `barracuda/src/bin/sarkas_gpu.rs` — re-enabled cell-list mode for cells_per_dim >= 5

**Diagnostic binary:** `barracuda/src/bin/celllist_diag.rs` — comprehensive test
suite that validates all-pairs vs cell-list forces at 6 N values with multiple
isolation tests.

---

## 6. Implications for Scaling

### Cell-list is now unlocked

With the fix verified at all tested N values (108 to 10,976), the cell-list
kernel can be used for large-N simulations:

| N | Method | Est. steps/s | Est. total time (35k steps) |
|:---:|:---:|:---:|:---:|
| 10,000 | all-pairs | ~12 | ~50 min |
| 10,000 | cell-list | ~40-80 | ~8-15 min |
| 20,000 | all-pairs | ~3-4 | ~3-4 hours |
| 20,000 | cell-list | ~30-60 | ~10-20 min |
| 50,000 | cell-list only | ~20-40 | ~15-30 min |
| 100,000 | cell-list only | ~15-30 | ~20-40 min |

### HPC GPU potential

On an A100/H100 with more memory and compute:
- N = 100,000: feasible in minutes
- N = 500,000: feasible in ~1 hour
- N = 1,000,000: feasible in ~few hours

These scales far exceed the Murillo Group's published N=10,000 studies.

### Lesson for barraCUDA upstream

**Never use WGSL `%` operator for negative integer wrapping.** Use branch-based
wrapping instead. This is a Naga/NVIDIA portability issue that may affect other
GPU drivers as well.

---

*Experiment completed: Feb 14, 2026. Bug found and fixed. Cell-list validated
at all N values tested.*

---

## Related Experiments

- **[001_N_SCALING_GPU.md](001_N_SCALING_GPU.md)**: The N-scaling sweep that first triggered this bug (N=10,000 cell-list catastrophic failure). Post-fix results show 0.000% drift at all N.
- **[003_RTX4070_CAPABILITY_PROFILE.md](003_RTX4070_CAPABILITY_PROFILE.md)**: Full 9-case paper-parity run using the cell-list fix from this experiment. κ=2,3 cases use cell-list mode at N=10,000 (4.1× faster than all-pairs).
- **`whitePaper/STUDY.md` §5.7.2**: Narrative write-up of this diagnostic for journal publication.

---

## References

1. Choi, B., Dharuman, G., Murillo, M. S. "High-Frequency Response of Classical Strongly Coupled Plasmas." *Physical Review E* 100, 013206 (2019). — Published DSF study (N=10,000).
2. Silvestri, L. G. et al. "Sarkas: A fast pure-python molecular dynamics suite for plasma physics." *Computer Physics Communications* 272 (2022) 108245. doi:[10.1016/j.cpc.2021.108245](https://doi.org/10.1016/j.cpc.2021.108245) — MD simulation engine.
3. Dense Plasma Properties Database. GitHub: [MurilloGroupMSU/Dense-Plasma-Properties-Database](https://github.com/MurilloGroupMSU/Dense-Plasma-Properties-Database). — Reference DSF S(q,ω) spectra.
4. WGSL Specification: [W3C WebGPU Shading Language](https://www.w3.org/TR/WGSL/). — §16.4 "Integer Division": `e1 % e2` uses truncated-division semantics.
5. wgpu Features: [`SHADER_F64`](https://docs.rs/wgpu/latest/wgpu/struct.Features.html#associatedconstant.SHADER_F64). — Enables f64 types in WGSL via Vulkan backend.
