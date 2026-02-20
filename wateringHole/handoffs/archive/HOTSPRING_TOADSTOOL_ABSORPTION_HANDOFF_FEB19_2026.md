# hotSpring → ToadStool/BarraCUDA: v0.5.13–v0.5.14 Absorption Handoff

**Date:** 2026-02-19
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**hotSpring version:** v0.5.14 (281 passing tests, 5 GPU-ignored, 0 clippy, 0 doc warnings)
**Supplements:** `HOTSPRING_COMPREHENSIVE_HANDOFF_FEB19_2026.md` (v0.5.12 baseline)

---

## Executive Summary

Two evolution cycles since the last comprehensive handoff:

1. **v0.5.13 — GPU-Resident Cell-List**: Built a working 3-pass GPU cell-list
   locally because ToadStool's `CellListGpu` has a binding mismatch bug.
   Four new WGSL shaders ready for upstream absorption. VACF particle-identity
   bug fixed as a collateral benefit.

2. **v0.5.14 — Daligault Transport Model Evolution**: Discovered that the
   weak-coupling prefactor in the Daligault D* model requires κ-dependent
   correction. Crossover-regime errors dropped from 44–63% to <10%. All 12
   Sarkas N=2000 Green-Kubo D* values now stored as provenance constants.

**Action items for ToadStool**:
- Fix `CellListGpu` prefix_sum binding mismatch (P0)
- Consider absorbing hotSpring's corrected cell-list implementation (P0)
- Fix `cell_idx()` WGSL `i32 %` bug for negative coordinates (P1)

---

## 1. ToadStool Bug: `CellListGpu` Prefix Sum Binding Mismatch

### The Problem

**Location**: `crates/barracuda/src/ops/md/neighbor/cell_list_gpu.rs` + `prefix_sum.wgsl`

ToadStool's `prefix_sum.wgsl` declares **4 bindings**:
```
@binding(0) var<uniform>             params
@binding(1) var<storage, read>       input
@binding(2) var<storage, read_write> output
@binding(3) var<storage, read_write> block_sums
```

But `cell_list_gpu.rs` wires a **3-binding** `BindGroupLayout`:
```rust
// 2 storage + 1 uniform → 3 bindings total
```

This mismatch means `wgpu::Device::create_bind_group()` will either:
- Panic at pipeline creation time (binding count mismatch), or
- Silently produce incorrect results if the driver accepts the layout

### The Fix

hotSpring's corrected `exclusive_prefix_sum.wgsl` uses a consistent 3-binding layout:
```
@binding(0) var<storage, read>       counts    — per-cell particle counts
@binding(1) var<storage, read_write> starts    — exclusive prefix sum output
@binding(2) var<uniform>             params    — { size, pad, pad, pad }
```

Sequential scan (`workgroup_size(1)`) is optimal for Nc < 1000. At typical MD
parameters (box_side=10, rc=2 → 5×5×5 = 125 cells), this runs in sub-microsecond time.

**File**: `barracuda/src/md/shaders/exclusive_prefix_sum.wgsl` (hotSpring local)

---

## 2. ToadStool Bug: `cell_idx()` WGSL `i32 %` for Negative Coordinates

### The Problem

**Location**: `yukawa_celllist_f64.wgsl` lines 49–54

WGSL's `%` operator on `i32` can produce negative results for negative operands.
In `cell_idx()`, particles near cell boundaries can have `cx + dx = -1`, and
`(-1) % mx` returns `-1` (not `mx - 1`). On NVIDIA, this produces wrong cell
indices, leading to missed neighbor pairs and incorrect forces.

### The Fix

Use branch-based wrapping instead of modulo:
```wgsl
fn cell_idx(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32, nz: i32) -> u32 {
    var wx = cx;
    if (wx < 0)  { wx = wx + nx; }
    if (wx >= nx) { wx = wx - nx; }
    // ... same for wy, wz
    return u32(wx + wy * nx + wz * nx * ny);
}
```

**File**: `barracuda/src/md/shaders/yukawa_force_celllist_indirect_f64.wgsl` (hotSpring local)
**Alert document**: `wateringHole/handoffs/TOADSTOOL_CELLLIST_BUG_ALERT.md`

---

## 3. Absorption Candidate: `GpuCellList` (3-Pass GPU Cell-List)

### What It Is

A GPU-resident cell-list builder that eliminates ALL CPU readbacks during
MD cell-list rebuilds. Three compute passes in a single encoder submission:

| Pass | Shader | Purpose | Dispatch |
|------|--------|---------|----------|
| 1 — Bin | `cell_bin_f64.wgsl` | Assign particles to cells, atomically count | `(⌈N/64⌉, 1, 1)` |
| 2 — Scan | `exclusive_prefix_sum.wgsl` | Prefix sum → cell start offsets | `(1, 1, 1)` |
| 3 — Scatter | `cell_scatter.wgsl` | Write particle indices to sorted positions | `(⌈N/64⌉, 1, 1)` |

### Key Design Decisions

1. **Indirect indexing**: Particles stay in original order on GPU. The force shader
   reads neighbors via `sorted_indices[cell_start[c] + k]` instead of physically
   sorting particle arrays. This eliminates the need for gather/scatter of N×3 f64
   position/velocity/force buffers on each rebuild.

2. **Atomic write cursors**: Pass 3 uses `atomicAdd` on per-cell write cursors to
   scatter particle indices without conflicts. Within-cell ordering is arbitrary
   (acceptable for force calculations).

3. **5 intermediate buffers**: `cell_ids[N]`, `cell_counts[Nc]`, `cell_start[Nc]`,
   `write_cursors[Nc]`, `sorted_indices[N]`. All u32.

### Performance

At N=500, cell-list rebuild is ~10 μs (3 GPU dispatches). No CPU-GPU synchronization.
The cell-list can be rebuilt every 20 timesteps (`rebuild_interval=20`) without any
CPU round-trip.

### Files to Absorb

| File | Lines | Purpose |
|------|-------|---------|
| `barracuda/src/md/shaders/cell_bin_f64.wgsl` | 54 | Pass 1: atomic particle binning |
| `barracuda/src/md/shaders/exclusive_prefix_sum.wgsl` | 38 | Pass 2: sequential scan |
| `barracuda/src/md/shaders/cell_scatter.wgsl` | 39 | Pass 3: index scatter |
| `barracuda/src/md/shaders/yukawa_force_celllist_indirect_f64.wgsl` | 120 | Force kernel with indirect neighbor lookup |
| `barracuda/src/md/celllist.rs` (struct `GpuCellList`) | ~150 | Rust orchestrator |

### VACF Particle-Identity Fix (Collateral Benefit)

The indirect indexing approach accidentally fixed a long-standing VACF computation bug.
Previously, physically sorting particle arrays every 20 steps scrambled particle
identities in velocity snapshots, producing incorrect velocity autocorrelation
functions. With indirect indexing, particles keep their original indices throughout
the simulation, so `v(t=0, particle_i)` and `v(t=τ, particle_i)` always refer to
the same physical particle.

---

## 4. Discovery: Daligault D* Model Weak-Coupling Gap

### What We Found

Running the Python `calibrate_daligault_fit.py` against all 12 Sarkas Green-Kubo
D* reference points revealed that the Daligault model with constant `C_w = 5.3`
has 44–63% errors in the crossover regime (low Γ near Γ_x):

| κ | Γ | D*(Sarkas) | D*(C_w=5.3) | Error |
|---|---|-----------|-------------|-------|
| 0 | 10 | 0.1253 | 0.1480 | 18% |
| 1 | 14 | 0.1049 | 0.0595 | **43%** |
| 2 | 31 | 0.0642 | 0.0229 | **64%** |
| 3 | 100 | 0.0290 | 0.0154 | **47%** |

### Root Cause

Yukawa screening suppresses the effective Coulomb logarithm faster than the
classical Landau-Spitzer formula captures. The correction needed grows
exponentially with κ: 4.2× at κ=0, 13× at κ=1, 87× at κ=2, 1325× at κ=3.

### The Fix (v0.5.14)

Replace constant `C_w = 5.3` with:
```
C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²)
```

Coefficients fitted from the 4 crossover-regime Sarkas data points where the
weak-coupling term dominates. After this correction:

| κ | Γ | Error (old) | Error (new) |
|---|---|------------|------------|
| 0 | 10 | 18% | <1% |
| 1 | 14 | 43% | ~1% |
| 2 | 31 | 64% | ~1% |
| 3 | 100 | 47% | ~1% |

Strong-coupling points (Γ >> Γ_x) remain at <6% error, essentially unchanged.

### Relevance to ToadStool

If ToadStool has any Daligault/Landau-Spitzer transport models or plans to add
them, the κ-dependent weak-coupling correction is essential for accuracy in the
crossover regime. The correction applies to all Chapman-Enskog transport
coefficients (D*, η*, λ*) since the Coulomb logarithm enters identically.

---

## 5. Sarkas D* Provenance Data

12 Green-Kubo D_MKS values from Sarkas N=2000 VACF integration are now stored as
`SARKAS_D_MKS_REFERENCE` in `barracuda/src/md/transport.rs`:

```
Physical parameters: Z=1, m=m_p, n=1.62e30 m⁻³
a_ws = 5.282005e-11 m, ω_p = 1.675694e15 rad/s
D_MKS → D* conversion: A2_OMEGA_P = 4.675114e-6 m²/s
```

| κ | Γ | D_MKS [m²/s] | D* (reduced) |
|---|---|-------------|-------------|
| 0 | 10 | 5.856e-7 | 0.12527 |
| 0 | 50 | 6.054e-8 | 0.01295 |
| 0 | 150 | 2.017e-8 | 0.00431 |
| 1 | 14 | 4.903e-7 | 0.10488 |
| 1 | 72 | 5.118e-8 | 0.01095 |
| 1 | 217 | 1.697e-8 | 0.00363 |
| 2 | 31 | 3.001e-7 | 0.06419 |
| 2 | 158 | 3.384e-8 | 0.00724 |
| 2 | 476 | 1.204e-8 | 0.00258 |
| 3 | 100 | 1.356e-7 | 0.02901 |
| 3 | 503 | 1.787e-8 | 0.00382 |
| 3 | 1510 | 7.712e-9 | 0.00165 |

Source: `control/sarkas/simulations/dsf-study/results/all_observables_validation.json`

---

## 6. Transport Grid Expansion

`transport_cases()` in `barracuda/src/md/config.rs` expanded from 12 → 20 points
to include all 9 Sarkas-matched κ>0 DSF study points:

**Added (Sarkas DSF-matched)**:
- κ=1: Γ = 14, 72, 217
- κ=2: Γ = 31, 158, 476
- κ=3: Γ = 503, 1510

New `sarkas_validated_cases()` function returns these 9 points as a filtered subset
for validation binaries that need ground-truth D* comparison.

---

## 7. Complete v0.5.12 → v0.5.14 Changelog

| Version | Key Changes |
|---------|-------------|
| v0.5.12 | `ReduceScalarPipeline` rewire, error bridge (`HotSpringError` ← `BarracudaError`), local `SHADER_SUM_REDUCE` removed |
| v0.5.13 | GPU cell-list (4 WGSL shaders, `GpuCellList` struct), indirect force shader, VACF identity fix, ToadStool binding mismatch discovery |
| v0.5.14 | `C_w(κ)` transport model, 12 Sarkas D_MKS provenance, transport grid 12→20, `validate_stanton_murillo` 2→6 points, 2 new tolerance constants |

**Tests**: 281 passing + 5 GPU-ignored = 286 total
**Clippy**: 0 warnings
**Doc**: 0 warnings

---

## 8. Priority Actions for ToadStool

| Priority | Action | Impact |
|----------|--------|--------|
| **P0** | Fix `prefix_sum.wgsl` binding mismatch in `CellListGpu` | Cell-list currently broken |
| **P0** | Consider absorbing hotSpring's 4 cell-list WGSL shaders as replacement | Working, tested implementation |
| **P1** | Fix `cell_idx()` WGSL `i32 %` bug for negative cell coordinates | Wrong forces near cell boundaries |
| **P1** | Add `GpuCellList` as a barracuda primitive (`ops::md::neighbor`) | Eliminates CPU readback in MD |
| **P2** | Consider `C_w(κ)` weak-coupling correction for transport models | Major accuracy improvement |
| **P2** | Absorb `SARKAS_D_MKS_REFERENCE` provenance data | Ground-truth validation data |
| **P3** | `StatefulPipeline` — hotSpring's MD loops are good candidates | Architectural improvement |

---

## 9. What hotSpring Will Continue Evolving

1. **More transport points**: Run MD at all 20 transport grid configurations and
   validate D* vs Sarkas + Daligault at N=500 (statistical) and N=2000 (reference)
2. **GPU HFB energy integrands**: Wire the existing `batched_hfb_energy_f64.wgsl`
   shader through the pipeline (estimated ~200 lines of refactoring)
3. **Lattice QCD GPU promotion**: Port pure gauge SU(3) plaquette computation to WGSL
4. **Titan V proprietary driver**: Unlock full 6.9 TFLOPS fp64 for fair hardware comparison

ToadStool absorbs what works; hotSpring keeps pushing boundaries on new physics.
The feedback loop continues.
