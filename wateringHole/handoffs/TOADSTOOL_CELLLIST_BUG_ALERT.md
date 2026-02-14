# ALERT: Cell-List `cell_idx` Bug in ToadStool

**Date**: February 15, 2026  
**From**: hotSpring validation team  
**To**: ToadStool / BarraCUDA evolution team  
**Priority**: HIGH — affects correctness of all cell-list force computations

---

## The Bug

**File**: `crates/barracuda/src/ops/md/forces/yukawa_celllist_f64.wgsl`, lines 49-54

```wgsl
fn cell_idx(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32, nz: i32) -> u32 {
    let wx = ((cx % nx) + nx) % nx;   // ← BROKEN on NVIDIA/Naga/Vulkan
    let wy = ((cy % ny) + ny) % ny;
    let wz = ((cz % nz) + nz) % nz;
    return u32(wx + wy * nx + wz * nx * ny);
}
```

The WGSL `i32 %` operator produces **incorrect results for negative operands**
on NVIDIA GPUs via Naga/Vulkan. The expression `((cx % nx) + nx) % nx` should
wrap -1 → nx-1, but instead wraps to 0 on at least NVIDIA RTX 4070 + Naga.

## Impact

- Cell (0,0,0) is visited up to **8 times** instead of once
- 7 of 27 neighbor cells are **never visited**
- Per-particle PE is inflated 1.5-2.2× (tested at N=108 to N=10,976)
- Total energy grows linearly in production → catastrophic energy non-conservation
- Temperature explodes 15× above target during equilibration

## The Fix

Replace modular arithmetic with branch-based wrapping:

```wgsl
fn cell_idx(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32, nz: i32) -> u32 {
    var wx = cx;
    if (wx < 0)  { wx = wx + nx; }
    if (wx >= nx) { wx = wx - nx; }
    var wy = cy;
    if (wy < 0)  { wy = wy + ny; }
    if (wy >= ny) { wy = wy - ny; }
    var wz = cz;
    if (wz < 0)  { wz = wz + nz; }
    if (wz >= nz) { wz = wz - nz; }
    return u32(wx + wy * nx + wz * nx * ny);
}
```

## Verification

Post-fix, cell-list PE matches all-pairs to machine precision (relative diff < 1e-16)
across all tested N values: 108, 500, 2048, 4000, 8788, 10976.

## General Rule

**Never use `i32 %` for negative wrapping in WGSL.** Use branch-based conditionals.
This applies to ALL cell-list, grid, and spatial hashing code in BarraCUDA.

## Diagnostic Tool

hotSpring includes `celllist_diag` (6-phase isolation test) that verifies cell-list
correctness. See `hotSpring/experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md`.

## Also Found: Native f64 Builtins Available

`sqrt(f64)`, `exp(f64)`, `inverseSqrt(f64)`, `log(f64)`, `abs(f64)`, `floor(f64)`,
`ceil(f64)`, `round(f64)` all compile and work correctly on f64 types via Naga/wgpu.

Performance on RTX 4070 (1M elements):
- Native sqrt: **1.5× faster** than math_f64 `sqrt_f64`
- Native exp: **2.2× faster** than math_f64 `exp_f64`

The force kernels should use native builtins instead of `math_f64.wgsl` software
implementations for transcendentals. This directly improves MD simulation throughput.
