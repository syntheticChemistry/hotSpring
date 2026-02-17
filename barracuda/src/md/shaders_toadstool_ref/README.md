# ARCHIVED — ToadStool Barracuda WGSL Shader Reference (Snapshot)

**Status**: ARCHIVED — barracuda canonical shaders have evolved past these snapshots.
Shader .wgsl copies removed; barracuda repo is the single source of truth.

**Original Date**: Feb 14, 2026
**Original Source**: `phase1/toadstool/crates/barracuda/`

These were snapshot copies of toadstool's barracuda MD shaders for divergence
tracking. The divergence history and known-bug documentation below is retained
for fossil record and upstream coordination.

## Toadstool Evolution Status (cb89d054, Feb 14 2026)

### New GPU Operations Available for hotSpring Rewire

| Operation | Module Path | Use Case | Status |
|-----------|------------|----------|--------|
| `BatchedEighGpu` | `barracuda::ops::linalg::BatchedEighGpu` | L2 HFB: batch 791 eigensolves per optimizer step | **Wired** |
| `GenEighGpu` | `barracuda::ops::linalg::GenEighGpu` | L3 deformed HFB (Ax=λBx with overlap) | Ready |
| `SsfGpu` | `barracuda::ops::md::observables::SsfGpu` | MD paper-parity SSF observable | Ready |
| `PppmGpu` | `barracuda::ops::md::electrostatics::PppmGpu` | κ=0 Coulomb (L3 prep, MD extension) | Ready |
| `NelderMeadGpu` | `barracuda::optimize::NelderMeadGpu` | GPU-resident optimizer | Ready |
| `CumsumF64` | `barracuda::ops::CumsumF64` | GPU prefix sum (integration, CDFs) | Ready |

### New WGSL Shaders

| Shader | Path | Purpose |
|--------|------|---------|
| `batched_eigh_f64.wgsl` | `shaders/linalg/` | Batched symmetric eigensolve (Jacobi) |
| `eigh_f64.wgsl` | `shaders/linalg/` | Single-matrix eigensolve |
| `cumsum_f64.wgsl` | `shaders/reduce/` | f64 prefix sum |
| `ssf_f64.wgsl` | `ops/md/observables/` | Static structure factor |
| PPPM suite | `ops/md/electrostatics/` | erfc, Greens, charge spread, force interp |

## Divergence from hotSpring

hotSpring's `shaders.rs` has evolved beyond these toadstool originals:

| Feature | hotSpring | toadstool (this snapshot) |
|---------|-----------|--------------------------|
| f64 transcendentals | **Native builtins** (`sqrt`, `exp`, `round`, `floor`) | Software `math_f64.wgsl` (`sqrt_f64`, `exp_f64`, etc.) |
| Cell-list `cell_idx` | Branch-based wrapping (Naga/NVIDIA safe) | `i32 %` modulo (broken on NVIDIA, see bug alert) |
| math_f64 preamble | **Not needed** | Required for all force/integrator shaders |

## Known Issues in toadstool (reported)

1. **`cell_idx` i32 % bug**: `yukawa_celllist_f64.wgsl` line 50-52 uses
   `((cx % nx) + nx) % nx` which is broken on NVIDIA/Naga/Vulkan for negative
   operands. See `wateringHole/handoffs/TOADSTOOL_CELLLIST_BUG_ALERT.md`.

2. **Software transcendentals**: All shaders still use `sqrt_f64`/`exp_f64` from
   `math_f64.wgsl`. Native builtins are 1.5-2.2x faster (validated on RTX 4070).

## MD Shader Files (snapshot)

- `yukawa_f64.wgsl` — All-pairs Yukawa force (O(N^2))
- `yukawa_celllist_f64.wgsl` — Cell-list Yukawa force (O(N))
- `velocity_verlet_split.wgsl` — VV kick-drift-kick integrator
- `vv_half_kick_f64.wgsl` — VV second half-kick

## Bridge: GpuF64 → WgpuDevice

hotSpring's `GpuF64` connects to toadstool's `WgpuDevice` via:
```rust
use barracuda::device::WgpuDevice;
let wgpu_device = gpu.to_wgpu_device();  // Arc<WgpuDevice>
```
This enables all toadstool GPU ops (BatchedEighGpu, SsfGpu, etc.)
from hotSpring binaries.
