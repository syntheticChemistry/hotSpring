# ToadStool Barracuda WGSL Shader Reference (Snapshot)

**Date**: Feb 2026
**Source**: `phase1/toadstool/crates/barracuda/src/ops/md/`

These are **snapshot copies** of toadstool's barracuda MD shaders for reference
and divergence tracking. Main evolution happens in the toadstool repo.

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

## Files

- `yukawa_f64.wgsl` — All-pairs Yukawa force (O(N^2))
- `yukawa_celllist_f64.wgsl` — Cell-list Yukawa force (O(N))
- `velocity_verlet_split.wgsl` — VV kick-drift-kick integrator
- `vv_half_kick_f64.wgsl` — VV second half-kick
