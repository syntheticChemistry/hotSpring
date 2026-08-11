# Fossil: hotspring-barracuda v0.6.32-full

## Tag
`hotspring-barracuda-v0.6.32-fossil`

## Date
2026-08-11

## Reason for Fossilization

hotSpring-barracuda grew to 157 binaries, 156 shaders, and ~37 GPU HMC submodules
that substantially duplicate capabilities now living in upstream primals:

- **barraCuda**: geometry (`NeighborMode`, `DiracGpuLayout`), dispatch
  (`ComputeDispatch`), full HMC trajectory (`GpuHmcTrajectory` with Omelyan),
  all lattice WGSL (absorbed_shaders), reduction, CG, multi-shift CG
- **coralReef**: shader compilation (sovereign SPIR-V, DF64 lowering)
- **toadStool**: hardware discovery, compute.dispatch IPC, streaming pipeline

The crate is being rebuilt as a thin **node-atomic composition spring** that
consumes these primals rather than reimplementing them locally.

## Critical Bug Fixed in This Version

The 32⁴ (β=5.90) plaquette was stuck at ≈0.786 instead of ≈0.578.

**Root cause**: RADV NAVI21 (Mesa) has broken `@builtin(num_workgroups)` and
`global_invocation_id.y` for 2D compute dispatch. At 32⁴ volume,
`n_links / 64 = 65536` triggered the 2D split → `(32768, 2, 1)`. With gid.y
always reporting 0, only the first half of links received force/momentum/link
updates. The second half stayed at cold-start identity → P ≈ (1+0.578)/2.

**Fix**: Increased `@workgroup_size` from 64 to 128 in the 6 HMC-critical shaders,
keeping workgroup counts ≤ 32768 for 32⁴ (well under the 65535 limit). Avoids 2D
dispatch entirely.

## Contents

This archive contains the complete `src/` tree at the time of fossilization:
- `src/bin/` — 157 binary entry points
- `src/lattice/` — GPU HMC, CG, observables, streaming pipeline
- `src/lattice/shaders/` — 156 WGSL compute shaders
- `src/gpu/` — GpuF64 wrapper, dispatch, adapter, buffers
- `src/fleet_ember.rs` — toadStool fleet dispatch (deprecated)
- `src/lib.rs` — top-level module structure

## What Replaces This

A new `src/` structured as a node-atomic composition:
```
src/
  lib.rs
  node_atomic/    -- Wraps barraCuda ops + toadStool dispatch
  spring/         -- QCD tolerances, provenance, validation, campaign
  bin/            -- Thin orchestrator binaries
```

## Reference

- Fossilization standard: `infra/fossilRecord/`
- Deduplication handoff: `infra/wateringHole/handoffs/HOTSPRING_PRIMAL_DEDUPLICATION_HANDOFF_AUG07.md`
- Previous per-file fossils: `archive/_fossilized/*.rs`
