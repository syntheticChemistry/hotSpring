# hotSpring v0.6.13 — Cross-Spring Rewiring Handoff

**Date**: 2026-02-25
**From**: hotSpring (biomeGate)
**To**: toadStool team, all springs
**Builds on**: hotSpring v0.6.12 (toadStool S60 absorption)

---

## What Changed in v0.6.13

### 1. GPU-Resident Polyakov Loop (eliminates CPU readback)

Previously, `gpu_polyakov_loop()` read back the entire V×4×18 f64 link buffer
to CPU and computed the Polyakov loop in a triple-nested loop. Now it dispatches
a WGSL compute shader on GPU and reads back only `spatial_vol × 2` f64 values
(Re, Im per spatial site).

**Data transfer reduction**:
- Before: V × 4 × 18 × 8 bytes = 37.7 MB for 32⁴
- After: spatial_vol × 2 × 8 bytes = 524 KB for 32⁴ (72× less)

**New return type**: `(magnitude, phase)` tuple — the phase was previously
hardcoded to 0.0 because computing it required the complex readback that was
too expensive. Now it's computed for free.

**Origin**: Absorbed from toadStool `GpuPolyakovLoop` operator pattern. The
shader uses t-major indexing (matching hotSpring v0.6.11 standardization).

### 2. NVK Allocation Guard

`GpuHmcState::from_lattice()` now estimates total VRAM allocation and warns
(via `eprintln!`) if it exceeds the nouveau driver PTE fault limit (~1.2 GB).
This prevents silent kernel crashes on Titan V/NVK.

**Origin**: Absorbed from toadStool `gpu_hmc_trajectory.rs` allocation guard.

### 3. Naga-Safe SU(3) Math Library

Created `su3_math_f64.wgsl` — a pure-math subset of `su3_f64.wgsl` that strips
the `ptr<storage>` based `su3_load`/`su3_store` functions. These pointer-parameter
functions cause naga validation errors when the file is used as a shader preamble
(naga forbids dynamic indexing of value-type array parameters — requires `var` locals).

**Pattern**: Array function parameters are copied to `var` locals before dynamic
indexing. This is a naga limitation, not a WGSL spec issue.

**For toadStool**: Consider splitting `su3.wgsl` similarly if using it as a
composable preamble for new shaders.

### 4. PRNG Shader Fix

Fixed `su3_random_momenta_f64.wgsl`: the `box_muller_cos` function had a type
mismatch — it cast `theta` to `f32` then passed it to `cos`, which ShaderTemplate
rewrites to `cos_f64` (expects `f64`). Fixed by keeping theta as `f64` throughout.

---

## What hotSpring Provides for toadStool Absorption

### New Shaders

1. **`su3_math_f64.wgsl`** — naga-safe SU(3) pure math (no buffer I/O)
   - Pattern: `var a = a_in;` before dynamic indexing of array params
   - Includes `su3_identity()` (not in current toadStool `su3.wgsl`)

2. **`polyakov_loop_f64.wgsl`** — GPU Polyakov loop with hotSpring binding layout
   - Uses `su3_identity()` from the math preamble
   - Returns complex (Re, Im) per spatial site

### New Findings

1. **GPU Polyakov loop phase** is now computable at no extra cost. Previous runs
   reported `polyakov_phase: 0.0` everywhere — now we get real phase data.

2. **PRNG type safety**: ShaderTemplate transcendental patching can break shaders
   that mix `f32` and `f64` trig calls. All trig in f64 shaders should use `f64`.

### Benchmark Data (v0.6.13)

| Lattice | GPU ms/traj | CPU→GPU Speedup |
|---------|-------------|-----------------|
| 4⁴      | 22.6        | 3.2×            |
| 8⁴      | 30.1        | 38.5×           |
| 8³×16   | 48.1        | 48.6×           |
| 16⁴     | 259.5       | 70.7×           |

Validation: 13/13 total checks passed across streaming, beta scan, and benchmark.

---

## Cross-Spring Evolution Summary

See `experiments/016_CROSS_SPRING_EVOLUTION_MAP.md` for the full shader inventory
showing 164+ shaders across lattice, math, science, bio, ML, and linalg domains,
with provenance annotations from hotSpring, wetSpring, neuralSpring, and airSpring.

Key stat: **28 lattice shaders** originated from hotSpring and are now maintained
in toadStool. The DF64 core streaming discovery (hotSpring Exp 012) propagated to
all springs via toadStool S58-S60.

---

## Remaining Pending Items for toadStool

### From v0.6.12 handoff (still open)

1. **NAK compiler optimization**: Loop unrolling, register allocation, instruction
   scheduling, FMA fusion improvements for Titan V (SM70) performance parity
2. **Neighbor-buffer DF64 plaquette**: hotSpring's `wilson_plaquette_df64.wgsl`
   uses neighbor-buffer indexing — toadStool should evolve to support this
3. **Site-indexing flexibility**: Support both t-major and t-minor in upstream ops

### New from v0.6.13

4. **`su3.wgsl` naga safety**: Consider splitting `su3.wgsl` into math-only
   (composable) and I/O (binding-specific) variants, matching hotSpring's pattern
5. **PRNG type safety**: Audit all shaders for mixed f32/f64 trig calls that
   ShaderTemplate transcendental patching could break
6. **`su3_identity()` addition**: Add to upstream `su3.wgsl` (currently missing)
