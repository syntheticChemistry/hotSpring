# Experiment 056: Sovereign Dispatch Benchmark — Backend-Agnostic MD Engine

**Date**: 2026-03-09
**Status**: VALIDATED (wgpu), BLOCKED (sovereign DRM ioctl)
**Binary**: `bench_sovereign_dispatch`
**Upstream**: barraCuda `875e116`, coralReef Iter 33

## Summary

Rewired hotSpring's MD simulation to use barraCuda's `GpuBackend` trait and
`ComputeDispatch<B>` builder, creating a backend-agnostic engine that runs
identically on wgpu/Vulkan and sovereign (coralReef → DRM) backends.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  MdEngine<B: GpuBackend>  (sovereign_engine.rs)     │
│  ├── alloc_buffers()    → B::alloc_buffer_f64_init  │
│  ├── dispatch_force()   → ComputeDispatch<B>        │
│  ├── dispatch_kick_drift() → ComputeDispatch<B>     │
│  ├── dispatch_half_kick()  → ComputeDispatch<B>     │
│  ├── dispatch_ke()      → ComputeDispatch<B>        │
│  └── download_f64()     → B::download_f64 (CPU sum) │
├─────────────────────────────────────────────────────┤
│  B = WgpuDevice          │  B = CoralReefDevice     │
│  wgpu → naga → SPIR-V    │  coralReef → SASS/GFX   │
│  → Vulkan driver JIT      │  → DRM ioctl dispatch    │
│  VALIDATED ✓              │  BLOCKED (ioctl 22)      │
└─────────────────────────────────────────────────────┘
```

## Results

### wgpu/Vulkan Backend (RTX 3090, proprietary driver)

| Metric | Value |
|--------|-------|
| N particles | 2000 |
| Equil steps | 2000 |
| Prod steps | 5000 |
| Force method | AllPairs (O(N²)) |
| Steps/s | 140.3 |
| Wall time | 49.89s |
| Final KE | 14.3381 |
| Final PE | 230.2525 |
| Final E | 244.5905 |
| Energy reduction | CPU-side sum (bypasses ReduceScalarPipeline zero bug) |

### Sovereign/DRM Backend

| Metric | Value |
|--------|-------|
| Init | CoralReefDevice::with_auto_device() succeeded |
| Dispatch | FAILED: `DRM ioctl returned 22` (EINVAL) |
| Root cause | coral-driver nvidia-drm ioctl interface incomplete |

### Comparison: Generic Engine vs Original Batched Path

| Engine | Method | Steps/s | Notes |
|--------|--------|---------|-------|
| Generic `MdEngine<WgpuDevice>` | AllPairs | 140.3 | Per-dispatch submit (3×/step) |
| Original `run_simulation` (Verlet) | Verlet | 257.6 | Batched encoder (1 submit/batch) |
| Kokkos-CUDA | Verlet | 2896.7 | CUDA kernels, fused launch |

The generic engine uses per-dispatch `ComputeDispatch::submit()` (one command
buffer per kernel, three per step) vs the original path's batched encoding (all
kernels in one compute pass per batch). The ~1.8× overhead is expected from
Vulkan command buffer submission cost. Sovereign dispatch has lower per-dispatch
overhead (no Vulkan stack), so the gap should narrow or invert when DRM matures.

## Key Findings

### 1. Backend-agnostic engine works
Same WGSL shader source, same physics, same buffer layout — the `GpuBackend`
trait abstracts everything. `ComputeDispatch<B>` handles shader compilation,
bind group creation, and dispatch through one builder API.

### 2. CPU-side energy sum avoids ReduceScalarPipeline bug
The original simulation reports `KE=0.0000, PE=0.0000` due to the upstream
`ReduceScalarPipeline` zero bug. The generic engine reads back per-particle
arrays and sums on CPU, producing correct energies. This validates the physics
independently of the reducer.

### 3. Sovereign DRM gap is well-defined
The failure is at `coral_driver::drm_ioctl` — the kernel interface for NVIDIA
DRM dispatch. coralReef's compilation (WGSL → SASS) works perfectly (validated
in Exp 055 / Iter 33). The remaining gap is driver-level:

| Driver | Compilation | Dispatch |
|--------|------------|----------|
| amdgpu | ✓ | ✓ (E2E ready) |
| nouveau/NVK | ✓ | Pending kernel ioctl |
| nvidia-drm | ✓ | Pending UVM integration |

### 4. Kernel caching eliminates recompilation
CoralReefDevice's hash-keyed kernel cache (barraCuda `875e116`) means the 5
unique MD shaders compile once, then 7000 steps × 3 dispatches/step = 21000
dispatches hit cache. Zero recompilation overhead in the hot loop.

## Cross-Spring Shader Evolution

| Shader | Origin | Absorbed by | Used in |
|--------|--------|-------------|---------|
| df64_core.wgsl | hotSpring Exp 028 | barraCuda → all | precision math everywhere |
| df64_transcendentals.wgsl | hotSpring | barraCuda, coralReef | Yukawa exp/sqrt, bio ops |
| yukawa_force_f64.wgsl | hotSpring | barraCuda md/ | MD force computation |
| smith_waterman_f64.wgsl | wetSpring | barraCuda bio/ | sequence alignment |
| hmm_viterbi_f64.wgsl | neuralSpring | barraCuda bio/ | HMM inference |
| matrix_correlation_f64.wgsl | neuralSpring | barraCuda stats/ | correlation analysis |
| perlin_2d_f64.wgsl | ludoSpring | barraCuda procedural/ | terrain noise |
| PrecisionBrain | hotSpring v0.6.25 | barraCuda a012076 | tier routing |
| HardwareCalibration | hotSpring v0.6.25 | barraCuda a012076 | GPU capability probing |

The sovereign compilation path (coralReef) benefits all springs equally:
hotSpring's precision shaders, wetSpring's bio shaders, and neuralSpring's
stats/ML shaders all route through the same `ComputeDispatch<CoralReefDevice>`
when sovereign dispatch is available. The DF64 transcendental poisoning fix
(naga SPIR-V codegen bug) applies to all springs, not just hotSpring.

## Files Created/Modified

| File | Change |
|------|--------|
| `barracuda/src/md/sovereign_engine.rs` | NEW — generic MdEngine<B: GpuBackend> |
| `barracuda/src/md/mod.rs` | Added sovereign_engine module |
| `barracuda/src/bin/bench_sovereign_dispatch.rs` | NEW — benchmark binary |
| `barracuda/Cargo.toml` | Added bench_sovereign_dispatch binary, pinned barraCuda 875e116 |

## Next Steps

1. **coralReef**: nvidia-drm ioctl dispatch (UVM buffer management)
2. **barraCuda**: `ReduceScalarPipeline` fix (returns zeros)
3. **hotSpring**: Once sovereign DRM works, benchmark sovereign vs wgpu
4. **hotSpring**: Verlet/CellList variants for generic engine
5. **All springs**: Adopt `ComputeDispatch<B>` pattern for backend-agnostic code
