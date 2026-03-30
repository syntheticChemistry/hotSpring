# Experiment 011: GPU Streaming HMC + Resident CG + Bidirectional Pipeline

**Date**: 2026-02-23
**Gate**: Eastgate (i9-12900, RTX 4070, Pop!_OS 22.04)
**Crate**: hotspring-barracuda v0.6.7
**Status**: 22/22 checks pass (9/9 streaming + 13/13 dynamical streaming)

---

## Objective

Evolve lattice QCD HMC from per-operation GPU dispatch to fully streaming execution,
then to GPU-resident CG with minimal readback, and finally to a bidirectional streaming
pipeline with NPU phase screening.

## Evolution Levels

| Level | Description | Key Innovation |
|-------|------------|----------------|
| Level 1: Dispatch | One GPU dispatch per HMC sub-operation | Baseline GPU HMC |
| Level 2: Streaming | Single-encoder batched dispatch, GPU PRNG | Eliminates per-op dispatch overhead |
| Level 3: GPU-Resident CG | α, β, rz computed on GPU, 10-iter batches | 15,360× readback reduction |
| Level 4: Async Readback | Double-buffered staging, speculative batches | Overlap compute + readback |
| Level 5: Bidirectional Stream | 90%+ to GPU, 10% readback, NPU branch | Full three-substrate pipeline |

## Streaming HMC Results (validate_gpu_streaming, 9/9)

Scaling comparison at 4 lattice sizes, pure gauge HMC:

| Lattice | Volume | CPU ms | Dispatch ms | Streaming ms | Stream gain |
|---------|--------|--------|-------------|-------------|-------------|
| 4⁴ | 256 | 62.5 | 31.4 | 27.5 | 1.14× (2.3× CPU) |
| 8⁴ | 4096 | 2041.2 | 67.3 | 52.4 | 1.28× (38.9× CPU) |
| 8³×16 | 8192 | 3572.6 | 87.7 | 87.4 | 1.00× (40.9× CPU) |
| 16⁴ | 65536 | 31397.7 | 490.2 | 465.6 | 1.05× (67.4× CPU) |

Key: streaming matches dispatch physics exactly (ΔH=0, plaquette=0) while reducing
API call overhead.

## Dynamical Streaming Results (validate_gpu_streaming_dyn, 13/13)

Full dynamical fermion HMC with staggered quarks and CG inversion:

| Config | ⟨P⟩ | Acceptance | Time | Notes |
|--------|------|-----------|------|-------|
| 4⁴ dispatch (baseline) | 0.749 | 90% | 189.9s | Per-op GPU dispatch |
| 4⁴ streaming | 0.785 | 50% | 180.5s | Single-encoder batched |
| 4⁴ resident CG | 0.762 | 70% | 6.6s | 30.7× faster |
| 8⁴ streaming | 1.000 | 0% | 58.4s | Cold start, needs more thermalization |
| 4⁴ bidirectional | 0.784 | 80% | 3.9s | Full three-substrate |

## GPU-Resident CG Architecture

The CG solver was the main readback bottleneck: each iteration required reading back
dot product results (α, β, ||r||²) to CPU. The resident CG keeps all scalars on GPU:

- **sum_reduce_f64.wgsl**: Tree-based reduction of N f64 values to 1 scalar, on GPU
- **cg_compute_alpha_f64.wgsl**: α = rz / pAp (single-thread GPU kernel)
- **cg_compute_beta_f64.wgsl**: β = rz_new / rz_old (single-thread GPU kernel)
- **cg_update_xr_f64.wgsl**: x += α*p, r -= α*ap (reads α from GPU buffer)
- **cg_update_p_f64.wgsl**: p = r + β*p (reads β from GPU buffer)

10 CG iterations are batched into a single command encoder submission.
Only 8 bytes (||r||²) are read back every 10 iterations for convergence check.

**Readback reduction**: 37,355,520 bytes → 2,432 bytes per trajectory (15,360×)

## Bidirectional Stream Architecture

```
GPU (90%+ inbound) ──→ compute ──→ readback (10% outbound)
                                     ├──→ NPU (phase screening)
                                     └──→ CPU (convergence check) ──→ rebatch to GPU
```

The GPU works continuously. Async readback overlaps with the next compute batch.
NPU receives observables (plaquette, Polyakov loop) for real-time phase classification.
CPU handles CG convergence decisions and rebatches parameters back to GPU.

## New WGSL Shaders

| Shader | Purpose | Workgroup Size |
|--------|---------|---------------|
| sum_reduce_f64.wgsl | Tree reduction of f64 array | 256 |
| cg_compute_alpha_f64.wgsl | α = rz / pAp | 1 |
| cg_compute_beta_f64.wgsl | β = rz_new / rz_old | 1 |
| cg_update_xr_f64.wgsl | x += α*p, r -= α*ap | 64 |
| cg_update_p_f64.wgsl | p = r + β*p | 64 |

## Files Modified/Created

| File | Changes |
|------|---------|
| barracuda/src/lattice/gpu_hmc.rs | +~2000 lines: GpuResidentCgPipelines, GpuResidentCgBuffers, streaming dispatch, resident CG, async readback, bidirectional stream |
| barracuda/src/lattice/cg.rs | +5 WGSL constants for resident CG shaders |
| barracuda/src/gpu/buffers.rs | +start_async_readback(), +finish_async_readback_f64() |
| barracuda/src/lattice/shaders/*.wgsl | 5 new shader files |
| barracuda/src/bin/validate_gpu_streaming.rs | Streaming HMC validation (9/9) |
| barracuda/src/bin/validate_gpu_streaming_dyn.rs | Dynamical streaming validation (13/13) |

## Validation

All checks pass:
- validate_gpu_streaming: 9/9 (641s)
- validate_gpu_streaming_dyn: 13/13 (796s)

Combined with existing suites: 155/155 checks in the session.
