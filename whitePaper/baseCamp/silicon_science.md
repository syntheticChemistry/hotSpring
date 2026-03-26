# Silicon Science — All-Silicon GPU Experiments

**Domain:** GPU fixed-function hardware repurposing for computational physics
**Updated:** March 26, 2026
**Status:** Exp 096 complete — first non-shader-core silicon experiment (TMU), 12/12 pass
**Hardware:** RTX 3090 (Ampere GA102), RX 6950 XT (RDNA2 Navi21), llvmpipe
**Protocol:** `wateringHole/GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md`

---

## Thesis

A modern GPU die has at least 9 distinct hardware units. Each was designed for
graphics but computes a general mathematical function. The DF64 discovery proved
the pattern: fp32 ALUs emulate fp64 at 8-16x throughput. Every other unit on the
die is a hidden computer waiting to be repurposed for science.

## Silicon Units Under Investigation

| Unit | What It Computes | QCD Application | Status |
|------|-----------------|-----------------|--------|
| **Shader Core** | FP arithmetic (add, mul, fma) | All operations (baseline) | **LIVE** — fully characterized |
| **Texture Unit (TMU)** | 2D interpolated lookup (1 cycle) | EOS tables, exp/log at reduced precision | **LIVE** — Exp 096, 1.89x on RTX 3090 |
| **Tensor Core** | Matrix multiply-accumulate (MMA) | CG solver D†Dx (60-70% of HMC runtime) | PLAN — needs sovereign ISA (HMMA/IMMA) |
| **RT Core** | BVH traversal + ray-triangle | MD neighbor search, Monte Carlo transport | PLAN — needs RT pipeline or sovereign |
| **ROP** | Per-pixel scatter-add (blend) | Force accumulation, histogram, charge deposition | PLAN — needs render pass |
| **Rasterizer** | Point-in-polygon + barycentric | Particle binning (PIC), FEM cell assignment | PLAN — needs vertex pipeline |
| **Depth Buffer** | Per-pixel min reduction | Voronoi diagrams, distance fields | PLAN — needs render pass |
| **Tessellator** | Adaptive mesh subdivision | AMR, FEM mesh refinement | PLAN — needs hull/domain shaders |
| **Video Encoder** | Block transform + entropy coding | Trajectory compression, motion registration | PLAN — needs NVENC/VCN API |

## Experiment Results

### TMU Table Lookup (Exp 096)

Precomputed exp(x) into a 1024-entry R32Float texture. Compute shader reads
via `textureLoad` (nearest-neighbor, integer coordinates).

| Backend | Compute exp() | TMU exp() | Speedup | TMU Count |
|---------|--------------|-----------|---------|-----------|
| RTX 3090 | 25.1M ops/s | 47.6M ops/s | **1.89x** | 328 |
| RX 6950 XT | 13.9M ops/s | 17.2M ops/s | **1.24x** | 96 |
| llvmpipe | 6.6M ops/s | 6.1M ops/s | 0.92x | 0 (CPU) |

NVIDIA's advantage correlates with its 3.4x TMU count advantage.

### DF64 Silicon Differentiation

AMD RDNA2 outperforms NVIDIA Ampere on DF64 arithmetic:

| Operation | RTX 3090 | RX 6950 XT | AMD Advantage |
|-----------|----------|------------|---------------|
| DF64 chain (64 iters) | 16.9M ops/s | 23.4M ops/s | **+38%** |

Combined with higher DF64 fidelity (precision matrix), AMD is a stronger
substrate for double-float science computing.

### QCD Operation Throughput (Shader Cores)

| Operation | RTX 3090 | RX 6950 XT | llvmpipe |
|-----------|----------|------------|----------|
| Wilson plaquette (FMA) | 29.7M | 22.0M | 6.3M |
| CG dot product (reduce) | 50.0M | 17.2M | 5.3M |
| DF64 arithmetic chain | 16.9M | 23.4M | 4.9M |

NVIDIA leads on reduction (better shared memory bandwidth).
AMD leads on DF64 (better fp32 FMA pipeline for double-float).

## Evolution Path

```
Exp 096: TMU textureLoad (nearest-neighbor, compute shader)     ← WE ARE HERE
    → TMU textureSampleLevel (linear filtering, higher accuracy)
    → ROP scatter-add via render pass (additive blending)
    → Depth buffer Voronoi via Z-buffer cone rendering
    → Rasterizer binning via vertex pipeline
    → Tensor core MMA via sovereign ISA (coralReef Phase D)
    → RT core BVH via ray tracing pipeline (coralReef Phase D)
    → Mixed command stream: compute + draw + RT in one pass
```

## Cross-References

- **Experiment journal:** `experiments/096_SILICON_SCIENCE_TMU_QCD_MAPPING.md`
- **Handoff:** `wateringHole/handoffs/HOTSPRING_SILICON_SCIENCE_TMU_QCD_HANDOFF_MAR26_2026.md`
- **Protocol:** `wateringHole/GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md`
- **toadStool spec:** `phase1/toadStool/specs/ALL_SILICON_PIPELINE.md`
- **Binary:** `barracuda/src/bin/validate_silicon_science.rs`
- **Precision matrix:** `barracuda/src/bin/validate_precision_matrix.rs`
- **Silicon capabilities:** `barracuda/src/bin/validate_silicon_capabilities.rs`
