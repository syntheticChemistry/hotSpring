# Silicon Science — All-Silicon GPU Experiments

**Domain:** GPU fixed-function hardware repurposing for computational physics
**Updated:** March 29, 2026
**Status:** Exp 096 + Silicon Saturation Profiling complete. TMU PRNG wired into production RHMC. Subgroup reduce live. ROP atomics live. 7-tier routing operational
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
| **ROP** | Per-pixel scatter-add (blend) | Force accumulation, histogram, charge deposition | **LIVE** — i32 fixed-point `atomicAdd` for fermion force (Tier 3) |
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
Exp 096: TMU textureLoad (nearest-neighbor, compute shader)     ✅ DONE
    → TMU Box-Muller PRNG in production RHMC (Tier 0)          ✅ DONE (silicon saturation)
    → Subgroup reduce via subgroupAdd() (Tier 4)               ✅ DONE (silicon saturation)
    → ROP scatter-add via atomicAdd i32 (Tier 3)               ✅ DONE (silicon saturation)
    → TMU textureSampleLevel (linear filtering, higher accuracy)  NEXT
    → Depth buffer Voronoi via Z-buffer cone rendering
    → Rasterizer binning via vertex pipeline
    → Tensor core MMA via sovereign ISA (coralReef Phase D)
    → RT core BVH via ray tracing pipeline (coralReef Phase D)
    → Mixed command stream: compute + draw + RT in one pass
```

## Silicon Saturation Profiling (March 28-29, 2026)

Full 7-phase silicon saturation on strandgate (RTX 3090 + RX 6950 XT):

### Phase 1-2: Profile refresh + full-trajectory benchmark
Refreshed silicon profiles on both GPUs. Created `bench_full_trajectory_silicon`
for actual RHMC trajectory comparison across lattice sizes.

### Phase 3: TMU PRNG (Tier 0)
Box-Muller PRNG via texture lookup — offloads log/cos/sin transcendentals to TMU.
New shader: `su3_random_momenta_tmu_f64.wgsl`. New module: `tmu_tables.rs`.
Wired into `GpuHmcStreamingPipelines::new_with_tmu`.

### Phase 4: Subgroup Reduce (Tier 4)
`subgroupAdd()` for CG dot product reduction — eliminates shared memory barriers.
New shader: `sum_reduce_subgroup_f64.wgsl`. Conditional compilation: falls back
to shared-memory reduce when `wgpu::Features::SUBGROUP` unavailable.

### Phase 5: ROP Atomic Scatter-Add (Tier 3)
Fixed-point i32 `atomicAdd` for fermion force accumulation. Multiple RHMC poles
dispatch simultaneously — no barriers between poles. Scale factor 2^20 provides
~6 significant digits (sufficient for Omelyan integrator O(dt^2) error).
New shaders: `su3_fermion_force_accumulate_rop_f64.wgsl`,
`su3_force_atomic_to_momentum_f64.wgsl`. New module: `rop_force_accum.rs`.

### Phase 6: NPU Observation Extension
`TrajectoryObservation` extended with `SiliconRoutingTags` (TMU PRNG, subgroup
reduce, ROP force accumulation, FP64 strategy, native F64 flag). NPU input
vector expanded from 6D to 11D via `npu_canonical_input_v2`.

### Phase 7: Capacity Analysis
Max lattice per card under both `max_buffer_size` and total VRAM constraints:
- **RTX 3090** (24 GB): L=46⁴ dynamical RHMC (23.6 GB), L=56⁴ quenched
- **RX 6950 XT** (16 GB): L=40⁴ dynamical RHMC (13.5 GB), L=42⁴ quenched

## Cross-References

- **Experiment journal:** `experiments/096_SILICON_SCIENCE_TMU_QCD_MAPPING.md`
- **Handoff:** `wateringHole/handoffs/HOTSPRING_SILICON_SCIENCE_TMU_QCD_HANDOFF_MAR26_2026.md`
- **Protocol:** `wateringHole/GPU_FIXED_FUNCTION_SCIENCE_REPURPOSING.md`
- **toadStool spec:** `phase1/toadStool/specs/ALL_SILICON_PIPELINE.md`
- **Binary:** `barracuda/src/bin/validate_silicon_science.rs`
- **Precision matrix:** `barracuda/src/bin/validate_precision_matrix.rs`
- **Silicon capabilities:** `barracuda/src/bin/validate_silicon_capabilities.rs`
