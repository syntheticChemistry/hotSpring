# Silicon Inventory: Per-Card Per-Unit Utilization Matrix

**Status**: Active — Exp 108 (full-die utilization campaign)
**Version**: hotSpring v0.6.33
**License**: AGPL-3.0-only

## Principle

Every consumer GPU is a standalone compute system. A gaming card has 9+
distinct functional units — each a special-purpose computer. Traditional HPC
uses only shader cores. We intend to utilize every piece of silicon on every
card, turning $300 consumer GPUs into physics machines that rival $30,000 HPC
cards by saturating die area that HPC software ignores.

## Cards on strandgate

| Card | Arch | Die | VRAM | Boost | FP64:FP32 | Subgroup |
|------|------|-----|------|-------|-----------|----------|
| NVIDIA GeForce RTX 3090 | Ampere (GA102) | 628mm² | 24 GB GDDR6X | 1.70 GHz | 1:64 | 32 (warp) |
| AMD Radeon RX 6950 XT | RDNA 2 (Navi 21) | 520mm² | 16 GB GDDR6 | 2.31 GHz | 1:16 | 64 (wavefront) |

## Complete Silicon Unit Inventory

### RTX 3090 (GA102, Ampere)

| # | Silicon Unit | Count/Spec | Theoretical Peak | ecoPrimals Status | QCD Task | Access Path |
|---|-------------|-----------|-----------------|-------------------|----------|-------------|
| 1 | **FP32 ALU** (Shader Cores) | 10496 CUDA cores | 35.6 TFLOPS | **LIVE** | DF64 bulk compute, CG solver, all kernels | wgpu compute |
| 2 | **FP64 ALU** | 164 FP64 cores (1:64) | 0.556 TFLOPS | **LIVE** | Metropolis, observable accum, precision CG | wgpu compute (SHADER_F64) |
| 3 | **Tensor Cores** (3rd gen) | 328 units (Ampere) | 71.0 TF32 / 142.0 FP16 / 2.23 FP64 TFLOPS | **PLAN** | SU(3) matmul, preconditioner | coralReef SASS → HMMA |
| 4 | **RT Cores** (2nd gen) | 82 units (Ampere) | ~58 RT TFLOPS (BVH+intersect) | **PLAN** | Neighbor search, Monte Carlo transport | coralReef SASS → BVH |
| 5 | **TMU** (Texture Units) | 328 units | 557.6 GT/s | **LIVE** | PRNG Box-Muller transcendentals | wgpu texture lookup |
| 6 | **ROP** (Render Output) | 112 units | 190.4 GP/s | **LIVE** | Force scatter-add (atomicAdd i32) | wgpu atomics |
| 7 | **Subgroup** (Warp intrinsics) | 32-wide warps | N/A (saves barriers) | **LIVE** | CG dot-product reduce | wgpu subgroupAdd |
| 8 | **Shared Memory** (SMEM) | 128 KB/SM × 82 SMs | 10.5 MB aggregate | **LIVE** | CG tree reduce fallback, halo | wgpu workgroup |
| 9 | **L2 Cache** | 6 MB | ~2.5 TB/s | Implicit | Stencil reuse, link caching | Hardware managed |
| 10 | **Memory BW** (GDDR6X) | 384-bit, 19.5 Gbps | 936 GB/s | **LIVE** | All buffer I/O | wgpu storage |
| 11 | **NVENC** (Video Encoder) | 1 unit (7th gen) | 8K30 H.265 | **FUTURE** | Simulation frame compression | FFmpeg/NVENC API |
| 12 | **Rasterizer** | 82 GPC raster engines | ~165 Gpix/s | **FUTURE** | Voxelization, spatial binning | wgpu render pass |
| 13 | **Depth Buffer** (Z-test) | Per-ROP | Fill rate | **FUTURE** | Voronoi diagrams, distance fields | wgpu depth attachment |
| 14 | **Tessellator** | Fixed-function | Hardware-rate | **FUTURE** | AMR mesh refinement | wgpu vertex stage |

### RX 6950 XT (Navi 21, RDNA 2)

| # | Silicon Unit | Count/Spec | Theoretical Peak | ecoPrimals Status | QCD Task | Access Path |
|---|-------------|-----------|-----------------|-------------------|----------|-------------|
| 1 | **FP32 ALU** (CU cores) | 5120 stream procs (80 CUs) | 23.65 TFLOPS | **LIVE** | DF64 bulk compute, CG solver | wgpu compute |
| 2 | **FP64 ALU** | 1:16 rate | 1.478 TFLOPS | **LIVE** | Metropolis, observable, precision CG | wgpu compute (SHADER_F64) |
| 3 | **Tensor Cores** | None (RDNA 2) | 0 | N/A | — | — |
| 4 | **RT Cores** (1st gen RA) | 80 Ray Accelerators | ~44 RT TFLOPS (BVH only) | **PLAN** | Neighbor search (BVH traverse) | coralReef / Vulkan RT |
| 5 | **TMU** (Texture Units) | 320 units | 739.2 GT/s | **LIVE** | PRNG Box-Muller transcendentals | wgpu texture lookup |
| 6 | **ROP** (Render Output) | 128 units | 295.7 GP/s | **LIVE** | Force scatter-add (atomicAdd i32) | wgpu atomics |
| 7 | **Subgroup** (Wavefront) | 64-wide wavefronts | N/A (saves barriers) | **LIVE** | CG dot-product reduce | wgpu subgroupAdd |
| 8 | **Shared Memory** (LDS) | 64 KB/CU × 80 CUs | 5.1 MB aggregate | **LIVE** | CG tree reduce fallback | wgpu workgroup |
| 9 | **L2 Cache** | 4 MB | ~1.8 TB/s | Implicit | Stencil reuse | Hardware managed |
| 10 | **Infinity Cache** | 128 MB (L3) | ~2.0 TB/s effective | Implicit | Working-set caching for lattice | Hardware managed |
| 11 | **Memory BW** (GDDR6) | 256-bit, 18 Gbps | 576 GB/s | **LIVE** | All buffer I/O | wgpu storage |
| 12 | **VCN** (Video Core Next) | 1 unit (VCN 3.0) | 8K H.265 | **FUTURE** | Simulation frame compression | FFmpeg/VAAPI |
| 13 | **Rasterizer** | Per-SE raster engines | ~130 Gpix/s | **FUTURE** | Voxelization, spatial binning | wgpu render pass |
| 14 | **Depth Buffer** | Per-ROP | Fill rate | **FUTURE** | Voronoi, distance fields | wgpu depth attachment |
| 15 | **Tessellator** | Fixed-function | Hardware-rate | **FUTURE** | AMR mesh refinement | wgpu vertex stage |

## Utilization Score

### Current (Exp 108)

| Unit | RTX 3090 | RX 6950 XT | Notes |
|------|----------|-----------|-------|
| FP32 ALU | **LIVE** | **LIVE** | DF64 Dekker pairs, all shaders |
| FP64 ALU | **LIVE** | **LIVE** | Adaptive via Fp64Strategy |
| Tensor | PLAN | N/A | Needs coralReef SASS MMA |
| RT Cores | PLAN | PLAN | Needs coralReef BVH dispatch |
| TMU | **LIVE** | **LIVE** | Box-Muller PRNG, Tier 0 |
| ROP | **LIVE** | **LIVE** | Force accumulation atomicAdd |
| Subgroup | **LIVE** | **LIVE** | CG reduce (naga 28 fix applied) |
| Shared Mem | **LIVE** | **LIVE** | CG fallback, always available |
| Memory BW | **LIVE** | **LIVE** | All buffer streaming |
| Rasterizer | FUTURE | FUTURE | Needs wgpu render pass integration |
| Depth Buf | FUTURE | FUTURE | Needs wgpu render pass integration |
| Tessellator | FUTURE | FUTURE | Needs wgpu render pass integration |
| Video Enc | FUTURE | FUTURE | Needs FFmpeg/system API |

**Score**: 7/14 NVIDIA, 6/13 AMD (counting available units only)

### Tier Priority for Next Activation

1. **Tensor Cores (NVIDIA only)** — 71 TF32 TFLOPS idle. SU(3) 3×3 complex
   matmul maps naturally to HMMA tiles. Requires coralReef SASS emission OR
   wgpu FP16 compute via SHADER_F16 as intermediate precision for
   preconditioner inner loop.

2. **RT Cores (both cards)** — BVH hardware does spatial queries at ray-tracing
   speeds. QCD neighbor table construction (currently CPU) can be accelerated.
   Also: Monte Carlo radiation transport through lattice volumes. Requires
   Vulkan ray tracing extension or coralReef.

3. **Rasterizer + Depth Buffer** — The fixed-function rasterizer is a
   point-in-polygon engine at fill rate. Can be used for:
   - Voxelization of gauge field configurations
   - Spatial binning for particle-in-cell methods
   - Distance field computation via depth buffer min-reduction
   Requires switching from pure compute to render-pass mode.

4. **Video Encoder** — Compress simulation snapshots for archival. Low priority
   but free silicon.

## wgpu Feature Matrix

| Feature | RTX 3090 | RX 6950 XT | Used By |
|---------|----------|-----------|---------|
| SHADER_F64 | Yes | Yes | All f64 physics shaders |
| SHADER_F16 | Yes | Yes | Future: tensor proxy via f16 compute |
| SUBGROUP | Yes | Yes | CG reduce (subgroupAdd) |
| TIMESTAMP_QUERY | Yes | Yes | Profiling timestamps |
| PIPELINE_CACHE | Yes | Yes | Pipeline caching |

## Tensor Core Integration Path (Tier 1)

### NVIDIA Ampere Tensor Cores

GA102 has 328 tensor cores (4 per SM × 82 SMs). Each executes:
- FP16: 16×8×16 MMA → 142 TFLOPS
- TF32: 16×8×8 MMA → 71 TFLOPS  
- FP64: 8×4×8 DMMA → 2.23 TFLOPS (only on GA100, GA102 has limited FP64 MMA)

**QCD mapping**: SU(3) gauge link multiplication is 3×3 complex matmul = 6×6
real matmul. This maps to tensor core tiles:
- FP16 MMA for force evaluation inner loop (preconditioner)
- TF32 MMA for gauge force computation
- Exact physics via iterative refinement on FP32/FP64 ALU

**Access paths** (ordered by feasibility):
1. **wgpu SHADER_F16** — write CG preconditioner in FP16, hope naga lowers to
   tensor ops (unlikely but free to try)
2. **coralReef SASS** — emit HMMA instructions directly via SASS binary patching.
   coralReef already has SM75+ HMMA encoding.
3. **Vulkan cooperative matrix** — `VK_KHR_cooperative_matrix` extension for
   explicit MMA tiles. Requires wgpu integration.

### AMD RDNA 2 — No Tensor Cores

RDNA 2 has no dedicated matrix units. RDNA 3 (6950 successor) adds WMMA
instructions. For 6950 XT, the FP32 ALU IS the compute workhorse — DF64 is
the optimal strategy.

## RT Core Integration Path (Tier 1.5)

### NVIDIA 2nd Gen RT Cores (Ampere)

82 RT units, each with:
- BVH traversal unit (hardware tree walker)
- Ray-triangle intersection unit

**QCD/physics mapping**:
- **Neighbor table construction**: Build BVH from lattice sites, query for
  nearest neighbors. Currently O(V) CPU scan → could be O(log V) RT query.
- **Monte Carlo transport**: Ray-march through gauge field for photon/gluon
  propagation in thermal medium.
- **Gradient flow visualization**: Cast rays through lattice for volume rendering.

### AMD Ray Accelerators (RDNA 2)

80 RA units (1 per CU), BVH traversal only (no hardware intersection).
Slower than NVIDIA but still faster than compute BVH.

## Standalone GPU Compute Vision

Each GPU should operate as an independent physics computer:

```
┌─────────────────────────────────────────────────┐
│  Consumer GPU = Standalone Compute System        │
│                                                  │
│  FP32 ALU ──── DF64 bulk physics ────────┐      │
│  FP64 ALU ──── precision accumulation    │      │
│  Tensor   ──── SU(3) matmul (Tier 1)    │      │
│  RT Cores ──── neighbor search           ├─→ QCD│
│  TMU      ──── PRNG transcendentals      │      │
│  ROP      ──── force scatter-add         │      │
│  Subgroup ──── barrier-free reduce       │      │
│  SharedMem──── workgroup communication   │      │
│  MemBW    ──── data streaming            ┘      │
│                                                  │
│  CPU = async scheduler (never in hot path)       │
│  NPU = real-time observation steering            │
└─────────────────────────────────────────────────┘
```

Multiple consumer GPUs in a tower → independent trajectory parallelism.
No multi-GPU communication needed for RHMC — each card runs its own
Markov chain. $300 × 4 cards = $1200 for 4 independent chains.

## Related Documents

- `SILICON_TIER_ROUTING.md` — tier hierarchy and kernel routing table
- `profiles/silicon/*.json` — measured hardware profiles
- toadStool `silicon.rs` — silicon unit discovery and routing types
- barraCuda `DeviceCapabilities` — wgpu feature/limits query
- coralReef — sovereign shader compiler for tensor/RT/SASS paths
