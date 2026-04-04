# Experiment 100: Silicon Characterization at Scale

**Date:** March 26, 2026
**Status:** Phase 1-4 complete (local fleet), Phase 5 planned (HPC reference)
**Hardware:** RTX 3090, RX 6950 XT, llvmpipe (local); 2x Titan V, 2x MI50, Tesla P80 (incoming)
**Binaries:** `bench_silicon_budget`, `bench_silicon_saturation`, `bench_qcd_silicon`, `bench_silicon_composition`
**Depends on:** Exp 096 (TMU science), Exp 097-099 (silicon budget/saturation/composition/RHMC)

---

## Goal

Characterize the full silicon budget of every available GPU — not just the
shader ALU lane that conventional HPC codes optimize for. Use QCD as the
workload validator: 14 kernels spanning quenched, dynamical, and observable
phases, at lattice sizes from 4^4 to 32^4, in both FP32 and DF64 precision.

The hypothesis: modern GPUs have 10-100x more compute capacity than HPC
lattice QCD codes utilize, because those codes target one silicon unit
(FP64 ALU) and one transport lane (HBM bandwidth), ignoring tensor cores,
texture units, ROPs, cache hierarchy, and compound composition effects.

---

## Local Fleet Results (March 26, 2026)

### Phase 1: Theoretical Silicon Budget

| Metric | RTX 3090 | RX 6950 XT | Ratio | Winner |
|--------|----------|------------|-------|--------|
| FP32 TFLOPS | 35.60 | 23.65 | 1.51x | NVIDIA |
| DF64 TFLOPS (measured Dekker) | 3.24 | 5.90 | 0.55x | AMD 1.82x |
| FP64 TFLOPS (native ALU) | 0.556 | 1.478 | 0.38x | AMD 2.66x |
| Memory BW (GB/s) | 936 | 576 | 1.62x | NVIDIA |
| TMU throughput (GT/s) | 557.6 | 739.2 | 0.75x | AMD |
| ROP throughput (GP/s) | 190.4 | 295.7 | 0.64x | AMD |
| L2+IC cache (MB) | 6 | 132 | 0.05x | AMD 22x |
| VRAM (GB) | 24 | 16 | 1.50x | NVIDIA |
| Tensor FP16 (TFLOPS) | 142 | 0 | — | NVIDIA only |
| Tensor TF32 (TFLOPS) | 71 | 0 | — | NVIDIA only |

Compound budget (all silicon units parallel):
- RTX 3090: 36.7 – 58.2 TFLOPS (1.03x – 1.63x over shader-only)
- RX 6950 XT: 25.1 – 25.4 TFLOPS (1.06x – 1.08x; no tensor cores)

### Phase 2: Measured Saturation (Actual Per-Unit Peak)

| Unit | RTX 3090 | RX 6950 XT | Key Finding |
|------|----------|------------|-------------|
| FP32 FMA chain | 7.97 TFLOPS | 7.78 TFLOPS | Similar measured ALU throughput |
| DF64 Dekker chain | 17.48 TFLOPS | 24.13 TFLOPS | AMD 38% faster (underlying FP32 ops) |
| Bandwidth (16 MB) | 367 GB/s | 358 GB/s | Similar small-buffer BW |
| Bandwidth (64 MB) | 455 GB/s | 1,100 GB/s | AMD Infinity Cache 2.4x advantage |
| Cache boundary | 8 MB (sharp cliff) | flat through 128 MB | IC eliminates cache misses |
| TMU textureLoad | 302.2 GT/s | 291.1 GT/s | Near-parity at saturation |
| LDS reduce | 72.8 Gop/s (582 GB/s) | 75.0 Gop/s (600 GB/s) | Near-parity |
| Atomic throughput | 15.6 Gatom/s | 93.6 Gatom/s | **AMD 6x faster** |

### Phase 3: QCD Kernel Profiling (32^4, 1M sites)

| Kernel | RTX 3090 GFLOP/s | RX 6950 XT GFLOP/s | Winner |
|--------|-------------------|---------------------|--------|
| Gauge force (FP32) | 2,151 | 2,338 | AMD 1.09x |
| Plaquette (FP32) | 2,211 | 2,681 | AMD 1.21x |
| SU(3) matmul (FP32) | 1,326 | 3,148 | AMD 2.37x |
| Link update (FP32) | 2,604 | 3,266 | AMD 1.25x |
| Momentum update (FP32) | 1,582 | 1,601 | Tie |
| CG dot+reduce (FP32) | 243 | 179 | NVIDIA 1.36x |
| Dirac stencil (FP32) | 1,615 | 3,646 | AMD 2.26x |
| Pseudofermion force (FP32) | 1,942 | 3,421 | AMD 1.76x |
| PRNG heat bath (FP32) | 5,853 | 3,820 | NVIDIA 1.53x |
| Polyakov loop (FP32) | 2,721 | 7,216 | AMD 2.65x |
| Gradient flow acc. (FP32) | 1,886 | 3,706 | AMD 1.97x |
| Force (DF64) | 11,695 | 21,008 | AMD 1.80x |
| Plaquette (DF64) | 14,094 | 25,134 | AMD 1.78x |
| CG dot (DF64) | 1,018 | 759 | NVIDIA 1.34x |

AMD dominates 11/14 kernels at 32^4. NVIDIA wins on CG reduction (shared
memory bandwidth) and PRNG (transcendental throughput/TMU composition).

### Phase 4: Silicon Composition (Multi-Unit Parallel)

| Experiment | RTX 3090 | RX 6950 XT | Interpretation |
|------------|----------|------------|----------------|
| ALU + TMU composition | **2.71x** | 1.73x | NVIDIA TMU runs fully parallel with ALU |
| ALU + BW overlap | 1.16x | **1.65x** | AMD memory controller more independent |
| CG pattern (Dirac→dot) | 137% | **202%** | AMD wave scheduler handles transition free |

---

## What This Means: HPC Silicon Waste

### A100 SXM — What QUDA Uses vs What Exists

| Silicon Unit | A100 Capacity | QUDA Utilization | Idle |
|-------------|---------------|-----------------|------|
| FP64 ALU | 9.7 TFLOPS | ~5 TFLOPS (~50%) | 50% |
| FP64 Tensor Core | 19.5 TFLOPS | 0% | **100%** |
| FP32 ALU | 19.5 TFLOPS | 0% (for QCD) | **100%** |
| TF32 Tensor | 156 TFLOPS | 0% | **100%** |
| BF16 Tensor | 312 TFLOPS | 0% | **100%** |
| FP16 Tensor | 312 TFLOPS | ~5% (mixed-prec CG) | 95% |
| HBM2e bandwidth | 2,039 GB/s | ~1,500 GB/s (75%) | 25% |
| TMU (432 units) | ~630 GT/s | 0% | **100%** |
| ROP (160 units) | ~230 GP/s | 0% | **100%** |
| L2 Cache | 40 MB | passive | not targeted |

Total non-sparsity compute: ~830 TFLOPS. QUDA sustains ~5 TFLOPS. **0.6% utilization.**

### H100 SXM — Same Pattern, Worse

| Silicon Unit | H100 Capacity | QUDA Utilization | Idle |
|-------------|---------------|-----------------|------|
| FP64 ALU | 34 TFLOPS | ~15 TFLOPS | 56% |
| FP64 Tensor | 67 TFLOPS | 0% | **100%** |
| TF32 Tensor | 495 TFLOPS | 0% | **100%** |
| FP16 Tensor | 990 TFLOPS | ~some | >95% |
| FP8 Tensor | 1,979 TFLOPS | 0% | **100%** |
| HBM3 bandwidth | 3,350 GB/s | ~2,500 GB/s | 25% |
| TMU (528 units) | ~900 GT/s | 0% | **100%** |

Total compute: ~3,600 TFLOPS. QUDA sustains ~15 TFLOPS. **0.4% utilization.**

### Why the Waste Exists

The Wilson/staggered Dirac operator — the dominant QCD cost — has arithmetic
intensity ~3.5 FLOP/byte. On any modern GPU, it's memory-bound. QUDA
optimizes the bandwidth path (achieving ~75-80% of peak HBM) and accepts
that most ALU capacity sits idle because data can't reach the cores fast
enough through the bandwidth bottleneck.

The insight: **other silicon units have their own data paths.** TMUs read
through the texture cache (separate from L1/L2). Tensor cores operate on
register-resident tiles after a single load. ROPs scatter-write through
the blend unit. These parallel data paths are why our composition experiments
show >1x multipliers — the silicon units don't all compete for the same bus.

---

## The Opportunity: Silicon-Aware QCD on HPC Cards

### FP64 Tensor Cores (A100: 19.5 TFLOPS, H100: 67 TFLOPS)

A100 DMMA (Double-precision Matrix Multiply-Accumulate) operates on 8x4
tiles at 2x the FP64 ALU rate. SU(3) multiplication = 3x3 complex = 6x6
real matmul. With coralReef SASS/PTX emission and tile padding:

- Current (QUDA): force eval limited by FP64 ALU → ~5 TFLOPS
- With FP64 tensor: 19.5 TFLOPS → **2x on the most expensive QCD operation**

### TF32 for Inner MD Steps (A100: 156 TFLOPS)

Omelyan integrator inner force evaluations need ~19-bit mantissa. TF32
(10-bit mantissa + FP32 accumulation) suffices for inner steps. Final
Metropolis evaluation stays FP64 for correctness:

- Current: all steps at FP64 = ~5 TFLOPS
- With TF32 inner: 156 TFLOPS peak, 10% utilization = **15 TFLOPS, 3x**

### TMU for PRNG (432 units on A100)

Our Exp 096 showed TMU exp() lookup at 1.89x on RTX 3090. Box-Muller PRNG
is transcendental-heavy (log, cos). TMU lookup tables free ALU for force
computation — compound composition effect proven at 2.71x on consumer cards.

### Multi-Precision Routing (PrecisionBrain)

Each HMC trajectory phase has different precision needs:

```
Momenta generation     → FP32 + TMU  (Gaussian noise)
Inner MD force (×38)   → TF32 tensor (19-bit sufficient)
Dirac inversion (CG)   → FP16 tensor (preconditioner) + FP64 (accumulate)
Final force (×2)       → FP64 tensor (full precision for ΔH)
Metropolis test        → FP64 ALU    (single comparison)
Observables            → DF64 or FP64 (plaquette/Polyakov trace)
```

No HPC lattice QCD code does this. They run everything on one precision lane.

### Projected Impact

| Approach | Sustained TFLOPS (A100) | Silicon util | Speedup |
|----------|------------------------|-------------|---------|
| QUDA (FP64 only) | ~5 | 0.6% | 1x |
| + FP64 tensor | ~10 | 1.2% | 2x |
| + TF32 inner MD | ~20 | 2.4% | 4x |
| + TMU composition | ~25 | 3.0% | 5x |
| + multi-precision routing | ~30 | 3.6% | 6x |

At cluster scale (1000 H100s): 6x effective speedup = same physics in 1/6
the wall time, or 6x larger lattice volumes at the same wall time.

---

## Local Fleet: Incoming Hardware

### Current Fleet

| GPU | Architecture | FP64 | VRAM | HBM | Role |
|-----|-------------|------|------|-----|------|
| RTX 3090 | Ampere GA102 | 1:64 | 24 GB GDDR6X | No | Production compute |
| RX 6950 XT | RDNA2 Navi21 | 1:16 | 16 GB GDDR6 | No | DF64 champion |
| RTX 4070 | Ada AD104 | 1:64 | 12 GB GDDR6X | No | Validation |
| 2x Titan V | Volta GV100 | 1:2 | 12 GB HBM2 | **Yes** | NVK oracle, sovereign target |

### Arriving / Planned

| GPU | Architecture | FP64 | VRAM | HBM | Role |
|-----|-------------|------|------|-----|------|
| 2x MI50 | GCN5 Vega 20 | 1:2 | 16 GB HBM2 | **Yes** | AMD HBM2 characterization |
| Tesla P80 | Pascal GP100 | 1:2 | 16 GB HBM2 | **Yes** | Third HBM2 vendor data point |
| More decommissioned HBM2 | Various | 1:2 | 16-32 GB | **Yes** | Scale fleet |

### HBM2 Strategy

Decommissioned data center HBM2 cards (Titan V, MI50, Tesla P80, V100) are
available at $100-300 each — 1/100th the cost of new A100s. They share the
critical silicon characteristics that matter for this experiment:

1. **Full-rate FP64** (1:2 ratio, not 1:64 consumer throttle)
2. **HBM2 bandwidth** (650-1024 GB/s vs 500-936 GB/s GDDR)
3. **Tensor cores** (Titan V: 110 TFLOPS FP16; MI50: none but 1:1 FP64:FP32)
4. **TMU/ROP units** (same architecture as data center cards)

The silicon characterization pipeline reveals per-card personality. Running
bench_silicon_budget → bench_silicon_saturation → bench_qcd_silicon →
bench_silicon_composition on each incoming card builds a performance surface
that toadStool uses for hardware-aware routing.

---

## The Science Argument

### What We Prove Locally

Every piece of silicon on a GPU can contribute to physics compute. QCD
validates this because it exercises every type of operation: dense matmul
(force), stencil (Dirac), reduction (CG dot), transcendental (PRNG),
serial chain (Polyakov), and bandwidth-limited streaming (momentum update).
Our 14-kernel benchmark across 2 GPUs, 5 lattice sizes, and 2 precision
tiers is the first systematic measurement of QCD kernel behavior across
silicon units on consumer hardware.

### What CERN Could Prove

If a collaborator (Bazavov, Chuna, or the HotQCD/MILC consortium) ran the
same characterization pipeline on A100/H100/MI250X hardware:

1. The silicon waste table above becomes empirically measured, not estimated
2. Composition multipliers (ALU+TMU, ALU+tensor) quantify the real compound effect
3. Multi-precision routing across an HMC trajectory demonstrates 4-6x speedup
4. A 1000-GPU cluster effectively becomes a 4000-6000 GPU cluster for QCD

The same binary. The same Cargo.lock. The same physics. Just more silicon.

### The Path

```
Local characterization (2 consumer GPUs)                    ← DONE (Exp 097-100)
  → Extend to HBM2 fleet (Titan V, MI50, Tesla P80)        ← NEXT
    → Prove full-rate FP64 + HBM2 BW changes the picture
      → Publish characterization methodology + results
        → Collaborator runs on A100/H100/MI250X
          → Demonstrate 4-6x effective speedup on same hardware
            → QCD at continuum limit on fewer GPUs
```

---

## Cross-References

- Exp 096: `experiments/096_SILICON_SCIENCE_TMU_QCD_MAPPING.md`
- Exp 097: `experiments/097_SILICON_BUDGET_SATURATION_COMPOSITION.md`
- Exp 098: `experiments/098_QCD_SILICON_BENCHMARK_V2.md`
- Exp 099: `experiments/099_GPU_RHMC_ALL_FLAVORS.md`
- BaseCamp: `whitePaper/baseCamp/silicon_characterization_at_scale.md`
- Binaries: `bench_silicon_budget`, `bench_silicon_saturation`, `bench_qcd_silicon`, `bench_silicon_composition`
- toadStool: `compute.performance_surface.report`, `compute.route.multi_unit`
- Hardware: `metalForge/gpu/nvidia/HARDWARE.md`
- Sovereign goal: `SOVEREIGN_VALIDATION_GOAL.md`
