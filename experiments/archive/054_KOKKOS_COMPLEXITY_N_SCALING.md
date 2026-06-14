# Experiment 054: N-Scaling Complexity Benchmark — barraCuda vs Kokkos-CUDA

**Date**: 2026-03-11  
**hotSpring**: v0.6.29  
**barraCuda**: v0.3.5 (8d63c77) — native f64 fallback on Ampere  
**toadStool**: S146 (751b3849)  
**coralReef**: Iter 31 (9d63b72)  
**Hardware**: NVIDIA RTX 3090 (Ampere, GA102), NVIDIA proprietary 570.133.07  
**Driver**: NvidiaProprietary / NvidiaPtxas / Ampere  
**Binary**: `bench_kokkos_complexity`

## Objective

Profile both Rust (barraCuda wgpu/Vulkan) and Kokkos-CUDA (LAMMPS) across problem
sizes N=500, 2000, 10000, 50000 to:

1. Measure scaling exponents (α where steps/s ∝ 1/N^α)
2. Identify whether the gap widens, narrows, or stays stable with N
3. Determine root causes: arithmetic throughput vs memory access vs dispatch overhead
4. Produce absorption targets for the toadStool/barraCuda team

## Configuration

- Physics: Yukawa OCP (pair_style yukawa), NVT thermostat
- Equilibration: 2000 steps, Production: 10,000 steps
- Three physics cases × four problem sizes = 12 benchmarks per backend
- Algorithm auto-selected per case by `AlgorithmSelector`
- f64 strategy: native f64 (DF64 transcendentals unsafe on Ampere proprietary)

## Results

### Case 1: AllPairs, κ=1, Γ=72, r_c=8.0

| N | barraCuda (steps/s) | Kokkos (steps/s) | Gap | bc wall (s) | kk wall (s) |
|---:|---:|---:|---:|---:|---:|
| 500 | 607.2 | 1,676.6 | 2.8× | 19.8 | 7.5 |
| 2,000 | 184.8 | 1,839.3 | 10.0× | 65.0 | 6.8 |
| 10,000 | 63.3 | 559.4 | 8.8× | 190.0 | 21.9 |
| 50,000 | 17.0 | 166.8 | 9.8× | 706.1 | 73.1 |

Note: N=10,000 and N=50,000 auto-selected Verlet (AlgorithmSelector) despite this being
the "AllPairs" case template — at high N, the selector correctly switches algorithms.

### Case 2: Verlet, κ=2, Γ=158, r_c=6.5

| N | barraCuda (steps/s) | Kokkos (steps/s) | Gap | bc wall (s) | kk wall (s) |
|---:|---:|---:|---:|---:|---:|
| 500 | 600.6 | 2,565.6 | 4.3× | 20.0 | 5.0 |
| 2,000 | 251.6 | 2,979.8 | 11.8× | 48.0 | 4.3 |
| 10,000 | 97.8 | 955.1 | 9.8× | 123.0 | 13.0 |
| 50,000 | 24.6 | 286.8 | 11.7× | 487.8 | 42.4 |

### Case 3: Verlet, κ=3, Γ=503, r_c=6.0

| N | barraCuda (steps/s) | Kokkos (steps/s) | Gap | bc wall (s) | kk wall (s) |
|---:|---:|---:|---:|---:|---:|
| 500 | 611.1 | 3,551.9 | 5.8× | 19.7 | 3.7 |
| 2,000 | 277.0 | 3,425.4 | 12.4× | 43.7 | 3.8 |
| 10,000 | 111.8 | 1,165.8 | 10.4× | 107.7 | 10.7 |
| 50,000 | 29.7 | 359.5 | 12.1× | 404.9 | 33.9 |

## Complexity Analysis

### Scaling Exponents (α where steps/s ∝ 1/N^α)

| Case | barraCuda α | Kokkos α | Interpretation |
|------|---:|---:|---|
| AP_k1_G72 | 0.78 | 0.50 | Both sub-linear — GPU not saturated at small N |
| VL_k2_G158 | 0.69 | 0.48 | Both sub-linear — Verlet reduces scaling |
| VL_k3_G503 | 0.66 | 0.50 | Both sub-linear — shorter cutoff helps |

**Key observation**: Both stacks show α < 1.0, meaning neither is compute-bound at
these sizes. The GPU is underutilized. But barraCuda's α is ~0.15-0.28 higher than
Kokkos's in every case — barraCuda degrades faster with N.

### Gap Trend Analysis

| Case | N=500 | N=2k | N=10k | N=50k | Trend |
|------|---:|---:|---:|---:|---|
| AP_k1_G72 | 2.8× | 10.0× | 8.8× | 9.8× | Widening 500→2k, stable 2k→50k |
| VL_k2_G158 | 4.3× | 11.8× | 9.8× | 11.7× | Widening 500→2k, stable 2k→50k |
| VL_k3_G503 | 5.8× | 12.4× | 10.4× | 12.1× | Widening 500→2k, stable 2k→50k |

## Root Cause Analysis

### 1. The N=500 → N=2000 jump (gap triples)

At N=500, the GPU is barely utilized. Both stacks are dispatch-bound. barraCuda shows
~600 steps/s regardless of algorithm at N=500 — a ceiling imposed by wgpu dispatch
overhead (~1.6ms round-trip per step). Kokkos starts slower (~1,700 for AllPairs,
~3,500 for Verlet) because CUDA kernel launch overhead is much lower (~10μs).

The gap jump from ~3-6× to ~10-12× between N=500 and N=2000 reveals that **dispatch
overhead** is the dominant factor at small N, but **arithmetic throughput** takes over
at N≥2000.

### 2. The stable ~10-12× gap at N≥2000

Once the GPU is reasonably occupied, the gap stabilizes. This means the dominant factor
is **per-step arithmetic cost**, not scaling behavior. The root cause is:

- **Native f64 on Ampere**: RTX 3090 has 1:32 f64:f32 throughput ratio. Kokkos uses
  CUDA with `double` which maps to hardware f64 SASS instructions. barraCuda uses WGSL
  compiled to SPIR-V to PTX to SASS — same hardware f64, but the WGSL→SPIR-V→PTX
  path introduces:
  - Register spilling from 64-bit variables in a 32-bit register file model
  - No shared-memory tiled force accumulation (Kokkos uses `var<workgroup>` equivalent)
  - Extra synchronization barriers from wgpu's validation layer
  - SPIR-V→PTX translation overhead in the driver's JIT

### 3. barraCuda scales worse (higher α)

barraCuda's α=0.66-0.78 vs Kokkos α=0.48-0.50 means barraCuda loses ~30% more
throughput per 10× increase in N. Likely causes:

- **No workgroup shared memory** in force kernels — every thread reads from global
  memory. At N=50k, this is massive bandwidth waste.
- **Verlet rebuild overhead**: 1000 rebuilds in 10k production steps with the Nautilus
  brain not yet calibrated (0/12 heads trusted) — every rebuild recomputes the full
  neighbor list from scratch.
- **No occupancy tuning**: fixed workgroup_size(64) regardless of register pressure.

## Absorption Targets for toadStool/barraCuda

### Priority 1: Shared-Memory Tiled Force Kernels
The single biggest win. Kokkos's `pair_compute` uses shared memory to load tiles of
particle data, reducing global memory reads from O(N²) to O(N²/T) where T is tile size.
barraCuda's force shaders read every pair from global memory. Estimated impact: 2-3×
improvement.

### Priority 2: DF64 Transcendental Resolution
The NVVM poisoning fix forced native f64. If coralReef's sovereign compiler can emit
safe DF64 transcendentals (exp, sqrt) via SASS, this would use f32 ALUs for most of
the computation. Estimated impact: 2-4× on Ampere.

### Priority 3: Dispatch Overhead Reduction
At N=500, barraCuda is dispatch-bound (~600 steps/s ceiling). Batching multiple MD
steps into a single GPU submission (multi-step kernel fusion) would amortize wgpu's
~1.6ms per-dispatch overhead. Estimated impact: 2-3× at small N.

### Priority 4: Occupancy-Aware Workgroup Sizing
`toadStool` S146 has `WorkgroupOptimizer` — use it to tune workgroup size based on
register pressure. The current fixed 64 may cause register spilling on f64-heavy
kernels.

### Priority 5: Verlet Rebuild Intelligence
The Nautilus brain should suppress unnecessary rebuilds — 1000 rebuilds in 10k steps
(every 10 steps) is excessive. With proper skin-distance tracking, rebuilds could be
reduced to ~50-100 for most Verlet cases.

## Known Issues

- **Energy reducer returns zero**: T*=0.000000 throughout — the `ReduceScalarPipeline`
  bug persists (upstream barraCuda). steps/s remains valid.
- **AlgorithmSelector override**: At N=10k+, the AllPairs case auto-switches to Verlet.
  This is correct behavior but means the AllPairs data only covers N=500, 2000 for pure
  O(N²) comparison.

## Conclusion

The gap between barraCuda and Kokkos is **10-12× at production sizes (N≥2000)** and
**stable** — it does not worsen with increasing N beyond 2k. This means the gap is
primarily an arithmetic throughput issue (native f64 penalty + no shared-memory tiling),
not a fundamental scaling problem. With shared-memory tiling and DF64 resolution, a
3-4× closure is achievable, bringing the gap to ~3× which approaches the theoretical
wgpu/Vulkan vs CUDA overhead floor.

## Files

- Binary: `barracuda/src/bin/bench_kokkos_complexity.rs`
- Results: `experiments/054_kokkos_complexity_results.json`
- Related: Exp 053 (fixed-N parity), `specs/MULTI_BACKEND_DISPATCH.md`
