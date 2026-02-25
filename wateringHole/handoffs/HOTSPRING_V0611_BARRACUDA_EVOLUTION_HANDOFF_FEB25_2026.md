# hotSpring v0.6.11 — BarraCuda Evolution & Cross-Spring Handoff

**Date:** February 25, 2026
**From:** hotSpring (biomeGate compute campaign)
**To:** ToadStool / BarraCuda core team + all springs
**Crate:** hotspring-barracuda v0.6.11
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring v0.6.11 completes the DF64 core streaming activation and site-indexing
standardization. The RTX 3090 runs quenched HMC at 32⁴ in 7.7s/trajectory —
2× faster than native f64. Dynamical fermion HMC is fully validated (13/13
streaming checks) with GPU-resident CG achieving 15,360× readback reduction.
This handoff documents the full barracuda evolution, cross-spring provenance,
and actionable work items for toadStool and sibling springs.

---

## Part 1: BarraCuda Evolution in hotSpring

### Local Shader Inventory (20 WGSL files)

| Category | Files | Upstream Status |
|----------|:-----:|----------------|
| SU(3) gauge force | `su3_gauge_force_f64.wgsl`, `su3_gauge_force_df64.wgsl` | f64 upstream, DF64 local (neighbor-buffer) |
| Wilson plaquette | `wilson_plaquette_f64.wgsl` | Upstream has f64 + DF64 |
| HMC integrator | `su3_link_update_f64.wgsl`, `su3_momentum_update_f64.wgsl` | Upstream |
| Kinetic energy | `su3_kinetic_energy_f64.wgsl` | Upstream has f64 + DF64 |
| CG solver (7) | `complex_dot_re_f64`, `axpy_f64`, `xpay_f64`, `cg_compute_alpha_f64`, `cg_compute_beta_f64`, `cg_update_xr_f64`, `cg_update_p_f64` | Absorbed (S31d) |
| Reduction | `sum_reduce_f64.wgsl` | Absorbed |
| Fermion | `dirac_staggered_f64.wgsl`, `staggered_fermion_force_f64.wgsl`, `gaussian_fermion_f64.wgsl` | Dirac absorbed, force/gaussian local |
| PRNG | `su3_random_momenta_f64.wgsl` | Local only |
| SU(3) algebra | `su3_f64.wgsl`, `complex_f64.wgsl` | Upstream identical |

### What hotSpring Contributed to toadStool

| Contribution | Session | Impact |
|-------------|---------|--------|
| `df64_core.wgsl` (Dekker splitting) | S58 | Foundation for all DF64 ops |
| DF64 core streaming strategy | S58 | 6.7-9.9× speedup concept |
| `Fp64Strategy` enum concept | S58 | Auto-detect Native vs Hybrid |
| Site-indexing incompatibility discovery | v0.6.10 | Led to v0.6.11 standardization |
| Neighbor-buffer pattern (20 shaders) | — | Proposed for upstream adoption |
| NAK deficiency analysis (5 issues) | — | Led to `WgslOptimizer` |
| PTE fault characterization (30⁴ boundary) | — | `NvkLargeBufferLimit` workaround |
| f64 transcendental crash (NVK) | — | `NvkExpF64Crash` workaround |
| Workgroup dispatch limit fix (2D dispatch) | — | All shaders use `num_workgroups` |

### What hotSpring Consumed from toadStool

| Primitive | Source Session | Usage |
|-----------|---------------|-------|
| `GpuF64` device management | S18+ | All GPU dispatch |
| `GpuDriverProfile` / `Fp64Strategy` | S58 | DF64 auto-selection |
| `WGSL_DF64_CORE` | S58 | DF64 gauge force shader |
| `WGSL_SU3_DF64` | S58 | DF64 SU(3) matmul in shader |
| `ShaderTemplate` + workarounds | S58+ | NVK/Ada transcendental fixes |
| `WgslOptimizer` | S62 | NAK performance mitigation |
| `split_workgroups()` | S53 | 2D dispatch for large lattices |
| Buffer size constants | S53 | 2 GB binding / 4 GB total limits |

### Version Evolution

| Version | Date | Headline |
|---------|------|----------|
| v0.1-v0.4 | Feb 14-18 | L1 nuclear EOS, L2 GPU MD, Yukawa OCP |
| v0.5 | Feb 19-20 | L3 lattice QCD, SU(3) HMC, CG solver |
| v0.6.0-v0.6.7 | Feb 20-23 | Dynamical fermions, streaming, NPU bridge |
| v0.6.8 | Feb 24 | Production β-scan (Exp 013, deconfinement) |
| v0.6.9 | Feb 24 | toadStool S62 sync, DF64 absorption confirmed |
| v0.6.10 | Feb 25 | DF64 core streaming live on RTX 3090 |
| **v0.6.11** | **Feb 25** | **Site-indexing standardization, full validation** |

---

## Part 2: Cross-Spring Evolution Map

Following wetSpring's cross-spring provenance model (see
`ecoPrimals/wetSpring/wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md`):

### hotSpring → toadStool → All Springs

| Primitive | Domain | Date | Cross-Spring Value |
|-----------|--------|------|-------------------|
| `df64_core.wgsl` | Precision | Feb 24 | All springs get DF64 on consumer GPUs |
| `su3_df64.wgsl` | Physics | Feb 24 | SU(3) algebra at ~14-digit precision |
| `Fp64Strategy` | Device | Feb 24 | Auto-detect Native vs Hybrid per-GPU |
| f64 transcendental workaround | Precision | Feb 24 | NVK shaders work on all GPUs |
| NAK deficiency analysis | Compiler | Feb 24 | `WgslOptimizer` helps all springs |
| Neighbor-buffer pattern | Indexing | Feb 25 | Proposed: decouple ops from layout |

### wetSpring → toadStool → hotSpring

| Primitive | Impact on hotSpring | Date |
|-----------|---------------------|------|
| `GemmCachedF64` | 60× speedup for HFB nuclear structure | Feb 20 |
| `math_f64.wgsl` precision fix | Correct f64 constants in all shaders | Feb 16 |
| `NvvmAdaF64Transcendentals` | RTX 4070 f64 shader correctness | Feb 24 |
| Bio shader patterns | Shared dispatch/streaming architecture | Feb 16-23 |

### neuralSpring → toadStool → hotSpring

| Primitive | Impact on hotSpring | Date |
|-----------|---------------------|------|
| ESN reservoir ops | GPU-trained, NPU-deployed inference | Feb 20 |
| `PairwiseHammingGpu` | Pattern for pairwise GPU metrics | Feb 22 |

---

## Part 3: Actionable Work Items for toadStool

### P0: Neighbor-Buffer Support in Lattice Ops

See `TOADSTOOL_SITE_INDEXING_NAK_SOLVER_HANDOFF_FEB25_2026.md` Part 1.

Add `NeighborMode::PrecomputedBuffer` to `Su3HmcForce`, `WilsonPlaquette`,
`GpuKineticEnergy`. This lets any spring use upstream ops regardless of
internal memory layout, and enables domain decomposition for multi-GPU.

**Acceptance**: hotSpring passes `validate_pure_gpu_hmc` using upstream ops
with its own neighbor buffer.

### P1: DF64 Expansion to Remaining Kernels

Apply DF64 to plaquette, kinetic energy, and momentum update (currently only
gauge force uses DF64). This increases the DF64-covered fraction from 40% to
~65% of HMC wall time, projecting a ~2.8× overall speedup on consumer GPUs.

### P2: Rust-Native NAK Solver

See `TOADSTOOL_SITE_INDEXING_NAK_SOLVER_HANDOFF_FEB25_2026.md` Part 2.

Build sovereign compiler infrastructure in Rust within toadStool. Phase 1:
architecture-aware SPIR-V scheduling. Phase 2: peephole optimizations. Phase 3:
native code generation.

### P3: NVK PTE Fault Resolution

Test Mesa git HEAD on Titan V. If still failing at 30⁴+, file upstream bug
with minimal reproducer. Consider buffer sub-allocation as workaround.

### P3: Local Shader Retirement

Once neighbor-buffer support exists upstream, hotSpring can retire local copies
of: `su3_gauge_force_f64.wgsl`, `wilson_plaquette_f64.wgsl`,
`su3_kinetic_energy_f64.wgsl`. The local `su3_gauge_force_df64.wgsl` becomes
unnecessary if upstream force op supports both DF64 and pre-computed neighbors.

---

## Part 4: What hotSpring Proved

### Validated Physics

| Domain | Papers | Checks | Status |
|--------|--------|--------|--------|
| Yukawa OCP (plasma) | 9 | 50+ | 9/9 paper parity |
| Nuclear EOS (HFB) | 4 | 200+ | AME2020 2,042 nuclei |
| Lattice QCD (quenched) | 3 | 100+ | Deconfinement at β=5.69 |
| Lattice QCD (dynamical) | 1 | 13/13 | Streaming + resident CG |
| Transport (Green-Kubo) | 2 | 30+ | 4 methods validated |
| Abelian Higgs | 1 | 10+ | 143× faster than Python |
| Spectral theory | 2 | 50+ | Anderson localization |
| **Total** | **22** | **400+** | **39/39 validation suites** |

### Hardware Envelope

| GPU | Max Quenched | Max Dynamical | DF64 Strategy |
|-----|-------------|---------------|---------------|
| RTX 4070 (12 GB) | 44⁴ | 40⁴ | Hybrid (3.24 TFLOPS) |
| RTX 3090 (24 GB) | 56⁴ | 48⁴ | Hybrid (3.24 TFLOPS) |
| Titan V (12 GB, NVK) | 30⁴ (PTE limit) | 24⁴ (est.) | Native (7.45 TFLOPS hw) |

### Cost of Science

| Campaign | Hardware | Time | Cost | Physics |
|----------|----------|------|------|---------|
| 22 papers reproduced | RTX 4070 | ~2 days | $0.20 | Complete |
| 32⁴ quenched β-scan | RTX 3090 | 13.6 hrs | $0.58 | Deconfinement |
| 16⁴ NVK validation | Titan V | 47 min | ~$0.01 | First open-driver QCD |
| 32⁴ DF64 benchmark | RTX 3090 | 2.6 min | ~$0.002 | 2× speedup confirmed |

---

## Part 5: metalForge & GPU→NPU Streaming (Next)

hotSpring's `metalForge/forge/` crate provides hardware discovery and
cross-substrate dispatch. The ESN reservoir pipeline is validated:
GPU trains → quantize → NPU deploys at 30mW.

The next frontier: **GPU→NPU streaming during HMC trajectories**.
The NPU (Akida AKD1000, 78 neural processors) can run inference models
that classify physical regimes, predict transport coefficients, or detect
phase transitions in real-time while the GPU computes. The Titan V can
serve as a final validation backend (native f64) for GPU→NPU predictions.

See `metalForge/npu/akida/EXPLORATION.md` for the deployment pipeline.
