# Experiment 016: Cross-Spring Shader Evolution Map

**Date**: Feb 25, 2026
**Version**: hotSpring v0.6.13, toadStool S60
**Purpose**: Document the full cross-spring shader evolution across the ecoPrimals ecosystem

## Summary

The ecoPrimals ecosystem evolves through "cross-spring" patterns: domain-specific
springs (hotSpring, wetSpring, neuralSpring, airSpring) contribute shaders to the
shared toadStool/barracuda library, which then redistributes evolved versions back.
This document maps every known evolution path.

## Evolution Flows

### hotSpring → toadStool (Physics → Core)

| Shader/Module | Origin | Absorption | Impact |
|---|---|---|---|
| `df64_core.wgsl` | hotSpring Exp 012 (FP64 core streaming) | toadStool S58 | All springs get DF64 arithmetic on FP32 cores |
| `su3_df64.wgsl` | hotSpring gauge force hybrid | toadStool S58 | Lattice QCD on consumer GPUs |
| `complex_f64.wgsl` | hotSpring `lattice/complex_f64.rs` | toadStool | Shared complex arithmetic library |
| `su3.wgsl` | hotSpring `lattice/su3.rs` (v0.5.16) | toadStool | SU(3) matrix operations for lattice QCD |
| 18 lattice shaders | hotSpring HMC pipeline | toadStool | Full lattice QCD on GPU |
| `gpu_cg_resident.rs` | hotSpring CG pipeline | toadStool S60 | GPU-resident CG types |
| CG solver shaders | hotSpring `lattice/cg.rs` (v0.6.1) | toadStool | CG, Dirac, pseudofermion operations |
| `BidirectionalStream` | hotSpring NPU streaming | toadStool | Cross-substrate observable pipeline |
| ESN reservoir shaders | hotSpring v0.6.0 (transport) | toadStool | Echo state networks on GPU |
| Nuclear HFB shaders | hotSpring nuclear EOS study | toadStool | Deformed nuclei, energy functionals, BCS |
| `su3_math_f64.wgsl` | hotSpring v0.6.13 | Candidate | Naga-safe SU(3) math for shader composition |
| `wilson_plaquette_df64.wgsl` | hotSpring v0.6.12 | Candidate | DF64 plaquette with neighbor-buffer indexing |
| NVK PTE bug report | hotSpring Titan V testing | toadStool driver profile | NVK allocation guard |

### toadStool → hotSpring (Core → Physics)

| Shader/Module | Origin | Absorption | Impact |
|---|---|---|---|
| FMA-optimized `df64_core.wgsl` | toadStool S60 | hotSpring v0.6.12 | 8-12% faster HMC trajectories |
| `df64_transcendentals.wgsl` | toadStool S60 | hotSpring v0.6.12 | sqrt/exp/log in DF64 mode |
| `kinetic_energy_df64.wgsl` | toadStool S60 | hotSpring v0.6.12 | KE on FP32 cores |
| `GpuDriverProfile` | toadStool S58 | hotSpring v0.6.10 | Auto-detect Fp64Strategy |
| `Fp64Strategy` enum | toadStool S58 | hotSpring v0.6.10 | Native vs Hybrid selection |
| `GpuPolyakovLoop` pattern | toadStool | hotSpring v0.6.13 | GPU-resident Polyakov loop |
| `check_allocation_safe()` | toadStool | hotSpring v0.6.13 | NVK PTE fault prevention |
| `su3_df64_preamble()` | toadStool S60 | hotSpring v0.6.12 | Unified DF64 shader composition |
| `ShaderTemplate` | toadStool | hotSpring v0.6.10 | NVK transcendental workarounds |
| `WgslOptimizer` | toadStool Sovereign Compute | hotSpring | NAK ILP reorderer + loop unroller |

### wetSpring → toadStool (Bio → Core)

| Shader/Module | Origin | Impact |
|---|---|---|
| `bray_curtis_f64.wgsl` | wetSpring metagenomics | Diversity metrics for all springs |
| `hill_f64.wgsl` | wetSpring quorum sensing (Exp019) | Hill activation, c-di-GMP |
| `unifrac_propagate.wgsl` | wetSpring phylogenetics | Phylogenetic distance |
| `smith_waterman_banded_f64.wgsl` | wetSpring alignment | Sequence alignment |
| `tree_inference_f64.wgsl` | wetSpring taxonomy | Decision tree inference |
| `kmer_histogram.wgsl` | wetSpring metagenomics | k-mer counting |
| `quality_filter.wgsl` | wetSpring handoff v5 | Metagenomic QC |
| `taxonomy_fc.wgsl` | wetSpring taxonomy | Taxonomic classification |
| `rf_batch_inference.wgsl` | wetSpring handoff v5 | Random forest inference |
| ESN reservoir (V17) | wetSpring time series | Echo state networks |
| f64 constant precision fix | wetSpring finding | Math precision for all springs |

### neuralSpring → toadStool (Neural → Core)

| Shader/Module | Origin | Impact |
|---|---|---|
| `hessian_column.wgsl` | neuralSpring V18 | Hessian computation |
| `histogram.wgsl` | neuralSpring V18 | Statistical histograms |
| `metropolis.wgsl` | neuralSpring V18 | Metropolis sampling |
| `symmetrize.wgsl` | neuralSpring V18 | Matrix symmetrization |
| `laplacian.wgsl` | neuralSpring V18 | Laplacian operators |
| `matmul_gpu_evolved.wgsl` | neuralSpring handoff #11 | Evolved matrix multiply |
| `pairwise_l2.wgsl` | neuralSpring novelty search | Diversity metrics for MODES |
| `swarm_nn_forward.wgsl` | neuralSpring neuroevolution | Swarm neural controllers |
| `batch_fitness_eval.wgsl` | neuralSpring metalForge | Evolutionary fitness |
| `spatial_payoff.wgsl` | neuralSpring metalForge | Game theory payoff |

### airSpring → toadStool (Agriculture → Core)

| Shader/Module | Origin | Impact |
|---|---|---|
| `batched_elementwise_f64.wgsl` (FAO-56) | airSpring ET₀ | Evapotranspiration calculation |
| `kriging_f64.wgsl` | airSpring soil moisture | Spatial interpolation |
| `moving_window.wgsl` | airSpring IoT sensors | Time series windowing |

### Multi-Spring Convergence (Shared by 2+ Springs)

| Shader | Springs | Use Cases |
|---|---|---|
| `batched_elementwise_f64.wgsl` | hotSpring + wetSpring + airSpring | Nuclear structure, diversity metrics, ET₀ |
| `fused_map_reduce_f64.wgsl` | hotSpring + wetSpring + airSpring | Convergence norms, Shannon entropy, water balance |
| `df64_core.wgsl` | hotSpring + all via toadStool | FP32 core streaming everywhere |
| ESN reservoir shaders | hotSpring + wetSpring | Phase prediction, biofilm dynamics |
| `cosine_similarity_f64.wgsl` | wetSpring + neuralSpring | Diversity metrics, similarity search |

## Shader Count by Domain (toadStool)

| Category | Count | Primary Springs |
|---|---|---|
| `shaders/lattice/` | 28 | hotSpring |
| `shaders/math/` | 104 | All springs |
| `shaders/science/` | 13 | hotSpring (HFB), airSpring (hydrology) |
| `shaders/bio/` | 8+ | wetSpring, neuralSpring |
| `shaders/ml/` | 5+ | neuralSpring, wetSpring |
| `shaders/linalg/` | 6+ | hotSpring, neuralSpring |
| **Total** | **~164** | Cross-spring ecosystem |

## Validation Results (v0.6.13)

### GPU Streaming HMC — 7/7 PASS

- Bit-identical streaming vs dispatch parity (ΔH error: 0.0)
- GPU-resident PRNG + streaming: plaquette 0.502665
- 4⁴ → 16⁴ scaling validated

### GPU Beta Scan — 6/6 PASS

- Monotonic plaquette ✓
- GPU Polyakov loop positive across all β ✓ (first time computed on GPU)
- 8³×16 cross-check within 5% of 8⁴ ✓
- 99.4s total GPU time for 330 trajectories

### Performance (v0.6.13 RTX 3090 DF64 Hybrid)

| Lattice | CPU ms/traj | GPU ms/traj | Speedup |
|---------|-------------|-------------|---------|
| 4⁴      | 73.4        | 22.6        | 3.2×    |
| 8⁴      | 1157.3      | 30.1        | 38.5×   |
| 8³×16   | 2341.7      | 48.1        | 48.6×   |
| 16⁴     | 18342.1     | 259.5       | 70.7×   |

## Key Insight

The ecoPrimals ecosystem demonstrates that **constrained evolution** (Rust compiler
as selection pressure, AI as mutation operator) naturally produces a shared shader
library where domain discoveries in one spring (hotSpring's DF64 core streaming,
wetSpring's diversity metrics, neuralSpring's neuroevolution) benefit all others.
ToadStool's barracuda crate serves as the evolutionary reservoir — absorbing
mutations from all springs and redistributing evolved variants.

This is analogous to horizontal gene transfer in microbial ecosystems: innovations
don't stay siloed in one organism (spring) but propagate through the shared gene
pool (toadStool) to benefit the entire community.
