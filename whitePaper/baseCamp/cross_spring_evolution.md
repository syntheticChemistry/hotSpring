# Cross-Spring Shader Evolution

**Domain:** ecoPrimals ecosystem — shader provenance and cross-pollination
**Updated:** March 26, 2026
**Status:** 164+ WGSL shaders tracked across 4 springs + toadStool core. Synced to toadStool S155b+. coralReef Iter 67+ (falcon boot solver, sysmem DMA, 30+ files changed for sovereign GPU pipeline).

---

## The Ecosystem

The ecoPrimals project comprises domain-specific "springs" that independently
evolve compute capabilities, with toadStool (barracuda) as the shared fungal
network absorbing what works and redistributing it.

```
hotSpring (nuclear/plasma physics)  ──→ toadStool ←── wetSpring (bio/phylogenetic)
                                          ↓
                                    neuralSpring (ML)
                                    airSpring (atmospheric)
```

## Shader Count by Domain (March 2026)

| Domain | Spring | Count | Key Shaders |
|--------|--------|-------|-------------|
| Precision math | toadStool | 12 | df64_core, complex_f64, su3_math_f64 |
| Lattice QCD | hotSpring | 24 | gauge force (f64/df64), plaquette, HMC, CG, Polyakov |
| MD simulation | hotSpring | 11 | Yukawa force, cell-list, Verlet, observables |
| HFB nuclear | hotSpring | 14 | potentials, density, BCS, deformed |
| Spectral | toadStool | 8 | SpMV, Lanczos (originated in hotSpring) |
| Bio/phylo | wetSpring | ~30 | distance, alignment, tree kernels |
| ESN/reservoir | shared | 6 | reservoir update, readout, prediction |
| NPU substrate | metalForge | 4 | quantization, int4 cascade |
| Abelian Higgs | hotSpring | 3 | U(1) gauge, Higgs field, HMC |
| Transport | hotSpring | 5 | VACF, stress ACF, Green-Kubo |
| **Total** | **all** | **164+** | |

## Key Cross-Spring Evolution Events

| Date | Event | From → To |
|------|-------|-----------|
| Feb 12 | Complex64 WGSL template | hotSpring → toadStool |
| Feb 13 | SU(3) matrix algebra | hotSpring → toadStool |
| Feb 14 | Native f64 builtins discovery | hotSpring → all springs |
| Feb 18 | DF64 core streaming | hotSpring → toadStool `df64_core.wgsl` |
| Feb 20 | ESN reservoir GPU | hotSpring → toadStool → neuralSpring |
| Feb 22 | CellListGpu fix | toadStool → hotSpring (lean) |
| Feb 23 | Spectral module absorption | hotSpring → toadStool (41 KB deleted) |
| Feb 24 | Site-indexing standardization | toadStool t-major → hotSpring adopts |
| Feb 25 | GPU Polyakov loop | toadStool → hotSpring → toadStool (bidirectional) |
| Feb 25 | NVK allocation guard | toadStool → hotSpring |
| Feb 25 | DF64 plaquette + KE | toadStool S60 → hotSpring v0.6.12 |
| Mar 2 | Spectral stats (bandwidth, condition_number) | hotSpring proxy → toadStool S78 `spectral/stats.rs` |
| Mar 2 | SpectralAnalysis (Marchenko-Pastur) | hotSpring → toadStool S78 → used by wetSpring bio spectral |
| Mar 2 | `level_spacing_ratio` cross-spring | hotSpring (born) → toadStool S78 → wetSpring + neuralSpring |
| Mar 2 | NeighborMode 4D precompute | hotSpring `build_neighbors` → toadStool S80 `precompute_periodic_4d` |
| Mar 2 | MultiHeadEsn serde compat | hotSpring `ExportedWeights` ↔ toadStool `MultiHeadEsn` (bidirectional) |
| Mar 2 | Batched Nelder-Mead GPU | neuralSpring → toadStool S79 → benchmarked in hotSpring (HMC tuning) |

## Detailed Experiment Journal

See `../../experiments/016_CROSS_SPRING_EVOLUTION_MAP.md` for the full shader
provenance graph, validation matrix, and absorption timeline.

## Philosophy

> Springs don't reference each other — they learn from each other by reviewing
> code in `ecoPrimals/`, not by importing. toadStool is the only shared dependency.
> This ensures each spring can evolve independently while benefiting from the
> collective discoveries. The constraint is the driver of evolution.

## S155b+ Cross-Spring Pathways (March 2026)

toadStool has advanced to S155b+ with PcieTransport, ResourceOrchestrator, and
GPU sysmon telemetry. barraCuda stabilized at `7c1fd03a` with 806+ WGSL shaders.
coralReef (Iter 67+) evolved the falcon boot solver with strategy pattern
(VRAM/hybrid/sysmem), experiment loop infrastructure, and DMA backend exposure.
Shader counts are stable — the active frontier is hardware (SEC2 DMA, FBHUB
bypass) rather than new shader development.

## S80 Cross-Spring Pathways (Historical)

The S78→S80 toadStool sync revealed three new cross-spring pathways:

1. **Spectral statistics** (`level_spacing_ratio`, `spectral_bandwidth`, `spectral_condition_number`):
   Born in hotSpring's Anderson localization work, absorbed into toadStool `spectral/stats.rs`,
   now used by wetSpring for bio-molecular spectral analysis and neuralSpring for
   RMT-based phase classification. hotSpring now leans on the upstream versions
   in its Anderson 3D proxy.

2. **GPU batch optimization** (`batched_nelder_mead_gpu`): Born in neuralSpring's
   hyperparameter tuning, absorbed into toadStool S79 `optimize/`. hotSpring
   benchmarks it for HMC parameter tuning (1000 parallel problems in 205ms).

3. **MultiHeadEsn** (toadStool `esn_v2`): GPU-backed multi-head ESN. hotSpring
   validated `ExportedWeights` serde compatibility and `HeadGroup` enum alignment.
   Migration path from CPU `MultiHeadNpu` to GPU `MultiHeadEsn` is now clear —
   the ESN becomes a first-class GPU citizen.
