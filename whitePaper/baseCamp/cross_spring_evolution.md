# Cross-Spring Shader Evolution

**Domain:** ecoPrimals ecosystem — shader provenance and cross-pollination
**Updated:** February 25, 2026
**Status:** 164+ WGSL shaders tracked across 4 springs + toadStool core

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

## Shader Count by Domain (Feb 2026)

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

## Detailed Experiment Journal

See `../../experiments/016_CROSS_SPRING_EVOLUTION_MAP.md` for the full shader
provenance graph, validation matrix, and absorption timeline.

## Philosophy

> Springs don't reference each other — they learn from each other by reviewing
> code in `ecoPrimals/`, not by importing. toadStool is the only shared dependency.
> This ensures each spring can evolve independently while benefiting from the
> collective discoveries. The constraint is the driver of evolution.
