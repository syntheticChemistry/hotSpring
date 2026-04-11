# baseCamp: Research Domain Briefings

**Date:** April 10, 2026  
**Project:** hotSpring (ecoPrimals)  
**Status:** v0.6.32 — 956 lib tests, 145 binaries, 128 WGSL shaders; research index current through audit/execution cycle; NUCLEUS composition validation operational; ecoBin in `infra/plasmidBin/hotspring/`. NVIDIA GPFIFO (RTX 3090) and AMD scratch/local (RX 6950 XT) paths operational; SovereignInit and multi-ember fleet work complete per domain briefings.

**Validation arc:** Science stacks are validated on three tiers — **Python (and legacy HPC) baselines → sovereign Rust → NUCLEUS IPC composition** — so peer-reviewed physics and methods are checked at the **primal composition layer**, not only in standalone binaries.

---

## Overview

Each document below summarizes what hotSpring reproduced, evolved, or validated in a research domain. Implementation depth follows:

```
Published paper (Python/FORTRAN/HPC)
  → Rust CPU reproduction (validated against reference)
    → GPU acceleration (toadStool/barracuda WGSL)
      → Pure GPU / cross-substrate paths (DF64, fleet, NPU where applicable)
```

**Composition tier:** Rust-direct results are the baseline for **IPC-composed** primal stacks (Tower / Node / Nest / full NUCLEUS), using the same tolerance-and-exit-code discipline as Python→Rust.

---

## Documents (18)

| File | Role | Status (Apr 2026) |
|------|------|-------------------|
| **README.md** | Index for this folder | Current; points to 17 domain briefings |
| [`murillo_plasma.md`](murillo_plasma.md) | Dense plasma MD | 60/60 checks, paper parity |
| [`murillo_lattice_qcd.md`](murillo_lattice_qcd.md) | Lattice QCD (quenched & dynamical) | Asymmetric 64³×8, N_f=4 infra, deconfinement at β_c≈5.69 |
| [`chuna_gradient_flow.md`](chuna_gradient_flow.md) | Chuna 43 — SU(3) gradient flow | 11/11 core; LSCFRK; strong GPU speedup |
| [`chuna_bgk_dielectric.md`](chuna_bgk_dielectric.md) | Chuna 44 — conservative BGK dielectric | 20/20; multi-component paths |
| [`chuna_kinetic_fluid.md`](chuna_kinetic_fluid.md) | Chuna 45 — kinetic–fluid multi-species | 10/10; coupled GPU pipeline |
| [`kachkovskiy_spectral.md`](kachkovskiy_spectral.md) | Anderson / Hofstadter spectral | 45/45; GPU Lanczos |
| [`cross_spring_evolution.md`](cross_spring_evolution.md) | Shader ecosystem across springs | 164+ shaders; four-spring coordination |
| [`neuromorphic_silicon.md`](neuromorphic_silicon.md) | AKD1000 / NPU exploration | Exps 020–031; NPU in production QCD paths |
| [`reality_ladder_rung0.md`](reality_ladder_rung0.md) | Mass × volume × β scan | Exp 033; many trajectories |
| [`esn_baseline_validation.md`](esn_baseline_validation.md) | ESN baseline & capability map | Training/capability coverage |
| [`npu_dynamic_programming.md`](npu_dynamic_programming.md) | NPU as DP / memoization insight | Architecture note for multigrid/HMC |
| [`neuromorphic_native_field_theory.md`](neuromorphic_native_field_theory.md) | Long-term NPU field-theory vision | Roadmap / coralForge angle |
| [`sovereign_gpu_compute.md`](sovereign_gpu_compute.md) | GlowPlug, PFIFO, DRM, ACR, dual-GPU | GPFIFO operational; deep debt cleared; large test count |
| [`silicon_science.md`](silicon_science.md) | All-silicon GPU experiments | TMU/subgroup/ROP paths; tiered routing |
| [`silicon_characterization_at_scale.md`](silicon_characterization_at_scale.md) | Consumer to CERN-scale silicon | Saturation profiling; memory ceilings |
| [`self_tuning_rhmc.md`](self_tuning_rhmc.md) | Self-tuning RHMC | RhmcCalibrator; GPU spectral probes |
| [`true_multishift_cg_validated.md`](true_multishift_cg_validated.md) | Multi-shift CG + fermion force | Production RHMC; ΔH controlled |

---

## Cross-References

- **wetSpring** (`../../../wetSpring/whitePaper/baseCamp/`): Per-faculty briefings for bio/phylogenetic domains. Shares DF64, ESN, and pairwise distance shaders via barraCuda.
- **barraCuda** (`../../../barraCuda/`): Standalone compute primal. WGSL library, DF64, lattice QCD, precision system.
- **coralReef** (`../../../coralReef/`): Sovereign shader compiler; WGSL→native; coral-driver backends.
- **toadStool** (`../../../phase1/toadStool/`): Hardware discovery and orchestration; `shader.dispatch`.
- **Experiment journals**: `../../experiments/`
- **Handoffs**: `ecoPrimals/infra/wateringHole/handoffs/` (including April 10, 2026 NUCLEUS deployment, composition validation, and audit/execution entries)
