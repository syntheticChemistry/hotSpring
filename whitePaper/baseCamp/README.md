# baseCamp: Research Domain Briefings

**Date:** 2026-03-27
**Project:** hotSpring (ecoPrimals)
**Status:** v0.6.32+, experiments 001-123, 4,065+ tests pass, 39/39 validation suites, 120+ binaries, 86+ WGSL shaders

**Notes:** Upstream primal sync: barraCuda `7c1fd03a`, coralReef Iter 67+ (Kepler PIO loader, GK210 identity, parasitic probe). toadStool S155b+. **Sovereign GPU compute: L10 ROOT CAUSE DEFINITIVE (Exp 122).** WPR2 hardware-locked, FWSEC inaccessible, FBPA offline — explains all WPR copy stalls. HS mode achieved (Exp 112, SCTL=0x3002) but PMU-dependent (Exp 113). **K80 (GK210, Kepler) arriving 2026-03-26** — zero firmware security, direct PIO FECS/GPCCS boot, validates full pipeline. Exp 110-122: consolidation matrix, dual-phase boot, WPR2 tracking/preservation/resolution. **Fleet: 2x Titan V + RTX 5070 (GB206) + K80 (incoming).** Chuna Papers 43-45: **44/44 overnight checks pass**. All AGPL-3.0-only.

---

## Overview

Each document in this directory summarizes what hotSpring reproduced, evolved,
and validated from a specific research domain. The evolution follows a consistent path:

```
Published paper (Python/FORTRAN/HPC)
  → Rust CPU reproduction (sovereign, validated against reference)
    → GPU acceleration (toadStool/barracuda WGSL shaders)
      → Pure GPU streaming (zero CPU round-trips)
        → DF64 core streaming (FP32 cores for f64 math)
          → Cross-substrate (GPU + NPU + Titan V oracle)
```

---

## Briefings

| File | Domain | Papers | Status |
|------|--------|--------|--------|
| [`murillo_plasma.md`](murillo_plasma.md) | Murillo Group — Dense Plasma MD | Papers 1-6 | 60/60 checks, paper parity |
| [`murillo_lattice_qcd.md`](murillo_lattice_qcd.md) | Lattice QCD — Quenched & Dynamical | Papers 7-12 | Asymmetric 64³×8, N_f=4 infra, deconfinement at β_c=5.69 |
| [`chuna_gradient_flow.md`](chuna_gradient_flow.md) | **Chuna Paper 43** — SU(3) Gradient Flow | arXiv:2101.05320 | **11/11 core** (LSCFRK derived, 38.5× GPU); dynamical N_f=4 ext 3/3 complete |
| [`chuna_bgk_dielectric.md`](chuna_bgk_dielectric.md) | **Chuna Paper 44** — Conservative BGK Dielectric | Phys. Rev. E 111, 035206 | **20/20**, standard+completed+multi-comp, DSF vs MD, 322× Rust/Py |
| [`chuna_kinetic_fluid.md`](chuna_kinetic_fluid.md) | **Chuna Paper 45** — Multi-Species Kinetic-Fluid | J. Comput. Phys. (2024) | **10/10**, BGK+Euler+coupled GPU pipeline |
| [`kachkovskiy_spectral.md`](kachkovskiy_spectral.md) | Spectral Theory — Anderson & Hofstadter | Kachkovskiy | 45/45 checks, GPU Lanczos |
| [`cross_spring_evolution.md`](cross_spring_evolution.md) | Cross-Spring Shader Ecosystem | All springs | 164+ shaders, 4 springs |
| [`neuromorphic_silicon.md`](neuromorphic_silicon.md) | Neuromorphic Silicon — AKD1000 Exploration | None (hardware exploration) | Exp 020-031, live NPU in production QCD, 4-layer brain, NPU parameter control |
| [`reality_ladder_rung0.md`](reality_ladder_rung0.md) | Reality Ladder — Mass x Volume x Beta Scan | None (original experiment) | Exp 033, 479 trajectories, Nf=4, mass 0.01-0.5, volumes 2^4-6^4 |
| [`esn_baseline_validation.md`](esn_baseline_validation.md) | ESN Baseline — CPU Training & Capability Map | None (NPU engineering) | 116 pts, 5 synthetic probes, 3 heads EXCELLENT, HeadConfidence tracker live |
| [`npu_dynamic_programming.md`](npu_dynamic_programming.md) | NPU as DP — Activation Parity & Subproblem Memoization | None (architecture insight) | tanh≈ReLU validated, NPU-as-memoization-table architecture for multigrid HMC |
| [`neuromorphic_native_field_theory.md`](neuromorphic_native_field_theory.md) | Neuromorphic-Native Field Theory — Lattice Physics on Spiking Hardware | None (long-term hardware architecture) | 5-level path from NPU steering → NPU IS the simulation, coralForge isomorphism |
| [`sovereign_gpu_compute.md`](sovereign_gpu_compute.md) | Sovereign GPU Compute — GlowPlug, Falcon, PFIFO, DRM Dispatch | None (hardware exploration) | Exp 060-123: **L10 ROOT CAUSE DEFINITIVE** (Exp 122). WPR2 HW-locked, FWSEC inaccessible. HS achieved (Exp 112). K80 (Kepler, no security) arriving — validates full stack. 123 experiments, 2 GPU architectures. DRM dual-track (AMD PM4 + NVIDIA EXEC), iommufd/cdev VFIO (607 tests), **RTX 5070 Blackwell DRM** (SM120) |
| [`silicon_science.md`](silicon_science.md) | Silicon Science — All-Silicon GPU Experiments | None (hardware exploration) | Exp 096: **TMU table lookup 1.89x** (RTX 3090), **AMD DF64 38% faster** than NVIDIA. 11 QCD ops mapped to 9 silicon units. `validate_silicon_science` binary. toadStool performance surface integration |
| [`silicon_characterization_at_scale.md`](silicon_characterization_at_scale.md) | Silicon Characterization at Scale — Consumer to CERN | None (hardware exploration + HPC analysis) | Exp 097-100: **4-phase silicon pipeline** (budget/saturation/composition/QCD). AMD 6x atomics, 2.26x Dirac, 202% CG efficiency. HPC silicon waste: A100 0.6%, H100 0.4% utilization. Decommissioned HBM2 fleet strategy |
| [`self_tuning_rhmc.md`](self_tuning_rhmc.md) | Self-Tuning RHMC — Physics-Validated Parameter Discovery | None (methodology evolution) | Exp 103: `RhmcCalibrator` eliminates hand-tuned magic numbers. GPU spectral probe (power iteration λ_max, m² λ_min), acceptance-driven dt/n_md, consistency-driven pole count. 12 tolerance constants. NPU bridge pending |

---

## Cross-References

- **wetSpring** (`../../../wetSpring/whitePaper/baseCamp/`): Per-faculty briefings for bio/phylogenetic domains. Shares DF64, ESN, and pairwise distance shaders via barraCuda.
- **barraCuda** (`../../../barraCuda/`): Standalone compute primal (budded from toadStool S89). 792+ WGSL shaders, DF64, precision system, lattice QCD, eigensolver, activations API.
- **coralReef** (`../../../coralReef/`): Sovereign shader compiler primal (Phase 10, Iter 67+). WGSL→native binary compilation. VFIO dispatch pipeline (PFIFO + MMU + DMA). coral-glowplug persistent PCIe broker daemon. coral-driver pure Rust GPU backends (AMD DRM, NVIDIA nouveau, NVIDIA VFIO). **iommufd/cdev** kernel-agnostic VFIO (607 tests). **RTX 5060 Blackwell DRM** (SM120, 4/4 tests). SEC2 HS mode via sysmem DMA (Exp 095). Falcon boot solver with strategy pattern (VRAM/hybrid/sysmem).
- **toadStool** (`../../../phase1/toadStool/`): Hardware discovery and orchestration (S155b+). PcieTransport, ResourceOrchestrator, GPU sysmon telemetry.
- **Experiment journals**: `../../experiments/` (001-123)
- **Handoffs**: `../../wateringHole/handoffs/` (fossil record of all cross-project exchanges)
