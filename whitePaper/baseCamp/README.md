# baseCamp: Research Domain Briefings

**Date:** 2026-04-05
**Project:** hotSpring (ecoPrimals)
**Status:** v0.6.32, experiments 001-143, 4,065+ tests pass, 39/39 validation suites, 139 binaries, 99 WGSL shaders. **NVIDIA GPFIFO pipeline OPERATIONAL on RTX 3090.** **AMD scratch/local memory OPERATIONAL on RX 6950 XT.** AMD sovereign compiler: 24/24 QCD shaders → native GFX ISA. **ACR HS auth under investigation** — Exp 141 identified VBIOS DEVINIT, **Exp 142-143 contradicted** (fails on POSTed GPU). SEC2 PTOP/PMC bit discovery next. Uncrashable GPU safety architecture validated (Exp 140). coralReef Deep Debt Evolution complete (P1-P7).

**Notes:** Sovereign GPU frontier: ACR authentication loop (PC 0x2d78) initially traced to uninitialized SEC2 crypto engine (Exp 141), but Exp 142-143 show failure persists even on BIOS-POSTed GPU — VBIOS DEVINIT is NOT the sole root cause. SEC2 falcon PTOP discovery and PMC bit enumeration are the active track. DMA path fully debugged: FBIF locked in VIRT mode, sysmem page tables via PRAMIN, falcon MMU routing verified. K80 cold boot wired into `coralctl`. D-state resilience validated. coralReef Deep Debt: socket consistency, ChipCapability trait, unsafe reduction, large file refactoring, ecosystem discovery. **Fleet: 2x Titan V + RTX 5070 (GB206) + K80 + RTX 3090 + RX 6950 XT.** Chuna Papers 43-45: **44/44 overnight checks pass**. All AGPL-3.0-only.

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
| [`sovereign_gpu_compute.md`](sovereign_gpu_compute.md) | Sovereign GPU Compute — GlowPlug, Falcon, PFIFO, DRM Dispatch, ACR Investigation | None (hardware exploration) | Exp 060-143: **NVIDIA GPFIFO OPERATIONAL** on RTX 3090, **AMD scratch memory OPERATIONAL** on RX 6950 XT. ACR HS auth under investigation (Exp 141-143: VBIOS DEVINIT contradicted, SEC2 PTOP/PMC next). Dual GPU sovereign boot (Exp 135-136). Uncrashable safety arch (Exp 140). Deep Debt complete. K80 cold boot pipeline. 1672+ tests |
| [`silicon_science.md`](silicon_science.md) | Silicon Science — All-Silicon GPU Experiments | None (hardware exploration) | Exp 096 + **Silicon Saturation**: TMU PRNG, subgroup reduce, ROP atomics **LIVE** in production RHMC. 7-tier routing operational. NPU 11D observation vector |
| [`silicon_characterization_at_scale.md`](silicon_characterization_at_scale.md) | Silicon Characterization at Scale — Consumer to CERN | None (hardware exploration + HPC analysis) | Exp 097-100 + **Saturation Profiling**: 4-phase pipeline + 7-phase saturation. RTX 3090 L=46⁴ max dynamical (23.6 GB), RX 6950 XT L=40⁴ (13.5 GB). HPC silicon waste: A100 0.6%, H100 0.4% |
| [`self_tuning_rhmc.md`](self_tuning_rhmc.md) | Self-Tuning RHMC — Physics-Validated Parameter Discovery | None (methodology evolution) | Exp 103: `RhmcCalibrator` eliminates hand-tuned magic numbers. GPU spectral probe (power iteration λ_max, m² λ_min), acceptance-driven dt/n_md, consistency-driven pole count. 12 tolerance constants. NPU bridge pending |
| [`true_multishift_cg_validated.md`](true_multishift_cg_validated.md) | True Multi-Shift CG + Fermion Force — Validated RHMC Production | None (algorithmic evolution) | Exp 105: True multi-shift CG (shared Krylov), fermion force sign fix (−η convention), ΔH=O(1), 8.5 GFLOP/s, `std::hint::black_box` optimizer fix |

---

## Cross-References

- **wetSpring** (`../../../wetSpring/whitePaper/baseCamp/`): Per-faculty briefings for bio/phylogenetic domains. Shares DF64, ESN, and pairwise distance shaders via barraCuda.
- **barraCuda** (`../../../barraCuda/`): Standalone compute primal (budded from toadStool S89). 792+ WGSL shaders, DF64, precision system, lattice QCD, eigensolver, activations API.
- **coralReef** (`../../../coralReef/`): Sovereign shader compiler primal (Phase 10, Iter 70). WGSL→native binary compilation for AMD GFX10.3 and NVIDIA SM86. **NVIDIA GPFIFO pipeline operational** on RTX 3090 (BIND + TSG schedule + Volta WST). **AMD sovereign compiler:** 24/24 QCD shaders → native ISA, 38/39 dispatch tests pass. coral-driver pure Rust GPU backends (AMD DRM, NVIDIA UVM). 350+ unit tests. coral-glowplug persistent PCIe broker. iommufd/cdev VFIO (607 tests).
- **toadStool** (`../../../phase1/toadStool/`): Hardware discovery and orchestration (S155b+). PcieTransport, ResourceOrchestrator, GPU sysmon telemetry.
- **Experiment journals**: `../../experiments/` (001-141)
- **Handoffs**: `ecoPrimals/infra/wateringHole/handoffs/` (fossil record of all cross-project exchanges)
