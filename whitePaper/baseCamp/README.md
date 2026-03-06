# baseCamp: Research Domain Briefings

**Date:** 2026-03-06
**Project:** hotSpring (ecoPrimals)
**Status:** v0.6.18, experiments 001-040, 685 lib tests, 39/39 validation suites, 92 binaries, 62 WGSL shaders

**Notes:** Deep debt resolution (v0.6.18): Clippy 0 warnings, file-size compliance, unwrap/expect removal. Brain B2 (memory pressure) and D1 (force anomaly) evolved from placeholder to real runtime estimates.

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
| [`murillo_lattice_qcd.md`](murillo_lattice_qcd.md) | Lattice QCD — Quenched & Dynamical + Chuna Flow | Papers 7-12 + arXiv:2101.05320 | Asymmetric 64³×8, gradient flow, Chuna integrators derived, N_f=4 infra |
| [`kachkovskiy_spectral.md`](kachkovskiy_spectral.md) | Spectral Theory — Anderson & Hofstadter | Kachkovskiy | 45/45 checks, GPU Lanczos |
| [`cross_spring_evolution.md`](cross_spring_evolution.md) | Cross-Spring Shader Ecosystem | All springs | 164+ shaders, 4 springs |
| [`neuromorphic_silicon.md`](neuromorphic_silicon.md) | Neuromorphic Silicon — AKD1000 Exploration | None (hardware exploration) | Exp 020-031, live NPU in production QCD, 4-layer brain, NPU parameter control |
| [`reality_ladder_rung0.md`](reality_ladder_rung0.md) | Reality Ladder — Mass x Volume x Beta Scan | None (original experiment) | Exp 033, 479 trajectories, Nf=4, mass 0.01-0.5, volumes 2^4-6^4 |
| [`esn_baseline_validation.md`](esn_baseline_validation.md) | ESN Baseline — CPU Training & Capability Map | None (NPU engineering) | 116 pts, 5 synthetic probes, 3 heads EXCELLENT, HeadConfidence tracker live |
| [`npu_dynamic_programming.md`](npu_dynamic_programming.md) | NPU as DP — Activation Parity & Subproblem Memoization | None (architecture insight) | tanh≈ReLU validated, NPU-as-memoization-table architecture for multigrid HMC |
| [`neuromorphic_native_field_theory.md`](neuromorphic_native_field_theory.md) | Neuromorphic-Native Field Theory — Lattice Physics on Spiking Hardware | None (long-term hardware architecture) | 5-level path from NPU steering → NPU IS the simulation, coralForge isomorphism |

---

## Cross-References

- **wetSpring** (`../../../wetSpring/whitePaper/baseCamp/`): Per-faculty briefings for bio/phylogenetic domains. Shares DF64, ESN, and pairwise distance shaders via barraCuda.
- **barraCuda** (`../../../barraCuda/`): Standalone compute primal (budded from toadStool S89). 767+ WGSL shaders, DF64, precision system, lattice QCD.
- **toadStool** (`../../../phase1/toadStool/`): Shared compute ecosystem. barraCuda retains legacy copy; Springs migrating to standalone.
- **Experiment journals**: `../../experiments/` (001-040)
- **Handoffs**: `../../wateringHole/handoffs/` (fossil record of all cross-project exchanges)
- **Reality Ladder plan**: `../../.cursor/plans/reality_ladder_plan_3571df11.plan.md`
