# baseCamp: Research Domain Briefings

**Date:** February 25, 2026
**Project:** hotSpring (ecoPrimals)
**Status:** v0.6.14, 17 experiments, 39/39 validation suites, 664 tests

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
| [`murillo_lattice_qcd.md`](murillo_lattice_qcd.md) | Lattice QCD — Quenched & Dynamical | Papers 7-12 | 32⁴ production, deconfinement seen |
| [`kachkovskiy_spectral.md`](kachkovskiy_spectral.md) | Spectral Theory — Anderson & Hofstadter | Kachkovskiy | 45/45 checks, GPU Lanczos |
| [`cross_spring_evolution.md`](cross_spring_evolution.md) | Cross-Spring Shader Ecosystem | All springs | 164+ shaders, 4 springs |

---

## Cross-References

- **wetSpring** (`../../../wetSpring/whitePaper/baseCamp/`): Per-faculty briefings for bio/phylogenetic domains. Shares DF64, ESN, and pairwise distance shaders via toadStool.
- **toadStool** (`../../../phase1/toadstool/`): Shared compute library. 164+ shaders absorbing from all springs.
- **Experiment journals**: `../../experiments/` (001-017)
- **Handoffs**: `../../wateringHole/handoffs/` (fossil record of all cross-project exchanges)
