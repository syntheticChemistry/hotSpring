+++
title = "hotSpring Validation Summary"
description = "Computational physics on consumer GPU — 993 tests, 181 experiments, guideStone Level 5, 22 papers reproduced, $0.30 total science cost"
date = 2026-05-07

[taxonomies]
primals = ["barracuda", "toadstool", "coralreef", "beardog", "songbird", "nestgate", "rhizocrypt", "loamspine", "sweetgrass", "squirrel"]
springs = ["hotspring"]
+++

## Status

- **993 tests** passing (lib), 0 failed, 6 GPU-heavy ignored
- **181 experiments** across 12 physics categories
- **22 papers** reproduced (Sarkas, Chabanat, Kortelainen, Bender, AME2020, ...)
- **guideStone Level 5** — 30/30 bare checks, BLAKE3 Property 3, 5/5 properties
- **166 binaries**, **64/64 validation suites**, **128 WGSL shaders**
- **$0.30** total science cost on consumer hardware
- **9 primals** required (10 with optional Squirrel), capability-based routing via `by_domain()`
- **Phase 46 composition template** absorbed — event-driven QCD + DAG memoization
- **Deep debt evolution** complete — capability-based discovery, smart file refactoring

## Key Validation Binaries

- `hotspring_guidestone` — 5 guideStone properties (bare + NUCLEUS IPC parity)
- `validate_primal_proof` — end-to-end primal composition validation
- `validate_nuclear_eos_*` — AME2020 binding energies (L1/L2/L3)
- `validate_lattice_qcd_*` — SU(3) HMC/RHMC, gradient flow, beta-scan
- `validate_sarkas_md` — Yukawa OCP molecular dynamics
- `validate_anderson_3d` — Anderson localization (1D/2D/3D)
- `validate_pure_gauge` — 16/16 quenched QCD checks

## sporePrint Notebooks (5)

| # | Notebook | Focus |
|---|----------|-------|
| 01 | Composition Validation | Deploy graph, guideStone Level 5, capability routing, atomic types |
| 02 | Benchmark Comparison | Python vs Rust (44.8x–2274x), GPU vs CPU (44–72x), DF64 3.24 TFLOPS |
| 03 | Experiment Evidence | 181 experiments, 22 science ladder milestones, evolution timeline |
| 04 | Cross-Spring Connections | 10 primals consumed, 5 patterns handed back, ecosystem flows |
| 05 | Physics Deep Dive | Nuclear EOS, lattice QCD, sovereign GPU pipeline, code safety |

## Paper Baseline Notebooks (12)

Publishable Python baselines for 22 reproduced papers — `notebooks/papers/`.

| # | Notebook | Domain |
|---|----------|--------|
| 01 | SEMF Binding Energy | Nuclear physics (live) |
| 02 | Yukawa Screening | Screened Coulomb (live) |
| 03 | Sarkas Yukawa MD | Plasma MD (live + frozen) |
| 04 | TTM Laser-Plasma | Laser heating (live) |
| 05 | Transport Coefficients | Daligault fit (live) |
| 06 | Surrogate Learning | ML sampling (live) |
| 07 | Quenched QCD | SU(3) HMC (live 4^4) |
| 08 | Dynamical Fermions | Staggered QCD (live + frozen) |
| 09 | Abelian Higgs | U(1) gauge-Higgs (live) |
| 10 | Spectral Theory | Anderson/Hofstadter (live) |
| 11 | Gradient Flow | Wilson flow (live 4^4) |
| 12 | Plasma Dielectric | BGK/Mermin (live) |

## Workload TOMLs

Not yet created — contribute to `projectNUCLEUS/workloads/hotspring/`.

## See Also

- [Spring Catalog](https://primals.eco/architecture/spring-catalog-status-science-and-evolution/) on primals.eco
- [Lab Notebooks](https://primals.eco/lab/notebooks/) for rendered notebook views
- [baseCamp Papers](https://primals.eco/science/) — nuclear EOS, lattice QCD, plasma physics
