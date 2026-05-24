+++
title = "hotSpring Validation Summary"
description = "Computational physics on consumer GPU — 700/596/1,045 tests, 219 experiments, guideStone Level 6, 25 papers reproduced, $0.30 total science cost"
date = 2026-05-23

[taxonomies]
primals = ["barracuda", "toadstool", "coralreef", "beardog", "songbird", "nestgate", "rhizocrypt", "loamspine", "sweetgrass", "squirrel"]
springs = ["hotspring"]
+++

## Status

- **700 (cylinder) / 596 / 1,045 tests** passing (IPC-first default / barracuda-local), 0 failed, 6 GPU-heavy ignored
- **219 experiments** across 12 physics categories + sovereign GPU
- **25 papers** reproduced (25/25 CPU, 20/25 GPU)
- **guideStone Level 6 CERTIFIED** — NUCLEUS Deployment Validation
- **167 binaries**, **65 validation suites** (smoke/nucleus/silicon), **128 WGSL shaders**, **7 deploy graphs**
- **$0.30** total science cost on consumer hardware
- **Tier 4 IPC-first** — `primal-proof` feature, `barracuda` optional
- **Fleet: 2× Titan V (GV100) + RTX 5060 (Blackwell)** — Tier 1 sovereign infrastructure validated
- **Sovereignty Tier Model** — Tier 0 (cold), Tier 1 (warm infra — validated), Tier 2 (warm compute — blocked by GPC power), Tier 3 (full sovereign)
- **183ms warm pipeline** — falcon preservation, fd store e2e, 76× faster than cold

## Key Validation Binaries

- `hotspring_unibin` — eukaryotic UniBin: certify (L0–L6), validate (18/24 scenarios), status
- `validate_primal_proof` — end-to-end primal composition validation
- `validate_nuclear_eos_*` — AME2020 binding energies (L1/L2/L3)
- `validate_lattice_qcd_*` — SU(3) HMC/RHMC, gradient flow, beta-scan
- `validate_sarkas_md` — Yukawa OCP molecular dynamics
- `validate_anderson_3d` — Anderson localization (1D/2D/3D)
- `validate_pure_gauge` — 16/16 quenched QCD checks

## sporePrint Notebooks (5)

| # | Notebook | Focus |
|---|----------|-------|
| 01 | Composition Validation | Deploy graphs (7), guideStone Level 6, capability routing, atomic types |
| 02 | Benchmark Comparison | Python vs Rust (44.8x–2274x), GPU vs CPU (44–72x), DF64 3.24 TFLOPS |
| 03 | Experiment Evidence | 219 experiments, science ladder milestones, evolution timeline |
| 04 | Cross-Spring Connections | 10 primals consumed, 5 patterns handed back, ecosystem flows |
| 05 | Physics Deep Dive | Nuclear EOS, lattice QCD, sovereign GPU pipeline, code safety |

## Paper Baseline Notebooks (13)

Publishable Python baselines for 25 reproduced papers — `notebooks/papers/`.

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
| 13 | LTEE Anderson Fitness | Anderson & Wiser (live statistics) |

## Workload TOMLs

Not yet created — contribute to `projectNUCLEUS/workloads/hotspring/`.

## See Also

- [Spring Catalog](https://primals.eco/architecture/spring-catalog-status-science-and-evolution/) on primals.eco
- [Lab Notebooks](https://primals.eco/lab/notebooks/) for rendered notebook views
- [baseCamp Papers](https://primals.eco/science/) — nuclear EOS, lattice QCD, plasma physics
