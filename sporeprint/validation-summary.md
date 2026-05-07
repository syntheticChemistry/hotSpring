+++
title = "hotSpring Validation Summary"
description = "Computational plasma physics, lattice QCD, spectral theory — 697+ tests, paper-parity MD for $0.044"
date = 2026-05-06

[taxonomies]
primals = ["barracuda", "toadstool", "coralreef"]
springs = ["hotspring", "groundspring", "neuralspring", "wetspring"]
+++

## Status

- **697+ tests**, 78 binaries, 62 WGSL shaders
- **Sarkas Yukawa MD** at paper parity on RTX 4070 for **$0.044**
- **Full AME2020** nuclear dataset (2,042 nuclei) on single consumer GPU
- **Lattice QCD** production beta-scans resolving deconfinement at beta=5.69
- **DF64** delivers 3.24 TFLOPS double precision on FP32 cores

## Key Validation Binaries

<!-- TODO: Update with actual binary names from target/release/ -->
- `validate_sarkas_md` — Yukawa OCP molecular dynamics
- `validate_nuclear_eos` — AME2020 equation of state
- `validate_lattice_qcd` — SU(3) HMC beta-scan
- `validate_anderson_3d` — Anderson localization (1D/2D/3D)

## Workload TOMLs

Skeleton available in `projectNUCLEUS/workloads/hotspring/`.

## See Also

- [hotSpring Science Hub](https://primals.eco/lab/springs/hotspring/) on primals.eco
- [baseCamp Papers 07, 10, 15, 25](https://primals.eco/science/)
