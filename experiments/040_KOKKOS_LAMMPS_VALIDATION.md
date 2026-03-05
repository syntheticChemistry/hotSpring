# Experiment 040: Kokkos/LAMMPS Validation Baseline

**Date**: March 4, 2026
**Source**: Murillo → groundSpring → wateringHole → hotSpring
**Status**: QUEUED (sketch during 32⁴ dynamical run GPU-bound time)
**Priority**: P0 for hotSpring — entry point for Chuna review

---

## Objective

Establish BarraCuda vs Kokkos/LAMMPS performance comparison across the
9 PP Yukawa DSF cases already validated in Sarkas. This provides a
quantitative baseline for Thomas Chuna's review.

**Evolution path**:
```
Python (Sarkas) → Kokkos (LAMMPS) → Rust (hotSpring) → GPU (BarraCuda) → sovereign
```

Python validates correctness. Kokkos validates competitiveness.

---

## Phase 1: LAMMPS Input File Mapping

Map the 9 Sarkas PP Yukawa DSF cases to equivalent LAMMPS configurations:

| # | Sarkas Case | Γ | κ | LAMMPS pair_style | kspace |
|---|-------------|---|---|-------------------|--------|
| 1 | `dsf_k0_G10_mks` | 10 | 2.0 | `yukawa` | `pppm` |
| 2 | `dsf_k0_G50_mks` | 50 | 2.0 | `yukawa` | `pppm` |
| 3 | `dsf_k0_G150_mks` | 150 | 2.0 | `yukawa` | `pppm` |
| 4 | `dsf_k1_G10_mks` | 10 | 1.0 | `yukawa` | `pppm` |
| 5 | `dsf_k1_G50_mks` | 50 | 1.0 | `yukawa` | `pppm` |
| 6 | `dsf_k1_G150_mks` | 150 | 1.0 | `yukawa` | `pppm` |
| 7 | `dsf_k3_G10_mks` | 10 | 3.0 | `yukawa` | `pppm` |
| 8 | `dsf_k3_G50_mks` | 50 | 3.0 | `yukawa` | `pppm` |
| 9 | `dsf_k3_G150_mks` | 150 | 3.0 | `yukawa` | `pppm` |

### LAMMPS Configuration Template

```lammps
package kokkos
units lj
atom_style atomic
lattice fcc ${density}
region box block 0 ${Lbox} 0 ${Lbox} 0 ${Lbox}
create_box 1 box
create_atoms 1 box
pair_style yukawa ${kappa} ${cutoff}
pair_coeff 1 1 ${epsilon}
kspace_style pppm 1e-4
fix 1 all nvt temp ${T_target} ${T_target} ${tau}
timestep ${dt}
run ${nsteps}
```

Compile with: `-DKokkos_ENABLE_CUDA=ON`

---

## Phase 2: Tier 1 Benchmarks

For each of the 9 cases, record:

| Metric | Source | Tool |
|--------|--------|------|
| Wall time per timestep | LAMMPS log | built-in |
| Energy drift | LAMMPS thermo | built-in |
| Memory bandwidth | GPU | `nsys`/`nvprof` |
| GPU utilization | GPU | `nsys`/`nvprof` |
| FLOPS (if measurable) | GPU | `nsys`/`nvprof` |

Compare against BarraCuda numbers from `sarkas_gpu` (9/9, 0.000% drift).

---

## Phase 3: PPPM Comparison

| Aspect | BarraCuda | LAMMPS/Kokkos |
|--------|-----------|---------------|
| Language | WGSL (runtime JIT) | CUDA (compile-time) |
| FFT | WGSL 3D FFT | cuFFT |
| Ewald splitting | same | same |
| B-spline order | same | same |
| Vendor portability | Any Vulkan GPU | NVIDIA only (CUDA Kokkos) |
| Driver | Open source (NVK/Mesa) | Proprietary (nvidia.ko) |
| Dispatch | Runtime WGSL JIT | Compile-time CUDA |

---

## Phase 4: Chuna Review Package

Deliverables for Thomas Chuna (profile in
`whitePaper/attsi/non-anon/contact/murillo/chuna_profile.md`):

1. **Sarkas reproduction** — his group's code validated in hotSpring
   (12 cases, 9/9 GPU, 0.000% drift)
2. **Lattice QCD β-scan** — HMC shaders on consumer GPU
   (quenched 10/10, dynamical 7/7, GPU 8/8)
3. **PPPM implementation** — vendor-agnostic Ewald sum
4. **DF64 precision validation** — f64 storage + f32-pair arithmetic
5. **Performance data** — wall times, speedups vs CPU, vs Python
6. **Public repos**: hotSpring, barraCuda (AGPL-3.0)

Strategy: lead with physics, let him discover the infrastructure.

---

## References

- Kokkos: Edwards et al., JPDC 74(12), 2014
- Kokkos v4: Trott et al., IEEE CiSE 24(4), 2022
- Cabana (particle methods on Kokkos): https://github.com/ECP-copa/Cabana
- LAMMPS Kokkos: https://docs.lammps.org/Speed_kokkos.html
- Chuna & Bazavov (2021): arXiv:2101.05320
- Chuna & Murillo (2024): arXiv:2405.07871
