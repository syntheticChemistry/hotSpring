# hotSpring: Kokkos/LAMMPS Validation Target

**Date**: March 4, 2026
**From**: groundSpring (cross-spring notice via wateringHole)
**To**: hotSpring
**Priority**: P0 for hotSpring — you are the entry point
**Depends on**: `wateringHole/handoffs/BARRACUDA_KOKKOS_VALIDATION_BASELINE_NOTICE_MAR04_2026.md`

---

## TL;DR

Murillo pointed us at Kokkos (Sandia's C++ performance portability framework)
as the right comparison for BarraCuda. His PhD student **Thomas Chuna** — who
co-authored with Bazavov on SU(3) Lie group integrators for MILC and published
on the DSF/dielectric theory with Murillo — is the person to prepare a review
package for. He can evaluate our shaders against MILC production code he
contributed to.

**New evolution path**:
```
Python (Sarkas) → Kokkos (LAMMPS) → Rust (hotSpring) → GPU (BarraCuda) → sovereign
```

Python validates correctness. Kokkos validates competitiveness.

---

## Action: Sketch LAMMPS Kokkos Benchmark While Dynamical QCD Runs

The 32⁴ dynamical run is GPU-bound for days. Use that time to:

### 1. Identify Overlapping Cases

Map the 9 PP Yukawa DSF cases (already validated in Sarkas) to equivalent
LAMMPS input files:

| Sarkas Case | Γ | κ | LAMMPS Equivalent |
|-------------|---|---|-------------------|
| `dsf_k0_G10_mks` | 10 | 2.0 | `pair_style yukawa`, PPPM |
| `dsf_k0_G50_mks` | 50 | 2.0 | Same, stronger coupling |
| `dsf_k0_G150_mks` | 150 | 2.0 | Same, near-crystalline |
| ... (6 more) | | | |

LAMMPS uses `pair_style yukawa` + `kspace_style pppm` for Yukawa OCP.
Kokkos backend: `package kokkos` + compile with `-DKokkos_ENABLE_CUDA=ON`.

### 2. Record Tier 1 Benchmarks

For each case, record from LAMMPS/Kokkos:
- Wall time per timestep
- Energy drift
- Memory bandwidth utilization
- GPU utilization (nvprof/nsys)
- FLOPS (if measurable)

Compare against same-case BarraCuda numbers already in the validation suite.

### 3. Document PPPM Comparison

Our BarraCuda PPPM (WGSL, vendor-agnostic) vs LAMMPS PPPM (CUDA via Kokkos):
- Same Ewald splitting, same B-spline order
- Different FFT: our WGSL 3D FFT vs cuFFT
- Different dispatch: runtime WGSL JIT vs compile-time CUDA

### 4. Chuna Review Package

Prepare a focused package for Thomas Chuna (details in
`whitePaper/attsi/non-anon/contact/murillo/chuna_profile.md`):

- hotSpring Sarkas reproduction (his group's code)
- Lattice QCD β-scan and HMC shaders (his thesis domain)
- PPPM implementation (his DSF theory work)
- DF64 precision validation (he can evaluate sufficiency)
- Performance data vs MILC/ICER (he knows those numbers)

Lead with physics. Let him discover the infrastructure.

---

## References

- Kokkos: Edwards et al., JPDC 74(12), 2014; Trott et al., IEEE CiSE 24(4), 2022
- Cabana (particle methods on Kokkos): https://github.com/ECP-copa/Cabana
- LAMMPS Kokkos: https://docs.lammps.org/Speed_kokkos.html
- Chuna & Bazavov (2021): arXiv:2101.05320 (SU(3) gradient flow integrators)
- Chuna & Murillo (2024): arXiv:2405.07871 (conservative dielectric functions)

---

*Pull the full cross-spring notice from wateringHole for complete context.*
