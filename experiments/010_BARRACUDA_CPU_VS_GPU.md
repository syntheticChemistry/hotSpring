# Experiment 010: BarraCUDA CPU vs GPU — Systematic Parity Validation

**Date**: February 22, 2026
**Goal**: Consolidated evidence that BarraCUDA pure Rust CPU math and WGSL GPU
dispatch produce identical physics across all validated domains.
**Binary**: `validate_barracuda_evolution`

---

## Motivation

With 34/34 validation suites passing, we have CPU and GPU validations scattered
across individual binaries. This experiment consolidates all CPU-vs-GPU comparisons
into a single evidence chain, proving:

1. **Pure Rust math matches GPU dispatch** — same physics, bit-level or machine-epsilon parity
2. **Each substrate adds value** — GPU provides speedup, NPU provides energy efficiency
3. **The evolution path works** — Python baseline → Rust CPU → WGSL GPU → metalForge

## Substrate Coverage by Paper

| # | Paper | Python | Rust CPU | Rust GPU | metalForge | CPU-GPU Δ |
|---|-------|:---:|:---:|:---:|:---:|:---:|
| 1 | Sarkas Yukawa MD | ✅ | ✅ | ✅ 9/9 | — | 0.000% drift |
| 3 | Diaw surrogate | ✅ | ✅ | ✅ GPU RBF | — | — |
| 4 | Nuclear EOS | ✅ | ✅ 195/195 | ✅ L2+L3 | — | 1.7× faster |
| 5 | Stanton-Murillo | ✅ | ✅ 13/13 | ✅ parity | ✅ NPU transport | — |
| 6 | Screened Coulomb | ✅ | ✅ 23/23 | — (CPU eigensolve) | — | N/A |
| 8 | Pure gauge SU(3) | ✅ | ✅ 12/12 | ✅ plaquette+HMC | ✅ NPU phase | — |
| 9 | Production QCD | ✅ | ✅ 10/10 | ✅ CG 9/9 + Dirac 8/8 | ✅ NPU classify | 22.2× at 16⁴ |
| 10 | Dynamical QCD | ✅ | ✅ 7/7 | ⬜ pseudofermion GPU pending | — | — |
| 13 | Abelian Higgs | ✅ | ✅ 17/17 | ✅ shader | — | 143× Rust/Python |
| 14-17 | Spectral (1D) | ✅ | ✅ 10/10 | — (CPU-natural) | — | N/A |
| 18 | Lanczos | ✅ | ✅ 11/11 | ✅ 6/6 | — | parity 1e-15 |
| 19 | 2D Anderson | ✅ | ✅ | ✅ via SpMV | — | parity 1.78e-15 |
| 20 | 3D Anderson | ✅ | ✅ 10/10 | — (large matrix, P2) | — | — |
| 21-22 | Hofstadter | ✅ | ✅ 10/10 | — (Sturm, CPU-natural) | — | N/A |

## Key CPU-GPU Parity Results

| Domain | Binary | CPU (ms) | GPU (ms) | Speedup | Max Error |
|--------|--------|----------|----------|---------|-----------|
| Dirac (16⁴) | `validate_gpu_dirac` | 532.6 | 24.0 | **22.2×** | 4.44e-16 |
| CG solver (16⁴) | `validate_gpu_cg` | 532.6 | 24.0 | **22.2×** | exact iters |
| SpMV CSR | `validate_gpu_spmv` | — | — | — | 1.78e-15 |
| Lanczos eigensolve | `validate_gpu_lanczos` | — | — | — | 1e-15 |
| MD forces | `validate_barracuda_pipeline` | — | — | — | 0.000% drift |
| HFB eigensolve | `validate_barracuda_hfb` | 4.30s | 3.65s | **1.2×** | 2.4e-12 |

## Evolution Evidence

### Python → Rust: Pure Math Validation
- CG solver: Rust 200× faster, identical iteration counts
- Lattice QCD: Rust 56× faster (pure gauge), 143× (Abelian Higgs)
- Nuclear EOS: Rust 478× faster (L1)
- Spectral: Rust 8× faster
- Transport: Rust validates all Sarkas-calibrated Green-Kubo points

### Rust CPU → GPU: Dispatch Validation
- Machine-epsilon parity (4.44e-16 Dirac, 4.10e-16 pure GPU QCD)
- Identical iteration counts (CG, HMC)
- 22.2× speedup at production lattice sizes
- 0.000% energy drift maintained on GPU

### GPU → metalForge: Heterogeneous Validation
- GPU→NPU streaming: 9,017× less energy (transport)
- NPU phase classification: β_c = 5.715 (0.4% error)
- Cross-substrate parity: f64→f32 error 5.1e-7, f64→int4 error 0.13

---

## Remaining Gaps

| Gap | What's Needed | Priority |
|-----|--------------|----------|
| Paper 10 GPU pseudofermion | WGSL kernel for fermion force (toadstool absorption target) | P1 |
| Paper 20 GPU 3D Anderson | Large sparse matrix → GPU SpMV Lanczos | P2 |
| Papers 17, 21-22 GPU | Sturm/transfer matrix inherently sequential — no GPU gain | N/A |
| Paper 6 GPU | Sturm bisection eigensolve — sequential, no GPU gain | N/A |
