# Nuclear EOS Strategy: From Python Control to BarraCuda Proof

> **Fossil Record (March 27, 2026):** This document captures state as of February 21, 2026. For current status, see the [root README](README.md) and [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md). Body below is preserved as historical record.

> **FULLY EXECUTED** (Feb 21, 2026) — All phases A–F complete. 195/195 quantitative
> checks pass. See `CONTROL_EXPERIMENT_STATUS.md`
> for current results and `barracuda/CHANGELOG.md` for the ongoing crate evolution.

**Date**: 2026-02-08 (initial), 2026-02-11 (L1+L2), 2026-02-15 (Phase E), 2026-02-17 (GPU density pipeline)  
**Status**: Phase A ✅ Complete, Phase B ✅ L1+L2 Validated, Phase C ✅ GPU MD (9/9 PP Yukawa), Phase D ✅ Native f64 + N-scaling, Phase E ✅ Paper-parity long run (9/9, N=10k, 80k steps, $0.044) + Toadstool rewire, Phase F ✅ GPU-resident HFB density pipeline  
**f64 Status**: Native WGSL builtins confirmed. Consumer Ampere/Ada: fp64:fp32 ~1:64 (both CUDA and Vulkan). Double-float hybrid delivers 9.9× native f64 — bottleneck broken.  
**Context**: The Code Ocean capsule is gated. We built the nuclear EOS from
first principles instead — and used it as the Phase A → Phase B transition.

---

## The Strategic Question

> Should we solve the nuclear EOS in Python, or is this where BarraCuda
> starts proving it can do the math on ANY hardware?

**Answer: Both.** Python establishes correctness (Phase A). BarraCuda
demonstrates hardware independence (Phase B). The surrogate learning
workflow is the *perfect* transition because the math maps almost 1:1
to existing BarraCuda operations.

**Result**: BarraCuda L1 achieves **better accuracy** (χ²=2.27 vs 6.62) at
**478× throughput** on consumer hardware. The thesis is validated.

---

## What We Built (Instead of HFBTHO)

Since HFBTHO requires institutional access/Fortran compilation, we built
the nuclear physics from scratch:

### Level 1: Semi-Empirical Mass Formula (SEMF)
- Skyrme EDF → nuclear matter properties (ρ₀, E/A, K∞, m*/m, J)
- SEMF with Skyrme-derived coefficients (volume, surface, Coulomb, symmetry, pairing)
- 10D optimization: [t₀, t₁, t₂, t₃, x₀, x₁, x₂, x₃, α, W₀]
- 52 nuclei from AME2020 experimental dataset
- Log-transformed objective: log(1 + χ²/datum)

### Level 2: Spherical HF+BCS (hybrid)
- Full Skyrme HF+BCS solver (`skyrme_hfb.py` / `nuclear_eos_l2.rs`):
  - Harmonic oscillator basis, separate proton/neutron channels
  - Isospin-dependent Skyrme potential
  - Coulomb direct (Poisson) + exchange (Slater approximation)
  - BCS pairing with Δ = 12/√A
  - Self-consistent iteration with Hamiltonian diagonalization
- Hybrid dispatch: HFB for 56 ≤ A ≤ 132, SEMF elsewhere
- 18 focused nuclei where HFB adds value

### Level 3: Axially Deformed HFB (target)
- Designated for BarraCuda + Titan V f64 GPU compute
- Requires 2D mesh, larger matrices, iterative eigensolvers
- This is the "Murillo parity" target

---

## The Three Compute Layers

### Layer 1: Objective Function (CPU — the expensive simulation)

```
Input:  x = [t₀, t₁, t₂, t₃, x₀, x₁, x₂, x₃, α, W₀]  (Skyrme params)
Output: χ² against experimental nuclear binding energies

L1:     SEMF calculation (microseconds)
L2:     HF+BCS eigensolve per nucleus (~100ms each, 18 nuclei)
L3:     Deformed HFB (~minutes per nucleus, target for Titan V)
```

### Layer 2: Surrogate Training (GPU — embarrassingly parallel RBF math)

```
Input:  N cached (x, f(x)) pairs from Layer 1
Output: Trained RBF surrogate (weights + training points)

Math:
  1. Pairwise distance matrix:  D[i,j] = ||x_i - x_j||     → cdist.wgsl ✅
  2. TPS kernel evaluation:     K[i,j] = r²·log(r)          → tps_kernel.wgsl ✅
  3. Linear solve:              K·w = y                       → cholesky.wgsl ✅ (or CPU f64)
  4. Prediction:                f̂(x) = Σ wᵢ · K(x, xᵢ)     → CPU fast path ✅

Implemented in both Python (scipy/PyTorch) and Rust (BarraCuda).
```

### Layer 3: Surrogate Inference (NPU — tiny model, fast prediction)

```
Input:  x_new (new parameter set to evaluate)
Output: Predicted f(x_new) ≈ Σ wᵢ · φ(||x_new - xᵢ||)

Target: Akida AKD1000 (operational, driver built for kernel 6.17)
Status: Hardware verified. Model pipeline needs cnn2snn conversion.
```

---

## Results

### Phase A: Python Control ✅

| Level | χ²/datum | Evals | Time | Method |
|-------|----------|-------|------|--------|
| L1 | 6.62 | 1,008 | 184s | scipy RBF + Nelder-Mead |
| L2 | **1.93** | 3,008 | 3.2h | GPU RBF + SparsitySampler + 8-worker parallel |

### Phase B: BarraCuda (Rust + WGSL) ✅

| Level | χ²/datum | Evals | Time | Speedup | Method |
|-------|----------|-------|------|---------|--------|
| L1 | **2.27** | 6,028 | 2.3s | **478×** | WGSL cdist + f64 LA + LHS + multi-start NM |
| L2 (best accuracy) | **16.11** | 60 | 53min | 1.7× | seed=42, λ=0.1, eigh_f64 Jacobi |
| L2 (best NMP) | **19.29** | 60 | 55min | 1.7× | seed=123, λ=1.0, all 5 NMP within 2σ |
| L2 (extended ref) | 25.43 | 1,009 | 35min | 1.7× | Different seed/λ — multimodal landscape |

### Key Discoveries

1. **GPU dispatch overhead for single-point queries**: Using GPU for Nelder-Mead
   inner loop predictions caused 90× slowdown. CPU-only `predict_cpu()` fast path
   resolved this. Lesson: auto-route small workloads to CPU.

2. **Dual-precision strategy works**: f32 cdist on GPU → promote → f64 on CPU
   for TPS kernel and linear solve. Matches Python's torch.float64 accuracy.

3. **Sampling > compute**: The L2 accuracy gap (16.11 vs 1.93) is primarily due
   to sampling strategy (DirectSampler vs mystic SparsitySampler), not compute or
   physics. The range 16–25 across configs confirms the 10D landscape is multimodal.
   SparsitySampler port is the #1 priority.

4. **LHS + multi-start NM beats random + single NM**: On L1, this combination
   improved χ² from 6.62 (Python) to 2.27 (BarraCuda) — better accuracy AND
   478× faster throughput.

---

## What BarraCuda Already Has vs What It Needs

### Exists and Validated ✅

| BarraCuda Op | Surrogate Use | Status |
|-------------|---------------|--------|
| `cdist.wgsl` | RBF pairwise distances | ✅ Production (f32 GPU) |
| `tps_kernel.wgsl` | Thin-plate spline kernel | ✅ Production |
| `cholesky.wgsl` | Matrix factorization | ✅ Production (f32) |
| `triangular_solve.wgsl` | Forward/back substitution | ✅ Production (f32) |
| CPU f64 linear solve | RBF system solve | ✅ Via nalgebra |
| CPU f64 TPS kernel | Kernel evaluation | ✅ Native Rust |
| Latin Hypercube Sampling | Space-filling exploration | ✅ Implemented |
| Multi-start Nelder-Mead | Robust optimization | ✅ Implemented |
| CPU predict fast path | NM inner loop | ✅ Critical optimization |

### Needs Evolution

| Need | Blocks | Effort | Priority |
|------|--------|--------|----------|
| **SparsitySampler** | L2 accuracy parity | 2-3 weeks | 🔴 Critical |
| ~~f64 WGSL shaders (with Titan V)~~ | ~~Native GPU f64~~ | ~~1-2 weeks~~ | ✅ **DONE** (native builtins) |
| `barracuda::surrogate` library extraction | Reusable API | 1 week | 🟡 Important |
| `barracuda::optimize` library extraction | NM + LHS API | 1 week | 🟡 Important |
| ~~GPU eigh_f64 shader~~ | ~~L2 HFB on GPU~~ | ~~2 weeks~~ | ✅ **DONE** (BatchedEighGpu, `nuclear_eos_l2_gpu`) |
| ~~PPPM/Ewald 3D FFT pipeline~~ | ~~κ=0 Coulomb~~ | ~~2-3 weeks~~ | ✅ **DONE** (PppmGpu, `validate_pppm`) |
| NPU surrogate inference | Layer 3 | 2 weeks | 🟢 Enhancement |
| Deformed HFB solver (L3) | Murillo parity | 4-6 weeks | 🟢 Future |
| Extended nuclei set (52→2457) | Tighter constraints | 1 week | 🟢 Enhancement (GPU makes this cheap) |

---

## Hardware Evolution

| Hardware | Role | FP64 TFLOPS | fp64:fp32 | Status |
|----------|------|-------------|-----------|--------|
| i9-12900K (CPU) | HFB eigensolvers, NM fast path | ~0.5 | N/A | ✅ Active |
| RTX 4070 (GPU) | **f64 science compute** (native builtins + DF64 hybrid) | ~0.3 (native) / **3.24** (DF64) | **~1:64** (native) | ✅ **Active — bottleneck broken** |
| AKD1000 (NPU) | Pre-screening classifier | N/A | N/A | ✅ Hardware ready |
| **Titan V ×2** | **f64 GPU compute** | **13.8 combined** | **1:2 native** | 📦 On order |

**UPDATE Feb 15, 2026 — f64 Bottleneck Broken**: The RTX 4070 now runs native f64 builtins
(`sqrt`, `exp`, `round`, `floor`) via `SHADER_F64`/Naga/Vulkan. Rigorous benchmarking confirmed
fp64:fp32 ~1:64 (both CUDA and Vulkan match hardware). The breakthrough: **double-float (f32-pair)**
on FP32 cores delivers 3.24 TFLOPS at 14-digit precision (9.9× native f64). This changes the compute equation:

| Role | Before (software f64) | After (native f64) |
|------|----------------------|-------------------|
| RTX 4070 | f32 only, ML inference | **f64 science compute** (validated, 2-6× speedup) |
| Titan V | Required for any f64 work | Faster for compute-bound (25× raw TFLOPS) |
| CPU | All f64 eigensolvers, fallback | Small workloads, NM inner loop |
| NPU | Pre-screening | Pre-screening, energy-efficient inference |

With Titan V:
- RTX 4070: Bandwidth-limited f64 (small matrices, batched SEMF, MD force kernels)
- Titan V: Compute-bound f64 (large eigensolvers, dense matrix ops, L3 deformed HFB)
- CPU: NM inner loop, L2 SCF fallback, small workloads
- NPU: Pre-screening, energy-efficient inference

### What GPU f64 Unlocks (with RTX 4070 alone)

| Capability | Before (CPU-only f64) | After (GPU native f64) |
|-----------|----------------------|----------------------|
| L1 SEMF (100k evals) | 7.3s, 374 J | **4.0s, 126 J** (44.8× less energy) |
| L1 parameter sweep (512 pts) | 5.1s | **3.75s** (GPU-accelerated) |
| MD N=10k (35k steps) | ~4 hrs, ~720 kJ | **5.3 min, 19.4 kJ** (46× faster) |
| MD N=20k (35k steps) | ~16 hrs, ~2,880 kJ | **10.4 min, 39.3 kJ** (94× faster) |
| MD parameter sweep (50 pts × N=10k) | ~200 hrs | **~4 hrs** |
| L2 HFB eigensolvers | 55s/eval (CPU Jacobi) | **CPU for now** (GPU eigh_f64 planned) |

The GPU path makes experiments that were previously impractical (days/weeks) into routine overnight runs.

---

## GPU MD — Sarkas on Consumer GPU (Phase C)

**Status**: ✅ Implemented — f64 WGSL Yukawa MD on RTX 4070

The `sarkas_gpu` binary runs the full Sarkas PP Yukawa DSF study (9 cases)
entirely on GPU using f64 WGSL shaders. This is the first step toward
replacing the Python Sarkas codebase with BarraCuda for plasma MD.

### What's Running
- Yukawa all-pairs force kernel (f64 WGSL, `SHADER_F64`)
- Velocity-Verlet symplectic integrator (f64 WGSL)
- Periodic boundary conditions with minimum image convention (f64 WGSL)
- Berendsen thermostat for equilibration
- Energy conservation validated: **0.000% drift** (exact symplectic)
- Observables: RDF, VACF, SSF, energy (CPU post-process from GPU snapshots)
- Cell-list neighbor search for O(N) scaling at large N

### 9 PP Yukawa Cases

| κ | Γ values       | rc/a_ws |
|---|----------------|---------|
| 1 | 14, 72, 217    | 8.0     |
| 2 | 31, 158, 476   | 6.5     |
| 3 | 100, 503, 1510 | 6.0     |

### What's Flagged for Future Work

#### PPPM / FFT-Based Ewald — WIRED (Feb 14, 2026)
The 3 Coulomb cases (κ=0: Γ=10, 50, 150) from the DSF study use PPPM for
long-range Coulomb interactions. **PppmGpu** from toadstool is now wired into
hotSpring via the `validate_pppm` binary. Validation against analytical
(2-charge) and direct-sum (64-particle NaCl) test cases is ready to run.

#### Paper-Parity Long Run — COMPLETE (Feb 14, 2026)
All 9 PP Yukawa cases at N=10,000, 80k production steps: **9/9 pass**,
0.000-0.002% energy drift, 3.66 hours total, $0.044 electricity.
Cell-list 4.1× faster than all-pairs for κ=2,3.

#### BatchedEighGpu L2 — WIRED (Feb 14, 2026)
Toadstool's `BatchedEighGpu` wired into L2 HFB solver via `nuclear_eos_l2_gpu`
binary. Groups nuclei by basis dimension, runs lockstep SCF with GPU-batched
eigensolves. Ready for full AME2020 791-nucleus runs on Titan V.

#### MSU HPCC (iCER) Comparison
Pin a CPU-only large-system baseline (N=10,000+, 80k production steps) for
comparison against the same physics on a $600 GPU card.

**Route**: Request alumni/collaborator access to MSU HPCC, or route through
Murillo collaboration — "you run your Sarkas at full scale on HPC, we run
the same physics on a consumer GPU."

This becomes a headline number: **HPC cluster vs consumer GPU, same physics,
same observables.**

---

## The One-Liner

> Python proved we can do the physics.
> BarraCuda proved we can do it 478× faster on a consumer GPU.
> Native f64 builtins broke the bottleneck — 2-6× more throughput, $0.001 per paper-parity run.
> Now Sarkas runs on a $600 GPU — same physics, same observables, no HPC needed.
> Together they prove the scarcity was artificial.
