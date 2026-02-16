# Handoff: hotSpring → Toadstool/BarraCUDA

**Date:** February 16, 2026 (revised)
**From:** hotSpring (Nuclear EOS validation study)
**To:** Toadstool/BarraCUDA core team
**License:** AGPL-3.0-or-later

---

## STATED GOAL: Pure GPU Faster Than CPU

hotSpring's L2 GPU mega-batch (Experiment 005) confirmed: **CPU is 70x faster
than GPU** for spherical HFB with 12×12 matrices. GPU utilization is 95%,
dispatch count is down to 101 — the dispatch overhead pathology is solved.
The remaining bottleneck is Amdahl's Law: the eigensolve is 1% of the SCF
iteration; the other 99% (Hamiltonian construction, BCS pairing, density
updates) still runs on CPU.

**The fix is not better batching. It is moving ALL physics to GPU.**

Target: GPU-resident SCF loop where data never leaves VRAM until convergence.
Zero CPU↔GPU round-trips during iteration. Estimated time: ~40s for 791 nuclei
(vs CPU's 35s), surpassing CPU at larger basis sizes. This is the architecture
ToadStool needs to enable.

---

## 1. What hotSpring Proved

| Capability | Validated By | Result |
|-----------|-------------|--------|
| f64 GPU on consumer hardware | 169/169 acceptance checks | Science-grade precision |
| Batched eigensolve (f64 Jacobi) | 2,042 nuclei × 200 SCF iterations | Correct to 10⁻⁶ MeV |
| Mega-batch dispatch | 791 HFB nuclei, 1 dispatch/iter | 101 total dispatches, 95% GPU util |
| GPU utilization ≠ efficiency | Exp 004 (79% util, 16x slower) | Dispatch overhead diagnosed |
| Amdahl's Law boundary | Exp 005 (95% util, 70x slower) | CPU physics dominates |
| MD Yukawa + Coulomb | Sarkas paper parity | ΔE < 10⁻⁸, ΔT < 1% |
| PPPM electrostatics | kappa=0 pure Coulomb | Matches Ewald to 0.1% |

---

## 2. The Complexity Boundary (Key Finding)

For HFB eigensolves on matrices of size n×n:

| n | GPU compute/dispatch | Dispatch overhead | GPU wins? |
|:-:|:--------------------:|:-----------------:|:---------:|
| 12 (current L2) | ~8 ms | ~50 ms | **No** (14% compute) |
| 30 | ~125 ms | ~50 ms | Marginal (71%) |
| **50 (L3 target)** | **~580 ms** | **~50 ms** | **Yes (92%)** |
| 100+ (beyond MF) | >4.6 s | ~50 ms | Dominant |

**Below n≈30**: CPU cache coherence beats GPU parallelism. nalgebra solves
a 12×12 eigenproblem in ~5 μs (L1 cache resident). GPU dispatch overhead
is 10,000x larger than the compute it replaces.

**Above n≈50**: GPU's massively parallel Jacobi sweeps dominate. This is
where L3 deformed, beyond-mean-field, and future methods live.

**For n<30**: The ONLY way GPU wins is by running the ENTIRE iteration on GPU
— eliminating all CPU↔GPU round-trips. This is what ToadStool must enable.

---

## 3. L2 Performance History

| Implementation | Dispatches | Wall Time | Per HFB | GPU Util |
|----------------|:----------:|:---------:|:-------:|:--------:|
| CPU-only (nalgebra) | 0 | **35.1s** | 44ms | — |
| GPU v1 (5 groups/iter) | 206 | 66.3 min | 5,029ms | ~80% |
| **GPU v2 (mega-batch)** | **101** | **40.9 min** | **3,104ms** | **95%** |
| **Target (GPU-resident)** | **~101** | **~40s** | **~50ms** | **>95%** |

Physics output is identical across all substrates (chi2=224.52, NMP=0.63).

---

## 4. What ToadStool Needs to Build (Priority Order)

### CRITICAL — GPU-Resident Physics Pipeline

These five items, together, enable the GPU-resident SCF loop that makes
pure GPU faster than CPU:

#### 4.1 Multi-Kernel Pipeline Without CPU Round-Trips

Chain dependent GPU operations: output of shader 1 = input of shader 2,
no intermediate CPU readback. The SCF iteration is:

```
H-build → eigensolve → BCS → density → convergence check
```

Each step's output feeds the next. Currently each step requires CPU readback
+ re-upload. ToadStool's `begin_batch()` / `end_batch()` is the foundation;
the gap is **dependent op chaining** where shader A's output buffer becomes
shader B's input buffer without CPU involvement.

**Deliverable**: `PipelineBuilder` API or `begin_batch()` extension that
accepts a DAG of dependent ops with shared buffer handles.

#### 4.2 GPU Hamiltonian Construction Kernel

The Hamiltonian is built element-wise on a radial grid:

```
H[i,j] = T_eff[i,j] + ∫ φ_i(r) · V(ρ,τ,J; params) · φ_j(r) · r²dr
```

This is a weighted inner product over grid points — embarrassingly parallel
per matrix element. ToadStool already has `weighted_dot_f64`. The extension:
a **batched grid-quadrature GEMM** where the weight function V depends on
density profiles that change each SCF iteration.

**Deliverable**: `ops/physics/grid_quadrature_gemm_f64` — or a composable
pattern using existing `GemmF64` + `weighted_dot_f64` + grid evaluation.

#### 4.3 GPU BCS Pairing Kernel

BCS pairing requires a root-finding step (Brent/bisection) to find the
chemical potential μ such that particle number is conserved:

```
Σ_k v²_k(μ) = N,  where v²_k = ½(1 - (ε_k - μ)/√((ε_k - μ)² + Δ²))
```

This is per-nucleus, per-isospin — 1,582 independent bisection problems.
Each is ~20 iterations of a scalar function evaluation.

**Deliverable**: `ops/optimize/batched_bisection_f64` — batch of independent
1D root-finding problems on GPU. Input: function parameters per problem.
Output: root per problem.

#### 4.4 GPU Convergence Reduction

The SCF loop checks max|E_new - E_old| across all nuclei. Only a single
scalar (continue/stop) needs to come back to CPU.

**Deliverable**: `ops/reduce/max_abs_diff_f64` — returns a single f64 (the
maximum absolute difference). This is a trivial extension of `SumReduceF64`
but with `abs(a-b)` and `max` instead of `+`.

#### 4.5 Persistent Buffer Management for Iterative Solvers

The SCF loop runs 100-200 iterations. All nucleus data (wavefunctions,
densities, potentials, eigenvalues) should be allocated ONCE at startup and
reused across iterations. Current pattern re-creates buffers per dispatch.

**Deliverable**: `BufferPool` "pin for solver lifetime" mode — allocate a
named buffer set at solver start, reuse across N dispatches, release at end.
The existing `BufferPool` in `tensor_context.rs` is close; needs explicit
"pin" / "release" lifecycle.

### Previously Completed (Confirmed Working)

| Primitive | Status | Location |
|-----------|:------:|----------|
| Broyden mixing (f64) | DONE | `ops/mixing/broyden_f64` |
| Weighted inner product (f64) | DONE | `ops/reduce/weighted_dot_f64` |
| Hermite/Laguerre (f64 GPU) | DONE | `shaders/special/` |
| FD gradient (f64) | DONE | `ops/grid/fd_gradient_f64` |
| Science buffer limits (512 MiB / 1 GiB) | DONE | `science_limits()` |
| BatchedEighGpu (mega-batch validated) | DONE | `ops/linalg/batched_eigh_gpu` |

---

## 5. What Stays in hotSpring (Physics-Specific)

These encode nuclear physics and should NOT migrate to toadstool:

| Shader | Purpose | Why Physics-Specific |
|--------|---------|---------------------|
| `batched_hfb_potentials_f64.wgsl` | Skyrme nuclear potential | t0-t3, x0-x3, α parameters |
| `batched_hfb_hamiltonian_f64.wgsl` | HFB Hamiltonian assembly | Nuclear effective mass, l(l+1) |
| `batched_hfb_density_f64.wgsl` | BCS pairing + density | Nuclear pairing gaps |
| `batched_hfb_energy_f64.wgsl` | Nuclear energy functional | Skyrme E_t0, E_t3, Coulomb |
| `deformed_potentials_f64.wgsl` | Deformed mean field | Skyrme + Coulomb on cylindrical grid |
| `deformed_density_energy_f64.wgsl` | Observables | Q20 quadrupole moment, β₂ |

These demonstrate that ToadStool's GPU primitives support arbitrary domain
physics. The same pattern applies to fluid dynamics, protein folding, ray
tracing — any domain that needs iterative PDE/eigenvalue solvers on GPU.

---

## 6. Resource Partitioning Vision (Strandgate)

### Already in ToadStool

| Component | Location | Status |
|-----------|----------|--------|
| `enumerate_adapters()` | `device/wgpu_device.rs` | Working |
| `from_adapter_index(n)` | `device/wgpu_device.rs` | Working |
| `Substrate` classification | `device/substrate.rs` | Working |
| `HardwareManager` | `device/toadstool_integration.rs` | Working |
| `BufferPool` | `device/tensor_context.rs` | Working |
| `AutoTuner` | `device/autotune.rs` | Working |

### Needed for Strandgate

| Component | Description | Priority |
|-----------|------------|----------|
| `ResourceQuota` | Per-task VRAM budget enforcement | Medium |
| `ComputePartition` | Fraction of GPU for a task | Medium |
| `WorkloadRouter` | Route tasks to best available device | Medium |
| `MultiDevicePool` | Manage heterogeneous GPU array | Low |

---

## 7. Evolution Summary: What Each Experiment Taught

| Experiment | Key Finding | ToadStool Action |
|------------|------------|-----------------|
| 004: L3 Dispatch Overhead | 145k dispatches → 16x slower than CPU | Mega-batch (DONE) |
| 005: L2 Mega-Batch | 101 dispatches, 95% util, still 70x slower | GPU-resident pipeline (NEW) |
| 005: Complexity Boundary | n<30 CPU wins, n>50 GPU wins | Multi-kernel chaining (NEW) |
| MD (Phase E) | GPU 50-100x faster than CPU | Large-N work suits GPU (confirmed) |
| f64 Validation | wgpu SHADER_F64 = 1:2 fp64:fp32 | Consumer GPU viable (confirmed) |

The pattern: dispatch overhead is solved. The next evolution is **eliminating
CPU physics from the iteration loop entirely**. Every component that remains
on CPU during the hot loop is wasted potential.

---

## 8. The Proof

BarraCUDA, through hotSpring's validation:

- **f64 GPU compute works on consumer hardware** (RTX 4070, $599)
- **169/169 acceptance checks** — zero failures across all physics domains
- **GPU-first architecture validated** — dispatch overhead diagnosed and fixed
- **Complexity boundary characterized** — small matrices (CPU wins) vs large
  matrices (GPU wins), with GPU-resident pipeline as the universal solution
- **Any domain can build on this** — the math primitives are domain-agnostic
- **Sovereign Science** — AGPL-3.0, fully reproducible, no institutional access

---

*Revised: February 16, 2026 — Added Experiment 005 complexity boundary
analysis. Refocused priorities on GPU-resident physics pipeline. Previous
items 6-10 completed. New critical items 4.1-4.5 define the path to
pure GPU faster than CPU.*
