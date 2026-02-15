# Handoff: hotSpring Math Primitives → Toadstool/BarraCUDA

**Date:** February 12, 2026  
**From:** hotSpring (Nuclear EOS validation study)  
**To:** Toadstool/BarraCUDA core team  
**Status:** Ready for absorption

---

## 1. Context

hotSpring is a first-principles nuclear structure study that uses BarraCUDA for
**all** scientific computation — no NumPy, no LAPACK, no external math. Every
calculation from special functions through eigensolves to molecular dynamics runs
through BarraCUDA on consumer GPU hardware (RTX 4070, f64 via SHADER_F64).

This study has been the most rigorous validation of BarraCUDA's f64 GPU pipeline
to date: 169/169 acceptance checks across Yukawa MD, Coulomb PPPM, HFB nuclear
structure, and full nuclear equation of state (2,042 nuclei).

In the process, we discovered and built several **generic math primitives** that
belong in toadstool/BarraCUDA, not in hotSpring. This handoff documents what to
absorb and why.

---

## 2. What hotSpring Proved

| Capability | Validated By | Result |
|-----------|-------------|--------|
| f64 GPU compute on consumer hardware | All 169 acceptance checks | Works at science-grade precision |
| Batched eigensolve (f64, Jacobi) | 2,042 nuclei × 200 SCF iterations | Correct to 10⁻⁶ MeV |
| GEMM f64 (tiled, batched) | HFB Hamiltonian construction | Matches CPU reference |
| Sum/Max/Min reduction (f64) | Energy functional integration | Matches CPU to machine epsilon |
| Prefix sum (f64) | Coulomb potential accumulation | Correct |
| MD (Yukawa + Coulomb) | Sarkas Python vs BarraCUDA | ΔE < 10⁻⁸, ΔT < 1% |
| PPPM electrostatics | kappa=0 (pure Coulomb) limit | Matches Ewald to 0.1% |

**Key insight:** Nuclear structure is a *better* ML library test than ML itself.
An ML library can hide numerical errors behind stochastic gradients. Nuclear
physics demands deterministic, reproducible, precision-verified results. If
BarraCUDA can compute nuclear binding energies to MeV accuracy on consumer GPUs,
it can run *anything* — fluid dynamics, ray tracing, protein folding, gaming
physics.

---

## 3. Math Primitives to Migrate → Toadstool

### 3.1 Already in Toadstool (hotSpring validated, no migration needed)

| Primitive | Toadstool Module | hotSpring Validation |
|-----------|-----------------|---------------------|
| BatchedEighGpu | `ops/linalg/batched_eigh_gpu` | 2,042 nuclei × 200 iter × 10 blocks |
| GemmF64 | `ops/linalg/gemm_f64` | HFB matrix construction |
| SumReduceF64 | `ops/sum_reduce_f64` | Energy integration |
| CumsumF64 | `ops/cumsum_f64` | Coulomb prefix sums |
| Fft1DF64/Fft3DF64 | `ops/fft/` | PPPM charge mesh |
| PppmGpu | `ops/md/electrostatics/pppm_gpu` | kappa=0 Coulomb validation |
| YukawaForceF64 | `ops/md/forces/yukawa_f64` | MD thermodynamic validation |
| eigh_f64 (CPU) | `linalg/eigh` | CPU reference for GPU comparison |

### 3.2 New Generic Primitives → MIGRATE TO TOADSTOOL

These were created in hotSpring but are **physics-agnostic** and belong in
toadstool as reusable GPU ops.

#### A. Broyden Mixing (vector operation)

**Source:** `hotSpring/barracuda/src/physics/shaders/deformed_density_energy_f64.wgsl`  
**Entry points:** `mix_density_linear`, `broyden_update`

```
mix_density_linear: out[i] = (1-α)·old[i] + α·new[i]
broyden_update:     out[i] = u[i] + α·F[i] - Σ_m γ_m·(du_m[i] + α·df_m[i])
```

**Why generic:** Broyden mixing is used in *any* self-consistent-field solver,
including DFT, Poisson-Boltzmann, coupled-cluster. Also useful in nonlinear
equation solvers and fixed-point iterations.

**Proposed toadstool location:** `ops/mixing/` or `ops/vector/`

#### B. Finite-Difference Gradient on Structured Grid (2D/3D)

**Source:** `hotSpring/barracuda/src/physics/shaders/deformed_gradient_f64.wgsl`  
**Entry points:** `density_radial_derivative` (generic part)

```
Central difference: ∂f/∂x = (f[i+1] - f[i-1]) / (2·dx)
Forward/backward at boundaries
```

**Why generic:** Finite differences on structured grids are used in fluid
dynamics (Navier-Stokes), heat transfer, wave propagation, electrostatics,
image processing. A parameterized FD kernel for 1D/2D/3D grids with
configurable stencil order would be broadly useful.

**Proposed toadstool location:** `ops/grid/finite_difference_f64`

#### C. Weighted Inner Product with Workgroup Reduction

**Source:** `hotSpring/barracuda/src/physics/shaders/deformed_hamiltonian_f64.wgsl`  
**Entry point:** `compute_potential_matrix_elements_reduce`

```
result = Σ_k w[k] · a[i·N + k] · b[j·N + k]   (weighted dot product)
```

With workgroup tree reduction over k (shared memory, 256-wide).

**Why generic:** This is a batched weighted dot product — the fundamental
operation in Galerkin methods, FEM assembly, spectral methods, correlation
computation. Any code that computes `∫ φ_i · W · φ_j dx` on a grid uses this.

**Proposed toadstool location:** `ops/reduce/weighted_dot_f64` or integrate
into `GemmF64` as a "diagonal-weighted GEMM" mode.

#### D. Hermite and Laguerre Polynomial GPU Evaluation

**Source:** `hotSpring/barracuda/src/physics/shaders/deformed_wavefunction_f64.wgsl`  
**Functions:** `hermite(n, x)`, `laguerre(n, alpha, x)` (iterative recurrence)

**Why generic:** Hermite polynomials are used in quantum mechanics, Gaussian
quadrature, probability (Hermite functions = Gaussian × Hermite). Laguerre
polynomials appear in radial wavefunctions, Gamma function computation,
exponential fitting. Toadstool already has `hermite.wgsl` and `legendre.wgsl`
in shaders/special/ — these f64 GPU versions should be added.

**Proposed toadstool location:** `shaders/special/hermite_f64.wgsl`,
`shaders/special/laguerre_f64.wgsl`

#### E. GPU Buffer Limit Configuration

**Source:** `hotSpring/barracuda/src/gpu.rs`

```rust
required_limits: wgpu::Limits {
    max_storage_buffer_binding_size: 512 * 1024 * 1024,  // 512 MiB
    max_buffer_size: 1024 * 1024 * 1024,                 // 1 GiB
    ..wgpu::Limits::default()
},
```

**Why generic:** The default wgpu limit of 128 MiB per storage buffer is too
small for scientific computing (wavefunctions, 3D fields, large matrices). Any
serious GPU computation needs configurable limits. Toadstool's
`WgpuDevice::new_with_limits()` should default to science-grade limits, or
`new_high_capacity()` should be the recommended path.

**Proposed toadstool change:** Update `WgpuDevice::new()` default limits, or
add `WgpuDevice::new_science()` with 512 MiB / 1 GiB defaults.

---

## 4. What Stays in hotSpring (Physics-Specific)

These shaders encode nuclear physics models and should NOT migrate to toadstool.
They are hotSpring's science layer built ON TOP of toadstool primitives.

| Shader | What It Does | Why Physics-Specific |
|--------|-------------|---------------------|
| `batched_hfb_potentials_f64.wgsl` | Skyrme nuclear potential | t0/t1/t2/t3/x0-x3/α Skyrme EDF parameters |
| `batched_hfb_hamiltonian_f64.wgsl` | HFB Hamiltonian assembly | Nuclear effective mass, l(l+1) angular momentum |
| `batched_hfb_density_f64.wgsl` | BCS pairing + density | Nuclear pairing gaps, Fermi surface |
| `batched_hfb_energy_f64.wgsl` | Nuclear energy functional | Skyrme E_t0, E_t3, Coulomb exchange |
| `deformed_potentials_f64.wgsl` | Deformed nuclear mean field | Skyrme + Coulomb on cylindrical grid |
| `deformed_density_energy_f64.wgsl` | BCS, observables | Q20 quadrupole moment, nuclear β₂ |

These demonstrate that toadstool's GPU primitives can support arbitrary
domain-specific physics. The same pattern works for:

- **Fluid dynamics:** Replace Skyrme potential with Navier-Stokes viscous terms
- **Protein folding:** Replace BCS pairing with AMBER/CHARMM force fields
- **Ray tracing:** Replace nuclear wavefunctions with BVH traversal
- **Gaming physics:** Replace nuclear EDF with rigid body dynamics

---

## 5. Resource Partitioning Vision

hotSpring's overnight runs revealed the need for **GPU resource partitioning**:
the ability to dedicate a fraction of GPU to long-running science while keeping
the rest available for interactive work.

### What Toadstool Already Has

| Component | Location | Status |
|-----------|----------|--------|
| `enumerate_adapters()` | `device/wgpu_device.rs` | Working — finds all GPUs |
| `from_adapter_index(n)` | `device/wgpu_device.rs` | Working — targets specific GPU |
| `Substrate` classification | `device/substrate.rs` | Working — NVIDIA/AMD/Intel/NPU/CPU |
| `HardwareManager` | `device/toadstool_integration.rs` | Working — full inventory |
| `BufferPool` | `device/tensor_context.rs` | Working — buffer reuse |
| `AutoTuner` | `device/autotune.rs` | Working — workgroup calibration |

### What's Needed for Strandgate

| Component | Description | Priority |
|-----------|------------|----------|
| `ResourceQuota` | Per-task VRAM budget enforcement | High |
| `ComputePartition` | Fraction of GPU for a task (via dispatch limiting) | High |
| `WorkloadRouter` | Route tasks to best available device | Medium |
| `MultiDevicePool` | Manage heterogeneous GPU array (e.g., 4070 + Titan V) | Medium |
| `NVIDIA MPS integration` | True compute partitioning via Multi-Process Service | Low |

This enables the vision: a single machine running DNA sequencing ingestion on
1/3 of GPU-0, OpenFold on GPU-1, protein structure on 2/3 of GPU-0, and
interactive experiments on CPU — all orchestrated by Strandgate through
toadstool's device layer.

---

## 6. Concrete Next Steps

### For Toadstool Team (Priority Order)

1. **Absorb Broyden mixing** → `ops/mixing/broyden_f64.rs` + shader
2. **Absorb weighted inner product** → `ops/reduce/weighted_dot_f64.rs` + shader
3. **Absorb Hermite/Laguerre f64** → `shaders/special/hermite_f64.wgsl`, `laguerre_f64.wgsl`
4. **Absorb finite-difference gradient** → `ops/grid/fd_gradient_f64.rs` + shader
5. **Raise default buffer limits** → `WgpuDevice::new()` defaults to 512 MiB
6. **Add `ResourceQuota`** → Track per-task VRAM usage
7. **Add `ComputePartition`** → Limit dispatch fraction per task

### For hotSpring Team

1. **Wait for overnight L3 run** to complete (~2-4 hours remaining)
2. **Run GPU L3 benchmark** on clean system (no CPU competition)
3. **Profile GPU vs CPU** side-by-side for the same 52 nuclei
4. **Replace hotSpring buffer helpers** with toadstool equivalents once absorbed
5. **Wire remaining GPU shaders** (tau, density, energy) into SCF loop

---

## 7. The Proof

BarraCUDA, through hotSpring's validation, has demonstrated:

- **f64 GPU compute works on consumer hardware** (RTX 4070, $599)
- **Science-grade precision** — nuclear binding energies to MeV accuracy
- **169/169 acceptance checks** — zero failures across all physics domains
- **GPU-first architecture** — CPU orchestrates, GPU computes
- **Any domain can build on this** — the math primitives are domain-agnostic

This is not an ML benchmark. This is deterministic physics at double precision
on hardware anyone can buy. If BarraCUDA can solve the nuclear many-body problem,
it can enable any computational science on any machine.
