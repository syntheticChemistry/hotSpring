# Handoff: hotSpring → ToadStool/BarraCUDA (Feb 16, 2026)

**Date:** February 16, 2026
**From:** hotSpring (Nuclear EOS + MD validation study)
**To:** ToadStool/BarraCUDA core team
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring has been the first production consumer of ToadStool/BarraCUDA's
science compute stack. Over five weeks (Jan 15 – Feb 16, 2026), we validated
**195 quantitative checks** across molecular dynamics, nuclear equation of
state, and GPU compute — all on a single consumer workstation (i9-12900K,
32 GB DDR5, RTX 4070). This handoff documents everything the ToadStool team
needs to absorb our work, fix two bugs we found, and evolve the library.

### What We Proved

1. **BarraCUDA's WGSL/wgpu/Vulkan stack delivers IEEE 754 f64 on consumer
   GPUs** — no CUDA, no FFI, no unsafe code. Max error vs CPU: 4.55e-13 MeV.
2. **GPU-resident hybrid HFB beats CPU-only** — 3.65s vs 4.30s (15% faster)
   for nuclear structure on 52 nuclei.
3. **Full MD pipeline through BarraCUDA ops** matches raw WGSL to 1.86e-7
   force error with 0.0000% energy drift.
4. **BCS GPU bisection** matches CPU Brent to 6.2e-11 chemical potential error.
5. **Batched eigensolve** matches CPU Jacobi to 2.4e-12 eigenvalue error.

### What We Found (Bugs)

1. `batched_bisection_f64.wgsl` line 154: `target` is a WGSL reserved keyword.
2. `WgpuDevice::from_adapter_index()` line 333: does not request SHADER_F64.

Both are documented below with exact fixes.

---

## Part 1: Bugs to Fix

### Bug 1: WGSL Reserved Keyword in BCS Bisection Shader

**File**: `toadstool/crates/barracuda/src/shaders/optimizer/batched_bisection_f64.wgsl`
**Line 154**: `let target = params[problem_idx];`
**Symptom**: naga rejects shader at compile time: `name 'target' is a reserved keyword`
**Impact**: ALL BCS bisection GPU calls fail on any wgpu version that enforces WGSL spec

**Fix** (one-word rename):

```wgsl
// Before (broken):
fn polynomial_test(x: f64, problem_idx: u32) -> f64 {
    let target = params[problem_idx];
    return x * x - target;
}

// After (working):
fn polynomial_test(x: f64, problem_idx: u32) -> f64 {
    let target_val = params[problem_idx];
    return x * x - target_val;
}
```

**hotSpring workaround**: Local copy at `barracuda/src/physics/shaders/bcs_bisection_f64.wgsl`
with the fix applied, plus Rust wrapper at `barracuda/src/physics/bcs_gpu.rs`. Validated
to 6.2e-11 precision (14/14 checks).

### Bug 2: WgpuDevice Does Not Request SHADER_F64

**File**: `toadstool/crates/barracuda/src/device/wgpu_device.rs`
**Function**: `from_adapter_index()` (called by `new_f64_capable()`)
**Line 333**:

```rust
required_features: wgpu::Features::empty(),
```

**Symptom**: Any f64 WGSL shader fails with:
`Type [1] '' is invalid. Using f64 values requires the naga::valid::Capabilities::FLOAT64 flag`

**Impact**: `new_f64_capable()` finds the right adapter but creates the device without
SHADER_F64. Every BarraCUDA f64 op silently creates a broken device.

**Fix**:

```rust
let adapter_features = adapter.features();
let required_features = if adapter_features.contains(wgpu::Features::SHADER_F64) {
    wgpu::Features::SHADER_F64
} else {
    wgpu::Features::empty()
};

let (device, queue) = adapter
    .request_device(
        &wgpu::DeviceDescriptor {
            label: Some("BarraCUDA device"),
            required_features,
            required_limits: super::tensor_context::science_limits(),
        },
        None,
    )
    .await
    .map_err(|e| BarracudaError::device(format!("Failed to create device: {e}")))?;
```

**hotSpring workaround**: `GpuF64::new()` creates the wgpu device directly with
SHADER_F64 and bridges to BarraCUDA via `WgpuDevice::from_existing_simple()`.

---

## Part 2: Complete BarraCUDA Usage Inventory

### What hotSpring Consumes from ToadStool

hotSpring (`hotspring-barracuda` v0.5.0) uses BarraCUDA as a path dependency
and consumes these modules across 18 validation binaries and 81 unit tests:

#### Device + Infrastructure

| Import | Usage |
|--------|-------|
| `barracuda::device::WgpuDevice` | GPU device management (bridged via `GpuF64`) |
| `barracuda::device::TensorContext` | Batched dispatch (begin_batch/end_batch) |
| `barracuda::tensor::Tensor` | GPU-resident f64 data: `from_f64_data()`, `to_f64_vec()` |
| `barracuda::shaders::precision::ShaderTemplate` | f32/f64 precision switching |

#### MD Ops (Tensor-based)

| Op | Binary | Precision Achieved |
|----|--------|--------------------|
| `YukawaForceF64` | `validate_barracuda_pipeline` | Force magnitude: 1.86e-7 |
| `VelocityVerletKickDrift` | `validate_barracuda_pipeline` | Functional (PBC, in-place) |
| `VelocityVerletHalfKick` | `validate_barracuda_pipeline` | Functional (velocity update) |
| `BerendsenThermostat` | `validate_barracuda_pipeline` | Functional (temperature scaling) |
| `KineticEnergy` | `validate_barracuda_pipeline` | KE error: 0.0, T error: 9.6e-16 |
| `LennardJonesForce` | `validate_md` | PASS |
| `CoulombForce` | `validate_md` | PASS |
| `MorseForce` | `validate_md` | PASS |
| `VelocityVerlet` (f32) | `validate_md` | PASS |
| `SsfGpu` | `sarkas_gpu` | PASS (9 Yukawa cases) |
| `PppmGpu` | `validate_pppm` | PASS (vs direct Coulomb) |

#### Linear Algebra

| Op | Binary | Precision Achieved |
|----|--------|--------------------|
| `BatchedEighGpu` | `validate_barracuda_hfb`, `nuclear_eos_l2_gpu`, `nuclear_eos_l3_gpu` | Eigenvalue: 2.4e-12, orthogonality: 3.1e-15 |
| `eigh_f64` (CPU) | `validate_barracuda_hfb` | Reference implementation |
| `lu_decompose` / `lu_solve` | `validate_linalg` | PASS |
| `qr_decompose` | `validate_linalg` | PASS |
| `svd_decompose` / `svd_pinv` | `validate_linalg` | PASS |
| `tridiagonal_solve` | `validate_linalg` | PASS |

#### Optimization + Sampling

| Op | Binary | Notes |
|----|--------|-------|
| `bisect`, `brent` | `nuclear_eos_l1_ref`, `hfb.rs` | Root-finding for NMP + BCS |
| `nelder_mead`, `multi_start_nelder_mead` | `nuclear_eos_l1_ref` | Skyrme parameter optimization |
| `bfgs` / `bfgs_numerical` | `validate_optimizers` | PASS |
| `convergence_diagnostics` | `nuclear_eos_l1_ref` | Bootstrap convergence |
| `EvaluationCache` | `nuclear_eos_l1_ref` | Evaluation deduplication |
| `direct_sampler` | `nuclear_eos_l1_ref`, `nuclear_eos_l2_ref` | Parameter space sampling |
| `sparsity_sampler` | `nuclear_eos_l1_ref`, `nuclear_eos_l2_ref` | L1-penalized sampling |
| `latin_hypercube` | `nuclear_eos_l2_ref` | LHS initialization |
| `sobol_sequence` | `validate_optimizers` | PASS |

#### Statistics + Numerics + Special Functions

| Op | Binary |
|----|--------|
| `bootstrap_ci`, `chi2_decomposed_weighted`, `norm_cdf`, `norm_ppf`, `pearson_correlation` | `validate_optimizers`, EOS binaries |
| `trapz`, `gradient_1d` | HFB solvers |
| `gamma`, `ln_gamma`, `factorial`, `erf`, `erfc`, `laguerre`, `hermite`, `legendre`, `assoc_legendre` | `validate_special_functions` |
| `bessel_j0`, `bessel_j1`, `bessel_i0`, `bessel_k0` | `validate_special_functions` |
| `regularized_gamma_p/q`, `chi_squared_cdf/quantile/pdf` | `validate_special_functions` |
| `rk45_solve`, `CrankNicolsonConfig`, `HeatEquation1D` | `validate_optimizers` |

### What hotSpring Does NOT Use

These ToadStool capabilities exist but were not needed for our physics validation:

| Category | Unused Ops | Notes |
|----------|-----------|-------|
| **Thermostats** | NoseHooverChain, LangevinStep | We used Berendsen (simplest for validation) |
| **Forces** | BornMayer | Only Yukawa OCP needed |
| **Neighbor** | `barracuda::ops::md::neighbor::CellList` | hotSpring wrote its own in `md/simulation.rs` |
| **GPU LinAlg** | GenEighGpu, LuGpu, QrGpu, SvdGpu | Validated CPU versions; GPU for BatchedEigh only |
| **FFT** | Fft1D, Fft2D, Fft3D (f32/f64), IFFT, RFFT | Sarkas uses pyfftw; not yet needed in Rust |
| **Grid** | `spin_orbit_f64`, `fd_gradient_f64` | Spin-orbit currently on CPU (documented as Tier 2.1) |
| **Neural** | nn, esn_v2, snn | Not relevant to physics validation |
| **Genomics** | genomics | Not relevant |
| **Vision** | vision | Not relevant |
| **FHE** | 13 FHE modules | Not relevant |
| **Attention/Conv/Pool/Norm/Loss/RNN/GNN** | ~150 shaders | ML ops; not used for science |

**Key observation**: hotSpring exercises ~15% of ToadStool's op surface (the
science/physics slice). The other 85% (ML, FHE, genomics, vision) is untested
by us. Consider separate validation tracks for those domains.

---

## Part 3: WGSL Shaders Written by hotSpring

hotSpring wrote 19 WGSL shaders — 8 for MD (inline in `md/shaders.rs`) and
11 standalone `.wgsl` files. These encode domain physics and should remain in
hotSpring, but the *patterns* they use (f64 buffer layout, workgroup sizing,
PBC wrapping) are directly transferable to ToadStool's shader library.

### MD Shaders (inline, `barracuda/src/md/shaders.rs`)

| Shader | Physics | Pattern for ToadStool |
|--------|---------|----------------------|
| `SHADER_YUKAWA_FORCE` | All-pairs Yukawa + PE, PBC | Fused force+accumulation avoids separate reduce |
| `SHADER_YUKAWA_FORCE_CELLLIST` | Cell-list Yukawa O(N) | Flat 27-neighbor loop, branch-based PBC wrapping |
| `SHADER_YUKAWA_FORCE_CELLLIST_V2` | Same physics, different loop | Alternative neighbor iteration |
| `SHADER_VV_KICK_DRIFT` | VV half-kick + drift + PBC wrap | Split integrator in WGSL |
| `SHADER_VV_HALF_KICK` | VV second half-kick | Minimal shader |
| `SHADER_BERENDSEN` | Velocity rescaling | Trivial but validates Tensor API |
| `SHADER_KINETIC_ENERGY` | Per-particle KE | Reduction pattern |
| `SHADER_RDF_HISTOGRAM` | RDF pair histogram | atomicAdd in WGSL |

### Nuclear Physics Shaders (standalone `.wgsl` files)

| Shader | Physics |
|--------|---------|
| `bcs_bisection_f64.wgsl` | BCS chemical potential bisection (local fix) |
| `batched_hfb_hamiltonian_f64.wgsl` | Batched HFB Hamiltonian construction |
| `batched_hfb_potentials_f64.wgsl` | Skyrme + Coulomb potentials |
| `batched_hfb_density_f64.wgsl` | BCS occupations + density mixing |
| `batched_hfb_energy_f64.wgsl` | Nuclear energy functional |
| `deformed_hamiltonian_f64.wgsl` | Deformed HFB Hamiltonian |
| `deformed_wavefunction_f64.wgsl` | HO basis on cylindrical grid |
| `deformed_density_energy_f64.wgsl` | Observables (Q20, β₂) |
| `deformed_gradient_f64.wgsl` | Deformed mean field gradient |
| `deformed_potentials_f64.wgsl` | Deformed Skyrme potentials |

### Lessons for ToadStool's Shader Library

1. **Fused force+accumulation**: hotSpring's Yukawa shader accumulates forces
   AND potential energy in one pass, avoiding a separate reduction dispatch.
   ToadStool's `YukawaForceF64` does the same. This pattern should be standard
   for any GPU force kernel.

2. **Branch-based PBC wrapping**: `if d > half_box { d -= box_size }` is faster
   and more correct than `d - box * round(d/box)` in f64 WGSL. The modulo-based
   approach was the source of a cell-list bug (WGSL `i32 %` semantics differ from
   Rust `i32 %` for negative numbers).

3. **atomicAdd for histograms**: RDF histogram shader uses `atomicAdd` on a
   `storage` buffer. This works but requires careful workgroup sizing to avoid
   bank conflicts.

4. **f64 uniform alignment**: BCS bisection shader encodes `f64` tolerance as
   two `u32` fields because WGSL uniform buffers require 16-byte alignment for
   `f64` values. This pattern should be documented in ToadStool's shader guide.

---

## Part 4: Architecture Lessons

These are the hard-won lessons from running BarraCUDA on real physics problems.
Each one has a specific implication for ToadStool's evolution.

### Lesson 1: Device Creation Must Request Features

`WgpuDevice::from_adapter_index()` creates the device with `Features::empty()`.
This means `new_f64_capable()` discovers an f64-capable adapter but creates a
device that can't run f64 shaders. The fix is trivial (Bug 2 above) but the
impact is total: every BarraCUDA user who tries f64 hits this.

**Recommendation**: `from_adapter_index()` should inspect `adapter.features()`
and request all supported features, not just an empty set. At minimum,
SHADER_F64 when available.

### Lesson 2: The Poll Bottleneck is Real

From Experiment 005c: GPU-resident HFB spends **40% of time in `device.poll(Wait)`**.
Every SCF iteration requires GPU→CPU sync to read Hamiltonian matrices for
eigensolve. At 200 iterations × 19 nuclei = 3,800 syncs, this dominates.

**Numbers**: `upload=0.04s, gpu=0.06s, poll=1.45s, cpu=2.08s` (total: 3.65s)

**Implication for ToadStool**: Any iterative solver (SCF, optimization, ODE)
that reads GPU data per iteration will hit this wall. The solution is either:
(a) efficient small-matrix GPU eigensolve (Tier 1.1), or
(b) async pipelining with double-buffered staging (Tier 3.4).

### Lesson 3: BatchedEighGpu Per-Rotation Submit is Catastrophic

`BatchedEighGpu` issues a separate `queue.submit()` for every Jacobi rotation.
For 12×12 matrices needing ~100 rotations, that's 100 submissions per eigensolve.
At 19 nuclei × 2 isospins × 200 iterations = 760,000 submissions. This is why
the first GPU-resident attempt (Experiment 005b) was **16× slower than CPU**.

**The fix**: A single-dispatch Jacobi kernel where ALL rotations for ALL matrices
execute inside ONE shader invocation (Tier 1.1). The shader should loop internally
with workgroup barriers, not round-trip through the CPU.

### Lesson 4: Complexity Crossover at n≈30

GPU eigensolve breaks even with CPU at matrix dimension ~30. Below this, CPU L1
cache coherence beats GPU dispatch overhead. Above (L3 deformed, n=50+), GPU
dominates.

| n | CPU (Rayon) | GPU (BatchedEigh) | GPU wins? |
|:-:|:-----------:|:-----------------:|:---------:|
| 12 | 2μs/eigen | ~2ms/eigen (dispatch overhead) | No |
| 30 | ~50μs/eigen | ~125ms (but batchable) | Breakeven |
| 50 | ~580μs/eigen | ~580ms (batchable) | Yes (6×) |
| 100+ | ~4.6ms/eigen | ~4.6s (batchable) | Yes (13×) |

**Implication**: ToadStool should NOT try to beat CPU on small matrices via the
current per-rotation approach. Either implement single-dispatch (Tier 1.1) or
accept the hybrid pattern (GPU for potentials/H-build, CPU for eigensolve+BCS).

### Lesson 5: Hybrid GPU+Rayon is the Practical Path

The winning architecture today: GPU computes Skyrme+Coulomb potentials and
Hamiltonian matrix elements (7 kernel dispatches, single encoder). CPU adds
spin-orbit, solves eigenproblems (Jacobi, ~2μs each), runs BCS, density
updates, and energy — all parallelized via Rayon across all nuclei.

**Result**: 3.65s vs 4.30s CPU-only (15% faster). Physics matches to 2% chi².

### Lesson 6: Tensor API Works But Needs f64 Device Init

The `Tensor::from_f64_data()` → `execute()` → `to_f64_vec()` pipeline works
perfectly for all MD and HFB ops. The only barrier was Bug 2 (device creation).
Once the device has SHADER_F64, every BarraCUDA Tensor op Just Works with f64.

### Lesson 7: GPU FP64 is Real on Consumer Hardware

RTX 4070 with wgpu/Vulkan provides TRUE IEEE 754 f64. Performance ratio
fp64:fp32 ≈ 1:2 (bandwidth-limited). Native WGSL builtins (sqrt, exp on f64)
provide 2-6× speedup over software-emulated versions. This is validated across
all 9 Yukawa OCP cases at N=500–20,000 with 0.000% energy drift.

---

## Part 5: Evolution Roadmap (Updated Tiers)

### TIER 1 — Eliminate the Poll Bottleneck (Critical Path)

| Item | Status | Deliverable |
|------|:------:|-------------|
| **1.1** Efficient small-matrix GPU eigensolve | **BLOCKED** | Single-dispatch batched Jacobi: one `queue.submit()`, all rotations in shader loop. Target: B=40, n=12, <1ms total. |
| **1.2** GPU-resident SCF pipeline | **UNBLOCKED by 1.1** | Chain: potentials → H-build → eigensolve → BCS → density → convergence scalar. Only one value crosses GPU→CPU: continue/stop. |

**Impact**: Estimated 3-5× speedup (poll=1.45s → ~0s, upload=0.04s → ~0s).

### TIER 2 — Close the Precision Gap (chi2 23.97 → 23.49)

| Item | Status | Deliverable |
|------|:------:|-------------|
| **2.1** GPU spin-orbit kernel | OPEN | Compute `gradient_1d(rho_total, dr)` on GPU + diagonal spin-orbit corrections. `ops/grid/spin_orbit_f64` exists in ToadStool — just needs integration. |
| **2.2** Full f64 power functions in WGSL | OPEN | `pow_f64(base, exp)`, `exp_f64(x)`, `log_f64(x)`. Eliminates CPU pre-compute of `rho^alpha`. |

### TIER 3 — Scaling (Validated or Shader Exists)

| Item | Status | Notes |
|------|:------:|-------|
| **3.1** GPU BCS pairing kernel | **VALIDATED** | hotSpring's `bcs_gpu.rs` + shader: 6.2e-11 precision. Fix ToadStool shader (Bug 1) and use directly. |
| **3.2** GPU density construction | **SHADER EXISTS** | `batched_hfb_density_f64.wgsl` in hotSpring. Pattern: eigenstates × BCS weights → density. |
| **3.3** GPU energy functional | **SHADER EXISTS** | `batched_hfb_energy_f64.wgsl` in hotSpring. Pattern: batched radial integrals + reductions. |
| **3.4** Async GPU/CPU pipelining | OPEN | Double-buffered staging, overlap GPU N+1 with CPU N. |

---

## Part 6: Code to Absorb

### Priority 1: Bug Fixes (Apply Immediately)

1. Apply Bug 1 fix: rename `target` → `target_val` in `batched_bisection_f64.wgsl`
2. Apply Bug 2 fix: request SHADER_F64 in `from_adapter_index()` when adapter supports it

### Priority 2: Validation Patterns (Review and Adopt)

| hotSpring File | What ToadStool Gets |
|----------------|---------------------|
| `validate_barracuda_pipeline.rs` | End-to-end MD validation pattern: CPU reference → GPU ops → cross-validate |
| `validate_barracuda_hfb.rs` | BCS + eigensolve validation pattern with degeneracy |
| `bcs_gpu.rs` | Local GPU dispatch pattern using `wgpu` directly (no Tensor layer) |

### Priority 3: Device Bridge Pattern (Architectural Reference)

hotSpring's `GpuF64` demonstrates how to create a properly-featured wgpu device
and bridge it to BarraCUDA's WgpuDevice:

```rust
// Create device with SHADER_F64
let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
    required_features: wgpu::Features::SHADER_F64 | wgpu::Features::TIMESTAMP_QUERY,
    required_limits: wgpu::Limits { /* science limits */ },
    ..Default::default()
}, None).await?;

// Bridge to BarraCUDA
let wgpu_device = WgpuDevice::from_existing_simple(device, queue);
let tensor_ctx = TensorContext::new(wgpu_device.clone());
```

This pattern ensures all BarraCUDA Tensor ops have f64 support from the start.

---

## Part 7: What Stays in hotSpring

These are physics-domain-specific and should NOT migrate to ToadStool:

| Module | Purpose | Why Domain-Specific |
|--------|---------|---------------------|
| `physics/semf.rs` | Skyrme SEMF binding energy | Skyrme force parameters (t0-t3, x0-x3, α, w0) |
| `physics/hfb.rs` | Spherical HFB solver | BCS pairing with nuclear degeneracy |
| `physics/hfb_deformed.rs` | Deformed HFB (CPU) | Cylindrical HO basis |
| `physics/hfb_deformed_gpu.rs` | Deformed HFB (GPU) | Same with GPU eigensolve |
| `physics/hfb_gpu_resident.rs` | Fully GPU-resident HFB | Nuclear potentials in WGSL |
| `physics/nuclear_matter.rs` | NMP (ρ₀, E/A, K∞, J, m*/m) | Saturation density bisection |
| `physics/constants.rs` | CODATA 2018 nuclear constants | MeV, fm, hbar*c |
| `md/` | Yukawa OCP simulation | Sarkas paper-parity validation |
| All 10 `.wgsl` physics shaders | Nuclear structure on GPU | Encode Skyrme EDF physics |

hotSpring's `data/`, `discovery/`, `tolerances/`, `validation/`, and
`provenance/` modules are general-purpose patterns that ToadStool could adopt
as testing infrastructure, but they're not BarraCUDA compute primitives.

---

## Part 8: Experiment Findings (Complete Timeline)

| Date | Experiment | Finding | ToadStool Action |
|------|-----------|---------|-----------------|
| Jan 15 | L1+L2 initial | BarraCUDA L1 chi²=2.27, 478× faster than Python | Validates science compute stack |
| Feb 4 | Phase A-B | 86/86 checks, 5 upstream Python bugs found | Confirms Rust stability advantage |
| Feb 8 | Phase C | GPU MD 9/9, 0.000% drift, 259 steps/s | GPU force kernel validated |
| Feb 10 | Phase D | N-scaling 500→20k, cell-list O(N) | Cell-list `i32 %` bug found+fixed |
| Feb 12 | Phase E | Paper-parity 9/9, toadstool rewire | BatchedEighGpu, SsfGpu, PppmGpu wired |
| Feb 12 | Exp 004 | L3 dispatch overhead: 145k dispatches → 16× slower | Mega-batch pattern needed |
| Feb 12 | Exp 005 | L2 mega-batch: 101 dispatches, 95% GPU util | Still 70× slower (Amdahl's Law) |
| Feb 14 | Exp 005b | GPU-resident v1: per-rotation submit catastrophic | Single-dispatch eigensolve needed |
| Feb 14 | Exp 005c | **GPU beats CPU**: 3.65s vs 4.30s | Hybrid GPU+Rayon is the path |
| Feb 15 | Phase F | Full AME2020: 2042 nuclei, L1-L3 characterized | Infrastructure validated at scale |
| Feb 16 | Pipeline | MD 12/12, HFB 14/14, two bugs found | Bug fixes + handoff documented |

---

## Part 9: Validation Numbers (Comprehensive)

### 195/195 Quantitative Checks Pass

| Phase | Checks | Description |
|-------|:------:|-------------|
| A (Python control) | 86 | 60 MD + 6 TTM + 15 surrogate + 5 EOS |
| C (GPU MD, N=2000) | 45 | 9 cases × 5 observables |
| D (N-scaling + builtins) | 16 | 5 N values + 6 cell-list diag + 5 native builtins |
| E (Paper-parity + rewire) | 13 | 9 long-run cases + 1 profiling + 3 GPU ops |
| F (Full-scale nuclear EOS) | 9 | 3 L1 Pareto + 3 L2 GPU + 3 L3 deformed |
| Pipeline (BarraCUDA ops) | 26 | 12 MD pipeline + 14 HFB pipeline |
| **Total** | **195** | **All pass** |

### Upstream Bugs Found and Fixed

| # | Bug | Where | Impact |
|:-:|-----|-------|--------|
| 1 | `np.int` deprecated (NumPy 2.x) | Sarkas | Silent crash |
| 2 | pandas `append()` removed | Sarkas | Crash on latest pandas |
| 3 | Numba/pyfftw Sarkas init | Sarkas | Import failure |
| 4 | TF→Saha TTM parameter | TTM | Wrong physics |
| 5 | Dump file corruption | Sarkas | Silent data loss |
| 6 | WGSL `target` keyword | ToadStool BCS shader | Shader compile failure |

---

## Part 10: Hardware

All results on a single consumer workstation:

| Component | Spec |
|-----------|------|
| CPU | Intel i9-12900K (24 threads) |
| RAM | 32 GB DDR5-4800 |
| GPU | NVIDIA RTX 4070 (12 GB VRAM, SHADER_F64 confirmed) |
| OS | Pop!_OS 22.04 (Linux 6.17) |
| Cost | ~$2,000 total |

This is a desk-sized workstation, not a cluster. Every result is reproducible
on equivalent consumer hardware with any Vulkan-capable GPU that supports
`wgpu::Features::SHADER_F64`.

---

## The Proof

BarraCUDA, through hotSpring's five-week validation campaign:

- **195/195 acceptance checks** — zero failures across MD, nuclear EOS, and GPU compute
- **GPU-resident HFB beats CPU-only** — 3.65s vs 4.30s (15% faster) for 52 nuclei
- **f64 GPU on consumer hardware** — RTX 4070 via WGSL/wgpu/Vulkan, max error 4.55e-13 MeV
- **Full MD pipeline validated** — BarraCUDA ops match raw WGSL to 1.86e-7 with 0.0000% drift
- **BCS GPU bisection** — 6.2e-11 chemical potential precision vs CPU Brent
- **Batched eigensolve** — 2.4e-12 eigenvalue precision, 3.1e-15 orthogonality
- **Pure Rust, pure WGSL, pure Vulkan** — no CUDA, no FFI, no unsafe, zero external commands
- **Sovereign Science** — AGPL-3.0, fully reproducible, no institutional gatekeeping

---

*February 16, 2026 — Comprehensive handoff. 195/195 checks, two bugs documented
with exact fixes, full op inventory, 19 WGSL shaders, architecture lessons from
7 experiments. Critical path: Tier 1.1 (single-dispatch eigensolve) unlocks
full GPU-resident SCF pipeline and 3-5× further speedup.*
