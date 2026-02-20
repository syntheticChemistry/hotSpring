# hotSpring v0.5.16 → ToadStool/BarraCUDA: Consolidated Handoff

**Date:** 2026-02-20
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** All 14 prior handoffs (archived to `archive/`)

---

## Executive Summary

hotSpring has completed its Tier 0-2 science portfolio: **9 papers reproduced**,
**320 unit tests**, **18/18 validation suites**, **~$0.20 total compute cost** on
consumer hardware. This handoff consolidates everything the toadstool/barracuda
team needs to absorb and evolve from hotSpring's validated workloads.

The key deliverable since the last consolidated handoff: **Paper 13 (Abelian
Higgs)** — U(1) gauge + complex scalar Higgs field on a (1+1)D lattice with HMC.
Rust CPU is **143x faster than Python**. This completes the lattice gauge theory
pipeline from SU(3) pure gauge → U(1) Higgs → quantum simulation bridge.

### What Changed Since Last Handoff

| Change | Impact |
|--------|--------|
| Paper 13: Abelian Higgs implemented + validated | 17/17 checks, new HMC pattern with complex fields |
| Wirtinger derivative force for complex scalar | New calculus pattern for any complex-field HMC |
| 320 unit tests (was 281) | +39 tests from Abelian Higgs |
| 18/18 validation suites (was 16/16) | +2 suites (screened Coulomb, Abelian Higgs) |
| 68 centralized tolerances (was 58) | +10 from new physics domains |
| PHYSICS.md expanded | Sections 13 (Screened Coulomb) and 14 (Abelian Higgs) added |
| All 14 prior handoffs archived | This document is the single source of truth |

---

## Part 1: Complete BarraCUDA Primitive Usage

Every upstream barracuda primitive hotSpring consumes, organized by domain.

### GPU Device & Shader Infrastructure

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `WgpuDevice` | `barracuda::device` | All GPU paths — MD, HFB, eigensolve, PPPM, SSF |
| `TensorContext` | `barracuda::device` | GPU tensor allocation |
| `GpuDriverProfile` | `barracuda::device::capabilities` | Hardware-accurate ILP scheduling, NAK workarounds |
| `ShaderTemplate` | `barracuda::shaders::precision` | All WGSL compilation routes through `for_driver_profile()` |
| `Tensor` | `barracuda::tensor` | GPU buffer abstraction for MD and HFB |

### Pipeline & Reduction

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `ReduceScalarPipeline` | `barracuda::pipeline` | KE/PE sum-reduction in MD (both all-pairs and cell-list) |

### Molecular Dynamics

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `YukawaForceF64` | `barracuda::ops::md::forces` | Validation binary (Tensor API path) |
| `CoulombForce`, `LennardJonesForce`, `MorseForce` | `barracuda::ops::md::forces` | Validation binaries |
| `VelocityVerletHalfKick`, `VelocityVerletKickDrift` | `barracuda::ops::md::integrators` | Validation binary |
| `BerendsenThermostat` | `barracuda::ops::md::thermostats` | Validation binary |
| `KineticEnergy` | `barracuda::ops::md::observables` | Validation binary |
| `SsfGpu` | `barracuda::ops::md::observables` | Production: static structure factor |
| `PppmGpu`, `PppmParams` | `barracuda::ops::md::electrostatics` | PPPM Ewald validation |

### Linear Algebra & Eigensolve

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `BatchedEighGpu` | `barracuda::ops::linalg` | GPU-batched HFB (791 nuclei), multi-GPU benchmarks |
| `eigh_f64` | `barracuda::linalg` | CPU HFB, deformed HFB, NAK eigensolve validation |
| `lu_decompose`, `lu_solve` | `barracuda::ops::linalg` | Linear algebra validation |
| `qr_decompose` | `barracuda::ops::linalg` | Linear algebra validation |
| `svd_decompose`, `svd_pinv` | `barracuda::ops::linalg` | Linear algebra validation |
| `tridiagonal_solve` | `barracuda::ops::linalg` | Linear algebra validation |
| `compute_ls_factor` | `barracuda::ops::grid` | HFB radial grid |
| `gradient_1d`, `trapz` | `barracuda::numerical` | HFB density integration |

### Special Functions

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `hermite`, `factorial` | `barracuda::special` | HFB harmonic oscillator basis |
| `gamma`, `laguerre` | `barracuda::special` | HFB/deformed-HFB basis functions |

### Optimization & Sampling

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `brent` | `barracuda::optimize` | HFB Fermi energy root-finding |
| `bfgs`, `bfgs_numerical` | `barracuda::optimize::bfgs` | Optimizer validation |
| `nelder_mead`, `multi_start_nelder_mead` | `barracuda::optimize` | Nuclear EOS parameter search |
| `bisect` | `barracuda::optimize` | General root-finding |
| `direct_sampler` | `barracuda::sample::direct` | Nuclear EOS L1/L2 sampling |
| `latin_hypercube` | `barracuda::sample` | Nuclear EOS parameter space |
| `sparsity_sampler` | `barracuda::sample::sparsity` | Nuclear EOS sparse search |
| `sobol_sequence` | `barracuda::sample::sobol` | Quasi-random validation |
| `convergence_diagnostics` | `barracuda::optimize` | Nuclear EOS convergence |

### Statistics

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `chi2_decomposed_weighted` | `barracuda::stats` | Nuclear EOS fit quality |
| `bootstrap_ci` | `barracuda::stats` | Confidence intervals |
| `norm_cdf`, `norm_ppf`, `pearson_correlation` | `barracuda::stats` | Optimizer validation |

### PDE & Numerical

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `rk45_solve` | `barracuda::numerical::rk45` | ODE validation |
| `CrankNicolsonConfig`, `HeatEquation1D` | `barracuda::pde` | PDE validation |

### Surrogate

| Primitive | Import Path | hotSpring Usage |
|-----------|-------------|-----------------|
| `RBFKernel`, `RBFSurrogate`, `EvaluationCache` | `barracuda::surrogate` | Nuclear EOS L2 hetero |

---

## Part 2: Local Implementations (Not Upstream)

These are things hotSpring built locally that either fill gaps in barracuda
or diverge from the upstream pattern.

### GPU Cell-List (Local — barracuda's `CellListGpu` Has Bugs)

**Location:** `md/celllist.rs` + `md/shaders/`

barracuda's `CellListGpu` has a binding mismatch in the prefix-sum scan and
a `cell_idx` bug in `yukawa_celllist_f64.wgsl` (lines 49-54: `((cx % nx) + nx) % nx`
is broken on NVIDIA/Naga/Vulkan due to WGSL `i32 %` truncation semantics).

hotSpring's local GPU cell-list uses a 3-pass approach:
1. `cell_bin_f64.wgsl` — bin particles into cells (atomic increment)
2. `exclusive_prefix_sum.wgsl` — compute cell start offsets
3. `cell_scatter.wgsl` — scatter particles to sorted order

The force kernel `yukawa_force_celllist_indirect_f64.wgsl` uses `sorted_indices`
for neighbor lookup while keeping positions/velocities/forces in original order.

**Recommendation:** Fix the upstream `CellListGpu` bugs (binding mismatch +
`i32 %` truncation), then hotSpring can delete its local implementation.
The 3-pass pattern (bin → prefix sum → scatter) is sound and can be adopted.

### Custom MD Force Shaders (Local)

**Location:** `md/shaders.rs`

hotSpring maintains its own WGSL force kernels:
- `SHADER_YUKAWA_FORCE` — all-pairs Yukawa (production path)
- `SHADER_YUKAWA_FORCE_CELLLIST` variants — cell-list force with indirect indexing
- `SHADER_VELOCITY_VERLET`, `SHADER_BERENDSEN` — integrator/thermostat

These parallel barracuda's `ops::md::forces::YukawaForceF64` etc. but operate
on raw `wgpu::Buffer` rather than `Tensor`. Both paths produce identical physics
(validated in CPU/GPU parity checks). The local shaders are optimized for the
stateful simulation pattern (GPU-resident buffers, minimal readback).

### Screened Coulomb Solver (Local — No Upstream Equivalent)

**Location:** `physics/screened_coulomb.rs`

Sturm bisection eigensolve for tridiagonal Hamiltonian — O(N) per eigenvalue.
This is specialized atomic physics (Murillo & Weisheit 1998), not a general
barracuda primitive. Could become a `barracuda::ops::spectral` module if other
Springs need Sturm-Liouville eigenproblems.

### Lattice QCD Stack (Local — Pure CPU, GPU Promotion Ready)

**Location:** `lattice/` (~3,300 lines across 10 modules)

| Module | Lines | Purpose | GPU Status |
|--------|-------|---------|------------|
| `complex_f64.rs` | 316 | Complex f64 arithmetic | WGSL template included in source |
| `su3.rs` | 460 | SU(3) matrix algebra | WGSL template included in source |
| `wilson.rs` | 338 | Wilson gauge action, plaquettes, staples | Needs WGSL shader |
| `hmc.rs` | 350 | HMC with Cayley exponential | Needs WGSL shader |
| `dirac.rs` | 297 | Staggered Dirac operator | Needs WGSL shader |
| `cg.rs` | 214 | Conjugate gradient for D†D | Needs WGSL shader |
| `eos_tables.rs` | 307 | HotQCD EOS reference data | CPU-only (data) |
| `abelian_higgs.rs` | ~500 | U(1) gauge + Higgs HMC | Needs WGSL shader |
| `multi_gpu.rs` | 237 | Temperature scan dispatcher | CPU-threaded, GPU-ready |
| `constants.rs` | ~50 | LCG PRNG constants | Shared by all lattice modules |

The lattice stack uses zero barracuda imports — it's pure Rust math. This is
intentional: validate the math on CPU first, then promote to GPU via WGSL
templates that are already embedded in `complex_f64.rs` and `su3.rs`.

---

## Part 3: Key Discoveries for ToadStool Evolution

### 1. Wirtinger Derivatives for Complex Field HMC (NEW)

Paper 13 required HMC over a complex scalar field φ(x). The force calculation
for the Higgs momentum requires Wirtinger derivatives:

```
dp/dt = -2 ∂S/∂φ*
```

The factor of 2 is critical. It arises because `dp/dt = -(∂S/∂φ_R + i ∂S/∂φ_I)`
and the Wirtinger derivative `∂/∂φ* = ½(∂/∂φ_R + i ∂/∂φ_I)`. Missing this
factor causes HMC to produce |ΔH| >> 1 and 0% acceptance.

**Implication for ToadStool:** Any future GPU kernel that does HMC over complex
fields (lattice QCD with dynamical fermions, condensed matter BCS, etc.) must
include this factor of 2. It's easy to miss because real-field HMC doesn't have it.

### 2. Gauge Link Force Sign Convention

For U(1) gauge links U = exp(iθ), the derivative of the hopping term:

```
d/dθ Re[φ*(x) e^{iθ} φ(x+μ̂)] = -Im[φ*(x) e^{iθ} φ(x+μ̂)]
```

The force is `F = -dS/dθ`, so the Higgs contribution to the link force is
`+2κ Im(hop)`. Getting this sign wrong (using `+Im` instead of `-Im` in the
derivative) causes the link to diverge from equilibrium.

### 3. Stateful Pipeline Pattern Confirmed

The MD and lattice HMC workloads both confirm the `StatefulPipeline` pattern:

1. Upload initial state to GPU buffers (once)
2. Iterate: dispatch kernel chain N times on same buffers
3. Dispatch reduction (sum, norm) within same compute pass
4. Copy scalar result to staging buffer
5. Read back minimal bytes (8-16 bytes per iteration)
6. Decision on CPU (Metropolis accept/reject, thermostat adjust)
7. Repeat

This pattern applies to: MD (Verlet loop), HFB (SCF iteration), lattice HMC
(leapfrog + Metropolis), eigensolve (Jacobi sweeps), CG solver (iterations).
It is the dominant compute pattern for physics simulations.

### 4. GPU Sum-Reduction is Universal

Every physics workload hotSpring runs needs the pattern:
- N per-element values → tree reduction → 1 scalar
- MD: total KE, total PE
- HFB: density norm, energy convergence
- Lattice: plaquette average, Polyakov loop
- Eigensolve: residual norm
- CG solver: inner product

`ReduceScalarPipeline` is the most-used upstream primitive after `WgpuDevice`.
Promoting it to a first-class API (e.g., `pipeline.reduce_sum(buffer, n)`) would
eliminate boilerplate across all Springs.

### 5. NAK Shader Compiler Deficiencies (Still Open)

The 149× performance gap between proprietary NVIDIA and NVK (nouveau) on
loop-heavy f64 kernels is a NAK issue, not hardware. Five specific deficiencies
identified and documented in `archive/NVK_EIGENSOLVE_PERF_ANALYSIS_FEB18_2026.md`:

1. No loop unrolling (4× of the gap)
2. Register spills to local memory
3. Source-order instruction scheduling (no latency hiding)
4. No f64 FMA fusion
5. No shared memory bank conflict avoidance

NAK is written in Rust (Mesa). Community patches are feasible. Loop unrolling
alone would close ~4× of the 9× gap for compute-bound kernels.

hotSpring delivered a NAK-optimized eigensolve shader (`batched_eigh_nak_optimized_f64.wgsl`)
with manual workarounds for all 5 deficiencies. Same API as existing shader —
ToadStool can absorb by replacing the WGSL file.

### 6. CellListGpu Bug Still Open

barracuda's `CellListGpu` has two bugs:
1. **Binding mismatch** in prefix-sum scan dispatch
2. **`i32 %` truncation** in `yukawa_celllist_f64.wgsl` — `((cx % nx) + nx) % nx`
   produces wrong cell indices on NVIDIA/Naga/Vulkan

hotSpring works around this with a local 3-pass GPU cell-list. Fix the upstream
bugs and all Springs can use `CellListGpu` directly.

---

## Part 4: Paper Portfolio (9/9 Complete)

| # | Paper | Domain | Checks | Rust vs Python | Key Result |
|---|-------|--------|--------|:--------------:|------------|
| 1 | Sarkas Yukawa OCP MD | Plasma MD | 9/9 PP cases | 82× GPU | N=10k, 80k steps, $0.044 |
| 2 | Two-Temperature Model | Plasma | 6/6 | CPU only | TTM equilibration |
| 3 | Diaw et al. Surrogate | ML | 15/15 | CPU only | Benchmark functions |
| 4 | Nuclear EOS (AME2020) | Nuclear | 195/195 | 478× | 2,042 nuclei, GPU-batched HFB |
| 5 | Stanton-Murillo Transport | Plasma | 13/13 | CPU only | Green-Kubo D*/η*/λ* |
| 6 | Murillo-Weisheit Screening | Atomic | 23/23 | 2274× | Sturm eigensolve |
| 7 | HotQCD EOS Tables | Lattice | Thermo | CPU only | Reference data |
| 8 | Pure Gauge SU(3) | Lattice | 12/12 | CPU only | 4^4 HMC, 96-100% acceptance |
| 13 | Abelian Higgs | Lattice | 17/17 | 143× | U(1)+Higgs, Wirtinger forces |

**Total: 300+ quantitative validation checks, ~$0.20 total compute cost.**

---

## Part 5: Codebase Health

| Metric | Value |
|--------|-------|
| Unit tests | **320** pass, 5 GPU-ignored (325 total) |
| Validation suites | **18/18** pass (CPU); GPU suites require hardware |
| Clippy warnings | **0** (default + pedantic on library code) |
| Doc warnings | **0** |
| Unsafe blocks | **0** |
| TODO/FIXME/HACK | **0** |
| Centralized tolerances | **68** constants in `tolerances.rs` |
| Provenance records | All targets traced to Python origins or DOIs |
| AGPL-3.0 compliance | All `.rs` and `.wgsl` files |

---

## Part 6: GPU Promotion Roadmap

### What's Ready Now (WGSL Templates Exist)

| Component | CPU Module | WGSL Template | Effort to Promote |
|-----------|-----------|---------------|-------------------|
| Complex f64 | `lattice/complex_f64.rs` | Embedded in source | Low — extract, compile, validate |
| SU(3) matrix ops | `lattice/su3.rs` | Embedded in source | Low — extract, compile, validate |

### What Needs WGSL Shaders

| Component | CPU Module | Blocking | Effort |
|-----------|-----------|----------|--------|
| Wilson plaquette + force | `lattice/wilson.rs` | Nothing | Medium |
| HMC leapfrog + Cayley exp | `lattice/hmc.rs` | Wilson shader | Medium |
| U(1) Abelian Higgs HMC | `lattice/abelian_higgs.rs` | Complex f64 shader | Medium |
| Staggered Dirac D†D | `lattice/dirac.rs` | SU(3) + Complex f64 | High |
| CG solver | `lattice/cg.rs` | Dirac shader | High |

### Blocked on Missing Primitives

| Primitive | Needed For | Priority |
|-----------|-----------|----------|
| **FFT (3D, f64)** | Full QCD with dynamical fermions (Tier 3 papers) | P0 |
| **SpMV (sparse matrix-vector)** | Large-lattice Dirac operator | P1 |
| **Lanczos eigensolve** | Spectral theory, large sparse matrices | P1 |

### Recommended Promotion Order

1. **Complex f64 WGSL** — already has template, validate on GPU
2. **SU(3) WGSL** — already has template, validate on GPU
3. **Wilson plaquette shader** — enables GPU SU(3) HMC
4. **U(1) Higgs force shader** — enables GPU Abelian Higgs HMC
5. **CG solver shader** — enables GPU Dirac inversion
6. **FFT** — unlocks full QCD (Tier 3 papers)

---

## Part 7: Evolution Lessons

### What Worked

1. **Python control → Rust CPU → GPU pipeline.** Every paper followed this
   exact sequence. Python catches physics bugs early. Rust CPU validates
   pure math. GPU validates portability. No shortcuts.

2. **Centralized tolerances.** Every numerical threshold in one file
   (`tolerances.rs`) with doc-comments explaining the physics. This caught
   3 bugs where ad-hoc thresholds masked failures.

3. **`ValidationHarness` for structured checks.** Standardized pass/fail
   reporting with named checks. Made it trivial to add new papers.

4. **LCG PRNG for bitwise determinism.** Matching the same LCG constants
   between Python and Rust enables exact trajectory comparison. Critical for
   debugging force calculation bugs (Wirtinger sign, factor of 2).

5. **Unidirectional readback pattern.** GPU sum-reduction cutting readback
   from 160 KB to 16 bytes per dump at N=10,000. This is the pattern.

### What Didn't Work (First Time)

1. **CellListGpu upstream bugs.** Wasted ~2 hours debugging before building
   local GPU cell-list. Should have validated upstream with a minimal test
   case first.

2. **Wirtinger factor of 2.** Took significant debugging to find. The HMC
   acceptance rate was 0% with |ΔH| >> 1. The fix was one line: multiply
   the Higgs force by 2. Complex-field HMC is subtle.

3. **Daligault fit coefficients.** Published Table I coefficients were in
   different reduced units than Sarkas. Required 12-point recalibration.
   Lesson: never trust published fit coefficients without checking units.

### What Springs Need from ToadStool

| Need | Why | Priority |
|------|-----|----------|
| Fix `CellListGpu` bugs | All Springs with short-range forces | High |
| `reduce_sum()` convenience API | Eliminate boilerplate in every physics loop | Medium |
| NAK loop unrolling patch | 4× speedup for all f64 kernels on NVK | Medium |
| `StatefulPipeline` for HMC | Lattice QCD GPU promotion | Medium |
| FFT primitive | Full QCD (Tier 3 papers), PPPM improvements | High |

---

## Part 8: Archived Handoffs

All 14 prior handoffs have been moved to `wateringHole/handoffs/archive/`.
They contain detailed technical specifics that may be useful for archaeology
but are superseded by this document for active development.

| Archived File | Original Date | Topic |
|---------------|:------------:|-------|
| `HANDOFF_HOTSPRING_TO_TOADSTOOL_FEB_12_2026.md` | Feb 12 | GPU-resident HFB |
| `TOADSTOOL_EVOLUTION_REVIEW_FEB14_2026.md` | Feb 14 | ToadStool pull review |
| `TOADSTOOL_CELLLIST_BUG_ALERT.md` | Feb 15 | Cell-list `i32 %` bug |
| `HANDOFF_HOTSPRING_TO_TOADSTOOL_FEB_16_2026.md` | Feb 16 | First production use |
| `HANDOFF_HOTSPRING_BARRACUDA_V055.md` | Feb 16 | Code quality hardening |
| `TOADSTOOL_MULTI_GPU_NVK_HANDOFF_FEB17_2026.md` | Feb 17 | Multi-GPU NVK (superseded) |
| `HOTSPRING_MULTI_GPU_BENCHMARK_FEB17_2026.md` | Feb 17 | Multi-GPU benchmark (superseded) |
| `HOTSPRING_BARRACUDA_FULL_GPU_HANDOFF_FEB17_2026.md` | Feb 17 | Full GPU validation |
| `NVK_EIGENSOLVE_PERF_ANALYSIS_FEB18_2026.md` | Feb 18 | NAK perf decomposition |
| `HOTSPRING_GPU_SOVEREIGNTY_HANDOFF_FEB18_2026.md` | Feb 18 | GPU sovereignty path |
| `HOTSPRING_COMPREHENSIVE_HANDOFF_FEB19_2026.md` | Feb 19 | Consolidated (prev) |
| `HOTSPRING_LATTICE_QCD_HANDOFF_FEB19_2026.md` | Feb 19 | Lattice QCD audit |
| `HOTSPRING_TOADSTOOL_ABSORPTION_HANDOFF_FEB19_2026.md` | Feb 19 | Absorption handoff |
| `HOTSPRING_UNIDIRECTIONAL_FEEDBACK_FEB19_2026.md` | Feb 19 | Unidirectional pipeline |

---

## Part 9: Validation Summary

### Full Suite (Feb 20, 2026)

```
cargo test --lib                                     # 320 pass, 5 GPU-ignored
cargo run --release --bin validate_all -- --skip-gpu  # 18/18 CPU suites
cargo clippy -- -D warnings                          # 0 warnings
```

| # | Suite | Checks | Status |
|---|-------|:------:|:------:|
| 1 | Special Functions | pass | PASS |
| 2 | Linear Algebra | pass | PASS |
| 3 | Optimizers & Numerics | pass | PASS |
| 4 | MD Forces & Integrators | pass | PASS |
| 5 | Nuclear EOS (Pure Rust) | pass | PASS |
| 6 | HFB Verification (SLy4) | pass | PASS |
| 7 | WGSL f64 Builtins | pass | PASS (GPU) |
| 8 | BarraCUDA HFB Pipeline | pass | PASS (GPU) |
| 9 | BarraCUDA MD Pipeline | pass | PASS (GPU) |
| 10 | PPPM Coulomb/Ewald | pass | PASS (GPU) |
| 11 | CPU/GPU Parity | pass | PASS (GPU) |
| 12 | NAK Eigensolve | pass | PASS (GPU) |
| 13 | N-Scaling Benchmark | pass | PASS (GPU) |
| 14 | HotQCD EOS | pass | PASS |
| 15 | Pure Gauge SU(3) | pass | PASS |
| 16 | Stanton-Murillo Transport | pass | PASS |
| 17 | Screened Coulomb | pass | PASS |
| 18 | Abelian Higgs | pass | PASS |

---

*License: AGPL-3.0-only. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required.*
