# hotSpring v0.6.0 / v0.6.1 → ToadStool/BarraCUDA: Consolidated Handoff

**Date:** 2026-02-21
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Supersedes:** `HOTSPRING_V0516_CONSOLIDATED_HANDOFF_FEB20_2026.md` (archived)

> **Note:** v0.6.1 builds on v0.6.0 with additional code quality evolution
> (modular tolerances, pedantic lints eliminated, shared Wood-Saxon impl,
> provenance completeness). This handoff covers both releases.

---

## Executive Summary

hotSpring v0.6.0 / v0.6.1 is a hardened, audit-clean release. The v0.5.16 → v0.6.0 bump
reflects a full code quality audit: **zero `.expect()` and zero `.unwrap()` in
library code**, full `Result` propagation across all GPU and simulation APIs,
146 centralized tolerances with zero inline magic numbers, 16 determinism tests
for every stochastic algorithm, and idiomatic `Arc::clone` everywhere.

The science portfolio is unchanged — **18 papers reproduced, 33/33 validation
suites, 463 unit tests**. What changed is the Rust code quality: the crate is
now ready for absorption and evolution without needing cleanup first.

### What Changed: v0.5.16 → v0.6.0

| Change | Before (v0.5.16) | After (v0.6.0) |
|--------|-----------------|----------------|
| `.expect()` in library code | ~10 calls | **0** — all converted to `Result` + `?` |
| `.unwrap()` in library code | 0 | 0 (maintained) |
| `HotSpringError` variants | 5 | **7** (+`GpuCompute`, +`InvalidOperation`) |
| Tolerance constants | 122 | **146** (+24 ESN, BCS, phase boundary) |
| Inline magic numbers | ~8 in validation bins | **0** — all wired to `tolerances::*` |
| `Arc::clone` idiom | Mixed `.clone()` | **All `Arc::clone(&...)`** |
| Determinism tests | 0 | **16** (ESN, HMC, Anderson 2D/3D, MD forces, MD lattice) |
| SSF CPU tests | 0 | **4** (empty, multi-frame, k-spacing, single-particle) |
| Unit tests | 441 | **454** |
| Clippy `expect_used` | not enforced | **zero warnings** |

### What Changed: v0.6.0 → v0.6.1

| Change | Before (v0.6.0) | After (v0.6.1) |
|--------|-----------------|----------------|
| `#![deny(clippy::expect_used)]` | warnings only | **crate-level deny** |
| `tolerances.rs` | 1384-line monolith | **`tolerances/` module tree** (5 submodules) |
| `const fn` library functions | 0 | **13** (lattice, spectral accessors) |
| Redundant `.clone()` | 5 in library | **0** |
| `cast_lossless` warnings | 76 | **0** — all `From` conversions |
| `unreadable_literal` | 18 | **0** — underscore-separated |
| `missing_errors_doc` | 21 | **0** — all `# Errors` documented |
| `items_after_statements` | 21 | **0** |
| `needless_pass_by_value` | 5 | **0** — `&Arc<WgpuDevice>` pattern |
| Wood-Saxon density | 3 duplicate impls | **1 shared** `hfb_common::initial_wood_saxon_density` |
| GPU energy pipeline | dead allocation | **feature-gated** (`gpu_energy`) |
| Provenance records | missing 2 | **all** baselines traced |
| Unit tests | 454 | **463** (+2 Wood-Saxon, +3 summary, +4 discovery) |

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
| `WgslOptimizer` | `barracuda::shaders` | Shader optimization (loop unrolling, instruction reordering) |

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
| `CellListGpu` | `barracuda::ops::md::neighbor` | Fixed in Session 25 — local workaround deprecated |

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

### GPU Cell-List (Deprecated — Upstream Fixed)

**Location:** `md/celllist.rs` + `md/shaders/`
**Status:** Deprecated. Upstream `CellListGpu` fixed in ToadStool Session 25.
Migration to upstream API is next evolution step (removes ~400 lines + 3 WGSL shaders).

### Custom MD Force Shaders (Local — Production Path)

**Location:** `md/shaders.rs` + `md/shaders/`
Optimized for GPU-resident stateful simulation (minimal readback). Parallel to
upstream `ops::md::forces` but operates on raw `wgpu::Buffer` rather than `Tensor`.
Both paths produce identical physics (validated in CPU/GPU parity).

### Screened Coulomb Solver (Local — Specialized)

**Location:** `physics/screened_coulomb.rs`
Sturm bisection eigensolve for tridiagonal Hamiltonian. Specialized atomic physics
(Murillo & Weisheit 1998). Could become `barracuda::ops::spectral` if needed.

### Lattice QCD Stack (Local — GPU Promotion Ready)

**Location:** `lattice/` (~3,300 lines across 10 modules)

| Module | Lines | GPU Status |
|--------|-------|------------|
| `complex_f64.rs` | 316 | WGSL template in source → absorbed as `complex_f64.wgsl` |
| `su3.rs` | 460 | WGSL template in source → absorbed as `su3.wgsl` |
| `wilson.rs` | 338 | GPU shader absorbed: `wilson_plaquette_f64.wgsl` |
| `hmc.rs` | 350 | GPU shader absorbed: `su3_hmc_force_f64.wgsl` |
| `abelian_higgs.rs` | ~500 | GPU shader absorbed: `higgs_u1_hmc_f64.wgsl` |
| `dirac.rs` | 297 | GPU shader exists: Dirac SpMV validated on GPU |
| `cg.rs` | 214 | GPU CG validated: solution parity 4.10e-16 |
| `eos_tables.rs` | 307 | CPU-only data tables |
| `multi_gpu.rs` | 237 | CPU-threaded dispatch |
| `constants.rs` | ~50 | LCG PRNG constants shared by all lattice modules |

### Spectral Theory Stack (Local)

**Location:** `spectral/` (6 modules)

| Module | Purpose |
|--------|---------|
| `tridiag.rs` | Sturm bisection eigensolve |
| `csr.rs` | CsrMatrix + SpMV (CPU + GPU WGSL shader) |
| `lanczos.rs` | Lanczos algorithm + tridiagonal extraction |
| `anderson.rs` | Anderson 1D/2D/3D + Lyapunov exponent |
| `hofstadter.rs` | Almost-Mathieu + Hofstadter butterfly |
| `stats.rs` | Level spacing ratio, band detection |

### ESN / Reservoir Computing (Local)

**Location:** `md/reservoir.rs`
Echo State Network for transport prediction (D*, η*, λ*). Includes `NpuSimulator`
for substrate-independent validation. `predict()` now returns `Result` (v0.6.0).

### Shared Wood-Saxon Density (v0.6.1)

**Location:** `physics/hfb_common.rs`
`hfb_common::initial_wood_saxon_density()` provides a single shared implementation
used by all HFB paths (spherical, deformed, GPU). Replaces three duplicate impls.

---

## Part 3: Error Handling Architecture (v0.6.0)

`HotSpringError` enum with 7 variants — all GPU pipelines, HFB solvers, and ESN
prediction propagate errors via `Result<T, HotSpringError>` and `?` operator.

| Variant | When |
|---------|------|
| `NoAdapter` | No compatible GPU found |
| `NoShaderF64` | GPU lacks `SHADER_F64` feature |
| `DeviceCreation(String)` | wgpu device creation failed |
| `DataLoad(String)` | File I/O or parse error |
| `GpuCompute(String)` | Buffer map, dispatch, or readback failed |
| `InvalidOperation(String)` | State error (e.g. ESN predict before train) |
| `Barracuda(BarracudaError)` | Propagated from upstream primitives |

Functions that now return `Result`:
- `hfb::solve()`, `hfb::binding_energy_l2()`
- `hfb_deformed::solve()`, `hfb_deformed::binding_energy_l3()`
- `hfb_gpu::binding_energies_l2_gpu()`
- `hfb_deformed_gpu::binding_energies_l3_gpu()`, `binding_energies_l3_gpu_auto()`
- `reservoir::EchoStateNetwork::predict()`

---

## Part 4: Paper Portfolio (18 Papers, 33 Suites)

| # | Paper | Domain | Checks | Rust vs Python | Key Result |
|---|-------|--------|--------|:--------------:|------------|
| 1 | Sarkas Yukawa OCP MD | Plasma MD | 60/60 | 82× GPU | N=10k, 80k steps, $0.044 |
| 2 | Two-Temperature Model | Plasma | 6/6 | CPU only | TTM equilibration |
| 3 | Diaw et al. Surrogate | ML | 15/15 | CPU only | Benchmark functions |
| 4 | Nuclear EOS (AME2020) | Nuclear | L1-L3 | 478× | 2,042 nuclei, GPU-batched HFB |
| 5 | Stanton-Murillo Transport | Plasma | 13/13 | CPU only | Green-Kubo D*/η*/λ* |
| 6 | Murillo-Weisheit Screening | Atomic | 23/23 | 2274× | Sturm eigensolve |
| 7 | HotQCD EOS Tables | Lattice | Thermo | CPU only | Reference data |
| 8 | Pure Gauge SU(3) | Lattice | 12/12 | CPU only | HMC, Dirac CG |
| 9-12 | GPU Lattice QCD | Lattice | 34/34 | 200× | GPU SpMV, Lanczos, Dirac, CG, pure GPU QCD |
| 13 | Abelian Higgs | Lattice | 17/17 | 143× | U(1)+Higgs HMC |
| — | Spectral (Kachkovskiy) | Spectral | 45/45 | CPU+GPU | Anderson 1D/2D/3D, Hofstadter, GPU SpMV+Lanczos |
| — | NPU (metalForge) | Hardware | 68/68 | — | Quantization, beyond-SDK, pipeline, lattice NPU, hetero monitor |

---

## Part 5: Codebase Health (v0.6.0 / v0.6.1)

| Metric | Value |
|--------|-------|
| Unit tests | **463** pass (458 + 5 GPU-ignored) |
| Determinism tests | **16** (all stochastic algorithms) |
| Validation suites | **33/33** pass |
| Clippy warnings | **0** (pedantic + `expect_used` + `unwrap_used`) |
| Doc warnings | **0** |
| `.expect()` in library | **0** |
| `.unwrap()` in library | **0** |
| `unsafe` blocks | **0** |
| Centralized tolerances | **146** in `tolerances/` module tree |
| Pedantic lints resolved | **141** (5 categories eliminated) |
| Inline magic numbers | **0** in validation binaries |
| SPDX compliance | **All** 106 `.rs` + 34 `.wgsl` files |
| Provenance records | All targets traced to Python origins or DOIs |
| Coverage | ~63% overall / ~96% unit-testable library |

```bash
cargo test               # 463 pass, 5 GPU-ignored
cargo clippy --all-targets -- -W clippy::expect_used -W clippy::unwrap_used  # 0 warnings
cargo doc --no-deps      # 0 warnings
cargo run --release --bin validate_all  # 33/33 suites
```

---

## Part 6: GPU Promotion Roadmap

### Absorbed by ToadStool (Sessions 18-25)

| Component | ToadStool Shader | Status |
|-----------|-----------------|--------|
| Complex f64 | `complex_f64.wgsl` | Absorbed |
| SU(3) matrix ops | `su3.wgsl` | Absorbed |
| Wilson plaquette | `wilson_plaquette_f64.wgsl` | Absorbed |
| HMC force | `su3_hmc_force_f64.wgsl` | Absorbed |
| Abelian Higgs HMC | `higgs_u1_hmc_f64.wgsl` | Absorbed |
| CellListGpu | Fixed (Session 25) | Local deprecated |
| GPU FFT f64 | `Fft1DF64` / `Fft3DF64` | 14 tests pass |

### Remaining Promotion Targets

| Component | CPU Module | Priority | Effort |
|-----------|-----------|----------|--------|
| GPU SpMV (CSR) | `spectral/csr.rs` | P1 | Low — WGSL shader exists |
| GPU Lanczos | `spectral/lanczos.rs` | P1 | Medium |
| GPU Dirac SpMV | `lattice/dirac.rs` | P1 | Medium — pattern matches CSR |
| Local GpuCellList migration | `md/celllist.rs` | P1 | Low — upstream fixed |
| ESN `export_weights()` on `esn_v2` | `md/reservoir.rs` | P2 | Low |

---

## Part 7: Key Patterns for ToadStool

### Stateful Pipeline Pattern

All physics workloads confirm:
1. Upload initial state to GPU buffers (once)
2. Iterate: dispatch kernel chain N times on same buffers
3. Reduce scalar result within same compute pass
4. Read back minimal bytes (8-16 bytes per iteration)
5. CPU decision (Metropolis, thermostat, convergence)
6. Repeat

Applies to: MD (Verlet), HFB (SCF), lattice HMC (leapfrog + Metropolis),
eigensolve (Jacobi), CG solver (iterations).

### Wirtinger Derivatives for Complex Field HMC

`dp/dt = -2 ∂S/∂φ*` — the factor of 2 is critical for any complex-field HMC.
Missing it causes |ΔH| >> 1 and 0% acceptance. Baked into `higgs_u1_hmc_f64.wgsl`.

### Quantization Error Budget

| Precision | Max Error vs f64 | Physics Use |
|-----------|:---:|------------|
| f32 | <0.001% | Production — identical for all practical purposes |
| int8 | <5% | Production — within MD statistical uncertainty |
| int4 | <30% | Screening only — not for final predictions |

---

## Part 8: Companion Handoffs

| Document | Scope |
|----------|-------|
| `HOTSPRING_TOADSTOOL_REWIRE_V3_FEB20_2026.md` | Session 25 absorption audit, CellListGpu fix, FFT f64 |
| `HOTSPRING_METALFORGE_NPU_HANDOFF_FEB20_2026.md` | AKD1000 NPU discoveries, 10 overturned SDK assumptions |

### Archived (16 prior handoffs)

All in `archive/`. Historical technical detail preserved for archaeology.

---

*License: AGPL-3.0-only. All discoveries, code, and documentation are
sovereign community property. No proprietary dependency required.*
