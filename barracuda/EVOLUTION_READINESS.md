# Evolution Readiness: Rust → WGSL Shader Promotion

This document maps each Rust module to its GPU shader readiness tier
and identifies blockers for full GPU-resident pipeline promotion.

## Evolution Path

```
Python baseline → Rust validation → GPU acceleration → sovereign pipeline
```

## Tier Definitions

| Tier | Label | Meaning |
|------|-------|---------|
| **A** | Rewire | Shader exists and is validated; wire into pipeline |
| **B** | Adapt | Shader exists but needs modification (API, precision, layout) |
| **C** | New | No shader exists; must be written from scratch |

## Physics Modules

| Rust Module | WGSL Shader(s) | Tier | Status | Blocker |
|-------------|----------------|------|--------|---------|
| `physics/semf.rs` | `SHADER_SEMF_BATCH`, `SHADER_CHI2` (inline in `nuclear_eos_gpu.rs`) | **A** | GPU pipeline exists | None — production-ready |
| `physics/hfb.rs` | `batched_hfb_*.wgsl` (4 shaders) via `hfb_gpu.rs` | **A** | GPU pipeline exists | None — validated against CPU |
| `physics/hfb_gpu.rs` | Uses `BatchedEighGpu::execute_single_dispatch` | **A** | Production GPU — single-dispatch (v0.5.3) | None — all rotations in one shader |
| `physics/bcs_gpu.rs` | `bcs_bisection_f64.wgsl` | **A** | Production GPU — pipeline cached (v0.5.3) | None — ToadStool `target` bug absorbed (`0c477306`) |
| `physics/hfb_gpu_resident.rs` | `batched_hfb_potentials_f64.wgsl`, `batched_hfb_hamiltonian_f64.wgsl`, `BatchedEighGpu`, `SpinOrbitGpu` | **A** | GPU H-build + eigensolve + spin-orbit (v0.5.6) | BCS/density still CPU |
| `physics/hfb_deformed.rs` | — | **C** | CPU only | Deformed HFB needs new shaders for 2D grid Hamiltonian build |
| `physics/hfb_deformed_gpu.rs` | `deformed_*.wgsl` (5 shaders exist, not all wired) | **B** | Partial GPU | H-build on CPU; deformed Hamiltonian shaders exist but unwired. Needs `GenEighGpu` (Ax=λBx) for overlap matrix |
| `physics/nuclear_matter.rs` | — | **C** | CPU only | Uses `barracuda::optimize::bisect` (CPU); no NMP shader. Low priority — fast on CPU |
| `physics/hfb_common.rs` | — | N/A | Shared utilities | Pure CPU helpers (WS radii, deformation estimation) |
| `physics/constants.rs` | — | N/A | Physical constants | Data only |

## MD Modules

| Rust Module | WGSL Shader(s) | Tier | Status | Blocker |
|-------------|----------------|------|--------|---------|
| `md/simulation.rs` | Yukawa (all-pairs + cell-list), VV integrator, Berendsen thermostat, KE reduction, RDF histogram (inline in `md/shaders.rs`) | **A** | Full GPU pipeline | None — production-ready |
| `md/shaders.rs` | 8 WGSL shaders (5 extracted to `src/md/shaders/`, 3 inline) | **A** | Production | v0.5.3: 5 large shaders extracted to `.wgsl` files |
| `md/observables.rs` | Uses `SsfGpu` from BarraCUDA | **A** | SSF on GPU; RDF/VACF CPU post-process | RDF/VACF are cheap post-processing; GPU not needed |
| `md/cpu_reference.rs` | — | N/A | Validation reference | Intentionally CPU-only for baseline comparison |
| `md/config.rs` | — | N/A | Configuration | Data structures only |

## WGSL Shader Inventory

### Physics Shaders (`src/physics/shaders/`, 10 files, ~1950 lines)

| Shader | Lines | Pipeline Stage |
|--------|-------|----------------|
| `batched_hfb_density_f64.wgsl` | 148 | Density + BCS + energy for spherical HFB |
| `batched_hfb_potentials_f64.wgsl` | 170 | Skyrme potentials (U_total, f_q) |
| `batched_hfb_hamiltonian_f64.wgsl` | 123 | HFB Hamiltonian H = T_eff + V |
| `batched_hfb_energy_f64.wgsl` | 147 | HFB energy functional (shared-memory reduce) |
| `bcs_bisection_f64.wgsl` | 141 | BCS chemical-potential bisection |
| `deformed_wavefunction_f64.wgsl` | 241 | Nilsson HO wavefunctions on 2D (ρ,z) grid |
| `deformed_hamiltonian_f64.wgsl` | 214 | Block Hamiltonian for deformed HFB |
| `deformed_density_energy_f64.wgsl` | 293 | Deformed density, energy, Q20, RMS radius |
| `deformed_gradient_f64.wgsl` | 205 | Gradient of deformed densities |
| `deformed_potentials_f64.wgsl` | 268 | Deformed mean-field potentials |

### MD Reference Shaders (`src/md/shaders_toadstool_ref/`, 4 files, ~334 lines)

| Shader | Lines | Pipeline Stage |
|--------|-------|----------------|
| `yukawa_f64.wgsl` | 97 | Yukawa all-pairs force with PBC |
| `yukawa_celllist_f64.wgsl` | 132 | Cell-list Yukawa force |
| `velocity_verlet_split.wgsl` | 73 | VV kick-drift-kick integrator |
| `vv_half_kick_f64.wgsl` | 32 | VV second half-kick |

### MD Production Shaders (`src/md/shaders/` + inline)

| Shader / Constant | Physics | Location |
|-------------------|---------|----------|
| `yukawa_force_f64.wgsl` | Yukawa all-pairs (native f64) | `.wgsl` file |
| `yukawa_force_celllist_f64.wgsl` | Cell-list v1 (27-neighbor) | `.wgsl` file |
| `yukawa_force_celllist_v2_f64.wgsl` | Cell-list v2 (flat loop) | `.wgsl` file |
| `vv_kick_drift_f64.wgsl` | Velocity-Verlet kick+drift | `.wgsl` file |
| `rdf_histogram_f64.wgsl` | RDF histogram binning | `.wgsl` file |
| `SHADER_VV_HALF_KICK` | VV second half-kick | inline (26 lines) |
| `SHADER_BERENDSEN` | Berendsen thermostat rescale | inline (22 lines) |
| `SHADER_KINETIC_ENERGY` | Kinetic energy reduction | inline (22 lines) |

## BarraCUDA Primitives Used

| BarraCUDA Module | hotSpring Usage |
|------------------|-----------------|
| `barracuda::linalg::eigh_f64` | Symmetric eigendecomposition (CPU) |
| `barracuda::ops::linalg::BatchedEighGpu` | Batched GPU eigensolve |
| `barracuda::ops::grid::SpinOrbitGpu` | GPU spin-orbit correction in HFB **(v0.5.6)** |
| `barracuda::ops::grid::compute_ls_factor` | Canonical l·s factor for spin-orbit **(v0.5.6)** |
| `barracuda::numerical::{trapz, gradient_1d}` | Radial integration, gradient |
| `barracuda::optimize::*` | Bisection, Brent, Nelder-Mead, multi-start NM |
| `barracuda::sample::*` | Latin hypercube, Sobol, direct |
| `barracuda::surrogate::*` | RBF surrogate, kernels |
| `barracuda::stats::*` | Chi², bootstrap CI, correlation |
| `barracuda::special::*` | Gamma, Laguerre, Bessel, Hermite, Legendre, erf |
| `barracuda::ops::md::*` | Forces, integrators, thermostats, observables |
| `barracuda::device::{WgpuDevice, TensorContext}` | GPU device bridge |

No duplicate math — all mathematical operations use BarraCUDA primitives.
WGSL `abs_f64` and `cbrt_f64` have inline copies pending preamble injection refactor.

## Promotion Priority

1. **GPU energy integrands + SumReduceF64** → Wire `batched_hfb_energy_f64.wgsl` into
   `hfb_gpu_resident.rs` SCF loop. Shader already exists; needs pipeline creation,
   Skyrme parameter upload, and `barracuda::ops::SumReduceF64` to reduce
   per-grid-point integrands to scalar total. Eliminates CPU `compute_energy_with_v2`
   and its `trapz` calls. **Estimated: ~100 lines of wiring code.**
   Note: `SumReduceF64::sum()` takes `&[f64]` (CPU-side data); full GPU-resident
   reduction requires keeping integrand buffer on GPU — this is the real target.
2. **BCS on GPU** → Move BCS occupations + density accumulation to GPU shader
   (`batched_hfb_density_f64.wgsl` exists, needs pipeline wiring)
3. ~~**SpinOrbitGpu**~~ ✅ **DONE (v0.5.6)** — Wired with CPU fallback
4. **WGSL preamble injection** → Replace inline `abs_f64`, `cbrt_f64` with
   `ShaderTemplate::math_f64_subset()` preamble from ToadStool canonical math
5. **hfb_deformed_gpu.rs** → Wire existing deformed_*.wgsl shaders for full GPU H-build
6. **nuclear_matter.rs** → Low priority; CPU bisection is fast enough

## Completed (v0.5.6)

- ✅ `SpinOrbitGpu` wired into `hfb_gpu_resident.rs` with CPU fallback
- ✅ `compute_ls_factor` from barracuda replaces manual `(j(j+1)-l(l+1)-0.75)/2` in `hfb.rs`, `hfb_gpu_resident.rs`
- ✅ Physics guard constants centralized: `DENSITY_FLOOR`, `SPIN_ORBIT_R_MIN`, `COULOMB_R_MIN`
  - 20+ inline `1e-15`, `0.1`, `1e-10` guards replaced across 5 physics modules
- ✅ SPDX headers added to all 17 WGSL shaders that were missing them (30/30 total)
- ✅ `panic!()` in library code converted to `expect()` (GPU buffer map failures)
- ✅ WGSL math duplicates annotated with `TODO(evolution)` for preamble injection

## Completed (v0.5.5)

- ✅ `data::load_eos_context()` → Shared EOS context loading for all nuclear EOS binaries
- ✅ `data::chi2_per_datum()` → Shared χ² computation with `tolerances::sigma_theo`
- ✅ `tolerances::BFGS_TOLERANCE` → Corrected from 0.1 to 1e-4 with proper justification
- ✅ `validate_optimizers` → Wired to use `tolerances::BFGS_TOLERANCE`
- ✅ All inline WGSL extracted from `celllist_diag.rs`
- ✅ 16 new unit tests (176 total)
- ✅ `verify_hfb` added to `validate_all` meta-validator

## Completed (v0.5.4)

- ✅ `hfb_gpu_resident.rs` → GPU eigensolve via `execute_single_dispatch` (was CPU `eigh_f64`)
- ✅ `validate_nuclear_eos` → Formal L1/L2/NMP validation with harness (37 checks)
- ✅ `validate_all` → Meta-validator for all 9 validation suites

## Completed (v0.5.3)

- ✅ `bcs_gpu.rs` → ToadStool `target` keyword fix absorbed (commit `0c477306`)
- ✅ `hfb_gpu.rs` → Single-dispatch eigensolve wired
- ✅ `hfb_deformed_gpu.rs` → Single-dispatch with fallback wired
- ✅ MD shaders → 5 large shaders extracted to `.wgsl` files
- ✅ BCS pipeline → Shader compilation cached at construction

## Evolution Gaps Identified

| Gap | Impact | Priority | Status |
|-----|--------|----------|--------|
| GPU energy integrands not wired in spherical HFB | CPU bottleneck in SCF energy | High | Shader exists, needs pipeline wiring |
| `SumReduceF64` not used for HFB energy sums | CPU readback for reduction | High | barracuda primitive available; needs GPU-buffer variant |
| BCS + density shader not wired | CPU readback after eigensolve | High | `batched_hfb_density_f64.wgsl` exists |
| WGSL inline math (`abs_f64`, `cbrt_f64`) | Maintenance drift from canonical | Medium | Annotated, pending preamble injection |
| 4 files > 1000 lines (HFB lib modules) | Code organization | Medium | Documented deviation; physics-coherent |
| `pow_f64(base, exp)` in ToadStool exists but not used | Available when needed | Low | WGSL-only |
| `FusedMapReduceF64` / `KrigingF64` unused | Available for MD post-processing | Low | |

## Gaps Resolved (v0.5.5)

- ✅ `celllist_diag.rs` inline WGSL → Extracted 8 shaders to `.wgsl` files (1672 → 1124 lines)
- ✅ Dead_code in deformed HFB → 6 field renames, 3 documented GPU-reserved functions
- ✅ Nuclear EOS path duplication → Shared `data::load_eos_context()` replaces 9 inline path constructions
- ✅ Inline tolerances → 30+ magic numbers replaced with `tolerances::` constants
- ✅ Inline `sigma_theo` → 19 instances replaced with `tolerances::sigma_theo()`
