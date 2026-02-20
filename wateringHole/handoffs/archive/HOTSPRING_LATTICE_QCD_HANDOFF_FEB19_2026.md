# hotSpring → ToadStool/BarraCUDA: Lattice QCD + Comprehensive Audit Handoff

**Date:** 2026-02-19
**From:** hotSpring (computational physics Spring)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only
**Builds on:** All previous handoffs (11 total in `wateringHole/handoffs/`)

---

## Executive Summary

hotSpring has completed a major extension: **lattice gauge theory infrastructure**
validated on CPU, with WGSL template strings ready for GPU promotion. Pure gauge
SU(3) simulations run on 4^4 lattices with working HMC, 96-100% acceptance rates,
and validated Dirac CG solver. This proves the Python → Rust → GPU evolution path
extends beyond plasma physics to quantum field theory.

A comprehensive audit was also performed: zero unsafe code, zero clippy warnings,
zero doc warnings, 254 unit tests passing, 14/15 validation suites passing.

### What Was Delivered

| Deliverable | Status | Files |
|-------------|--------|-------|
| Complex f64 arithmetic (Rust + WGSL) | **Done** | `lattice/complex_f64.rs` (316 lines) |
| SU(3) matrix algebra (Rust + WGSL) | **Done** | `lattice/su3.rs` (460 lines) |
| Wilson gauge action | **Done** | `lattice/wilson.rs` (338 lines, 100% coverage) |
| HMC integrator (Cayley exponential) | **Done** | `lattice/hmc.rs` (350 lines) |
| Staggered Dirac operator | **Done** | `lattice/dirac.rs` (297 lines) |
| Conjugate gradient solver | **Done** | `lattice/cg.rs` (214 lines) |
| HotQCD EOS tables | **Done** | `lattice/eos_tables.rs` (307 lines) |
| Multi-GPU temperature scan | **Done** | `lattice/multi_gpu.rs` (237 lines) |
| Transport coefficient fits | **Done** | `md/transport.rs` (291 lines) |
| Green-Kubo observables | **Partial** | `md/observables.rs` — normalization bug |
| Validation binary: pure gauge | **Done** | `bin/validate_pure_gauge.rs` (12/12) |
| Validation binary: HotQCD EOS | **Done** | `bin/validate_hotqcd_eos.rs` |
| Validation binary: transport | **Partial** | `bin/validate_stanton_murillo.rs` (D* fails) |
| Comprehensive code audit | **Done** | Zero unsafe, zero clippy, zero doc warnings |

---

## What ToadStool/BarraCUDA Should Absorb

### 1. Complex f64 Primitive

`lattice/complex_f64.rs` defines `Complex64` with full arithmetic, conjugate,
inverse, exponential, polar form. The `WGSL_COMPLEX64` string constant contains
equivalent WGSL functions ready for `ShaderTemplate` injection.

**Recommendation**: Absorb `Complex64` into `barracuda::numerical` or
`barracuda::linalg` as a reusable primitive. Many scientific workloads need
complex f64 (FFT, quantum mechanics, signal processing).

### 2. SU(3) Matrix Algebra

`lattice/su3.rs` defines `Su3Matrix` (3×3 complex) with multiply, adjoint,
trace, determinant, Gram-Schmidt reunitarization, random near-identity, and
Lie algebra momentum generation. `WGSL_SU3` contains WGSL equivalents.

**Recommendation**: This is domain-specific (lattice QCD) but the 3×3 complex
matrix multiply is general. Consider adding complex GEMM to `barracuda::linalg`.

### 3. Cayley Matrix Exponential

The HMC integrator uses the Cayley transform `(I + X/2)(I - X/2)^{-1}` for
the SU(3) matrix exponential. This is **exactly unitary** when X is
anti-Hermitian, unlike Taylor approximations which drift. The 3×3 inverse
uses cofactor expansion (exact, O(1)).

**Key lesson**: Second-order Taylor `I + X + X²/2` caused 0% HMC acceptance
on 4^4 lattices. The Cayley transform fixed it immediately. This is a known
result in numerical lattice QCD but wasn't obvious from first principles.

### 4. SumReduceF64 Buffer-to-Buffer API

hotSpring's MD loops use a local adaptation of `sum_reduce_f64.wgsl` because
barracuda's `SumReduceF64::sum()` takes `&[f64]` (CPU-side data). The MD
pipeline needs GPU-buffer → scalar reduction without readback.

**Request**: Add `SumReduceF64::sum_from_buffer(device, buffer, len) -> f64`
or equivalent that reduces a GPU buffer to a scalar. This would eliminate the
last duplicated shader in hotSpring.

### 5. Transport Coefficient Bug

The Stanton-Murillo validation fails because `compute_stress_acf` and
`compute_heat_acf` in `md/observables.rs` produce values orders of magnitude
off from the analytical fits. The diffusion coefficient D* is ~30× too large,
viscosity η* is ~36M× too large, thermal conductivity λ* is ~10M× too large.

**Root cause**: Almost certainly a dimensional analysis error in the Green-Kubo
integrals — missing volume factor, temperature normalization, or incorrect
reduced-unit conversion. The VACF → D* path should be:
`D = (1/3) ∫₀^∞ <v(0)·v(t)> dt` in reduced units.

This is documented as a known bug. The analytical fits themselves (Daligault
2012, Stanton & Murillo 2016) are validated against literature.

---

## Audit Findings Relevant to BarraCUDA Evolution

### Code Quality

| Metric | Value |
|--------|-------|
| Unit tests | 254 pass, 5 ignored (GPU-required) |
| Validation suites | 14/15 pass |
| Clippy (default) | 0 warnings |
| Clippy (pedantic) | 465 warnings (cast_precision_loss, similar_names — physics code) |
| Doc warnings | 0 |
| Unsafe blocks | 0 |
| TODO/FIXME/HACK | 0 |
| AGPL-3.0 compliance | All .rs and .wgsl files |
| Test coverage (llvm-cov) | 46% line, 65% function |

### Files Over 1000 Lines

9 files exceed the wateringHole 1000-line guideline. All are physics-coherent
modules where splitting would create artificial boundaries:

- `physics/hfb_gpu_resident.rs` (1,928) — GPU pipeline with 14 buffers
- `physics/hfb.rs` (1,405) — core HFB solver
- `physics/hfb_deformed.rs` (1,290) — deformed HFB
- `physics/hfb_deformed_gpu.rs` (1,227) — GPU deformed HFB
- `bin/nuclear_eos_l1_ref.rs` (1,250) — L1 reference
- `bin/nuclear_eos_l2_hetero.rs` (1,207) — heterogeneous L2
- `bin/celllist_diag.rs` (1,158) — cell-list diagnostic
- `md/simulation.rs` (1,116) — MD engine
- `bin/nuclear_eos_gpu.rs` (1,039) — GPU nuclear EOS

### Barracuda Primitive Usage (Comprehensive)

hotSpring correctly delegates to barracuda for: `eigh_f64`, `BatchedEighGpu`,
`trapz`, `gradient_1d`, `gamma`, `laguerre`, `hermite`, `factorial`, `brent`,
`bisect`, `nelder_mead`, `bfgs`, `chi2_decomposed_weighted`, `bootstrap_ci`,
`latin_hypercube`, `sobol_sequence`, `RBFSurrogate`, YukawaForceF64, VV
integrators, Berendsen thermostat, `WgpuDevice`, `TensorContext`.

Only two justified duplications remain:
1. `SHADER_SUM_REDUCE` — needs buffer-to-buffer API (see above)
2. Inlined WGSL math in `deformed_wavefunction_f64.wgsl` — WGSL has no `#include`

---

## GPU Promotion Roadmap for Lattice QCD

### Phase 1: SU(3) on GPU (Tier B — templates exist)

1. Compile `WGSL_COMPLEX64` and `WGSL_SU3` via `ShaderTemplate`
2. Create plaquette computation shader (input: link buffer, output: average plaquette)
3. Validate GPU plaquette vs CPU plaquette on cold and hot lattices

### Phase 2: HMC on GPU (Tier C — new shaders)

1. Port Cayley exponential to WGSL (3×3 inverse via cofactor)
2. Port leapfrog link update (exp(dtP) × U per link)
3. Keep Metropolis accept/reject on CPU (single random number)
4. Validate GPU HMC acceptance rate matches CPU

### Phase 3: Dirac CG on GPU (Tier C — dominant cost)

1. Port staggered Dirac operator to WGSL (sparse matrix-vector product)
2. Port CG iteration to GPU (dot products via SumReduceF64)
3. Memory layout: SoA for coalesced access across lattice sites

### Phase 4: Multi-GPU Temperature Scan

1. Each GPU runs independent HMC at different β (temperature)
2. Embarrassingly parallel — no inter-GPU communication
3. Distribute across basement HPC mesh (9 GPUs, 10G backbone)

### Data Layout for GPU

Each SU(3) link = 3×3 complex = 18 f64 = 144 bytes.
4^4 lattice: 256 sites × 4 directions × 144 bytes = **147 KB** (fits in L1).
8^4 lattice: 4096 × 4 × 144 = **2.4 MB** (fits in L2).
16^4 lattice: 65536 × 4 × 144 = **37.7 MB** (fits in VRAM easily).
32^4 lattice: 1M × 4 × 144 = **603 MB** (fits in 12GB VRAM).

---

## Lessons Learned

1. **Sign conventions in lattice QCD are treacherous.** The gauge force
   `dP/dt = -(β/3) Proj_TA(UV)` has the staple V (not V†), and the
   traceless anti-Hermitian projection uses `(M - M†)/2`, not `(M + M†)/2`.
   Getting any sign wrong causes 0% HMC acceptance with no obvious error.

2. **Cayley > Taylor for matrix exponentials.** The Cayley transform is
   exactly unitary and costs one 3×3 inverse (cofactor, exact). Taylor
   requires reunitarization which introduces uncontrolled error.

3. **Lattice QCD and plasma MD share more structure than expected.**
   The HMC integrator is a leapfrog (same as Velocity Verlet), the
   gauge force is analogous to the Yukawa force derivative, and the
   Metropolis step is standard. The main difference is the configuration
   space: SU(3) manifold vs Euclidean positions.

4. **WGSL template strings in Rust source are effective.** Having
   `WGSL_COMPLEX64` and `WGSL_SU3` as `&str` constants in the Rust
   modules means the GPU code evolves alongside the CPU code. No
   separate shader files to keep in sync.

5. **Green-Kubo transport is fragile.** Dimensional analysis must be
   perfect — any missing factor shows up as orders-of-magnitude error,
   not a subtle drift. The VACF path works; stress and heat current
   do not (yet).

---

## Files Changed Since Last Handoff

```
barracuda/Cargo.toml                          # 3 new [[bin]] entries
barracuda/src/lib.rs                          # pub mod lattice
barracuda/src/lattice/mod.rs                  # NEW — 8 submodules
barracuda/src/lattice/complex_f64.rs          # NEW — 316 lines
barracuda/src/lattice/su3.rs                  # NEW — 460 lines
barracuda/src/lattice/wilson.rs               # NEW — 338 lines
barracuda/src/lattice/hmc.rs                  # NEW — 350 lines
barracuda/src/lattice/dirac.rs                # NEW — 297 lines
barracuda/src/lattice/cg.rs                   # NEW — 214 lines
barracuda/src/lattice/eos_tables.rs           # NEW — 307 lines
barracuda/src/lattice/multi_gpu.rs            # NEW — 237 lines
barracuda/src/md/transport.rs                 # MODIFIED — η*, λ* fits added
barracuda/src/md/observables.rs               # MODIFIED — stress ACF, heat ACF
barracuda/src/bin/validate_stanton_murillo.rs # NEW — Paper 5 validation
barracuda/src/bin/validate_hotqcd_eos.rs      # NEW — Paper 7 validation
barracuda/src/bin/validate_pure_gauge.rs      # NEW — Paper 8 validation (12/12)
barracuda/src/bin/validate_all.rs             # MODIFIED — 15 suites
```

Total new code: ~2,519 lines of library code + ~516 lines of validation binaries.
