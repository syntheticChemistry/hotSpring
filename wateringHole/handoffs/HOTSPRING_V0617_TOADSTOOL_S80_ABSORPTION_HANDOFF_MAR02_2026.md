SPDX-License-Identifier: AGPL-3.0-only

# hotSpring → toadStool: S80 Sync Complete + Absorption Targets + Learnings

**Date:** 2026-03-02
**From:** hotSpring v0.6.17
**To:** toadStool/barracuda team
**Covers:** S78 → S80 sync, cross-spring evolution audit, full experiment arc (001–031)
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring is synced to toadStool S80. 660 tests pass, 0 clippy warnings.
The S80 sync wired three new upstream primitive families into hotSpring's
production pipeline (spectral stats, NeighborMode 4D, Nelder-Mead GPU)
and validated serde compatibility for the upcoming `MultiHeadEsn` migration.

This handoff identifies **what hotSpring has learned** that should flow back
to toadStool, and **what toadStool should absorb next** based on 31 experiments,
85 binaries, and 660 tests of cross-substrate physics validation.

---

## Part 1: What hotSpring Wired from S78→S80

| Upstream Primitive | Where Used | Result |
|--------------------|-----------|--------|
| `spectral_bandwidth(&eigenvalues)` | `proxy.rs` Anderson 3D | Replaced manual `last - first` calc |
| `spectral_condition_number(&eigenvalues)` | `proxy.rs` Anderson 3D | New field in `ProxyFeatures` |
| `SpectralAnalysis::from_eigenvalues()` | `bench_cross_spring_evolution` | Marchenko-Pastur phase classification validated |
| `classify_spectral_phase` | `bench_cross_spring_evolution` | Linked, exercised |
| `NeighborMode::precompute_periodic_4d` | `bench_cross_spring_evolution` | 3.6× faster than hotSpring at 16^4 |
| `batched_nelder_mead_gpu` | `bench_cross_spring_evolution` | 1000 problems in 205ms (GPU batch optimizer) |
| `MultiHeadEsn` / `ExportedWeights` | `md/reservoir/tests.rs` | Serde round-trip validated, HeadGroup aligned |

---

## Part 2: What toadStool Should Absorb from hotSpring

### Tier 1 — Ready now (validated, WGSL exists)

| Module | Lines | Checks | Notes |
|--------|-------|--------|-------|
| `lattice/pseudofermion/` | 477+ | 7/7 | Pseudofermion HMC: heat bath, CG action, fermion force, combined leapfrog. CPU with WGSL-ready patterns |
| `lattice/dirac.rs` + WGSL | ~400 | 8/8 | Staggered Dirac SpMV. `WGSL_DIRAC_STAGGERED_F64` ready |
| `lattice/cg.rs` + 3 WGSL | ~350 | 9/9 | CG solver: `WGSL_COMPLEX_DOT_RE_F64`, axpy, xpay |
| `md/reservoir/` (ESN) | ~1200 | 35/35 | 11-head ESN with physics heads. `ExportedWeights` already compatible with `MultiHeadEsn`. Migration to GPU `MultiHeadEsn` is the highest-impact absorption |
| `production/npu_worker.rs` | ~1000 | — | NPU parameter controller: dt/n_md control, safety clamps, acceptance targeting. Pattern for neuromorphic steering |

### Tier 2 — Ready but needs WGSL extraction

| Module | Lines | Checks | Notes |
|--------|-------|--------|-------|
| `physics/hfb_deformed_gpu/` | ~1500 | 16/16 | Deformed HFB with 5 WGSL shaders (potentials, density, BCS, coulomb, kinetic) |
| `lattice/abelian_higgs.rs` | ~400 | 17/17 | U(1)+Higgs HMC on (1+1)D. 3 WGSL shaders. 143× Python→Rust speedup |
| `lattice/gpu_hmc/resident_cg*.rs` | ~800 | — | GPU-resident CG: 15,360× readback reduction, 30.7× speedup. 4 submodules |

### Tier 3 — Local evolution (stays in hotSpring)

| Module | Why |
|--------|-----|
| `production/` (orchestration) | Spring-specific pipeline wiring |
| `tolerances/` | Spring-specific physics thresholds |
| `provenance.rs` | Spring-specific baseline records |
| `validation.rs` | Spring-specific harness |

---

## Part 3: Learnings for toadStool Evolution

### 3a. NeighborMode Site Indexing Mismatch

hotSpring uses z-fastest site ordering: `idx = t*V3 + x*V2 + y*Nz + z`
toadStool uses x-fastest site ordering: `idx = t*V3 + z*V2 + y*Nx + x`

Both conventions are internally consistent. The issue only matters when
sharing neighbor tables between springs. **Recommendation**: document the
convention in `NeighborMode` and consider a builder that accepts ordering.

### 3b. MultiHeadEsn → MultiHeadNpu Migration Path

hotSpring validated that `ExportedWeights` serializes/deserializes identically
between hotSpring's CPU ESN and toadStool's `MultiHeadEsn`. The `HeadGroup`
enum values (A=0, B=1, C=2, D=3) align with hotSpring's `heads::GROUP_A/B/C/D`
base indices.

**What toadStool should provide for the migration**:
1. A `MultiHeadEsn::from_exported_weights()` constructor (currently only `new()`)
2. Per-head training API (hotSpring trains heads independently based on physics)
3. Async inference (hotSpring's NPU worker runs on a separate thread)

### 3c. Anderson Proxy → toadStool Spectral Primitives

The Anderson 3D proxy (`proxy.rs`) generates `ProxyFeatures` for NPU steering:
- `level_spacing_ratio` (GOE vs Poisson → extended vs localized)
- `spectral_bandwidth` (eigenvalue spread)
- `spectral_condition_number` (stiffness indicator)
- `participation_ratio` (localization fraction)
- `SpectralAnalysis` phase (Marchenko-Pastur classifier)

All five are now upstream in toadStool. hotSpring's proxy.rs leans on them.
wetSpring and neuralSpring can use the same primitives for their own spectral
classification tasks (bio-molecular eigenmodes, RMT classifier).

### 3d. NPU Parameter Control — A Pattern for toadStool

Exp 029–031 proved that neuromorphic hardware can steer physics simulations
in real time. The pattern:

```
Physics step (GPU) → Feature extraction (CPU) → ESN prediction (NPU) → Parameter suggestion → Safety clamp → Apply
```

Key learnings:
- NPU input alignment is fragile: the NPU expects exactly N features in the
  order it was trained. Misalignment silently corrupts predictions.
- Safety clamps are essential: dt must be bounded [0.001, 0.5], n_md [2, 50].
- The ESN must be trained on acceptance rate as the target (not plaquette).
- Cross-run bootstrap (loading weights from previous experiments) works but
  requires careful feature normalization.

### 3e. Cross-Spring Shader Evolution (S80 census)

Shaders born in hotSpring that toadStool absorbed and redistributed:

| Shader | Born | Absorbed | Used By |
|--------|------|----------|---------|
| `complex_f64.wgsl` | hotSpring Feb 12 | toadStool S20 | all springs |
| `su3_math_f64.wgsl` | hotSpring Feb 13 | toadStool S21 | hotSpring only (QCD) |
| `df64_core.wgsl` | hotSpring Feb 18 | toadStool S42 | all springs |
| Spectral (Lanczos, SpMV) | hotSpring Feb 23 | toadStool S31 | hotSpring, wetSpring, neuralSpring |
| `level_spacing_ratio` | hotSpring Feb 23 | toadStool S78 | hotSpring, wetSpring, neuralSpring |
| VACF GPU | hotSpring Feb 25 | toadStool S70+ | hotSpring, wetSpring |
| `esn_reservoir_update.wgsl` | hotSpring Feb 20 | toadStool S42 | hotSpring, neuralSpring |

Shaders born in other springs that hotSpring uses:

| Shader | Born | Via | Used For |
|--------|------|-----|----------|
| `matrix_correlation` | neuralSpring | toadStool S25 | Plaquette correlation analysis |
| `batched_nelder_mead_gpu` | neuralSpring | toadStool S79 | HMC parameter tuning |
| `compute_stress_virial` | hotSpring | toadStool S70+ | Transport (Green-Kubo η*) |
| Bio spectral analysis | wetSpring | toadStool S78 | Spectral stats primitives |

---

## Part 4: Test Validation Summary

| Category | Tests | Status |
|----------|-------|--------|
| Library tests | 660 | Pass |
| Integration tests | 53 | Pass |
| Clippy (pedantic + nursery) | — | 0 warnings |
| `bench_cross_spring_evolution` | — | S80 primitives exercised |
| `ExportedWeights` serde round-trip | 1 | Pass |
| `HeadGroup` alignment | 1 | Pass |

---

## Part 5: Recommended toadStool Evolution Priorities

1. **MultiHeadEsn `from_exported_weights()` constructor** — unblocks hotSpring's
   CPU→GPU ESN migration. The 36-head ESN becomes a GPU citizen.
2. **NeighborMode ordering documentation** — prevent silent bugs when springs
   share neighbor tables.
3. **Pseudofermion HMC absorption** — 477+ lines of validated lattice QCD physics
   ready for GPU promotion. This is the last major hotSpring→toadStool absorption.
4. **Resident CG pattern** — GPU-resident iterative solvers with minimal readback
   are a general pattern useful beyond QCD.
5. **NPU steering integration** — the `ProxyFeatures` → `NpuRequest` → parameter
   suggestion pattern could become a toadStool primitive for any adaptive simulation.

---

## Action Items for toadStool Team

- [ ] Review `MultiHeadEsn` constructor API for `ExportedWeights` import
- [ ] Document `NeighborMode` site-indexing convention
- [ ] Schedule pseudofermion HMC shader extraction from hotSpring `lattice/pseudofermion/`
- [ ] Consider `ProxyFeatures` as a generic adaptive-simulation primitive
- [ ] Review resident CG pattern for generalization

---

*hotSpring v0.6.17 — 660 tests, 85 binaries, 62 WGSL shaders, synced to toadStool S80.
31 experiments from N-scaling to neuromorphic parameter control. The scarcity was artificial.*
