# hotSpring v0.6.4 → ToadStool/BarraCUDA: Comprehensive Evolution Handoff

**Date:** February 22, 2026
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCUDA core team (barracuda v0.2.0)
**License:** AGPL-3.0-only
**Context:** hotSpring v0.6.4 (spectral lean), ToadStool Sessions 25-31h complete

---

## Executive Summary

hotSpring has completed four ToadStool rewire cycles. The spectral module
(Anderson localization, Lanczos, Hofstadter, Sturm, CSR SpMV, level statistics)
is now fully leaning on upstream `barracuda::spectral` — 41 KB of local code
deleted, re-exports and a backward-compatible `CsrMatrix` alias retained. All
33 validation suites pass. 22 papers reproduced. 637 unit + 24 integration tests.

This handoff identifies:
1. What hotSpring has ready for ToadStool to absorb next
2. What ToadStool has that hotSpring should use more
3. Cross-spring shader evolution insights
4. Lessons learned from the absorption cycle

---

## Part 1: What hotSpring Has Ready for Absorption

### Tier 1 — High-Value GPU Primitives

These have WGSL shaders, validation suites, and binding documentation.

#### 1.1 Staggered Dirac Operator

| Property | Value |
|----------|-------|
| Source | `barracuda/src/lattice/dirac.rs` |
| Shader | `WGSL_DIRAC_STAGGERED_F64` (inline, ~300 lines) |
| Tests | 8/8 checks (cold, hot, asymmetric lattices) |
| Precision | Max error 4.44e-16 (machine epsilon) |
| Bindings | `[positions (SU3 links), source, result, uniforms (Lx,Ly,Lz,Lt)]` |
| Workgroup | `(64, 1, 1)`, grid = `ceil(volume/64)` |
| Suggested upstream location | `barracuda::ops::lattice::StaggeredDiracGpu` |

The shader applies the staggered Dirac operator D[n,m] to a complex vector.
It handles periodic boundary conditions, staggered phases `η_μ(n)`, and SU(3)
color transport. Validated on 4⁴, 6⁴, and 8³×4 lattices.

#### 1.2 Conjugate Gradient Solver (3 shaders)

| Shader | Purpose | Bindings |
|--------|---------|----------|
| `WGSL_COMPLEX_DOT_RE_F64` | Re(∑ a̅ᵢbᵢ) reduction | `[a, b, partial_sums, uniforms(n)]` |
| `WGSL_AXPY_F64` | y ← αx + y | `[x, y, uniforms(alpha_re, alpha_im, n)]` |
| `WGSL_XPAY_F64` | y ← x + αy | `[x, y, uniforms(alpha_re, alpha_im, n)]` |

- **Tests**: 9/9 checks, CG iterations match CPU exactly
- **Transfer**: Only 24 bytes/iteration (3 scalar coefficients) cross CPU↔GPU
- **Benchmark**: Rust 200× faster than Python (identical iterations, identical seeds)
- **Suggested upstream**: `barracuda::ops::linalg::CgSolverGpu`

Together with the Dirac shader, these form the **complete GPU lattice QCD pipeline**.
`validate_pure_gpu_qcd` proves it works on thermalized HMC configurations (3/3 checks,
solution parity 4.10e-16).

#### 1.3 ESN Reservoir (2 shaders)

| Shader | Purpose |
|--------|---------|
| `esn_reservoir_update.wgsl` | Reservoir state update (tanh activation, f32) |
| `esn_readout.wgsl` | Linear readout layer (f32) |

- **Tests**: 16+ checks across CPU, NpuSimulator, quantized, and AKD1000 hardware
- **Pipeline**: MD trajectory → ESN → transport predictions (D*, η*, λ*)
- **Cross-substrate**: Same weights work on GPU (f32), NPU (int4), CPU (f64)
- **Note**: hotSpring has `md/reservoir.rs` locally; ToadStool has `esn_v2::ESN`.
  The upstream ESN could absorb these shaders and expose `export_weights()`
  for GPU-train → NPU-deploy. Currently hotSpring does this manually.

### Tier 2 — Physics Modules (CPU → upstream library)

These are CPU implementations that enrich barracuda's physics coverage:

| Module | Tests | What it does | Absorption value |
|--------|-------|--------------|------------------|
| `physics/screened_coulomb.rs` | 23/23 | Murillo-Weisheit Sturm eigensolve | Screened Coulomb eigenvalues for plasmas |
| `md/transport.rs` | 13/13 | Stanton-Murillo analytical fits (D*, η*, λ*) | Transport coefficient models |
| `md/observables/transport.rs` | 5 | Green-Kubo stress/heat current ACFs | First-principles transport |
| `lattice/eos_tables.rs` | — | HotQCD EOS reference data (Bazavov 2014) | Lattice QCD thermodynamics |

### Tier 3 — Nuclear Physics Shaders (domain-specific but reusable patterns)

| Module | Shaders | Notes |
|--------|---------|-------|
| `physics/bcs_gpu.rs` | `bcs_bisection_f64.wgsl` | GPU root-finding via bisection. Pattern reusable. |
| `physics/hfb_gpu_resident/` | 4 production shaders | Full GPU-resident SCF pipeline. Pattern: iterate on GPU, only converged scalars cross. |
| `physics/hfb_deformed_gpu/` | 5 shaders (partially wired) | Deformed nuclear structure. H-build pattern generalizable. |

### Tier 4 — metalForge Substrate Discovery

| Component | Source | Absorption target |
|-----------|--------|-------------------|
| NPU probe | `forge/src/probe.rs::probe_npus()` | `/dev/akida*` + PCIe sysfs + SRAM |
| CPU probe | `forge/src/probe.rs::probe_cpu()` | `/proc/cpuinfo` + `/proc/meminfo` |
| GPU probe | `forge/src/probe.rs::probe_gpus()` | wgpu adapters + VRAM from `adapter.limits()` |
| Capability model | `forge/src/substrate.rs` | 12-variant enum (F64Compute, QuantizedInference, ...) |
| Dispatch | `forge/src/dispatch.rs` | Capability-based workload routing |
| Bridge | `forge/src/bridge.rs` | forge substrate ↔ barracuda `WgpuDevice` |

The bridge module is the explicit absorption seam. See
`archive/HOTSPRING_V061_FORGE_HANDOFF_FEB21_2026.md` for full forge documentation.

---

## Part 2: What ToadStool Has That hotSpring Should Use More

These upstream primitives exist but hotSpring hasn't adopted yet. Some would
provide significant improvements.

### High-Impact Candidates

| Primitive | Module | hotSpring Benefit |
|-----------|--------|-------------------|
| **`GemmCachedF64`** | `ops::linalg` | **60× speedup for repeated GEMM** (wetSpring origin). hotSpring rebuilds Hamiltonian matrices in HFB SCF loops — caching the workspace would eliminate per-iteration allocation. |
| **`NelderMeadGpu`** | `optimize` | GPU-accelerated Nelder-Mead. hotSpring uses `multi_start_nelder_mead()` for L1 parameter search (6,028 evaluations). GPU dispatch could parallelize simplex evaluations. |
| **`GenEighGpu`** | `ops::linalg` | Generalized eigenvalue decomposition (Ax = λBx). Relevant for deformed HFB where overlap matrices are non-trivial. |
| **`BatchIprGpu`** | `spectral` | Already re-exported but not yet used. GPU inverse participation ratio for Anderson localization diagnostics. |
| **`UnidirectionalPipeline`** | `staging` | Ring-buffer staging for streaming GPU dispatch. Could replace manual encoder batching in MD loops. |
| **`WorkloadClassifier`** | `workload` | Intelligent workload routing. Synergy with metalForge's capability-based dispatch. |

### Medium-Impact Candidates

| Primitive | Module | hotSpring Benefit |
|-----------|--------|-------------------|
| `ESN` (v2) | `esn_v2` | Upstream ESN could replace local `md/reservoir.rs`. Need `export_weights()` for NPU deploy. |
| `BatchedOdeRK4F64` | `numerical` | Batched RK4 ODE solver. Could parallelize nuclear matter ODE evaluations. |
| New MD forces | `ops::md` | `LennardJonesForce`, `MorseForce`, `BornMayerForceF64` extend beyond Yukawa. |
| `QrGpu`, `SvdGpu` | `ops::linalg` | GPU decompositions. HFB needs eigensolve (already using `BatchedEighGpu`), but QR/SVD could help with basis orthogonalization. |
| Spline interpolation | `interpolate` | Cubic spline. Could improve EOS table interpolation vs current linear. |

### Already Using (Confirmation)

| Primitive | Usage in hotSpring |
|-----------|-------------------|
| `WgpuDevice`, `TensorContext` | All GPU pipelines |
| `BatchedEighGpu` | HFB eigensolve (spherical, deformed, GPU-resident) |
| `ReduceScalarPipeline` | MD energy reduction (KE, PE, thermostat) |
| `ShaderTemplate`, `GpuDriverProfile` | All shader compilation |
| `CellListGpu` | MD cell-list neighbor search |
| `SsfGpu`, `PppmGpu` | Observables, electrostatics |
| `spectral::*` | Full re-export (Anderson, Lanczos, Hofstadter, etc.) |
| `linalg::eigh_f64` | CPU eigensolve |
| `special::*` | Gamma, Laguerre, Bessel, Hermite, Legendre, erf |
| `optimize::*` | Bisect, Brent, NM, multi-start NM |
| `sample::*` | LHS, direct, Sobol, sparsity |
| `stats::*` | Chi², bootstrap CI |
| `numerical::*` | trapz, gradient_1d, RK45 |
| `surrogate::*` | RBF surrogate |

---

## Part 3: Cross-Spring Shader Evolution

The biome model works. Each Spring contributes domain-specific insights that
ToadStool absorbs, making them available to all Springs without inter-Spring
imports. The full map is in `CROSS_SPRING_EVOLUTION_FEB22_2026.md`. Highlights:

### WGSL Precision (wetSpring → all)

wetSpring's bio-math shaders discovered that WGSL f64 literal constants can
lose precision during compilation. The fix: `let zero: f64 = f64(0); let c = zero + 2.302585092994046;`
forces full f64 precision. This fixed `log_f64` from ~1e-3 to ~1e-15 accuracy.

**Impact on hotSpring**: Every f64 WGSL shader benefits. The fix is baked into
`ShaderTemplate::with_math_f64_auto()` which hotSpring uses for all shader compilation.

### NVK Driver Workarounds (hotSpring → all)

hotSpring profiled NVK/nouveau on Titan V and found `exp()` and `log()` produce
wrong results. The workaround (software-emulated transcendentals on NVK) is
in `ShaderTemplate::for_driver_profile()`. All Springs benefit transparently.

**Performance data**: NVK is 4-7× slower on compute-bound shaders (driver maturity,
not hardware). Warp-packed eigensolves on NAK: 5× faster than scalar, but 7× slower
than nvidia proprietary. These findings are in `device/driver_profile.rs`.

### Spectral Module (hotSpring → all)

The spectral theory stack (Anderson localization, Lanczos eigensolve, CSR SpMV,
Hofstadter butterfly) was evolved in hotSpring for Kachkovskiy's condensed matter
physics and handed off to ToadStool. Any Spring needing sparse eigensolve or
localization diagnostics now has it from `barracuda::spectral`.

### Bio Primitives (wetSpring → all)

ToadStool absorbed extensive bio primitives from wetSpring: Smith-Waterman
alignment, Gillespie SSA, phylogenetic inference, HMM, DADA2 E-step, SNP
calling, pangenome classification. hotSpring doesn't use these directly but
the pattern demonstrates absorption at scale (17+ bio ops).

### Neural Primitives (neuralSpring → all)

`TensorSession` with matmul, relu, softmax, attention was absorbed from
neuralSpring. `BatchIprGpu` (inverse participation ratio) came via
metalForge → toadStool path and is now available to hotSpring for GPU
Anderson localization diagnostics.

---

## Part 4: Lessons Learned

### What Makes Absorption Work Well

1. **WGSL shaders in dedicated `.wgsl` files** — ToadStool can copy directly
2. **Clear binding documentation** — binding index, type, purpose table
3. **CPU reference implementation** — gives ToadStool a test oracle
4. **Validation binary** — shows exactly how to call the primitive
5. **Type alias for renamed structs** — `CsrMatrix = SpectralCsrMatrix` provides
   backward compatibility without API churn in downstream Springs

### What Slowed Absorption Down

1. **Inline WGSL strings** — shaders embedded in `.rs` source required extraction
   before absorption. v0.6.3 eliminated all inline WGSL from the library.
2. **Binding layout mismatches** — CellListGpu BGL mismatch between shader and
   Rust code required debugging on both sides. Document bindings in a table.
3. **Renamed types** — `CsrMatrix` → `SpectralCsrMatrix` required downstream alias.
   Consider keeping names stable or providing a migration guide.

### Metrics from the Spectral Lean

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Local spectral source | 41 KB (6 files) | 0 KB (re-exports only) | −41 KB |
| Local tests | 648 | 637 | −11 (moved upstream) |
| Validation suites | 33/33 | 33/33 | No regression |
| Compilation time | — | Faster (less local code) | Improved |

---

## Part 5: Concrete Next Steps

### For ToadStool to Absorb

| Priority | What | Source | Impact |
|----------|------|--------|--------|
| **P1** | Staggered Dirac GPU shader | `lattice/dirac.rs` | Completes upstream lattice QCD pipeline |
| **P1** | CG solver (3 shaders) | `lattice/cg.rs` | D†D inversion on GPU — core lattice QCD |
| **P1** | ESN `export_weights()` | `esn_v2::ESN` | Enables GPU-train → NPU-deploy workflow |
| **P2** | Screened Coulomb | `physics/screened_coulomb.rs` | Plasma physics eigenvalue solver |
| **P2** | HFB shader suite | `physics/hfb_gpu_resident/` | 4 production shaders, GPU-resident SCF pattern |
| **P3** | forge substrate discovery | `metalForge/forge/src/` | NPU + CPU discovery for heterogeneous dispatch |

### For hotSpring to Adopt

| Priority | What | Upstream | Benefit |
|----------|------|----------|---------|
| **P1** | `GemmCachedF64` | `ops::linalg` | HFB SCF loop: eliminate per-iteration GEMM allocation |
| **P1** | `BatchIprGpu` | `spectral` | GPU Anderson localization diagnostics (already re-exported) |
| **P2** | `NelderMeadGpu` | `optimize` | GPU-parallel L1 parameter search |
| **P2** | `UnidirectionalPipeline` | `staging` | Streaming MD dispatch (replace manual encoder batching) |
| **P3** | Upstream `ESN` | `esn_v2` | Replace local `md/reservoir.rs` (pending `export_weights()`) |

### Cross-Spring Observations for ToadStool

- **Shader precision fixes should be documented in a central place** — the
  `(zero + literal)` f64 pattern is essential for all Springs but currently
  only noted in wetSpring's handoff and `ShaderTemplate` internals.
- **Driver profile data grows with each Spring** — hotSpring contributed NVK
  performance data, wetSpring contributed precision data. A formal
  `GpuDriverProfile` calibration database would benefit all.
- **The absorption cycle takes ~3 sessions** — hotSpring's experience: write (1),
  hand off + absorb (1), lean + validate (1). Plan accordingly.

---

## Appendix: File Inventory for Absorption

### WGSL Shaders Ready for Upstream

```
src/lattice/dirac.rs          → WGSL_DIRAC_STAGGERED_F64 (~300 lines)
src/lattice/cg.rs             → WGSL_COMPLEX_DOT_RE_F64 (~60 lines)
                              → WGSL_AXPY_F64 (~40 lines)
                              → WGSL_XPAY_F64 (~40 lines)
src/md/shaders/esn_reservoir_update.wgsl (~50 lines)
src/md/shaders/esn_readout.wgsl (~30 lines)
```

### Validation Binaries (test oracles)

```
src/bin/validate_gpu_dirac.rs          # 8/8 checks
src/bin/validate_gpu_cg.rs             # 9/9 checks
src/bin/validate_pure_gpu_qcd.rs       # 3/3 checks
src/bin/validate_reservoir_transport.rs # ESN pipeline
src/bin/validate_screened_coulomb.rs    # 23/23 checks
```

### Active Handoff Documents

```
wateringHole/handoffs/HOTSPRING_V064_TOADSTOOL_HANDOFF_FEB22_2026.md  ← this document
wateringHole/handoffs/HOTSPRING_TOADSTOOL_REWIRE_V4_FEB22_2026.md     ← spectral lean
wateringHole/handoffs/CROSS_SPRING_EVOLUTION_FEB22_2026.md            ← shader evolution map
wateringHole/handoffs/HOTSPRING_V063_EVOLUTION_HANDOFF_FEB22_2026.md  ← v0.6.3 extraction
```

22 prior handoffs archived to `wateringHole/handoffs/archive/`.

---

## Codebase Health (Feb 22, 2026)

| Metric | Value |
|--------|-------|
| Crate version | v0.6.4 |
| Unit tests | 637 pass (6 GPU/heavy-ignored; spectral tests now upstream) |
| Integration tests | 24 pass (3 suites) |
| Validation suites | 33/33 pass |
| metalForge forge tests | 19 pass |
| Coverage | 74.9% region / 83.8% function |
| Clippy warnings | 0 (pedantic + nursery) |
| Doc warnings | 0 |
| Unsafe blocks | 0 |
| `expect()`/`unwrap()` in library | 0 |
| Papers reproduced | 22 |
| Centralized tolerances | 154 constants |
| WGSL shaders (library) | 34 (zero inline) |
| Python control scripts | 34 |
| Rust validation binaries | 50 |

---

*This handoff follows the unidirectional pattern: hotSpring → wateringHole → ToadStool.
No inter-Spring imports. All code is AGPL-3.0.*
