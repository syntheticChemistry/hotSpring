# Evolution Readiness: Rust → WGSL Shader Promotion → ToadStool Absorption

This document maps each Rust module to its GPU shader readiness tier,
tracks what toadstool has absorbed, and identifies next absorption targets.

## Evolution Path

```
Python baseline → Rust validation → WGSL template → GPU shader → ToadStool absorption → Lean on upstream
```

## v0.6.32 Evolution Sprint (Mar 17-22, 2026)

### Rewired to modern barraCuda v0.3.7 (was f82d60c6 → now 32554b0a)
### coralReef synced to ce66de4 (BDF allowlist, preflight device checks, VRAM write-readback)

| New Primitive | Origin | Cross-Spring Value |
|---------------|--------|-------------------|
| `FmaPolicy` (Contract/Separate/Default) | hotSpring precision brain + coralReef Iter 30 | All springs: FMA contraction control for bit-exact QCD, gradient flow, nuclear EOS |
| `PrecisionTier` (F32/DF64/F64/F64Precise) | hotSpring v0.6.25 → barraCuda Sprint 2 | All springs: domain-aware shader compilation routing |
| `PhysicsDomain` (LatticeQcd, GradientFlow, ...) | hotSpring v0.6.25 → barraCuda Sprint 2 | Maps physics domains to precision requirements |
| `domain_requires_separate_fma()` | hotSpring + coralReef | Returns true for LatticeQcd, GradientFlow, NuclearEos |
| `special::stable_gpu::{log1p_f64, expm1_f64, erfc_f64, bessel_j0_minus1_f64}` | wetSpring + hotSpring → Sprint 2 | Cancellation-safe specials: hotSpring screened Coulomb (erfc), BCS (expm1); wetSpring HMM (log1p) |
| `GemmF64::execute_gemm_ex(trans_a, trans_b)` | neuralSpring → Sprint 6 | A^T*B without transpose materialization: Gram matrices, normal equations, covariance |
| `WGSL_MEAN_REDUCE_F64` | neuralSpring re-export → Sprint 6 | Custom mean reduction pipelines |

### Cross-Spring Shader Evolution Map

```
hotSpring (precision, lattice, MD, HFB, spectral)
  ├─ DF64 core streaming ────────→ wetSpring bio spectral, neuralSpring RMT
  ├─ ReduceScalarPipeline ───────→ wetSpring, all springs GPU reductions
  ├─ ESN reservoir/readout ──────→ neuralSpring GPU ESN, wetSpring bio ESN
  ├─ NeighborMode 4D ────────────→ neuralSpring graph lattices
  ├─ SpectralAnalysis+RMT ───────→ neuralSpring classifier, wetSpring bio spectral
  ├─ HFB/SEMF/chi2 shaders ─────→ (nuclear-specific, not yet cross-spring)
  └─ Precision brain architecture → ALL springs via PrecisionTier + FmaPolicy

wetSpring (bio, diversity, HMM, alignment)
  ├─ log_f64/pow_f64 fixes ──────→ hotSpring BCS gap equations
  ├─ Shannon/Simpson/Bray-Curtis → neuralSpring eco-diversity metrics
  ├─ HMM forward GPU ────────────→ neuralSpring sequence models
  ├─ Stable specials (log1p etc) → hotSpring screened Coulomb, dielectric
  └─ Gillespie SSA ──────────────→ (bio-specific, potential hotSpring kinetics)

neuralSpring (ML, pairwise, spectral, optimization)
  ├─ Batched Nelder-Mead GPU ────→ hotSpring HMC parameter tuning
  ├─ GemmF64 transpose ──────────→ hotSpring surrogate fitting, groundSpring least-squares
  ├─ Pairwise metrics ───────────→ wetSpring similarity, hotSpring MD clustering
  ├─ Linear regression GPU ──────→ hotSpring observable fits, groundSpring calibration
  └─ Matrix correlation GPU ─────→ hotSpring obs correlations (plaquette vs Polyakov)

groundSpring (noise, stats, hydrology)
  ├─ Chi-squared GPU ────────────→ hotSpring nuclear χ² fits, wetSpring enrichment
  ├─ FFT radix-2 GPU ────────────→ all springs Fourier analysis
  └─ Jackknife resampling ───────→ hotSpring bootstrap CI, wetSpring population stats
```

### Deep Debt Burndown + Cross-Vendor Dispatch (Mar 22, 2026, Exp 075)

13 deep-debt items resolved across coralReef + hotspring-barracuda:

| Item | Category | Impact |
|------|----------|--------|
| TOCTOU BusyGuard | P0 | Safe concurrent oracle captures on dual Titans |
| Buffer handle validation | P0 | Explicit error on invalid dispatch handles |
| BDF-specific dispatch | P0 | No silent fallback to wrong GPU |
| coralctl health fix | P1 | Accurate HEALTHY/DEGRADED/DOWN reporting |
| Async nvidia-smi | P1 | RPC responsiveness under load |
| `try_read_u32`/`try_write_u32` | P1 | Safe BAR0 access for PMU debugging |
| `OracleError` variant | P1 | Clean oracle error propagation |
| Optional deps (`cuda-validation` feature) | P2 | `cudarc`/`base64` gated — `cargo check` fast on non-CUDA |
| saxpy.ptx sm_70 | P2 | Volta+ compatible PTX (was sm_90 Hopper-only) |
| BufReader 64KB | P2 | Reduced per-connection memory footprint |

**New validation binaries:**
- `validate_5060_dual_use` — RTX 5060 display + CUDA compute proof-of-concept
- `validate_cross_vendor_dispatch` — CUDA dispatch via glowplug daemon RPC (zero pkexec)

**pkexec-free pipeline:** Entire compute lifecycle (enumerate, swap, capture, dispatch, health) operates through Unix socket RPC. No pkexec, no sudo, no SUID.

### Vendor-Agnostic Register Maps (Mar 18, 2026)

New `src/register_maps/` module with `RegisterMap` trait for GPU introspection:

| Component | Description |
|-----------|-------------|
| `RegisterMap` trait | `vendor()`, `arch()`, `registers()`, `decode_temp_c()`, `decode_boot_id()`, `thermal_offset()` |
| `NvGv100Map` | 127 NVIDIA GV100 BAR0 registers (PMC, PBUS, PFIFO, PBDMA, PFB, FBHUB, PMU, PCLOCK, GR, FECS, GPCCS, LTC, FBPA, PRAMIN, THERM) |
| `AmdGfx906Map` | AMD Vega 20 / MI50 registers (SRBM, GRBM, MMHUB, GFX, SDMA, IH, THM, SMN, HDP) |
| `detect_register_map(vendor_id)` | Runtime vendor selection → `Box<dyn RegisterMap>` |
| `RegisterDump` / `RegisterEntry` | Unified JSON output types with vendor field |

**Absorption target**: This module belongs in barraCuda long-term. hotSpring will lean on upstream after absorption.

### Triangle Architecture + VendorProfile Convergence (Mar 20, 2026)

The compute trio now operates as a triangle (coralReef ↔ toadStool ↔ barraCuda).
barraCuda's `RegisterMap` trait and coralReef's `VendorLifecycle` trait both dispatch
from PCI vendor IDs — these should converge into a unified `VendorProfile`:

```
VendorProfile
  ├── RegisterMap      (hardware introspection: registers, temp decode, boot ID)
  ├── VendorLifecycle  (swap orchestration: rebind strategy, settle times, health)
  └── ShaderISA        (compilation hints: wave size, subgroup ops, FMA policy)
```

This convergence is the trio's next evolution priority. barraCuda owns the math,
but it needs to know what hardware is available (via toadStool, which queries
coralReef GlowPlug). The triangle data flow:

```
barraCuda: "I need DF64 matrix multiply on 16GB HBM2"
  → toadStool: "Radeon VII available, compiling shader..."
    → coralReef: WGSL → SPIR-V, vendor-optimized
  → toadStool: submit to hardware
  → barraCuda: receives results, validates physics
```

**BrainChip Akida NPU** (0x1e7c:0xbca1) is now integrated into GlowPlug with
`BrainChipLifecycle` and `AkidaPersonality`. This proves the lifecycle system
handles any PCIe device, not just GPUs. The `RegisterMap` trait could be
extended to cover NPU register spaces for hardware telemetry.

### Ember/GlowPlug Hardening (Mar 22, 2026 — coralReef ce66de4)

Three deep debts resolved in the sovereign GPU lifecycle layer:

| Debt | Resolution | coralReef Location |
|------|-----------|-------------------|
| VRAM health false positives | Write-readback canary (`0xC0A1_BEEF`) replaces read-nonzero | `coral-glowplug/src/device/health.rs` |
| Unmanaged BDF operations | `HashSet<String>` allowlist in ember, JSON-RPC `-32001` error | `coral-ember/src/{lib,ipc}.rs` |
| D-state kernel hangs | Pre-flight: sysfs existence, D0 power state, config space 0xFFFF guard | `coral-ember/src/swap.rs` |

Additional hardening:
- Display GPU safety guard prevents unbind of active display devices
- Test isolation via `EmberClient::disable_for_test()` thread-local RAII guard
- 86 ember + 178 glowplug tests pass (264 total)

**Relevance to barraCuda**: The `RegisterMap` module's health-check patterns should adopt
the write-readback canary approach rather than simple register reads. The BDF allowlist
pattern (config-driven device scope) is a model for any multi-device management.

### Sovereign Command Submission Pipeline (Mar 21, 2026 — Exp 071)

The PFIFO diagnostic matrix in coralReef's `coral-driver` crate proved 6/10
sovereign pipeline layers working on GV100 via VFIO. Key discoveries:

- **PFIFO re-init sequence**: PMC reset (bit 8) → soft enable → preempt all → clear PBDMAs
- **GP_FETCH register at 0x050** (not 0x048 as documented — 0x048 is GP_STATE)
- **PBDMA context loading works**: GP_BASE, USERD, SIG, GP_PUT all loaded correctly
- **MMU translation is the blocker**: 0xbad00200 PBUS timeout fetching GPU VA 0x1000

**Relevance to barraCuda**: Once sovereign dispatch works, `MdEngine<B: GpuBackend>`
can route through a `SovereignBackend` that uses coral-driver for GPFIFO submission.
The `RegisterMap` module's GV100 register definitions should incorporate the PBDMA
operational register corrections (CTX_GP_FETCH_BYTE at 0x050).

### DRM Dispatch Dual-Track (Mar 21, 2026 — Exp 072, Phase 3 Preswap)

In parallel with sovereign VFIO, coral-driver has **fully coded DRM dispatch**:

- **AMD** (`AmdDevice`): `ComputeDevice` impl with GEM buffers, PM4 command
  construction (`build_compute_dispatch`), `DRM_AMDGPU_CS` submission, fence sync.
  **Full preswap 6/6 PASS** on MI50 (GFX906/GCN5): f64 write, f64 arithmetic,
  multi-workgroup, multi-buffer read/write, HBM2 bandwidth, **f64 Lennard-Jones
  force (Newton's 3rd law verified)**. 18 GCN5 bugs fixed. 85 coral-reef tests pass.
- **NVIDIA** (`NvDevice`): new UAPI (`VM_INIT`/`VM_BIND`/`EXEC`) + syncobj.
  Blocked on Titan V (missing PMU firmware for `CHANNEL_ALLOC`). K80 (Kepler,
  incoming) has no PMU requirement.

**Relevance to barraCuda**: DRM dispatch is the **fastest path to working DF64
compute** — it bypasses the Naga WGSL→SPIR-V poisoning (Exp 055) entirely:

```
WGSL → coral-reef AmdBackend → native GCN ISA → coral-driver AmdDevice → GPU
```

coral-reef now has a `Gcn5` variant in `AmdArch` — **GCN5 preswap complete**
(March 2026): WGSL → coral-reef → coral-driver PM4 → MI50. **6/6 phases PASS**
(f64 write, f64 arith, multi-workgroup, multi-buffer, HBM2 bandwidth, f64 LJ force).
18 coral-reef bugs fixed. The MI50's 1/4 rate f64 (3.5 TFLOPS) is **4× faster
than RDNA2** for DF64. **DF64 Lennard-Jones verified via DRM** — the Naga-poisoned
kernel produces correct forces through the sovereign bypass path. Newton's 3rd
law confirmed (equal and opposite forces, CPU ref match to tol=1e-8).

**DF64 kernel candidates (ready for DRM dispatch)**:
- `SHADER_YUKAWA_FORCE` — Lennard-Jones (**VALIDATED** — the Naga-poisoned kernel works via DRM)
- `wilson_plaquette_df64.wgsl` — lattice QCD gauge action (ready)
- `su3_gauge_force_df64.wgsl` — HMC gauge force (ready)

**`RegisterMap` evolution**: DRM dispatch works on MI50 — the GFX906 register
map is now testable against real hardware via `AmdDevice::dispatch()`. This
validates `RegisterMap` encodings that were previously theoretical.

### Code Quality Improvements

- `#![forbid(unsafe_code)]` added to lib.rs (compiler-enforced zero-unsafe)
- License: `AGPL-3.0-only` → `AGPL-3.0-or-later` (scyBorg trio alignment)
- Zero clippy pedantic+nursery warnings across all targets (was 8)
- barraCuda pin: `f82d60c6` → `32554b0a` (4 commits, Sprint 12-14, v0.3.7)
- Cross-spring benchmark: 4 new phases (FMA routing, stable specials, GemmF64 transpose, precision tiers)
- 848 tests, 0 failures, 6 GPU-ignored

## Tier Definitions

| Tier | Label | Meaning |
|------|-------|---------|
| **A** | Rewire | Shader exists and is validated; wire into pipeline |
| **B** | Adapt | Shader exists but needs modification (API, precision, layout) |
| **C** | New | No shader exists; must be written from scratch |
| **✅** | Absorbed | ToadStool has absorbed this as a first-class barracuda primitive |

## ToadStool Absorption Status (Mar 22, 2026 — v0.6.32 synced to toadStool S158 + coralReef ce66de4 + barraCuda v0.3.7, 848 tests)

| hotSpring Module | ToadStool Primitive | Absorbed At | Status |
|-----------------|--------------------| -------|--------|
| `lattice/complex_f64.rs` WGSL template | `shaders/math/complex_f64.wgsl` + `ops/lattice/complex_f64` | S25 | ✅ Absorbed |
| `lattice/su3.rs` WGSL template | `shaders/math/su3.wgsl` + `ops/lattice/su3` + `cpu_su3` | S25 | ✅ Absorbed |
| Wilson plaquette design | `ops/lattice/plaquette` + `gpu_wilson_action` | S25 | ✅ Absorbed |
| HMC force design | `ops/lattice/hmc_force_su3` | S25 | ✅ Absorbed |
| Abelian Higgs design | `ops/lattice/higgs_u1` | S25 | ✅ Absorbed |
| `lattice/dirac.rs` (Dirac SpMV) | `ops/lattice/dirac` + `cpu_dirac` | S46-S52 | ✅ **Absorbed** (was Tier A) |
| `lattice/cg.rs` (CG solver) | `ops/lattice/cg` + `gpu_cg_solver` + `gpu_cg_resident` | S51-S52 | ✅ **Absorbed** (was Tier A) |
| `lattice/pseudofermion/` (pseudo HMC) | `ops/lattice/pseudofermion` + `gpu_pseudofermion` | S46 | ✅ **Absorbed** |
| `lattice/wilson.rs` (CPU Lattice) | `ops/lattice/wilson` (test-gated) | S46 | ✅ Absorbed (CPU ref) |
| `lattice/constants.rs` (LCG, PRNG) | `ops/lattice/constants` (test-gated) | S46 | ✅ Absorbed |
| `lattice/hmc.rs` (CPU HMC) | `ops/lattice/gpu_hmc_leapfrog` + `gpu_hmc_trajectory` | S52 | ✅ Absorbed (GPU version) |
| GPU HMC trajectory (full) | `ops/lattice/gpu_hmc_trajectory` | S52 | ✅ **Absorbed** — full dynamical HMC on GPU |
| Local `GpuCellList` | `CellListGpu` (BGL fixed) | S25 | ✅ Absorbed (local deprecated) |
| NAK eigensolve workarounds | `batched_eigh_nak_optimized_f64.wgsl` | `82f953c8` | ✅ Absorbed |
| FFT need documented | `Fft1DF64` + `Fft3DF64` | `1ffe8b1a` | ✅ Absorbed |
| `ReduceScalar` feedback | `ReduceScalarPipeline` (`scalar_buffer`, `max_f64`, `min_f64`) | v0.5.16 | ✅ Absorbed |
| Driver profiling feedback | `GpuDriverProfile` + `WgslOptimizer` | v0.5.15 | ✅ Absorbed |
| Spectral module (41 KB) | `barracuda::spectral::*` | S25-S31 | ✅ Absorbed (local deleted) |
| ESN reservoir + readout | `esn_v2::MultiHeadEsn` + `ExportedWeights` | S79 | ✅ Absorbed |
| Nautilus brain + shell | `nautilus::{NautilusBrain, Shell, Evolution, Board, Population}` | S80 | ✅ **Absorbed** |
| `BatchedEncoder` pattern | `device::BatchedEncoder` | S80 | ✅ Absorbed |

### Completed Absorption Targets (historical)

| hotSpring Module | Status | Notes |
|-----------------|--------|-------|
| `spectral/csr.rs::CsrMatrix::spmv()` | ✅ Done | `WGSL_SPMV_CSR_F64` validated 8/8 checks |
| `spectral/lanczos.rs::lanczos()` | ✅ Done | GPU SpMV inner loop + CPU control, 6/6 checks |
| `lattice/dirac.rs` | ✅ Done | `WGSL_DIRAC_STAGGERED_F64` validated 8/8 checks |
| `lattice/cg.rs` | ✅ Done | GPU CG (D†D) validated 9/9 checks |
| `lattice/pseudofermion/` | ✅ Done | Pseudofermion HMC: heat bath, CG action, fermion force |
| GPU HMC trajectory (full) | ✅ Done | `gpu_hmc_trajectory` in toadStool S52 |
| ESN reservoir + readout | ✅ Done | `esn_v2::MultiHeadEsn` in toadStool S79 |
| Nautilus brain + shell | ✅ Done | toadStool S80 |
| `BatchedEncoder` pattern | ✅ Done | toadStool S80 |

### Active Absorption Targets (Mar 6, 2026)

| hotSpring Module | Lines | Priority | Absorption Value |
|-----------------|-------|----------|-----------------|
| `lattice/rhmc.rs` | ~600 | **P0** | RHMC: Remez+pole-optimization rational approximation, multi-shift CG, `rhmc_heatbath`/`action`/`force` — unlocks Nf=2, 2+1 for all springs |
| `lattice/gpu_hmc/hasenbusch.rs` | ~350 | **P1** | GPU Hasenbusch mass preconditioning — multi-scale leapfrog, unlocks light quarks (m < 0.05) |
| Multi-field `Vec<phi_bufs>` pattern | Δ+300 | **P1** | `dynamical.rs`, `streaming.rs`, `resident_cg.rs`, `resident_cg_brain.rs` — arbitrary Nf via field loop |
| `gpu_hmc/resident_cg_brain.rs` | ~400 | **P2** | NPU-interleaved CG with `BrainInterrupt` — neuromorphic steering during solve |
| `gpu_hmc/resident_cg_async.rs` | ~300 | **P2** | Latency-adaptive CG check intervals, async GPU→CPU readback |
| `production/npu_worker.rs` | ~1000 | **P3** | NPU parameter controller pattern: dt/n_md control, safety clamps, acceptance targeting |

## Physics Modules

| Rust Module | WGSL Shader(s) | Tier | Status | Blocker |
|-------------|----------------|------|--------|---------|
| `physics/semf.rs` | `SHADER_SEMF_BATCH`, `SHADER_CHI2` (inline in `nuclear_eos_gpu.rs`) | **A** | GPU pipeline exists | None — production-ready |
| `physics/hfb.rs` | `batched_hfb_*.wgsl` (4 shaders) via `hfb_gpu.rs` | **A** | GPU pipeline exists | None — validated against CPU |
| `physics/hfb_gpu.rs` | Uses `BatchedEighGpu::execute_single_dispatch` | **A** | Production GPU — single-dispatch (v0.5.3) | None — all rotations in one shader |
| `physics/bcs_gpu.rs` | `bcs_bisection_f64.wgsl` | **A** | Production GPU — pipeline cached (v0.5.3) | None — ToadStool `target` bug absorbed (`0c477306`) |
| `physics/hfb_gpu_resident/` | `batched_hfb_potentials_f64.wgsl`, `batched_hfb_hamiltonian_f64.wgsl`, `batched_hfb_density_f64.wgsl`, `batched_hfb_energy_f64.wgsl`, `BatchedEighGpu`, `SpinOrbitGpu` | **A** | GPU H-build + eigensolve + spin-orbit + density + mixing + energy (v0.6.2) | BCS Brent on CPU (root-finding not GPU-efficient) |
| `physics/hfb_deformed/` | — | **C** | CPU only (refactored: mod, potentials, basis, tests) | Deformed HFB needs new shaders for 2D grid Hamiltonian build |
| `physics/hfb_deformed_gpu/` | `deformed_*.wgsl` (5 shaders exist, not all wired) | **B** | Partial GPU (refactored: mod, types, physics, gpu_diag, tests) | H-build on CPU; deformed Hamiltonian shaders exist but unwired |
| `physics/nuclear_matter.rs` | — | **C** | CPU only | Uses `barracuda::optimize::bisect` (CPU); no NMP shader. Low priority — fast on CPU |
| `physics/hfb_common.rs` | — | N/A | Shared utilities | Pure CPU helpers (WS radii, deformation estimation) |
| `physics/constants.rs` | — | N/A | Physical constants | Data only |

## MD Modules

| Rust Module | WGSL Shader(s) | Tier | Status | Blocker |
|-------------|----------------|------|--------|---------|
| `md/simulation.rs` | Yukawa (all-pairs + cell-list), VV integrator, Berendsen thermostat, KE per-particle, `ReduceScalarPipeline` (inline in `md/shaders.rs`) | **A** | Full GPU pipeline | None — production-ready |
| `md/celllist.rs` | GPU cell-list via upstream `CellListGpu` + indirect force shader. Zero CPU readback | **✅** | **Migrated** (v0.6.2) — local `GpuCellList` deleted, upstream `barracuda::ops::md::CellListGpu` | None |
| `md/shaders.rs` | 11 WGSL shaders (all `.wgsl` files, zero inline). GPU cell-list shaders added v0.5.13 | **A** | Production | v0.6.3: all inline extracted to `.wgsl` |
| `md/observables/` | Uses `SsfGpu` from BarraCuda | **A** | SSF on GPU; RDF/VACF CPU post-process | VACF now correct (particle identity preserved by indirect indexing) |
| `md/cpu_reference.rs` | — | N/A | Validation reference | Intentionally CPU-only for baseline comparison |
| `md/config.rs` | — | N/A | Configuration | Data structures only |

## WGSL Shader Inventory

### Physics Shaders (`src/physics/shaders/`, 10 files, ~1950 lines)

| Shader | Lines | Pipeline Stage |
|--------|-------|----------------|
| `batched_hfb_density_f64.wgsl` | 150 | Density + BCS + mixing for spherical HFB (batched wf) |
| `batched_hfb_potentials_f64.wgsl` | 170 | Skyrme potentials (U_total, f_q) |
| `batched_hfb_hamiltonian_f64.wgsl` | 123 | HFB Hamiltonian H = T_eff + V |
| `batched_hfb_energy_f64.wgsl` | 147 | HFB energy functional (shared-memory reduce) |
| `bcs_bisection_f64.wgsl` | 141 | BCS chemical-potential bisection |
| `deformed_wavefunction_f64.wgsl` | 241 | Nilsson HO wavefunctions on 2D (ρ,z) grid |
| `deformed_hamiltonian_f64.wgsl` | 214 | Block Hamiltonian for deformed HFB |
| `deformed_density_energy_f64.wgsl` | 293 | Deformed density, energy, Q20, RMS radius |
| `deformed_gradient_f64.wgsl` | 205 | Gradient of deformed densities |
| `deformed_potentials_f64.wgsl` | 268 | Deformed mean-field potentials |

### MD Reference Shaders (absorbed — directory removed)

Toadstool reference shaders were absorbed upstream and the
`src/md/shaders_toadstool_ref/` directory was deleted in v0.6.3.
Production equivalents live in `src/md/shaders/`.

### MD Production Shaders (`src/md/shaders/`)

| Shader | Physics | Location |
|--------|---------|----------|
| `yukawa_force_f64.wgsl` | Yukawa all-pairs (native f64) | `.wgsl` file |
| `yukawa_force_celllist_f64.wgsl` | Cell-list v1 (27-neighbor, sorted positions) | `.wgsl` file |
| `yukawa_force_celllist_v2_f64.wgsl` | Cell-list v2 (flat loop, sorted positions) | `.wgsl` file |
| `yukawa_force_celllist_indirect_f64.wgsl` | Cell-list indirect (unsorted positions + `sorted_indices`) **(v0.5.13)** | `.wgsl` file |
| `vv_kick_drift_f64.wgsl` | Velocity-Verlet kick+drift | `.wgsl` file |
| `vv_half_kick_f64.wgsl` | VV second half-kick | `.wgsl` file **(v0.6.3)** |
| `berendsen_f64.wgsl` | Berendsen thermostat rescale | `.wgsl` file **(v0.6.3)** |
| `kinetic_energy_f64.wgsl` | Kinetic energy reduction | `.wgsl` file **(v0.6.3)** |
| `rdf_histogram_f64.wgsl` | RDF histogram binning | `.wgsl` file |
| `esn_reservoir_update.wgsl` | ESN reservoir state update (f32) | `.wgsl` file |
| `esn_readout.wgsl` | ESN readout layer (f32) | `.wgsl` file |

**Note**: `cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl` were deleted in v0.6.2 (GPU cell-list build migrated to upstream `CellListGpu`).

## BarraCuda Primitives Used

| BarraCuda Module | hotSpring Usage |
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
| `barracuda::pipeline::ReduceScalarPipeline` | GPU f64 sum-reduction (KE, PE, thermostat) **(v0.5.12)** |
| `GpuCellList` (local, **deprecated**) | GPU-resident 3-pass cell-list build — upstream `CellListGpu` fixed (toadstool `8fb5d5a0`) |
| `barracuda::ops::lattice::*` | Complex f64, SU(3), Wilson plaquette, HMC force, Abelian Higgs GPU shaders (toadstool `8fb5d5a0`) |
| `barracuda::ops::fft::Fft1DF64` | GPU FFT f64 for momentum-space (toadstool `1ffe8b1a`) |
| `barracuda::ops::fft::Fft3DF64` | GPU 3D FFT for lattice QCD / PPPM (toadstool `1ffe8b1a`) |
| `barracuda::spectral::{spectral_bandwidth, spectral_condition_number}` | **NEW S80** — Proxy bandwidth and condition number |
| `barracuda::spectral::{classify_spectral_phase, SpectralAnalysis}` | **NEW S80** — RMT-based Marchenko–Pastur phase classifier |
| `barracuda::ops::lattice::NeighborMode` | **NEW S80** — 4D precomputed neighbor table (tested, different index convention) |
| `barracuda::optimize::batched_nelder_mead_gpu` | **NEW S79** — Batched parallel Nelder-Mead on GPU |
| `barracuda::esn_v2::{MultiHeadEsn, HeadGroup, HeadConfig}` | **NEW S78** — GPU ESN with per-head training (serde-compatible weights) |
| `barracuda::device::BatchedEncoder` | **NEW S79** — Fuse multiple dispatches into single submission |
| `barracuda::device::{WgpuDevice, TensorContext}` | GPU device bridge |

No duplicate math — all mathematical operations use BarraCuda primitives.
`hermite_value` now delegates to `barracuda::special::hermite` (v0.5.7).
`factorial_f64` now delegates to `barracuda::special::factorial` (v0.5.10).
`solve_linear_system` in `reservoir/` delegates to `barracuda::ops::linalg::lu_solve` for dense linear solves.
WGSL `abs_f64` and `cbrt_f64` now injected via `ShaderTemplate::with_math_f64_auto()` (v0.5.8).
Force shaders compiled via `GpuF64::create_pipeline_f64()` → barracuda driver-aware path **(v0.5.11)**.
`GpuCellList` migrated to upstream `barracuda::ops::md::CellListGpu` (v0.6.2) — 3 local shaders deleted.

## Completed (v0.6.3, Feb 22 2026)

- ✅ **Inline WGSL extraction**: 5 more inline shader strings extracted to `.wgsl` files:
  - `md/shaders.rs`: `SHADER_VV_HALF_KICK` → `vv_half_kick_f64.wgsl`, `SHADER_BERENDSEN` → `berendsen_f64.wgsl`, `SHADER_KINETIC_ENERGY` → `kinetic_energy_f64.wgsl`
  - `lattice/complex_f64.rs`: `WGSL_COMPLEX64` → `shaders/complex_f64.wgsl`
  - `lattice/su3.rs`: `WGSL_SU3` → `shaders/su3_f64.wgsl`
- ✅ **Deformed HFB coverage**: 13 new tests covering `diagonalize_blocks` (V=0, constant V, sharp Fermi), `potential_matrix_element` (constant V, Hermitian symmetry), `solve()` SCF loop (smoke test, determinism, physical bounds), `binding_energy_l3`, and Hermite/Laguerre norm integrals
- ✅ **648 tests** (was 638), 0 failures, 6 ignored
- ✅ **Stale documentation cleaned**: Deleted shader references (`cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl`) removed from shader inventory; extracted shaders added

## Completed (v0.6.2, Feb 21 2026)

- ✅ **Zero clippy pedantic+nursery warnings**: was ~1500 in v0.6.1, now 0. Systematic resolution of `mul_add` (150+), `doc_markdown` (600+), `must_use` (186+), `imprecise_flops` (30+), `use_self` (14), `const_fn` (4), `option_if_let_else` (5), `HashMap` hasher (2), `significant_drop_tightening` (1).
- ✅ **Duplicate math eliminated**: `reservoir/` Gaussian elimination → `barracuda::linalg::solve_f64`
- ✅ **GPU energy pipeline wired**: `batched_hfb_energy_f64.wgsl` dispatched in SCF loop behind `gpu_energy` feature flag
- ✅ **Large file refactoring**: `bench.rs` (1005→4 files), `hfb_gpu_resident/mod.rs` (7 helpers extracted), `celllist_diag.rs` (1156→951)
- ✅ **Cast safety documentation**: Crate-level `#![allow]` with mantissa/range analysis; per-function annotations on critical GPU casts
- ✅ **MutexGuard tightening**: `PowerMonitor::finish()` clones samples immediately, drops lock before processing
- ✅ **561 tests** (was 505), 0 failures, 67.4% region / 78.8% function coverage
- ✅ **metalForge/forge**: zero pedantic warnings
- ✅ **Version**: 0.6.0 → 0.6.2

## Completed (v0.6.1, Feb 21 2026)

- ✅ **Zero `expect()`/`unwrap()` in library code**: `#![deny(clippy::expect_used, clippy::unwrap_used)]` enforced crate-wide. All 15 production `expect()` calls replaced with `Result` propagation, `bytemuck` zero-copy, or safe pattern matching.
- ✅ **Tolerances module tree**: `tolerances.rs` (1384 lines) refactored into `tolerances/{mod,core,md,physics,lattice,npu}.rs`. Each submodule under 300 lines. Zero API change via `pub use` re-exports.
- ✅ **Solver config centralized**: 8 new constants (`HFB_MAX_ITER`, `BROYDEN_WARMUP`, `BROYDEN_HISTORY`, `HFB_L2_MIXING`, `HFB_L2_TOLERANCE`, `FERMI_SEARCH_MARGIN`, `CELLLIST_REBUILD_INTERVAL`, `THERMOSTAT_INTERVAL`) extracted from 7 files. Zero hardcoded solver params in library code.
- ✅ **Large file refactoring**: 4 monolithic files decomposed into module directories: `hfb/` (3 files), `hfb_deformed/` (4 files), `hfb_deformed_gpu/` (5 files), `hfb_gpu_resident/` (3 files). All new files under 500 LOC except `hfb_gpu_resident/mod.rs` (1456 — monolithic GPU pipeline).
- ✅ **Integration test suites**: 3 new suites: `integration_physics.rs` (11), `integration_data.rs` (8), `integration_transport.rs` (5) — 24 tests covering cross-module interactions.
- ✅ **Provenance completeness**: `SCREENED_COULOMB_PROVENANCE` and `DALIGAULT_CALIBRATION_PROVENANCE` added. Commit verification documentation included.
- ✅ **Capability-based discovery**: `try_discover_data_root()` returns `Result`; `available_capabilities()` probes runtime validation domains.
- ✅ **Tolerances tightened**: `ENERGY_DRIFT_PCT` 5%→0.5%, `RDF_TAIL_TOLERANCE` 0.15→0.02 (both still 10×+ above measured worst case).
- ✅ **Zero-copy GPU buffer reads**: `bytemuck::try_cast_slice` with alignment fallback replaces manual byte conversion.
- ✅ **Control JSON policy documented** in tolerances module.
- ✅ **505 unit tests + 24 integration + 8 forge (505 passing + 5 GPU-ignored), 0 clippy warnings, 0 doc warnings, 154 centralized constants**

## Promotion Priority

1. ~~**GPU energy integrands + SumReduceF64**~~ ✅ **DONE (v0.6.2)** — `batched_hfb_energy_f64.wgsl` wired into SCF loop behind `gpu_energy` feature flag. `compute_energy_integrands` + `compute_pairing_energy` GPU passes, staging buffer readback. CPU fallback preserved.
2. ~~**BCS on GPU**~~ ✅ **DONE (v0.5.10)** — Density + mixing on GPU; BCS Brent remains on CPU (root-finding not GPU-efficient)
3. ~~**SpinOrbitGpu**~~ ✅ **DONE (v0.5.6)** — Wired with CPU fallback
4. ~~**WGSL preamble injection**~~ ✅ **DONE (v0.5.8)** — `ShaderTemplate::with_math_f64_auto()`
5. **hfb_deformed_gpu.rs** → Wire existing deformed_*.wgsl shaders for full GPU H-build
6. **nuclear_matter.rs** → Low priority; CPU bisection is fast enough

## Completed (v0.5.9)

- ✅ **Final tolerance wiring pass**: 6 new constants (`BCS_DENSITY_SKIP`, `SHARP_FILLING_THRESHOLD`,
  `DEFORMED_COULOMB_R_MIN`, `DEFORMATION_GUESS_WEAK/GENERIC/SD`) — 15 remaining inline values
  in `hfb.rs`, `hfb_deformed.rs`, `hfb_deformed_gpu.rs`, `md/observables/` → named constants
- ✅ **Clippy pedantic**: 0 clippy warnings across all targets
- ✅ **Full audit report**: specs, wateringHole compliance, validation fidelity, dependency health,
  evolution readiness, test coverage, code size, licensing, data provenance

## CPU vs GPU Scaling (v0.5.11, Feb 19 2026)

See `experiments/007_CPU_GPU_SCALING_BENCHMARK.md` for full data.

| N | GPU mode | CPU steps/s | GPU steps/s | Speedup |
|------:|:-----------|----------:|----------:|--------:|
| 108 | all-pairs | 10,734 | 4,725 | 0.4× |
| 500 | all-pairs | 651 | 1,167 | **1.8×** |
| 2,000 | all-pairs | 67 | 449 | **6.7×** |
| 5,000 | all-pairs | ~6.5* | 158 | **~24×** |
| 10,000 | cell-list | ~1.6* | 136 | **~84×** |

Paper-parity run (N=10k, 80k steps): 9.8 min, $0.0012. 98 runs/day idle.

### Streaming dispatch (v0.5.11)

- `GpuF64::begin_encoder()` / `submit_encoder()` / `read_staging_f64()`:
  batch multiple VV steps into single GPU submission
- Production MD: `dump_step` iterations batched per encoder
- Cell-list: `rebuild_interval=20` steps between CPU-side rebuilds
- Result: N=500 GPU went from 1.0× to 1.8×; N=10000 cell-list at 136 steps/s

### Next evolution targets

1. ~~**GPU-resident reduction**~~ ✅ **DONE (v0.5.12)** — `ReduceScalarPipeline` from ToadStool
2. ~~**GPU-resident cell-list**~~ ✅ **DONE (v0.5.13)** — 3-pass GPU build (bin→scan→scatter) + indirect force shader. ToadStool's `CellListGpu` had prefix_sum binding mismatch; hotSpring built corrected implementation locally. Ready for ToadStool absorption.
3. ~~**Daligault D* model evolution**~~ ✅ **DONE (v0.5.14)** — κ-dependent weak-coupling correction `C_w(κ)` replaces constant `C_w=5.3`. Crossover-regime error reduced from 44-63% to <10% across all 12 Sarkas calibration points. See Completed (v0.5.14).
4. **`StatefulPipeline` for MD loops**: ToadStool's `run_iterations()` / `run_until_converged()` could replace manual encoder batching. Architectural change — deferred.
5. **GPU HFB energy integrands**: Shader exists (`batched_hfb_energy_f64.wgsl`); wiring requires threading 10+ buffer references through `GroupResources`. Estimated ~200 lines of refactoring.
6. **Titan V proprietary driver**: unlock 6.9 TFLOPS fp64, fair hardware comparison

## Completed (v0.5.14)

- ✅ **Daligault D* model evolution**: κ-dependent weak-coupling correction
  - Replaced constant `C_w=5.3` with `C_w(κ) = exp(1.435 + 0.715κ + 0.401κ²)`
  - Root cause: Yukawa screening suppresses the Coulomb logarithm faster than the classical formula captures; correction grows exponentially with κ (4.2× at κ=0 → 1332× at κ=3)
  - Crossover-regime errors: 44-63% → <10% (12/12 points pass 20% per-point, RMSE<10%)
  - Same `C_w(κ)` applied to η*_w and λ*_w (Chapman-Enskog transport)
  - Calibrated from `calibrate_daligault_fit.py` weak-coupling correction analysis
- ✅ **12 Sarkas D_MKS provenance constants**: all 12 Green-Kubo D_MKS values from
  `all_observables_validation.json` stored in `transport.rs::SARKAS_D_MKS_REFERENCE`
  with `A2_OMEGA_P` conversion constant and `sarkas_d_star_lookup()` function
- ✅ **Transport grid expanded**: `transport_cases()` now includes 9 Sarkas DSF points
  (κ=1: Γ=14,72,217; κ=2: Γ=31,158,476; κ=3: Γ=503,1510) alongside original 12 → 20 total
- ✅ **`sarkas_validated_cases()` added**: returns 9 κ>0 Sarkas-matched transport configs
- ✅ **`validate_stanton_murillo.rs` extended**: 2→6 transport points; Sarkas D* reference
  displayed for matched cases; cross-case D* ordering checks per-κ and cross-κ
- ✅ **Tolerance constants added**: `DALIGAULT_FIT_VS_CALIBRATION` (20% per-point),
  `DALIGAULT_FIT_RMSE` (10% over 12 points) in `tolerances.rs`
- ✅ **454 tests, 0 clippy warnings, 0 failures**

## Completed (audit, Feb 19 2026)

- ✅ **`validate_transport` added to `validate_all.rs`**: was missing from meta-validator SUITES
- ✅ **37 tolerance constants centralized**: 16 new constants in `tolerances.rs` covering transport parity,
  lattice QCD, NAK eigensolve, PPPM; wired into 7 validation binaries (was ~21 inline magic numbers)
- ✅ **`lattice/constants.rs` created**: centralizes LCG PRNG (MMIX multiplier/increment), `lcg_uniform_f64()`,
  `lcg_gaussian()` Box-Muller, `LATTICE_DIVISION_GUARD`, `N_COLORS`, `N_DIM`; wired into `su3.rs`,
  `hmc.rs`, `dirac.rs`, `cg.rs` — 14 magic-number sites eliminated
- ✅ **Provenance expanded**: `HOTQCD_EOS_PROVENANCE` struct + `HOTQCD_DOI`, `PURE_GAUGE_REFS` for
  lattice QCD validation targets
- ✅ **25 new tests**: `hfb_gpu_types` (7), `celllist` (7), `lattice/constants` (7), tolerance (2),
  provenance (2); total 441 (436 passing + 5 GPU-ignored)
- ✅ **Clippy warnings: 0** (was 3 `uninlined_format_args` in `md/transport.rs`)
- ✅ **`validate_pppm.rs` semantic fix**: multi-particle net force checks now use
  `PPPM_MULTI_PARTICLE_NET_FORCE` instead of `PPPM_NEWTON_3RD_ABS`

## Completed (v0.5.13)

- ✅ **GPU-resident cell-list build**: 3-pass GPU pipeline (bin → exclusive prefix sum → scatter)
  replaces CPU cell-list rebuild. Zero CPU readback for neighbor-list updates.
  - 4 new WGSL shaders: `cell_bin_f64.wgsl`, `exclusive_prefix_sum.wgsl`, `cell_scatter.wgsl`,
    `yukawa_force_celllist_indirect_f64.wgsl`
  - `GpuCellList` struct: compiles 3 pipelines, manages 5 intermediate buffers, single-encoder dispatch
  - Cell-list rebuild reduced from 7 lines (3 readbacks + CPU sort + 5 uploads) to 1 line (`gpu_cl.build()`)
  - Eliminated 1.4 MB of PCIe round-trip data per rebuild at N=10,000

- ✅ **Indirect force shader**: Positions, velocities, and forces stay in original particle order.
  The force shader uses `sorted_indices[cell_start[c] + jj]` for indirect neighbor access.
  VV integrator, KE shader, and thermostat are unchanged (operate on original-order arrays).

- ✅ **VACF particle-identity fix**: The old sorted-array approach scrambled particle ordering
  every 20 steps when the cell-list was rebuilt. Velocity autocorrelation C(t) = ⟨v(t)·v(0)⟩
  requires consistent particle identity across snapshots. The indirect approach preserves
  original particle ordering — VACF is now correct by construction.

- ✅ **ToadStool `CellListGpu` binding mismatch found**: ToadStool's `prefix_sum.wgsl` has a
  4-binding layout (uniform + 3 storage) but `cell_list_gpu.rs` wires a 3-binding BGL
  (2 storage + 1 uniform) — the pipeline would fail to create. hotSpring's local
  `exclusive_prefix_sum.wgsl` matches the intended 3-binding layout. Ready for ToadStool absorption.

- ✅ **Zero regressions**: 279 tests pass, 0 clippy warnings, 0 doc warnings, 0 linter errors.

## Completed (v0.5.12)

- ✅ **ReduceScalarPipeline rewire**: Local `SHADER_SUM_REDUCE` (inline WGSL copy of barracuda's
  `sum_reduce_f64.wgsl`) removed from `md/shaders.rs`. Both `md/simulation.rs` and `md/celllist.rs`
  now use `barracuda::pipeline::ReduceScalarPipeline::sum_f64()` for KE, PE, and thermostat reductions.
  Eliminated ~50 lines of boilerplate per MD path (4 bind groups, 6 buffers, reduce_pipeline).
- ✅ **Error bridge**: `HotSpringError::Barracuda(BarracudaError)` variant + `From` impl enables
  clean `?` propagation from barracuda primitive calls into hotSpring result types.
- ✅ **Zero regressions**: 454 tests pass (449 + 5 GPU-ignored), 0 clippy warnings, 0 doc warnings.

## Completed (v0.5.11)

- ✅ **Barracuda op integration**: Validation binaries now use `barracuda::ops::md::forces::YukawaForceF64`
  directly instead of the local `yukawa_nvk_safe` workaround module (deleted)
- ✅ **Driver-aware shader compilation**: Added `GpuF64::create_pipeline_f64()` which delegates to
  barracuda's `WgpuDevice::compile_shader_f64()` — auto-patches `exp()` on NVK/nouveau
- ✅ **Production MD uses `create_pipeline_f64()`**: Force shaders in `simulation.rs` now compile
  via driver-aware path; VV/berendsen/KE use raw path (no exp/log, safe on all drivers)
- ✅ **Removed `yukawa_nvk_safe.rs`** and `yukawa_force_f64_nvk_safe.wgsl` — barracuda handles NVK
- ✅ **Clippy pedantic+nursery: 0 warnings** in library code (was 167+ cast_precision_loss, 87 mul_add)
- ✅ **Zero doc warnings**: All rustdoc links resolve correctly
- ✅ **Auto-fixed 28 clippy suggestions** across binaries (redundant clone, From, let-else)

## Completed (v0.5.10)

- ✅ **GPU density pipeline**: `batched_hfb_density_f64.wgsl` wired into `hfb_gpu_resident.rs`
  - Shader updated for batched per-nucleus wavefunctions
  - 14 GPU buffers, 3 compute pipelines (`compute_density`, `mix_density`)
  - Full staging readback + CPU density fallback
  - SCF loop restructured: CPU Brent → GPU density + mixing → CPU energy
  - L2 GPU validation confirmed: chi2/datum=5.42, physics consistent
- ✅ **Energy pipeline stubs**: bind groups, buffers, staging for future GPU energy dispatch
- ✅ **rho buffers upgraded**: `COPY_SRC` added for GPU-to-staging density transfer

## Completed (v0.5.8)

- ✅ **WGSL preamble injection**: `abs_f64` (bcs_bisection) and `cbrt_f64` (potentials) replaced
  with `ShaderTemplate::with_math_f64_auto()` — zero duplicate WGSL math
- ✅ **Exhaustive tolerance wiring**: 8 new constants (`FACTORIAL_TOLERANCE`, `ASSOC_LEGENDRE_TOLERANCE`,
  `DIGAMMA_FD_TOLERANCE`, `BETA_VIA_LNGAMMA_TOLERANCE`, `INCOMPLETE_GAMMA_TOLERANCE`,
  `BESSEL_NEAR_ZERO_ABS`, `RHO_POWF_GUARD`, `GPU_JACOBI_CONVERGENCE`)
- ✅ **Core physics tolerance wiring**: `hfb.rs`, `hfb_gpu.rs`, `hfb_gpu_resident.rs` — all
  inline density floors, powf guards, GPU eigensolve thresholds → named constants
- ✅ **Comprehensive audit**: zero unsafe, 8 scoped TODO(B2) markers (GPU-resident migration), zero mocks, zero hardcoded paths,
  all AGPL-3.0 licensed, all validation binaries follow hotSpring pattern

## Completed (v0.5.7)

- ✅ `hermite_value` → delegates to `barracuda::special::hermite` (zero duplicate math)
- ✅ Validation binary tolerances fully wired: `validate_linalg`, `validate_special_functions`,
  `validate_md`, `validate_barracuda_pipeline`, `validate_optimizers` (~50 inline → ~12 niche)
- ✅ `HFB_TEST_NUCLEI_PROVENANCE` — machine-readable `BaselineProvenance` struct
- ✅ L1/L2 provenance environments expanded: NumPy 1.24, SciPy 1.11, mystic 0.4.2
- ✅ 7 new determinism + coverage tests (189 total, 44% line / 61% function coverage)
- ✅ `GpuResidentL2Result` fully documented

## Completed (v0.5.6)

- ✅ `SpinOrbitGpu` wired into `hfb_gpu_resident.rs` with CPU fallback
- ✅ `compute_ls_factor` from barracuda replaces manual `(j(j+1)-l(l+1)-0.75)/2` in `hfb.rs`, `hfb_gpu_resident.rs`
- ✅ Physics guard constants centralized: `DENSITY_FLOOR`, `SPIN_ORBIT_R_MIN`, `COULOMB_R_MIN`
  - 20+ inline `1e-15`, `0.1`, `1e-10` guards replaced across 5 physics modules
- ✅ SPDX headers added to all 17 WGSL shaders that were missing them (30/30 total)
- ✅ `panic!()` in library code converted to `expect()` (GPU buffer map failures)
- ✅ WGSL math duplicates resolved via `ShaderTemplate::with_math_f64_auto()` preamble injection

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
- ✅ `validate_all` → Meta-validator for all 33 validation suites

## Completed (v0.5.3)

- ✅ `bcs_gpu.rs` → ToadStool `target` keyword fix absorbed (commit `0c477306`)
- ✅ `hfb_gpu.rs` → Single-dispatch eigensolve wired
- ✅ `hfb_deformed_gpu.rs` → Single-dispatch with fallback wired
- ✅ MD shaders → 5 large shaders extracted to `.wgsl` files
- ✅ BCS pipeline → Shader compilation cached at construction

## GPU Reduction via ReduceScalarPipeline (v0.5.12, Feb 19 2026)

**Rewired**: Local `SHADER_SUM_REDUCE` (inline WGSL copy) replaced by
`barracuda::pipeline::ReduceScalarPipeline` from ToadStool (Feb 19 2026
absorption). Both `simulation.rs` and `celllist.rs` now call
`reducer.sum_f64(&buffer)` instead of manually managing 4 bind groups,
6 intermediate buffers, and 4 reduce dispatches per energy readback.

| Before (N=10000) | After | Reduction |
|-------------------|-------|-----------|
| KE readback: 80 KB (N×8) | 8 bytes (1 scalar) | 10,000× |
| PE readback: 80 KB (N×8) | 8 bytes (1 scalar) | 10,000× |
| Equil thermostat: 80 KB | 8 bytes | 10,000× |
| Total per dump: 160 KB | 16 bytes | 10,000× |

**Code eliminated**: ~50 lines of boilerplate per MD path (reduce pipeline,
partial buffers, scalar buffers, param buffers, bind groups, inline dispatches).
`SHADER_SUM_REDUCE` removed from `shaders.rs`.

**Validation**: 454 tests pass, 0 clippy warnings, 0 doc warnings.

**Remaining readback**: position/velocity snapshots for VACF and cell-list
rebuilds. Both can be eliminated with GPU-resident VACF and `CellListGpu`.

See `wateringHole/handoffs/HOTSPRING_UNIDIRECTIONAL_FEEDBACK_FEB19_2026.md`
for full design feedback to ToadStool on the unidirectional pattern,
`StatefulPipeline` proposal, and NAK universal solution.

## Multi-GPU Benchmark Results (v0.5.10, Feb 17 2026)

RTX 4070 (nvidia proprietary) vs Titan V (NVK/nouveau, open-source).
See `experiments/006_GPU_FP64_COMPARISON.md` for full analysis.

| Workload | RTX 4070 | Titan V | Ratio | Bottleneck |
|----------|----------|---------|-------|------------|
| BCS bisection (2048 nuclei) | 1.54 ms | 1.72 ms | 1.11× | Dispatch overhead |
| Eigensolve 128×20 | 4.64 ms | 31.11 ms | 6.7× | NVK shader compiler |
| Eigensolve 128×30 | 12.95 ms | 93.33 ms | 7.2× | NVK shader compiler |
| L2 HFB pipeline (18 nuclei) | 8.1 ms | 8.6 ms | 1.06× | CPU-dominated (Amdahl) |

**Key finding**: NVK is functionally correct (identical physics to 1e-15)
but 4–7× slower on compute-bound shaders. This is a driver maturity gap,
not hardware. Proprietary driver on Titan V would unlock its 6.9 TFLOPS fp64.

### NVK Reinstated (Feb 21, 2026)

NVK rebuilt from Mesa 25.1.5 source (`-Dvulkan-drivers=nouveau`) after Pop!_OS
Mesa package was found to omit NVK. Titan V now visible as GPU1 alongside
RTX 4070 (GPU0). Full validation re-confirmed:

- `validate_cpu_gpu_parity`: **6/6 checks passed** (energy, temperature, D* parity)
- `validate_stanton_murillo`: **40/40 checks passed** (6 (κ,Γ) transport cases, 16.5 min)
- `bench_gpu_fp64`: BCS bisection + batched eigensolve — all correct

ToadStool barracuda auto-detects NVK via adapter info and applies Volta
driver profile (`DriverKind::Nvk`, `CompilerKind::Nak`, `GpuArch::Volta`).

## Lattice QCD Modules (Updated Feb 20, 2026 — Post ToadStool Session 25)

| Rust Module | Lines | WGSL | Tier | Status |
|-------------|-------|------|------|--------|
| `lattice/complex_f64.rs` | 257 | `WGSL_COMPLEX64` → `shaders/complex_f64.wgsl` | **✅** | **Absorbed** — toadstool `8fb5d5a0`; extracted v0.6.3 |
| `lattice/su3.rs` | 393 | `WGSL_SU3` → `shaders/su3_f64.wgsl` | **✅** | **Absorbed** — toadstool `8fb5d5a0`; extracted v0.6.3 |
| `lattice/wilson.rs` | 338 | → `wilson_plaquette_f64.wgsl` | **✅** | **Absorbed** — GPU plaquette shader |
| `lattice/hmc.rs` | 350 | → `su3_hmc_force_f64.wgsl` | **✅** | **Absorbed** — GPU HMC force shader |
| `lattice/abelian_higgs.rs` | ~500 | → `higgs_u1_hmc_f64.wgsl` | **✅** | **Absorbed** — toadStool `ops/lattice/higgs_u1` |
| `lattice/dirac.rs` | 440+ | `WGSL_DIRAC_STAGGERED_F64` | **✅** | **Absorbed** — toadStool `ops/lattice/dirac` + `cpu_dirac`. Local retained for hotSpring validation |
| `lattice/cg.rs` | 320+ | `WGSL_COMPLEX_DOT_RE_F64` + `WGSL_AXPY_F64` + `WGSL_XPAY_F64` | **✅** | **Absorbed** — toadStool `ops/lattice/cg` + `gpu_cg_solver` + `gpu_cg_resident`. Local retained |
| `lattice/pseudofermion/` | 812 | CPU pseudofermion HMC | **✅** | **Absorbed** — toadStool `ops/lattice/pseudofermion` + `gpu_pseudofermion`. Local retained |
| `lattice/gpu_hmc/` | ~2800 | GPU streaming HMC + resident CG + brain | **✅** | **Absorbed** — toadStool `ops/lattice/gpu_hmc_trajectory`. hotSpring retains for brain/NPU integration |
| `lattice/eos_tables.rs` | 307 | — | N/A | HotQCD reference data (CPU-only) |
| `lattice/multi_gpu.rs` | 237 | — | **C** | CPU-threaded dispatcher; needs GPU dispatch |

## Spectral Theory Modules (Feb 22, 2026 — Fully Leaning on Upstream)

**All spectral source code deleted from hotSpring.** The `spectral/mod.rs` now
contains only re-exports from `barracuda::spectral` plus a `CsrMatrix` type
alias for backward compatibility. ~41 KB of local source removed.

ToadStool absorbed the entire spectral module in Sessions 25-31h (commit
`dc540afd`..`0bd6a92d`), including Anderson 1D/2D/3D, Lanczos, Hofstadter,
Sturm tridiagonal, level statistics, CSR SpMV, and a new `BatchIprGpu`.

| Upstream Module | hotSpring Status |
|----------------|------------------|
| `barracuda::spectral::anderson` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::SpectralCsrMatrix` | ✅ **Leaning** — aliased as `CsrMatrix` |
| `barracuda::spectral::lanczos` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::hofstadter` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::tridiag` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::stats` | ✅ **Leaning** — re-exported |
| `barracuda::spectral::BatchIprGpu` | ✅ **NEW** — available via re-export |

### HMC Implementation Notes

The HMC leapfrog integrator uses the **Cayley transform** for the SU(3) matrix
exponential: `exp(X) ≈ (I + X/2)(I - X/2)^{-1}`. This is exactly unitary when
X is anti-Hermitian, eliminating unitarity drift that plagues Taylor approximations.
The 3×3 inverse uses cofactor expansion (exact, no iteration).

Gauge force: `dP/dt = -(β/3) Proj_TA(U × V)` where V is the staple sum
(NOT V†). This was debugged from first principles during the Feb 19 audit —
the original sign and adjoint were both wrong, causing 0% HMC acceptance.

### Lattice QCD GPU Promotion Roadmap (Updated Feb 20, 2026)

1. ~~**Complex f64 + SU(3)**~~ ✅ **Absorbed** — toadstool `8fb5d5a0`
2. ~~**Plaquette shader**~~ ✅ **Absorbed** — `wilson_plaquette_f64.wgsl`
3. ~~**HMC force on GPU**~~ ✅ **Absorbed** — `su3_hmc_force_f64.wgsl` + `higgs_u1_hmc_f64.wgsl`
4. ~~**FFT**~~ ✅ **Absorbed** — `Fft1DF64` + `Fft3DF64` (toadstool `1ffe8b1a`)
5. ~~**Dirac apply on GPU**~~ ✅ **Done** — `WGSL_DIRAC_STAGGERED_F64` validated
   8/8 checks (cold/hot/asymmetric lattices, max error 4.44e-16).
6. ~~**GPU CG solver**~~ ✅ **Done** — `WGSL_COMPLEX_DOT_RE_F64` + `WGSL_AXPY_F64`
   + `WGSL_XPAY_F64` validated 9/9 checks. Full CG iteration on GPU:
   D†D + dot + axpy + xpay, only scalar coefficients transfer per iteration.
   **Full GPU lattice QCD pipeline: COMPLETE.**
7. ✅ **Pure GPU workload validated** — `validate_pure_gpu_qcd` (3/3 checks):
   CPU HMC thermalization (10 traj, 100% accepted) → GPU CG on thermalized
   configs (5 solves, 32 iters each, exact iteration match, solution parity
   4.10e-16). Only 24 bytes/iter CPU↔GPU transfer. **Production-like workload: VALIDATED.**
8. ✅ **Python baseline established** — `bench_lattice_cg` + `lattice_cg_control.py`:
   Rust 200× faster than Python on identical CG algorithm. Iterations match exactly
   (5 cold, 37 hot). Dirac apply: 0.023ms (Rust) vs 4.59ms (Python) = 200×.

## Evolution Gaps Identified

| Gap | Impact | Priority | Status |
|-----|--------|----------|--------|
| ~~GPU energy integrands not wired in spherical HFB~~ | ~~CPU bottleneck in SCF energy~~ | ~~High~~ | ✅ Resolved v0.6.2: `batched_hfb_energy_f64.wgsl` wired behind `gpu_energy` feature flag |
| ~~`SumReduceF64` not used for MD energy sums~~ | ~~CPU readback for reduction~~ | ~~High~~ | ✅ Resolved v0.5.12: `ReduceScalarPipeline` (GPU-buffer variant) |
| ~~Lattice QCD GPU shaders~~ | ~~CPU-only lattice modules~~ | ~~Medium~~ | ✅ Absorbed by toadstool S25 (5 GPU shaders) |
| ~~GPU SpMV (CSR)~~ | ~~CPU-only sparse matrix-vector product~~ | ~~**P1**~~ | ✅ **Done** — `WGSL_SPMV_CSR_F64` validated, `validate_gpu_spmv` binary (28th suite) |
| ~~GPU Lanczos~~ | ~~CPU-only iterative eigensolve~~ | ~~**P1**~~ | ✅ **Done** — GPU SpMV Lanczos validated, `validate_gpu_lanczos` (29th suite) |
| ~~GPU Dirac SpMV~~ | ~~CPU-only staggered Dirac operator~~ | ~~**P1**~~ | ✅ **Done** — `WGSL_DIRAC_STAGGERED_F64` validated, `validate_gpu_dirac` binary (30th suite) |
| ~~Pure GPU QCD workload~~ | ~~Thermalized-config CG validation~~ | ~~**P1**~~ | ✅ **Done** — `validate_pure_gpu_qcd` (3/3): HMC → GPU CG, 4.10e-16 parity (31st suite) |
| ~~Python baseline~~ | ~~Interpreted-language benchmark~~ | ~~**P1**~~ | ✅ **Done** — Rust 200× faster: CG iters match exactly, Dirac 0.023ms vs 4.59ms |
| ~~8 files > 1000 lines~~ | ~~Code organization~~ | ~~Medium~~ | ✅ Resolved v0.6.1: 4 monolithic files decomposed into module dirs. Only `hfb_gpu_resident/mod.rs` (1456) remains >1000 (justified — monolithic GPU pipeline) |
| ~~Stanton-Murillo transport normalization~~ | ~~Paper 5 calibration~~ | ~~High~~ | ✅ Resolved: Sarkas-calibrated (12 points, N=2000) |
| ~~BCS + density shader not wired~~ | ~~CPU readback after eigensolve~~ | ~~High~~ | ✅ Resolved v0.5.10 |
| ~~WGSL inline math~~ | ~~Maintenance drift~~ | ~~Medium~~ | ✅ Resolved v0.5.8 |
| ~~Hardcoded tolerances in validation binaries~~ | ~~Not traceable/justified~~ | ~~High~~ | ✅ Resolved: 37 constants in `tolerances.rs` |
| ~~Lattice LCG magic numbers scattered~~ | ~~Maintenance risk~~ | ~~Medium~~ | ✅ Resolved: `lattice/constants.rs` centralizes all |

## Gaps Resolved (v0.5.5)

- ✅ `celllist_diag.rs` inline WGSL → Extracted 8 shaders to `.wgsl` files (1672 → 1124 lines)
- ✅ Dead_code in deformed HFB → 6 field renames, 3 documented GPU-reserved functions
- ✅ Nuclear EOS path duplication → Shared `data::load_eos_context()` replaces 9 inline path constructions
- ✅ Inline tolerances → 30+ magic numbers replaced with `tolerances::` constants
- ✅ Inline `sigma_theo` → 19 instances replaced with `tolerances::sigma_theo()`

## Absorption Note: Why Local Code is Retained (March 2, 2026)

As of toadStool S80, nearly ALL of hotSpring's lattice QCD stack has been absorbed:
- CPU reference: wilson, su3, complex64, dirac, CG, pseudofermion, constants
- GPU ops: plaquette, HMC force, CG solver, pseudofermion heatbath/force, full HMC trajectory
- ESN: MultiHeadEsn + ExportedWeights
- Nautilus: brain, shell, evolution, population, readout, board

**hotSpring retains local copies** for two reasons:

1. **toadStool's CPU lattice types are `#[cfg(test)]` only.** hotSpring uses `Lattice`,
   `Su3Matrix`, `Complex64` in production binaries (cold_start, HMC config, validation).
   Deleting them would require toadStool to make these public.

2. **hotSpring's GPU HMC orchestration is spring-specific.** The `gpu_hmc/` module
   integrates with brain architecture, NPU worker, streaming dispatch, and resident CG
   in ways that are not captured by toadStool's generic `gpu_hmc_trajectory`. The
   production binary `production_dynamical_mixed.rs` depends on ~15 hotSpring-specific
   types (`GpuHmcState`, `GpuDynHmcStreamingPipelines`, `BrainInterrupt`, etc.).

**Lean path**: When toadStool makes CPU lattice types public (not `#[cfg(test)]`),
hotSpring can re-export them (like spectral did). When toadStool's GPU HMC trajectory
gains brain/NPU hooks, hotSpring can migrate its orchestration.

## Exp 032: NVK Dynamical HMC Validation (March 2, 2026)

### RTX 3090 (NVK/NAK GA102) — Full Pipeline Validated

1. **4^4 dynamical HMC**: 3 trajectories completed, 100% acceptance, ⟨P⟩=0.497
2. **8^4 dynamical HMC**: Running (n_md=50, dt=0.01, trajectory length 0.5)
3. **Sovereign compilation path**: f64 shaders via toadStool's naga→SPIR-V passthrough
4. **CG-precise compilation**: Dirac/dot/axpy/xpay use WGSL-text path (no FMA fusion)

### Latency-Adaptive CG (new)

Brain-mode CG auto-detects NVK readback latency (>5ms threshold) and scales
check_interval to maintain 5× compute-to-readback ratio. Reduces readback count
from ~500 to ~3-5 per CG solve on NVK.

### Dispatch Coalescing (new)

Pre-CG (gauge force + mom_update) and post-CG (Dirac + fermion_force + fermion_mom)
batched into single encoder submissions. Saves 180 vkQueueSubmit calls per trajectory.

### NVK Performance (RTX 3090)

- Readback: 10-50ms (50× slower than proprietary)
- Submit overhead: ~2ms per vkQueueSubmit (20× slower)
- 4^4 trajectory: ~42s (vs ~2s est. proprietary)
- 8^4 trajectory: ~140s (dt=0.0125, n_md=40)
- Full handoff: `experiments/NVK_F64_DISPATCH_HANDOFF.md`

## S80 Sync Validation (March 2, 2026)

- 660 tests passing (was 658 before S80 sync)

## v0.6.18 Refactoring (March 6, 2026)

- 724 tests (lib), 95 binaries, 71 WGSL shaders (all AGPL-3.0-only)
- npu_worker → 6 modules, simulation → 4 modules, dynamical_mixed → library module
- esn_baseline extracted to library (ready for absorption), sarkas → library module
- brain B2/D1 evolved from placeholder to real implementations
- New test: `exported_weights_serde_compatible_with_toadstool` — bidirectional JSON round-trip
- New test: `head_group_layout_matches_toadstool_head_group` — 6×6 head mapping confirmed
- `spectral_bandwidth` and `spectral_condition_number` wired into `proxy.rs` Anderson proxy
- `ProxyFeatures` now carries `condition_number` field (condition number κ = max|λ|/min|λ|)
- Cross-spring benchmark updated to S80: spectral stats, neighbor precompute, Nelder-Mead GPU
- NeighborMode 4D tested: hotSpring and toadStool both produce correct inverse-consistent tables

## Evolution v0.6.9 → v0.6.13 (Feb 24-25, 2026)

| Version | Key Change | Status |
|---------|-----------|--------|
| v0.6.9 | toadStool S62 sync. Spectral lean (41 KB deleted). CsrMatrix alias | ✅ Fully leaning |
| v0.6.10 | DF64 gauge force on RTX 3090. 9.9× FP32 core throughput | ✅ Production |
| v0.6.11 | t-major site indexing standardization. 119/119 unit tests | ✅ Convention adopted |
| v0.6.12 | toadStool S60 DF64 expansion (plaquette, KE, transcendentals). 60% HMC DF64 | ✅ Absorbed |
| v0.6.13 | GPU Polyakov loop (72× less transfer). NVK guard. su3_math_f64. PRNG fix | ✅ 13/13 checks |

### New Shaders (v0.6.10-v0.6.13)

| Shader | Version | Origin | Absorption Status |
|--------|---------|--------|-------------------|
| `su3_gauge_force_df64.wgsl` | v0.6.10 | hotSpring | Local (DF64 neighbor-buffer variant) |
| `wilson_plaquette_df64.wgsl` | v0.6.12 | toadStool S60 | Bidirectional |
| `su3_kinetic_energy_df64.wgsl` | v0.6.12 | toadStool S60 | Bidirectional |
| `polyakov_loop_f64.wgsl` | v0.6.13 | toadStool → hotSpring | Bidirectional (pending upstream sync) |
| `su3_math_f64.wgsl` | v0.6.13 | hotSpring | Local (pending upstream absorption) |

### Key Discoveries

- **DF64 core streaming** (v0.6.10): FP32 cores deliver 3.24 TFLOPS at 14-digit precision on consumer GPUs. 9.9× native f64 throughput.
- **Naga composition bug** (v0.6.13): `naga` rejects WGSL with unused `ptr<storage>` functions when prepended as preamble. Workaround: split into `su3_math_f64.wgsl`.
- **PRNG type mismatch** (v0.6.13): `ShaderTemplate` patches `cos`→`cos_f64`, but `f32(theta)` broke the call. Fix: keep theta as `f64` throughout.
