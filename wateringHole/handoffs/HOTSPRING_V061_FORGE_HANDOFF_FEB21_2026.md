# hotSpring v0.6.1 → ToadStool/BarraCUDA: Forge & Evolution Handoff

**Date:** 2026-02-21
**From:** hotSpring (computational physics biome)
**To:** ToadStool / BarraCUDA core team
**License:** AGPL-3.0-only

---

## Context: The Biome Model

hotSpring is a biome. ToadStool/barracuda is the fungus. The fungus lives
in every biome (hotSpring, neuralSpring, desertSpring). Each biome leans on
toadstool for what it already provides, evolves shaders and systems locally,
and toadstool absorbs what works. Springs don't reference each other — they
learn from each other by reviewing code in `ecoPrimals/`, not by importing.

This handoff covers:
1. New local infrastructure (metalForge forge) ready for review
2. Absorption-ready shaders and systems
3. NPU substrate support (new capability for toadstool)
4. Lessons learned from the Write → Absorb → Lean cycle

---

## 1. metalForge Forge: Local Hardware Discovery

**Location:** `hotSpring/metalForge/forge/`
**Status:** 16 tests, zero clippy warnings, zero fmt issues
**Deps:** `barracuda` (from toadstool), `wgpu` 22, `tokio` 1

### What It Does

The forge crate discovers compute substrates on the local machine using
wgpu adapter enumeration (the same path barracuda uses for GPU discovery),
plus local probing for NPU (`/dev/akida0`) and CPU (`/proc/cpuinfo`).

```
wgpu::Instance::enumerate_adapters() → GPU substrates with SHADER_F64, timestamps, driver info
/proc/cpuinfo → CPU model, core count, thread count, cache, AVX2 flag
/dev/akida0 → BrainChip AKD1000 NPU (quantized inference, weight mutation)
```

### What This Machine Discovers

| # | Kind | Name | Capabilities |
|---|------|------|-------------|
| 0 | CPU | i9-12900K (16C/24T, AVX2) | f64, f32, spmv, eigen, cg, simd |
| 1 | GPU | RTX 4070 (NVIDIA 580.82.09, Vulkan, SHADER_F64) | f64, f32, shader, reduce, spmv, eigen, cg, timestamps |
| 2 | GPU | Titan V (NVK/Mesa 25.1.5, Vulkan, SHADER_F64) | f64, f32, shader, reduce, spmv, eigen, cg, timestamps |
| 3 | NPU | BrainChip AKD1000 (/dev/akida0) | f32, quant(8), quant(4), batch(8), weight-mut |

### Capability-Based Dispatch

The forge routes workloads to substrates by capability, not by name:
- "Who can do f64 + reduce?" → RTX 4070 (GPU)
- "Who can do quantized inference?" → AKD1000 (NPU)
- "Who can do f64 + simd?" → i9-12900K (CPU)

### Architecture

| Module | Lines | Purpose |
|--------|-------|---------|
| `substrate.rs` | ~212 | Substrate, Identity, Properties, Capability types |
| `probe.rs` | ~207 | wgpu GPU probe, procfs CPU probe, /dev NPU probe |
| `inventory.rs` | ~99 | Assemble all probes into unified inventory |
| `dispatch.rs` | ~189 | Route workloads to best capable substrate |
| `bridge.rs` | ~140 | **Forge↔barracuda device bridge** (absorption seam) |

### Bridge Module (New v0.6.1)

The `bridge.rs` module is the explicit absorption seam. It connects
forge substrates to barracuda's device creation API:

| Function | Direction | What it does |
|----------|-----------|--------------|
| `create_device()` | Forge → barracuda | Creates `WgpuDevice` from forge substrate via `from_adapter_index()` |
| `best_f64_gpu()` | Forge scan | Returns best f64-capable GPU substrate from inventory |
| `substrate_from_device()` | barracuda → Forge | Wraps existing `WgpuDevice` as a forge substrate |

This means code that uses barracuda for GPU compute can now also
participate in capability-based dispatch. The bridge makes the
absorption path explicit — toadstool can see exactly how the pieces
connect.

### Absorption Opportunity

The bridge module, NPU probing logic (`probe_npus()`), CPU probing
(`probe_cpu()`), and the `Capability` enum (12 variants including
QuantizedInference, BatchInference, WeightMutation) are the contributions.

toadstool's `barracuda::device::substrate` already has `SubstrateType::Npu`,
but no actual NPU discovery logic or CPU substrate support. The forge
provides working implementations for both plus the capability-based
dispatch that toadstool's `HardwareWorkload` enum could evolve toward.

| Forge Module | Toadstool Target | Gap |
|-------------|------------------|-----|
| `substrate::Capability` | `device::unified::Capability` | Forge has 12 variants vs toadstool's 4 |
| `probe::probe_cpu()` | `substrate::Substrate::discover_all()` | Toadstool has no CPU substrate |
| `probe::probe_npus()` | `device::akida` | Forge `/dev` check complements PCIe scan |
| `dispatch::route()` | `toadstool_integration::select_best_device()` | Capability sets vs workload enum |
| `bridge::create_device()` | Already delegates to `WgpuDevice::from_adapter_index()` | Direct mapping |

---

## 2. Shaders Ready for Absorption

These WGSL shaders are validated, documented, and self-contained:

### Tier 1 — GPU acceleration benefit, high priority

| Shader | Location | Tests | Binding layout |
|--------|----------|-------|---------------|
| `WGSL_SPMV_CSR_F64` | `spectral/csr.rs` (inline) | 8/8 GPU checks, parity 1.78e-15 | `@group(0)@binding(0-5)` — row_ptr, col_idx, values, x, y, params |
| `WGSL_DIRAC_STAGGERED_F64` | `lattice/dirac.rs` (inline) | 8/8 GPU checks, parity 4.44e-16 | `@group(0)@binding(0-3)` — gauge_links, input, output, params |
| `WGSL_COMPLEX_DOT_RE_F64` | `lattice/cg.rs` (inline) | Part of 9/9 CG suite | `@group(0)@binding(0-2)` — vec_a, vec_b, result |
| `WGSL_AXPY_F64` | `lattice/cg.rs` (inline) | Part of 9/9 CG suite | `@group(0)@binding(0-3)` — alpha, x, y, result |
| `WGSL_XPAY_F64` | `lattice/cg.rs` (inline) | Part of 9/9 CG suite | `@group(0)@binding(0-3)` — alpha, x, y, result |
| `esn_reservoir_update.wgsl` | `md/shaders/` | 16+ ESN tests + NPU pipeline | `@group(0)@binding(0-4)` — W_in, W, state, input, output |
| `esn_readout.wgsl` | `md/shaders/` | 16+ ESN tests | `@group(0)@binding(0-2)` — W_out, state, output |

### Tier 2 — CPU physics ready for upstream library

| Module | Location | Tests | What it provides |
|--------|----------|-------|-----------------|
| Anderson localization 1D/2D/3D | `spectral/anderson.rs` | 31 checks | Disorder models, Lyapunov exponent, mobility edge |
| Hofstadter butterfly | `spectral/hofstadter.rs` | 10 checks | Almost-Mathieu operator, fractal band counting |
| Screened Coulomb | `physics/screened_coulomb.rs` | 23 checks | Sturm bisection eigensolve (2274× faster than Python) |
| HotQCD EOS tables | `lattice/eos_tables.rs` | Thermo checks | Bazavov 2014 reference data with interpolation |

---

## 3. NPU Substrate: What We Learned

This is the most novel contribution — toadstool doesn't have NPU support yet.
hotSpring validated the entire physics pipeline on BrainChip AKD1000 hardware.

### Hardware Facts (measured, not from SDK)

| Property | SDK says | We measured |
|----------|----------|------------|
| Input channels | 1 or 3 | Any count (tested 1-64) |
| FC chain | Independent layers | Single HW pass (SkipDMA), 6.7% overhead for 7 layers |
| Batch size | 1 | 8 → 2.35× throughput (427μs/sample) |
| FC width | ~hundreds | 8192+ neurons, all map to HW |
| Weight update | Not supported | `set_variable()` works — exact linearity (error=0) |
| Power | "30mW" | Board floor 918mW, chip inference below measurement threshold |
| Memory | 8MB SRAM | PCIe BAR1 exposes 16GB address space |
| Program format | Opaque | FlatBuffer (program_info + program_data) |

### Physics Pipeline Validated End-to-End

```
GPU (MD simulation) → ESN training (CPU/GPU) → quantize (f64→int8) → NPU inference
```

| Metric | Value |
|--------|-------|
| NPU throughput | 2,469 inf/s streaming |
| NPU energy | 8,796× less than CPU Green-Kubo |
| Phase detection | β_c = 5.715 (known 5.692, error 0.4%) |
| HMC monitoring overhead | 0.09% (9μs per prediction vs 10.3ms trajectory) |
| Compute savings (steering) | 62% fewer evaluations |
| Cross-substrate parity | f64→f32 error 5.1e-7, int8 error <5% |

### NPU Validation Suites

| Suite | Checks | Status |
|-------|--------|--------|
| Python HW (npu_beyond_sdk.py) | 13/13 | All pass on AKD1000 |
| Rust math (validate_npu_beyond_sdk) | 16/16 | All pass |
| Quantization (npu_quantization_parity.py) | 4/4 | f32/int8/int4/act4 |
| Rust quantization (validate_npu_quantization) | 6/6 | All pass |
| Physics pipeline (npu_physics_pipeline.py) | 10/10 | MD→ESN→NPU→D*,η*,λ* |
| Rust pipeline (validate_npu_pipeline) | 10/10 | All pass |
| Lattice NPU (validate_lattice_npu) | 10/10 | Phase classification |
| Hetero monitor (validate_hetero_monitor) | 9/9 | Real-time steering |

### What toadstool needs to absorb NPU

1. **Substrate discovery**: Probe `/dev/akida*` device nodes (working code in
   `metalForge/forge/src/probe.rs::probe_npus()`)
2. **Capability model**: QuantizedInference, BatchInference, WeightMutation
   (working types in `metalForge/forge/src/substrate.rs`)
3. **Device abstraction**: Open device node, load FlatBuffer program, run inference.
   The Python SDK (`akida` 2.19.1) works today. A Rust driver is the goal.
   See `metalForge/npu/akida/BEYOND_SDK.md` for the full hardware analysis.
4. **Quantization pipeline**: f64→f32→int8→int4 cascade validated. The math is
   in `barracuda/src/md/reservoir.rs` (ESN quantization).

---

## 4. What's Already Absorbed (Lean Phase)

For reference, these hotSpring contributions are already upstream:

| Component | Toadstool Commit | hotSpring Status |
|-----------|-----------------|------------------|
| Complex f64 WGSL | `8fb5d5a0` | Leaning on upstream |
| SU(3) WGSL | `8fb5d5a0` | Leaning on upstream |
| Wilson plaquette | `8fb5d5a0` | Leaning on upstream |
| SU(3) HMC force | `8fb5d5a0` | Leaning on upstream |
| Abelian Higgs HMC | `8fb5d5a0` | Leaning on upstream |
| CellListGpu fix | `8fb5d5a0` | Local deprecated |
| NAK eigensolve | `82f953c8` | Leaning on upstream |
| ReduceScalarPipeline | v0.5.16 | Leaning on upstream |
| GpuDriverProfile | v0.5.15 | Leaning on upstream |
| FFT f64 | `1ffe8b1a` | Leaning on upstream |

---

## 5. Lessons Learned (for toadstool evolution)

### Architecture patterns that worked

1. **Inline WGSL in Rust source**: Embedding shader strings as `const` in the
   module that uses them. Makes the shader + CPU reference + test suite a single
   unit. All 7 absorption-ready shaders follow this pattern.

2. **Tolerance module tree**: 146 constants in `tolerances/{core,md,physics,lattice,npu}.rs`
   with physical justification comments. Zero inline magic numbers in validation
   code. This pattern is worth absorbing — a `barracuda::tolerances` module
   would serve all springs.

3. **Provenance records**: `BaselineProvenance` and `AnalyticalProvenance` structs
   trace every validation constant to its origin (Python script, git commit, DOI).
   Reproducibility demands this.

4. **Capability-based discovery**: Both GPU adapter selection and data path
   resolution use runtime discovery, not hardcoded paths. The forge extends
   this to cross-substrate dispatch.

### Things we got wrong and fixed

1. **Software f64 emulation**: We spent weeks with `math_f64.wgsl` (hundreds of
   lines of f32-pair arithmetic). The real fp64:fp32 ratio on consumer GPUs via
   Vulkan is ~1:2, not 1:64. Once we switched to native WGSL f64 builtins,
   throughput increased 2-6×. Lesson: always test the hardware path first.

2. **Cell-list i32 modulo bug**: WGSL `%` on negative `i32` returns negative
   remainder (C semantics, not Euclidean). Diagnosis took days. The fix was
   `((x % n) + n) % n`. This is documented in `experiments/002_CELLLIST_FORCE_DIAGNOSTIC.md`.
   toadstool absorbed the fix in CellListGpu.

3. **NAK compiler workarounds**: NVK's NAK compiler has bugs with `exp()` and `log()`
   on Volta. toadstool absorbed polynomial workarounds in `GpuDriverProfile`.
   Both RTX 4070 (nvidia proprietary) and Titan V (NVK/nouveau) now produce
   identical physics to 1e-15.

4. **`.expect()` in library code**: We started with ~10 `.expect()` calls in
   production. Crate-level `#![deny(clippy::expect_used)]` forced us to convert
   all to `Result` propagation. Zero panics in library code is worth enforcing
   ecosystem-wide.

### Performance findings relevant to toadstool

| Finding | Value | Source |
|---------|-------|--------|
| Native fp64:fp32 ratio (Vulkan) | ~1:2 (not 1:64) | RTX 4070, Titan V |
| GPU CG solver vs Python | 200× faster | Identical iterations, identical seeds |
| GPU 16⁴ lattice vs CPU | 22.2× faster (24ms vs 533ms) | bench_lattice_scaling |
| Cell-list vs all-pairs | 4.1× at N=10,000, κ=2,3 | Paper-parity long run |
| NPU inference throughput | 2,469 inf/s | Streaming, batch=8 |
| NPU vs CPU energy | 8,796× less | Green-Kubo equivalent |
| Rust vs Python (screened Coulomb) | 2,274× faster | Sturm bisection |
| Rust vs Python (Abelian Higgs) | 143× faster | U(1)+Higgs HMC |
| Unidirectional GPU reduce | 10,000× bandwidth reduction at N=10k | Sum reduction eliminates per-particle readback |

---

## 6. Codebase Health Summary

| Metric | Value |
|--------|-------|
| Unit tests | **505** pass, 5 GPU-ignored |
| Integration tests | **24** pass (3 suites: physics, data, transport) |
| Validation suites | **33/33** pass |
| `expect()`/`unwrap()` in library | **0** (crate-level deny) |
| Clippy warnings (pedantic + nursery) | **0** |
| Doc warnings | **0** |
| Unsafe blocks | **0** |
| External FFI/C bindings | **0** (all pure Rust except wgpu GPU driver bridge) |
| Centralized tolerances | **154** constants (including 8 new solver config) |
| Hardcoded solver params in library | **0** (all centralized in `tolerances/`) |
| Files over 1000 LOC | **1** (`hfb_gpu_resident/mod.rs` at 1456 — monolithic GPU pipeline) |
| AGPL-3.0 compliance | All `.rs` and `.wgsl` files |
| metalForge forge tests | **13** pass |

---

## 6b. Structural Evolution (v0.6.1)

### Large File Refactoring

All monolithic files from v0.6.0 that exceeded 1000 LOC have been decomposed
into module directories with smart domain boundaries:

| Original File | Original LOC | Result | Largest Module |
|---|---|---|---|
| `physics/hfb.rs` | 1,408 | `physics/hfb/{mod,potentials,tests}.rs` | 856 |
| `physics/hfb_deformed.rs` | 1,254 | `physics/hfb_deformed/{mod,potentials,basis,tests}.rs` | 441 |
| `physics/hfb_deformed_gpu.rs` | 1,218 | `physics/hfb_deformed_gpu/{mod,types,physics,gpu_diag,tests}.rs` | 467 |
| `physics/hfb_gpu_resident.rs` | 1,775 | `physics/hfb_gpu_resident/{mod,types,tests}.rs` | 1,456 |

`hfb_gpu_resident/mod.rs` remains at 1,456 lines because the GPU pipeline
(shader setup → buffer allocation → SCF dispatch → eigensolve → readback)
is a single coherent unit where splitting would break GPU resource lifetimes.
Types and physics helpers have been extracted to keep the pipeline function
self-contained.

### Clippy Configuration

Workspace-level `[workspace.lints.clippy]` enables pedantic + nursery with
physics-justified allows (`cast_precision_loss`, `similar_names`, etc.).
Both `barracuda` and `metalForge/forge` compile with zero warnings.

### Solver Configuration Centralization

All hardcoded solver tuning parameters have been extracted to `tolerances/`:

| Constant | Value | Used By | Rationale |
|----------|-------|---------|-----------|
| `HFB_MAX_ITER` | 200 | All HFB solvers | Sufficient for Z=8..120 chart |
| `BROYDEN_WARMUP` | 50 | Deformed HFB | Stabilize density before quasi-Newton |
| `BROYDEN_HISTORY` | 8 | Deformed HFB | Memory vs convergence balance |
| `HFB_L2_MIXING` | 0.3 | Spherical HFB | Conservative linear mixing |
| `HFB_L2_TOLERANCE` | 0.05 MeV | Spherical HFB | ~0.005% relative for A~100 |
| `FERMI_SEARCH_MARGIN` | 50.0 MeV | BCS bisection | Bracket loosely-bound systems |
| `CELLLIST_REBUILD_INTERVAL` | 20 | MD GPU | Rebuild vs accuracy tradeoff |
| `THERMOSTAT_INTERVAL` | 10 | MD equilibration | Smooth Berendsen coupling |

All 8 constants have sanity-check unit tests verifying physical reasonableness.

### Test Coverage Improvements

| Module | Before | After | New Tests |
|--------|--------|-------|-----------|
| `physics/hfb_deformed_common.rs` | 32.88% | **100%** | 13 unit tests |
| `physics/hfb_deformed/` | 0% (new) | 44% basis, 100% tests | 12 unit tests |
| `md/celllist.rs` (CPU paths) | ~80% | ~90% | 5 edge-case tests |
| `md/observables/ssf.rs` | 73.18% | ~90% | 4 edge-case tests |
| `bench.rs` | ~60% | ~80% | 3 BenchReport tests (JSON round-trip) |
| `md/reservoir.rs` (ESN) | ~70% | ~85% | 3 tests (velocity values, NPU state) |
| `physics/hfb_common.rs` | ~85% | ~95% | 2 Coulomb exchange density tests |
| `tolerances/mod.rs` | existing | +2 | solver_config + md_config sanity |
| Integration tests | none | 24 | 3 new test suites |

### Dependency & Safety Audit

- All dependencies are pure Rust (wgpu is Rust API over GPU drivers — unavoidable)
- Zero `unsafe` blocks in all library code
- Zero FFI (`extern "C"` / `extern "system"`) declarations
- Platform-specific paths (`/proc/`, `/sys/`, `/dev/`) degrade gracefully on non-Linux

### GpuCellList Migration Status

The deprecated local `GpuCellList` is documented with a clear migration path
to upstream `barracuda::ops::md::neighbor::CellListGpu` (fixed in toadstool
commit `8fb5d5a0`). Migration deferred to GPU-validated session to avoid
risk to the MD simulation pipeline.

---

## 7. Recommended Absorption Sequence

1. **CSR SpMV + Dirac + CG shaders** → These form the GPU lattice QCD pipeline.
   All tested, all follow the inline-WGSL pattern. Absorb as
   `barracuda::ops::lattice::{spmv, dirac, cg}`.

2. **ESN reservoir shaders** → GPU training + readout for transport/phase
   prediction. Absorb as `barracuda::ops::ml::esn`.

3. **Forge bridge + substrate discovery** → The `bridge.rs` module shows
   exactly how forge substrates connect to `WgpuDevice::from_adapter_index()`.
   Add CPU+NPU probing to `barracuda::device::substrate::Substrate::discover_all()`.
   The forge `Capability` enum (12 variants) can extend toadstool's 4.

4. **Tolerance module pattern** → Consider a `barracuda::tolerances` module
   for upstream validation constants. The module tree structure (core, md,
   physics, lattice, npu) scales cleanly. The 8 solver config constants
   demonstrate the pattern: physically justified defaults with doc comments.

---

## Files Changed Since Last Handoff

```
barracuda/src/lib.rs                     — crate-level deny(expect_used, unwrap_used)
barracuda/src/tolerances/                — NEW module tree (5 submodules, 146 constants)
barracuda/src/physics/hfb_common.rs      — shared initial_wood_saxon_density()
barracuda/src/physics/hfb_deformed_common.rs — NEW shared deformation physics (100% coverage)
barracuda/src/physics/hfb/               — REFACTORED from hfb.rs (1408 → 3 files)
barracuda/src/physics/hfb_deformed/      — REFACTORED from hfb_deformed.rs (1254 → 4 files)
barracuda/src/physics/hfb_deformed_gpu/  — REFACTORED from hfb_deformed_gpu.rs (1218 → 5 files)
barracuda/src/physics/hfb_gpu_resident/  — REFACTORED from hfb_gpu_resident.rs (1775 → 3 files)
barracuda/tests/integration_physics.rs   — NEW (8 tests)
barracuda/tests/integration_data.rs      — NEW (11 tests)
barracuda/tests/integration_transport.rs — NEW (5 tests)
barracuda/src/md/celllist.rs             — 5 new edge-case tests + centralized rebuild/thermostat config
barracuda/src/md/reservoir.rs            — 3 new tests (velocity features, NPU state consistency)
barracuda/src/md/observables/ssf.rs      — 4 new edge-case tests
barracuda/src/bench.rs                   — 3 new tests (BenchReport JSON round-trip, serialize)
barracuda/src/physics/hfb_common.rs      — 2 new tests (Coulomb exchange energy density)
barracuda/src/physics/hfb/mod.rs         — centralized solver config (HFB_MAX_ITER, HFB_L2_TOLERANCE, etc.)
barracuda/src/physics/hfb_deformed/mod.rs — centralized solver config
barracuda/src/physics/hfb_deformed_gpu/  — centralized solver config across mod/physics/gpu_diag
barracuda/src/bin/bench_gpu_fp64.rs      — use centralized HFB_L2_TOLERANCE, HFB_L2_MIXING
barracuda/src/tolerances/physics.rs      — NEW: 6 solver config constants (HFB_MAX_ITER, BROYDEN_*, FERMI_*)
barracuda/src/tolerances/md.rs           — NEW: 2 MD config constants (CELLLIST_REBUILD_INTERVAL, THERMOSTAT_INTERVAL)
barracuda/src/tolerances/mod.rs          — re-export new constants, 2 new sanity tests
barracuda/src/provenance.rs              — corrected paths, extended provenance test
barracuda/Cargo.toml                     — pedantic + nursery lint configuration
metalForge/forge/Cargo.toml              — pedantic + nursery lint configuration
barracuda/ABSORPTION_MANIFEST.md         — NEW comprehensive inventory
barracuda/EVOLUTION_READINESS.md         — updated absorption status
metalForge/forge/                        — Rust crate (16 tests, 5 modules including bridge)
metalForge/forge/src/bridge.rs           — NEW: forge↔barracuda device bridge (absorption seam)
metalForge/forge/Cargo.toml              — added tokio dependency for device creation
metalForge/README.md                     — forge docs, bridge, absorption mapping table
whitePaper/README.md                     — evolution architecture, codebase health updated
README.md                               — test counts (505), tolerance counts (154), directory tree
barracuda/ABSORPTION_MANIFEST.md         — bridge module, absorption mapping table
barracuda/EVOLUTION_READINESS.md         — v0.6.1 structural evolution section
```
