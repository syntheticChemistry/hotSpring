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
**Status:** 13 tests, zero clippy warnings, zero fmt issues
**Deps:** `barracuda` (from toadstool), `wgpu` 22

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
| `substrate.rs` | ~170 | Substrate, Identity, Properties, Capability types |
| `probe.rs` | ~180 | wgpu GPU probe, procfs CPU probe, /dev NPU probe |
| `inventory.rs` | ~70 | Assemble all probes into unified inventory |
| `dispatch.rs` | ~100 | Route workloads to best capable substrate |

### Absorption Opportunity

The NPU probing logic (`probe_npus()`) and the `Capability` enum
extensions (QuantizedInference, BatchInference, WeightMutation) are
the new contributions. toadstool's `barracuda::device::substrate` already
has `SubstrateType::Npu`, but no actual NPU discovery logic. The forge
provides a working implementation that probes `/dev/akida*` device nodes.

The dispatch logic (`dispatch::route()`) complements
`barracuda::device::toadstool_integration::select_best_device()` by adding
NPU-aware routing. The priority model (GPU > NPU > CPU) with capability
matching is a pattern toadstool could absorb.

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
| Unit tests | **463** pass, 5 GPU-ignored |
| Validation suites | **33/33** pass |
| `expect()`/`unwrap()` in library | **0** (crate-level deny) |
| Clippy warnings | **0** |
| Doc warnings | **0** |
| Unsafe blocks | **0** |
| Centralized tolerances | **146** constants |
| AGPL-3.0 compliance | All `.rs` and `.wgsl` files |
| metalForge forge tests | **13** pass |

---

## 7. Recommended Absorption Sequence

1. **CSR SpMV + Dirac + CG shaders** → These form the GPU lattice QCD pipeline.
   All tested, all follow the inline-WGSL pattern. Absorb as
   `barracuda::ops::lattice::{spmv, dirac, cg}`.

2. **ESN reservoir shaders** → GPU training + readout for transport/phase
   prediction. Absorb as `barracuda::ops::ml::esn`.

3. **NPU substrate discovery** → Add `/dev/akida*` probing to
   `barracuda::device::substrate::Substrate::discover_all()`. The forge
   code is a working reference.

4. **Tolerance module pattern** → Consider a `barracuda::tolerances` module
   for upstream validation constants. The module tree structure (core, md,
   physics, lattice, npu) scales cleanly.

---

## Files Changed Since Last Handoff

```
barracuda/src/lib.rs                     — crate-level deny(expect_used, unwrap_used)
barracuda/src/tolerances/                — NEW module tree (5 submodules, 146 constants)
barracuda/src/physics/hfb_common.rs      — shared initial_wood_saxon_density()
barracuda/src/physics/hfb_deformed_common.rs — NEW shared deformation physics
barracuda/ABSORPTION_MANIFEST.md         — NEW comprehensive inventory
barracuda/EVOLUTION_READINESS.md         — updated absorption status
metalForge/forge/                        — NEW Rust crate (13 tests, 4 modules)
metalForge/README.md                     — forge docs, biome model
whitePaper/README.md                     — evolution architecture section
README.md                               — biome model, forge in directory tree
```
