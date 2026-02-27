# toadStool / barracuda — v0.6.14 Evolution & Absorption Handoff

**Date:** February 25, 2026
**From:** hotSpring (biomeGate)
**To:** toadStool / barracuda core team
**Covers:** v0.6.14 debt reduction + barracuda usage review + absorption roadmap
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring v0.6.14 completes a full-crate debt reduction audit. The crate is now
at its cleanest state: 0 clippy warnings, 0 TODOs, 0 mocks in production, 0
unsafe blocks, and 0 hardcoded cross-primal references. This handoff documents:

1. **How hotSpring uses barracuda** — what we lean on, what's local, and why
2. **What to absorb next** — 3 shaders + 2 patterns ready for upstream
3. **Evolution discoveries** — DF64 strategy, PRNG composition, NVK patterns
4. **Paper validation controls** — confirming open data/systems for 22 papers
5. **Hardware validation matrix** — CPU, GPU, and metalForge mixed-substrate results

---

## Part 1: How hotSpring Uses BarraCUDA

### Upstream Dependencies (lean — we use barracuda for this)

| barracuda Module | hotSpring Usage | Validation |
|------------------|-----------------|------------|
| `ops::linalg::lu_solve` | ESN reservoir weight solving (replaced local Gauss-Jordan in v0.6.14) | 664 tests pass |
| `ops::linalg::BatchedEighGpu` | Batched eigensolve for L2/L3 HFB nuclear structure | 16/16 HFB checks |
| `ops::md::CellListGpu` | GPU cell-list spatial decomposition for Yukawa MD | 9/9 PP Yukawa |
| `ops::md::observables::SsfGpu` | Static structure factor on GPU | Transport validation |
| `ops::md::electrostatics::PppmGpu` | PPPM κ=0 Coulomb | PPPM validation pass |
| `spectral::*` | Anderson, Lanczos, Hofstadter (full re-export) | 45/45 spectral checks |
| `device::WgpuDevice` | GPU adapter selection, shader compilation | All GPU binaries |
| `pipeline::ReduceScalarPipeline` | GPU reduction for MD/QCD observables | All GPU validation |
| `shaders::precision::ShaderTemplate` | f64 transcendental patching | All WGSL compilation |
| `linalg::eigh_f64` | CPU eigensolve for HFB spherical/deformed | L2/L3 validation |
| `numerical::{trapz, gradient_1d}` | Numerical integration, gradient stencils | HFB potentials |
| `special::{gamma, laguerre, hermite}` | Special functions for nuclear physics | Physics validation |
| `optimize::{brent, bisect}` | Root-finding for BCS, screened Coulomb | 23/23 screened Coulomb |
| `sample::*`, `surrogate::*` | Sampling, optimization for nuclear EOS | L1/L2 EOS validation |

### Local Implementation (hotSpring owns the physics, barracuda owns the compute)

| hotSpring Module | Why Local | Absorption Candidate? |
|------------------|-----------|:---------------------:|
| `lattice/` (8 modules + gpu_hmc/) | Lattice QCD is hotSpring's domain physics | Shaders: yes. Solvers: after generalization |
| `physics/` (HFB, SEMF, nuclear matter) | Nuclear structure is hotSpring's domain physics | Shaders: yes. Physics: stays local |
| `md/` (simulation, transport, reservoir) | Yukawa OCP MD orchestration | Mostly leaning already. ESN shaders: yes |
| `tolerances/`, `provenance/`, `validation/` | hotSpring-specific validation infrastructure | No — stays local by design |
| `discovery/`, `data/`, `error/` | hotSpring-specific data paths and error types | No — primal self-knowledge |

### Evolution Principle

> hotSpring owns domain physics. barracuda owns compute primitives.
> Local code that becomes a reusable compute primitive gets absorbed.
> Local code that encodes hotSpring-specific physics stays local.

---

## Part 2: What to Absorb

### Immediate (validated, zero risk)

#### 2a. `prng_pcg_f64.wgsl` — Shared PRNG Library

**File:** `barracuda/src/lattice/shaders/prng_pcg_f64.wgsl` (new in v0.6.14)

```wgsl
fn pcg_hash(inp: u32) -> u32 { /* PCG hash */ }
fn hash_u32(idx: u32, seq: u32) -> u32 { /* seeded hash */ }
fn uniform_f64(idx: u32, seq: u32) -> f64 { /* [0,1) */ }
```

- Used by: `su3_random_momenta_f64.wgsl`, `gaussian_fermion_f64.wgsl`
- Composition: Rust `LazyLock<String>` concatenation at compile time
- Requires: `Params` struct with `traj_id`, `seed_lo`, `seed_hi` fields
- Box-Muller/gaussian stays per-consumer (f32 vs f64 cos difference)

**toadStool action:** Absorb as `shaders/math/prng_pcg_f64.wgsl`. This is the
third shared WGSL math library after `complex_f64.wgsl` and `su3_math_f64.wgsl`.
Any spring that needs GPU-side pseudorandom numbers can compose this.

#### 2b. `su3_math_f64.wgsl` — Naga-Safe SU(3) Pure Math

**File:** `barracuda/src/lattice/shaders/su3_math_f64.wgsl` (v0.6.13)

- Fixes real naga validation error: prepending `su3_f64.wgsl` to another shader
  that doesn't use its `ptr<storage>` I/O functions causes naga rejection
- This version has ONLY pure math functions (identity, mul, trace, adj, add, sub)
- Array params copied to `var` locals for dynamic indexing (naga workaround)

**toadStool action:** Consider splitting upstream `su3_f64.wgsl` into math + I/O.
The math-only version is safe for shader composition. This is a recurring pattern:
shaders that mix pure math with buffer I/O break when composed.

#### 2c. `polyakov_loop_f64.wgsl` — GPU-Resident Observable

**File:** `barracuda/src/lattice/shaders/polyakov_loop_f64.wgsl` (v0.6.13)

- Computes temporal Wilson line per spatial site entirely on GPU
- Returns `(Re, Im)` per spatial site: 72× less transfer than full link readback
- Uses t-major indexing (v0.6.11 convention)
- Validated: 6/6 β-scan points

**toadStool action:** Absorb as `ops/lattice/polyakov.rs` + shader. This is the
first GPU-resident lattice observable; same pattern can extend to plaquette
susceptibility, Polyakov loop susceptibility, and Wilson loop ratios.

### Medium Priority (validated but needs adaptation)

#### 2d. ESN Reservoir Shaders

**Files:** `barracuda/src/md/shaders/esn_reservoir_update.wgsl`, `esn_readout.wgsl`

- GPU-accelerated Echo State Network for transport prediction
- Already validated on GPU + NPU pipeline (10/10 hardware checks)
- neuralSpring also benefits from these as a shared primitive

**toadStool action:** Absorb alongside the ESN CPU solver. The reservoir update
is a sparse matrix-vector product + tanh activation; the readout is a dense
matrix-vector product. Both are general-purpose ML primitives.

#### 2e. Neighbor Buffer Abstraction

hotSpring's lattice shaders use a `nbr_buf[site * 8 + dir]` pattern that makes
them site-indexing agnostic. toadStool uses direct-index arithmetic. Absorbing
the neighbor buffer pattern would make all lattice operators portable across
indexing conventions (t-major, t-minor, or custom).

---

## Part 3: Evolution Discoveries

### DF64 Hybrid Precision Strategy

The key discovery from v0.6.10-v0.6.14: consumer GPUs (Ampere/Ada) have
fp64:fp32 = 1:64, but **double-float (f32-pair) on FP32 cores delivers
3.24 TFLOPS at 14-digit precision — 9.9× native f64**.

| Operation | f64 TFLOPS | DF64 TFLOPS | DF64/f64 | Digits |
|-----------|:----------:|:-----------:|:--------:|:------:|
| Gauge force | 0.33 | 3.24 | **9.9×** | 14 |
| Plaquette | 0.33 | 3.24 | **9.9×** | 14 |
| Kinetic energy | 0.33 | 3.24 | **9.9×** | 14 |
| CG solver | 0.33 | — | — | 16 (needs full f64) |

**Recommendation:** DF64 for accumulation-heavy operations (force, plaquette,
energy). Native f64 for precision-critical paths (CG convergence, Dirac
operator). The HMC pipeline is now 60% DF64 by operation count.

### WGSL Shader Composition Patterns

Two patterns emerged for multi-file WGSL:

1. **`include_str!` per-file** (toadStool default): Simple, but requires each
   shader to be self-contained. Breaks when composing math libraries.

2. **`LazyLock<String>` concatenation** (hotSpring v0.6.14): Concatenates shared
   library + consumer shader at compile time. More flexible but requires runtime
   initialization.

**Recommendation:** Adopt pattern #2 for any shader that needs shared math
libraries. The `prng_pcg_f64.wgsl` composition is the reference implementation.

### NVK/nouveau Discovery

| Finding | Impact |
|---------|--------|
| PTE faults at ~1.2 GB allocation | Limit per-buffer size; use buffer splitting |
| naga rejects unused `ptr<storage>` functions | Split shaders into math + I/O layers |
| Dynamic array indexing in naga | Copy array params to `var` locals |
| Titan V NVK produces identical physics to proprietary | Full open-driver sovereignty |

### Capability-Based Discovery

hotSpring v0.6.14 eliminates all hardcoded device paths. The pattern:

```rust
pub fn probe_npu_available() -> bool {
    #[cfg(feature = "npu-hw")]
    { crate::md::npu_hw::NpuHardware::discover().is_some() }
    #[cfg(not(feature = "npu-hw"))]
    { std::fs::read_dir("/dev").map(|e| e.filter_map(Result::ok)
        .any(|e| e.file_name().to_string_lossy().starts_with("akida")))
        .unwrap_or(false) }
}
```

**toadStool action:** Similar pattern for GPU discovery: `DeviceManager::discover()`
should enumerate all Vulkan adapters, filter by capabilities (`SHADER_F64`,
VRAM size, driver name), and provide a sorted priority list. No hardcoded
device names or PCI IDs.

---

## Part 4: Paper Validation Controls (Open Data & Systems)

### Papers Using Open Data

All 22 validated papers use publicly available data and open-source tools:

| Paper | Data Source | Open? | Control Binary |
|-------|------------|:-----:|----------------|
| 1 (Sarkas MD) | Dense Plasma Properties DB (GitHub MIT) | ✅ | `validate_md` |
| 2 (TTM) | UCLA-MSU TTM (GitHub) | ✅ | `validate_ttm` |
| 3 (Surrogate) | Zenodo 10908462 (CC-BY) | ✅ | `nuclear_eos_l1_ref` |
| 4 (Nuclear EOS) | AME2020 (IAEA open) | ✅ | `validate_nuclear_eos` |
| 5 (Transport) | Sarkas + Daligault 2012 (public) | ✅ | `validate_stanton_murillo` |
| 6 (Screening) | Murillo & Weisheit 1998 (public) | ✅ | `validate_screened_coulomb` |
| 7 (HotQCD EOS) | Bazavov 2014 (published tables) | ✅ | `validate_hotqcd_eos` |
| 8 (Pure gauge) | Known β_c literature values | ✅ | `validate_pure_gauge` |
| 9-10 (QCD) | Known plaquette values, β_c | ✅ | `validate_production_qcd` + v2 |
| 10 (Dynamical) | Python control (algorithm-identical) | ✅ | `validate_dynamical_qcd` |
| 11 (HVP g-2) | Lattice QCD literature | ✅ | `validate_hvp_g2` |
| 12 (Freeze-out) | β_c literature + susceptibility | ✅ | `validate_freeze_out` |
| 13 (Abelian Higgs) | Python control + Bazavov 2015 | ✅ | `validate_abelian_higgs` |
| 14-22 (Spectral) | Anderson 1958, exact solutions | ✅ | `validate_spectral` + 3 more |

**No paper depends on gated or proprietary data.** The only gated resource
(Code Ocean capsule for Paper 3) was worked around — we built from first
principles using the open Zenodo data archive.

### Control Substrate Matrix

Every paper is validated across multiple substrates. The progression proves
correctness at each level before promoting:

| Level | Engine | Purpose | Papers Covered |
|-------|--------|---------|:--------------:|
| **Python Control** | Original authors' code | Algorithmic baseline | 18/22 |
| **BarraCuda CPU** | Pure Rust math | Correctness foundation | 22/22 |
| **BarraCuda GPU** | WGSL shaders (wgpu/Vulkan) | GPU acceleration | 20/22 |
| **metalForge** | GPU + NPU + CPU mixed | Heterogeneous pipeline | 9/22 |

---

## Part 5: Hardware Validation Matrix

### BarraCuda CPU (Pure Rust)

| Gate | Hardware | Tests | Status |
|------|----------|:-----:|:------:|
| Eastgate | i9-12900K, 32GB DDR5 | 664 | ✅ PASS |
| biomeGate | 3970X, 256GB DDR4 | 664 | ✅ PASS |

All 22 papers validate on CPU. Rust consistently 50×–2000× faster than Python.

### BarraCuda GPU (WGSL/Vulkan)

| Gate | GPU | Driver | Tests | Key Result |
|------|-----|--------|:-----:|------------|
| Eastgate | RTX 4070 (12GB) | nvidia 580.x | All GPU bins | 9/9 PP Yukawa, 0.000% drift |
| Eastgate | Titan V (12GB HBM2) | NVK/nouveau (Mesa 25.1.5) | 6/6 parity, 40/40 transport | Identical physics to proprietary |
| biomeGate | RTX 3090 (24GB) | nvidia 580.x | 12/12 β-scan at 32⁴ | β_c = 5.69 (deconfinement) |
| biomeGate | Titan V (12GB HBM2) | NVK/nouveau | 16⁴ 9/9 | First NVK QCD |

All GPUs produce identical physics to machine epsilon (1e-15).

### metalForge Mixed Hardware

| Configuration | Gate | Tests | Key Result |
|--------------|------|:-----:|------------|
| RTX 3090 + NPU (NpuSimulator) | biomeGate | 9/9 mixed | 4 domains validated |
| RTX 3090 + NPU + Titan V | biomeGate | 13/13 streaming | Full 3-substrate pipeline |
| RTX 3090 + AKD1000 (real NPU HW) | biomeGate | 16/16 streaming | 100% classification, 2.3e-7 error |
| Titan V + NPU | Eastgate | 10/10 lattice NPU | β_c=5.715, 0.4% error |

The mixed-hardware pipeline validates that GPU-generated physics survives
quantization cascade (f64→f32→int8→int4) and cross-substrate transfer.

---

## Part 6: WGSL Shader Inventory (62 total)

### Ready for Upstream Absorption

| Shader | Lines | Domain | Priority |
|--------|:-----:|--------|:--------:|
| `prng_pcg_f64.wgsl` | 25 | Shared PRNG | **P0** |
| `su3_math_f64.wgsl` | ~110 | SU(3) pure math | **P0** |
| `polyakov_loop_f64.wgsl` | ~80 | Lattice observable | **P0** |
| `esn_reservoir_update.wgsl` | ~60 | ML/ESN | P1 |
| `esn_readout.wgsl` | ~30 | ML/ESN | P1 |
| `su3_gauge_force_df64.wgsl` | ~200 | DF64 gauge force | P1 |

### Already Upstream (via previous absorption cycles)

| Shader | Origin Session |
|--------|:-------------:|
| `complex_f64.wgsl` | S25 |
| `su3_f64.wgsl` | S25 |
| `wilson_plaquette_f64.wgsl` | S25 |
| `su3_hmc_force_f64.wgsl` | S25 |
| `higgs_u1_hmc_f64.wgsl` | S25 |
| `df64_core.wgsl` | S58 |
| `wilson_plaquette_df64.wgsl` | S60 |
| `su3_kinetic_energy_df64.wgsl` | S60 |
| `dirac_staggered_f64.wgsl` | S31d |
| `staggered_fermion_force_f64.wgsl` | S31d |
| CG shaders (axpy, xpay, dot, update, reduce) | S31d |

---

## Part 7: Remaining Evolution Gaps

| Gap | Blocker | Priority |
|-----|---------|:--------:|
| Pseudofermion HMC GPU shader | Complex CG + heat bath on GPU | P1 |
| DF64 CG solver | Need DF64 complex arithmetic | P2 |
| GPU-resident Lanczos (N > 100k) | GPU dot + axpy + scale | P2 |
| Deformed HFB full GPU | Large matrix eigensolve on GPU | P3 |
| SparsitySampler port | Sampling strategy for L2 EOS | P3 |
| Buffer splitting for NVK | Workaround for nouveau PTE limit | P2 |

---

## Closing

hotSpring v0.6.14 is the cleanest the crate has ever been. All 22 papers
validate on open data with open tools across CPU, GPU, and mixed hardware.
The 3 P0 shaders (`prng_pcg_f64`, `su3_math_f64`, `polyakov_loop_f64`) are
ready for immediate absorption. The DF64 hybrid strategy, WGSL composition
patterns, and NVK findings are reusable across all springs.

The barracuda team has everything needed to evolve the shared library and
propagate these capabilities to wetSpring, neuralSpring, and airSpring.
