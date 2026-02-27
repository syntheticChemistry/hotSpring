# toadStool / barracuda Team — Comprehensive Absorption Handoff

**Date:** February 25, 2026
**From:** hotSpring (biomeGate)
**To:** toadStool / barracuda core team
**Covers:** hotSpring v0.6.10 → v0.6.14 (full DF64 + cross-spring evolution + debt reduction)
**License:** AGPL-3.0-only

---

## Executive Summary

hotSpring has completed its most intensive evolution cycle: from native f64
baseline through DF64 core streaming, site-indexing standardization, and
cross-spring rewiring. This handoff documents everything toadStool needs to
absorb, evolve, and redistribute to all springs.

**Key metrics (v0.6.14):**
- 76 binaries, 25 lattice WGSL shaders, 664 tests (629 lib + 31 integration + 4 doc)
- 32⁴ quenched HMC at 7.7s/trajectory (2× faster than v0.6.8)
- GPU Polyakov loop: 72× less data transfer
- DF64: 60% of HMC pipeline, 3.24 TFLOPS on FP32 cores
- Zero TODO/FIXME in 167 .rs files, zero clippy warnings (lib + bins)
- Zero mocks in production code, zero unsafe blocks
- All cross-primal hardcoding eliminated → capability-based discovery
- β_c provenance centralized, ~150 validation tolerances with physics justification

---

## Part 1: What to Absorb

### Priority 1 — Immediate (validated, tested, ready)

#### 1a. GPU Polyakov Loop Shader

**File:** `barracuda/src/lattice/shaders/polyakov_loop_f64.wgsl`
**Bindings:** `@group(0) @binding(0)` links (read), `@binding(1)` neighbors (read),
`@binding(2)` params (uniform), `@binding(3)` poly output (read_write)
**Entry:** `@compute @workgroup_size(64) fn main`
**Math preamble:** Requires `complex_f64.wgsl` + `su3_math_f64.wgsl` prepended

- Computes temporal Wilson line per spatial site: `L(x) = Tr[∏_t U_0(x,t)] / 3`
- Returns `(Re, Im)` per spatial site — 72× less transfer than full link readback
- Uses t-major indexing (standardized in v0.6.11)
- Validated: 6/6 beta scan points, magnitude + phase both correct

**toadStool action:** Absorb into `ops/lattice/gpu_polyakov.rs` alongside existing
implementation. hotSpring's version uses prepended math libraries; toadStool's uses
`include_str!` per-file — reconcile the composition pattern.

#### 1b. Naga-Safe SU(3) Math Library

**File:** `barracuda/src/lattice/shaders/su3_math_f64.wgsl`

- Pure mathematical SU(3) functions only (no `ptr<storage>` I/O)
- Safe for shader composition via prepending (naga doesn't reject unused storage fns)
- Array parameters copied to `var` locals for dynamic indexing (naga workaround)
- Functions: `su3_identity`, `su3_mul`, `su3_trace`, `su3_adj`, `su3_add`, `su3_sub`

**toadStool action:** Consider splitting upstream `su3_f64.wgsl` similarly. The I/O
functions (`su3_load`, `su3_store`) cause naga validation errors when the shader is
prepended to another shader that doesn't use them. This affects any operator that
wants to compose SU(3) math with custom buffer layouts.

#### 1c. NVK Allocation Guard Pattern

**Location:** `barracuda/src/lattice/gpu_hmc.rs`, `GpuHmcState::from_lattice()`

```rust
let total_estimate = /* sum of buffer sizes */;
gpu.driver_profile().check_allocation_safe(total_estimate);
```

- Already uses toadStool's `check_allocation_safe()` API
- Logs warning when estimated VRAM exceeds nouveau PTE limit (~1.2 GB)
- Prevents silent kernel crashes on Titan V/NVK

**toadStool action:** The guard is already upstream. Ensure all new `ops/lattice/`
operators also call `check_allocation_safe()` before buffer creation.

#### 1d. PRNG Type-Safety Fix

**File:** `barracuda/src/lattice/shaders/su3_random_momenta_f64.wgsl`
**Bug:** `box_muller_cos` cast theta to `f32` before `cos()`, but `ShaderTemplate`
patches `cos` → `cos_f64` (expecting `f64` argument). The cast was removed.

**toadStool action:** Audit all WGSL shaders for similar `f32`/`f64` type mismatches
in transcendental function arguments, especially where `ShaderTemplate` patching
occurs. The pattern `f64(cos(f32(x)))` silently breaks when `cos` is remapped.

### Priority 2 — Evolution Targets

#### 2a. DF64 Gauge Force (local, not yet upstream)

**File:** `barracuda/src/lattice/shaders/su3_gauge_force_df64.wgsl`

- Uses hotSpring's neighbor-buffer indexing (t-major since v0.6.11)
- Depends on toadStool's `df64_core.wgsl` for arithmetic
- 9.9× native f64 throughput on consumer Ampere/Ada GPUs

**toadStool action:** This is the highest-value DF64 shader to absorb. The
neighbor-buffer pattern differs from toadStool's direct-index pattern — adding
a `neighbor_table` buffer to toadStool's gauge force would make it portable.

#### 2b. DF64 Plaquette and Kinetic Energy

**Files:**
- `wilson_plaquette_df64.wgsl` (absorbed from toadStool S60 in v0.6.12)
- `su3_kinetic_energy_df64.wgsl` (absorbed from toadStool S60 in v0.6.12)

**toadStool action:** These are already upstream via S60. Confirm parity with
hotSpring's local copies after any upstream changes.

#### 2c. Site-Indexing Convention

hotSpring adopted toadStool's t-major convention in v0.6.11. The conversion:
```
t-minor: idx = x + Nx*(y + Ny*(z + Nz*t))  → spatial dims innermost
t-major: idx = t + Nt*(x + Nx*(y + Ny*z))  → time innermost
```

**toadStool action:** Document t-major as the canonical convention. Consider
adding a utility function for convention conversion so springs that haven't
migrated can interop.

### Priority 3 — Patterns Worth Absorbing

| Pattern | Location | Benefit |
|---------|----------|---------|
| `PolyParams` uniform struct | `gpu_hmc.rs` | Reusable for any shader needing lattice geometry |
| Shader composition via `format!("{}\n{}\n{}", ...)` | `gpu_hmc.rs` | Simple alternative to `include_str!` for multi-file shaders |
| Validation harness with `(magnitude, phase)` | All binaries | Standard observable return type |

---

## Part 2: Cross-Spring Evolution Map

### Shader Provenance (as of Feb 25, 2026)

| Shader | Origin | Current Home | Users |
|--------|--------|-------------|-------|
| `complex_f64.wgsl` | hotSpring | toadStool | all springs |
| `su3_f64.wgsl` | hotSpring | toadStool | hotSpring, toadStool |
| `su3_math_f64.wgsl` | hotSpring v0.6.13 | hotSpring (pending absorption) | hotSpring |
| `polyakov_loop_f64.wgsl` | toadStool → hotSpring (bidirectional) | both | hotSpring |
| `df64_core.wgsl` | hotSpring discovery → toadStool | toadStool | all springs |
| `wilson_plaquette_df64.wgsl` | toadStool S60 → hotSpring v0.6.12 | both | hotSpring |
| `su3_gauge_force_df64.wgsl` | hotSpring | hotSpring (local) | hotSpring |
| `distance_f64.wgsl` | hotSpring MD | toadStool | wetSpring (phylogenetics) |
| `esn_*.wgsl` | hotSpring → toadStool | toadStool | neuralSpring |
| `(zero + literal)` f64 fix | wetSpring | toadStool `math_f64.wgsl` | all springs |

### Cross-References to Other Springs

| Document | Location | Content |
|----------|----------|---------|
| wetSpring cross-spring evolution | `wetSpring/wateringHole/CROSS_SPRING_SHADER_EVOLUTION.md` | 612-shader census, full timeline |
| wetSpring cross-spring provenance | `wetSpring/wateringHole/handoffs/CROSS_SPRING_PROVENANCE_FEB22_2026.md` | Per-shader contribution attribution |
| hotSpring evolution map | `hotSpring/experiments/016_CROSS_SPRING_EVOLUTION_MAP.md` | 164+ shader tracking, validation matrix |
| hotSpring baseCamp briefings | `hotSpring/whitePaper/baseCamp/` | Per-domain summaries with cross-spring notes |

---

## Part 3: Pending toadStool Tasks

These were identified during hotSpring evolution and documented in the
site-indexing/NAK handoff (`TOADSTOOL_SITE_INDEXING_NAK_SOLVER_HANDOFF_FEB25_2026.md`):

### NAK / NVK Driver Issues (Rust-native solution)

1. **Buffer size PTE faults**: nouveau driver crashes at ~1.2 GB. The allocation
   guard warns but doesn't solve the underlying limit. toadStool should evolve
   buffer-splitting strategies for large lattices on NVK.

2. **NAK compiler issues**: Some WGSL patterns cause NAK shader compilation
   failures (especially dynamic array indexing and complex control flow).
   toadStool should build a Rust-native WGSL→SPIR-V→NIR path that can work
   around NAK limitations — more portable than depending on Mesa updates.

3. **Titan V + RTX 3090 side-by-side**: Both GPUs are validated independently.
   Multi-GPU dispatch (Titan V for precision work, 3090 for throughput) is
   architecturally ready but not yet stress-tested in production.

### Neighbor Buffer Abstraction

hotSpring uses a `nbr_buf` (8 neighbors per site: ±μ for μ=0..3) that makes
gauge-theory shaders site-indexing agnostic. toadStool's direct-index pattern
assumes a specific convention. Absorbing the neighbor buffer pattern would make
all lattice operators indexing-convention portable.

### Mixed Precision Strategy Documentation

The DF64 hybrid strategy (FP32 cores for bulk math, native FP64 for
precision-critical operations) should be documented as a first-class toadStool
capability with:
- When to use DF64 vs native f64 (14 vs 16 digits, 9.9× vs 1× throughput)
- Per-GPU fp64:fp32 ratios (Titan V 1:2, consumer Ampere/Ada 1:64)
- Which operations benefit most from DF64 (gauge force > plaquette > KE)

---

## Part 4: Barracuda Evolution Summary

### Version Timeline

| Version | Date | Key Change |
|---------|------|-----------|
| v0.6.8 | Feb 24 | Production β-scan baseline (32⁴, 13.6h, native f64) |
| v0.6.9 | Feb 24 | toadStool S62 sync, spectral lean (41 KB deleted) |
| v0.6.10 | Feb 24 | DF64 gauge force on RTX 3090 (9.9× FP32 throughput) |
| v0.6.11 | Feb 25 | t-major site indexing standardization (119/119 unit tests pass) |
| v0.6.12 | Feb 25 | toadStool S60 DF64 expansion (plaquette, KE, transcendentals). 60% HMC in DF64 |
| v0.6.13 | Feb 25 | GPU Polyakov loop, NVK guard, su3_math_f64, PRNG fix. 13/13 checks |
| v0.6.14 | Feb 25 | Debt reduction: cross-primal discovery, gauss-jordan→lu_solve, WGSL dedup, 57 clippy fixes, β_c provenance |

### Crate Health

| Metric | Value |
|--------|-------|
| Version | 0.6.14 |
| Tests | 664 (629 lib + 31 integration + 4 doc) |
| Binaries | 76 |
| WGSL shaders (lattice) | 25 (incl. prng_pcg_f64 shared library) |
| Open TODO/FIXME | 0 |
| Clippy warnings (lib+bins) | 0 |
| `#[allow(dead_code)]` | 9 files (HFB GPU pipeline, future wiring) |
| Unsafe blocks | 0 |
| Mocks in production | 0 (NpuSimulator is a legitimate software impl) |
| Cross-primal hardcoding | 0 (all capability-based discovery) |
| Validation tolerances | ~150 centralized with physics justification |

---

## Part 5: v0.6.14 Debt Reduction (New)

### Cross-Primal Discovery

All hardcoded cross-primal references eliminated from production code:
- `/dev/akida0` → `discovery::probe_npu_available()` (sysfs scan / akida-driver)
- `control/metalforge_npu` → `control/npu` (generic capability name)
- toadstool/showcase path → removed from error messages
- metalForge doc reference → replaced with env var documentation

**toadStool action:** Adopt the `probe_npu_available()` pattern for hardware discovery.
The sysfs scan fallback (`/dev/akida*`) works without the `npu-hw` feature gate.

### Dependency Evolution

- `gauss_jordan_solve` (68-line local implementation) → `barracuda::ops::linalg::lu_solve`
- All external deps analyzed: serde, rayon, bytemuck, tokio, wgpu are KEEP (essential, pure Rust, or required for GPU). `akida-models` is EVOLVE (optional, not directly used).

### WGSL Shader Deduplication

Created `prng_pcg_f64.wgsl` shared PRNG library (PCG hash → uniform f64).
Both PRNG consumer shaders now compose via `include_str!` + `LazyLock<String>`.
The `box_muller_cos`/`gaussian` functions remain per-consumer (f64 cos vs f32 cast
difference is intentional and validated separately).

**toadStool action:** Consider absorbing `prng_pcg_f64.wgsl` as a shared PRNG library
alongside `complex_f64.wgsl` and `su3_math_f64.wgsl`.

### Validation Fidelity Hardening

- `KNOWN_BETA_C_SU3_NT4` (5.6925) added to `provenance.rs` with Bali/Engels/Creutz citations
- 7 binaries migrated from local `const KNOWN_BETA_C` to centralized provenance
- 4 new GPU validation tolerances in `tolerances::lattice` (streaming parity, fermion force, CG action, dynamical streaming)
- ~150 tolerance constants, ~95% of validation thresholds now centralized

### Evolution Readiness Assessment

| Status | Modules |
|--------|---------|
| **READY** | `lattice/gpu_hmc` (GPU-resident HMC, DF64, streaming, resident CG), `physics/hfb_gpu_resident` (full L2 pipeline), `physics/bcs_gpu` |
| **NEAR-READY** | `md` (GPU simulation, needs DEPRECATED shader migration to `barracuda::ops::md::*`) |
| **NOT READY** | CPU-only: `hmc`, `pseudofermion`, `abelian_higgs`, `screened_coulomb`, `nuclear_matter` |

---

## Part 6: Recommended Absorption Order

1. **su3_math_f64.wgsl** — immediate, fixes a real naga composition bug
2. **polyakov_loop_f64.wgsl** — immediate, GPU-resident observable
3. **prng_pcg_f64.wgsl** — immediate, shared PRNG library for all Gaussian generation
4. **PRNG type-safety audit** — quick, prevents silent breakage
5. **Neighbor buffer abstraction** — medium, unlocks indexing portability
6. **DF64 gauge force upstream** — high-value, biggest DF64 performance gain
7. **NAK Rust-native workarounds** — longer term, enables open-driver sovereignty
