SPDX-License-Identifier: AGPL-3.0-or-later

# hotSpring → barraCuda/toadStool: S93 + v0.3.1 Rewire, Revalidation, and Shader Audit

**Date:** 2026-03-03
**From:** hotSpring v0.6.16
**To:** barraCuda team + toadStool/barracuda team
**Covers:** Full rewire to barraCuda v0.3.1, toadStool S93 pull, shader deduplication audit, upstream bug reports, metalForge rewire
**License:** AGPL-3.0-or-later

---

## Executive Summary

hotSpring has completed a full rewire and revalidation against:
- **barraCuda v0.3.1** (commit `f6895ca`) — up from v0.2.0
- **toadStool S93** (commit `9319668d`) — up from S80

**Results:** 663 lib tests + 25 metalForge tests = **688 total pass, 0 failures**.
Release build clean. All 25 WGSL lattice shaders audited against barraCuda equivalents.

---

## Part 1: Dependency Rewire

### 1.1 hotspring-barracuda

```toml
# barracuda/Cargo.toml — already wired (prior session):
barracuda = { path = "../../barraCuda/crates/barracuda" }

# Fixed path case for NPU deps (was lowercase "toadstool"):
akida-driver = { path = "../../phase1/toadStool/crates/neuromorphic/akida-driver", optional = true }
akida-models = { path = "../../phase1/toadStool/crates/neuromorphic/akida-models", optional = true }
```

### 1.2 metalForge

```toml
# metalForge/forge/Cargo.toml — rewired from toadStool to standalone:
# Before:
barracuda = { path = "../../../phase1/toadstool/crates/barracuda" }
# After:
barracuda = { path = "../../../barraCuda/crates/barracuda" }
```

### 1.3 Stashed Local Fixes

Our previous `sin_f64_safe` reciprocal-multiply fix and sovereign shader compilation
changes were stashed in both toadStool and barraCuda repos. barraCuda v0.2.1 absorbed
the `sin_f64_safe` fix upstream (replacing `x % two_pi` with `x - floor(x / two_pi) * two_pi`).
The stashed changes are no longer needed.

---

## Part 2: API Compatibility (S84-S93)

### 2.1 What's New Since S80

| Session | Key Changes | hotSpring Impact |
|---------|-------------|------------------|
| S84-86 | ComputeDispatch migration (144 ops total) | No impact — hotSpring uses high-level ops |
| S87 | 6 hotSpring fault tests fixed; AFIT→NOTE; unsafe audit | Tests already pass; no code changes needed |
| S88 | `MultiHeadEsn::from_exported_weights()`, `SeasonalGpuParams::new()`, cross-spring tolerances | Different API from hotSpring's local `MultiHeadNpu` — not absorbed |
| S89 | barraCuda budding complete, standalone extraction | Already wired |
| S90-92 | Deep audit, sovereignty evolution, debris cleanup | Internal to toadStool |
| S93 | Root docs cleanup, D-DF64 transfer to barraCuda | No hotSpring impact |

### 2.2 New APIs Not Absorbed (by design)

- **`MultiHeadEsn::from_exported_weights()`** — barraCuda's is async, GPU-backed tensor API. hotSpring's `MultiHeadNpu` is synchronous CPU f32 for NPU steering. Different use cases.
- **`SeasonalGpuParams::new()`** — hotSpring doesn't use seasonal parameters.
- **Domain feature gates** (v0.3.1: `lattice`, `md`, `spectral`) — hotSpring uses default `gpu` feature; compile-time opt-in not needed.
- **IPC/tarpc parity** (v0.3.1) — hotSpring runs in-process; cross-process barraCuda not needed yet.

### 2.3 Deprecated APIs

hotSpring does NOT use any deprecated APIs:
- `PppmGpu::new()` → already uses `PppmGpu::from_device()` ✅
- `WgpuDevice::from_synthetic()` → not used ✅

---

## Part 3: Shader Deduplication Audit

### 3.1 Methodology

Diffed all 25 hotSpring local lattice WGSL shaders against barraCuda equivalents
in `shaders/lattice/` and `shaders/math/`.

### 3.2 Results

| Category | Count | Shaders |
|----------|-------|---------|
| **Identical** (after license align) | 7 | `axpy_f64`, `complex_dot_re_f64`, `su3_f64`, `su3_gauge_force_df64`, `su3_kinetic_energy_df64`, `su3_math_f64`, `xpay_f64` |
| **Comment-only divergence** | 2 | `cg_compute_alpha_f64`, `cg_compute_beta_f64` (barraCuda added "Absorbed from hotSpring" comments) |
| **2D dispatch divergence** | 8 | `cg_update_p/xr_f64`, `su3_gauge_force/kinetic_energy/link_update/momentum_update_f64`, `sum_reduce_f64` |
| **PRNG divergence** | 1 | `prng_pcg_f64` (hotSpring uses reciprocal multiply for NVK; see upstream bug report below) |
| **Architectural divergence** | 7 | `complex_f64`, `dirac_staggered_f64`, `gaussian_fermion_f64`, `polyakov_loop_f64`, `staggered_fermion_force_f64`, `su3_random_momenta_f64`, `wilson_plaquette_f64/df64` |

### 3.3 Divergence Rationale

**2D Dispatch (keep local):** hotSpring uses `gid.x + gid.y * nwg.x * 64u` to work
around NVK/Nouveau workgroup-count limits. barraCuda simplified to `gid.x` which may
hit dispatch limits on older hardware. The local pattern is NVK-safe and tested.

**Architectural (keep local):** hotSpring's `wilson_plaquette_f64`, `polyakov_loop_f64`,
etc. are self-contained single-file shaders with inline SU(3) ops. barraCuda's versions
use prepend chains (`complex_f64.wgsl + su3.wgsl + ...`). hotSpring's GPU HMC pipeline
composes shaders via `include_str!` concatenation, not the `ShaderTemplate` prepend system.
Converting would require rewriting the entire shader compilation path.

**Complex64 (keep local):** hotSpring uses `struct Complex64 { re: f64, im: f64 }`
throughout all lattice shaders. barraCuda's `math/complex_f64.wgsl` uses `vec2<f64>`.
barraCuda's lattice shaders still use the struct form internally, so there's an
inconsistency in barraCuda itself.

### 3.4 License Alignment

All 25 lattice + 14 MD + 14 physics WGSL shaders (53 total) updated from
`AGPL-3.0-only` to `AGPL-3.0-or-later` to match barraCuda's convention.

---

## Part 4: Upstream Bug Reports

### 4.1 prng_pcg_f64.wgsl — f64 Division on NVK

**File:** `barraCuda/crates/barracuda/src/shaders/lattice/prng_pcg_f64.wgsl`

**Problem:** Line 25 uses `/ f64(4294967296.0)` which triggers naga f64 division
rejection on NVK drivers.

**hotSpring fix:** Uses `* f64(2.3283064365386963e-10)` (precomputed reciprocal).
Numerically equivalent, NVK-safe.

**Recommendation:** Replace division with reciprocal multiply in barraCuda's version.

### 4.2 2D Dispatch Pattern

barraCuda's lattice shaders use 1D dispatch (`gid.x`). On NVK/Nouveau with older
Vulkan drivers, dispatch X-dimension can be limited to ~65535 workgroups. For volumes
≥10^4 (10,000 sites × 4 directions × 18 components = 720K elements), 1D dispatch
hits this limit.

hotSpring's 2D pattern works up to ~4 billion elements. Consider adopting for
barraCuda lattice shaders or adding a dispatch helper.

### 4.3 Complex64 Inconsistency

barraCuda has two Complex64 representations:
- `shaders/math/complex_f64.wgsl`: `vec2<f64>` (`.x`, `.y`)
- `shaders/lattice/*.wgsl`: `struct Complex64 { re, im }` (`.re`, `.im`)

These are incompatible. Lattice shaders that prepend `math/complex_f64.wgsl` will
get type conflicts if they also use the struct form.

---

## Part 5: NPU Ownership Architecture

The primal split clarifies NPU ownership:

```
toadStool (hardware orchestration)
  └── crates/neuromorphic/
      ├── akida-driver       # PCIe device access, register I/O
      ├── akida-models       # Model parsing, layer config
      ├── akida-setup        # Device initialization binary
      ├── akida-reservoir-research  # Research tooling
      ├── cross-substrate-validation
      └── neurobench-runner

barraCuda (math + compilation)
  ├── esn_v2/              # MultiHeadEsn, ExportedWeights, GPU reservoir
  ├── npu/                 # EventCodec, quantization (quantize_affine_i8_f64)
  ├── npu_executor/        # NPU inference dispatch
  ├── npu_bridge (ops)     # CPU↔NPU data bridging
  └── snn/                 # Spiking neural network ops
```

**Principle:** toadStool exposes capabilities (what hardware exists, what it supports).
barraCuda compiles and executes (shader compilation, weight quantization, inference math).
Springs consume both: toadStool for device discovery, barraCuda for compute.

hotSpring's current wiring reflects this:
- `barracuda = { path = "../../barraCuda/crates/barracuda" }` — math, ESN, shaders
- `akida-driver = { path = "../../phase1/toadStool/crates/neuromorphic/akida-driver" }` — hardware

### 5.1 What hotSpring Contributes to barraCuda NPU Math

| Component | File | What It Does | Absorption Target |
|-----------|------|-------------|-------------------|
| `ReluTanhApprox` | `md/reservoir/mod.rs` | 5-segment piecewise-linear tanh for AKD1000 bounded-ReLU | `esn_v2` or `npu/activations` |
| `HeadConfidence` | `production/npu_worker.rs` | Rolling R² per head, trust/fallback switching | `esn_v2::multi_head` |
| `MultiHeadNpu` | `md/reservoir/npu.rs` | CPU f32 NPU simulator with 36-head readout | Reference for `esn_v2::MultiHeadEsn` sync path |
| `NpuSimulator` | `md/reservoir/npu.rs` | Lightweight CPU ESN for real-time steering | Could be `esn_v2::CpuEsn` |
| Canonical 6D input | `production/npu_worker.rs` | `[beta_norm, plaq, mass, chi/1000, acc, lat/8]` | Pattern for `esn_v2` input normalization |

### 5.2 What hotSpring Contributes to toadStool Hardware

| Component | File | What It Does | Absorption Target |
|-----------|------|-------------|-------------------|
| NPU sysfs discovery | `discovery.rs` | `/dev/akida*` + `/sys/class/akida/` probing | `neuromorphic/akida-driver` |
| Brain 4-layer arch | `production/npu_worker.rs` | Cortex/Cerebellum/Motor/Pre-motor pipeline | `neuromorphic/` orchestration pattern |
| CG residual monitoring | `production/npu_worker.rs` | Real-time attention state machine (Green/Yellow/Red) | Hardware interrupt pattern |

---

## Part 6: NAK/NVK Testing for barraCuda Absorption

hotSpring has extensive NAK (nouveau shader compiler) testing that exercises edge cases
barraCuda's driver profile system should absorb. These tests run on real hardware
(Titan V via NVK + RTX 3090 via proprietary) and expose compilation and dispatch gaps.

### 6.1 NAK Test Inventory

| Binary | File | Tests | What It Proves |
|--------|------|-------|---------------|
| `validate_nak_eigensolve` | `bin/validate_nak_eigensolve.rs` | Correctness + perf | NAK-optimized eigensolve shader vs baseline; proves `workgroup_size(32)` cooperative pattern beats `workgroup_size(1)` on NVK by 10-50× |
| `bench_wgsize_nvk` | `bin/bench_wgsize_nvk.rs` | Perf diagnostic | Proves NVK penalizes `workgroup_size(1)` much more than proprietary; motivates cooperative shader design |
| `bench_fp64_ratio` | `bin/bench_fp64_ratio.rs` | FP64:FP32 ratio | Measures native f64 throughput; Titan V = 1:2, RTX 3090 = 1:64 |
| `bench_gpu_fp64` | `bin/bench_gpu_fp64.rs` | FP64 throughput | Raw FP64 GFLOPS on each GPU |

### 6.2 NAK Workarounds hotSpring Evolved

| Workaround | Location | Pattern |
|------------|----------|---------|
| f64 modulo → floor-multiply | `lattice/shaders/prng_pcg_f64.wgsl` | `x - floor(x / y) * y` instead of `x % y` |
| 2D dispatch for large volumes | `lattice/shaders/*.wgsl` (18 shaders) | `gid.x + gid.y * nwg.x * 64u` for >65535 workgroups |
| NVK allocation guard | `lattice/gpu_hmc/mod.rs` | `profile.check_allocation_safe()` before PTE fault |
| SPIR-V sovereign path | `gpu/mod.rs` | `SPIRV_SHADER_PASSTHROUGH` bypasses naga f64 validation |
| `enable f64;` stripping | via `ShaderTemplate::for_sovereign()` | Sovereign compiler strips WGSL extension for SPIR-V emit |
| Cooperative eigensolve | `shaders/batched_eigh_nak_optimized_f64.wgsl` | `workgroup_size(32)` shared-memory Jacobi rotation |

### 6.3 Absorption Recommendation

barraCuda already has `GpuDriverProfile` and `Fp64Strategy` in `device/driver_profile/`.
The next round of absorption should:

1. **Add `dispatch_2d_threshold`** to `GpuDriverProfile` — when volume × components exceeds
   the NVK X-dimension workgroup limit (~65535), auto-switch to 2D dispatch.

2. **Absorb `batched_eigh_nak_optimized_f64.wgsl`** — the cooperative eigensolve is already
   in barraCuda's `shaders/linalg/` but should be the default path when `is_nak()` returns true.

3. **Absorb the NAK benchmarks** — `bench_wgsize_nvk` and `validate_nak_eigensolve` belong
   in barraCuda's `src/bin/` for CI regression testing on NVK hardware.

4. **Add `prng_reciprocal_multiply` workaround** to `polyfill.rs` — the NVK f64 division
   issue affects any consumer using `prng_pcg_f64.wgsl` on nouveau.

---

## Part 7: Cross-Spring Evolution Notes

### 7.1 Precision Shaders

hotSpring contributed:
- `sin_f64_safe` reciprocal-multiply fix (absorbed in barraCuda v0.2.1)
- `for_sovereign()` shader template with `enable f64;` stripping (compilation.rs)
- NVK f64 modulo workaround (floor-multiply pattern)

### 7.2 NPU/ESN Evolution

hotSpring's Brain architecture (4-layer: Cortex, Cerebellum/NPU, Motor, Pre-motor)
and `MultiHeadNpu` with 36 heads are local to hotSpring. Key innovations:
- `ReluTanhApprox`: 5-segment piecewise-linear tanh for AKD1000 bounded-ReLU
- `HeadConfidence` tracker: rolling R² per head, trust/fallback switching
- 6D canonical input vector with normalized lattice size
- DP memoization architecture for cross-volume knowledge transfer

These are specific to hotSpring's physics domain but the `ReluTanhApprox` activation
and `HeadConfidence` patterns could be generalized in barraCuda's ESN v2.

### 7.3 Shader Absorption from Other Springs

barraCuda's lattice shaders carry "Absorbed from hotSpring lattice QCD (Feb 2026)"
provenance comments. The 7 identical shaders confirm successful absorption. The
18 divergent ones reflect hotSpring's hardware-specific evolution (NVK 2D dispatch,
self-contained compilation model).

---

## Part 8: Test Results

| Crate | Tests | Result |
|-------|-------|--------|
| hotspring-barracuda (lib) | 663 | ✅ PASS |
| hotspring-forge (metalForge) | 25 | ✅ PASS |
| hotspring-barracuda (release check) | — | ✅ PASS |
| **Total** | **688** | **0 failures** |

**barraCuda version:** v0.3.1 (commit `f6895ca`)
**toadStool version:** S93 (commit `9319668d`)

---

## Part 9: Remaining Work

| Item | Owner | Priority |
|------|-------|----------|
| Fix prng_pcg_f64 reciprocal multiply | barraCuda | Medium |
| Absorb NAK benchmarks + eigensolve default | barraCuda | Medium |
| Add `dispatch_2d_threshold` to GpuDriverProfile | barraCuda | Medium |
| Fix Complex64 vec2 vs struct inconsistency | barraCuda | Low |
| Absorb `ReluTanhApprox` into esn_v2 activations | barraCuda | Low |
| Absorb `HeadConfidence` pattern into esn_v2 | barraCuda | Low |
| `barracuda-core` + `sourdough-core` availability | barraCuda | Blocked |
| Phase 1 boundary hardening (API audit) | barraCuda | Next |
| Adopt barraCuda domain feature gates when mature | hotSpring | Future |

---

## Part 10: Files Changed in This Rewire

| File | Change |
|------|--------|
| `barracuda/Cargo.toml` | Path case fix (toadstool→toadStool), comment updates |
| `metalForge/forge/Cargo.toml` | Rewired from toadStool to barraCuda v0.3.1 |
| `barracuda/src/lattice/shaders/*.wgsl` (25) | License: AGPL-3.0-only → AGPL-3.0-or-later |
| `barracuda/src/md/shaders/*.wgsl` (14) | License aligned |
| `barracuda/src/physics/shaders/*.wgsl` (14) | License aligned |
| `barracuda/README.md` | Updated to reference barraCuda v0.3.1 |
| `barracuda/docs/REDUCE_PIPELINE_ANALYSIS.md` | Path reference updated |
| `barracuda/src/lattice/complex_f64.rs` | Provenance comment updated |
| `metalForge/README.md` | Dependency reference updated |
| `specs/README.md` | Status updated to barraCuda v0.3.1 |
| `README.md` | Dependency tree references updated |
