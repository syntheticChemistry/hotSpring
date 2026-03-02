SPDX-License-Identifier: AGPL-3.0-only

# hotSpring → toadStool: S78 Sync Status + Remaining Absorption Targets

**Date:** 2026-03-02
**From:** hotSpring v0.6.15
**To:** toadStool/barracuda team
**Covers:** S68 (e96576ee) → S78 (223b2007) sync
**License:** AGPL-3.0-only

---

## Executive Summary

- hotSpring pulled toadStool S68→S78 (18 commits) and compiles cleanly
- 711 tests pass, 0 clippy warnings, 0 compile errors
- **One breaking change handled**: `Fp64Strategy::Concurrent` added to enum — hotSpring
  now matches it alongside `Hybrid` in all 4 `gpu_hmc/mod.rs` match sites
- **box_muller_cos fix** (ad9e9dea) already applied in hotSpring's local shaders
- hotSpring does NOT rewire deeply integrated local modules to upstream —
  the Write→Absorb→Lean cycle works at the shader/primitive level, not
  by replacing hotSpring's validated orchestration code

---

## What toadStool Absorbed Since S68 (from hotSpring's perspective)

| toadStool Session | What Changed | hotSpring Impact |
|-------------------|-------------|------------------|
| S68++ | AGPL-3.0 audit, chrono eliminated, 0 clippy | Compatible |
| S68+++ | Dead code cleanup, unsafe evolved, deep debt | Compatible |
| S69++ | ComputeDispatch migration (71 ops) | New pattern available |
| S70 | Deep debt, concurrent Rust | Compatible |
| S70+ | Cross-spring absorption, DF64 ML shaders | New shaders available |
| S71 | GPU dispatch wiring, sovereignty constants | Compatible |
| S71+ | DF64 transcendentals (gamma, erf, trig) | New math available |
| S71++ | ComputeDispatch batch 2-3, unsafe reduction | Compatible |
| S71+++ | ComputeDispatch batch 4-6, deps audit | Compatible |
| S78 | libc→rustix, AFIT migration, wildcard narrowing | **Fp64Strategy::Concurrent** added |

---

## What hotSpring Still Needs from toadStool

### Priority 1: Already handed off, not yet absorbed

| Item | Handoff | Status |
|------|---------|--------|
| Pseudofermion HMC force + heatbath | `lattice/pseudofermion/` (477+ lines, 7/7 checks) | Handed off, not absorbed |
| ESN 11-head weight migration | `md/reservoir/` (11 physics heads, NPU integration) | Handed off, pending |
| `NeighborMode::PrecomputedBuffer` | Precomputed neighbor table for repeated HMC | Not yet in toadStool |
| `GpuDriverProfile` sin/cos workarounds | needs_sin_f64_workaround, needs_cos_f64_workaround | Not yet in toadStool |

### Priority 2: New upstream primitives hotSpring could lean on (future)

| Upstream Primitive | Current hotSpring Local | Migration Effort |
|-------------------|------------------------|------------------|
| `ComputeDispatch` builder | ~80 lines BGL/BG/pipeline per op | Medium — would clean up gpu_hmc |
| `ops::md::stress_virial` GPU | Local CPU stress tensor | Low — add GPU path |
| `ops::md::vacf` batch GPU | Local `compute_vacf` CPU | Low — add GPU path |
| `ops::stats_f64::linear_regression` | Local CPU regression | Low |
| DF64 gamma/erf transcendentals | barracuda::special::gamma/erf CPU | Low — already using CPU |

### Priority 3: hotSpring-specific (will never be absorbed)

- Transport fits (Stanton-Murillo, Daligault) — domain-specific
- Nuclear physics (SEMF, HFB, BCS) — domain-specific
- Production pipeline orchestration (4-layer brain, NPU worker) — domain-specific
- Validation harness, provenance, tolerances — hotSpring-specific

---

## What hotSpring Is Already Leaning On

| Module | Upstream Source | Since |
|--------|---------------|-------|
| `spectral/` | `barracuda::spectral::*` | S25-31h (v0.6.4) |
| `md/celllist.rs` | `barracuda::ops::md::CellListGpu` | S25 (v0.6.2) |
| Complex64 WGSL | `barracuda::ops::lattice::complex_f64` | S25 |
| SU(3) WGSL | `barracuda::ops::lattice::su3` | S25 |
| DF64 core math | `barracuda::ops::lattice::su3::WGSL_DF64_CORE` | S58 (v0.6.9) |
| DF64 gauge force | `barracuda::ops::lattice::su3::WGSL_SU3_DF64` | S58 (v0.6.10) |
| GPU Polyakov loop | bidirectional with toadStool | S68 (v0.6.13) |
| `Fp64Strategy` | `barracuda::device::driver_profile::Fp64Strategy` | S58 (v0.6.10) |
| `WgslOptimizer` | `barracuda::device::WgslOptimizer` | S42 (v0.5.15) |
| `GpuDriverProfile` | `barracuda::device::driver_profile::GpuDriverProfile` | S42 (v0.5.15) |

---

## Validation

```
cargo test: 711 passed, 0 failed, 6 ignored
cargo clippy --all-targets --all-features: 0 warnings
cargo check: clean compile against toadStool S78
```
