# hotSpring v0.6.26 → toadStool/barraCuda Absorption Handoff

**Date:** March 10, 2026
**From:** hotSpring v0.6.26 (842 lib tests, 0 failures, 0 clippy warnings)
**To:** toadStool / barraCuda teams
**Covers:** hotSpring evolution v0.6.18 → v0.6.26 (Dec 2025 — Mar 2026)
**License:** AGPL-3.0-only

## Executive Summary

hotSpring has completed 50 experiments spanning nuclear structure, molecular
dynamics, lattice QCD, spectral theory, transport coefficients, and
heterogeneous hardware orchestration. This handoff documents what hotSpring
has built on top of barraCuda that is ready for upstream absorption, plus
critical hardware findings that should inform toadStool/barraCuda evolution.

Key findings for upstream:

- **NVVM device poisoning** — NVIDIA's proprietary NVVM permanently kills
  the wgpu device on specific f64 shader patterns. hotSpring discovered,
  gated, and built a full workaround via coralReef sovereign bypass.
- **Self-routing precision brain** — hardware-probed, per-tier calibration
  with automatic routing across F32/F64/DF64/F64Precise. Upstream could
  absorb this as a first-class barraCuda primitive.
- **Heterogeneous dual-GPU patterns** — DevicePair, WorkloadPlanner,
  dual dispatch with PCIe transfer cost modeling across Titan V + RTX 3090.
- **170+ tolerance constants** — centralized, physically justified, with
  DOI provenance. Pattern ready for upstream adoption.
- **Sovereign compile validation** — 45/46 hotSpring WGSL shaders compile
  to native SM70/SM86 SASS via coralReef Iter 29. 12/12 NVVM bypass patterns
  confirmed across 6 GPU targets.

---

## Part 1: Absorption Candidates

### P1 — Precision System (high value for all springs)

| Module | Location | Tests | Description |
|--------|----------|-------|-------------|
| `HardwareCalibration` | `hardware_calibration.rs` | 8 | Per-tier safe probing of GPU capabilities. Probes F32→F64→DF64→F64Precise, detects NVVM risk, tracks sovereign bypass. |
| `PrecisionBrain` | `precision_brain.rs` | 6 | Data-driven domain→tier routing. 7 physics domains auto-routed based on calibration. Sovereign-aware. |
| `PrecisionRoutingAdvice` | `precision_routing.rs` | 4 | Capability-aware shader compilation advice. |
| `precision_eval` | `precision_eval.rs` | — | Per-shader precision/throughput profiler across all 4 tiers. |
| `bench_precision_tiers` | `bin/bench_precision_tiers.rs` | — | 4-tier evaluation harness (F32/F64/DF64/F64Precise). |

**Why upstream should absorb:** Every spring needs precision routing. The
current pattern (probe → calibrate → route) is general-purpose. barraCuda
could offer `PrecisionBrain` as a first-class API so springs don't each
reinvent hardware probing.

**toadStool action:** Consider absorbing `HardwareCalibration` into
`barracuda::device` alongside `GpuDriverProfile`. The calibration data
(which tiers are safe, which need sovereign bypass) is hardware-universal.

### P1 — NVVM Poisoning Knowledge

hotSpring discovered that NVIDIA's proprietary NVVM compiler permanently
poisons the wgpu device when compiling:

1. **f64 transcendentals** (exp, log, pow on f64 values)
2. **DF64 pipeline with transcendentals** (double-float emulation + exp/log)
3. **F64Precise no-FMA** (f64 with `FmaPolicy::Separate`)

These patterns cause an unrecoverable device loss on RTX 3090 (proprietary
driver). The Titan V on NVK is immune.

**What hotSpring built:**
- `nvvm_transcendental_risk` flag in `HardwareCalibration`
- Conditional shader skip in probe sequence (never dispatch poisoning shaders)
- coralReef sovereign bypass detection and routing

**toadStool action:** barraCuda's `GpuDriverProfile` should track
`nvvm_transcendental_risk` natively. Any spring dispatching f64
transcendentals on NVIDIA proprietary needs this guard.

### P2 — Heterogeneous GPU Dispatch

| Module | Location | Tests | Description |
|--------|----------|-------|-------------|
| `DevicePair` | `device_pair.rs` | 4 | Heterogeneous dual-GPU pairing with PCIe bandwidth estimation |
| `WorkloadPlanner` | `workload_planner.rs` | 3 | Domain-aware workload splitting (which card gets what) |
| `dual_dispatch` | `dual_dispatch.rs` | 6 | Cooperative dispatch: Split BCS (2.2×), Split HMC, Redundant profiling |
| `transfer_eval` | `transfer_eval.rs` | 2 | PCIe transfer cost profiler |

**Why upstream should absorb:** toadStool S142 introduced `PcieTransport`
and `ResourceOrchestrator`. hotSpring's `DevicePair`/`WorkloadPlanner`
validate the concepts with real physics workloads. The measured PCIe
bandwidth (1.2 GB/s CPU-mediated between Titan V and RTX 3090) and
workload split ratios are useful data for toadStool's transport layer.

**toadStool action:** When `PcieTransport` API stabilizes, hotSpring will
migrate from spec-sheet bandwidth estimates to runtime-measured values.
Consider absorbing hotSpring's workload split heuristics into
`ResourceOrchestrator`.

### P2 — Tolerance Pattern

| Module | Location | Constants | Description |
|--------|----------|-----------|-------------|
| `tolerances/core.rs` | Core thresholds | ~30 | GPU parity, float comparison, convergence |
| `tolerances/physics.rs` | Physics thresholds | ~40 | Nuclear binding, EOS, dielectric |
| `tolerances/lattice.rs` | Lattice thresholds | ~50 | Plaquette, CG residual, flow scale |
| `tolerances/md.rs` | MD thresholds | ~30 | Energy drift, RDF, VACF, transport |
| `tolerances/npu.rs` | NPU thresholds | ~20 | Quantization parity, pipeline accuracy |

**Why upstream should absorb:** The centralized tolerance pattern (named
constants with physical justification + DOI provenance) prevents magic
numbers. barraCuda could offer a `tolerances` module pattern that springs
fill with domain-specific values.

### P3 — Physics Modules (domain-specific, lower absorption priority)

These are hotSpring-specific physics implementations. They use barraCuda
primitives but are unlikely candidates for upstream absorption (too
domain-specific). Listed for completeness:

| Module | Description | Shaders |
|--------|-------------|---------|
| `physics/hfb*` | Hartree-Fock-Bogoliubov nuclear structure (spherical + deformed) | 10 |
| `physics/semf.rs` | Semi-empirical mass formula | 2 |
| `physics/bcs_gpu.rs` | BCS superconductivity on GPU | 1 |
| `physics/dielectric*` | Mermin dielectric function + multicomponent | 2 |
| `physics/kinetic_fluid*` | BGK kinetic-fluid equations | 2 |
| `physics/screened_coulomb.rs` | Screened Coulomb eigenvalues (Sturm) | 0 |
| `lattice/gradient_flow*` | Wilson gradient flow with LSCFRK integrators | 2 |
| `lattice/pseudofermion*` | Pseudofermion HMC for dynamical QCD | 4 |
| `lattice/rhmc.rs` | Rational HMC for fractional flavors | 0 |
| `md/reservoir*` | ESN reservoir for transport prediction | 2 |

---

## Part 2: Hardware Findings

### GPU Precision Landscape (biomeGate)

| GPU | Driver | F32 | F64 | DF64 | F64Precise | NVVM Risk |
|-----|--------|-----|-----|------|------------|-----------|
| Titan V | NVK (Mesa) | Full | Full (1:2) | Full | Full | None |
| RTX 3090 | NVIDIA proprietary | Full | Throttled (1:64) | Arith only | Arith only | **Yes** — transcendentals poison device |
| AKD1000 | akida-driver | N/A | N/A | N/A | N/A | N/A (NPU) |

### Key Hardware Insights

1. **NVK is the safest NVIDIA path for f64** — no NVVM, no proprietary
   driver quirks, genuine 1:2 throughput on Volta. Mesa 25.1.5.
2. **DF64 on RTX 3090 delivers 3.24 TFLOPS** at 14-digit precision —
   9.9× native f64 throughput. This is the workhorse for production.
3. **Titan V + RTX 3090 cooperation** — precision (Titan V) + throughput
   (RTX 3090) split validated in production β-scans.
4. **PCIe CPU-mediated transfer** — 1.2 GB/s between cards. Split
   workloads viable when per-card compute time >> transfer time.
5. **coralReef sovereign bypass** — compiles the exact NVVM-poisoning
   shaders to native SASS. 45/46 hotSpring shaders compile. Dispatch
   pending NVIDIA DRM maturation.

### Sovereign Compilation Results (coralReef Iter 29)

| Test Suite | Result | Notes |
|------------|--------|-------|
| hotSpring sovereign compile (SM70+SM86) | **45/46** | `complex_f64` expected fail (utility include) |
| NVVM bypass patterns | **12/12** | All 3 poisoning patterns × 6 GPU targets |
| Spring absorption wave 3 | 9/14 | AMD discriminant + SM70 vec3/log2 gaps |
| Spring absorption wave 1+2 | 17/24 | RDNA2 f64 literal constants |

---

## Part 3: Evolution Recommendations

### For barraCuda

| Priority | Recommendation | Impact |
|----------|----------------|--------|
| P1 | Absorb `HardwareCalibration` into `barracuda::device` | All springs get safe per-tier GPU probing |
| P1 | Add `nvvm_transcendental_risk` to `GpuDriverProfile` | Prevents device poisoning across all springs |
| P2 | Consider `PrecisionBrain` as `barracuda::precision::Router` | Springs get domain→tier routing out of the box |
| P2 | Absorb tolerance pattern (`centralized.rs` with provenance) | Prevents magic numbers across springs |
| P3 | `tridiagonal_ql` eigensolver for Lanczos completion | hotSpring may adopt for HFB/Lanczos termination |

### For toadStool

| Priority | Recommendation | Impact |
|----------|----------------|--------|
| P1 | Stabilize `PcieTransport` API | hotSpring ready to adopt measured bandwidth |
| P1 | Expose `nvvm_transcendental_risk` in hardware discovery | Runtime-safe probing needs this data |
| P2 | Absorb `DevicePair` workload split heuristics | Multi-GPU dispatch for all springs |
| P2 | `ResourceOrchestrator` multi-spring integration | biomeGate needs cross-spring GPU allocation |
| P3 | `transport.open/stream/status` JSON-RPC client | Cross-primal GPU sharing |

---

## Part 4: Metrics

| Metric | v0.6.18 (Dec) | v0.6.26 (Mar 10) |
|--------|---------------|-------------------|
| lib tests | 685 | 842 |
| binaries | 85+ | 111+ |
| WGSL shaders | 84 | 84 |
| experiments | 031 | 050 |
| clippy warnings | 0 | 0 |
| unsafe blocks | 0 | 0 |
| barraCuda pin | `cdd748d` | `83aa08a` |
| toadStool sync | S93 | S142 |
| coralReef sync | — | Iter 29 |
| sovereign compile | — | 45/46 SM70+SM86 |
| NVVM bypass | — | 12/12 |
| precision tiers | — | 4 (F32/F64/DF64/F64Precise) |

---

## Part 5: Code Locations

All code referenced in this handoff lives in `hotSpring/barracuda/src/`:

```
hardware_calibration.rs    — HardwareCalibration struct + probing
precision_brain.rs         — PrecisionBrain router + sovereign detection
precision_routing.rs       — PrecisionRoutingAdvice
precision_eval.rs          — Per-shader evaluation harness
device_pair.rs             — Heterogeneous dual-GPU pairing
workload_planner.rs        — Domain-aware workload splitting
dual_dispatch.rs           — Cooperative dispatch patterns
transfer_eval.rs           — PCIe transfer cost profiler
tolerances/                — ~170 centralized validation thresholds
provenance.rs              — DOI + Python origin provenance tracking
```

Shaders: `physics/shaders/` (18), `lattice/shaders/` (36), `md/shaders/` (21).
Experiments: `experiments/001-050`.
Full changelog: `barracuda/CHANGELOG.md`.

---

*hotSpring v0.6.26 — 50 experiments, 842 tests, 84 shaders, 3 GPUs + 1 NPU.
The precision brain routes. The sovereign pipeline compiles. The fungus
absorbs what the biome proves.*
