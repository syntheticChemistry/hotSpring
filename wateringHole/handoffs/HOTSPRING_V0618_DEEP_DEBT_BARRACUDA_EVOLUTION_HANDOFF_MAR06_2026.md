# hotSpring v0.6.18 → toadStool/barraCuda — Deep Debt Resolution & Evolution Handoff

**Date:** 2026-03-06  
**From:** hotSpring v0.6.18 (685 lib tests, 47 validation suites, 0 clippy warnings)  
**To:** toadStool (S93+) / barraCuda (v0.3.3+)  
**License:** AGPL-3.0-only  

## Executive Summary

- Deep debt audit and resolution across 226 .rs files
- NautilusShell API sync (brain.rs): `output_dim`, `population_respond`, `retrain_readout` evolved
- 5 oversized files refactored into 15 focused modules (all <1000 lines)
- Zero clippy warnings (lib, pedantic+nursery), zero unsafe, zero TODO/FIXME
- Production error handling: unwrap→Result in 9 sites, error-swallowing fixed
- Brain B2 (memory pressure) and D1 (force anomaly) evolved from placeholder to runtime
- Stream-first I/O for trajectory log loading
- Dependency alignment: pollster 0.3, bytemuck 1.25, tokio 1.50

## Part I: barraCuda API Evolution Findings

### NautilusShell API Drift
- `ShellConfig` removed `output_dim` field; readout must be reconstructed with correct dimensions
- `NautilusShell` removed `population_respond()` → use `population.respond_all()`
- `NautilusShell` removed `retrain_readout()` → use `readout.train()`
- **Recommendation:** Add `ShellConfig::with_output_dim(n)` convenience or document that `readout` must be replaced for non-default output dimensions

### NpuResponse Field Naming
- `NpuResponse::RejectPrediction._confidence` was underscore-prefixed (clippy violation)
- Renamed to `confidence` — barraCuda should adopt same convention

## Part II: Absorption Candidates (P0)

| Item | Location | Rationale |
|------|----------|-----------|
| `derive_lscfrk3` | `lattice/gradient_flow.rs` | Algebraic derivation of Lüscher-style integrators; eliminates magic numbers |
| `lscfrk_step` | Same | Generic LSCFRK step applicable to any gradient flow |
| `esn_baseline` | `md/reservoir/esn_baseline.rs` | Reusable ESN training/evaluation harness with synthetic datasets |
| `sarkas_harness` | `md/sarkas_harness.rs` | Brain-steered MD case runner with N-scaling |
| `DynamicalMixedConfig` | `production/dynamical_mixed_pipeline/` | Validated multi-substrate scan orchestration |
| `memory_pressure` | `md/brain.rs` | Algorithm-aware GPU memory estimate (AllPairs/CellList/Verlet) |
| `force_anomaly` | `md/brain.rs` | Energy deviation anomaly detection (10σ threshold) |

## Part III: Absorption Candidates (P1)

| Item | Location | Rationale |
|------|----------|-----------|
| `sum_reduce_f64.wgsl` unification | `lattice/shaders/` | hotSpring keeps local multi-pass encode-only reduce; upstream `ReduceScalarPipeline` needs `encode_reduce_to_buffer()` for GPU-resident CG |
| `su3_math_f64.wgsl` | `lattice/shaders/` | Naga composition workaround; should be absorbed when naga supports modular includes |
| RHMC shaders | `lattice/rhmc.rs` | P0 evolution target; needs WGSL rational approximation + multi-shift CG |
| Hasenbusch preconditioning | `gpu_hmc/hasenbusch.rs` | P1; GPU preconditioning for dynamical HMC |
| Deformed HFB wiring | `physics/hfb_deformed_gpu/` | 5 WGSL shaders exist; H-build pipeline not wired |

## Part IV: Codebase State

| Metric | Value |
|--------|-------|
| Lib tests | 685 (0 fail, 6 ignored) |
| Validation binaries | 47 (39 suites) |
| WGSL shaders | 81 |
| Clippy (lib) | 0 warnings |
| Unsafe blocks | 0 |
| TODO/FIXME | 0 |
| Max file size | 995 lines |
| Coverage | 51% lines, 63% functions (production/GPU modules at 0% — hardware-dependent) |
| SPDX | 100% AGPL-3.0-only |

## Part V: Dependency Alignment

| Dependency | hotSpring | barraCuda | Notes |
|------------|-----------|-----------|-------|
| wgpu | 28 | 28 | Aligned |
| pollster | 0.3 | 0.3 | Aligned |
| bytemuck | 1.25 | 1.25 | Aligned |
| tokio | 1.50 | 1.50 | Aligned |
| Edition | 2021 | 2024 | hotSpring on 2021; consider migration |

## Part VI: Refactoring Map

| Original | Lines | Split Into |
|----------|-------|------------|
| npu_worker.rs | 1,562 | mod.rs (995) + messages.rs + head_confidence.rs + training.rs + trajectory_input.rs + checkpoint.rs |
| simulation.rs | 1,066 | mod.rs (455) + types.rs + init.rs + verlet.rs |
| production_dynamical_mixed.rs | 1,299 | binary (351) + pipeline/mod.rs + pipeline/single_beta.rs |
| esn_baseline_validation.rs | 1,088 | binary (415) + esn_baseline.rs (716) |
| sarkas_gpu.rs | 1,085 | binary (866) + sarkas_harness.rs (217) |

## Part VII: Open Items for barraCuda

- [ ] Add `encode_reduce_to_buffer()` to `ReduceScalarPipeline` for GPU-resident CG unification
- [ ] Consider `ShellConfig::with_output_dim()` convenience method
- [ ] GPU multi-shift CG for RHMC (hotSpring has CPU-only `rhmc.rs`)
- [ ] Deformed HFB pipeline integration (5 shaders waiting)
- [ ] Matrix exponentiation shader for gradient flow (currently CPU `exp(-epsilon*Z)`)
- [ ] Edition 2024 migration guide for Springs

## Part VIII: Evolution Pipeline

```text
hotSpring (Write)
  ↓ validated shaders + patterns
toadStool (Absorb)
  ↓ upstream primitives
barraCuda (Lean)
  ↓ GPU-native ops
Springs (Use)
```

---
*hotSpring v0.6.18 — AGPL-3.0-only*
