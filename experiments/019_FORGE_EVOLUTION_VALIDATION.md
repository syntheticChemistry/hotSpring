# Experiment 019: Forge Evolution & Comprehensive Validation

**Date**: 2026-02-26
**Gate**: biomeGate (Threadripper 3970X, RTX 3090 24GB + Titan V 12GB HBM2, 256GB DDR4)
**Crate**: hotspring-barracuda v0.6.14, hotspring-forge v0.2.0
**Status**: CPU VALIDATION COMPLETE, FORGE EVOLUTION COMPLETE, pivoting to NPU subtask experiments

---

## Objective

1. Confirm all validation suites pass after Exp 017 (debt audit) and Exp 018 (DF64 benchmark)
2. Evolve metalForge forge crate with DF64 capabilities, streaming pipeline, and PCIe transfer model
3. Validate CPU vs GPU parity for pure Rust math
4. Build local implementations for toadstool absorption

## Part 1: Control Validation — All Green

### Library Tests
- **629 tests**: ALL PASS (19.07s)
- **6 ignored**: GPU-dependent tests (skipped in --lib)

### Integration Tests
- **19 tests**: ALL PASS (26.77s)

### Code Quality
- `cargo fmt --check`: CLEAN
- `cargo clippy -- -D warnings`: 0 warnings (lib + bins)
- `cargo doc --no-deps`: generates 76 doc files

### CPU-Only Validation Suites (20 binaries)

| Suite | Status | Notes |
|-------|--------|-------|
| Special Functions | ✅ PASS | Gamma, Bessel, Laguerre, χ², digamma |
| Linear Algebra | ✅ PASS | LU, QR, SVD, Thomas, eigh |
| Optimizers & Numerics | ✅ PASS | NM, BFGS, RK45, Sobol, Crank-Nicolson |
| MD Forces & Integrators | ✅ PASS | VV, Yukawa, Coulomb, Morse |
| Nuclear EOS (SEMF+HFB) | ✅ PASS | SLy4, UNEDF0, cross-param |
| HFB Verification | ✅ PASS | Zr-94 binding, Rust vs Python |
| Screened Coulomb | ✅ PASS | Stewart-Pyatt, ion-sphere limit |
| HotQCD EOS | ✅ PASS | Thermodynamic consistency |
| Pure Gauge SU(3) | ✅ PASS | CG convergence, plaquette order |
| Production QCD β-scan | ✅ PASS | Determinism, susceptibility peak |
| Dynamical Fermion QCD | ✅ PASS | Mass dependence, plaquette ordering |
| Abelian Higgs | ✅ PASS | ΔH conservation, Rust speedup 77× |
| NPU Quantization | ✅ PASS | Sparsity monotonic, int4>int8>f32 |
| NPU Beyond-SDK | ✅ PASS | Wide FC parity, determinism |
| NPU Pipeline | ✅ PASS | Streaming determinism |
| Lattice QCD + NPU | ✅ PASS | Phase classification |
| Hetero Monitor | ✅ PASS | Refined β_c, adaptive efficiency |
| Spectral Theory | ✅ PASS | W² scaling, γ monotonic |
| Lanczos + 2D Anderson | ✅ PASS | ⟨r⟩ disorder scaling |
| Anderson 3D | ✅ PASS | Dimensional hierarchy, symmetry |
| Hofstadter Butterfly | ✅ PASS | Cantor convergence, gap counting |
| BarraCuda Evolution | ✅ PASS | α=1/5 bands, scaling theory |
| Stanton-Murillo | ✅ PASS | D* ordering + screening (36.5min CPU) |
| Mixed Substrate | ✅ PASS | 9/9 cross-domain (4 ESN + NpuSim) |

### GPU Validation Suites
- Queued after mixed pipeline Stage 3 completes (GPU at 100%)
- 18 GPU suites including CPU/GPU parity, PPPM, CG, SpMV, Dirac

## Part 2: Forge Evolution — v0.2.0

### New Capabilities Added

1. **`Capability::DF64Compute`** — double-float (f32-pair) emulated f64 on FP32 cores
2. **`Capability::PcieTransfer`** — PCIe DMA transfer support
3. **`Capability::StreamingStage`** — streaming pipeline stage participation
4. **`Fp64Rate` enum** — Full (1:1), Half (1:2), Narrow (1:64) classification
5. **`Fp64Strategy` enum** — Native, Hybrid, Concurrent selection
6. **`Pipeline` module** — streaming DAG with Stage, Edge, WorkUnit, ChannelKind

### Strategy Selection (Empirical)

| GPU | FP64 Rate | DF64 | Strategy |
|-----|-----------|------|----------|
| RTX 3090 | Narrow (1:64) | Yes | **Concurrent** |
| Titan V | Half (1:2) | Yes | **Native** |
| RTX 3090 GL | Narrow | No | Native (fallback) |

### Pipeline Topologies

5 predefined QCD topologies for silicon efficiency discovery:

- **A: GPU → NPU → Oracle → CPU** — daisy-chain with Titan V validation
- **B: Parallel GPUs → NPU → CPU** — both GPUs simultaneous, cross-card merge
- **C: NPU-first → GPU → CPU** — ESN predicts before expensive HMC
- **D: CPU-only** — pure Rust baseline
- **E: GPU-only (DF64)** — current Exp 018 benchmark

### Dispatch Profiles Added

- `lattice_cg_df64()` — CG with DF64 extension requirement
- `streaming_compute()` — streaming pipeline stage
- `validation_oracle()` — cross-card validation

### Test Results

- **25 forge tests**: ALL PASS (0.80s)
- `cargo clippy -- -D warnings`: 0 warnings
- `cargo fmt --check`: CLEAN

## Part 3: PCIe Transfer Model

### Current Architecture

```
GPU ──PCIe──▶ CPU ──PCIe──▶ NPU
      16B/traj    24B/traj
      (~650μs)    (~650μs)
```

No GPU↔NPU peer-to-peer DMA available (Akida on PCIe, separate address space).
CPU mediates all cross-device transfers. Latency dominated by PCIe roundtrip,
not compute (~0.7μs ESN inference vs ~1300μs PCIe roundtrip).

### Transfer Budget (per QCD trajectory)

| Transfer | Bytes | Latency | Notes |
|----------|-------|---------|-------|
| CPU→GPU | 0 | 0 | GPU PRNG generates momenta |
| GPU→CPU | 16 | ~650μs | plaquette (8B) + Polyakov (8B) |
| GPU→NPU | 24 | ~1300μs | feature vector via CPU bridge |
| NPU→CPU | 8 | ~650μs | classification label |

### Evolution Path

Phase 1 (current): CPU-mediated, ~2.6ms total PCIe per trajectory
Phase 2 (toadstool): Batch transfers, amortize PCIe overhead across N trajectories
Phase 3 (future): GPU→NPU DMA if hardware supports it (requires IOMMU cooperation)

## Part 4: Mixed Pipeline Status (Stage 3)

The production mixed pipeline continues in background:
- Phase 1: Seed scan complete (3 points × 500 meas) — 4.4 hours
- Phase 2: ESN trained — 0.4ms
- Phase 3: Adaptive steering Round 1 (β=5.5254) — IN PROGRESS

GPU utilization: 100%, 72-74°C.

### Mixed Pipeline Progress

| Phase | Status | Duration | Notes |
|-------|--------|----------|-------|
| Phase 1: Seed Scan | ✅ Complete | 4.4h | 3 points × 500 meas, 7.64s/traj |
| Phase 2: ESN Train | ✅ Complete | 0.4ms | β_c estimate: 5.5051 |
| Phase 3 Round 1 | ✅ Complete | 1.68h | β=5.5254, ⟨P⟩=0.493±0.001, β_c→5.5253 |
| Phase 3 Round 2 | ⏳ Running | ~1.7h est | β=5.4237 |
| Phase 3 Rounds 3-6 | Pending | ~6.7h est | Up to 4 more adaptive rounds |

Stopped after Round 1 complete to pivot to NPU subtask experiments.
Full pipeline would have been ~15-16 hours; seed scan alone was 4.4h (70% of time).

## Part 5: Next Phase — NPU Workload Optimization

The key insight from Stage 3: **70% of compute time is seed scanning** (3 coarse
β points × 500 measurements each). The NPU/ESN operates *after* the expensive GPU
work, steering the *next* point — but cannot reduce the cost of *this* point.

Evolution targets:
1. NPU pre-screening to skip thermalization trajectories
2. Multi-model NPU (requires multi-chip or model swapping)
3. GPU→NPU streaming at natural data production points
4. Combined workload reduction through PCIe-linked NPU+GPU

## Files Changed

| File | Change |
|------|--------|
| `metalForge/forge/src/substrate.rs` | Added `Fp64Rate`, `Fp64Strategy`, `DF64Compute`, `PcieTransfer`, `StreamingStage` |
| `metalForge/forge/src/probe.rs` | DF64 detection, FP64 rate classification, PCIe/streaming caps |
| `metalForge/forge/src/dispatch.rs` | New profiles: `lattice_cg_df64`, `streaming_compute`, `validation_oracle` |
| `metalForge/forge/src/pipeline.rs` | NEW: streaming pipeline DAG with 5 QCD topologies |
| `metalForge/forge/src/bridge.rs` | DF64 capability in `substrate_from_device` |
| `metalForge/forge/src/lib.rs` | Added `pipeline` module |
| `metalForge/forge/examples/inventory.rs` | Full inventory + strategy + topology display |
| `experiments/019_FORGE_EVOLUTION_VALIDATION.md` | This file |

## Provenance

- Exp 017: Debt reduction audit
- Exp 018: DF64 production benchmark
- METALFORGE_STREAMING_PIPELINE_EVOLUTION_FEB26_2026.md: Daisy-chain design handoff
- HOTSPRING_V0614_TOADSTOOL_BARRACUDA_EVOLUTION_HANDOFF_FEB25_2026.md: Absorption roadmap
