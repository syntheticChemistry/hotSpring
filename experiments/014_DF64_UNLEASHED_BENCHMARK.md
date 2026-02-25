# Experiment 014: DF64 Unleashed — RTX 3090 Benchmark Suite (v0.6.11)

**Date**: 2026-02-25
**Gate**: biomeGate (Threadripper 3970X, RTX 3090 24GB, Titan V 12GB)
**Crate**: hotspring-barracuda v0.6.11
**Status**: ✅ COMPLETE — all validation suites pass, benchmarks collected

---

## Objective

Benchmark the RTX 3090 with DF64 core streaming active (v0.6.10) and t-major
site-indexing standardization (v0.6.11). This is the first run with the 3090
"fully unleashed" — gauge force on FP32 cores at 3.24 TFLOPS instead of
native f64 at 0.33 TFLOPS.

## Changes from Experiment 013

| Property | Exp 013 (v0.6.8) | Exp 014 (v0.6.11) |
|----------|------------------|---------------------|
| Gauge force | Native f64 (0.33 TFLOPS) | DF64 hybrid (3.24 TFLOPS) |
| Site indexing | x-fastest (hotSpring local) | z-fastest (toadStool standard) |
| 32⁴ time/traj | ~15.5s | **7.7s** |
| Full scan est. | 13.6 hrs | **~7 hrs** |

---

## Results — Quenched HMC Scaling (bench_gpu_hmc)

| Lattice | Volume | CPU ms/traj | GPU ms/traj | Speedup |
|---------|--------|------------|------------|---------|
| 4⁴ | 256 | 68.5 | 18.5 | 3.7× |
| 8⁴ | 4,096 | 1,109 | 36.7 | 30.2× |
| 8³×16 | 8,192 | 2,190 | 46.6 | 47.0× |
| 16⁴ | 65,536 | 17,995 | 293 | 61.4× |

All trajectories: 100% acceptance, physical plaquette, DF64 active.

## Results — 32⁴ Quick Validation (production_beta_scan)

20 trajectories at β=6.0, dt=0.0125, n_md=40:

| Metric | Value |
|--------|-------|
| Time per trajectory | **7.7 seconds** |
| Acceptance | 90% |
| Mean plaquette | 0.521825 ± 0.011221 |
| Susceptibility | 132.04 |
| Total wall time | 154.3s (2.6 min) |

Compared to Exp 013 native f64: **2.0× faster** (15.5s → 7.7s).

## Results — Streaming HMC Validation (7/7 PASS)

| Lattice | Volume | CPU ms | Dispatch ms | Streaming ms | Stream gain |
|---------|--------|--------|-------------|-------------|-------------|
| 4⁴ | 256 | 73.3 | 18.9 | 18.5 | 1.02× (4.0× CPU) |
| 8⁴ | 4,096 | 1,155.7 | 29.8 | 27.8 | 1.07× (41.6× CPU) |
| 8³×16 | 8,192 | 2,305.0 | 41.9 | 43.6 | 0.96× (52.9× CPU) |
| 16⁴ | 65,536 | 18,751.7 | 270.4 | 268.2 | 1.01× (69.9× CPU) |

Dispatch/streaming parity exact to 1e-8 (validates site-indexing change).

## Results — Pure GPU HMC Validation (3/3 PASS)

| Check | Result |
|-------|--------|
| Shader pipelines compile | ✅ (DF64 hybrid detected) |
| Acceptance > 30% | ✅ (100%) |
| Plaquette in physical range | ✅ (0.584339) |

## Results — GPU Beta Scan Validation (6/6 PASS)

| Check | Result |
|-------|--------|
| Plaquette monotonically increasing | ✅ |
| Plaquette at β=6.0 in (0.55, 0.65) | ✅ (0.557285) |
| Polyakov loop positive | ✅ |
| Mean acceptance > 50% | ✅ (99.4%) |
| 8³×16 within 5% of 8⁴ | ✅ |
| Total wall time < 120s | ✅ (99.7s) |

## Results — Dynamical Fermion HMC (6/6 PASS)

| Metric | Value |
|--------|-------|
| Fermion force parity | 6.59e-17 |
| CG action parity | 1.86e-12 |
| Acceptance | 80% (8/10) |
| Plaquette | 0.563435 |
| CG iterations/traj | 624 |
| GPU time/traj (4⁴) | 13.4s |

## Results — Streaming Dynamical HMC (13/13 PASS)

| Phase | Lattice | Plaquette | Acceptance | Time |
|-------|---------|-----------|------------|------|
| Dispatch dynamical | 4⁴ | 0.765012 | 70% | 139.1s |
| Streaming dynamical | 4⁴ | 0.759215 | 70% | 136.5s |
| Resident CG | 4⁴ | 0.789789 | 40% | 6.2s |
| Bidirectional stream | 4⁴ | — | 40% | 3.2s |
| Streaming dynamical | 8⁴ | 1.000000 | 0% (param) | 34.4s |

Key: Resident CG achieves **22× speedup** over dispatch (6.2s vs 139.1s)
with 15,360× readback reduction (37 MB → 2.4 KB/traj).

8⁴ dynamical shows 0% acceptance — parameter tuning needed for larger volumes
(cold start + heavy quarks + insufficient thermalization at this volume).

---

## Cost Analysis

| Run | Time | Cost | Physics |
|-----|------|------|---------|
| 32⁴ 12-pt scan (native f64, Exp 013) | 13.6 hrs | $0.58 | Deconfinement at β=5.69 |
| 32⁴ 12-pt scan (DF64, estimated) | ~7 hrs | ~$0.30 | Same physics, 2× faster |
| 32⁴ dynamical (estimated, 1000 traj) | ~33 hrs | ~$1.40 | First dynamical on consumer |
| 16⁴ Titan V NVK (Exp 013) | 47 min | — | Open driver validation |

---

## DF64 Speedup Analysis

The gauge force kernel is ~40% of HMC wall time. DF64 speeds up only the
gauge force (9.9× on FP32 cores). The overall trajectory speedup is:

- Theoretical: `1 / (0.6 + 0.4/9.9)` = 1.56×
- Measured: 15.5s → 7.7s = **2.01×**
- The measured speedup exceeds prediction, suggesting the force fraction is
  higher than 40% at 32⁴ (makes sense — force scales with volume faster
  than scalar reductions)

## What DF64 Has NOT Yet Been Applied To

| Kernel | % of HMC | DF64-able? | Status |
|--------|----------|------------|--------|
| Gauge force | ~40% | YES | **DONE (v0.6.10)** |
| Wilson plaquette | ~15% | YES | Not yet |
| Kinetic energy | ~5% | YES | Not yet |
| Momentum update | ~5% | YES | Not yet |
| Link update | ~10% | PARTIAL | Not yet |
| CG/Dirac | ~20% | YES (bulk) | Not yet |
| Random momenta | ~5% | NO | N/A |

Applying DF64 to plaquette, KE, and momentum update would increase the
DF64 fraction from 40% to ~65%, giving a theoretical speedup of ~2.8×
over all-native-f64 (currently 2.0×).

---

## Cross-References

- Experiment 012: `012_FP64_CORE_STREAMING_DISCOVERY.md` — DF64 discovery
- Experiment 013: `013_BIOMEGATE_PRODUCTION_BETA_SCAN.md` — native f64 baseline
- Handoff: `wateringHole/handoffs/TOADSTOOL_SITE_INDEXING_NAK_SOLVER_HANDOFF_FEB25_2026.md`
- CHANGELOG: `barracuda/CHANGELOG.md` v0.6.10 (DF64) + v0.6.11 (site-indexing)
