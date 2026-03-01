# Experiment 029: NPU-Steered Dynamical Production

**Status:** IN PROGRESS (started 2026-03-01)
**Date:** March 1, 2026
**Depends on:** Exp 024 (parameter sweep), Exp 028 (brain architecture)
**License:** AGPL-3.0-only

---

## Motivation

Exp 024 generated 2,400 trajectories across 17 beta points but used a fixed
beta schedule. Exp 028 built the 4-layer brain architecture (RTX 3090 motor,
Titan V pre-motor, CPU cortex, NPU cerebellum) and proved concurrent
multi-substrate operation. Exp 029 combines both: the brain architecture
runs a production dynamical HMC scan where the NPU steers beta selection,
Titan V pre-thermalizes the next point, and the CPU runs Anderson proxies
concurrently.

This is the first production run where the NPU controls the measurement
schedule rather than following a fixed script.

## Configuration

| Parameter | Value |
|-----------|-------|
| Lattice | 8⁴ (4,096 sites) |
| Mass | 0.1 (1 staggered flavor) |
| β values (seed) | 4.5, 5.25, 5.69, 6.5 |
| dt | 0.01 |
| n_md | 100 |
| Trajectory length | 1.0 |
| CG tolerance | 1e-8 |
| CG max iterations | 65,000 |
| CG check interval | 100 |
| Quenched pre-therm | 10 (NPU-predicted) |
| Dynamical thermalization | 20 max |
| Measurements per β | 65 |
| Seed | 2029 |
| FP64 strategy | Hybrid DF64 (force + plaquette + KE) |

## Hardware

| Substrate | Role | Details |
|-----------|------|---------|
| RTX 3090 (NVK) | Motor cortex — primary CG/HMC | 24 GB, GA102, PCIe Gen3 x16 |
| Titan V (NVK) | Pre-motor — quenched pre-therm | 12 GB, GV100, concurrent thread |
| CPU (TR 3970X) | Cortex — Anderson 3D proxy | 32 cores, 256 GB DDR4 |
| AKD1000 (PCIe) | Cerebellum — 15-head steering | Hardware NPU, 30 mW inference |

## Bootstrap

ESN weights loaded from Exp 028 (`results/exp028_brain_weights.json`).
Training data combined from:
- `results/exp024_production_8x8.jsonl` → 17 beta points, 1,071 trajectories
- `results/exp028_brain_production_8x8.jsonl` → 8 beta points, 760 trajectories
- Combined: 25 unique beta points for ESN bootstrap

## Brain Architecture Layers Active

**Layer 1 (NPU Cerebellum):** 15-head worker thread. 4 pre-GPU heads for
beta screening and CG estimation. 5 during-trajectory heads. 3 post-trajectory
heads. 3 proxy heads. 1 brain-coordination head. NPU priority ranking
determines beta measurement order.

**Layer 2 (Titan V Pre-Motor):** Concurrent quenched pre-thermalization.
While the 3090 measures at β_i, the Titan V runs 10 quenched HMC trajectories
for β_{i+1}. Warm configs transferred via CPU (~2.4 MB, <1 ms).

**Layer 3 (CPU Cortex):** Anderson 3D proxy pipeline. Computes ⟨r⟩ (level
spacing ratio) and |λ|_min (minimum eigenvalue) for each beta point. Features
fed to NPU for physics-informed CG prediction.

**Layer 4 (Attention/Interrupt):** NPU monitors all streams. GREEN/YELLOW/RED
state machine for anomaly detection.

## Results (Partial — Run Active)

| β | Phase | Trajectories | ⟨P⟩ | ⟨CG⟩ | Acceptance | ⟨r⟩ | \|λ\|_min | Wall Time |
|---|-------|-------------|------|------|------------|-----|---------|-----------|
| 4.5000 | Complete | 65 meas | 0.3343 ± 0.0037 | 61,057 | 55% | 0.482 | 0.048 | 5,514s |
| 5.2500 | Complete | 65 meas | 0.4647 ± 0.0075 | 60,400 | 62% | 0.537 | 0.001 | 5,407s |
| 5.6900 | In progress | — | — | — | — | 0.533 | 0.013 | — |
| 6.5000 | Pending | — | — | — | — | — | — | — |

## NPU Steering Decisions

Initial priority ranking (highest = measured first):
1. β = 4.5000 (priority 0.745) — highest NPU uncertainty
2. β = 5.2500 (priority 0.363)
3. β = 5.6900 (priority -0.000) — near critical point, well-constrained
4. β = 6.5000 (priority -1.041) — deconfined, least novel

NPU CG estimate for β=4.5000: ~78,585 iterations (actual: 61,057 — 22% overestimate, conservative).

## Anderson Proxy Results

| β | ⟨r⟩ | |λ|_min | Regime | Compute Time |
|---|-----|---------|--------|--------------|
| 4.5000 | 0.482 | 0.048 | Extended (GOE-like) | 248 ms |
| 5.2500 | 0.537 | 0.001 | Extended (near mobility edge) | 210 ms |
| 5.6900 | 0.533 | 0.013 | Extended | 213 ms |

⟨r⟩ near 0.53 (GOE value) at all measured betas — consistent with
extended/delocalized phase at 8⁴ with m=0.1 dynamical fermions.
|λ|_min dropping toward zero at β=5.25 is notable: may signal
proximity to a spectral gap closing.

## Titan V Pre-Thermalization Log

| Target β | Quenched Trajs | Final P | Wall Time |
|----------|---------------|---------|-----------|
| 5.2500 | 10 | 0.4406 | 57.4s |
| 5.6900 | 10 | 0.5115 | 56.1s |

The Titan V successfully pre-thermalizes each point concurrently with
the 3090's measurement phase. Config transfer is sub-millisecond.
One discard occurred: Titan V had cached β=5.25 but β=5.69 was needed.

## Key Observations

1. **Brain architecture works**: All 4 layers operate concurrently.
   Titan V, CPU cortex, and NPU all produce useful results during
   the 3090's CG solve.

2. **NPU priority ordering is reasonable**: High-uncertainty confined
   points measured first, well-constrained deconfined points deferred.

3. **CG cost is flat**: ~60,400–61,057 across β=4.5–5.25 at 8⁴. This
   suggests the CG solver is hitting the check_interval ceiling
   (65,000 max ÷ 100 check) rather than converging naturally — the
   lattice may be too small for meaningful CG variation.

4. **Anderson proxy is fast**: <250 ms per beta point, negligible
   compared to the ~5,400s HMC measurement. The ⟨r⟩ values are
   physics-informative (GOE statistics in the extended regime).

## Connections

- **Exp 024**: Provides bootstrap training data (17 β points)
- **Exp 028**: Brain architecture spec and NVK deadlock fix
- **Nautilus Shell** (`bingoCube/nautilus`): Planned integration for
  evolutionary reservoir alongside ESN. See `specs/BIOMEGATE_BRAIN_ARCHITECTURE.md`.
- **Exp 025** (planned): 16⁴ validation for GPU saturation
- **Exp 026** (planned): 4D Anderson proxy for deeper spectral features

## Files

| File | Purpose |
|------|---------|
| `results/exp029_npu_steering_8x8.jsonl` | Per-trajectory data |
| `results/exp029_npu_steering.log` | Full console output |
| `results/exp028_brain_weights.json` | Bootstrap ESN weights |
| `barracuda/src/bin/production_dynamical_mixed.rs` | Production binary |
| `specs/BIOMEGATE_BRAIN_ARCHITECTURE.md` | Brain architecture spec |
