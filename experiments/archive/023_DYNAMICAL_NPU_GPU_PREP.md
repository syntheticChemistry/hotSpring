# Experiment 023: Dynamical NPU GPU-Prep + 11-Head Offload

**Date**: February 27, 2026  
**Binary**: `production_dynamical_mixed`  
**Hardware**: NVIDIA GeForce RTX 3090 (DF64), BrainChip AKD1000 (live PCIe NPU)  
**Crate**: hotspring-barracuda v0.6.15 (11-head ESN, adaptive CG, quenched monitoring)  
**Status**: Code complete — pending first production run

---

## Objective

Every function the NPU handles is GPU time reclaimed. Exp 022 proved the
NPU-in-the-loop architecture (4 placement points, 9-head ESN, 60% therm
early-exit). This experiment extends NPU utilization into the **quenched phase**
and GPU preparation — areas where the GPU previously ran blind.

The NPU is stupidly efficient at inference (2.8 µs/step on AKD1000). The cost
is amortized training, which is already the hallmark of data science. The
question: can the NPU orchestrate the GPU rather than just observe it?

## Changes from Experiment 022

| Property | Exp 022 (v0.6.14) | Exp 023 (v0.6.15) |
|----------|--------------------|--------------------|
| NPU heads | 9 | **11** (added QUENCHED_LENGTH, QUENCHED_THERM) |
| Quenched phase | Fixed length, unmonitored | **NPU-predicted length + early-exit monitoring** |
| GPU prep | None | **Pipelined NPU predictions during GPU upload** |
| CG tuning | Fixed check_interval | **NPU CG estimate → adaptive check_interval** |
| β steering | Post-scan recommendation | **Intra-scan adaptive gap-filling** |
| Architecture | NPU observes GPU | **NPU orchestrates GPU** |
| Quenched→dynamical | Fixed pretherm count | **NPU learns optimal allocation per (β,mass,lattice)** |

---

## Architecture: NPU as GPU Conductor

```
PRE-GPU (pipelined)          DURING QUENCHED (monitored)    DURING DYNAMICAL             POST + STEERING
──────────────────           ───────────────────────────    ────────────────             ───────────────
H1: β priority               H11: quenched therm detect    H3: therm early-exit         H7: quality score
H2: param suggest (dt,n_md)                                H4: reject prediction        H8: anomaly detect
H6: CG estimate → check_int                                H5: phase classification     H9: next-run recommend
H10: quenched length predict                                                            Adaptive: gap-fill steer
```

### Key Innovation: Pipelined NPU During GPU Upload

Before the quenched phase starts, the NPU fires three predictions simultaneously:
1. `PredictQuenchedLength` — optimal number of quenched pre-therm steps
2. `SuggestParameters` — dt, n_md for the dynamical phase
3. `PredictCgIters` — CG iteration estimate → adaptive check_interval

While the GPU uploads the lattice and runs quenched trajectories, the NPU
computes in parallel. Results are collected after the quenched phase completes —
zero GPU stall, free NPU utilization.

### Key Innovation: Quenched Phase Monitoring

Previously the quenched phase ran for a fixed number of steps regardless of
convergence. Near β_c more steps are needed; far from it, most are wasted.
Head 11 (QUENCHED_THERM) monitors plaquette convergence every 10 steps and
triggers early exit when the quenched phase has served its purpose.

### Key Innovation: Adaptive CG Check Interval

The NPU CG estimate determines how frequently the CG solver checks its residual:
- Easy solves (< 200 iters predicted): check every 20 iterations (fewer stalls)
- Medium (200–1000): check every 10
- Hard (> 1000): check every 5 (catch divergence early)

This is a direct GPU throughput gain — fewer residual readbacks per CG solve.

### Key Innovation: Intra-Scan Adaptive Steering

After 3+ measured β points, the NPU evaluates 40 candidate β values by
priority × uncertainty. If it finds a gap worth filling that isn't already
in the queue, it inserts a new β point into the live scan. The scan
self-heals — if the initial grid misses an interesting region, the NPU
catches it mid-run.

---

## New Heads

| Head | Index | Phase | Input | Output | Target |
|------|-------|-------|-------|--------|--------|
| QUENCHED_LENGTH | 9 | Pre-GPU | (β, plaq_meta, mass, lattice/32, 0) | 0–1 → 0–200 steps | Actual steps used / 200 |
| QUENCHED_THERM | 10 | During quenched | (β, mean_plaq, σ_plaq, n/100, 1.0) | >0.5 = converged | Early-exit flag |

Both heads share the same ESN reservoir as the existing 9 heads. On AKD1000,
Akida's SkipDMA merges all FC layers into one hardware pass — 11 outputs
add zero latency over 9.

---

## Expected vs Exp 022 Predictions

| Metric | Exp 022 (9-head) | Exp 023 (11-head) Predicted | Basis |
|--------|------------------|-----------------------------|-------|
| Quenched steps wasted | 100% of budget | **30–50% saved** | NPU predicts allocation |
| Quenched early-exits | 0% (unmonitored) | **40–60%** | Same ESN accuracy as dyn therm |
| Therm early-exits (dyn) | 60% at 8⁴ | **60–70%** (warm ESN) | Cross-run bootstrap |
| CG check overhead | Fixed 10 | **Adaptive 5–20** | ~5% GPU throughput gain |
| β gap insertions | 0 | **1–3 per scan** | Uncertainty-driven |
| Total NPU calls/β | ~15 | **~25–30** | More placements |
| NPU overhead/β | 1.2ms | **~2.0ms** | Still <0.03% of GPU time |

## Contrast: Quenched vs Dynamical NPU Effectiveness

A key experiment-within-the-experiment: does the NPU learn the quenched phase
as effectively as the dynamical phase? Quenched trajectories are simpler
(no fermion determinant, no CG), so plaquette convergence patterns should
be more predictable. We expect:

- Quenched therm detection accuracy ≥ dynamical (simpler signal)
- Quenched length prediction to cluster near β_c (more steps needed in transition region)
- The heuristic fallback (proximity-based) to be beaten by the trained ESN after 5+ β points

---

## Files Changed

| File | Change |
|------|--------|
| `src/md/reservoir.rs` | +2 head constants (QUENCHED_LENGTH=9, QUENCHED_THERM=10), NUM_HEADS=11 |
| `src/bin/production_dynamical_mixed.rs` | Full 11-head architecture: PredictQuenchedLength, QuenchedThermCheck, pipelined predictions, adaptive CG, intra-scan steering |
| `src/bin/production_dynamical_scan.rs` | New: GPU-only dynamical scan (no NPU, baseline) |
| `src/bin/meta_table_scan.rs` | New: meta table-driven scan with quenched/dynamical mode selection |

### Compile Fixes (prerequisite)

51 pre-existing errors resolved:
- `rust-toolchain.toml`: 1.93.0 → stable (1.93.1) to match dependency workspace
- Type annotations added for wgpu 22 API (closures in adapter.rs, dispatch.rs, etc.)
- Iterator pattern fixes in gpu_diag.rs, physics.rs, reservoir.rs

---

## Run Plan

First production run will use:
```bash
HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
  --lattice=8 --betas=5.0,5.3,5.5,5.6,5.69,5.8,6.0 --mass=0.1 \
  --therm=200 --meas=500 --seed=42 \
  --bootstrap-from=results/exp022_meta.jsonl \
  --output=results/exp023_8x8.json \
  --trajectory-log=results/exp023_8x8_trajectories.jsonl \
  --save-weights=results/exp023_weights.json
```

Bootstrap from Exp 022 meta table gives the ESN a warm start — it already knows
the β landscape from previous runs. The new heads (9, 10) will train from scratch
during this run and be available for the 32⁴ production follow-up.
