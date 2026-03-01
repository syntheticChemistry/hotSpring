# Experiment 030: Adaptive Steering Production Run

**Status:** IN PROGRESS (started 2026-03-01)
**Binary:** `production_dynamical_mixed`
**Predecessor:** Exp 029 (4-seed baseline, steering bug identified)

## Motivation

Exp 029 revealed a bug in the adaptive steering logic: the condition
`bi + 1 < beta_order.len()` prevented the NPU from inserting new beta points
after the final seed. The NPU's `SteerAdaptive` request was never sent on the
last iteration because `bi + 1 == beta_order.len()`, making the condition false.

This experiment runs the fixed binary where:
1. The guard was replaced with `results.len() >= 3` — steering fires on every
   point after 3 results, regardless of queue position
2. `remaining` is set to `vec![]` when no queued points exist (NPU handles this)
3. A `--max-adaptive=N` budget (default 12) caps total insertions
4. The main loop `while bi < beta_order.len()` naturally continues when
   `beta_order.push(new_beta)` extends the vec

## Configuration

```
--betas=4.5,5.25,5.69,6.5
--lattice=8
--mass=0.1
--therm=20
--quenched-pretherm=10
--meas=65
--seed=3001
--cg-tol=1e-8
--cg-max-iter=65000
--max-adaptive=12
--bootstrap-from=exp028_brain_weights.json,exp024_production_8x8.jsonl,
                 exp028_brain_production_8x8.jsonl,exp029_npu_steering_8x8.jsonl
--save-weights=exp030_brain_weights.json
```

## Bootstrap

Trained ESN from 29 combined beta points:
- Exp 024: 17 beta points from 1,071 trajectories
- Exp 028: 8 beta points from 760 trajectories
- Exp 029: 4 beta points from 325 trajectories

## Hardware

- **Layer 1 (Motor):** RTX 3090 — GPU-resident dynamical HMC + CG
- **Layer 2 (Pre-motor):** Titan V — quenched pre-thermalization
- **Layer 3 (Cortex):** CPU — Anderson 3D proxy pipeline
- **Layer 4 (Cerebellum):** AKD1000 NPU — 15-head ESN, adaptive steering

## Key Fix

The NPU now scans for gaps after every measured point (not just between queued
seeds). When the queue is empty, `remaining = vec![]` lets the NPU evaluate
the full measured range for gaps. The `--max-adaptive=12` budget prevents
runaway expansion.

NPU reprioritized the scan order based on learned data:
- β=5.2500 (priority=0.212) — transition region, highest value
- β=5.6900 (priority=0.167)
- β=4.5000 (priority=0.013)
- β=6.5000 (priority=-0.784) — lowest, already well-characterized

## Expected Outcome

With 4 seeds + up to 12 adaptive insertions, the NPU should map the full
β=4.4..6.6 phase curve with ~16 points, concentrating density around the
crossover region (β≈5.5-5.8) where prediction uncertainty is highest.

## Connections

- **Exp 029:** 4-seed baseline (steering bug prevented adaptive insertions)
- **Exp 024:** HMC parameter sweep (training data)
- **Exp 028:** Brain concurrent pipeline (training data + weights)
- **Nautilus Shell:** `predict_live_exp029` example validated 2.6% blind CG error
