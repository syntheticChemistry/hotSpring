# Experiment 028: Brain Concurrent Pipeline

**Status:** COMPLETE (2026-02-28 — 2026-03-01)
**Date:** February 28, 2026
**Depends on:** Exp 024 (production data), Exp 026 (4D proxy), Spec: BIOMEGATE_BRAIN_ARCHITECTURE
**License:** AGPL-3.0-only

---

## Motivation

The production dynamical pipeline (Exp 024) runs serially: GPU, then
NPU, then GPU. During a 40-second CG solve, the Titan V, CPU (256 GB),
and NPU are all idle. The 3090 itself is 90% idle (10% occupancy at 8⁴).

This experiment validates a concurrent "brain" architecture where all
four substrates run simultaneously, with the NPU as the coordination
layer. See `specs/BIOMEGATE_BRAIN_ARCHITECTURE.md` for the full design.

---

## Phase 1: Layer 1 Validation — NPU Residual Watcher

### Test 1A: Residual streaming

Run a single dynamical trajectory at β=5.69, 8⁴, with `gpu_cg_solve_brain`
replacing `gpu_cg_solve_resident`. Verify:

- [ ] NPU receives `CgResidualUpdate` at every batch boundary
- [ ] Residual values match what the synchronous CG would produce
- [ ] Final iteration count matches synchronous CG (physics unchanged)
- [ ] Wall time overhead < 5% vs synchronous CG

```bash
cargo run --release --bin production_dynamical_mixed -- \
  --lattice=8 --betas=5.69 --mass=0.1 --dt=0.01 --n-md=100 \
  --therm=3 --quenched-pretherm=3 --meas=5 --seed=42 \
  --brain-mode=layer1 \
  --trajectory-log=../results/exp028_layer1_test.jsonl \
  2>&1 | tee ../results/exp028_layer1.log
```

### Test 1B: Interrupt delivery

Inject a synthetic anomaly (force CG residual to increase) and verify:

- [ ] NPU detects anomaly within 2 batch boundaries
- [ ] `BrainInterrupt` with `CgDiverging` arrives at main thread
- [ ] Main thread aborts CG solve
- [ ] Attention state transitions GREEN → YELLOW → RED

### Test 1C: Dynamic check interval

Run 10 trajectories and verify:

- [ ] Check interval starts at 100 (GREEN)
- [ ] If any trajectory shows slow convergence, interval decreases to 20 (YELLOW)
- [ ] After recovery, interval returns to 100 (GREEN)

---

## Phase 2: Layer 2 Validation — Titan V Pre-Motor

### Test 2A: Titan V device creation

Verify that two `GpuF64` instances can coexist:

- [ ] `GpuF64::new()` with `HOTSPRING_GPU_ADAPTER=3090` succeeds
- [ ] `GpuF64::new()` with `HOTSPRING_GPU_ADAPTER=titan` succeeds
- [ ] Both devices report different adapter names
- [ ] Simple shader dispatch works on both simultaneously

```bash
cargo run --release --bin production_dynamical_mixed -- \
  --lattice=8 --betas=5.69 --mass=0.1 --dt=0.01 --n-md=100 \
  --therm=3 --quenched-pretherm=3 --meas=5 --seed=42 \
  --brain-mode=layer2 \
  --trajectory-log=../results/exp028_layer2_test.jsonl \
  2>&1 | tee ../results/exp028_layer2.log
```

### Test 2B: Pre-therm overlap

Run a 3-beta scan (5.0, 5.69, 6.0) with Titan V pre-therm enabled:

- [ ] Titan V starts pre-therm for β_{i+1} while 3090 measures at β_i
- [ ] Warm config arrives before 3090 finishes measurement
- [ ] Plaquette from warm config is valid (within 10% of expected)
- [ ] 3090 skips quenched pre-therm (uses warm config directly)
- [ ] Total wall time decreases vs serial baseline

### Test 2C: Config transfer integrity

Transfer a gauge config from Titan V to 3090 and verify:

- [ ] Plaquette computed on 3090 matches plaquette computed on Titan V
- [ ] Polyakov loop matches
- [ ] First dynamical trajectory from warm config is physically valid

---

## Phase 3: Layer 3 Validation — CPU Cortex Proxy

### Test 3A: Concurrent proxy execution

Run the cortex thread alongside a CG solve and verify:

- [ ] `anderson_4d_proxy` completes within one CG solve wall time (< 40s)
- [ ] Proxy features delivered to NPU before next trajectory starts
- [ ] CPU utilization increases (cortex thread uses Threadripper cores)
- [ ] GPU performance unaffected (no contention)

### Test 3B: Proxy feature integration

Feed proxy features to NPU Head 14 and verify:

- [ ] CG iteration predictions improve (lower MAE vs no-proxy baseline)
- [ ] NPU receives `ProxyFeatures` struct with valid diagnostics
- [ ] Feature values match offline proxy pipeline (Exp 025) results

### Test 3C: Multi-tier proxy

Run all three proxy tiers concurrently using Rayon:

- [ ] 3D scalar (200ms) + 4D scalar (3s) + 4D Wegner (20s) complete < 25s
- [ ] All three tiers produce valid features
- [ ] NPU receives features from all tiers in priority order

---

## Phase 4: Layer 4 Validation — Attention State Machine

### Test 4A: State transitions

Inject a sequence of synthetic residual updates and verify:

- [ ] Starts in GREEN
- [ ] Single anomaly (residual plateau) → YELLOW
- [ ] Sustained anomaly (3+ consecutive) → RED
- [ ] Recovery (3 consecutive normal) → GREEN
- [ ] Interrupt emitted only on RED transition

### Test 4B: Full integration

Run a 5-beta scan with all four layers active:

- [ ] Layer 1: NPU receives residuals during every CG solve
- [ ] Layer 2: Titan V pre-thermalizes next beta while 3090 measures
- [ ] Layer 3: CPU runs proxy during CG, features available before next traj
- [ ] Layer 4: Attention state reflects actual system health
- [ ] All four substrates show non-zero utilization in parallel

### Test 4C: Performance comparison

Run the same 5-beta scan with brain mode enabled and disabled:

| Metric | Serial (baseline) | Brain (concurrent) |
|--------|:-----------------:|:------------------:|
| Wall time | — | Expected 10-20% faster |
| CG prediction MAE | — | Expected 50%+ better |
| Wasted trajectories | — | Expected fewer (interrupt kills bad CG) |
| Substrate utilization | 1/4 | 4/4 |

---

## Hardware Assignment

| Substrate | Layer | Work |
|-----------|-------|------|
| RTX 3090 | Motor | CG solve, HMC trajectories (primary) |
| Titan V | Pre-motor | Quenched pre-therm for next β |
| Threadripper CPU | Cortex | Anderson 4D, Wegner block proxy |
| AKD1000 NPU | Cerebellum | Residual monitoring, attention, steering |

---

## Output Files

| File | Contents |
|------|----------|
| `results/exp028_layer1_test.jsonl` | Layer 1 residual streaming test |
| `results/exp028_layer2_test.jsonl` | Layer 2 Titan V overlap test |
| `results/exp028_layer3_test.jsonl` | Layer 3 proxy concurrency test |
| `results/exp028_full_brain.jsonl` | Full 4-layer integration test |
| `results/exp028_brain_vs_serial.jsonl` | Performance comparison |
