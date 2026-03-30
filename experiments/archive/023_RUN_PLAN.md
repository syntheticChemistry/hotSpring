# Exp 023 Production Run Plan — Predictions & Contrasts

**Date**: February 27, 2026
**Hardware**: biomeGate (RTX 3090 DF64 + AKD1000 PCIe NPU)
**Binary**: `production_dynamical_mixed` (11-head ESN GPU conductor)

---

## Baseline Runs for Comparison

| Experiment | Version | Lattice | Type | GPU | Wall Time | s/traj | β points | Cost |
|------------|---------|---------|------|-----|-----------|--------|----------|------|
| 013 | v0.6.8 | 32⁴ | Quenched, native f64 | 3090 | 13.6h | ~16.3s | 12 | $0.58 |
| 018 | v0.6.14 | 32⁴ | Quenched, DF64 | 3090 | 6.37h | 7.64s | 12 | $0.27 |
| 015 | v0.6.11 | 32⁴ | Quenched, DF64 + NPU steer | 3090 | 4.42h (partial, 3 β) | 7.58s | 3 (seed) | ~$0.19 |
| 022 | v0.6.14 | 8⁴ | Dynamical, 9-head NPU | 3090 | — (validation) | — | 10 | — |

---

## Exp 023 Phase 1: 8⁴ Validation (11-Head Dynamical)

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Lattice | 8⁴ | Fast turnaround for NPU validation |
| β values | 5.0, 5.3, 5.5, 5.6, 5.69, 5.8, 6.0 | Core transition region + wings |
| Quark mass | 0.1 | Heavy quark limit (fast CG) |
| Therm trajectories | 200 (NPU-monitored) | Heads 2+10 can early-exit |
| Meas trajectories | 500 | Sufficient for ESN training |
| CG solver | GPU-resident | Adaptive check_interval via head 5 |
| NPU bootstrap | Exp 022 weights (if available) | Cross-run warm start |
| Seed | 42 | Reproducible |

### Predictions vs Exp 022 (9-Head, 8⁴)

| Metric | Exp 022 (9-head) | Exp 023 Predicted (11-head) | Basis |
|--------|:----------------:|:---------------------------:|-------|
| Therm early-exits (dynamical) | 60% (6/10) | **60–70%** | Warm ESN + cross-run bootstrap |
| Therm early-exits (quenched) | 0% (unmonitored) | **40–60%** | Head 10 plaquette convergence is simpler signal |
| Rejection accuracy | 86% | **86–90%** | Same head, warmer ESN |
| Quenched steps wasted | 100% of budget | **30–50% saved** | Head 9 learns allocation |
| CG check overhead | Fixed interval=10 | **Adaptive 5–20** | ~5% GPU throughput gain |
| β points inserted | 0 | **1–2** | Adaptive steering with 7 seed points |
| Total NPU calls | 5,947 | **~8,000–10,000** | More heads active + quenched monitoring |
| NPU overhead/trajectory | 1.2ms (0.016%) | **~2.0ms (0.026%)** | Still negligible |

### What to Watch

1. **Head 9 (QUENCHED_LENGTH)** learning curve — does it improve after 3+ β points?
2. **Head 10 (QUENCHED_THERM)** false positive rate — premature exit corrupts dynamical phase
3. **Adaptive CG check_interval** — does it actually reduce CG wall time?
4. **Intra-scan β insertion** — does the NPU pick sensible gap-filling points?
5. **Quenched vs dynamical therm detection accuracy** — is quenched easier to learn?

---

## Exp 023 Phase 2: 32⁴ Production (Full Pipeline)

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Lattice | 32⁴ (1,048,576 sites) | Production scale |
| β values | 5.0, 5.3, 5.5, 5.6, 5.65, 5.69, 5.72, 5.8, 5.9, 6.0, 6.5 | 11 points + adaptive insertions |
| Quark mass | 0.1 | Heavy quark, fast CG baseline |
| Therm | 50 (NPU budget per β) | Head 9 predicts, Head 10 monitors |
| Meas | 500 | Statistics for ESN + physics |
| GPU | RTX 3090 (DF64 hybrid) | Primary compute |
| NPU | AKD1000 via PCIe | Hardware inference |
| Bootstrap | Phase 1 weights | Trained on 7 β validation |

### Predictions vs Previous 32⁴ Runs

| Metric | Exp 013 (v0.6.8) | Exp 018 (v0.6.14) | Exp 023 Predicted | Improvement over 018 |
|--------|:-----------------:|:------------------:|:-----------------:|:--------------------:|
| Type | Quenched | Quenched | **Dynamical** | — |
| s/trajectory | ~16.3s | 7.64s | **7.6–8.5s** | ~0.9× (CG adds time) |
| CG overhead | N/A | N/A | **+0.5–1.0s/traj** | — (new physics) |
| Quenched steps/β | 50 (fixed) | 50 (fixed) | **25–45** (NPU-predicted) | 10–50% saved |
| Quenched early-exits | N/A | N/A | **40–60%** | New capability |
| Therm early-exits (dyn) | N/A | N/A | **50–70%** | New capability |
| β points | 12 (fixed) | 12 (fixed) | **11+2** (adaptive) | Gap-filling |
| Wall time | 13.6h | 6.37h | **5.5–7.0h** | ~0.9–1.2× |
| Electricity | $0.58 | $0.27 | **$0.24–0.30** | Similar (CG adds load) |

### Key Contrast: Dynamical vs Quenched

This will be the **first dynamical fermion run at 32⁴** in the hotSpring pipeline. Previous 32⁴ runs (Exp 013, 018) were pure gauge (quenched). The differences:

| Property | Quenched (013, 018) | Dynamical (023) |
|----------|:-------------------:|:---------------:|
| Fermion determinant | Absent | Staggered fermions via pseudofermion |
| CG solver | Not needed | GPU-resident CG per trajectory |
| Per-trajectory cost | 7.6s (DF64) | **8–9.5s** (HMC + CG) |
| Physics content | Pure gauge: Wilson action only | **Full QCD**: gauge + fermion |
| Phase transition | Deconfinement at β_c≈5.69 | **Shifted** β_c (fermions screen) |
| Plaquette | Gauge action only | Gauge + fermion backreaction |
| Polyakov loop | Order parameter for deconfinement | **Modified** by dynamical quarks |
| NPU value | Steer + recommend | **GPU prep + monitor + steer + classify** |

### Expected Physics

1. **Plaquette**: Higher than quenched at same β (fermion backreaction pushes plaquette up)
2. **β_c shift**: Lower than 5.69 — dynamical fermions smooth the transition into a crossover (for physical quark masses)
3. **CG iterations**: 50–200 for m=0.1 at most β; may spike near transition region
4. **Acceptance rate**: Lower than quenched (fermion force adds noise to trajectory)
5. **χ peak**: Broader and lower than quenched (crossover, not first-order transition)

### Expected NPU Performance (New Heads)

| Head | β region | Expected behavior |
|------|----------|-------------------|
| QUENCHED_LENGTH (9) | Wings (β<5.3, β>6.0) | Short budget (15–25 steps) |
| QUENCHED_LENGTH (9) | Transition (5.5–5.8) | Full budget (40–50 steps) |
| QUENCHED_THERM (10) | Wings | Early-exit at step 10–20 |
| QUENCHED_THERM (10) | Transition | No early-exit (needs full budget) |
| CG_ESTIMATE (5) | Wings | Low (50–100), check_interval=20 |
| CG_ESTIMATE (5) | Transition | High (200+), check_interval=5–10 |
| STEER_ADAPTIVE | After 3+ points | Insert 1–3 points in 5.5–5.8 gap |

---

## Success Criteria

### Minimum (Phase 1, 8⁴)

- [ ] All 7 β points complete without crash
- [ ] NPU calls succeed (no channel timeout)
- [ ] Quenched therm detector (H10) triggers at least once
- [ ] Quenched length predictor (H9) produces non-trivial predictions (not all max)
- [ ] Dynamical therm detector (H2) matches or exceeds 60% early-exit rate
- [ ] Plaquette values physically reasonable (0.4–0.65 range)
- [ ] CG converges at all β points

### Target (Phase 2, 32⁴)

- [ ] Complete 11-point scan + adaptive insertions in <7.5 hours
- [ ] Quenched phase savings ≥20% averaged over all β points
- [ ] Dynamical therm early-exit ≥50%
- [ ] CG adaptive check_interval active and varied (not all defaulting to 10)
- [ ] At least 1 adaptive β insertion executed
- [ ] Physics: clear crossover signal in χ(β) with peak shifted from quenched β_c≈5.69
- [ ] JSON output + trajectory log complete for ESN analysis

### Stretch

- [ ] Quenched savings ≥40%
- [ ] NPU steering inserts a point that reveals physics missed by initial grid
- [ ] Cross-run bootstrap shows measurable improvement over cold-start ESN
- [ ] Energy cost <$0.30 for full dynamical 32⁴ scan

---

## Command

```bash
# Phase 1: 8⁴ validation
HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
  --lattice=8 --betas=5.0,5.3,5.5,5.6,5.69,5.8,6.0 --mass=0.1 \
  --therm=200 --meas=500 --seed=42 \
  --output=results/exp023_8x8.json \
  --trajectory-log=results/exp023_8x8_trajectories.jsonl \
  --save-weights=results/exp023_8x8_weights.json

# Phase 2: 32⁴ production (after Phase 1 validates)
HOTSPRING_GPU_ADAPTER=3090 cargo run --release --bin production_dynamical_mixed -- \
  --lattice=32 --betas=5.0,5.3,5.5,5.6,5.65,5.69,5.72,5.8,5.9,6.0,6.5 --mass=0.1 \
  --therm=50 --meas=500 --seed=42 \
  --bootstrap-from=results/exp023_8x8_weights.json \
  --output=results/exp023_32x32.json \
  --trajectory-log=results/exp023_32x32_trajectories.jsonl \
  --save-weights=results/exp023_32x32_weights.json
```
