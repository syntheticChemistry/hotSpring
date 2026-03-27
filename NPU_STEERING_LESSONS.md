# NPU Steering — Lessons Learned

> **Fossil Record (March 27, 2026):** This document captures state as of March 21, 2026. For current status, see the [root README](README.md) and [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md). Body below is preserved as historical record.

> **Note (March 21, 2026):** Point-in-time lessons from the dynamical-only NPU steering run. For current NPU integration status, see [root README](README.md) and [`wateringHole/handoffs/HOTSPRING_TRIO_EVOLUTION_AMD_AKIDA_HANDOFF_MAR20_2026.md`](wateringHole/handoffs/HOTSPRING_TRIO_EVOLUTION_AMD_AKIDA_HANDOFF_MAR20_2026.md).

**Date**: March 9, 2026
**Run**: `validate_chuna_overnight --dynamical-only`
**Hardware**: biomeGate — Akida AKD1000 NPU, Threadripper 3970X, Titan V, RTX 3090
**Result**: 3/3 checks pass, 85% acceptance, all 4 mass annealing stages complete

---

## The Problem

Dynamical N_f=4 staggered fermion HMC on an 8⁴ lattice at β=5.4 with
target mass m=0.1.  This is a notoriously stiff system: the fermion
force is orders of magnitude larger than the pure gauge force, and
the CG solver becomes expensive as mass decreases.

Previous runs had 0% acceptance with catastrophic ΔH (18,000 → 1.4M).

---

## What Worked

### 1. Titan V plaquette guard

The NVK (nouveau/Vulkan) driver on the Titan V GV100 produces **all-zero
link buffers** from GPU compute shaders.  The readback gives P=0.000000
(cold start gives P=1.0, thermalized gives P≈0.47–0.60).

**Fix**: reject Titan V warm config if P < 0.1 or P > 0.999, fall back to
CPU quenched pre-thermalization.  This caught the broken config on every run.

### 2. CPU quenched pre-thermalization

100 quenched HMC trajectories from hot start at β=5.4 reliably thermalize
to P≈0.470.  Wall time: ~90s.  This replaces the broken Titan V path and
provides a clean starting lattice for mass annealing.

### 3. Mass annealing (m: 1.0 → 0.5 → 0.2 → 0.1)

Standard lattice QCD technique.  Start at heavy mass (easy CG, mild force)
and anneal toward the target.  Each stage runs 15–20 adaptive HMC
trajectories.  Acceptance:

| Stage | Mass | Final acc | |ΔH| | CG/traj |
|-------|------|-----------|-------|---------|
| 1     | 1.00 | 80%       | 0.24  | 44      |
| 2     | 0.50 | 100%      | 0.01  | 86      |
| 3     | 0.20 | 90%       | 0.20  | 200     |
| 4     | 0.10 | 80%       | 0.33  | 382     |

### 4. Emergency dt scaling (|ΔH| > 2 → scale to target 0.5)

Instead of blind halving (which doubles n_md and wall time), the controller
now estimates the needed dt reduction from the Omelyan scaling |ΔH| ∝ dt².

```
scale = (0.5 / |ΔH|)^(1/2), clamped to [0.1, 0.9]
new_dt = dt × scale
n_md unchanged → trajectory shortens proportionally
```

This recovered from ΔH=185.8 on the first trajectory to ΔH≈1 within 2–3
trajectories, without exploding the per-trajectory cost.

### 5. Short initial trajectory length (τ = 0.05)

The original heuristic used τ = dt × n_md = 0.5 (standard for quenched).
Dynamical fermions at strong coupling are far stiffer.  τ = 0.05 with
n_md ≈ 100 keeps trajectories fast (~1 min each with m=1.0) while the
adaptive controller probes the landscape.

---

## What Did NOT Work

### 1. Titan V GPU HMC via NVK

200 quenched trajectories ran in 18s but produced all-zero link buffers.
The NVK (nouveau Vulkan) driver for GV100 (Volta) appears to have compute
shader bugs.  **Do not trust Titan V readbacks until NVK matures or
proprietary driver is available.**

### 2. NPU overriding the adaptive controller

The untrained NPU (default reservoir weights) kept suggesting dt=0.001,
n_md=500 — the same parameters that caused catastrophic ΔH.  When the
adaptive controller reduced dt in response to 0% acceptance, the NPU
immediately overrode it back to the failing parameters.

**Root cause**: the NPU's `suggest_params` always returned `Some(...)` with
untrained weights, trumping the heuristic controller on every feedback
interval.

### 3. τ = 0.5 for dynamical fermions

A trajectory length of 0.5 with dt=0.001 means 500 MD steps, each requiring
a CG solve.  The fermion force at m=1.0 on a freshly thermalized quenched
lattice is enormous — |ΔH| ≈ 18,000.  The system never accepted a single
configuration.

### 4. Fixed n_md during dt emergency reduction

The first emergency reducer halved dt but kept τ constant by doubling n_md.
This doubled the per-trajectory wall time (each CG solve costs ~10ms on 8⁴)
without improving acceptance enough to justify the cost.

---

## The NPU Trust Model: "Teach the Apprentice"

The NPU should be treated like a talented apprentice — we listen, we teach,
but we don't let it override the chef until its suggestions consistently
beat the heuristic.

### Current implementation

```
if acceptance < 40% || |ΔH| > 5.0:
    NPU defers → adaptive controller runs alone
else:
    NPU suggests (dt, n_md) → applied if within bounds
```

This means:
- **Crisis regime** (acc < 40% or |ΔH| > 5): NPU is silent, heuristic
  controller drives recovery.  The "adult" handles the emergency.
- **Stable regime** (acc ≥ 40%, |ΔH| ≤ 5): NPU begins to steer, fine-tuning
  dt and n_md around the heuristic's baseline.

### What the NPU needs to evolve

1. **Earned trust**: Track NPU suggestions vs. actual outcomes.  If the NPU's
   suggested dt would have produced better acceptance than the heuristic over
   the last N trajectories, increase its authority.  If not, reduce it.

2. **Bounded deviation**: Even in stable regime, the NPU should only adjust
   dt by ±30% of the heuristic's value.  A master chef experiments at the
   margins, not by replacing the entire menu.

3. **Cross-run learning**: The `RunHistoryWriter` logs every trajectory to
   JSONL.  After each run, `retrain_npu_from_history()` warms the reservoir.
   Over many runs, the NPU builds a force-landscape model for each (β, mass,
   lattice) combination.

4. **Multi-head specialization**:
   - `PARAM_SUGGEST`: dt/n_md tuning (current)
   - `ANOMALY_DETECT`: flag divergent forces before they corrupt the trajectory
   - `CG_PREDICT`: estimate CG iteration count for budgeting
   - `WARM_START`: suggest initial dt for a new (β, mass) from learned history

5. **The graduation test**: When the NPU's suggestions match or exceed the
   heuristic's acceptance rate over 100+ trajectories across 3+ β values,
   promote it from "advisor" to "driver" — let it set initial params without
   the heuristic fallback.

### The metaphor

> The NPU is a child learning to cook.  We teach it what works (heuristic
> baseline), let it taste the results (trajectory outcomes), and listen when
> it suggests variations.  But until it consistently produces better meals
> than the recipe book, we don't let it plan dinner alone.  Skittles and
> spaghetti might be creative, but the physics needs to converge first.
>
> Over time — with enough runs, enough JSONL history, enough reservoir
> warming — the child becomes a chef.  And a chef sees patterns the recipe
> book never captured: force landscapes that predict CG cost, β-mass
> combinations where aggressive dt actually helps, anomalies that signal
> topological transitions.  That's when the NPU earns the wheel.

---

## Technical Parameters (final working configuration)

```
ADAPTIVE_DT_MIN     = 1e-6     (was 5e-5 — too high for light fermions)
ADAPTIVE_DT_MAX     = 0.02
ADAPTIVE_NMD_MIN    = 20
ADAPTIVE_NMD_MAX    = 10,000
ADAPTIVE_DT_DROP    = 0.50     (standard adaptation)
ADAPTIVE_DT_BUMP    = 1.15     (standard adaptation)
ADAPTIVE_HIGH_ACC   = 0.85
ADAPTIVE_LOW_ACC    = 0.40
EMERGENCY_DH_THRESH = 2.0      (was 10 → 100 in earlier iterations)
NPU_DEFER_ACC       = 0.40     (was 0.10 → 0.15 in earlier iterations)
NPU_DEFER_DH        = 5.0
NPU_TARGET_TAU      = 0.02     (was 0.5 → 0.1 → 0.05 in earlier iterations)
NPU_MAX_NMD         = 200
```

---

## Iteration History

| Run | dt_init | n_md | τ | |ΔH| stage 1 | Accept | Issue |
|-----|---------|------|---|-------------|--------|-------|
| 1 (broken) | 0.001 | 500 | 0.5 | 0.0 | 100% (frozen) | Titan V all-zero links |
| 2 | 0.001 | 500 | 0.5 | 18,000 | 0% | NPU override + τ too long |
| 3 | 0.0001 | 1000 | 0.1 | 185.8 | 0% | NPU override + n_md too high |
| 4 | 0.0005 | 100 | 0.05 | 185.8→1.23 | 40%→80% | Emergency recovery worked |
| **5 (final)** | **0.0005** | **100** | **0.05** | **185.8→0.09** | **80–100%** | **All stages pass** |

---

## Files Changed

- `barracuda/src/lattice/pseudofermion/mod.rs` — AdaptiveStepController, NpuSteering, warm-start
- `barracuda/src/bin/validate_chuna_overnight.rs` — `--dynamical-only`, Titan V guard, NPU telemetry
- `barracuda/src/production/titan_worker.rs` — plaquette diagnostic
- `barracuda/src/tolerances/lattice.rs` — ADAPTIVE_DT_MIN lowered to 1e-6
