# Reality Ladder — Rung 0: Mass x Volume x Beta Scan

**Experiment:** 033
**Date:** 2026-03-02 → 2026-03-03 (overnight)
**Status:** Complete — 479 trajectories, GPU free
**Hardware:** RTX 3090 (NVK GA102), AKD1000 NPU, Threadripper 3970X

---

## What This Experiment Is

Rung 0 is the first step of the Reality Ladder: a systematic scan across quark
mass, gauge coupling (beta), and lattice volume at fixed Nf=4 (one staggered
fermion field). No new code was required — the existing `production_dynamical_mixed`
binary was run with `--mass` varied across 5 values spanning 50x in quark mass.

The goal is to give the NPU a **three-dimensional training manifold** (mass x beta
x volume) so it learns how the physics changes as quarks get lighter, the coupling
weakens, and the lattice grows. Before this experiment, the NPU only knew one
point: mass=0.1, beta=5.69, 4^4.

---

## Run Structure

| Phase | Volume | Masses | Betas | Traj | NPU Mode | Time |
|-------|--------|--------|-------|------|----------|------|
| 0a | 2^4 (16 sites) | 0.5, 0.2, 0.1, 0.05, 0.01 | 5.0, 5.5, 6.0 | 105 | Steered | ~25 min |
| 0b | 4^4 (256 sites) | 0.5, 0.2, 0.1, 0.05, 0.01 | 5.0, 5.3, 5.5, 5.69, 6.0 | 275 | Steered | ~2.5 hr |
| 0c | 6^4 (1296 sites) | 0.2, 0.1, 0.05 | 5.5, 5.69, 6.0 | 99 | Manual | ~2 hr |
| **Total** | | | | **479** | | **~5 hr** |

Phase 0a/0b: NPU controls dt and n_md, predicts CG iterations, classifies phases.
Phase 0c: Manual params (dt=0.012, n_md=42) calibrated from Exp 032 volume scaling.

---

## Results: 4^4 Mass x Beta Grid (Primary Training Data)

| mass | beta=5.0 | beta=5.3 | beta=5.5 | beta=5.69 | beta=6.0 |
|------|----------|----------|----------|-----------|----------|
| **0.5** | 0.396 C | 0.420 C | 0.440 C | 0.473 C | 0.502 T |
| **0.2** | 0.407 C | 0.432 C | 0.447 C | 0.487 T | 0.515 T |
| **0.1** | 0.414 C | 0.444 C | 0.440 C | 0.498 T | 0.523 T |
| **0.05** | 0.418 C | 0.438 C | 0.446 C | 0.497 T | 0.524 T |
| **0.01** | 0.358 X | 0.393 X | 0.407 X | 0.457 X | 0.463 X |

Values are mean plaquette. C = confined, T = transition/deconfined, X = broken (0% acceptance).

### CG Cost Scaling

| mass | avg CG iters | acceptance | regime |
|------|-------------|------------|--------|
| 0.5 | 13,020 | 100% | Easy — heavy quarks, well-conditioned |
| 0.2 | 13,100 | 96% | Easy — moderate quarks |
| 0.1 | 13,020 | 88% | Standard — baseline parameters |
| 0.05 | 25,420 | 84% | Hard — CG cost doubles, condition number ~400 |
| 0.01 | 28,500 | 0% | **Broken** — CG hits max iter, delta_H ~ 10^9 |

### 6^4 Volume Scaling

| mass | beta=5.5 | beta=5.69 | beta=6.0 | avg CG |
|------|----------|-----------|----------|--------|
| 0.2 | 0.463 (80%) | 0.504 (100%) | 0.529 (100%) | 26,880 |
| 0.1 | 0.466 (60%) | 0.514 (100%) | 0.503 (80%) | 26,878 |
| 0.05 | 0.471 (60%) | 0.500 (80%) | 0.494 (40%) | 52,470 |

At 6^4, CG cost doubles again vs 4^4. Mass=0.05 at beta=6.0 hits 40% acceptance —
the algorithmic boundary is closing in.

---

## Physics Learned

### 1. Mass shifts the confinement transition

At heavy mass (0.5), the fermion determinant suppresses gauge fluctuations —
the system stays confined up to beta=5.69 and only barely crosses at beta=6.0.
As mass decreases toward the chiral limit, the determinant contributes less
suppression, and the transition moves to lower beta. This is the expected
behavior: lighter dynamical quarks make the vacuum more "active."

### 2. The chiral limit is an algorithmic cliff

At mass=0.01 on 4^4, the condition number of D†D is ~1/m² = 10,000. The CG
solver maxes out at 5000 iterations and returns wrong forces. The MD integrator
then computes a trajectory with delta_H ~ 10^9 (a billion), and Metropolis
correctly rejects everything. This is not a physics failure — it's an algorithm
failure. The cure is Hasenbusch mass preconditioning (Rung 2, already implemented
in `barracuda/src/lattice/gpu_hmc/hasenbusch.rs`).

### 3. Volume amplifies everything

Going from 4^4 to 6^4 at mass=0.1:
- CG cost: 13,020 → 26,878 (2.1x)
- Acceptance: 88% → 80% (tighter integrator tolerance needed)
- Plaquette fluctuations narrow (the system is more "thermal")

At mass=0.05 on 6^4, acceptance drops to 40-60%, indicating that `dt=0.012` is
too aggressive for this regime. The NPU will learn this boundary.

---

## NPU Training Value

The NPU now has:

| Dimension | Range | Points |
|-----------|-------|--------|
| Mass | 0.01 → 0.5 (50x) | 5 |
| Beta | 5.0 → 6.0 | 3–5 |
| Volume | 2^4 → 6^4 | 3 |

**New capabilities the NPU can learn from this data:**
- Predict CG iteration count as a function of (mass, beta, volume)
- Predict acceptance rate as a function of (mass, dt, volume)
- Classify the algorithmic regime (easy / hard / broken)
- Steer dt/n_md based on mass — lighter quarks need smaller steps
- Identify the Hasenbusch boundary (where standard CG fails)

---

## What Comes Next

| Rung | Goal | Status |
|------|------|--------|
| 0 | Mass scan at Nf=4 | **COMPLETE** — this experiment |
| 1 | Multi-field Nf=8, 12 comparison | Code ready (`--n-fields`), run pending |
| 2 | GPU Hasenbusch preconditioning | Code ready (`hasenbusch.rs`), unlocks mass < 0.05 |
| 3 | RHMC for Nf=2, 2+1 | Code ready (`rhmc.rs`), unlocks physical QCD |

---

## Files

- Script: `experiments/033_reality_ladder_rung0.sh`
- Results: `results/exp033_{2x2,4x4,6x6}_m{mass}.jsonl`
- Master log: `results/exp033_rung0_master.log`
- Plan: `.cursor/plans/reality_ladder_plan_3571df11.plan.md`
