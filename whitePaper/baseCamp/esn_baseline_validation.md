# ESN Baseline Validation: What the Brain Can and Cannot Learn

**Date:** March 3, 2026
**Binary:** `barracuda/src/bin/esn_baseline_validation.rs` (v2)
**Data:** 116 aggregated points from 9 experiments (exp023–034), 2,538 trajectories, 4 lattice sizes (2–8), 5 masses (0.01–0.5)

---

## Executive Summary

The 50-neuron CPU Echo State Network is **not the bottleneck**. On synthetic data it achieves R² > 0.98 for smooth functions, volume-dependent scaling, and noisy signals. On real physics data, 3 heads achieve production-grade accuracy (R² > 0.97) while 5 heads fail — entirely due to target design and data distribution, not model capacity.

The `HeadConfidence` tracker (now live in `npu_worker.rs`) gates untrusted heads to heuristic fallbacks while continuing to train them. Heads graduate to "trusted" automatically when their rolling R² crosses 0.3.

---

## Tier 1: Production-Ready Heads

| Head | R² (CV) | Notes |
|---|---|---|
| **REJECT_PREDICT** | 0.984 | Learns rejection probability from (beta, plaq, mass, chi, acc). Plateaus at 50% data (~60 points). Generalizes across volumes and masses. |
| **PLAQUETTE** | 0.999 | Direct observable prediction. Essentially perfect. |
| **ACCEPTANCE** | 1.000 | Acceptance rate prediction. Perfect across all regimes. |

These heads should be trusted immediately by the HeadConfidence tracker.

## Tier 2: Useful, Needs More Data

| Head | R² (CV) | Limiting Factor |
|---|---|---|
| **PARAM_SUGGEST** | 0.618 | dt suggestion works (R²=0.84 on fold 5) but inconsistent across folds. The target is discretized (only 4 levels). A continuous target from actual dt_used would help. Reservoir sweep shows N=25 is actually better (R²=0.85) — the model is **overfitting** at larger reservoirs. |
| **PHASE_CLASSIFY** | 0.284 | Works well on folds with balanced data (R²=0.83 on fold 1) but fails when test set has unseen volumes (fold 4: R²=-0.95). The phase transition is a crossover on small lattices, making classification genuinely hard. Larger volumes where the transition sharpens would help. |

## Tier 3: Failing — Fixable with Target Redesign

| Head | R² (CV) | Root Cause | Fix |
|---|---|---|---|
| **LOG_CG** | -83.3 | CG iterations span 5 orders of magnitude. Even with log-scale targets, a single fold (exp028/029 holdout) has R²=-382, destroying the mean. The ESN learns CG trends within a volume but cannot extrapolate to unseen experiment configurations. | Train per-volume ESN ensembles, or add experiment-type as an input feature. |
| **BETA_PRIORITY** | -1.86 | Susceptibility is a derived quantity (std_plaq² × L⁴) with extremely skewed distribution. Log-scale helped but not enough. | Use rank ordering (percentile) as target instead of raw value. |
| **ACTION_DENSITY** | -9.87 | Action density is redundant with plaquette (action = 6(1-P) for SU(3)). The ESN gets confused trying to learn a near-perfect correlation with a different normalization. | Remove this head or replace with a non-trivially derived observable. |

---

## Synthetic Dataset Insights

| Dataset | What It Tests | Key Finding |
|---|---|---|
| **Sine Wave** | Smooth function approximation | ESN hits R²=0.98 on REJECT, PLAQUETTE, ACCEPTANCE, ACTION. It can learn any smooth 1D mapping. |
| **Step Function** | Sharp classification boundary | R²=0.000 on all heads. The ESN's tanh reservoir smooths the step — it cannot represent discontinuities. This explains why PHASE_CLASSIFY struggles at the transition. **Implication: phase classification needs a post-processing threshold, not a continuous output.** |
| **Power-Law CG** | Multi-scale regression (CG ~ 1/m²) | REJECT and ACCEPTANCE: R²=0.998+. LOG_CG: R²=-3.4. Even log-scaling isn't enough when the power-law exponent varies with other parameters. |
| **Volume Scaling** | L⁴-dependent observables | LOG_CG: R²=1.000. With volume as an explicit input (6D), the ESN perfectly learns volume scaling on synthetic data. The failure on real data is not a volume-awareness issue — it's the interaction between volume, mass, and beta that creates unseen regimes. |
| **Noisy Signal** | 30% uniform noise robustness | LOG_CG: R²=0.986, PLAQUETTE: R²=1.000. The ESN handles noise extremely well. Our real-data failures are not noise-related. |

---

## Reservoir Size: Bigger Is Not Better

| Reservoir Size | REJECT R² | PARAM R² | PLAQUETTE R² |
|---|---|---|---|
| N=25 | 0.998 | **0.850** | 0.953 |
| N=50 | 0.991 | 0.732 | 1.000 |
| N=100 | 0.999 | 0.520 | 0.999 |
| N=200 | 0.997 | -5.720 | 0.995 |
| N=400 | 0.998 | -3.958 | 0.997 |

PARAM_SUGGEST performance **degrades** at N>25 — classic overfitting with ridge regression. The ESN has too many free parameters relative to the 116 training points.

**Recommendation:** Use N=50 for heads with strong signal (REJECT, PLAQ, ACC). Use N=25 for heads with weaker signal (PARAM, PHASE). Or: implement per-head reservoir sizing.

---

## Cross-Regime Generalization

### Volume Transfer

| Test Volume | REJECT R² | PLAQUETTE R² | Overall |
|---|---|---|---|
| 6^4 | 0.993 | 0.999 | Excellent |
| 4^4 | 0.968 | 0.996 | Good |
| 8^4 | 0.832 | 0.999 | REJECT degrades |
| 2^4 | 0.000 | 0.995 | REJECT fails (toy lattice) |

PLAQUETTE generalizes perfectly across all volumes. REJECT degrades when extrapolating to volumes with different physics (8^4 has qualitatively different dynamics, 2^4 is a toy).

### Mass Transfer

| Direction | REJECT R² | PLAQUETTE R² | PARAM R² |
|---|---|---|---|
| Heavy → Light | **0.973** | 0.999 | 0.914 |
| Light → Heavy | -0.614 | 0.998 | 0.576 |

Training on heavy masses (m≥0.1) and predicting light masses works well. The reverse fails for REJECT. Light-mass physics is a subset of heavy-mass physics (less CG stress, higher acceptance). The ESN can extrapolate "easier" but not "harder."

**Implication:** Always include some heavy-mass data in training. The brain should prioritize collecting diverse mass data early.

---

## Recommendations for Brain Evolution

### Immediate (no code changes needed)
1. HeadConfidence tracker will auto-gate failing heads — already deployed
2. Run 50+ diverse beta points before trusting any head for steering
3. Include at least 3 mass values in early runs to seed mass-dependent learning

### Short-term (target/input redesign)
1. **LOG_CG:** Train per-volume sub-models or use `(log_cg - mean_log_cg_for_volume) / std` as target
2. **PARAM_SUGGEST:** Use continuous dt_used from results (already in Schema B data) instead of discretized acceptance buckets
3. **PHASE_CLASSIFY:** Apply sigmoid post-processing + threshold (0.5) instead of raw ESN output; the step-function synthetic test proves the ESN cannot represent sharp boundaries

### Medium-term (architecture changes)
1. **Per-head reservoir sizing:** N=25 for PARAM/PHASE, N=50 for REJECT/PLAQ
2. **Ensemble ESN:** Train 3 ESNs with different seeds, average predictions, use spread as uncertainty
3. **Online incremental training:** Instead of full retrain from scratch, update readout weights incrementally as new data arrives (rank-1 update to ridge regression)

### Long-term (brain architecture)
1. **Hierarchical heads:** Phase classifier → gates → regime-specific CG predictor
2. **Memory buffer:** Store last 200 trajectory-level records in ring buffer, retrain every 20 trajectories instead of per-beta
3. **Cross-spring knowledge transfer:** Import wetSpring/neuralSpring ESN weights for related prediction tasks
4. **NPU-as-DP memoization:** See [`npu_dynamic_programming.md`](npu_dynamic_programming.md) — the NPU predicts downward (hard→easy) but not upward, which makes it a natural memoization table for multigrid HMC

### Activation: tanh ≈ ReLU-approx-tanh
The `Activation::ReluTanhApprox` mode (5-segment piecewise-linear) matches native tanh
to Δ R² < 0.03 on all working heads and actually *improves* PARAM_SUGGEST by +0.31.
One system deploys to AKD1000. See [`npu_dynamic_programming.md`](npu_dynamic_programming.md) §1 for full comparison.

---

## Data Inventory

| Mass | Points | Mean CG | Mean Acc | Coverage |
|---|---|---|---|---|
| 0.010 | 8 | 21,301 | 0.38 | Sparse — need more |
| 0.050 | 14 | 30,951 | 0.81 | Moderate |
| 0.100 | 75 | 49,462 | 0.75 | Dense |
| 0.200 | 11 | 16,891 | 0.96 | Sparse |
| 0.500 | 8 | 13,148 | 1.00 | Sparse |

**Critical gap:** m=0.01 has only 8 points and 38% acceptance — this is the regime where Hasenbusch (Rung 2) is needed. The ESN cannot learn light-quark physics from 8 points.
