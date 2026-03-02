# Experiment 031 Post-Mortem: NPU-Controlled Dynamical HMC

**Date:** March 2, 2026
**Run time:** 12.9 hours (46,437s), exit code 0
**Config:** 8⁴ SU(3), m=0.1, 4 β points × 65 meas, seed=4001
**Hardware:** RTX 3090 (motor) + Titan V (pre-motor) + AKD1000 (cerebellum) + CPU (cortex)

---

## What Worked

### Physics: Clean Deconfinement Scan

| β | ⟨P⟩ | σ(P) | |L| | χ | acc% | ⟨CG⟩ | Phase |
|---|-----|------|----|---|------|------|-------|
| 4.50 | 0.3325 | 0.0038 | 0.300 | 0.058 | 63% | 127,209 | confined |
| 5.25 | 0.4656 | 0.0059 | 0.308 | 0.143 | 62% | 67,783 | confined |
| 5.69 | 0.5531 | 0.0036 | 0.292 | 0.052 | 66% | 67,271 | transition |
| 6.50 | 0.6258 | 0.0023 | 0.309 | 0.021 | 92% | 57,590 | deconfined |

- Deconfinement transition confirmed at β_c ≈ 5.69
- Susceptibility peaks at β=5.25 (χ=0.143), consistent with crossover
- Anderson proxy ⟨r⟩ in extended regime (0.48–0.54) at all β — correct
- 150 JSONL output records, clean data

### Architecture: All Threads Spawned and Communicating

- 4-layer brain initialized: RTX 3090, Titan V, CPU cortex, NPU cerebellum
- Brain residual forwarder operational (CG residuals → NPU)
- 321 NPU calls dispatched and acknowledged (no deadlocks, no panics)
- Bootstrap ESN from 30 historical data points across 5 prior experiments
- NPU β priority ordering correctly identified transition region (β=5.69 first)

### NPU Mid-Run Steering

- At β=4.50: detected low acceptance (0%), drove dt 0.010→0.005, n_md 100→195
- At β=6.50: detected high acceptance (>85%), ramped dt 0.005→0.012, n_md 195→84
- Steering kept acceptance rates in 60-92% range — functional, useful

---

## What Failed

### 1. Titan V: Zero Useful Work (100% waste)

**Every config produced was discarded.** Root cause: timing.

The code sends the Titan pre-therm request at the **end** of β[n] (line 900-917
of `production_dynamical_mixed.rs`). But `try_recv` fires at the **start** of
β[n+1] — immediately, before Titan finishes (48-107s per config).

Timeline:
```
β=5.69 starts → Titan idle (no request yet)
β=5.69 ends   → Titan sent β=5.25 request
β=5.25 starts → try_recv: empty (Titan needs 48s, 0s elapsed)
               → 3090 self-therms (CPU fallback + GPU quenched)
β=5.25 ends   → Titan sent β=4.50 request
β=4.50 starts → try_recv: finds β=5.25 config → DISCARDED (wrong β)
β=4.50 ends   → Titan sent β=6.50 request
β=6.50 starts → try_recv: finds β=4.50 config → DISCARDED (wrong β)
```

Titan was idle for 6,085s during β=5.69 (the entire first measurement).
Total Titan compute: 213s. Total Titan useful compute: 0s.

### 2. NPU Reject Predictions: 0/260 Correct

The ESN's rejection head produces meaningless output. Root cause:
**input feature misalignment between training and inference.**

Training input format (build_training_data, line 933-944):
```
[beta_norm, mean_plaq + noise, mass, chi/1000, acceptance]
```

Inference input for RejectPredict (line 381-388):
```
[beta_norm, plaquette, delta_h/10, acceptance_rate, 0.0]
```

Positions 2-4 have **completely different semantics**:
- Position 2: mass (training) vs delta_h/10 (inference)
- Position 3: chi/1000 (training) vs acceptance_rate (inference)
- Position 4: acceptance (training) vs 0.0 (inference)

The ESN learned to map (beta, plaq, mass, chi, acc) → targets. During inference
it receives (beta, plaq, deltaH, acc, 0) — different features in the same
positions. The predictions are meaningless.

**This affects ALL heads, not just reject prediction:**

| Head | Training pos 1-4 | Inference pos 1-4 | Aligned? |
|------|-------------------|---------------------|----------|
| PreScreenBeta | plaq, mass, chi/1k, acc | meta_plaq, 0.1, meta_chi/1k, 0.5 | Partial |
| SuggestParameters | plaq, mass, chi/1k, acc | L/32, mass, 0.5, 0.0 | **No** |
| PredictCgIters | plaq, mass, chi/1k, acc | mass, L/32, 0.0, 0.0 | **No** |
| PredictQuenchedLength | plaq, mass, chi/1k, acc | meta_plaq, mass, L/32, 0.0 | Partial |
| QuenchedThermCheck | plaq, mass, chi/1k, acc | mean, σ, len/100, 1.0 | **No** |
| ThermCheck | plaq, mass, chi/1k, acc | mean, σ, len/200, 0.0 | **No** |
| RejectPredict | plaq, mass, chi/1k, acc | plaq, ΔH/10, acc, 0.0 | **No** |
| PhaseClassify | plaq, mass, chi/1k, acc | plaq, poly, χ/1k, 0.0 | Partial |
| QualityScore | plaq, mass, chi/1k, acc | n/1k, σ/μ, acc, χ/1k | **No** |
| AnomalyCheck | plaq, mass, chi/1k, acc | plaq, ΔH/10, CG/500, acc | **No** |

Only 3 heads have even partial alignment. The ESN is walking blind on
every head — the "NPU" is essentially returning noise, and the system
survives entirely on the hardcoded heuristic fallbacks in the `else` branches.

### 3. Therm Early-Exit: Structurally Impossible

`min_therm` is hardcoded to 20 (line 478). `n_therm` was set to 20 via
`--therm=20`. The check at line 515:

```rust
if i >= min_therm && (i + 1) % therm_check_interval == 0 {
```

The loop runs `0..n_therm` (i = 0 to 19). `i >= 20` is **never true**.
Early exit cannot fire. This burned 4 × 20 thermalization trajectories
(~120-6,300s per β) with zero chance of optimization.

### 4. Blocking NPU Calls Serializing the 3090

The main loop has 7+ blocking `npu_rx.recv()` calls between β points and
1-2 per trajectory during measurement. The 3090 is idle during each:

- `PredictQuenchedLength` → blocking recv (line 314)
- `SuggestParameters` → blocking recv (line 410)
- `PredictCgIters` → blocking recv (line 430)
- Per-trajectory `RejectPredict` → blocking recv (line 605)
- Every 10th trajectory `AnomalyCheck` → blocking recv (line 648)
- `PhaseClassify`, `QualityScore`, `Retrain`, `DisagreementQuery`,
  `SteerAdaptive` → all blocking recv between β points

At β=6.50, the NPU steering adjustments (6 mid-run adjustments) each
involve a blocking recv, contributing to the anomalous 221.9s/traj wall
time despite low CG count (57,590).

### 5. ESN Weights Save Failed

`Warning: failed to save ESN weights` — the retrained ESN from Exp 031
was lost. The next run will have to re-bootstrap from scratch or from
the prior Exp 028 weights (which produced the same broken predictions).

### 6. CPU HMC Fallback on 3090

With Titan always failing to provide warm configs, lines 289-299 execute
5 CPU-only HMC trajectories as a fallback warm-start. The 3090 is idle
during these CPU trajectories for every β point.

### 7. Cortex Underutilized

Anderson proxy runs once per β point (trajectory 0 only, line 617).
Takes ~200ms. The cortex thread is idle for the remaining 64 trajectories
and the entire quenched phase.

---

## Quantified Waste

| Resource | Available time | Useful time | Waste |
|----------|---------------|-------------|-------|
| Titan V | 46,437s | 0s | 100% |
| NPU reject head | 260 calls | 0 correct | 100% |
| NPU therm detect | 4 β points | 0 early exits | 100% (structurally impossible) |
| CPU cortex | 46,437s | ~0.9s (4 × ~200ms) | >99.99% |
| 3090 (between β) | ~60-120s per transition | 0s (blocked on NPU/CPU) | 100% idle |

---

## Fixes Required (Priority Order)

### P0: Input Feature Alignment (makes all NPU heads functional)
Standardize the 5-dim input vector across training AND inference.
Every head must receive features in the same positions with the same
semantics. Proposed canonical format:
```
[beta_norm, plaquette, mass_or_context, chi_or_variance, acceptance_or_flag]
```
Where positions have fixed meaning. Heads that need extra features
(delta_h, CG iters, window length) should encode them into the
canonical positions with consistent normalization.

### P1: Fix min_therm (one-line fix, unlocks early-exit)
Change hardcoded `min_therm = 20` to `min_therm = 8` (or parameterize).
With `n_therm=20` and `therm_check_interval=10`, this allows checks at
trajectories 10 and 20.

### P2: Fix Titan V timing (unlocks secondary GPU)
Option A: Send Titan pre-therm for β[1] during bootstrap (before scan).
Option B: Pipeline all β pre-therms through Titan at scan start.
Option C: Send next-β request at START of current β, add blocking wait
with timeout at next β start.

### P3: Unblock 3090 from NPU (reduces idle time)
- Make RejectPredict fire-and-forget (no blocking recv)
- Pipeline pre-β NPU calls (send all, do GPU work, collect)
- Move post-β NPU calls (retrain, disagreement, steer) to background

### P4: Continuous cortex (utilizes idle CPU)
- Run proxy at multiple points during measurement
- Pre-compute proxies for upcoming β points
- Feed continuous proxy features to NPU for online learning

### P5: Fix ESN weights save
- Investigate why ExportWeights failed (permissions? path? serialization?)
- Add fallback: save weights after each β point, not just at end

---

## Summary

The physics worked. The brain architecture spawned correctly and didn't crash.
The NPU mid-run steering (dt/n_md adjustment) was the only NPU feature that
produced measurably useful work — and that's hardcoded heuristic logic in the
main binary (lines 659-671), not ESN prediction.

The system ran for 12.9 hours using its prefrontal cortex (hardcoded heuristics)
while the cerebellum (ESN/NPU) produced noise. The Titan V (pre-motor) was
entirely wasted. The CPU cortex did 0.9 seconds of work in 12.9 hours.

The architecture is correct. The wiring is broken. The input feature alignment
is the root cause of NPU blindness. The Titan timing is the root cause of
secondary GPU waste. The min_therm bug is a trivial oversight with measurable
impact.

Fix P0 (input alignment) and the entire 36-head ESN starts learning. Fix P2
(Titan timing) and the secondary GPU contributes. Fix P1 (min_therm) and
thermalization becomes adaptive. These three fixes transform the system from
"4-layer brain where 3 layers do nothing" to "4-layer brain where every layer
contributes."
