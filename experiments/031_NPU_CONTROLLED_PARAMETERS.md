# Experiment 031: NPU-Controlled HMC Parameters

**Status:** IN PROGRESS (started 2026-03-01)
**Binary:** `production_dynamical_mixed`
**Predecessor:** Exp 030 (brain architecture, but NPU param suggestions ignored)

## Motivation

Exp 030 demonstrated the 4-layer brain architecture (RTX 3090 motor, Titan V
pre-motor, CPU cortex, AKD1000 cerebellum) was operational, but analysis
revealed a critical inefficiency: HMC acceptance was 97.5% — far above the
60-80% sweet spot. This wasted ~2x CG iterations per useful trajectory because
the step size `dt` was over-conservative (0.0032 due to the old
`mass_scale.sqrt()` penalty) and the NPU's `ParameterSuggestion` was printed
but never applied.

This experiment makes the NPU the **actual controller** of HMC parameters:

### Changes to the binary

1. **Fixed auto-dt formula:** Removed `mass_scale.sqrt()` penalty.
   `auto_dt = (0.01 * scale).max(0.002)` — lets the NPU refine from data.

2. **dt and n_md are now mutable:** NPU-suggested values are applied per-beta
   with safety clamps (`dt ∈ [0.001, 0.02]`, `n_md ∈ [20, 500]`).

3. **Mid-beta feedback loop:** Every 10 measurement trajectories, if acceptance
   > 85% → dt bumps 15%; if < 50% → dt drops 15%. Trajectory length
   (`dt * n_md`) is kept roughly constant by adjusting `n_md` inversely.

4. **Improved ESN training target:** Instead of `0.01 + acc * 0.04` (crude
   linear guess), the target is `dt_used * (1 - 0.5 * (acc - 0.70))` — trains
   the ESN to predict the dt that would have given 70% acceptance.

5. **dt_used and n_md_used recorded in BetaResult:** Enables post-hoc analysis
   of how the NPU adapted parameters across the phase curve.

6. **`--no-npu-control` safety flag:** Reverts to old behavior (print-only).

7. **Titan V pre-therm uses NPU-controlled dt:** The pre-motor thread now
   receives the current (adapted) dt for the next beta point.

## Configuration

```
--betas=4.5,5.25,5.69,6.5
--lattice=8
--mass=0.1
--therm=20
--quenched-pretherm=10
--meas=65
--seed=4001
--cg-tol=1e-8
--cg-max-iter=65000
--max-adaptive=12
--bootstrap-from=exp028_brain_weights.json,exp024_production_8x8.jsonl,
                 exp028_brain_production_8x8.jsonl,exp029_npu_steering_8x8.jsonl,
                 exp030_adaptive_steering_8x8.jsonl
--save-weights=exp031_brain_weights.json
--output=exp031_npu_controlled_params_8x8.jsonl
--trajectory-log=exp031_npu_controlled_params_8x8.jsonl
```

No `--dt` or `--n-md` override — the NPU controls from first beta onward.

## Bootstrap

Bootstraps from all prior data (Exps 024, 028, 029, 030) so the ESN has a
trained dt-prediction model from the start.

## What the NPU Controls

| Parameter       | Before (Exp 030)           | After (Exp 031)                                |
| --------------- | -------------------------- | ---------------------------------------------- |
| dt              | Fixed at startup           | NPU-suggested per-beta + mid-run adaptation    |
| n_md            | Fixed at startup           | Derived from dt to keep trajectory length ~1.0  |
| check_interval  | Partially from CG estimate | Already wired (no change)                      |
| quenched length | NPU predicts budget        | Already wired (no change)                      |
| therm length    | NPU early exit             | Already wired (no change)                      |
| beta order      | NPU priority sort          | Already wired (no change)                      |
| adaptive betas  | NPU steers                 | Already wired (fixed in Exp 030)               |

## Expected Outcome

- Acceptance should converge to 60-80% range within a few beta points
- Effective CG cost per useful trajectory should drop ~2x vs Exp 030
- The NPU should learn beta-dependent dt preferences across the phase curve
  (larger dt in confined phase, smaller near transition)
- `dt_used` in the output will show the NPU's parameter evolution

## Key Metrics to Watch

- `acceptance` per beta (target: 60-80%)
- `dt_used` / `n_md_used` evolution across beta points
- `mean_cg_iters` compared to Exp 030 (should decrease at same beta)
- Quality scores (should improve with better-tuned HMC)
- NPU mid-run adaptation count (how often the feedback loop fires)
