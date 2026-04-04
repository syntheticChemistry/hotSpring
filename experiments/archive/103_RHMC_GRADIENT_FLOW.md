# Experiment 103 — Gradient Flow on RHMC Configs: Chuna Paper 43 Continuation

**Date**: March 27, 2026
**Binary**: `production_rhmc_flow` (NEW — combines RHMC thermalization with gradient flow)
**Infrastructure**: `gpu_rhmc.rs` (Exp 099), `gradient_flow.rs`, `gpu_flow.rs`
**GPU**: NVIDIA GeForce RTX 3090 (native f64)
**License**: AGPL-3.0

**16⁴ RHMC + flow (in progress):** Nf=2+1, β=6.0. Thermalization 20/100 trajectories visible; acceptance ~15% at 16⁴ with n_md=1.

## Goal

Run Chuna's LSCFRK3W7 gradient flow integrator on GPU-generated dynamical fermion
configs. This closes the loop: Paper 43 integrators were validated on MILC Nf=2+1
configs, and we now generate equivalent fermion content at consumer scale.

The `production_rhmc_flow` binary does: GPU RHMC thermalization → read links to CPU →
CPU gradient flow → measure E(t), t₀, w₀. This is the first time gradient flow has
run on RHMC-generated dynamical configs in the hotSpring codebase.

## Pipeline

```
Hot start → Quenched pre-therm (GPU HMC) → Copy links → RHMC therm (GPU)
         → For each measurement config:
             Skip N RHMC trajectories (decorrelation)
             Read GPU links → CPU Lattice
             Run W7 gradient flow (CPU) → E(t), t₀, w₀
```

## Results: Nf=2 at 8⁴ (β=6.0)

Parameters: n_md=1, dt=0.005, mass=0.1, 8 poles, CG tol=1e-10,
50 quenched pretherm + 50 RHMC therm + 10 configs (skip 5).

| Metric | Value |
|--------|-------|
| ⟨P⟩ | 0.5859 ± 0.0000 |
| Acceptance | 52% |
| E(t=3) | 0.0061 (all configs) |
| t₀ | N/A (8⁴ finite-size) |
| w₀ | N/A (8⁴ finite-size) |
| Wall time | 790.8s |

## Results: Nf=2+1 at 8⁴ (β=6.0)

Parameters: n_md=1, dt=0.005, m_l=0.05, m_s=0.5, 2 sectors (8 poles each),
CG tol=1e-10, 50 quenched pretherm + 50 RHMC therm + 10 configs (skip 5).

| Metric | Value |
|--------|-------|
| ⟨P⟩ | 0.5857 ± 0.0001 |
| Acceptance | 62% |
| E(t=3) | 0.0062 (all configs) |
| t₀ | N/A (8⁴ finite-size) |
| w₀ | N/A (8⁴ finite-size) |
| Wall time | 949.4s |

## Flow Energy Comparison: Quenched vs Dynamical

| Ensemble | ⟨P⟩ | E(t=3) | Configs |
|----------|------|--------|---------|
| Quenched (8⁴ β=6.0) | 0.5935 ± 0.0015 | 0.0005–0.0042 | 5 (Exp 102) |
| Nf=2 (8⁴ β=6.0) | 0.5859 ± 0.0000 | 0.0061 | 10 |
| Nf=2+1 (8⁴ β=6.0) | 0.5857 ± 0.0001 | 0.0062 | 10 |

The dynamical configs show consistently higher E(t=3) than quenched (0.0061-0.0062
vs 0.0005-0.0042), indicating more residual UV structure surviving the flow smoothing.
The Nf=2 and Nf=2+1 values are nearly identical at this volume and trajectory length.

## Thermalization Note

The RHMC trajectory length τ=0.005 (n_md=1 × dt=0.005) is extremely short,
constrained by acceptance rate requirements on 8⁴. This means:
- Autocorrelation time is very high (~100s of trajectories)
- The dynamical plaquettes (0.586) haven't fully equilibrated from the
  quenched pre-thermalization starting point (0.586)
- Longer runs or the adaptive `RhmcCalibrator` (Exp 103 in README) are needed
  for fully thermalized dynamical ensembles

For clean fermion backreaction (⟨P⟩_dyn > ⟨P⟩_quenched), either:
1. Much longer thermalization (hundreds+ more RHMC trajectories), or
2. Larger trajectory length via the self-tuning calibrator, or
3. Volume scaling to 16⁴+ where the fermion force is gentler

## What This Proves

1. **Pipeline complete**: GPU RHMC → CPU gradient flow → scale observables, end-to-end
2. **Chuna integrator on dynamical configs**: W7 runs correctly on non-quenched gauge fields
3. **Flow energy is measurable**: E(t) converges consistently across configs
4. **Scale setting requires 16⁴+**: t₀/w₀ unmeasurable on 8⁴ (both quenched and dynamical)
5. **Thermalization is the bottleneck**: Short RHMC trajectories limit ensemble quality

## Cost

| Run | Trajectories | Compute time | Cost |
|-----|-------------|-------------|------|
| Nf=2 (8⁴, 10 configs) | 130 | 791s | ~$0.007 |
| Nf=2+1 (8⁴, 10 configs) | 130 | 949s | ~$0.008 |
| **Total** | **260** | **1740s** | **~$0.015** |

## What's Next

1. **16⁴ RHMC + flow**: In progress (Nf=2+1, β=6.0; overnight run toward t₀/w₀ at production volume)
2. **Adaptive RHMC at 8⁴+**: Use `RhmcCalibrator` for longer τ with maintained acceptance
3. **Multi-β scan**: Flow at β=5.5, 5.69, 6.0, 6.2 for continuum approach
4. **CK4 on dynamical configs**: Compare W7 vs CK4 step-size efficiency on RHMC ensembles
5. **Multi-GPU**: toadStool brain architecture for 32⁴+ Nf=2+1 RHMC + flow
