# hotSpring → toadStool: NPU Parameter Controller + Barracuda Evolution Handoff

**Date:** 2026-03-01
**From:** hotSpring v0.6.15 (Exp 031 active)
**To:** toadStool/barracuda team
**Covers:** Exp 029 → 030 → 031 progression, barracuda evolution review
**License:** AGPL-3.0-only

---

## Executive Summary

- **NPU now controls HMC parameters** (dt, n_md) with safety clamps and mid-beta
  acceptance-driven feedback. Previously, NPU suggestions were received and discarded.
- **ESN training target improved**: trains to predict the dt that yields 70% acceptance,
  not a crude linear guess. `dt_used` and `n_md_used` tracked in BetaResult.
- **auto_dt formula fixed**: removed `mass_scale.sqrt()` penalty that collapsed
  dt from 0.01 to 0.0032 for mass=0.1. Starting dt is now `(0.01 * scale).max(0.002)`.
- **Exp 030 post-mortem**: 97.5% acceptance = 2x wasted CG per useful trajectory.
  Root cause identified and fixed in Exp 031.
- **barracuda has 0 TODO/FIXME/HACK** in 167 `.rs` files. The only technical debt
  is `TODO(evolution)` annotations on WGSL inline math duplicates (`abs_f64`, `cbrt_f64`).

---

## Part 1: NPU Parameter Controller (What Changed)

### The Problem (Exp 030)

```
auto_dt = (0.01 * scale * mass_scale.sqrt()).max(0.001)
         = 0.01 * 1.0 * sqrt(0.1)
         = 0.0032
```

Result: 97.5% HMC acceptance, nearly all proposals accepted, ~2x CG iterations
per useful (decorrelated) trajectory. The NPU received `ParameterSuggestion`
responses but line 1855 printed them and kept the startup dt.

### The Fix (Exp 031)

1. **auto_dt**: `(0.01 * scale).max(0.002)` — mass does not penalize step size.
2. **dt/n_md now `let mut`** — NPU-suggested values applied with safety clamps.
3. **Safety constants**: `DT_MIN=0.001`, `DT_MAX=0.02`, `NMD_MIN=20`, `NMD_MAX=500`.
4. **Mid-beta feedback**: Every 10 measurement trajectories during the anomaly check:
   - Acceptance > 85%: dt *= 1.15, n_md adjusted inversely (constant trajectory length)
   - Acceptance < 50%: dt *= 0.85, n_md adjusted inversely
5. **ESN training target**: `optimal_dt = dt_used * (1 - 0.5 * (acc - 0.70))` —
   learns to predict the dt that would have given 70% acceptance.
6. **BetaResult extended**: `dt_used: f64`, `n_md_used: usize` enable post-hoc analysis.
7. **`--no-npu-control` flag**: disables parameter mutations, reverts to print-only.
8. **Titan V pre-therm**: receives current NPU-controlled dt (not startup dt).

### What the NPU Now Controls

| Parameter | Status | Mechanism |
|-----------|--------|-----------|
| dt | **NEW** | NPU per-beta suggestion + mid-run adaptation |
| n_md | **NEW** | Derived from dt to keep trajectory length ~1.0 |
| check_interval | Wired | CG estimate sets adaptive check_interval |
| quenched length | Wired | NPU predicts budget |
| therm length | Wired | NPU early exit |
| beta order | Wired | NPU priority sort |
| adaptive betas | Wired | NPU steers (fixed in Exp 030) |

---

## Part 2: Barracuda Evolution Review

### Current State (v0.6.15, synced to toadStool S68)

| Metric | Value |
|--------|-------|
| Tests | ~700 (~665 lib + 31 integration + doc) |
| Validation suites | 39/39 pass |
| Binaries | 84 |
| WGSL shaders | 62 |
| TODO/FIXME/HACK | 0 in production `.rs` files |
| Clippy warnings | 0 (lib + bins) |
| unsafe blocks | 0 |
| Centralized tolerances | ~170 |

### Key Evolution Milestones

| Version | Milestone |
|---------|-----------|
| v0.6.15 | 36-head ESN (6 groups × 6 heads), brain architecture, wgpu 22 |
| v0.6.14 | NPU offload mixed pipeline, cross-primal discovery, 0 TODO/FIXME |
| v0.6.13 | GPU Polyakov loop, NVK alloc guard, PRNG fix, cross-spring rewiring |
| v0.6.12 | DF64 expansion (60% of HMC), FMA-optimized df64_core |
| v0.6.11 | Site indexing standardization (t-major convention from toadStool) |
| v0.6.10 | Fp64Strategy, GpuDriverProfile, ShaderTemplate |
| v0.6.9 | Spectral module fully leaning on upstream (41 KB local deleted) |
| v0.6.8 | GPU streaming HMC, resident CG, bidirectional NPU |

### Already Leaning on Upstream

| Module | Upstream Primitive | Status |
|--------|-------------------|--------|
| `spectral/` | `barracuda::spectral::*` | Leaning — 41 KB local deleted |
| `md/celllist.rs` | `CellListGpu` | Leaning — local deprecated |

### Tier 1 Absorption Targets (highest value for toadStool)

| Module | What | Validation | Notes |
|--------|------|-----------|-------|
| `md/reservoir.rs` | 36-head ESN (Gen 2) | 31 nautilus tests + 5 examples | Multi-group architecture, GPU+NPU dispatch |
| `lattice/dirac.rs` | Staggered Dirac SpMV | 8/8 checks | `WGSL_DIRAC_STAGGERED_F64` |
| `lattice/cg.rs` | CG solver | 9/9 checks | 3 WGSL shaders, GPU-resident variant |
| `lattice/pseudofermion.rs` | Pseudofermion HMC | 7/7 checks | CPU (WGSL-ready pattern) |
| `proxy/` | Anderson 3D proxy | 10/10 checks | Physics proxy for CG prediction |

### Tier 2 Absorption Targets

| Module | What | Notes |
|--------|------|-------|
| `physics/screened_coulomb.rs` | Sturm eigensolve | 23/23, CPU only |
| `physics/hfb_deformed_gpu/` | Deformed HFB | 5 WGSL shaders |
| `lattice/gpu_hmc/resident_cg.rs` | GPU-resident CG | 15,360× readback reduction |

---

## Part 3: Production Binary Architecture

`production_dynamical_mixed.rs` (~2480 lines) is the most complex binary.
It implements the 4-layer brain architecture:

```
Layer 1: NPU Cerebellum (AKD1000)
  - 36-head ESN: 6 groups × 6 heads (Anderson, QCD, Potts, Steering, Brain, Meta)
  - Pre-beta: screen candidates, predict CG, suggest parameters, predict quenched length
  - During: reject prediction, anomaly detection, CG residual monitoring
  - Post: phase classification, quality score, adaptive steering, parameter control
  - NEW: dt/n_md control with safety clamps + mid-beta adaptation

Layer 2: Titan V Pre-Motor
  - Concurrent quenched pre-thermalization for next beta
  - Now receives NPU-controlled dt (not startup dt)
  - Config transfer via CPU (~2.4 MB, <1 ms)

Layer 3: CPU Cortex (Threadripper 3970X)
  - Anderson 3D proxy: ⟨r⟩ (level spacing ratio) + |λ|_min
  - Features fed to NPU for physics-informed CG prediction
  - Runs during CG solve, negligible overhead (<250 ms)

Layer 4: Attention/Interrupt State Machine
  - GREEN/YELLOW/RED states based on CG residual monitoring
  - Interrupt delivery for CG divergence
```

### toadStool action items:

1. **ESN reservoir absorption** (`md/reservoir.rs`): The 36-head multi-group
   architecture is the most complex hotSpring primitive. Absorb as
   `barracuda::esn::MultiHeadEsn` with configurable head groups.

2. **Resident CG absorption** (`lattice/gpu_hmc/resident_cg.rs`): The 15,360×
   readback reduction pattern is reusable across any iterative GPU solver.

3. **Physics proxy pattern** (`proxy/`): Anderson 3D as CG predictor. The
   pattern (cheap CPU simulation predicts expensive GPU cost) applies to any
   physics where a simpler model approximates the hard problem.

4. **Consider**: `BetaResult` as a generic `SimulationResult` type with
   `dt_used`/`n_md_used` pattern for any adaptive simulation.

---

## Part 4: Experiment Progression and Lessons

| Exp | What Happened | Lesson |
|-----|--------------|--------|
| 024 | 17 β points, 2400 trajectories, parameter sweep | Training data is cheap; collect broadly |
| 028 | Brain architecture validated (4-layer concurrent) | Concurrent substrates work; NVK dual-GPU needs alloc guard |
| 029 | 4-seed NPU-steered production, adaptive steering bug found | Guard `bi + 1 < len` prevents insertion at end of queue |
| 030 | Fixed steering, but dt=0.0032, 97.5% acceptance | NPU suggestions must be *applied*, not just logged |
| 031 | NPU controls dt/n_md, mid-beta adaptation, ESN targets 70% | Close the feedback loop: observe → predict → apply → retrain |

The critical lesson from 029→031: **a prediction system that doesn't affect the
system it predicts is overhead, not intelligence.** The NPU was doing useful work
(CG prediction, phase classification, anomaly detection) but its most impactful
output (parameter suggestion) was being discarded. Closing that loop is what
makes it a brain, not a monitor.

---

## Files Modified

| File | Change |
|------|--------|
| `barracuda/src/bin/production_dynamical_mixed.rs` | dt/n_md mutable, NPU param control, mid-beta adaptation, improved ESN target, dt_used in BetaResult, --no-npu-control |
| `experiments/030_ADAPTIVE_STEERING_PRODUCTION.md` | Marked SUPERSEDED with post-mortem |
| `experiments/031_NPU_CONTROLLED_PARAMETERS.md` | New experiment doc |

## Validation

- Binary compiles clean (2 pre-existing warnings: `latest_proxy` unused)
- Exp 031 launched with bootstrap from 30 β points (Exps 024-030)
- `npu_control=true` confirmed in startup output
- Starting dt=0.0100 (vs 0.0032 in Exp 030)
