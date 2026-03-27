# Self-Tuning RHMC — Physics-Validated Parameter Discovery

**Domain:** Lattice QCD self-tuning and adaptive simulation
**Updated:** March 27, 2026
**Status:** Phase 1-4 complete — spectral discovery, step adaptation, approximation quality, solver calibration
**Hardware:** RTX 3090 (GPU), BrainChip AKD1000 (NPU integration pending)
**Protocol:** Physics as sole validator; zero hand-tuned magic numbers

---

## Thesis

HPC lattice QCD codes require expert-tuned simulation parameters: step sizes,
CG tolerances, rational approximation ranges, pole counts. These are chosen by
experienced practitioners through trial runs and domain intuition. The knowledge
is non-reproducible — it lives in notebooks, scripts, and institutional memory.

The self-tuning RHMC calibrator replaces human guessing with a feedback loop:
the simulation measures its own physics observables (eigenvalue spectrum,
acceptance rate, Hamiltonian conservation, heatbath-action consistency) and
adjusts every parameter from those measurements. The result is a system where
the input is `(Nf, mass, β, lattice_dims)` and the output is a fully
configured, physics-validated RHMC simulation.

This is the ecoPrimals philosophy applied to simulation methodology: constrained
evolution (acceptance rate targets, consistency thresholds) replacing
unconstrained human choices. The NPU bridge (Phase 5) extends this to
neuromorphic prediction of spectral properties, closing the loop between
hardware intelligence and simulation configuration.

---

## What Changed From Exp 101 → Exp 103

### Before (Exp 101): Hand-tuned fixed parameters

```
RhmcConfig::nf2(mass, beta)
  spectral range: [0.01, 64.0]   ← hardcoded guess
  n_poles: 8                      ← experience
  dt: 0.005                       ← trial and error
  n_md_steps: 10                  ← trial and error
  cg_tol: 1e-6                    ← convention
```

Works at 4⁴/8⁴ with moderate quark masses. Fails silently at 16⁴+ or
physical quark masses (spectral range too wide → wasted poles; dt too
large → low acceptance; dt too small → autocorrelation).

### After (Exp 103): Physics-discovered parameters

```
let mut cal = RhmcCalibrator::new(2, mass, beta, dims);
cal.calibrate_spectral(&gpu, &pipelines, &state);
// spectral range: [0.5×m², 1.5×λ_max]  ← measured
// n_poles: 8 (auto-increases if needed)  ← consistency-checked
// dt: volume-scaled initial, adapted     ← acceptance-driven
// n_md_steps: τ-preserving              ← coupled to dt
// cg_tol: 1e-6 force, 1e-8 Metropolis   ← physics-justified split
```

Self-corrects as gauge field evolves. Spectral re-probe every 50
trajectories detects drift. Emergency dt scaling prevents runaway |ΔH|.

---

## Parameter Classification

Every RHMC parameter falls into exactly one of five categories:

### 1. Mathematical (Theory)

| Parameter | Value | Derivation |
|-----------|-------|------------|
| Omelyan λ | 0.1932 | Optimal 2nd-order symplectic parameter |
| det_power | Nf/8 | Rooting trick for staggered fermions |
| Rational powers | −α (action/force), +α/2 (heatbath) | Detailed balance |
| Consistency: r_hb² · r_act = 1 | Exact | Identity of approximation |

### 2. Discovered (Measured)

| Parameter | Method | Cost |
|-----------|--------|------|
| λ_max(D†D) | GPU power iteration (20 Dirac applications) | ~1 CG solve |
| λ_min(D†D) | Analytical: λ_min ≥ m² (staggered, positive mass) | Zero |
| Spectral range | [0.5 × λ_min, 1.5 × λ_max] with safety margins | Trivial |

### 3. Adapted (Feedback-Controlled)

| Parameter | Observable | Response |
|-----------|-----------|----------|
| dt | Acceptance rate | >85%: bump 10%, <50%: drop 15% |
| dt (emergency) | |ΔH| > 2 | Scale toward |ΔH| ≈ 0.5 via dt² scaling |
| n_md_steps | Coupled to dt | Preserve τ = dt × n_md |
| n_poles | Consistency ratio | S_f/η†η > 5% off → add 2 poles |

### 4. Validated (Checked)

| Parameter | Check | Threshold |
|-----------|-------|-----------|
| max_relative_error | Exceeds 1e-3 | Regenerate with more poles |
| Spectral range | Re-probe every 50 trajectories | Detect >20% drift |
| CG convergence | max_iter = 5000 | Log warning if reached |

### 5. Learned (NPU — Phase 5)

| Parameter | NPU Head | Status |
|-----------|----------|--------|
| λ_min prediction | A2_ANDERSON_LAMBDA_MIN | Pending |
| Anomaly detection | ANOMALY_DETECT | Pending |
| CG iteration estimate | CG_ESTIMATE | Pending |
| Parameter suggestion | PARAM_SUGGEST | Pending |

---

## How It Works: The Feedback Loop

```
  ┌─────────────────────────────────────────────────────────┐
  │                   RhmcCalibrator                         │
  │                                                         │
  │  Physics params ──→ Initial heuristics (conservative)   │
  │        ↓                                                │
  │  SpectralProbe ───→ λ_min, λ_max (GPU measurement)     │
  │        ↓                                                │
  │  produce_config() → RhmcConfig (all derived)            │
  │        ↓                                                │
  │  gpu_rhmc_trajectory() → result (accepted, ΔH, S_f)    │
  │        ↓                                                │
  │  observe(result) ──→ StepController (dt, n_md)          │
  │                  ──→ ApproxMonitor (n_poles)             │
  │                  ──→ SolverCalibrator (cg_tol)           │
  │        ↓                                                │
  │  [every 50 traj] ──→ reprobe_spectral() ─→ update range │
  │        ↓                                                │
  │  [NPU available] ──→ NpuBridge predictions (Phase 5)    │
  │        ↓                                                │
  │  Loop: produce_config() → run → observe → adapt         │
  └─────────────────────────────────────────────────────────┘
```

---

## Connection to Akida / NPU

The NPU bridge (Phase 5) integrates neuromorphic predictions into the
calibrator. The existing NPU head infrastructure (`heads.rs`) already
defines the relevant output categories:

- **A2_ANDERSON_LAMBDA_MIN**: Spectral minimum prediction from gauge
  field features. Could replace or cross-check the analytical m² bound.
- **ANOMALY_DETECT**: Flag configurations where the gauge field has
  evolved into a regime where current parameters are unsafe.
- **CG_ESTIMATE**: Predict CG iteration count before solving, enabling
  preemptive tolerance adjustment.
- **PARAM_SUGGEST**: Direct dt/n_md suggestions from ESN trained on
  trajectory history.

The calibrator degrades gracefully without NPU — all four parameter
categories (mathematical, discovered, adapted, validated) function on
GPU alone. The NPU adds a fifth category (learned) that accelerates
convergence to optimal parameters.

---

## Evolution Path

### Volume scaling (next)

At 16⁴+ (65K+ sites), the spectral range widens (λ_max grows with volume),
quark mass effects become more pronounced (λ_min shrinks toward m²), and
CG iteration counts increase. The calibrator adapts automatically — the
spectral probe measures the actual range, the step controller finds the
right dt, and the consistency monitor ensures the pole count is sufficient.

### Physical quark masses (future)

At physical m_π ≈ 135 MeV, λ_min(D†D) drops by 1-2 orders of magnitude
compared to our current m=0.05-0.5 test masses. The hardcoded [0.01, 64]
range would be grossly wrong. The calibrator's spectral discovery handles
this automatically.

### Multi-GPU (fleet)

The calibrator state serializes trivially (it's just numbers). On a
multi-GPU fleet via toadStool, each GPU runs trajectories and reports
results. The calibrator can aggregate observations from multiple
devices, converging faster on optimal parameters.

---

## Cross-References

- **Experiments:** 099 (RHMC infrastructure), 101 (production), 102 (gradient flow), 103 (self-tuning)
- **Modules:** `barracuda/src/lattice/gpu_hmc/rhmc_calibrator.rs`, `spectral_probe.rs`
- **Tolerances:** `barracuda/src/tolerances/lattice.rs` (12 named constants)
- **Adaptive precedent:** `barracuda/src/lattice/pseudofermion/adaptive.rs`
- **NPU heads:** `barracuda/src/md/reservoir/heads.rs`
- **NPU steering:** `barracuda/src/lattice/pseudofermion/npu_steering.rs`
- **toadStool surface:** `compute.rhmc.calibrate`, `compute.rhmc.spectral_probe`
- **baseCamp paper 10:** `10_dynamical_qcd_production.md` (prior production work)
- **gen3 baseCamp:** `25_self_tuning_simulation.md` (cross-spring implications)
