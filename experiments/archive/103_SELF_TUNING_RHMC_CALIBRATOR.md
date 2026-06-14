# Experiment 103 — Self-Tuning RHMC Calibrator: Eliminating Hand-Tuned Magic Numbers

**Date**: March 27, 2026
**Modules**: `rhmc_calibrator.rs`, `spectral_probe.rs`, `tolerances/lattice.rs`
**Infrastructure**: GPU RHMC (Exp 099, 101), `production_rhmc_scan` binary
**GPU**: NVIDIA GeForce RTX 3090 (DF64)
**License**: AGPL-3.0

## Goal

Eliminate every hand-tuned magic number from the RHMC pipeline. Every simulation
parameter must be either mathematical (derived from theory), discovered (measured
from the gauge field), adapted (feedback-controlled from physics observables), or
validated (checked and auto-corrected). Physics is the only validator. No human
guessing required.

This directly addresses the reproducibility problem: hand-tuned parameters are
non-reproducible work. A tuning system that discovers its own parameters from
physics makes every run fully determined by the theory and the hardware.

## Background

Exp 101 demonstrated GPU RHMC production for Nf=2 and Nf=2+1 on consumer hardware.
The runs used hand-selected parameters:

| Parameter | Exp 101 value | Source |
|-----------|:------------:|--------|
| Spectral range [a, b] | [0.01, 64.0] | Hardcoded guess |
| n_poles | 8 | Experience |
| dt | 0.005–0.01 | Trial and error |
| n_md_steps | 5–10 | Trial and error |
| CG tolerance | 1e-6 | Convention |

These work for 4⁴ and 8⁴ at moderate β. They will fail at 16⁴+, at physical
quark masses, or on different hardware. The self-tuning calibrator replaces all
of them with physics-derived values.

## Architecture: Parameter Classification

Every RHMC parameter falls into exactly one category:

### Mathematical (from theory, never changes)

- Omelyan integrator parameter λ = 0.1932
- Determinant power: `det_power = Nf/8` (rooting trick)
- Rational approximation: `x^{-α}` for action/force, `x^{α/2}` for heatbath
- Consistency relation: `r_hb² · r_act = 1`

### Discovered (measured from the gauge field)

- **λ_max(D†D)**: Largest eigenvalue of the squared Dirac operator. Measured
  via GPU power iteration (20 iterations of v → D†D·v / ‖D†D·v‖). Converges
  geometrically. Cost: ~20 Dirac applications (cheap vs one CG solve).

- **λ_min(D†D)**: Smallest eigenvalue. Bounded analytically: λ_min ≥ m² for
  positive-mass staggered fermions (free-field bound, tight at weak coupling).
  Interaction effects can only increase λ_min.

- **Spectral range**: [safety_low × λ_min, safety_high × λ_max] with
  configurable safety factors (0.5× low, 1.5× high).

### Adapted (feedback-controlled from observables)

- **dt** (step size): Driven by acceptance rate. High acceptance (>85%) → bump
  dt by 10%. Low acceptance (<50%) → drop dt by 15%. Emergency: |ΔH| > 2 →
  scale dt toward |ΔH| ≈ 0.5 using the Omelyan scaling |ΔH| ∝ dt².

- **n_md_steps**: Coupled to dt to preserve trajectory length τ = dt × n_md.
  Clamped to [2, 100].

- **n_poles**: Driven by consistency check. If S_f(old)/η†η deviates from 1.0
  by more than 5%, or if max_relative_error exceeds 1e-3, add 2 poles (up to 24).

### Validated (checked, never guessed)

- CG tolerance: 1e-6 for force (MD integration error ∝ dt² dominates),
  1e-8 for Metropolis (detailed balance requires accurate ΔH).
- Spectral re-probe every 50 trajectories (detect drift as gauge field evolves).
- All thresholds are named constants in `tolerances/lattice.rs` with physics
  justifications in doc comments.

## Implementation

### SpectralProbe (`spectral_probe.rs`)

```
SpectralInfo {
    lambda_min: f64,     // analytical bound m²
    lambda_max: f64,     // GPU power iteration
    range_min: f64,      // safety_low × lambda_min
    range_max: f64,      // safety_high × lambda_max
    lambda_min_is_bound: bool,
}
```

- `gpu_power_iteration_lambda_max`: Deterministic initial vector (golden-ratio
  quasi-random), reuses existing Dirac dispatch + dot product pipelines. No
  new shaders needed.
- `probe_spectral_range`: Combines power iteration λ_max with analytical m²
  bound for λ_min. Safety margins from tolerance constants.

### RhmcCalibrator (`rhmc_calibrator.rs`)

Central orchestrator. Takes physics parameters (nf, mass, β, dims) and
produces `RhmcConfig` with all derived parameters:

```
let mut cal = RhmcCalibrator::new(2, mass, beta, dims);
let spectral = cal.calibrate_spectral(&gpu, &pipelines, &state);
for _ in 0..n_trajectories {
    let config = cal.produce_config();
    let result = gpu_rhmc_trajectory(&gpu, ..., &config, &mut seed);
    cal.observe(&result);
    if cal.needs_spectral_reprobe() {
        cal.reprobe_spectral(&gpu, &pipelines, &state);
    }
}
```

Sub-components integrated into the calibrator struct:
- **SpectralProbe**: eigenvalue discovery
- **StepController**: dt/n_md from acceptance + ΔH
- **ApproxMonitor**: pole count from consistency ratio
- **SolverCalibrator**: CG tolerance split (force vs Metropolis)

### Tolerance Constants (`tolerances/lattice.rs`)

12 new named constants, each with physics justification:

| Constant | Value | Role |
|----------|:-----:|------|
| `RHMC_APPROX_ERROR_THRESHOLD` | 1e-3 | Trigger pole increase |
| `RHMC_SPECTRAL_SAFETY_LOW` | 0.5 | 2× margin below λ_min |
| `RHMC_SPECTRAL_SAFETY_HIGH` | 1.5 | 50% margin above λ_max |
| `RHMC_POWER_ITERATION_COUNT` | 20 | Converges λ_max to ~1e-6 |
| `RHMC_SPECTRAL_REPROBE_INTERVAL` | 50 | Trajectories between re-probes |
| `RHMC_CG_TOL_FORCE` | 1e-6 | Loose (integrator error dominates) |
| `RHMC_CG_TOL_METROPOLIS` | 1e-8 | Tight (detailed balance) |
| `RHMC_CONSISTENCY_THRESHOLD` | 0.05 | S_f/η†η deviation trigger |
| `RHMC_POLE_INCREMENT` | 2 | Poles added per upgrade |
| `RHMC_MAX_POLES` | 24 | Marginal returns beyond this |

### Production Integration (`production_rhmc_scan`)

New `--adaptive` CLI flag. When enabled:
1. Quenched pre-thermalization (same as before)
2. `cal.calibrate_spectral()` on pre-thermalized config
3. Each trajectory: `cal.produce_config()` → run → `cal.observe()`
4. Periodic `cal.reprobe_spectral()` every 50 trajectories
5. Progress logging includes dt, n_md, poles, spectral range

Fixed-parameter mode remains the default for backward compatibility.

### Telemetry Extensions (`gpu_rhmc.rs`)

`GpuRhmcResult` extended with Hamiltonian decomposition fields:
- `s_gauge_old`, `s_gauge_new` — gauge action before/after
- `t_old`, `t_new` — kinetic energy before/after
- `s_ferm_old`, `s_ferm_new` — fermion action before/after

These enable the calibrator to diagnose which Hamiltonian component is
driving large |ΔH|.

## Design Invariants

1. **No parameter requires human guessing** — physics or measurement provides everything
2. **Physics is always the validator** — acceptance rate, ΔH, consistency ratio
3. **Graceful degradation** — works without NPU; conservative defaults if spectral probe fails
4. **All thresholds are named constants** — documented, discoverable, evolvable
5. **Reproduces Exp 101** — same physics, same results, zero hand-tuning

## Next Steps

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Spectral Discovery | GPU power iteration + analytical λ_min | ✅ Complete |
| 2. Step Controller | Acceptance-driven dt/n_md | ✅ Complete |
| 3. Approx Quality Monitor | Consistency-driven pole count | ✅ Complete |
| 4. Solver Calibrator | Force/Metropolis CG split | ✅ Complete |
| 5. NPU Bridge | Akida spectral predictions + anomaly detection | Pending (NpuBridge) |
| 6. Production Validation | Weekend runs at 16⁴+ | Pending |

## Cross-References

- **Exp 101:** GPU RHMC production (fixed parameters)
- **Exp 102:** Gradient flow at volume (consumes RHMC configs)
- **Exp 024:** HMC parameter sweep (prior adaptive step work)
- **baseCamp:** `self_tuning_rhmc.md`
- **tolerances:** `barracuda/src/tolerances/lattice.rs`
- **Adaptive precedent:** `barracuda/src/lattice/pseudofermion/adaptive.rs`
- **NPU heads:** `barracuda/src/md/reservoir/heads.rs` (A2_ANDERSON_LAMBDA_MIN)
