# Experiment 024: HMC Parameter Sweep — NPU Training Data

**Status:** IN PROGRESS
**Date:** February 27, 2026
**Depends on:** Exp 023 (pseudofermion heatbath fix)
**License:** AGPL-3.0-only

---

## Motivation

Exp 023 revealed that HMC step size (dt) tuning for dynamical fermions is
critical and non-trivial. The quenched dt=0.05 is catastrophically unstable
for dynamical (ΔH ~ millions). The fermion force magnitude depends on:

- **β** (coupling): force structure changes near the crossover
- **m** (fermion mass): force ∝ 1/m for small masses
- **L** (lattice size): finite-volume effects change the spectrum
- **dt** (step size): acceptance ∝ exp(-ΔH), ΔH ∝ dt⁴ for Omelyan

This experiment systematically sweeps these parameters to:
1. Map the acceptance surface: acc(β, m, L, dt)
2. Find the optimal dt for each (β, m, L) — maximize acc × traj/sec
3. Generate training data for NPU Head 6 (SuggestParameters)
4. Build a cost model: wall_time(β, m, L, dt, n_md)

## Design

### Sweep Grid

| Parameter | Values | Notes |
|-----------|--------|-------|
| Lattice L | 4, 8 | Quick scan on small lattices |
| β | 5.0, 5.5, 5.69, 6.0 | Confined → crossover → deconfined |
| mass m | 0.05, 0.1, 0.2, 0.5 | Light to heavy staggered |
| dt | 0.002, 0.005, 0.01, 0.02, 0.05 | Conservative to aggressive |
| n_md | 1.0/dt (trajectory length = 1.0) | Fixed trajectory length |

**Total configurations:** 2 × 4 × 4 × 5 = 160 points
**Trajectories per point:** 5 therm + 10 meas = 15
**Total trajectories:** 2,400

### Measurements Per Point

For each (L, β, m, dt) we record:
- Mean ΔH and std(ΔH)
- Acceptance rate
- Mean CG iterations per trajectory
- Mean wall time per trajectory
- Mean plaquette and Polyakov loop
- CG convergence rate (iterations per solve)

### Output Format

JSONL file: one line per (L, β, m, dt) combination with all measurements.
This becomes the NPU training dataset for Head 6 (SuggestParameters).

### NPU Integration

After the sweep, the data feeds directly into the ESN:

**Input features (per point):**
- β, m, L⁴ (volume), L (linear size)
- Previous ΔH (if available from nearby point)

**Output targets:**
- Optimal dt (maximizing acceptance × throughput)
- Expected CG iterations
- Expected acceptance rate
- Expected ΔH

The NPU can then predict the optimal dt for any new (β, m, L) before
the GPU starts the trajectory — zero-overhead parameter suggestion.

## Binary

`production_dynamical_sweep` — standalone parameter sweep binary.
Iterates over the grid, runs short trajectories, collects statistics,
writes JSONL. Designed to complete in ~2-4 hours on 8⁴.

## Predictions

1. **Acceptance vs dt:** Expect sigmoidal drop-off. For m=0.1 at β=5.69,
   the critical dt should be ~0.01-0.02 (with proper heatbath).

2. **Mass dependence:** Lighter masses → smaller optimal dt. Expect
   dt_opt ∝ √m (fermion force ∝ 1/m, Omelyan error ∝ dt⁴ × F²).

3. **β dependence:** Near the crossover (β≈5.69), the force fluctuations
   are larger → need smaller dt. Well into confined or deconfined
   phase → larger dt is safe.

4. **Volume dependence:** Larger volume → more force contributions →
   slightly smaller optimal dt. But the effect is mild for Omelyan.

## Success Criteria

- [ ] Acceptance rate > 60% for at least one dt at each (β, m, L)
- [ ] ΔH < 10 for optimal dt values
- [ ] NPU Head 6 trained on sweep data achieves < 30% error on dt prediction
- [ ] Sweep completes in < 4 hours

## Files

- `barracuda/src/bin/production_dynamical_sweep.rs` — sweep binary
- `results/exp024_sweep.jsonl` — raw sweep data
- `results/exp024_sweep_summary.json` — aggregated results
