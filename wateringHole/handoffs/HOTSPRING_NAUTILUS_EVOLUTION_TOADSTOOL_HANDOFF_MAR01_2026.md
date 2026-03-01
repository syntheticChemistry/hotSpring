# hotSpring → toadStool: Nautilus Shell Evolution + Adaptive Steering Handoff

**Date:** 2026-03-01
**From:** hotSpring v0.6.15 (Exp 030 active)
**To:** toadStool/barracuda team
**Supersedes:** `HOTSPRING_NAUTILUS_BRAIN_TOADSTOOL_HANDOFF_MAR01_2026.md` (earlier today)

## What Changed Since Last Handoff

### 1. Nautilus Shell — Self-Regulating Evolution (NEW)

The Nautilus Shell (`primalTools/bingoCube/nautilus/`) now has a fully wired
self-regulation loop. Previously, the drift monitor and edge seeder existed
but were not connected to the evolution lifecycle. Now:

**Drift Monitor → Shell Evolution (automatic):**
- Every `evolve_generation()` records N_e * s and checks for drift
- `DriftAction::IncreaseSelection` — halves elite survivors or grows tournament size
- `DriftAction::IncreasePop { factor }` — grows population with fresh random boards
- Each `GenerationRecord` tracks `ne_s` and `drift_action` for audit trail

**Edge Seeding → Directed Mutagenesis (automatic):**
- `shell.set_concept_edges(edges)` registers prediction failure regions
- During evolution, bottom 25% of boards are replaced with edge-biased boards
- `shell.detect_concept_edges(inputs, targets, threshold)` does LOO CV to find edges

**DriftAction serialization:** Now derives `Serialize`/`Deserialize` — full
shell state including drift actions persists through save/restore cycles.

**Test count:** 26 → 31 tests. **Example count:** 3 → 5 examples.

### 2. AKD1000 Int4 Weight Export (NEW)

`shell.export_akd1000_weights()` returns `Akd1000Export`:
- Quantized weights in [-8, 7] (symmetric min-max: `w_q = round(w * 7/max|w|)`)
- Per-target dequantization scales and f64 biases
- `predict_dequantized()` for software validation against hardware
- `quantization_mse()` measures precision loss

Validated: quantization MSE = 0.004 on QCD-like data. Plaquette within ±0.02,
CG within ±0.04 of full-precision readout.

### 3. Full Brain Rehearsal (VALIDATED)

New example `full_brain_rehearsal` exercises the complete pipeline:
1. Bootstrap + drift-monitored evolution (15 gens, N_e*s stable 2.4–22.4)
2. Concept edge detection (4 edges at phase boundary)
3. Edge-seeded re-evolution (stable, no population destabilization)
4. AKD1000 int4 export + quantization validation
5. Save/restore: **bit-perfect** prediction match (delta < 10⁻¹⁶)
6. Instance transfer + merge (2 instances, 30 combined history entries)
7. Reset for production (cleared edges, saved production state)

### 4. Adaptive Steering Fix (Exp 030)

**Bug:** `bi + 1 < beta_order.len()` prevented NPU from inserting adaptive
beta points after the last seed. Exp 029 completed with 4 seeds, zero insertions.

**Fix:**
- Guard changed to `results.len() >= 3` — fires on every point after 3 results
- `remaining = vec![]` when queue empty (NPU handles gracefully)
- New `--max-adaptive=N` CLI flag (default 12) caps total insertions
- Main loop `while bi < beta_order.len()` naturally extends when NPU pushes

**Exp 030 running:** 4 seeds + up to 12 adaptive, bootstrapped from 29 data
points (Exp 024+028+029). NPU reprioritized scan: β=5.25 first (transition).

## Action Items for toadStool

### Priority 1: Absorb Nautilus Shell Self-Regulation

```
primalTools/bingoCube/nautilus/src/shell.rs    — evolve_generation() with drift + edge seeding
primalTools/bingoCube/nautilus/src/constraints.rs — DriftMonitor, DriftAction (now serializable)
```

Wire `DriftAction` into `MultiDevicePool` scheduler:
- When NPU reports drift, scheduler can adjust population sizes or selection pressure
- When concept edges are detected, scheduler can prioritize measurements there

### Priority 2: Absorb AKD1000 Export Path

```
primalTools/bingoCube/nautilus/src/shell.rs — Akd1000Export, export_akd1000_weights()
```

Map `Akd1000Export` → `akida-models` FullyConnected layer:
- `quantized_weights[t][i]` → int4 weight matrix
- `scales[t]` → dequantization scale per output
- `biases[t]` → bias in full precision

### Priority 3: Absorb Adaptive Steering Pattern

```
barracuda/src/bin/production_dynamical_mixed.rs — SteerAdaptive with --max-adaptive
```

The pattern: NPU evaluates candidate betas across the measured range, scores
by priority + uncertainty, inserts the best into the scan queue. Budget cap
prevents runaway expansion. This generalizes to any parameter scan.

### Priority 4: Wire `bingocube-nautilus` as Workspace Dependency

```toml
[workspace.dependencies]
bingocube-nautilus = { path = "../primalTools/bingoCube/nautilus" }
```

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `nautilus/src/shell.rs` | ~850 | Shell + drift + edges + AKD1000 export |
| `nautilus/src/constraints.rs` | ~300 | DriftMonitor, DriftAction, EdgeSeeder |
| `nautilus/src/brain.rs` | ~250 | NautilusBrain QCD integration |
| `nautilus/src/evolution.rs` | ~420 | Selection, crossover, mutation |
| `nautilus/src/readout.rs` | ~300 | Ridge regression readout (Cholesky) |
| `nautilus/src/response.rs` | ~200 | Board → ResponseVector projection |
| `nautilus/src/population.rs` | ~200 | Population + Pearson fitness |
| `nautilus/examples/full_brain_rehearsal.rs` | ~250 | End-to-end validation |
| `nautilus/examples/predict_live_exp029.rs` | ~300 | Blind prediction on live data |
| `barracuda/src/bin/production_dynamical_mixed.rs` | ~2400 | Adaptive steering fix |

## Validation

- 31 unit tests pass (nautilus crate)
- 5 examples run clean
- Full brain rehearsal: save/restore/transfer/merge/AKD1000 all validated
- Exp 030 running with fixed steering (NPU reprioritized scan order from bootstrap data)
