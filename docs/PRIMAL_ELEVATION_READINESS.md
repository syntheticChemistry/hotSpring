# Primal Elevation Readiness — GAP-HS-111

**Target**: Bonded force field shaders (GAP-HS-111)
**Status**: Tier 0 locked; Tier 1/2 parity confirmed; ready for primal draft.
**Date**: 2026-05-25

## Tier 0 Locked Checklist

| System | CV Dimension | Duration | Status | Reference File |
|--------|-------------|----------|--------|----------------|
| Free xylose | 1D θ | 10 ns | PASS | `fes_theta.dat` |
| Free xylose | 2D (qx, qy) | 20 ns | PASS | `cazyme_gh10_v2/fes_2d.dat` |
| Enzyme-bound (2D24) | 1D θ | 10 ns | PASS | `fes_theta.dat` |
| Enzyme-bound (2D24) | 2D (qx, qy) | 20 ns | PASS | `cazyme_2d24/fes_2d.dat` |
| Alanine dipeptide | 2D (φ, ψ) | 10 ns | PASS (reference only) | `fes_2d.dat` |

### Tier 0 Key Results
- **Free xylose 1D**: 4C1 → boat → 1C4 barriers ≈ 46/38 kJ/mol; ground state = 1C4
- **Enzyme-bound 1D**: Same topology but suppressed 4C1 basin (enzyme constraint)
- **2D (qx, qy)**: Full Stoddart landscape mapped; enzyme confines substrate to 1C4/2SO region

## Tier 1 (Python) Parity

| System | CV | Max Deviation | Status |
|--------|-----|--------------|--------|
| Free xylose | 1D θ | 0.7304 kJ/mol | MATCH |
| Enzyme-bound | 1D θ | 0.7585 kJ/mol | MATCH |
| Free xylose | 2D (qx, qy) | 1.7148 kJ/mol | MATCH |
| Enzyme-bound | 2D (qx, qy) | 1.7227 kJ/mol | MATCH |

Tolerance: 1.0 kJ/mol for 1D, 2.0 kJ/mol for 2D.

## Tier 2 (Rust) Parity

| System | CV | Max Deviation | Status |
|--------|-----|--------------|--------|
| Free xylose | 1D θ | 0.7869 kJ/mol | MATCH |
| Enzyme-bound | 1D θ | 0.7681 kJ/mol | MATCH |
| Free xylose | 2D (qx, qy) | 1.7148 kJ/mol | MATCH |
| Enzyme-bound | 2D (qx, qy) | 1.7227 kJ/mol | MATCH |

## Expected Values for Tier 3 (barraCuda / NUCLEUS IPC)

These are the reference values that the primal's Tier 3 implementation must reproduce:

```json
{
  "free_xylose_1d": {
    "ground_state": "1C4",
    "ground_state_theta_deg": 171.6,
    "4C1_barrier_kJmol": [44, 48],
    "boat_barrier_kJmol": [35, 40],
    "tolerance_kJmol": 2.0
  },
  "enzyme_bound_1d": {
    "ground_state": "1C4",
    "ground_state_theta_deg": 171.6,
    "4C1_relative_energy_kJmol": [10, 17],
    "tolerance_kJmol": 2.0
  }
}
```

## What barraCuda Needs to Implement for Tier 3

1. **HILLS parser** — Read PLUMED HILLS format (1D and 2D)
2. **Gaussian kernel summation** — Vectorized over grid, matching `staging/cazyme-fel` algorithm
3. **Min-to-zero normalization** — Shift F(s) so global minimum = 0
4. **Parity check** — Compare against reference fes.dat with interpolation on mismatched grids
5. **IPC protocol** — Report results via NUCLEUS message bus with `ParityCheck` schema

### Acceptance Criteria
- Max deviation from Tier 0 reference: < 2.0 kJ/mol (1D), < 3.0 kJ/mol (2D)
- Basin topology must match: same number of basins, same labels
- Barrier heights within 5 kJ/mol of reference

## Primal Drafting Notes

- GAP-HS-111 scope: bonded force field shader validation
- The FEL reconstruction is the primary acceptance test
- Input: HILLS file path, reference fes.dat path, tolerance
- Output: `ValidationResult` with basins, barriers, parity status
- The Rust implementation at `staging/cazyme-fel/src/lib.rs` is the canonical Tier 2 reference
