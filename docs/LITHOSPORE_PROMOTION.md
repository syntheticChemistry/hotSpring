# pseudoSpore → lithoSpore Chassis Promotion Plan

**Date**: May 24, 2026
**Artifact**: hotSpring-CAZyme-FEL v0.6.0
**Target**: Unified chassis — same binary validates LTEE data *and* CAZyme FEL data
**Prerequisite**: Module 3 completion + Alistaire convention confirmation

---

## Current State

The pseudoSpore tarball follows the lithoSpore *pattern* but uses slightly
different wire formats. The promotion path aligns them so the same `litho`
binary can validate both LTEE and CAZyme modules.

| Layer | pseudoSpore (now) | lithoSpore chassis (target) |
|-------|------------------|-----------------------------|
| Scope | `[artifact]` + `[[module]]` (custom fields) | `[guidestone]` + `[[module]]` with `binary`, `data_dir`, `expected`, `tier1_notebook` |
| Braid | `ferment_transcript.json` (wrapper object) | `FermentBraid` struct (flat, serde-compatible) |
| Validation | `validation.json` (custom schema) | `ModuleResult` / `ValidationReport` (litho-core types) |
| Module identity | `name = "ala-dipeptide-fel"` | `binary = "cazyme-fel"` (single crate, multi-module) |
| Parity | Not yet | `ParityReport` (Tier 0 GROMACS vs Tier 1 Python vs Tier 2 Rust) |

---

## Promotion Steps

### Step 1: Align ferment_transcript.json to FermentBraid wire format

The lithoSpore `FermentBraid` struct expects these top-level fields:

```json
{
  "dataset_id": "cazyme_fel_iglesias_2015",
  "spring": "hotSpring",
  "spring_version": "0.6.32",
  "braid_id": "urn:braid:hotspring-cazyme-fel-v0.6.0",
  "dag_session_id": "urn:dag:hotspring-cazyme-fel-2026-05-24",
  "dag_merkle_root": "cbf908fb...",
  "spine_id": "urn:spine:hotspring-cazyme-fel-v0.6.0",
  "timestamp": "2026-05-24T16:33:51Z",
  "computation": {
    "tool": "GROMACS 2026.0 + PLUMED 2.9.2",
    "tool_version": "2026.0",
    "substrate": "RTX 3090 (GPU+CPU)",
    "pipeline": "metadynamics-cremer-pople",
    "input_accession": "PDB:2D24",
    "input_blake3": "<blake3 of 2D24.pdb>",
    "output_blake3": "<blake3 of fes_theta.dat>",
    "wall_time_seconds": 6600
  }
}
```

**Delta from current**: Move `summary_stats` into `computation` block. Add
`dataset_id` and `spring` at top level. Compute BLAKE3 hashes (replacing
SHA-256). Drop the `ferment_transcript` wrapper key.

**When**: Before publishing to lithoSpore `provenance/braids/` directory.

### Step 2: Register hotSpring CAZyme contribution in lithoSpore scope.toml

Add to lithoSpore's `artifact/scope.toml`:

```toml
[[spring]]
name = "hotSpring"
contributes = [
  "CAZyme conformational free energy landscapes",
  "Cremer-Pople puckering metadynamics",
  "Enzyme-substrate conformational selection",
  "GROMACS/PLUMED Tier 0 industry control",
]
modules = ["ltee-anderson", "cazyme-fel"]
papers = ["B2", "B9", "IF2015"]

[[module]]
name = "cazyme_conformational_fel"
binary = "cazyme-fel"
data_dir = "artifact/data/iglesias_2015"
expected = "validation/expected/module8_cazyme.json"
tier1_notebook = "notebooks/module8_cazyme/puckering_fel.py"
```

**Delta**: lithoSpore gets a new `[[spring]]` update for hotSpring (adds
CAZyme alongside Anderson), a new `[[module]]` entry, and a new paper
reference for Iglesias-Fernández 2015.

### Step 3: Write validation expected values (module8_cazyme.json)

Once Module 3 completes and Alistaire confirms the convention:

```json
{
  "module": "cazyme_conformational_fel",
  "paper": "Iglesias-Fernandez 2015",
  "doi": "10.1039/C4SC02240H",
  "checks": [
    {
      "name": "free_xylose_two_chairs",
      "type": "topology",
      "expected": 2,
      "tolerance": "exact"
    },
    {
      "name": "free_xylose_boat_basin",
      "type": "topology",
      "expected": true
    },
    {
      "name": "chair_barrier_range_kJmol",
      "type": "range",
      "expected_range": [25, 60],
      "tolerance_name": "metadynamics_barrier"
    },
    {
      "name": "enzyme_conformational_shift",
      "type": "comparison",
      "description": "Enzyme-bound FEL differs from free FEL",
      "expected": true
    },
    {
      "name": "ground_state_is_4C1",
      "type": "identity",
      "expected": true,
      "note": "Depends on atom ordering convention confirmation"
    }
  ]
}
```

### Step 4: Write Tier 1 Python notebook (puckering_fel.py)

Standalone Python implementation that:
1. Reads HILLS file
2. Reconstructs F(θ) via Gaussian kernel summation (rebiasing)
3. Identifies basins and barriers
4. Produces same check outputs as GROMACS `sum_hills`

This proves the FEL reconstruction algorithm is correct independent of
PLUMED. Required primitives: NumPy, SciPy (no GROMACS dependency).

### Step 5: Wire Tier 2 Rust crate (cazyme-fel)

New crate at `lithoSpore/crates/cazyme-fel/` that:
1. Reads HILLS format (already defined)
2. Reconstructs F(θ) using native Rust kernel sum
3. Runs same checks as Python
4. Produces `ModuleResult` compatible with `litho validate --json`

This is the first non-LTEE module in lithoSpore — proving the chassis is
truly domain-agnostic. The same `litho validate` binary runs both LTEE
and CAZyme validation.

### Step 6: Wire Tier 0 → Tier 2 Parity

```toml
# In scope.toml or a parity config
[[parity_pair]]
module = "cazyme_conformational_fel"
tier0 = "GROMACS 2026.0 + PLUMED 2.9.2 (sum_hills)"
tier1 = "notebooks/module8_cazyme/puckering_fel.py"
tier2 = "crates/cazyme-fel"
acceptance = "FEL topology identical (same basins, barriers within 1 kJ/mol)"
```

`litho parity` runs all three and reports MATCH/DIVERGENCE. This is the
first module with a **Tier 0** (industry control) in addition to Tier 1/2.

### Step 7: Live provenance trio (Tier 3)

When NUCLEUS primals are running:
- `rhizoCrypt` records the computation DAG in real-time (replacing pseudo dag.json)
- `loamSpine` anchors ledger entries with DID signing (replacing pseudo spine.json)
- `sweetGrass` weaves the attribution braid (already partially live via `live_braid.json`)

The pseudoSpore's existing provenance structure is already shaped for this
— promotion is switching from offline hash → live IPC call.

---

## Unified Chassis Architecture

After promotion, lithoSpore's module registry looks like:

```
lithoSpore/crates/
├── litho-core/           # Chassis (unchanged, domain-agnostic)
├── ltee-fitness/         # Module 1 (LTEE)
├── ltee-mutations/       # Module 2 (LTEE)
├── ltee-alleles/         # Module 3 (LTEE)
├── ltee-citrate/         # Module 4 (LTEE)
├── ltee-biobricks/       # Module 5 (LTEE)
├── ltee-breseq/          # Module 6 (LTEE)
├── ltee-anderson/        # Module 7 (LTEE + hotSpring)
├── cazyme-fel/           # Module 8 (CAZyme + hotSpring)  ← NEW
└── ltee-cli/             # Unified CLI (dispatches all modules)
```

The `ltee-cli` rename to `litho-cli` is overdue — or the CLI stays
as-is and dispatches the new module by name from `scope.toml`.

---

## What's Needed Before Each Step

| Step | Blocker | Owner |
|------|---------|-------|
| 1 | BLAKE3 hash of input PDB + output FES | Tamison (trivial) |
| 2 | Module 3 completion + convention confirmed | Alistaire + running sim |
| 3 | Convention confirmation | Alistaire |
| 4 | None (can start now, validate against existing fes_theta.dat) | Tamison |
| 5 | Step 4 done (Python reference to match against) | Tamison |
| 6 | Steps 4+5 done | Tamison |
| 7 | NUCLEUS primals online (ongoing ecosystem work) | Ecosystem |

---

## Relationship to PRIMAL_GAPS.md

This promotion plan is the evolution path for:
- **GAP-HS-111** (Biomolecular Force Field Evolution) — Steps 4-6 require
  the bonded force field shaders documented there
- **GAP-HS-112** (petalTongue FEL Visualization) — the lithoSpore `litho
  visualize` command will need a FEL DataBinding adapter once the module
  exists

---

## What Makes This Different from LTEE Modules

| Dimension | LTEE modules | CAZyme FEL module |
|-----------|--------------|-------------------|
| Data source | Published datasets (CSV/TSV) | Computed (GROMACS simulation output) |
| Tier 0 | None (data is the ground truth) | GROMACS + PLUMED (industry control) |
| Reproducibility | Deterministic (same data → same answer) | Stochastic (MD is chaotic; topology should match, exact values won't) |
| Tolerances | Tight (curve fit parameters, mutation counts) | Topological (number of basins, barrier ranges, ground state identity) |
| Hardware | CPU-only (statistics) | GPU-accelerated (MD simulation) |
| Time to validate | Seconds (data analysis) | Hours (simulation) — or seconds if validating pre-computed FES |

The chassis handles this gracefully: the module's `lib::run_validation()`
can either:
- **Fast mode**: read pre-computed FES files and validate topology (seconds)
- **Full mode**: run metadynamics from configs and validate end-to-end (hours)

The `scope.toml` can declare both modes via a `validation_mode` field.

---

## Timeline

```
DONE (v0.6.0)    : pseudoSpore handoff → Alistaire
DONE             : Step 1 — braid format alignment (hotspring_cazyme_fel.json)
DONE             : Step 4 — Tier 1 Python (notebooks/cazyme_fel/puckering_fel.py)
DONE             : Step 5 — Tier 2 Rust (staging/cazyme-fel/, 3 unit tests + parity)

Post-Alistaire   : Convention confirmed → Steps 2, 3
                   Module 3 FEL → expected values finalized

Next             : Step 6 — `litho parity` wiring (cross-tier report)
                   Clone lithoSpore, move crate, wire into CLI

Ongoing          : Step 7 (Tier 3 when NUCLEUS primals land)
```

## Parity Results (May 24, 2026)

```
Tier 0 (GROMACS+PLUMED)  : fes_theta.dat           [industry reference]
Tier 1 (Python)          : max dev 0.83 kJ/mol     MATCH vs Tier 0
Tier 2 (Rust)            : max dev 0.75 kJ/mol     MATCH vs Tier 0
Tier 1 vs Tier 2         : max dev 0.00 kJ/mol     EXACT MATCH (identical algorithm)
```

All three tiers produce the same FEL topology (3 basins: 4C1, boat, 1C4;
barriers 42–43 kJ/mol; global min at 1C4 ~172°). The sovereign implementations
(Python and Rust) agree perfectly with each other and are within <1 kJ/mol of
the industry control (GROMACS+PLUMED `sum_hills`).
