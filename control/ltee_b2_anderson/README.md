# LTEE B2 — Anderson/Wiser Fitness Model

**Paper**: Wiser, Ribeck & Lenski (2013) — Long-Term Dynamics of Adaptation in Asexual Populations  
**Spring**: hotSpring  
**lithoSpore module**: 7 (`ltee-anderson`)  
**Status**: Tier 1 + Tier 2 COMPLETE  

## Artifacts

| File | Purpose |
|------|---------|
| `expected_values.json` | Frozen expected values for lithoSpore consumption |
| `../../barracuda/src/validation/scenarios/s_ltee_anderson.rs` | Tier 2 Rust validation scenario |
| `../../notebooks/papers/13-ltee-anderson-fitness.ipynb` | Paper reproduction notebook |

## Running

```bash
# Via UniBin (recommended)
hotspring_unibin validate --scenario ltee-anderson

# With structured JSON output (Tier 2 / toadstool.validate ready)
hotspring_unibin validate --scenario ltee-anderson --format json
```

## Expected Values

`expected_values.json` contains the frozen baseline from the Wiser et al. 2013
power-law fitness model: growth rate parameters, Anderson Hamiltonian
eigenvalues, GOE/Poisson level spacing ratios, and 12-population variance
diagnostics.

## lithoSpore Integration

The lithoSpore team consumes:
1. `expected_values.json` — parameter baselines and tolerance bounds
2. `s_ltee_anderson.rs` scenario — validation logic for Tier 2 parity checks
3. `--format json` output — structured results for `toadstool.validate` ingestion
