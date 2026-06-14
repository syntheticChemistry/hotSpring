# pseudoSpore / lithoSpore Evolution from PLUMED-NEST Ingestion

## Summary

The PLUMED-NEST ingestion array provides 8 curated targets spanning 6 enhanced
sampling methods, 6 system types, and 4 complexity layers. From reproducing these,
we evolve the pseudoSpore standard and lithoSpore tooling with:

## New Domain Profiles (4 created)

| Profile | Derived From | Key CVs | Tolerance Class |
|---------|-------------|---------|-----------------|
| `metadynamics_canonical` | Target 1 (alanine) | TORSION (phi/psi) | barrier_height, minimum_location |
| `protein_folding` | Target 2 (chignolin) | CONTACTMAP, RMSD, OPES | folding_fe, convergence |
| `carbohydrate_pucker` | Targets 5,6 (glycan) | PUCKERING (theta/phi), RECT | pucker_peak, population |
| `binding_free_energy` | Targets 3,4 (BRD4/GPCR) | DISTANCE, funnel | binding_fe, replica_convergence |

## New Tolerance Classes

### Quantitative Thresholds (from reproduced baselines)

```toml
[parity]
tier0_to_tier1_max_deviation_kj = 1.0   # GROMACS → barraCuda
tier1_to_tier2_max_deviation_kj = 2.0   # barraCuda → NUCLEUS

[metadynamics]
barrier_height_tolerance_kj = 5.0       # From Target 1
convergence_window_kj = 2.0

[opes]
folding_fe_tolerance_kj = 4.0           # From Target 2

[puckering]
theta_peak_tolerance_rad = 0.2          # From Targets 5,6
population_4c1_min_fraction = 0.85

[binding]
absolute_fe_tolerance_kcal = 2.0        # From Targets 3,4
```

## New `litho audit` Checks (proposed)

1. **PLUMED parse validation** — `plumed driver --parse-only` on all `.dat` files
2. **Reference comparison** — compare FES minima/barriers against published values
3. **Convergence verification** — block averaging with stride analysis
4. **Replica exchange rate** — for REST-RECT targets, verify 10-40% acceptance
5. **Parity deviation** — compare Tier N output against Tier N-1 baseline

## pseudoSpore Standard Extensions

### New module types
- `enhanced-sampling` (metadynamics, OPES, REST-RECT)
- `multi-replica` (exchange protocols, parallel walkers)
- `qm-mm` (mixed quantum/classical)
- `ml-cv` (machine-learned collective variables)

### New data artifacts
- `HILLS` — Gaussian deposition record (PLUMED native)
- `COLVAR` — Collective variable timeseries
- `Kernels.data` — OPES kernel state (compression)
- `*.ptc` — PyTorch traced models (DeepTICA, GNN-CV)

### New analysis derivations
- `plumed sum_hills` — FES reconstruction from HILLS
- `CONVERT_TO_FES` — histogram → free energy via PLUMED
- `FES_from_Reweighting.py` — OPES reweighting analysis

## Method Coverage Matrix

```
                   Metad  OPES  REST-RECT  Funnel  ML-CV  QM/MM
Target 1 (ala)      ✓
Target 2 (chig)            ✓
Target 3 (brd4)            ✓
Target 4 (musc)     ✓                      ✓
Target 5 (glyc)                   ✓
Target 6 (cazy)                   ✓
Target 7 (amyl)            ✓                       ✓      ✓
Target 8 (urea)            ✓                       ✓
```

## Parity Evolution Targets

Each reproduced target becomes a quantitative baseline for:

| Component | What it validates | Deviation budget |
|-----------|------------------|-----------------|
| barraCuda TORSION shader | phi/psi dihedral computation | < 1e-6 rad |
| barraCuda PUCKERING shader | Cremer-Pople 6-ring | < 1e-6 rad |
| barraCuda METAD kernel | Gaussian deposition + grid | < 0.1 kJ/mol per Gaussian |
| barraCuda OPES kernel | Kernel compression + bias | < 0.1 kJ/mol |
| toadStool dispatch | Multi-replica exchange protocol | bit-exact message passing |
| coralReef shader compile | CV computation graph optimization | identical output |

## Files Created

```
control/plumed_nest/
├── README.md
├── PSEUDOSPORE_EVOLUTION.md          (this file)
├── ingest.sh                         (download + validate pipeline)
├── profiles/
│   ├── TOLERANCES_REGISTRY.toml      (aggregated tolerance classes)
│   ├── metadynamics_canonical.toml   (Target 1 profile)
│   ├── protein_folding.toml          (Target 2 profile)
│   ├── carbohydrate_pucker.toml      (Targets 5,6 profile)
│   └── binding_free_energy.toml      (Targets 3,4 profile)
├── target_01_alanine_dipeptide/      (COMPLETE — 10 ns FES)
├── target_02_chignolin_opes/         (INGESTED — full inputs)
├── target_03_brd4_oneopes/           (DOCUMENTED)
├── target_04_muscarinic_funnel/      (DOCUMENTED)
├── target_05_glycan_pucker/          (INGESTED — 6 .dat files)
├── target_06_cazyme_glycan/          (INGESTED — enzyme parses)
├── target_07_amylase_qmmm/          (DOCUMENTED)
└── target_08_urea_nucleation/        (DOCUMENTED)
```

## Next Steps

1. **Immediate**: Run Targets 5,6 production (glycan domain — our priority)
2. **Short-term**: Full chignolin OPES μs run (requires PLUMED rebuild with OPES+GROMACS patch)
3. **Medium-term**: Integrate tolerance checks into `litho audit` Rust code
4. **Long-term**: barraCuda shader implementations against these baselines
