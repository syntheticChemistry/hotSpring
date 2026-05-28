# CompChem GuideStone v1.5.0

**Self-verifying baseline for enhanced sampling molecular dynamics.**

This pseudoSpore artifact contains validated free energy landscapes from 6 independent molecular simulations, spanning three enhanced sampling methods (well-tempered metadynamics, OPES, OPES_EXPLORE) and two scientific domains (protein folding, carbohydrate conformational landscapes).

## Quick Start

```bash
# Verify data integrity and scientific claims
./validate

# Check source data freshness
./refresh
```

## Modules

| # | System | Method | Time | Key Result |
|---|--------|--------|------|------------|
| 01 | Alanine dipeptide (vacuum) | WTMetaD φ/ψ | 10 ns | C7eq/C7ax basins, 20-45 kJ/mol barriers |
| 02 | Chignolin CLN025 (water) | OPES+OPES_E HLDA | 61.8 ns | 10 transitions, ΔG_fold=+19.5 kJ/mol (340K > Tm) |
| 03 | Free xylose 1D (water) | WTMetaD θ | 10 ns | 4C1/boat/1C4 basins, 38-53 kJ/mol |
| 04 | Free xylose 2D (water) | WTMetaD qx,qy | 20 ns | Full Stoddart diagram |
| 05 | Enzyme-bound GH10 1D | WTMetaD θ | 10 ns | Barrier lowering by enzyme |
| 06 | Enzyme-bound GH10 2D | WTMetaD qx,qy | 20 ns | Conformational selection |

## Scientific Significance

The CAZyme modules reproduce the key finding of Iglesias-Fernández et al. (2015): the GH10 xylanase active site lowers conformational barriers for the -1 subsite xylose ring, enabling the 4C1 → 2,5B → 2SO itinerary required for glycoside hydrolysis.

## Software Stack

- GROMACS 2026.0-conda_forge (Verlet scheme, PME, GPU-accelerated)
- PLUMED 2.10 (all modules enabled)
- RTX 3090 GPU + AMD EPYC 7452

## Validation Tools

- `nest-validate` v0.2.0 — Rust-native PLUMED-NEST analysis
- `cazyme-fel` v0.1.0 — Rust-native FES reconstruction and parity

## File Inventory

| File | Purpose |
|------|---------|
| `scope.toml` | Module manifest and metadata |
| `environment.toml` | Hardware/software snapshot |
| `domain_profile.toml` | Validation logic and audit checks |
| `tolerances.toml` | Quantitative acceptance criteria |
| `data.toml` | BLAKE3 hashes for integrity |
| `index_map.toml` | PDB ↔ GROMACS atom index translation |
| `validate` | Self-verification entry point |
| `refresh` | Data freshness check |
| `modules/` | Per-module data (HILLS, COLVAR, FES) |
| `configs/` | GROMACS MDP + PLUMED input files |
| `provenance/` | Emit provenance and braid chain |

## License

AGPL-3.0-or-later
