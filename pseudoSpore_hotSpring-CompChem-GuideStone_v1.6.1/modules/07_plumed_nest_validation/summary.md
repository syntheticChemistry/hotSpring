# PLUMED-NEST Validation Summary

## Overview

The hotSpring CompChem GuideStone pipeline ingested **8 PLUMED-NEST targets** spanning diverse enhanced sampling methods. Of these, **2 have been fully validated** (with self-consistent reproduction via GROMACS 2026.0 + PLUMED 2.10), and **6 have been profiled** (ingested, domain-profiled, pending full reproduction).

## Validated Targets (PASS)

### Target 01: Alanine Dipeptide (plumID 24.020)
- **Method**: Well-tempered metadynamics (vacuum)
- **CVs**: Backbone phi/psi dihedrals
- **Key Results**:
  - C7eq basin located at φ = -1.37 rad (ref: -1.40) ✓
  - C7ax basin located at φ = +1.08 rad (ref: +1.00) ✓
  - Barrier: 35.2 kJ/mol (ref range: 20-45 kJ/mol) ✓
  - Block averaging converged (max stderr 3.72 kJ/mol) ✓
  - Barrier convergence < 3 kJ/mol over last 3 windows ✓
- **Force Field**: AMBER99SB-ILDN
- **Simulation**: 10 ns, 10,000 Gaussians, 150×150 FES grid

### Target 02: Chignolin OPES (plumID 24.029)
- **Method**: OPES_METAD + OPES_METAD_EXPLORE
- **CVs**: HLDA (Harmonic Linear Discriminant Analysis)
- **Key Results**:
  - 10 folding/unfolding transitions (5 fold, 5 unfold) ✓
  - ΔG_fold = +19.5 kJ/mol (above Tm at 340K) ✓
  - FE convergence < 4 kJ/mol over last 3 windows ✓
- **Force Field**: a99SB-disp (explicit solvent, 14637 atoms)
- **Simulation**: 61.8 ns (309,569 frames)

## Profiled Targets (Ingested, Pending Reproduction)

| # | Target | Method | Domain Relevance |
|---|--------|--------|-----------------|
| 03 | BRD4 OneOPES | OPES protein-ligand | Drug binding methodology |
| 04 | Muscarinic Funnel | Funnel metadynamics | GPCR membrane systems |
| 05 | Glycan Pucker (22.028) | REST2-RECT | **Direct CAZyme domain** — N-glycan puckering |
| 06 | CAZyme Glycan (25.007) | REST2-RECT + steered | **Direct validation standard** — GH38 + Mannosidase II |
| 07 | Amylase QM/MM | QM/MM metadynamics | Enzyme catalysis (future QM/MM kernel) |
| 08 | Urea Nucleation | Crystallization metadynamics | Nucleation CVs methodology |

## Domain Profiles Applied

- `carbohydrate_pucker.toml` — Residue filters, puckering zone definitions, Cremer-Pople tolerance thresholds
- `metadynamics_canonical.toml` — Gaussian height/width/pace reference parameters, convergence criteria

## Validation Infrastructure

- **Tool**: `nest-validate` v0.2.0 (Rust)
- **Pipeline**: `nest-validate guidestone validate` → aggregate
- **Integrity**: BLAKE3 hashes for all FES output files
- **Elapsed**: ~31 seconds for full validation sweep (cached FES)

## Significance for Alistaire

Targets **05** (Grothaus 2022, plumID 22.028) and **06** (Grothaus 2025, plumID 25.007) are direct methodological predecessors to our CAZyme FEL work. They demonstrate:
1. REST2-RECT as a complementary enhanced sampling strategy to WTMetaD
2. Multi-glycan Cremer-Pople conformational analysis at GH-family scale
3. External validation standard from a major glycobiology computational group

Full reproduction of these targets using our pipeline is the natural next priority (see Module 08: Exploration Roadmap).
