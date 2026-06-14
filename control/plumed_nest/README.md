# PLUMED-NEST Ingestion Array

GROMACS validation suite for parity evolution via NUCLEUS primals.

## Purpose

Reproduce published PLUMED-NEST entries as Tier 0 GROMACS controls, establishing
quantitative baselines for barraCuda/NUCLEUS parity evolution.

## Targets

| # | System | Method | plumID | Status |
|---|--------|--------|--------|--------|
| 01 | Alanine dipeptide (vacuum) | Well-tempered metadynamics | 24.020 | **COMPLETE** (10 ns, FES verified) |
| 02 | Chignolin folding | OPES + OPES-Explore | 24.029 | **INGESTED** (inputs validated, OPES confirmed) |
| 03 | BRD4/HSP90 binding | OneOPES (8 replicas) | 24.017 | documented |
| 04 | Muscarinic M2 GPCR | Funnel metadynamics | 20.000 | documented |
| 05 | N-glycan pucker | REST2-RECT | 22.028 | **INGESTED** (6 .dat files, Cremer-Pople validated) |
| 06 | CAZyme glycan landscape | REST-RECT | 25.007 | **INGESTED** (enzyme-bound parses, our domain) |
| 07 | Alpha-amylase catalysis | OPES + QM/MM | 25.012 | documented |
| 08 | Urea/glycine nucleation | OPES + GNN | 22.039 | documented |

## Parity Ladder

```
Tier 0: GROMACS + PLUMED (this suite)
  ↓ parity < 1 kJ/mol
Tier 1: barraCuda shaders (CV computation + bias accumulation)
  ↓ IPC composition
Tier 2: NUCLEUS primals (toadstool dispatch + barracuda tensor)
```

## Per-Target Layout

```
target_NN_name/
├── README.md           # plumID, paper, reference values
├── archive/            # Downloaded PLUMED-NEST zip contents
├── inputs/             # GROMACS .tpr / .mdp / topology
├── plumed/             # PLUMED .dat files (tested)
├── output/             # COLVAR, HILLS, trajectory
├── analysis/           # FEL reconstruction, convergence
├── figures/            # Generated plots
└── reference/          # Published reference data for comparison
```

## Usage

```bash
# Ingest all targets (download + validate PLUMED inputs)
./ingest.sh --all

# Ingest a single target
./ingest.sh --target 01

# Run a target
cd target_01_alanine_dipeptide
gmx mdrun -plumed plumed/plumed.dat -deffnm output/md

# Reconstruct FEL
plumed sum_hills --hills output/HILLS --mintozero --outfile analysis/fes.dat
```
