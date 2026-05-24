# GROMACS FEL Control — CAZyme Conformational Energy Landscapes

**Experiment:** 220
**Env:** `conda activate gromacs-fel`
**GROMACS:** 2026.0 (CUDA, PLUMED, Colvars)
**Hardware:** strandGate RTX 3090

---

## Purpose

Industry-standard control runs for validating hotSpring/barraCuda biomolecular
MD against established tools. Results here are the parity targets.

## Directory Structure

```
control/gromacs_fel/
├── README.md              ← this file
├── tutorial/              ← Wei-Tse Hsu enhanced sampling tutorial
│   └── alanine_dipeptide/ ← metadynamics on alanine dipeptide (learning)
├── cazyme/                ← CAZyme FEL production runs
│   ├── systems/           ← prepared systems (PDB, topology, FF params)
│   ├── metadynamics/      ← metadynamics input files and results
│   └── docking/           ← AutoDock Vina comparison docking results
└── results/               ← extracted FEL data for parity comparison
```

## Quick Start

```bash
conda activate gromacs-fel
gmx --version          # verify 2026.0
gmx mdrun -version     # verify GPU support
```

## Workflow

1. **Tutorial**: Follow Wei-Tse Hsu's enhanced sampling tutorial to learn
   GROMACS metadynamics workflow on alanine dipeptide.
2. **System prep**: Build GH10 CAZyme + substrate system. Use GROMOS 45a4
   or GLYCAM06 force field.
3. **Metadynamics**: Define Cremer-Pople θ,φ as CVs. Run well-tempered
   metadynamics. Generate FEL.
4. **Extract**: Export FEL data as CSV/numpy for parity comparison with
   hotSpring Phase 4.

## Collaborator Notes

- **Alistaire**: Domain expert. QM/MM and metadynamics guidance. Most recent
  assessment: 2009–2011 era work uses GROMOS 45a4, modern work uses GLYCAM06.
  QM/MM typically PBE DFT for QM + FF for MM with TIP3P water.
- **Mark**: HPC compute via NSF ACES (Texas A&M, A100 GPUs).
- **Visualization**: → ludoSpring/petalTongue (FEL surface rendering, CV plots).
