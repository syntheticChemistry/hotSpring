# System Setup — Free β-D-Xylopyranose Puckering FEL (v0.7.0)

## Structure Source

**Crystal-derived**: Extracted from PDB 2D24, chain C, residue 4 (XYS → BXYL).
Stereochemistry verified by X-ray crystallography (1.90 Å resolution).

**Previous version (v0.6.0) used RDKit-generated structure which was identified by
Alistaire as β-D-Lyxose (wrong stereoisomer). This has been corrected.**

## GROMACS Pipeline

```bash
# 1. pdb2gmx: assigns CHARMM36 BXYL topology, adds hydrogens
gmx pdb2gmx -f xylose_crystal.pdb -o xylose.gro -p xylose.top \
    -water tip3p -ff charmm36-jul2022 -missing

# 2. Box + solvate
gmx editconf -f xylose.gro -o xylose_box.gro -c -d 1.2 -bt cubic
gmx solvate -cp xylose_box.gro -cs spc216.gro -o xylose_solv.gro -p xylose.top

# 3. Energy minimization → NVT → NPT → Production WTMetaD
gmx grompp -f em.mdp -c xylose_solv.gro -p xylose.top -o em.tpr && gmx mdrun -deffnm em
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p xylose.top -o nvt.tpr && gmx mdrun -deffnm nvt
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p xylose.top -t nvt.cpt -o npt.tpr && gmx mdrun -deffnm npt
gmx grompp -f md_meta.mdp -c npt.gro -t npt.cpt -p xylose.top -o md_meta.tpr
PLUMED_KERNEL=.../libplumedKernel.so gmx mdrun -deffnm md_meta -plumed plumed.dat
```

## System Composition

| Component | Count | Notes |
|-----------|-------|-------|
| BXYL (β-D-xylose) | 1 molecule, 20 atoms | CHARMM36 carb.rtp BXYL residue |
| TIP3P water | 879 molecules | Cubic box, d=1.2 nm buffer |
| Total atoms | 2657 | |

## PLUMED Ring Atom Indices

```
puck: PUCKERING ATOMS=1,9,13,17,5,8
```

| Index | Name | Role |
|-------|------|------|
| 1 | C1 | Ring carbon 1 |
| 9 | C2 | Ring carbon 2 |
| 13 | C3 | Ring carbon 3 |
| 17 | C4 | Ring carbon 4 |
| 5 | C5 | Ring carbon 5 |
| 8 | O5 | Ring oxygen |

Ordering: C1-C2-C3-C4-C5-O5 (sequential around ring).
Per PLUMED PUCKERING docs: θ=0 → 4C1 chair, θ=π → 1C4 chair.

## Solvation Evidence

- Topology (`xylose.top`) contains `SOL 879` in `[ molecules ]` section
- NPT MDP uses `pcoupl = c-rescale` (pressure coupling requires solvent)
- Production MDP uses `pbc = xyz`, `coulombtype = PME` (periodic boundary + electrostatics)
- Total atom count: 2657 = 20 (xylose) + 879×3 (water) = 2657 ✓
