# Atom Index Mapping — Enzyme-Bound Puckering (PDB 2D24)

## System Build

- **Input**: PDB 2D24 chain A (protein, 427 residues) + chain C substrate
- **-1 subsite**: XYS C residue 4 → renamed to BXYL
- **Tool**: `gmx pdb2gmx -f complex_AC.pdb -ff charmm36-jul2022 -water tip3p`
- **Solvation**: 28,758 TIP3P waters + 88 Na⁺ + 86 Cl⁻ = 92,745 total atoms

## GROMACS Topology Numbering

After pdb2gmx, atoms are numbered sequentially:
- Protein chain A: atoms 1–6277 (427 residues)
- BXYL (xylose): atoms 6278–6297 (1 residue, 20 atoms)
- Water + ions: atoms 6298–92745

## PLUMED Puckering Indices

```
puck: PUCKERING ATOMS=6278,6286,6290,6294,6282,6285
```

| PLUMED index | Atom name | Residue | Verified in npt.gro | Bond partner | Distance (Å) |
|-------------|-----------|---------|--------------------:|-------------|---------------|
| 6278 | C1 | BXYL | line 6280 | C2 (6286) | 1.565 |
| 6286 | C2 | BXYL | line 6288 | C3 (6290) | 1.533 |
| 6290 | C3 | BXYL | line 6292 | C4 (6294) | 1.529 |
| 6294 | C4 | BXYL | line 6296 | C5 (6282) | 1.597 |
| 6282 | C5 | BXYL | line 6284 | O5 (6285) | 1.434 |
| 6285 | O5 | BXYL | line 6287 | C1 (6278) | 1.459 |

All ring bonds in expected 1.4–1.6 Å range for pyranose ring.

## Why These Differ from Raw PDB Numbering

In the original 2D24.pdb, XYS C residue 4 has HETATM serial numbers 6497–6505.
After GROMACS processing (pdb2gmx), atoms are renumbered sequentially from 1.
The protein chain A has 6277 atoms, so the xylose starts at position 6278.

**PDB serial 6599–6607** (referenced by Alistaire) correspond to **XYS F residue 4**
(a symmetry-related copy in the other asymmetric unit of the crystal). Our simulation
uses chain A + chain C (one biological unit).

## Verification Commands

```bash
# Confirm atom identities in the solvated system
awk 'NR==6280 || NR==6288 || NR==6292 || NR==6296 || NR==6284 || NR==6287' npt.gro
# Should show: C1, C2, C3, C4, C5, O5 of BXYL residue

# Or from the full complex GRO (2 header lines: NR = atom_index + 2)
awk 'NR==6280' complex.gro   # → BXYL C1
```
