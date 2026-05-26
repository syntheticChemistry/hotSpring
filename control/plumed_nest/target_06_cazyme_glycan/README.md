# Target 06: CAZyme Glycan Landscape — REST-RECT (Enzyme-Bound)

## PLUMED-NEST Reference

- **plumID**: 25.007
- **Paper**: Grothaus et al., "Shaping the glycan landscape. Hidden relationships between linkage and ring distortion induced by carbohydrate-active enzymes" (2025, unpublished)
- **Method**: REST-RECT enhanced sampling + steered MD
- **System**: M5G0 glycan in solution and bound to mannosidase II (MII) enzyme

## Key Scientific Content

This is our EXACT domain — glycans bound to CAZymes (carbohydrate-active enzymes).
The same author (Grothaus) and methodology as Target 5, but now investigating how
enzyme binding affects glycan conformational landscapes.

Key insight: Enzymes induce pucker distortions in their substrate glycans, shifting
population from 4C1 chair toward conformations along the reaction pathway.

## Files Ingested

| File | Description | Atoms | Replicas |
|------|-------------|-------|----------|
| `plumed_M5G0.dat` | Glycan in solution (MOLINFO) | ~300 | 12 (RECT 1→14) |
| `plumed_M5G0+MII.dat` | Glycan bound to MII enzyme | ~16,322 | 16 (RECT 1→6) |
| `plumed_M5G0+MII_steer_conformer.dat` | Steered MD: conformer transition | ~16,322 | - |
| `plumed_M5G0+MII_steer_theta.dat` | Steered MD: theta puckering | ~16,322 | - |
| `plumed_M5G0_restrain.dat` | Restrained reference simulation | ~300 | - |

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Replicas (solution) | 12 (RECT: 1→14) |
| Replicas (enzyme) | 16 (RECT: 1→6, finer spacing) |
| PACE | 500 steps |
| SIGMA | 0.35 rad |
| TAU | 4.0 kJ/mol |
| Temperature | 310.15 K |
| Grid | 200 bins, -pi to pi |
| Enzyme | Mannosidase II (MII) |
| Substrate | M5G0 high-mannose glycan (7 monosaccharides) |

## CVs Monitored

- 14 glycosidic dihedral angles (phi, psi, omega for each linkage)
- 7 monosaccharide puckering coordinates (Cremer-Pople: phi, theta, amplitude)
- Steered: targeted theta transitions + conformer swaps

## Reproduction Status

- [x] Archive downloaded and extracted
- [x] PLUMED .dat files identified (5 files)
- [x] Enzyme-bound file validates (`plumed driver --parse-only` PASS)
- [ ] Solution PLUMED needs reference PDB (MOLINFO)
- [ ] GROMACS topology for MII + M5G0 system
- [ ] 16-replica REST-RECT production
- [ ] Pucker theta analysis comparing free vs. enzyme-bound
- [ ] Comparison to Grothaus et al. published distributions

## Why This Target Matters

THIS IS THE EXTERNAL VALIDATION STANDARD FOR OUR WORK:
- Same enzyme family (CAZymes)
- Same methodology (REST-RECT)
- Same CVs (Cremer-Pople puckering + glycosidic dihedrals)
- Same author (Grothaus, who pioneered this approach)
- Direct comparison of free vs. enzyme-bound glycan landscapes

## Parity Target

- barraCuda: PUCKERING + TORSION CVs in large protein-glycan system
- barraCuda: RECT bias accumulation with 16 replicas
- toadStool: Multi-walker replica exchange with enzyme topology
- NUCLEUS: Full CAZyme-glycan enhanced sampling composition
