# Target 05: N-Glycan Pucker — Cremer-Pople FEL (REST-RECT)

## PLUMED-NEST Reference

- **plumID**: 22.028
- **Paper**: Grothaus, Bussi, Colombi Ciacchi, JCIM 62, 4992–5008 (2022)
- **DOI**: 10.1021/acs.jcim.2c01049
- **Method**: REST2-RECT (Replica Exchange with Collective Tempering)
- **System**: N-glycans (M5, FM5, M9, A2G2, A2G2S2) in explicit solvent

## Key Scientific Content

Uses the Cremer-Pople puckering coordinates (phi, theta) to characterize
6-membered ring conformations of monosaccharides in N-glycans. The PUCKERING
CV in PLUMED computes these directly from the ring atom positions.

Biasing is done via RECT: 1D metadynamics on each dihedral angle independently,
with different bias strengths across replicas (RECT=1,1.2,...,14). Combined with
REST2 (Replica Exchange Solute Tempering) for effective temperature scaling.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Replicas | 12 (RECT scaling: 1→14) |
| PACE | 500 steps |
| SIGMA | 0.35 rad (all dihedrals) |
| TAU | 4.0 kJ/mol |
| Temperature | 310.15 K |
| Grid | 200 bins, -pi to pi |
| Glycans | M5, FM5, M9, A2G2, A2G2S2 |

## Puckering Analysis

Post-processing via `plumed_pucker.dat`:
- Reads COLVAR_theta from simulation output
- Computes 1D histograms of theta for each monosaccharide
- Generates 2D FEL (theta vs phi) using CONVERT_TO_FES

## Reference Values

- Monosaccharides in N-glycans predominantly in 4C1 chair (theta ≈ 0)
- Minor populations of 1C4 (theta ≈ pi) and boat/twist-boat (theta ≈ pi/2)
- Published theta distributions match experimental NMR J-couplings

## Files Ingested

- `plumed_M5.dat` — High-mannose M5 (7 monosaccharides, 14 dihedrals)
- `plumed_FM5.dat` — Fucosylated M5
- `plumed_M9.dat` — High-mannose M9
- `plumed_A2G2.dat` — Complex-type A2G2
- `plumed_A2G2S2.dat` — Sialylated A2G2
- `plumed_pucker.dat` — Post-processing: theta/phi histograms & FEL

## Reproduction Status

- [x] Archive downloaded and extracted
- [x] PLUMED .dat files identified (6 files)
- [ ] Reference PDBs needed (MOLINFO requirement)
- [ ] GROMACS topology generation (GLYCAM/CHARMM36)
- [ ] Solvation and equilibration
- [ ] 12-replica REST-RECT production
- [ ] Pucker theta/phi analysis
- [ ] Comparison to published distributions

## Why This Target Matters

DIRECTLY validates our Cremer-Pople CV implementation. Uses the same phi/theta
puckering coordinates as our CAZyme work. The published population distributions
provide quantitative baselines for ring conformer classification.

## Parity Target

- barraCuda: PUCKERING CV computation (6-membered ring Cremer-Pople)
- barraCuda: METAD bias accumulation with RECT scaling
- toadStool: Multi-replica dispatch and exchange protocol
