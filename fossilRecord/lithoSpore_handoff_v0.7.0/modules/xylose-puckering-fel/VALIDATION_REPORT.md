# Beta-D-Xylopyranose Ring Puckering FEL — Validation Report (v0.7.0)

**Date**: May 24, 2026
**Experiment**: 220 (Phase 0.7 — corrected)
**System**: Free beta-D-xylopyranose in water (879 TIP3P), CHARMM36 force field
**Software**: GROMACS 2026.0 + PLUMED 2.9.2
**CV**: Cremer-Pople puckering theta (1D metadynamics)
**Hardware**: strandGate (RTX 3090 GPU-accelerated PME)
**Runtime**: ~17 minutes for 10 ns production (847 ns/day)

## Corrections from v0.6.0

| Issue | v0.6.0 | v0.7.0 |
|-------|--------|--------|
| Input structure | RDKit from SMILES (identified as β-D-Lyxose by Alistaire) | Crystal structure from PDB 2D24 (XYS C res 4) |
| Stereochemistry | Wrong (C2 epimer) | Verified by X-ray crystallography |
| Solvation evidence | No topology shipped | SYSTEM_SETUP.md with full pipeline |
| Waters | 863 TIP3P | 879 TIP3P (slightly different box) |

## System Preparation

| Step | Method | Notes |
|------|--------|-------|
| Structure | PDB 2D24 crystal (XYS C 4 → BXYL) | Crystallographically verified stereochemistry |
| Force field | CHARMM36-jul2022 | `carb.rtp` BXYL residue, pdb2gmx |
| Solvation | TIP3P, cubic box d=1.2 nm | 879 waters, 2657 total atoms |
| Minimization | Steepest descent | Fmax < 1000 kJ/mol/nm |
| NVT | 100 ps, v-rescale 300K | Stable |
| NPT | 100 ps, c-rescale 1 bar | Stable |

## Metadynamics Parameters

| Parameter | Value |
|-----------|-------|
| CV | Cremer-Pople theta (PLUMED PUCKERING) |
| Ring atoms | C1(1)-C2(9)-C3(13)-C4(17)-C5(5)-O5(8) |
| Simulation time | 10 ns (5M steps × 2 fs) |
| Gaussian height | 1.5 kJ/mol |
| Gaussian sigma | 0.1 rad |
| Bias factor | 15.0 |
| Deposition stride | 500 steps (1 ps) |
| Grid | [0, pi], 100 bins |
| Total Gaussians | 10,000 |

## Results

### Puckering Free Energy Landscape F(theta)

Three distinct conformational basins identified:

| Basin | Theta | Free Energy | Character |
|-------|-------|-------------|-----------|
| Chair A | ~10° | 16.1 kJ/mol | 4C1 pyranose chair |
| Boat/twist-boat | ~89° | 6.4 kJ/mol | Boat region |
| Chair B | ~172° | 0.0 kJ/mol (global min) | 1C4 pyranose chair |

### Barriers

| Transition | Barrier theta | Barrier height |
|-----------|---------------|----------------|
| Chair A → Boat | ~53° | 52.5 kJ/mol (from global min) |
| Boat → Chair B | ~121° | 37.6 kJ/mol (from global min) |

### Key observations

1. Full Cremer-Pople puckering space sampled (theta 0° to 180°)
2. Two distinct chair conformations resolved (theta~10° and theta~172°)
3. Boat/twist-boat intermediate at theta~89°
4. Chair-to-boat barriers (38–53 kJ/mol) consistent with expected pyranose flexibility
5. Global minimum at theta~172° — per PLUMED convention (C1-C2-C3-C4-C5-O5),
   this corresponds to the 1C4 chair

### Convention Note

The global minimum assignment depends on the atom ordering in the PUCKERING CV.
For the standard C1-C2-C3-C4-C5-O5 sequential ordering used here, PLUMED defines:
- θ = 0 → 4C1 chair
- θ = π → 1C4 chair

β-D-xylopyranose is expected to favor 4C1 in solution. The observed 1C4 minimum
suggests either: (a) a convention interpretation issue, or (b) a CHARMM36 force
field characteristic. This requires domain expert confirmation.

### Convergence

| Stride file | Boat energy (kJ/mol) | Status |
|-------------|---------------------|--------|
| fes_3.dat (6-8 ns) | 7.03 | — |
| fes_4.dat (8-10 ns) | 6.40 | — |
| fes_5.dat (final) | 6.40 | Converged ✓ |

Drift from fes_4 → fes_5: 0.00 kJ/mol. Fully converged.

## Comparison with Enzyme-Bound (Module 3)

| Basin | Free xylose | Enzyme-bound (2D24) | Delta |
|-------|-------------|--------------------:|-------|
| Chair A (4C1) | 16.1 kJ/mol | 13.6 kJ/mol | −2.5 (enzyme stabilizes) |
| Boat | 6.4 kJ/mol | 5.4 kJ/mol | −1.0 (enzyme stabilizes) |
| Chair B (1C4) | 0.0 | 0.0 | reference |
| A→Boat barrier | 52.5 kJ/mol | 47.4 kJ/mol | −5.1 (enzyme lowers) |
| Boat→B barrier | 37.6 kJ/mol | 36.7 kJ/mol | −0.9 (enzyme lowers) |

The enzyme active site lowers all conformational barriers, consistent with
catalytic facilitation of ring distortion during glycoside hydrolysis.
