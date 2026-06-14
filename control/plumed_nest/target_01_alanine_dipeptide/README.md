# Target 01: Alanine Dipeptide — Well-Tempered Metadynamics

## PLUMED-NEST Reference

- **plumID**: 24.020 / Lugano 2019 / Masterclass 21.4
- **Paper**: Barducci, Bussi, Parrinello, PRL 100, 020603 (2008)
- **Method**: Well-tempered metadynamics on phi/psi backbone dihedrals
- **System**: Alanine dipeptide in pseudo-vacuum (large box, PBC, no solvent)
- **Atoms**: 22

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Integrator | MD (leapfrog) |
| Timestep | 2 fs |
| Total time | 10 ns (5M steps) |
| Temperature | 300 K (V-rescale) |
| Cutoff scheme | Verlet |
| BIASFACTOR | 10 |
| SIGMA | 0.35, 0.35 rad |
| HEIGHT | 1.2 kJ/mol |
| PACE | 500 steps (1 ps) |
| Gaussians deposited | 10,000 |
| Grid | 150x150 (-pi to pi) |

## Software

- GROMACS 2026.0-conda_forge
- PLUMED 2.9.2
- Force field: AMBER99SB (vacuum)

## Reference Values (canonical)

| Basin | phi (rad) | psi (rad) | Relative FE (kJ/mol) |
|-------|-----------|-----------|---------------------|
| C7eq (global min) | -1.4 | 1.3 | 0.0 |
| C7ax | 1.2 | -1.2 | ~8-12 |
| alpha-R | -1.4 | -0.6 | ~4-6 |
| Barrier (C7eq→alpha-R) | — | — | ~20-30 |

## Results

- HILLS: 10,007 lines (10,000 Gaussians + header)
- COLVAR: 500,006 frames (every 100 steps)
- FES reconstructed via `plumed sum_hills` (stride convergence + 2D)
- Global minimum at phi = -1.42 rad (C7eq), consistent with literature

## Status

- [x] PLUMED input validates (`plumed driver --parse-only`)
- [x] Simulation complete (10 ns)
- [x] FES reconstructed (2D + 1D projections)
- [x] Convergence verified (stride analysis: fes_0..fes_10)
- [x] Reference basins match literature

## Parity Target

This FEL becomes the Tier 0 baseline for:
1. barraCuda TORSION CV shader (phi/psi dihedral computation)
2. barraCuda METAD bias accumulation (Gaussian deposition + grid)
3. toadStool dispatch of metadynamics workload
