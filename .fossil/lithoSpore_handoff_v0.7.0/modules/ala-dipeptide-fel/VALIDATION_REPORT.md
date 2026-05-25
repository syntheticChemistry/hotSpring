# Alanine Dipeptide Well-Tempered Metadynamics — Validation Report

**Date**: May 24, 2026  
**Experiment**: 220 (Phase 0.4)  
**System**: Alanine dipeptide in vacuum, AMBER99SB-ILDN force field  
**Software**: GROMACS 2026.0-conda_forge + PLUMED 2.9.2  
**Hardware**: CPU (thread-MPI, 4 OpenMP threads)  
**Runtime**: ~10 minutes for 10 ns  

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Integrator | Leapfrog MD |
| Timestep | 2 fs |
| Duration | 10 ns (5,000,000 steps) |
| Temperature | 300 K (v-rescale) |
| Collective Variables | phi (C-N-CA-C), psi (N-CA-C-N) |
| Gaussian height | 1.2 kJ/mol |
| Gaussian sigma | 0.35 rad (both CVs) |
| Deposition stride | 500 steps (1 ps) |
| Bias factor (gamma) | 6.0 |
| Grid | [-pi, pi] × [-pi, pi], spacing 0.1 rad |
| Total Gaussians deposited | 10,000 |
| PBC | xyz (10 nm cubic box, pseudo-vacuum) |
| Cutoffs | 4.5 nm (all interactions captured) |
| Constraints | h-bonds (LINCS) |

## Results

### 2D Free Energy Landscape (phi, psi)

**Global minimum (C7eq)**: phi = -81.2°, psi = 52.9° (0.00 kJ/mol)

Top basins identified:
- **C7eq**: phi ≈ -81°, psi ≈ 53° — global minimum (0.0 kJ/mol)
- **C5 (beta)**: phi ≈ -149°, psi ≈ 159° — secondary minimum (+0.65 kJ/mol)
- **C7ax**: phi ≈ +60°, psi ≈ -39° — tertiary minimum (+5.57 kJ/mol)

Energy range: 0.00 to 68.51 kJ/mol

### 1D Projections

**F(phi)** local minima:
- phi = -81.2° (0.00 kJ/mol) — C7eq
- phi = -144.7° (1.91 kJ/mol) — C5/beta
- phi = +60.0° (5.57 kJ/mol) — C7ax

**F(psi)** local minima:
- psi = 158.8° (0.00 kJ/mol)
- psi = 52.9° (0.60 kJ/mol)
- psi = -38.8° (6.21 kJ/mol)

### Convergence

Basin free energy difference ΔF(C7ax − C7eq) over the last 4 ns:

| Time window | ΔF (kJ/mol) |
|-------------|-------------|
| 7 ns | +5.83 |
| 8 ns | +5.34 |
| 9 ns | +5.31 |
| 10 ns | +5.57 |

Convergence within ±0.5 kJ/mol over the last 3 ns. Well-tempered metadynamics smoothly converged.

## Validation Against Literature

| Property | Expected (AMBER99SB-ILDN) | Observed | Status |
|----------|--------------------------|----------|--------|
| C7eq phi | -80° ± 10° | -81.2° | PASS |
| C7eq psi | +80° ± 30° | +52.9° | PASS |
| C7ax phi | +60° ± 15° | +60.0° | PASS |
| ΔF(C7ax − C7eq) | 5–7 kJ/mol | 5.57 kJ/mol | PASS |
| C5 basin present | Yes | Yes (+0.65 kJ/mol) | PASS |
| Convergence | < 1 kJ/mol drift | ±0.5 kJ/mol | PASS |

## Conclusion

GROMACS 2026.0 + PLUMED 2.9.2 correctly reproduces the expected Ramachandran free energy landscape for alanine dipeptide in vacuum. The three canonical basins (C7eq, C5, C7ax) are resolved with correct relative energetics and positions. The simulation converged smoothly within 10 ns of well-tempered metadynamics.

This validates Phase 0.4 of Experiment 220 and confirms our GROMACS+PLUMED stack is operational for metadynamics-based FEL generation. The pipeline (grompp → mdrun+plumed → sum_hills → FEL analysis) is fully functional.

## Next Steps (Phase 0.5+)

1. Set up GH10 xylanase CAZyme system with GROMOS 45a4 force field
2. Define Cremer-Pople collective variables for sugar ring puckering
3. Run production metadynamics on the CAZyme substrate
4. Generate conformational FEL for CAZyme validation
