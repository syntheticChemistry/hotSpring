# Target 02: Chignolin Protein Folding — OPES + OPES-Explore

## PLUMED-NEST Reference

- **plumID**: 24.029
- **Paper**: Ray & Rizzi, JCTC 21, 58–69 (2024)
- **DOI**: 10.1021/acs.jctc.4c00592
- **Method**: OPES_METAD + OPES_METAD_EXPLORE combination
- **System**: Chignolin mini-protein (10 residues, 166 atoms) in explicit TIP3P

## Key Scientific Content

Combines two OPES algorithms to achieve rapid convergence on protein folding:
- OPES_METAD: deposits bias to enhance sampling of the folding CV
- OPES_METAD_EXPLORE: concurrent exploration bias for faster barrier crossing

Two CV approaches tested:
1. **HLDA** (Harmonic Linear Discriminant Analysis): 6 native contacts combined
2. **DeepTICA**: Deep learning-based time-lagged independent component

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Integrator | MD (leapfrog) |
| Timestep | 2 fs |
| Total time | 1 μs (500M steps) |
| Temperature | 340 K (V-rescale) |
| Force field | CHARMM22* |
| Water | TIP3P |
| Cutoff scheme | Verlet |
| Electrostatics | PME |
| OPES BARRIER (explore) | 20 kJ/mol |
| OPES BARRIER (metad) | 30 kJ/mol |
| PACE | 500 steps |

## CVs and Methods

### HLDA Collective Variable
```
hlda = 0.6188*d1 + 0.5975*d2 + 0.5045*d3 - 0.0708*d4 + 0.0217*d5 + 0.0140*d6
```
Where d1-d6 are native contact switches (CONTACTMAP with RATIONAL switching functions).

### DeepTICA
PyTorch model (`deep_tica.ptc`) trained on inter-atomic distances (~200 descriptors)
using time-lagged independent component analysis with deep neural network.

## Reference Values

- Folding free energy: ΔG_fold ≈ -5 to -8 kJ/mol (two-state folder)
- Folded state: RMSD_Cα < 0.1 nm
- Unfolded state: RMSD_Cα > 0.3 nm, end-to-end distance > 1.5 nm

## Files Ingested

- `plumed_hlda.dat` — OPES+OPES-Explore with HLDA CV
- `plumed_deeptica.dat` — OPES+OPES-Explore with DeepTICA model
- `plumed-descriptors.dat` — Inter-atomic distance definitions for DeepTICA
- `chignolin-ref.pdb` — Reference structure (MOLINFO)
- `chignolin-ca.pdb` — Cα reference for RMSD
- `prd.gro` — Production starting structure (solvated)
- `topol_01.top` — GROMACS topology (CHARMM22*)
- `md.mdp` — Production MDP (500M steps)
- `deep_tica.ptc` — PyTorch traced model for DeepTICA
- `charmm22star.ff/` — Full force field directory

## Reproduction Status

- [x] Archive downloaded and extracted (7.1 MB)
- [x] Full simulation inputs available (topology, structure, force field)
- [x] PLUMED HLDA validates (OPES_METAD + OPES_METAD_EXPLORE PASS)
- [x] PLUMED DeepTICA structure identified (requires PyTorch model)
- [ ] Short production test (10 ns subset)
- [ ] Full 1 μs production run
- [ ] FES reweighting analysis
- [ ] ΔG_fold comparison to reference

## Why This Target Matters

- Tests OPES (next-generation enhanced sampling, successor to metadynamics)
- Protein folding = fundamental biophysics benchmark
- Explicit solvent + full protein = validates force computation at scale
- Two CV approaches (linear vs. ML) provide comparison for barraCuda parity

## Parity Target

- barraCuda: CONTACTMAP CV computation (native contacts)
- barraCuda: OPES bias accumulation (kernel-based, different from Gaussian metad)
- barraCuda: RMSD computation (OPTIMAL alignment)
- toadStool: Long-timescale dispatch (μs simulations)
