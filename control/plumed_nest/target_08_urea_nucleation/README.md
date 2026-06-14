# Target 08: Urea/Glycine Crystal Nucleation — OPES + GNN CVs

## PLUMED-NEST Reference

- **plumID**: 22.039
- **Paper**: Zou & Tiwary, PNAS 2023
- **Archive**: https://github.com/zpengmei/Nucleation-OPES-GNN/archive/main.zip
- **Method**: OPES with graph neural network (GNN) learned CVs
- **Systems**: Urea and glycine polymorphs in water (nucleation)

## Key Scientific Content

Materials science application: uses OPES enhanced sampling with GNN-learned
collective variables to drive crystal nucleation of small molecules. Provides
nucleation barriers and polymorph free energy differences.

Computationally analogous to phase transitions — a barraCuda GPU parity target
for parallel force/energy computation on large homogeneous systems.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Method | OPES with GNN-learned CVs |
| Systems | Urea, glycine polymorphs in water |
| Software | GROMACS + PLUMED 2.8 |
| Runtime | ~500 ns, GPU-intensive |
| System size | Large (thousands of solute molecules) |

## Reproduction Status

- [ ] Archive download (GitHub)
- [ ] PLUMED input validation
- [ ] GNN model architecture review
- [ ] System topology preparation
- [ ] OPES production run
- [ ] Nucleation barrier comparison
- [ ] Polymorph free energy comparison

## Parity Target

- barraCuda: Large-system force computation (homogeneous = GPU-ideal)
- barraCuda: GNN-CV evaluation on GPU (graph convolution)
- toadStool: Large-system dispatch + checkpoint management
- NUCLEUS: Materials science composition (phase transition detection)
