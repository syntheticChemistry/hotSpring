# Target 03: BRD4/HSP90 — Absolute Binding Free Energy (OneOPES)

## PLUMED-NEST Reference

- **plumID**: 24.017
- **Paper**: Karrenbrock et al., JPCL 15, 9871–9880 (2024)
- **Archive**: https://zenodo.org/records/11126468/files/Supporting_Material.zip
- **Method**: OneOPES (8 replicas per system)
- **Systems**: BRD4 bromodomain + ligand, HSP90 + inhibitor

## Key Scientific Content

State-of-the-art absolute binding free energy calculation using the OneOPES
protocol: 8 replicas with OPES bias, achieving experimental accuracy within
1-2 kcal/mol for protein-ligand binding.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Replicas | 8 per system |
| Method | OPES (OneOPES protocol) |
| Temperature | 300 K |
| Software | GROMACS 2022 + PLUMED 2.9 |
| Runtime | ~200 ns per replica |

## Reproduction Status

- [ ] Archive download (Zenodo)
- [ ] PLUMED input validation
- [ ] Topology preparation
- [ ] 8-replica production (BRD4)
- [ ] Binding free energy analysis
- [ ] Comparison to experimental ΔG

## Parity Target

- barraCuda: OPES bias with multi-replica exchange
- toadStool: 8-replica parallel dispatch + exchange
- NUCLEUS: Full protein-ligand binding free energy pipeline
