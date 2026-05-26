# Target 04: Muscarinic M2 GPCR — Funnel Metadynamics

## PLUMED-NEST Reference

- **plumID**: 20.000
- **Paper**: Capelli et al., JCTC 2019 (Parrinello group)
- **Archive**: https://github.com/riccardocapelli/papers_data/raw/master/muscarinic_m2_2019/input_data.zip
- **Method**: Funnel metadynamics, multiple walkers, well-tempered
- **System**: Muscarinic M2 receptor + iperoxo (membrane-embedded GPCR)

## Key Scientific Content

Historical, well-cited entry from the Parrinello group. Demonstrates full
unbinding pathway of a ligand from a membrane-embedded GPCR using funnel
restraints to focus sampling on the binding/unbinding channel.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Method | Funnel metadynamics (well-tempered) |
| Walkers | Multiple (parallel-bias) |
| System | Membrane-embedded protein (~100k atoms) |
| Software | GROMACS + PLUMED 2.5 |
| Runtime | ~500 ns total across walkers |

## Reproduction Status

- [ ] Archive download (GitHub)
- [ ] PLUMED input validation
- [ ] Membrane system topology
- [ ] Multi-walker production
- [ ] Unbinding PMF reconstruction
- [ ] Binding free energy comparison

## Parity Target

- barraCuda: Funnel restraint geometry + TORSION/DISTANCE CVs
- barraCuda: Multiple-walker bias accumulation + sharing
- toadStool: Multi-walker parallel dispatch with bias exchange
- NUCLEUS: Membrane protein enhanced sampling pipeline
