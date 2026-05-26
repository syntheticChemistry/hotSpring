# Target 07: Alpha-Amylase Sugar Catalysis — OPES + QM/MM

## PLUMED-NEST Reference

- **plumID**: 25.012
- **Paper**: Das et al., ACS Catalysis 2025
- **Archive**: https://github.com/sudipdas789/Committor_Amylase/raw/main/Committor_Amylase_PLUMED_NEST.zip
- **Method**: OPES + committor function + machine learning CV
- **Systems**: Alpha-amylase enzyme + sugar substrate (QM/MM level)

## Key Scientific Content

Frontier study: uses OPES enhanced sampling with a machine-learning-based
committor CV to characterize the catalytic mechanism of alpha-amylase at
QM/MM level (GROMACS + CP2K). This is sugar enzyme catalysis — our domain
— at the highest theoretical level.

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Method | OPES with ML committor CV |
| QM engine | CP2K 9.1 (DFT) |
| MM engine | GROMACS 2021.5 |
| PLUMED | 2.9 |
| Level | QM/MM (enzyme active site = QM region) |

## Reproduction Status

- [ ] Archive download (GitHub)
- [ ] PLUMED input validation
- [ ] QM/MM setup analysis (CP2K input review)
- [ ] ML model architecture documentation
- [ ] Analysis of pre-generated data (if available)
- [ ] FES + committor comparison to published

## Parity Target

- Long-term: QM/MM interface for NUCLEUS (barracuda + quantum module)
- Intermediate: Analyze published data for reaction mechanism validation
- barraCuda: ML-CV computation on GPU
