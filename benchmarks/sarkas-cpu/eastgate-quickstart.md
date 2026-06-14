# Eastgate - Sarkas Quickstart Results

**Date**: February 7, 2026  
**Gate**: Eastgate (local dev)  
**Status**: PASS - First successful Sarkas simulation on the basement HPC

---

## Hardware

- **CPU**: (Eastgate - check `lscpu` for model)
- **GPU**: RTX 4070 (not used for this CPU-only run)
- **NPU**: BrainChip Akida (not used)
- **RAM**: (check `free -h`)
- **Numba threads**: 24

## Software

- **Python**: 3.9.23
- **Sarkas**: 1.0.2
- **Numba**: 0.60.0
- **NumPy**: 1.26.4
- **SciPy**: 1.13.1
- **pyfftw**: 0.12.0
- **FFTW3**: 3.3.8 (system)
- **Environment**: micromamba (sarkas env)

## Simulation Parameters

- **Input file**: `yocp_cgs_pp.yaml` (from Sarkas tutorial)
- **Particles**: 1,000
- **Species**: Carbon (C), Z=1.976599
- **Potential**: Yukawa (screened Coulomb), PP method
- **Cutoff**: 6.0e-8 cm
- **Integrator**: Verlet, dt=5.0e-17 s
- **Thermostat**: Berendsen (tau=2.0, relaxation=300 steps)
- **Equilibration**: 10,000 steps
- **Production**: 20,000 steps
- **Boundary**: Periodic
- **Units**: CGS

## Timing

| Phase | Time |
|-------|------|
| Preprocessing | 2.89s |
| Simulation | 154.92s |
| Post-processing | 1.67s |
| **Total** | **159.49s** |

## Output

- **g(r)**: 250 bins, C-C pair correlation function
- **Thermodynamics**: 2001 production data points
  - Total Energy: mean=6.851e-06, std=7.057e-08 (relative ~1.0%)
  - Kinetic, Potential, Temperature columns present
- **Dump files**: Position/velocity checkpoints every 10 steps

## Notes

- This is a 1000-particle PP (direct pairwise) simulation. No FFT involved.
- For PPPM (particle-particle particle-mesh with FFT), use `yukawa_mks_p3m.yaml` with 10K particles.
- Numba JIT compilation adds ~1-2s on first run (included in preprocessing).
- 24 Numba threads on Eastgate - will compare to Strandgate's 64.
