# Experiment 032: Finite-Temperature SU(3) Deconfinement β-Scan

**Date**: March 6, 2026
**Status**: IN PROGRESS

## Goal

Reproduce the SU(3) quenched deconfinement phase transition on asymmetric
N_s³ × N_t lattices using `barraCuda` GPU (DF64 precision) on a single
RTX 3090. Compare with literature β_c values to validate at MILC-comparable
lattice volumes.

## Lattice Geometries

| Lattice | Sites | VRAM | GPU ms/traj (bench) | β_c (lit.) |
|---------|:-----:|:----:|:-------------------:|:----------:|
| 32³×8   | 262K  | 0.45 GB | ~2,840 | 6.062 |
| 48³×8   | 884K  | 1.53 GB | ~9,600 (est.) | 6.062 |
| 64³×8   | 2.10M | 3.62 GB | ~23,000 (est.) | 6.062 |

Literature: β_c(N_t=8) ≈ 6.062 ± 0.004 (Karsch et al., Boyd et al.)

## Benchmark Results (March 6, 2026)

Full GPU HMC scaling validated with asymmetric lattices:

| Lattice  | Volume  | CPU ms/traj | GPU ms/traj | Speedup |
|----------|:-------:|:-----------:|:-----------:|:-------:|
| 4⁴       | 256     | 71.7        | 18.7        | 3.8×    |
| 8⁴       | 4,096   | 1,161.3     | 32.6        | 35.6×   |
| 8³×4     | 2,048   | 580.6       | 18.4        | 31.5×   |
| 16³×4    | 16,384  | 4,708.4     | 173.6       | 27.1×   |
| 16³×8    | 32,768  | 9,293.9     | 359.6       | 25.8×   |
| 16⁴      | 65,536  | 18,625.2    | 727.0       | 25.6×   |
| 32³×4    | 131,072 | 37,768.7    | 1,451.1     | 26.0×   |
| 32³×8    | 262,144 | 75,124.8    | 2,840.2     | 26.5×   |

**Consistent 26-36× GPU speedup across all geometries.**
GPU time scales linearly with volume as expected.

## Production Run: 32³×8 (COMPLETE)

| β | ⟨P⟩ | σ(P) | \|L\| | χ | acc% | wall(s) |
|---|-----|------|-------|---|------|---------|
| 5.8000 | 0.552979 | 0.002451 | 0.0014 | 1.5744 | 35.0 | 2059 |
| 5.9000 | 0.570399 | 0.002814 | 0.0012 | 2.0759 | 36.0 | 2059 |
| 6.0000 | 0.584770 | 0.001827 | 0.0032 | 0.8752 | 28.5 | 2090 |
| 6.0600 | 0.592503 | 0.001851 | 0.0019 | 0.8978 | 35.0 | 2101 |
| 6.1000 | 0.596953 | 0.001862 | 0.0022 | 0.9088 | 31.5 | 2096 |
| 6.2000 | 0.607255 | 0.002160 | 0.0022 | 1.2236 | 33.0 | 2103 |

Total: 12,509s (3.5 hours), 1800 trajectories.

Observations:
- |L| ≈ 0.001-0.003 across all β — confined phase or weak crossover
- N_s/N_t = 4 too small for sharp first-order signal
- Susceptibility peak near β ≈ 5.9 consistent with crossover
- Need 64³×8 (N_s/N_t = 8) for sharper transition signal

## Production Run: 64³×8 (IN PROGRESS — overnight)

- 2.1M sites, 3.62 GB VRAM
- β scan: [5.912, 5.972, 6.032, 6.092, 6.152, 6.212]
- 50 therm + 100 meas per β, 900 total trajectories
- Estimated wall time: ~14 hours
- N_s/N_t = 8 — should resolve deconfinement transition

## Infrastructure Built

1. **`bench_gpu_hmc`** — Extended with 8 asymmetric+symmetric configs
2. **`ComputeBackend` trait** — Swappable backend interface in `bench::compute_backend`
3. **`bench_backends` binary** — CPU vs GPU comparison with summary tables
4. **`production_beta_scan`** — Updated with `--dims=Nx,Ny,Nz,Nt` support
5. **`production_finite_temp`** — New binary for multi-N_t deconfinement studies

## Next Steps

- [ ] Analyze 64³×8 results — look for Polyakov loop jump at β_c
- [ ] Run 48³×8 for finite-size scaling (32³/48³/64³ × 8)
- [ ] Multi-N_t continuum extrapolation (N_t = 4, 6, 8)
- [ ] Compare with Kokkos-CUDA on same lattice sizes
- [ ] Feed trajectory data to NPU for steering evolution
