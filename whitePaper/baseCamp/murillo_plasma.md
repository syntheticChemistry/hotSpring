# Murillo Group — Dense Plasma Molecular Dynamics

**Papers:** 1-6 (Sarkas MD, TTM, Surrogate Learning, Transport, Screened Coulomb)
**Updated:** February 27, 2026
**Status:** ✅ 60/60 observable checks pass, paper parity achieved. Dynamical fermion run planned (Exp 023)

---

## What We Reproduced

The Murillo Group at Michigan State University published foundational work on
dense plasma properties using the Sarkas molecular dynamics package. Our
reproduction covers:

| Study | Paper | Checks | Result |
|-------|-------|--------|--------|
| Sarkas Yukawa OCP | Papers 1-3 | 60/60 | 9 PP cases, N=10k, 80k steps, 0.000-0.002% drift |
| TTM Laser-Plasma | Paper 4 | 6/6 | 3 local + 3 hydro, all species equilibrate |
| Stanton-Murillo Transport | Paper 5 | 13/13 | D*, η*, λ* Green-Kubo vs Sarkas-calibrated fits |
| Screened Coulomb | Paper 6 | 23/23 | Sturm bisection, Python parity Δ≈10⁻¹² |

## Evolution Path

1. **Python control** (Phase A): Ran original Sarkas v1.0.0 on our hardware. Fixed 3 upstream bugs (NumPy 2.x, pandas 2.x, Numba/pyfftw). 86/86 checks.

2. **Rust CPU** (Phase B): Built BarraCuda from first principles. LJ, Yukawa, Morse, Verlet, FCC lattice, cell-list, all in pure Rust. Nuclear EOS L1 achieved χ²/datum = 2.27 (478× faster than Python, better minimum).

3. **GPU acceleration** (Phase C): All force evaluation, integration, and observables on GPU via WGSL f64 shaders. 9/9 PP Yukawa DSF cases on RTX 4070. 0.000% energy drift at 80k steps.

4. **Native f64 builtins** (Phase D): Replaced software-emulated f64 transcendentals with hardware builtins. 2-6× throughput improvement. Paper parity in 5.3 minutes (N=10k).

5. **Paper-parity long run** (Phase E): 9-case sweep at N=10,000, 80k production steps. 3.66 hours, $0.044 electricity. Cell-list 4.1× faster than all-pairs.

## Key Finding

A $600 consumer GPU (RTX 4070) reproduces the exact same physics as the Murillo Group's HPC cluster runs — same particle count, same production steps, same energy conservation — in 3.66 hours for $0.044 in electricity.

## Next: Dynamical Fermion QCD (Exp 023)

The quenched SU(3) runs (Exp 013, 018, 022) validated pure gauge physics at 32⁴.
Exp 023 extends to dynamical staggered fermions — the first dynamical QCD at 32⁴
on consumer GPU. The NPU now runs 11 inference heads: GPU prep (quenched length,
parameter suggestion, CG estimate), quenched monitoring with early-exit, dynamical
thermalization and rejection, phase classification, and intra-scan adaptive steering.

Expected: crossover (not first-order) phase transition, β_c shifted lower than
5.69, CG solver at 50–200 iterations per trajectory. Run planned for weekend of
Feb 28–Mar 1, 2026. Results will inform the March 4 coffee with Professor Murillo.

## Cross-Spring Contributions

- **Pairwise distance shaders** → toadStool `distance_f64.wgsl` → used by wetSpring for phylogenetic distances
- **Cell-list GPU** → toadStool `CellListGpu` → used by all springs with spatial decomposition
- **ESN reservoir** → toadStool `esn_*.wgsl` → used by neuralSpring for time-series prediction
- **Transport coefficients** → GPU Green-Kubo D*/η*/λ* → toadStool `GpuVelocityRing`
