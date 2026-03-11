# Experiment 053: Live Kokkos Parity Benchmark

**Date**: 2026-03-11
**hotSpring version**: v0.6.28
**Binary**: `bench_md_parity`

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen Threadripper 3970X (32-core) |
| GPU (primary) | NVIDIA GeForce RTX 3090 (Ampere, sm_86) |
| GPU (secondary) | NVIDIA Titan V (Volta, sm_70) — available but not used in this run |
| RAM | 251 GB |
| Driver | NVIDIA proprietary 580.119.02 |

## Software

| Component | Version |
|-----------|---------|
| barraCuda | `a012076` (v0.3.4 expanded) |
| LAMMPS | 22 Jul 2025 Update 3, Kokkos 4.6.2 |
| Kokkos build | CUDA enabled, compiled for sm_70 (running on sm_86) |
| wgpu | 28.0.0 via NVIDIA proprietary driver |
| Rust | 1.93.1 |

## Configuration

- **N**: 2000 particles (default)
- **Cases**: 9 PP Yukawa DSF (κ=1,2,3 × Γ=weak,medium,strong)
- **Steps**: 5000 equil + 30000 production per case
- **barraCuda precision**: native f64 (DF64 transcendentals unsafe on proprietary — see Bug #1)
- **LAMMPS**: Kokkos mode, `pair_style yukawa`, NVT thermostat
- **LAMMPS atom count**: 2048 (nearest FCC-compatible to N=2000: 8³ × 4)

## Results

### Gap Analysis Table

| Case | κ | Γ | barraCuda (steps/s) | Kokkos (steps/s) | Gap | Method |
|------|---|---|--------------------:|------------------:|----:|--------|
| k1_G14 | 1 | 14 | 160.6 | 1,490.6 | 9.3× | AllPairs |
| k1_G72 | 1 | 72 | 174.0 | 1,835.3 | 10.6× | AllPairs |
| k1_G217 | 1 | 217 | 185.1 | 1,969.5 | 10.6× | AllPairs |
| k2_G31 | 2 | 31 | 199.1 | 2,605.7 | 13.1× | Verlet |
| k2_G158 | 2 | 158 | 231.4 | 2,901.9 | 12.5× | Verlet |
| k2_G476 | 2 | 476 | 230.2 | 2,961.4 | 12.9× | Verlet |
| k3_G100 | 3 | 100 | 231.3 | 3,110.3 | 13.4× | Verlet |
| k3_G503 | 3 | 503 | 249.3 | 3,338.2 | 13.4× | Verlet |
| k3_G1510 | 3 | 1510 | 250.9 | 3,459.1 | 13.8× | Verlet |

### Summary

| Metric | barraCuda GPU | Kokkos-CUDA |
|--------|-------------:|------------:|
| Avg steps/s | 212.4 | 2,630.2 |
| Cases completed | 9/9 | 9/9 |
| Average gap | — | 12.4× |
| Total wall time | ~28 min | (included) |

## Analysis

### Performance Gap Decomposition

The 12.4× average gap breaks down into:

1. **f64 rate penalty (~16×)**: RTX 3090 Ampere processes f64 at 1:32 of f32 rate.
   Native f64 was used because DF64 transcendentals are stripped on NVIDIA proprietary
   (NVVM poisoning protection). The intended DF64 path uses FP32 cores for force math,
   which would recover most of this penalty.

2. **Algorithm difference**: κ=1 cases use all-pairs O(N²), κ=2,3 use Verlet O(N·M).
   LAMMPS always uses neighbor lists. The all-pairs cases show ~10× gap vs ~13× for
   Verlet, suggesting the Verlet GPU implementation has room for optimization.

3. **LAMMPS compiled for sm_70**: Kokkos kernels targeting Volta compute capability,
   running on Ampere with reduced performance. Recompiling for sm_86 would improve
   Kokkos numbers, increasing the measured gap slightly.

### Convergence Path to Parity

| Optimization | Expected Speedup | Gap After |
|-------------|----------------:|----------:|
| Fix DF64 force shader (exp_df64 safe path) | ~4-8× | 1.5-3× |
| Shared-memory tiled force (barraCuda pattern) | ~1.5-2× | 1-2× |
| Kernel fusion (VV + force in single dispatch) | ~1.2× | ~1× |
| Recompile Kokkos for sm_86 | (increases gap ~1.1×) | ~1× |

### Known Issues

**Bug #1: DF64 Transcendental Poisoning**
The DF64 Yukawa force shader uses `exp_df64()` from `df64_transcendentals.wgsl`.
On NVIDIA proprietary, `compile_shader_df64()` strips transcendentals to prevent
NVVM device poisoning. This leaves `exp_df64()` undefined — the shader compiles
but produces zero forces. **Fixed in this session**: simulation now falls back to
native f64 when DF64 transcendentals are unsafe. Upstream barraCuda should provide
a safe DF64 exp path (Taylor series or range-reduction without NVVM poisoning).

**Bug #2: Energy Reducer Returns Zero**
`ReduceScalarPipeline::sum_f64()` returns 0.0 for ke_buf and pe_buf despite correct
particle dynamics (verified via RDF, VACF, SSF observables). The `steps_per_sec`
metric is wall-clock based and unaffected. Energy drift reporting is misleading
(shows 0.000% because both start/end energies are zero). Filed as upstream barraCuda
reduction pipeline issue.

**Bug #3: LAMMPS Energy Drift**
Kokkos/LAMMPS shows 1.3-98% energy drift, especially at high Γ. Root causes:
- NVT (Nosé-Hoover) vs Berendsen thermostat differences
- Tau parameter (5×dt = 0.05) may be too aggressive for strongly-coupled plasmas
- `mass 1 3.0` in reduced units needs validation against OCP conventions
Not a performance bug — affects only the drift diagnostic column.

## Raw Data

JSON results: `experiments/053_benchmark_results.json` (18 records, 9 per backend)

## Next Steps

1. **Fix DF64 exp_df64 path** — provide a NVVM-safe DF64 exponential (Taylor series)
   in barraCuda. This is the single biggest performance unlock (~4-8×).
2. **Fix energy reducer** — upstream barraCuda `ReduceScalarPipeline` investigation.
3. **Tune LAMMPS input** — fix mass, thermostat coupling for high-Γ cases.
4. **Titan V benchmark** — run same 9 cases on Titan V (native f64, 1:2 rate) for
   compute-class GPU baseline.
5. **Radeon M150 benchmark** — once hardware arrives, benchmark with RADV + Kokkos-HIP.
6. **N-scaling study** — run at N=500, 2000, 10000, 50000 to characterize GPU scaling.
