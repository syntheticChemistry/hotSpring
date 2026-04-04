# Experiment 099: GPU RHMC — All-Flavors Dynamical QCD

**Date**: 2026-03-26
**Status**: INFRASTRUCTURE BUILT — solver integration pending
**Binary**: TBD (production_rhmc_scan)
**Hardware**: RTX 3090, RX 6950 XT

## Goal

Enable GPU-accelerated Rational HMC (RHMC) for fractional determinant powers,
unlocking Nf=2 and Nf=2+1 dynamical fermion simulations at production scale.

## Background

The existing GPU dynamical HMC supports only Nf=4k (staggered, even multiples).
For Nf=2 and Nf=2+1 (the physically relevant case: 2 light quarks + 1 strange),
the rooting trick requires `det(D†D)^{1/2}` and `det(D†D)^{1/4}` via rational
approximations.

RHMC uses multi-shift CG to solve `(D†D + σ_i)x_i = b` for all poles
simultaneously. The key GPU advantage: only ONE D†D·p dispatch per CG iteration,
regardless of pole count. Shifted updates are cheap BLAS-1 operations.

## Infrastructure Delivered

### WGSL Shader: `multi_shift_zeta_f64.wgsl`

Scalar kernel computing the shifted CG recurrence parameters (zeta, alpha_s,
beta_s) for each pole. Runs with workgroup_size(1) since pole count is small
(typically 4-16 for 8-pole rational approximation).

### Rust Module: `gpu_hmc/gpu_rhmc.rs`

- `GpuRhmcSectorBuffers` — per-flavor-sector GPU buffer allocation
  - x_s, p_s vectors per shift (n_shifts × vol × 6 × f64)
  - Zeta recurrence state (zeta_curr, zeta_prev, beta_prev)
  - Alpha/beta output buffers
  - Active flags per shift
- `GpuRhmcPipelines` — zeta recurrence pipeline
- `GpuRhmcState` — combines gauge state with per-sector buffers
- `GpuRhmcResult` — trajectory result with total CG iterations

### CPU RHMC (existing, `lattice/rhmc.rs`)

- `RationalApproximation` with partial fractions and Remez fitting
- `multi_shift_cg_solve` — reference CPU implementation
- `RhmcConfig::nf2()` and `RhmcConfig::nf2p1()` — pre-configured sectors
- `rhmc_heatbath`, `rhmc_fermion_action`, `rhmc_fermion_force`

## What Remains

1. **GPU multi-shift CG solver function** — wire `GpuRhmcSectorBuffers` into
   a `gpu_multi_shift_cg_solve()` that:
   - Dispatches `D†D·p_0` via existing Dirac pipeline (shared Krylov)
   - Runs zeta recurrence kernel
   - Updates x_s, p_s for each shift via existing axpy/xpay shaders
   - Converges using the batched-readback pattern from resident CG

2. **GPU RHMC trajectory** — `gpu_rhmc_trajectory_streaming()` following
   the pattern of `gpu_dynamical_hmc_trajectory_streaming()`:
   - GPU PRNG for momenta + pseudofermion heat bath (existing)
   - RHMC heatbath: `φ = (D†D)^{-p/2} η` via multi-shift CG
   - Omelyan MD with RHMC fermion force
   - Metropolis accept/reject

3. **Production binary** — `production_rhmc_scan` for overnight runs at 32^4

## Memory Budget (32^4, Nf=2+1, 8 poles per sector)

| Component | Bytes | Notes |
|-----------|-------|-------|
| Links (4 × V × 18 × 8) | 603 MB | 4 dirs, SU(3) |
| Momenta | 603 MB | Same as links |
| Force buffer | 603 MB | Per-link |
| Neighbor table | 34 MB | 8 neighbors |
| Light sector (8 poles × (x + p) × V × 6 × 8) | 2×8×48 MB = 768 MB | BLAS vectors |
| Strange sector | 768 MB | Same |
| CG temporaries | ~150 MB | r, Ap, staging |
| **Total** | ~3.5 GB | Fits RTX 3090 (24 GB) |

The 16 GB RX 6950 XT is too tight for 32^4 RHMC. Target: 16^4 on AMD, 32^4 on NVIDIA.

## How This Enables All-Flavors Physics

| Configuration | CPU RHMC | GPU RHMC (target) |
|--------------|----------|-------------------|
| Nf=2 (rooted staggered) | rhmc.rs | gpu_rhmc.rs |
| Nf=2+1 (light + strange) | RhmcConfig::nf2p1() | GpuRhmcState with 2 sectors |
| Nf=2+1+1 (with charm) | 3 sectors | 3 sector buffers |

## Cross-Card Strategy

- **RTX 3090**: Production 32^4, Nf=2+1. Higher FP32 TFLOPS for Dirac dispatch.
  Tensor cores inaccessible via WGSL but ready for coralReef sovereign path.
- **RX 6950 XT**: Validation 16^4, Nf=2+1. DF64 advantage for precision-critical
  CG accumulation. 128 MB Infinity Cache helps cache-resident lattice sizes.
- **Cross-validation**: Run same physics on both cards, compare plaquettes and ΔH.
