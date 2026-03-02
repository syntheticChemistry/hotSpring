SPDX-License-Identifier: AGPL-3.0-only

# Cross-Spring Shader Evolution — hotSpring's View

**Date:** March 2, 2026
**Synced to:** toadStool S78 (barracuda v0.2.0)
**License:** AGPL-3.0-only

---

## How Springs Evolve ToadStool

Each spring (hotSpring, wetSpring, neuralSpring, airSpring, groundSpring) is an
independent biome. ToadStool is the shared fungus — present in every biome. The
pattern is Write → Validate → Hand off → Absorb → Lean:

1. **Write**: Spring implements physics locally with WGSL templates
2. **Validate**: Test against known baselines (Python, analytical, experimental)
3. **Hand off**: Document in `wateringHole/handoffs/` with code locations
4. **Absorb**: ToadStool team absorbs as GPU ops and WGSL shaders
5. **Lean**: Spring rewires imports to upstream, deletes local code

Cross-spring evolution happens when a shader written for one domain proves
useful in another. The shared library benefits all springs simultaneously.

---

## Cross-Spring Shader Provenance (from hotSpring's perspective)

### Shaders hotSpring Wrote → toadStool Absorbed → All Springs Benefit

| Shader | Origin | Absorbed | Used By |
|--------|--------|----------|---------|
| `complex_f64.wgsl` | hotSpring lattice QCD | S18 (Feb 16) | All springs (complex arithmetic) |
| `su3.wgsl` | hotSpring SU(3) gauge | S18 (Feb 16) | hotSpring, neuralSpring |
| `df64_core.wgsl` | hotSpring DF64 core streaming | S58 (Feb 26) | All springs (universal precision) |
| `su3_df64.wgsl` | hotSpring DF64 gauge math | S58 (Feb 26) | hotSpring |
| `esn_reservoir_update_f64.wgsl` | hotSpring MD transport | S21 (Feb 21) | hotSpring, wetSpring |
| `esn_readout_f64.wgsl` | hotSpring ESN readout | S21 (Feb 21) | hotSpring, wetSpring |
| `wilson_plaquette_f64.wgsl` | hotSpring plaquette | S18 (Feb 16) | hotSpring |
| `su3_hmc_force_df64.wgsl` | hotSpring HMC force | S18 (Feb 16) | hotSpring |
| HFB 10 shaders (spherical + deformed) | hotSpring nuclear | S39 (Feb 22) | hotSpring |
| `hermite_f64.wgsl`, `laguerre_f64.wgsl` | hotSpring nuclear basis | S25 (Feb 21) | hotSpring, neuralSpring |
| `heat_current_f64.wgsl` | hotSpring transport | S70+ (Feb 28) | hotSpring |
| `weighted_dot_f64.wgsl` | hotSpring nuclear potential | S25 | hotSpring |
| `vacf_batch_f64.wgsl` | hotSpring VACF design | S70+ (Feb 28) | hotSpring |
| `stress_virial_f64.wgsl` | hotSpring viscosity | S70+ (Feb 28) | hotSpring |

### Shaders Other Springs Wrote → hotSpring Benefits From

| Shader | Origin Spring | Absorbed | How hotSpring Benefits |
|--------|--------------|----------|----------------------|
| `bray_curtis_f64.wgsl` | wetSpring | S18 (Feb 16) | Available for diversity metrics |
| `hmm_forward_log_f64.wgsl` | wetSpring+neuralSpring | S21–S25 | Pattern recognition infrastructure |
| `linear_regression_f64.wgsl` | neuralSpring | S25 baseCamp V18 | GPU OLS for transport fits |
| `matrix_correlation_f64.wgsl` | neuralSpring | S25 baseCamp V18 | GPU correlation matrix |
| `pairwise_hamming_f64.wgsl` | neuralSpring | S25 | Distance metrics |
| `rk4_parallel_f64.wgsl` | neuralSpring | S25 | ODE integration |
| `hill_f64.wgsl` | wetSpring | S25 | Nonlinear response curves |
| `moving_window_f64.wgsl` | airSpring | S70+ | Streaming statistics |
| `kriging_f64.wgsl` | airSpring | S70+ | Spatial interpolation |
| `brent_f64.wgsl` | airSpring | S70+ | Root-finding |
| `rawr_weighted_mean_f64.wgsl` | groundSpring | S70+ | Bootstrap statistics |
| DF64 ML (gelu, sigmoid, softmax, sdpa) | airSpring+neuralSpring | S70+ | ML inference in DF64 |

### Cross-Pollination Examples

| From → To | What Traveled | Why It Helped |
|-----------|--------------|---------------|
| hotSpring → wetSpring | `math_f64` preamble, f64 polyfills | wetSpring's bio shaders needed precision |
| hotSpring → neuralSpring | `ReduceScalarPipeline`, `BatchedEighGpu` | Neural networks need reduction + eigensolve |
| wetSpring → hotSpring | `log_f64` fix, constant patterns | Fixed precision bug in hotSpring's shaders |
| neuralSpring → hotSpring | `rk4_parallel`, `eigh_f64` Householder+QR | ODE integration, eigensolve improvements |
| airSpring → all | `moving_window`, `kriging`, FAO-56 ET₀ | Streaming stats, interpolation, evapotranspiration |

---

## Benchmark Results (March 2, 2026 — RTX 3090 NVK)

### Special Functions (CPU, barracuda::special)

| Function | ns/eval | Provenance |
|----------|---------|------------|
| `gamma(x)` | 39 | hotSpring nuclear HFB |
| `erf(x)` | 16 | hotSpring tolerances |
| `bessel_j0(x)` | 22 | hotSpring spectral |
| `hermite(10,x)` | 11 | hotSpring nuclear basis |
| `laguerre(5,x)` | 18 | hotSpring deformed basis |

### GPU Ops (barracuda::ops via ComputeDispatch)

| Op | N | Time | Provenance |
|----|---|------|------------|
| VACF batch | 64×200 frames | 59ms (100 lags) | hotSpring→S70+ |
| VACF batch | 1024×200 frames | 584ms (100 lags) | hotSpring→S70+ |
| Matrix correlation | 100×10 | 10ms | neuralSpring→S25 |
| Matrix correlation | 2000×50 | 19ms | neuralSpring→S25 |
| Stress virial | 500 atoms | 7ms | hotSpring→S70+ |
| Stress virial | 2000 atoms | 13ms | hotSpring→S70+ |
| Linear regression | — | SKIP (naga dynamic indexing) | neuralSpring→S25 |

**Note**: VACF GPU is slower than CPU at N≤1024 because each lag value is a
separate dispatch (100 dispatches for 100 lags). The GPU path wins at larger
N (>10,000 atoms) or longer frame counts where the per-dispatch cost is
amortized. The CPU path remains the default for hotSpring's typical N=108–1000
MD simulations.

---

## Current Lean Inventory

| hotSpring Module | Upstream barracuda | Since |
|------------------|--------------------|-------|
| `spectral/` | `barracuda::spectral::*` | S25 (v0.6.4) |
| `md/celllist.rs` | `barracuda::ops::md::CellListGpu` | S25 (v0.6.2) |
| Complex64 WGSL | `barracuda::ops::lattice::complex_f64` | S25 |
| SU(3) WGSL | `barracuda::ops::lattice::su3` | S25 |
| DF64 core math | `barracuda::ops::lattice::su3::WGSL_DF64_CORE` | S58 (v0.6.9) |
| DF64 gauge force | `barracuda::ops::lattice::su3::WGSL_SU3_DF64` | S58 (v0.6.10) |
| GPU Polyakov loop | bidirectional | S68 (v0.6.13) |
| `Fp64Strategy` | `barracuda::device::driver_profile::Fp64Strategy` | S58 (v0.6.10) |
| `WgslOptimizer` | `barracuda::device::WgslOptimizer` | S42 (v0.5.15) |
| `GpuDriverProfile` | `barracuda::device::driver_profile` | S42 (v0.5.15) |
| Special functions | `barracuda::special::*` | S25–S68 |
| SSF GPU | `barracuda::ops::md::observables::SsfGpu` | S25 |
| **VACF GPU** (new) | `barracuda::ops::md::compute_vacf_batch` | **S78 sync** |
| **Stress virial GPU** (available) | `barracuda::ops::md::compute_stress_virial` | **S78 sync** |
| **Matrix correlation GPU** (available) | `barracuda::ops::stats_f64::matrix_correlation` | **S78 sync** |

---

## What Remains Local to hotSpring

| Module | Why It Stays |
|--------|-------------|
| `md/transport.rs` (Stanton-Murillo, Daligault) | Domain-specific transport fits |
| `md/reservoir/` (11-head ESN + NPU) | Transport-specific, NPU integration |
| `physics/` (SEMF, HFB, BCS) | Nuclear physics domain |
| `lattice/gpu_hmc/` (orchestration) | Deeply coupled to validation infrastructure |
| `gpu/adapter.rs` (primary/secondary discovery) | hotSpring-specific env vars and multi-GPU |
| `production/` (4-layer brain, NPU worker) | Domain-specific pipeline |
| `npu_experiments/` | Domain-specific NPU experiments |
| `discovery/` (data path discovery) | Data-specific, not device-specific |

---

## Summary

844 WGSL shaders now live in barracuda (toadStool S78). hotSpring contributed
~100 of those, primarily precision-critical lattice QCD, nuclear physics, and
MD transport shaders. The cross-spring model works: hotSpring's `math_f64`
preamble is used by wetSpring's bio shaders; neuralSpring's `rk4_parallel`
improves hotSpring's ODE paths; airSpring's `moving_window` is available for
streaming diagnostics. Every spring benefits from every other spring's work
through the shared toadStool fungus.
