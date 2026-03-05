SPDX-License-Identifier: AGPL-3.0-only

# Cross-Spring Shader Evolution — hotSpring's View

**Date:** March 5, 2026
**Synced to:** barraCuda v0.3.3 (standalone primal, wgpu 28, naga 28)
**hotSpring:** v0.6.17 — 669 tests, 9/9 Yukawa validation, Nautilus unified brain
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
| hotSpring → toadStool → all | `level_spacing_ratio`, Anderson proxy | Born in QCD, used by bio (wetSpring) and RMT (neuralSpring) |
| neuralSpring → hotSpring | `batched_nelder_mead_gpu` | GPU batch optimization for HMC parameter tuning |
| hotSpring → toadStool → all | ESN reservoir → `MultiHeadEsn` GPU | CPU sidecar → first-class GPU citizen with per-head training |

### S78→S80 New Modules (March 2, 2026)

| Module | Origin Spring | toadStool Session | Notes |
|--------|---------------|-------------------|-------|
| `spectral/stats.rs` | hotSpring→toadStool | S78 | `spectral_bandwidth`, `spectral_condition_number`, `classify_spectral_phase` |
| `SpectralAnalysis` + Marchenko–Pastur | hotSpring→toadStool | S78 | RMT-based phase classifier (Bulk/EdgeOfChaos/Chaotic) |
| `NeighborMode::precompute_periodic_4d` | hotSpring→toadStool | S80 | 4D precomputed neighbor table (note: index convention differs) |
| `MultiHeadEsn` GPU | hotSpring→toadStool | S78 | Per-head training, replaces CPU ESN sidecar |
| `NautilusBrain` | hotSpring+neuralSpring | S79 | Evolutionary reservoir for QCD steering |
| `BatchedEncoder` | toadStool | S79 | Fuse multiple GPU dispatches into one submission |
| `batched_nelder_mead_gpu` | neuralSpring | S79 | Parallel optimization on GPU |
| `fused_mlp` | neuralSpring | S80 | Single-dispatch multi-layer perceptron |
| `StatefulPipeline` | toadStool | S79 | CPU pipeline with mutable state between stages |
| `driver_profile/` centralized | hotSpring→toadStool | S80 | Driver detection and workarounds |
| `asin_df64` iterative | hotSpring+wetSpring | S80 | Taylor-series asin for DF64 |

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
| `spectral/stats` (new S80) | `spectral_bandwidth`, `spectral_condition_number`, `classify_spectral_phase`, `SpectralAnalysis` | **S78→S80** |
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
| VACF GPU | `barracuda::ops::md::compute_vacf_batch` | S78 |
| Stress virial GPU | `barracuda::ops::md::compute_stress_virial` | S78 |
| Matrix correlation GPU | `barracuda::ops::stats_f64::matrix_correlation` | S78 |
| **NeighborMode 4D** (tested) | `barracuda::ops::lattice::NeighborMode::precompute_periodic_4d` | **S80** |
| **Nelder-Mead GPU** (benchmarked) | `barracuda::optimize::batched_nelder_mead_gpu` | **S79** |
| **MultiHeadEsn** (serde-compatible) | `barracuda::esn_v2::MultiHeadEsn` | **S78** |

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

844+ WGSL shaders now live in barracuda (toadStool S80). hotSpring contributed
~100 of those, primarily precision-critical lattice QCD, nuclear physics, and
MD transport shaders. The cross-spring model works: hotSpring's `math_f64`
preamble is used by wetSpring's bio shaders; neuralSpring's `rk4_parallel`
improves hotSpring's ODE paths; airSpring's `moving_window` is available for
streaming diagnostics. Every spring benefits from every other spring's work
through the shared toadStool fungus.

S80 sync adds three new cross-spring pathways:
1. **Spectral stats** (`spectral_bandwidth`, `spectral_condition_number`,
   `SpectralAnalysis`) — born in hotSpring's Anderson proxy, now available to
   all springs for RMT-based diagnostics
2. **MultiHeadEsn** — hotSpring's 36-head CPU ESN layout is serde-compatible
   with toadStool's GPU `MultiHeadEsn`, enabling future migration to GPU-backed
   per-head training
3. **NeighborMode 4D** — toadStool's precomputed 4D neighbor table mirrors
   hotSpring's `build_neighbors` design (same purpose, different index convention:
   hotSpring uses z-fastest, toadStool uses x-fastest)

Index convention note: hotSpring `site_index = t*Nx*Ny*Nz + x*Ny*Nz + y*Nz + z`
(z fastest), toadStool `site_index = t*Nz*Ny*Nx + z*Ny*Nx + y*Nx + x` (x fastest).
Both precompute 8 neighbors per site `[+x,-x,+y,-y,+z,-z,+t,-t]`. Unification
is tracked as a future handoff — hotSpring's shaders assume z-fastest throughout.

---

## barraCuda v0.3.3 Sync (March 4, 2026)

### wgpu 22 → 28 Migration

The full catch-up to barraCuda v0.3.3 required migrating hotSpring from wgpu 22
to wgpu 28 — a breaking API change across ~12 files and ~40 call sites:

| Change | Old (wgpu 22) | New (wgpu 28) | Files |
|--------|---------------|---------------|-------|
| Device/Queue wrapping | `Arc<wgpu::Device>`, `Arc<wgpu::Queue>` | `device.clone()`, `queue.clone()` (Device is Clone) | `gpu/mod.rs` |
| `WgpuDevice::from_existing` | `from_existing(Arc::new(d), Arc::new(q), info)` | `from_existing(d, q, info)` | `gpu/mod.rs` (×2) |
| Adapter enumeration | Sync `instance.enumerate_adapters()` | Async `pollster::block_on(instance.enumerate_adapters())` | `gpu/adapter.rs` (×5) |
| Instance creation | `Instance::new(desc)` | `Instance::new(&desc)` | `gpu/mod.rs`, `adapter.rs`, `forge/probe.rs` |
| Device polling | `device.poll(Maintain::Wait)` | `device.poll(PollType::Wait { submission_index: None, timeout: None })` | 9 files |
| Error scopes | `device.push_error_scope(); device.pop_error_scope()` | `let scope = device.push_error_scope(); scope.pop()` | `gpu/mod.rs`, `f64_builtin_test.rs` |
| Pipeline layout | `push_constant_ranges: &[]` | `immediate_size: 0` | 4 files |
| Entry point | `entry_point: "main"` | `entry_point: Some("main")` | 6 files |
| Device descriptor | 4 fields | +`experimental_features`, +`trace` | `gpu/mod.rs` (×2) |
| `request_device` | 2 args (desc, trace) | 1 arg (desc only) | `gpu/mod.rs` (×2) |
| Buffer limits | Hardcoded 2 GB | `adapter_limits.min(2 GB)` (wgpu 28 strict validation) | `gpu/mod.rs` (×2) |
| SPIRV passthrough | `Features::SPIRV_SHADER_PASSTHROUGH` constant | Removed — `has_spirv_passthrough()` method | `gpu/mod.rs` |

### NVK Sovereign SPIR-V Discovery

**Cross-spring evolution from hotSpring → barraCuda**: hotSpring's NVK testing
discovered that barraCuda v0.3.3's sovereign SPIR-V passthrough produces invalid
shader modules on NVK. The fix was applied to barraCuda itself:

```rust
// barraCuda/src/device/wgpu_device/mod.rs — has_spirv_passthrough()
// Now excludes NVK: NAK compiler produces invalid modules from sovereign SPIR-V.
pub fn has_spirv_passthrough(&self) -> bool {
    if self.adapter_info.backend != wgpu::Backend::Vulkan { return false; }
    let d = self.adapter_info.driver.to_lowercase();
    !(d.contains("nvk") || d.contains("nouveau"))
}
```

Impact: Pipeline compilation 280× faster on NVK (664ms → 2.3ms for
`hmc_link_update`) because sovereign is skipped entirely. CG alpha/beta shaders
that previously failed validation now compile cleanly.

### Volta/Datacenter Native f64 Path

**Cross-spring evolution**: hotSpring's multi-GPU testing (RTX 3090 + Titan V,
both on NVK) revealed that `use_df64_compute = is_nvk` was wrong for Volta GPUs.
The Titan V has 1:2 f64 throughput — DF64-compute loses 5 bits of mantissa for
no throughput benefit.

Fix: hotSpring now classifies GPU f64 throughput and routes Volta/datacenter
adapters (Titan V, V100, A100, H100) to native f64 even on NVK. Consumer GPUs
(Ampere/Ada, 1:64 ratio) still use DF64-compute.

Result: Titan V achieves 100% HMC acceptance, P=0.584 (physically correct) with
native f64 on NVK.

### New Shaders from barraCuda v0.3.3

12 evolved lattice shaders copied from barraCuda to hotSpring:

| Shader | Purpose | Evolution |
|--------|---------|-----------|
| `cg_kernels_f64.wgsl` | Unified CG solver kernels | barraCuda consolidated from hotSpring's split CG |
| `hmc_leapfrog_f64.wgsl` | Leapfrog integrator (single-dispatch) | barraCuda fused from hotSpring's multi-pass |
| `kinetic_energy_f64.wgsl` | Standard KE | barraCuda universal math |
| `kinetic_energy_df64.wgsl` | DF64 KE | barraCuda precision specialization |
| `lattice_init_f64.wgsl` | Hot/cold lattice initialization | New in barraCuda |
| `pseudofermion_heatbath_f64.wgsl` | Gaussian pseudofermion heatbath | barraCuda absorbed from hotSpring |
| `pseudofermion_force_f64.wgsl` | Pseudofermion force | barraCuda absorbed from hotSpring |
| `su3_hmc_force_f64.wgsl` | SU(3) HMC force (native) | barraCuda refactored |
| `su3_hmc_force_df64.wgsl` | SU(3) HMC force (DF64) | barraCuda precision specialization |
| `wilson_action_f64.wgsl` | Wilson action | barraCuda universal math |
| `wilson_action_df64.wgsl` | Wilson action (DF64) | barraCuda precision specialization |
| `higgs_u1_hmc_f64.wgsl` | Abelian Higgs U(1) HMC | barraCuda absorbed from hotSpring Paper 13 |

### PRNG Division → Multiplication Fix

barraCuda v0.3.3 changed `/ f64(4294967296.0)` to `* f64(2.3283064365386963e-10)`
in PRNG shaders. hotSpring already had this fix (cross-spring evolution
propagated the optimization before barraCuda formalized it).

### Cross-Spring Evolution Timeline (Feb–Mar 2026)

| Date | Direction | What | Impact |
|------|-----------|------|--------|
| Feb 16 | hotSpring → toadStool | Lattice QCD shaders, complex_f64, su3 | Foundation for all lattice GPU ops |
| Feb 21 | hotSpring → toadStool | ESN reservoir/readout shaders | GPU ESN available to wetSpring |
| Feb 26 | hotSpring → toadStool | DF64 core streaming discovery | Universal precision for all springs |
| Feb 28 | toadStool → barraCuda | Primal budding: barraCuda splits from toadStool | Math/shaders/compilation independent |
| Mar 2 | hotSpring ← barraCuda | v0.2.0 sync, S80 modules | Spectral stats, MultiHeadEsn, 4D neighbors |
| Mar 4 | hotSpring ← barraCuda | **v0.3.3 full sync: wgpu 28, naga 28** | 12 new shaders, 280× faster pipeline compilation |
| Mar 4 | hotSpring → barraCuda | **NVK SPIR-V exclusion** | All NVK users benefit from the fix |
| Mar 4 | hotSpring | **Volta native f64 on NVK** | Titan V: 100% acceptance with native f64 |
| Mar 5 | hotSpring | **Nautilus unification** | ESN readout merged into Nautilus evolutionary reservoir |
| Mar 5 | hotSpring | **tol::/eps:: tolerance adoption** | All bare literals replaced with named constants |
| Mar 5 | hotSpring | **9/9 validation sweep** | All PP Yukawa DSF cases pass (0.001% drift), Verlet + DF64 |
| Mar 5 | hotSpring ← barraCuda | **DF64 naga rewriter fix** | Compound assignments + let bindings handled correctly |
| Mar 5 | hotSpring ← barraCuda | **Fused stats ops** | Welford mean+variance, Pearson correlation (CPU post-proc too small to benefit now) |
| Mar 5 | hotSpring ← toadStool S94b | **NpuDispatch + NpuParameterController** | Vendor-agnostic NPU interface (future wiring) |
| Mar 5 | hotSpring | **coralNAK awareness** | Sovereign Rust shader compiler cloned, not yet integrated |

### Current Benchmark Results (RTX 3090, DF64, N=2000, March 5 2026)

#### Without brain (raw physics, from Verlet handoff)

| Case | Algorithm | Steps/s | Gap vs Kokkos-CUDA |
|------|-----------|---------|:---:|
| k1_G14 | AllPairs | 181 | 4.0× |
| k2_G31 | Verlet | 368 | 3.0× |
| k2_G158 | Verlet | 846 | 3.6× |
| k3_G100 | Verlet | 977 | 3.2× |
| k3_G1510 | Verlet | 992 | 3.7× |

#### With Nautilus brain (full 9-case sweep, March 5)

| Case | Algorithm | Steps/s | Energy Drift | Brain Status |
|------|-----------|---------|:---:|:---:|
| k1_G14 | AllPairs | 292 | 0.001% | 12R, 0/12 trusted |
| k1_G72 | AllPairs | 307 | 0.001% | learning |
| k1_G217 | AllPairs | 329 | 0.001% | learning |
| k2_G31 | Verlet | 306 | 0.001% | learning |
| k2_G158 | Verlet | 303 | 0.001% | learning |
| k2_G476 | Verlet | 303 | 0.001% | learning |
| k3_G100 | Verlet | 314 | 0.001% | learning |
| k3_G503 | Verlet | 313 | 0.001% | learning |
| k3_G1510 | Verlet | 317 | 0.001% | learning |

Brain overhead: ~3× (Nautilus observation + readout retraining + board evolution).
This is a feature, not overhead — the brain is learning from physics data.
Kokkos has no equivalent intelligence layer.

### Remaining DF64 Precision Frontier

The RTX 3090 (NVK, DF64-compute) achieves correct quenched thermalization
(P=0.557 at β=6.0) but produces NaN ΔH in measurement trajectories. The 48-bit
mantissa of DF64 (vs 53-bit for native f64) causes the Metropolis ΔH test to
fail when comparing two large Hamiltonians that differ by O(1). This is a
fundamental precision boundary: DF64 is sufficient for local observables
(plaquette, force) but insufficient for global energy-difference tests.

Paths forward:
1. **Mixed-precision HMC**: Force/plaquette via DF64, kinetic energy + action
   accumulation via native f64 (Titan V as oracle)
2. **Stochastic ΔH**: Accept/reject based on estimated ΔH variance
3. **Proprietary driver**: NVIDIA proprietary Vulkan driver supports full f64
   on RTX 3090 (1:64 ratio but correct arithmetic)
