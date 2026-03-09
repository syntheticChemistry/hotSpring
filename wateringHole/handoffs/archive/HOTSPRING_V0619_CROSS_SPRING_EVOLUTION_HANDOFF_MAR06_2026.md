# hotSpring v0.6.19 — Cross-Spring Evolution & Modern Rewire Handoff

**Date:** 2026-03-06 (updated)
**From:** hotSpring v0.6.19 (724 lib tests, 19 integration tests, 0 clippy warnings)
**To:** toadStool (S96) / barraCuda (v0.3.3) / coralReef (Phase 5)
**License:** AGPL-3.0-only

## Executive Summary

Complete rewire of hotSpring to modern barraCuda v0.3.3. All shader compilation
now delegates to barraCuda's sovereign pipeline. Cross-spring evolved GPU ops
(autocorrelation, mean+variance, correlation, chi-squared) adopted and benchmarked.

**All three Chuna papers (43-45) now CPU-complete** with Python controls + BarraCuda
CPU validation. BarraCuda CPU covers 25/25 actionable papers. Python controls: 21/25.

**Math is universal. Precision is silicon. barraCuda owns both.**

## Part I: Shader Compilation Rewire

### DF64 Precision Delegation (gpu/mod.rs)

| Before | After |
|--------|-------|
| Local `compile_full_df64_pipeline()` — 30 lines, manual `downcast_f64_to_df64` + `strip_f64_from_df64_core` + WGSL concatenation | `wgpu_device.compile_shader_universal(source, Precision::Df64, label)` — barraCuda's two-layer naga+text rewrite |
| Local `create_pipeline_df64()` — manual `WGSL_DF64_CORE` + `WGSL_DF64_TRANSCENDENTALS` import from `ops::lattice::su3` | `wgpu_device.compile_shader_df64(source, label)` — barraCuda prepends core library |
| `strip_f64_from_df64_core()` helper — 26 lines | Removed |

### HFB GPU-Resident (physics/hfb_gpu_resident/mod.rs)

5 shader compilation sites rewired from raw `create_shader_module()` to `device.compile_shader_f64()`:

| Shader | Before | After | Benefit |
|--------|--------|-------|---------|
| Potentials | `ShaderTemplate::for_device_auto` + `create_shader_module` | `compile_shader_f64` | Sovereign SPIR-V, NVK polyfills |
| Hamiltonian | Raw `create_shader_module` (NO patching!) | `compile_shader_f64` | **Fixed NVK blindspot** |
| Density | Raw `create_shader_module` (NO patching!) | `compile_shader_f64` | **Fixed NVK blindspot** |
| Energy | `ShaderTemplate::for_device_auto` + `create_shader_module` | `compile_shader_f64` | Sovereign SPIR-V |
| SO pack | Raw `create_shader_module` (NO patching!) | `compile_shader_f64` | **Fixed NVK blindspot** |

### BCS GPU Bisection (physics/bcs_gpu.rs)

| Before | After |
|--------|-------|
| `ShaderTemplate::for_device_auto` + `create_shader_module` | `gpu.to_wgpu_device().compile_shader_f64()` |

### Workgroup Size Constants

| File | Before | After |
|------|--------|-------|
| `md/observables/transport_gpu.rs` | `const WORKGROUP_SIZE: u32 = 64` | `use barracuda::device::capabilities::WORKGROUP_SIZE_COMPACT` |
| `tolerances/md.rs` | `pub const MD_WORKGROUP_SIZE: usize = 64` | `barracuda::device::capabilities::WORKGROUP_SIZE_COMPACT as usize` |

## Part II: New barraCuda API Adoption

### GPU Autocorrelation (transport.rs)

New `compute_stress_acf_gpu()` using barraCuda's `AutocorrelationF64` — single-dispatch
GPU autocorrelation C(lag) for all lags, then Green-Kubo integration on host.

### Cross-Spring Evolution Benchmark (bench_cross_spring_evolution.rs)

Evolved from toadStool S80 to barraCuda v0.3.3 / S96. Added 4 new GPU benchmarks:

| Benchmark | barraCuda API | Origin | Cross-spring benefit |
|-----------|--------------|--------|---------------------|
| Autocorrelation GPU | `AutocorrelationF64` | hotSpring VACF + wetSpring time-series | All springs: spectral analysis |
| Mean+Variance GPU | `VarianceF64` | Kokkos/hotSpring Welford | All springs: observable stats |
| Correlation GPU | `CorrelationF64` | Kokkos/wetSpring 5-accum | groundSpring, neuralSpring, hotSpring |
| Chi-squared GPU | `FusedChiSquaredGpu` | groundSpring V74 | hotSpring nuclear χ², wetSpring enrichment |

## Part III: Cross-Spring Shader Evolution Map

```text
hotSpring (nuclear physics)
  → df64_core.wgsl (S58) → ALL springs get DF64 precision
  → sum_reduce_f64.wgsl (S46) → foundation for all GPU stats
  → spectral/anderson.rs (S26) → groundSpring localization validation
  → CG solver shaders (S46-48) → iterative GPU pattern used by neuralSpring
  → vacf_dot_f64.wgsl → barraCuda autocorrelation_f64 design
  → nuclear shaders (7) → barraCuda ops/nuclear/

wetSpring (metagenomics)
  → bray_curtis_f64.wgsl → airSpring sensor similarity
  → HMM forward/backward → neuralSpring log-domain batched inference
  → Shannon/Simpson → barraCuda fused_map_reduce pattern
  → correlation_full_f64.wgsl → groundSpring, neuralSpring stats

neuralSpring (ML/agents)
  → linear_regression_f64.wgsl → airSpring trend analysis
  → matrix_correlation_f64.wgsl → groundSpring multi-variate validation
  → fused_chi_squared_f64.wgsl → hotSpring nuclear χ² fits
  → batch_ipr_f64.wgsl → hotSpring spectral diagnostics

groundSpring (noise validation)
  → chi_squared_f64.wgsl (CDF+quantile) → all springs: statistical tests
  → rawr_weighted_mean_f64.wgsl → wetSpring bootstrap CI
  → anderson_lyapunov_f64.wgsl → hotSpring transfer-matrix spectral
  → 13-tier tolerance architecture → all springs: validation framework

airSpring (agriculture/hydro)
  → batched_elementwise_f64.wgsl → all springs: FAO-56 pattern for batch ops
  → seasonal_pipeline.wgsl → wetSpring environmental monitoring
  → moving_window_f64.wgsl → neuralSpring streaming inference
```

## Part IV: Chuna Papers — Complete CPU Validation

All three Thomas Chuna papers now have Python controls + BarraCuda CPU validation:

| Paper | Topic | Python | CPU | GPU | Speedup |
|-------|-------|:---:|:---:|:---:|:---:|
| 43 | SU(3) gradient flow | ✅ 11/11 | ✅ 14/14 | ✅ 7/7 | **38.5×** |
| 44 | BGK dielectric | ✅ 19/19 | ✅ 13+21/21 | ✅ 12/12 | 144× (CPU vs Py) |
| 45 | Kinetic-fluid | ✅ 18/18 | ✅ 16+20/20 | — | 322× (CPU vs Py) |

New modules: `physics/dielectric.rs`, `physics/gpu_dielectric.rs`, `physics/kinetic_fluid.rs`, `lattice/gpu_flow.rs`
New shaders: `su3_flow_accumulate_f64.wgsl`, `dielectric_mermin_f64.wgsl`
New binaries: `validate_dielectric`, `validate_kinetic_fluid`, `validate_gpu_gradient_flow`, `validate_gpu_dielectric`

**Paper 43 GPU**: 1 new shader, 38.5× speedup. **Paper 44 GPU**: batched Mermin with
numerically stable W(z) asymptotic (100% DSF positivity, 100% passive-medium compliance).

## Part V: Verification

| Check | Result |
|-------|--------|
| `cargo fmt --check` | Clean |
| `cargo clippy --all-targets --all-features` | 0 warnings on new code |
| `cargo test --lib` | 719 passed, 0 failed, 6 ignored |
| `cargo test --tests` | 19 passed, 0 failed |
| `cargo doc --no-deps` | Clean (95 files) |
| Max file size | 945 lines (< 1000 limit) |
| Unsafe blocks | 0 |

## Part VI: What Remains

| Item | Priority | Notes |
|------|----------|-------|
| Papers 44-45 Python vs Rust benchmark | P0 | Get speedup numbers for Chuna package |
| GPU promotion: Papers 43-45 | P1 | SU(3) GPU infra exists; dielectric + kinetic need new shaders |
| `GpuView<T>` adoption for MD | P1 | Eliminate per-call buffer upload/download |
| Buffer-resident CG unification | P1 | `ReduceScalarPipeline::encode_reduce_to_buffer()` |
| Edition 2024 migration | P2 | Align with barraCuda |
| coralReef sovereign pipeline | P3 | Blocked on coralDriver |
| `SubstrateCapabilityKind` dispatch | P2 | Use for hardware-aware routing |
| Deformed HFB full wiring | P2 | 5 WGSL shaders waiting for pipeline |

---
*hotSpring v0.6.19 — AGPL-3.0-only*
*Math is universal. Precision is silicon. The sovereign compiler decides.*
