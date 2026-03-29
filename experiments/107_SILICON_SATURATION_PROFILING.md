# Experiment 107: Silicon Saturation Profiling

**Date:** March 28-29, 2026
**Machine:** strandgate (RTX 3090 + RX 6950 XT)
**Status:** Complete — 7 phases
**License:** AGPL-3.0-only

---

## Objective

Profile and saturate all accessible silicon units on both GPUs. Wire TMU, subgroup,
and ROP hardware into the production RHMC pipeline. Extend NPU observation with
silicon routing metadata. Determine maximum physics capacity per card.

## Phases

### Phase 1: Refresh Silicon Profiles
Re-ran `bench_silicon_profile`, `bench_silicon_saturation`, `bench_fp64_ratio` on
both GPUs. Confirmed RTX 3090 TMU advantage (1.89x) and AMD DF64 advantage (38%).

### Phase 2: Full-Trajectory Benchmark
Created `bench_full_trajectory_silicon` for end-to-end RHMC trajectory comparison
across lattice sizes on both cards. Revealed RTX 3090 3.79x unidirectional speedup,
RX 6950 XT 2.06x speedup.

### Phase 3: TMU PRNG (Tier 0)
Wired Box-Muller PRNG via TMU `textureLoad` into production RHMC.
- New shader: `su3_random_momenta_tmu_f64.wgsl`
- New module: `tmu_tables.rs` (GPU-resident log + trig lookup textures)
- Integration: `GpuHmcStreamingPipelines::new_with_tmu`

### Phase 4: Subgroup Reduce (Tier 4)
Implemented `subgroupAdd()` for CG dot product reduction.
- New shader: `sum_reduce_subgroup_f64.wgsl` (with `enable subgroups;`)
- Integration: conditional in `GpuResidentCgPipelines::new`
- Fallback: shared-memory reduce when `wgpu::Features::SUBGROUP` unavailable
- Fixed: `SUBGROUPS` → `SUBGROUP` (singular) in wgpu v28

### Phase 5: ROP Atomic Scatter-Add (Tier 3)
Implemented fixed-point i32 `atomicAdd` for parallel fermion force accumulation.
- New shader: `su3_fermion_force_accumulate_rop_f64.wgsl` (fused force + atomicAdd)
- New shader: `su3_force_atomic_to_momentum_f64.wgsl` (i32 → f64 conversion)
- New module: `rop_force_accum.rs` (pipeline management + buffer lifecycle)
- Scale: 2^20 (1,048,576), ~6 significant digits
- All poles dispatch simultaneously — no inter-pole barriers

### Phase 6: NPU Observation Extension
Extended `TrajectoryObservation` with `SiliconRoutingTags`.
- New struct: `SiliconRoutingTags` (tmu_prng, subgroup_reduce, rop_force_accum,
  fp64_strategy_id, has_native_f64)
- NPU input: `npu_canonical_input_v2` (6D → 11D)

### Phase 7: Capacity Analysis
Buffer accounting for max lattice under VRAM + per-allocation constraints.
- RTX 3090 (24 GB): L=46⁴ dynamical RHMC (23.6 GB), L=56⁴ quenched
- RX 6950 XT (16 GB): L=40⁴ dynamical RHMC (13.5 GB), L=42⁴ quenched
- Both fit 32⁴ (5.5 GB). Neither fits 48⁴ (28 GB)

## New Files

| File | Type | Lines |
|------|------|-------|
| `su3_random_momenta_tmu_f64.wgsl` | WGSL shader | ~50 |
| `sum_reduce_subgroup_f64.wgsl` | WGSL shader | ~40 |
| `su3_fermion_force_accumulate_rop_f64.wgsl` | WGSL shader | ~120 |
| `su3_force_atomic_to_momentum_f64.wgsl` | WGSL shader | ~25 |
| `tmu_tables.rs` | Rust module | ~80 |
| `rop_force_accum.rs` | Rust module | ~130 |
| `bench_full_trajectory_silicon.rs` | Rust binary | ~350 |

## Modified Files

| File | Change |
|------|--------|
| `gpu/mod.rs` | `SUBGROUP` feature, `has_subgroups` field |
| `streaming.rs` | TMU PRNG integration, `new_with_tmu` constructor |
| `resident_cg_pipelines.rs` | Conditional subgroup reduce shader |
| `unidirectional_rhmc.rs` | ROP atomic force path in `uni_total_force_dispatch` |
| `brain_rhmc.rs` | `SiliconRoutingTags`, `npu_canonical_input_v2` |
| `dynamical.rs` | `WGSL_RANDOM_MOMENTA_TMU` static |
| `mod.rs` (gpu_hmc) | `pub mod tmu_tables`, `pub mod rop_force_accum` |

## Handoff

`HOTSPRING_V0632_SILICON_SATURATION_PRIMAL_EVOLUTION_HANDOFF_MAR29_2026.md`

## Cross-References

- Exp 096-100: Silicon characterization pipeline (prerequisite)
- Exp 105: Silicon-routed QCD revalidation (prerequisite)
- `specs/SILICON_TIER_ROUTING.md`: 7-tier routing architecture
- `whitePaper/baseCamp/silicon_science.md`: Updated briefing
- `whitePaper/baseCamp/silicon_characterization_at_scale.md`: Updated briefing
