# hotSpring v0.6.28 → toadStool/barraCuda: Kokkos Parity + DF64 Transcendental Fix

**Date:** March 11, 2026
**From:** hotSpring v0.6.28 (847 lib tests, 0 failures, 0 clippy warnings)
**To:** toadStool / barraCuda teams
**Covers:** hotSpring Experiments 052-053 (multi-backend dispatch + live Kokkos parity)
**Supersedes:** `HOTSPRING_V0628_UPSTREAM_SYNC_HANDOFF_MAR10_2026.md` (archived), `HOTSPRING_V0626_TOADSTOOL_BARRACUDA_ABSORPTION_HANDOFF_MAR10_2026.md` (archived), `HOTSPRING_V0625_PRECISION_BRAIN_NVVM_POISONING_HANDOFF_MAR10_2026.md` (archived)
**License:** AGPL-3.0-only

---

## Executive Summary

- **Live Kokkos benchmark completed**: 9/9 PP Yukawa DSF cases at N=2000, barraCuda vs LAMMPS/Kokkos-CUDA. Average **12.4× gap** (212 steps/s vs 2,630 steps/s).
- **DF64 transcendental poisoning bug found and fixed**: `compile_shader_df64()` strips `exp_df64()`/`sqrt_df64()` on NVIDIA proprietary, leaving the Yukawa force shader producing zero forces. hotSpring now falls back to native f64 when `has_nvvm_df64_poisoning_risk()` is true.
- **Gap decomposition**: 12.4× is almost entirely the native f64 penalty (1:32 on Ampere). An NVVM-safe DF64 exponential would recover ~4-8× immediately.
- **Energy reducer bug confirmed**: `ReduceScalarPipeline::sum_f64()` returns zero despite correct particle dynamics (RDF, VACF, SSF all verified). Filed as upstream issue.
- **Multi-backend benchmark infrastructure**: `MdBenchmarkBackend` trait + `bench_md_parity` binary — reusable for future N-scaling and vendor-comparison studies.

---

## Part 1: DF64 Transcendental Poisoning — Root Cause & Fix

### What Happened

On RTX 3090 (Ampere, NVIDIA proprietary driver 580.119.02), the DF64 Yukawa force shader (`yukawa_force_df64.wgsl`) calls `exp_df64()` from `df64_transcendentals.wgsl`. barraCuda's `compile_shader_df64()` correctly strips these transcendentals to prevent NVVM device poisoning. However, the shader still references `exp_df64()` — it compiles via `wgpu` (which logs validation errors but doesn't fail pipeline creation), but produces zero output at dispatch time.

### Impact

All 9 benchmark cases showed `T*=0.000`, `KE=0.000`, `PE=0.000` before the fix. Particles were stationary — the force computation silently returned zeros. The `steps_per_sec` timer was still valid (wall-clock), masking the bug until observables (RDF, VACF, SSF) were checked.

### hotSpring Fix

```rust
let df64_transcendentals_unsafe = gpu.driver_profile().has_nvvm_df64_poisoning_risk();
let use_df64_force = matches!(strategy, Fp64Strategy::Hybrid) && !df64_transcendentals_unsafe;

let force_pipeline = if use_df64_force {
    gpu.create_pipeline_df64(shaders::SHADER_YUKAWA_FORCE_DF64, "yukawa_force_df64")
} else {
    gpu.create_pipeline_f64(shaders::SHADER_YUKAWA_FORCE, "yukawa_force_f64")
};
```

Applied in `simulation/mod.rs`, `simulation/verlet.rs`, and `celllist.rs`.

### toadStool action: NVVM-safe DF64 exponential

The fix above is a workaround — it falls back to native f64, paying 1:32 throughput on Ampere. The real fix is an NVVM-safe `exp_df64()` implementation that avoids the transcendental patterns NVVM poisons:

- **Taylor series** for exp() in range-reduced DF64
- **Range reduction** using integer bit manipulation (no `frexp`/`ldexp`)
- **Polynomial approximation** (Cody-Waite or similar) that stays within DF64 add/mul/sub

This would recover ~4-8× throughput on all NVIDIA proprietary consumer GPUs.

---

## Part 2: Live Kokkos Parity Results

### Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen Threadripper 3970X (32-core) |
| GPU | NVIDIA GeForce RTX 3090 (Ampere, sm_86) |
| RAM | 251 GB |
| Driver | NVIDIA proprietary 580.119.02 |

### Gap Table

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

**Average gap: 12.4×**. Kokkos compiled for sm_70 (not sm_86).

### Gap Decomposition

| Factor | Expected Impact |
|--------|----------------|
| Native f64 fallback (1:32 on Ampere) | ~16× penalty |
| Fix DF64 exp (safe Taylor series) | Recovers **4-8×** |
| Shared-memory tiled force | Recovers **1.5-2×** |
| Kernel fusion (VV + force) | Recovers **~1.2×** |

After fixing DF64 exp + tiling + fusion, expected gap: **~1-2×**.

---

## Part 3: Energy Reducer Bug

### Symptom

`ReduceScalarPipeline::sum_f64()` returns `0.0` for both `ke_buf` and `pe_buf` despite correct particle dynamics. Verified by:
- RDF peak at correct position (Γ-dependent)
- VACF exponential decay (correct timescale)
- SSF peak structure at expected q values

The `steps_per_sec` metric is wall-clock and unaffected.

### Reproduction

```bash
cd hotSpring/barracuda
cargo run --release --features gpu --bin sarkas_gpu -- --quick
```

Output shows `T*=0.000000`, `KE=0.0000` on both RTX 3090 and Titan V (NVK), while RDF/VACF/SSF are physically correct.

### toadStool action: ReduceScalarPipeline investigation

The `sum_f64()` path likely has a buffer mapping or workgroup sizing issue specific to f64 values. The reduction shader itself may work correctly at small sizes but fail for the buffer dimensions used in MD (N=2000 particles × 3 components). Worth checking:
1. Buffer alignment for f64 (8-byte)
2. Workgroup dispatch count for large buffers
3. Read-back mapping after reduction dispatch

---

## Part 4: Multi-Backend Infrastructure

### New Trait: `MdBenchmarkBackend`

```rust
pub trait MdBenchmarkBackend {
    fn name(&self) -> &str;
    fn run(&self, spec: &MdBenchmarkSpec) -> Result<MdBenchmarkResult, String>;
}
```

Two implementations:
- `BarraCudaMdBackend` — wgpu/Vulkan GPU dispatch
- `KokkosLammpsBackend` — spawns external `lmp` process with auto-generated input

### Binary: `bench_md_parity`

Runs all 9 PP Yukawa DSF cases across available backends, outputs comparison table + JSON.

### Files

| File | Purpose |
|------|---------|
| `barracuda/src/bench/md_backend.rs` | Trait + impls + LAMMPS input gen |
| `barracuda/src/bin/bench_md_parity.rs` | Benchmark orchestrator binary |
| `experiments/053_benchmark_results.json` | Raw results (18 records) |
| `specs/MULTI_BACKEND_DISPATCH.md` | Three-tier dispatch architecture |

---

## Part 5: Absorption Candidates

### Priority 1 — NVVM-safe DF64 exponential

**What:** A `exp_df64()` implementation in `df64_transcendentals.wgsl` that does not trigger NVVM device poisoning on NVIDIA proprietary drivers.

**Why:** Single biggest performance unlock for all springs using f64 MD or similar force-like computations on consumer Ampere/Ada GPUs.

**Approach:** Taylor series with Cody-Waite range reduction. The DF64 add/mul/sub primitives are safe — only the transcendental patterns (involving intermediate f64→f32 splits in certain sequences) trigger poisoning.

### Priority 2 — ReduceScalarPipeline f64 fix

**What:** Fix `sum_f64()` to correctly reduce f64 buffers >1000 elements.

**Why:** Energy reporting is broken for all MD simulations. Physics is correct but the diagnostic output is misleading.

### Priority 3 — Shared-memory tiled force kernel

**What:** Tiled all-pairs force computation with shared memory (workgroup-level particle caching).

**Why:** The current kernel reads all N particles from global memory per thread. Tiling would reduce global memory bandwidth by ~32× for typical workgroup sizes.

---

## Metrics

| Metric | Value |
|--------|-------|
| hotSpring version | v0.6.28 |
| barraCuda pin | `a012076` (v0.3.4) |
| toadStool | S145 |
| coralReef | Iter 30 |
| Lib tests | 847 |
| Clippy warnings | 0 |
| Experiments | 53 |
| Kokkos gap (avg) | 12.4× |
| Primary blocker | DF64 exp_df64 NVVM-safe path |
