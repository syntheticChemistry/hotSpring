# Silicon Tier Routing Architecture

**Status**: Active — Exp 106
**Version**: hotSpring v0.6.32+
**License**: AGPL-3.0-only

## Principle

Traditional HPC sends everything to FP64 ALU and leaves the rest of the
GPU idle. ecoPrimals inverts this: fill every alternative silicon unit
first, reserve FP64 for precision-critical work last. Consumer GPUs have
more diverse silicon than HPC cards (TMU, ROP, tensor cores, RT cores) —
constraints guide us to use what we have.

## The 7-Tier Hierarchy

```
TIER 0  TMU           Lookup tables, PRNG transcendentals, stencil cache
TIER 1  Tensor Cores  SU(3) matmul, preconditioner (NVIDIA only, via coralReef SASS)
TIER 2  FP32 ALU      DF64 Dekker pairs — bulk compute, 48-bit mantissa
TIER 3  ROP / Atomics Scatter-add for force accumulation (AMD 7.4x advantage)
TIER 4  Subgroup      Warp/wavefront intrinsics for reductions (no shared mem)
TIER 5  Shared Memory Workgroup-level communication, halo exchange
TIER 6  FP64 ALU      LAST — Metropolis test, observable accumulation only
```

The tier number represents routing priority: lower tiers are cheaper silicon
that should be filled first. When a kernel's shader path exists for a lower
tier, it routes there. When not, it falls through to the next available tier.

## Data Model

The tier system is implemented in `barracuda/src/bench/silicon_profile.rs`:

- **`SiliconUnit`** — enum of 9 distinct functional units
- **`UnitThroughput`** — theoretical peak + measured peak + efficiency per unit
- **`SiliconProfile`** — full card personality (serde-serializable, saved to `profiles/silicon/`)
- **`QcdKernel`** — enum of 10 QCD workload phases
- **`TierRoute`** — preferred silicon ordering per kernel
- **`CompositionEntry`** — measured parallel speedup for unit pairs

Profiles persist as JSON in `profiles/silicon/<adapter_name>.json` and are
rebuilt by `cargo run --release --bin bench_silicon_profile`.

## QCD Kernel Routing Table

| Kernel | Tier 0 (preferred) | Tier 1 | Tier 2 | Tier 3+ (fallback) | Rationale |
|--------|-------------------|--------|--------|--------------------|----|
| PRNG (Box-Muller) | **TMU** | — | FP32 ALU | — | exp()/cos()/sin() → texture lookup frees ALU |
| Gauge Force | — | **Tensor** | FP32 ALU | FP64 ALU | SU(3)×SU(3) = 3×3 complex matmul → MMA tiles |
| Dirac Operator | **TMU** | — | FP32 ALU | FP64 ALU | 8-neighbor stencil → texture cache for link loads |
| CG Dot Product | — | — | — | **Subgroup** → Shared Mem → FP32 | shuffle-reduce without shared memory |
| CG axpy | — | — | — | **Mem BW** → FP32 ALU | Pure bandwidth: a*x+y streams |
| Force Accumulation | — | — | — | **ROP** → Shared Mem → FP32 | 8-neighbor scatter-add → atomicAdd |
| Metropolis ΔH | — | — | — | **FP64 ALU** → FP32 ALU | Single scalar, needs full precision |
| Observable Accum. | — | — | — | **FP64 ALU** → FP32 ALU | Plaquette/Polyakov precision accumulation |
| Link Update | — | — | **FP32 ALU** | Tensor | SU(3) exp(iH·dt) via Cayley-Hamilton |
| Gradient Flow | — | — | **FP32 ALU** | TMU → FP64 ALU | Link smearing = force + stencil pattern |

## Measured Hardware Profiles

### RTX 3090 (GA102, Ampere) — `profiles/silicon/nvidia_geforce_rtx_3090.json`

| Unit | Theoretical | Measured | Efficiency | Unit |
|------|------------|---------|-----------|------|
| FP32 ALU | 35.60 | 8.53 | 24.0% | TFLOPS |
| FP64 ALU | 0.56 | — | — | TFLOPS |
| DF64 (Dekker) | — | 18.13 | — | TFLOPS |
| TMU | 557.6 | 364.2 | 65.3% | GT/s |
| ROP | 190.4 | 16.0 | 8.4% | GP/s |
| Tensor (TF32) | 71.0 | — | — | TFLOPS |
| Memory BW | 936.0 | (cache) | — | GB/s |

Composition: ALU + TMU = **2.80x** multiplier (21.0ms serial → 7.5ms compound)

### RX 6950 XT (Navi 21, RDNA 2) — `profiles/silicon/amd_radeon_rx_6950_xt__radv_navi21_.json`

| Unit | Theoretical | Measured | Efficiency | Unit |
|------|------------|---------|-----------|------|
| FP32 ALU | 23.65 | 9.68 | 40.9% | TFLOPS |
| FP64 ALU | 1.48 | — | — | TFLOPS |
| DF64 (Dekker) | — | 24.07 | — | TFLOPS |
| TMU | 739.2 | 351.2 | 47.5% | GT/s |
| ROP | 295.7 | 117.7 | 39.8% | GP/s |
| Tensor | — | — | — | — |
| Memory BW | 576.0 | (cache) | — | GB/s |
| Infinity Cache | 128 MB | — | — | — |

Composition: ALU + TMU = **1.95x** multiplier (16.9ms serial → 8.7ms compound)

### Cross-GPU Summary

| Metric | RTX 3090 | RX 6950 XT | Advantage |
|--------|----------|------------|-----------|
| FP32 ALU (measured) | 8.5 TFLOPS | 9.7 TFLOPS | AMD |
| DF64 (measured) | 18.1 TFLOPS | 24.1 TFLOPS | AMD |
| TMU (measured) | 364 GT/s | 351 GT/s | ~parity |
| ROP/Atomics | 16.0 Gatom/s | 117.7 Gatom/s | **AMD 7.4x** |
| Composition mult. | 2.80x | 1.95x | NVIDIA |
| Subgroup size | 32 (warp) | 64 (wavefront) | — |

## Implementation Status

| Tier | Silicon Unit | Production Shader Path | Status |
|------|-------------|----------------------|--------|
| 0 | TMU | `bench_qcd_silicon_routing.rs` PRNG kernel | **BENCH ONLY** — not wired into RHMC |
| 0 | TMU stencil | `bench_qcd_silicon_routing.rs` stencil kernel | **BENCH ONLY** |
| 1 | Tensor Cores | — | **PLAN** — requires coralReef SASS MMA emission |
| 2 | FP32 ALU (DF64) | All production RHMC shaders | **LIVE** — default path |
| 3 | ROP atomics | — | **PLAN** — scatter-add force accumulation |
| 4 | Subgroup | — | **PLAN** — shuffle-reduce for CG dots |
| 5 | Shared Memory | CG tree reduction (existing) | **LIVE** — in resident CG reduce chain |
| 6 | FP64 ALU | — | **PLAN** — Metropolis/observable via native f64 |

## coralReef Integration Points

coralReef (external primal, `../../../coralReef/`) provides:

1. **WGSL → SASS compilation** — SM70 (Volta), SM86 (Ampere), SM120 (Blackwell)
2. **`FmaPolicy::Separate`** — FMA-free precision for F64Precise tier
3. **`CoralReefDevice`** — sovereign dispatch bypassing Vulkan
4. **AMD GCN5 E2E** — full dispatch working

What coralReef unlocks for silicon routing:

- **Tensor cores** — SASS MMA/WMMA/DMMA instructions for SU(3) matmul (Tier 1)
- **Subgroup intrinsics** — native `shfl.sync` for warp-reduce (Tier 4)
- **Native FP64** — direct SASS emission without NVVM transcendental risk (Tier 6)
- **Instruction scheduling** — manually ordered SASS to overlap ALU + TMU + memory

The `sovereign-dispatch` feature flag in `barracuda/Cargo.toml` enables these paths.
Compilation validates via `validate_sovereign_compile` binary (45/46 shaders → SASS).

## Bottleneck Roadmap: Path to 100% GPU

Three of five bottlenecks have been resolved. Two remain:

### B1: CPU Momenta Generation — ✅ RESOLVED (GPU PRNG)

**Status**: Already handled on GPU via `momenta_prng_pipeline` (WGSL shader)
in `unidirectional_rhmc.rs`. No CPU momenta generation in the hot path.

### B2: CPU Hamiltonian Assembly — ✅ RESOLVED (March 2026)

**Implemented**: Three WGSL kernels eliminate all CPU-side H assembly:
- `hamiltonian_assembly_f64.wgsl` — H = β(6V - plaq) + T + S_f from GPU scalars
- `fermion_action_sum_f64.wgsl` — S_f = α₀⟨φ|φ⟩ + Σαₛ⟨φ|xₛ⟩ on GPU, accumulates
- `compute_h_gpu()` orchestrates gauge+KE reduces + CG + fermion dots + H assembly
  with zero scalar readbacks for H_old or H_new.
**Result**: Eliminated 2×(1 + n_sectors) sync points per trajectory.

### B3: CPU Metropolis Test — ✅ RESOLVED (March 2026)

**Implemented**: `metropolis_f64.wgsl` — single GPU kernel reads H_old, H_new,
computes delta_H, accept/reject, and writes 7 f64 diagnostics. CPU generates
uniform random (single LCG step) and passes as parameter.
**Result**: Single 56-byte readback replaces all prior scalar readbacks.
Combined with B2, the RHMC hot path readback budget is now:
  ~100 × 8 bytes (CG convergence checks) + 56 bytes (Metropolis)

### B4: `gpu_links_to_lattice` Transfer ✅ RESOLVED

**Was**: After GPU RHMC, reads ALL gauge links back to CPU for gradient
flow: `18 * 8 * 4 * V` bytes (37 MB at 8^4, 590 MB at 16^4).
**Fix**: `GpuFlowState::from_gpu_gauge()` copies links on-device (GPU-GPU,
no PCI-e round-trip). The 37+ MB readback is eliminated entirely.
Production binaries now call `from_gpu_gauge()` instead of
`gpu_links_to_lattice()` for flow analysis.

### B5: CPU Gradient Flow ✅ RESOLVED

**Was**: `run_flow` in `gradient_flow.rs` runs W7/CK4 integrators on CPU.
**Fix**: `gpu_gradient_flow_resident()` in `gpu_flow.rs` runs the full
LSCFRK 2N-storage flow (Euler/W6/W7/CK4) on GPU with O(1) plaquette
readback via reduce chain. New components:
- `GpuFlowState::from_gpu_gauge()` — GPU-GPU link copy, no CPU round-trip
- `FlowReduceBuffers` — reduce chain for plaq_out → scalar (8 bytes vs O(V))
- `gpu_gradient_flow_resident()` — flow with `zero_buffer()` + reduced plaq
- `gpu_flow_plaquette_reduced()` — single 8-byte readback per measurement

Flow readback budget: 8 bytes per measurement point (vs 37+ MB full link
transfer before). `find_t0`/`find_w0` remain on CPU (trivial post-processing
on the small `Vec<FlowMeasurement>`).

### B6: CG Convergence Readback Optimization ✅ RESOLVED

**Was**: Fixed-interval convergence checking (every `check_interval` iterations)
meant ~820 GPU→CPU sync points per RHMC trajectory (41000 CG iters / 50).
Each sync flushes the GPU pipeline and blocks on `read_staging_f64`.
**Fix**: Exponential back-off in both `gpu_cg_solve_resident` and
`gpu_shifted_cg_solve_resident`: the check interval starts at the configured
value and doubles after each non-converged check, capping at 2000 iterations.
Sync count drops from O(I/C) to O(log(I/C)) — roughly 820 → 25 for typical
41000-iteration solves. Zero-fill optimization: replaced CPU `Vec` allocation
+ `upload_f64` with `zero_buffer()` in all CG init paths (resident, shifted,
async, brain, RHMC, dynamical).

### Priority Order

```
B1 + B2 + B3 + B4 + B5 + B6: ✅ ALL RESOLVED
  → Tensor cores via coralReef (cross-spring: WGSL→HMMA IR lowering needed)
  → Multi-GPU domain decomposition (halo exchange, multi-GPU CG)
  → True multi-shift CG (shared Krylov space across shifts)
```

### Frontier: Tensor Cores (coralReef Integration)

**Status**: PLANNED — cross-spring dependency.
coralReef has SASS-level HMMA/IMMA encoding + scheduling (SM75+) but the
WGSL→IR pipeline does not yet lower to tensor ops. SU(3) in the WGSL
preamble compiles to scalar FMA, not HMMA tiles. The F16 HMMA encoder works;
TF32/BF16 paths are commented out. Enabling this requires:
1. IR lowering pass: identify 3×3 complex matmul patterns → Op::Hmma
2. TF32/BF16 encoder completion in coralReef NV backend
3. Precision routing in barraCuda: staples/force in TF32 (Tier 1), CG in f64

### Frontier: Multi-GPU Dispatch

**Status**: Independent-trajectory dual-GPU works (bench_multi_gpu,
validate_dual_gpu_qcd, dual_gpu_trajectories). Missing: lattice domain
decomposition, halo exchange, multi-GPU CG with P2P. Large effort.

## Energy Efficiency Map

### The Hardware Reality

Per-silicon-unit power is not directly measurable. Neither NVIDIA (NVML / nvidia-smi)
nor AMD (sysfs `power1_average`) expose per-TMU, per-ALU, per-ROP, or per-tensor-core
wattage. The only hardware-reported number is **total GPU board power** (one scalar).
Individual functional units do not have dedicated shunt resistors.

### Differential Energy Cost Methodology

We infer per-unit energy efficiency using idle-vs-saturated power measurement:

```
idle_power         = GPU power at rest (no dispatches), averaged over 600ms settle + 3×200ms
unit_X_loaded      = GPU average power during X-only saturation micro-benchmark
delta_watts_X      = unit_X_loaded - idle_power
ops_per_watt_X     = measured_throughput_X / delta_watts_X
```

This gives **differential energy cost** per silicon unit — exactly what the tier
router needs for energy-aware decisions.

### Data Model

`UnitThroughput` carries four energy fields (all default to 0.0 for backward compat):

| Field         | Type  | Description                                    |
|---------------|-------|------------------------------------------------|
| `idle_watts`  | `f64` | GPU idle power before this benchmark           |
| `loaded_watts`| `f64` | GPU average power during unit saturation        |
| `delta_watts` | `f64` | `loaded - idle` (marginal cost, clamped ≥ 0)   |
| `ops_per_watt`| `f64` | `measured_peak / delta_watts` (0.0 if no data) |

`CompositionEntry` also carries `idle_watts`, `compound_watts`, and `delta_watts`
to answer whether compound dispatches (ALU + TMU) scale power linearly or
sub-linearly.

### What This Tells Us

- **Is TMU work "free" energy-wise?** If TMU saturates at +10W above idle vs
  +120W for ALU, routing PRNG to TMU saves both time AND energy.
- **AMD vs NVIDIA energy personality**: AMD ROP atomics are 7.4× faster — are
  they also more energy-efficient per operation?
- **Composition energy**: When ALU + TMU run simultaneously (2.8× throughput
  multiplier on RTX 3090), does power scale linearly or sub-linearly?
- **Future hardware signals**: If a hypothetical NPU on a GPU board handles PRNG
  at 0.5W vs TMU at 40W, the energy map makes the routing decision obvious.

### bench_silicon_profile Integration

The `bench_silicon_profile` binary wraps each saturation experiment with
`GpuTelemetry` snapshots:

1. Start `GpuTelemetry` once per adapter
2. Before each benchmark: settle 600ms, sample idle power (3×200ms average)
3. During benchmark: background thread samples power at ~100ms intervals
4. After: compute loaded average, call `profile.set_measured_energy(unit, idle, loaded)`
5. Composition benchmark: same pattern, stored in `CompositionEntry` energy fields

## Related Documents

- Exp 096-100: `experiments/100_SILICON_CHARACTERIZATION_AT_SCALE.md`
- Exp 105: `experiments/105_SILICON_ROUTED_QCD_REVALIDATION.md`
- Silicon profile data: `profiles/silicon/*.json`
- Silicon profile code: `barracuda/src/bench/silicon_profile.rs`
- Telemetry: `barracuda/src/bench/telemetry.rs`
- Precision routing: `barracuda/src/precision_routing.rs`
- Legacy deprecation: `barracuda/DEPRECATION_MIGRATION.md` (Lattice QCD section)
