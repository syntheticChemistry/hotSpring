# hotSpring v0.6.8 → ToadStool S53+ Absorption Handoff

**Date:** February 24, 2026
**From:** hotSpring (biomeGate compute campaign, Day 2)
**To:** ToadStool / BarraCuda core team
**License:** AGPL-3.0-only
**Context:** hotSpring synced to toadStool S53, compiled with zero errors/warnings,
and passed 39/39 validation suites in 6542.6s on biomeGate (RTX 3090 + Titan V + Akida).

---

## Executive Summary

hotSpring v0.6.8 produced three absorption-ready artifacts during the biomeGate
compute campaign:

1. **df64_core.wgsl** — Double-float (f32-pair) arithmetic library using Dekker splitting
2. **bench_fp64_ratio** pattern — Definitive GPU ALU throughput measurement
3. **Core streaming strategy** — Hardware-adaptive precision routing for consumer GPUs

Plus ongoing lattice QCD production infrastructure that's maturing toward absorption.

---

## Part 1: Items Ready for Absorption

### 1.1 df64_core.wgsl → `shaders/math/df64_core.wgsl`

**What:** Complete double-float arithmetic library in WGSL. Provides ~14 decimal
digits of precision using only f32 ALU operations (Dekker/Knuth error-free
transformations).

**Interface:**
```wgsl
struct Df64 { hi: f32, lo: f32 }

fn two_sum(a: f32, b: f32) -> Df64      // exact: s + e = a + b
fn two_prod(a: f32, b: f32) -> Df64     // exact: p + e = a * b
fn df64_add(a: Df64, b: Df64) -> Df64
fn df64_sub(a: Df64, b: Df64) -> Df64
fn df64_mul(a: Df64, b: Df64) -> Df64
fn df64_div(a: Df64, b: Df64) -> Df64
fn df64_from_f64(v: f64) -> Df64        // precision boundary
fn df64_to_f64(v: Df64) -> f64          // precision boundary
```

**Why absorb:** Every Spring doing f64 computation on consumer GPUs benefits
from the DF64 path. On RTX 3090, DF64 delivers 3.24 TFLOPS at 14-digit
precision — **9.9× faster than native f64** (0.33 TFLOPS). This is the
foundation for the core streaming strategy.

**No FMA intrinsic required.** Uses Dekker splitting (pure f32 ALU ops).
Works on NVIDIA (NVK or proprietary), AMD (RADV), Intel (ANV), Apple (Metal).

**Location in hotSpring:** `barracuda/src/lattice/shaders/df64_core.wgsl`

**Suggested toadStool location:** `crates/barracuda/src/shaders/math/df64_core.wgsl`
alongside `math_f64.wgsl`. ShaderTemplate could inject it via `with_df64_auto()`.

### 1.2 FP64 Strategy Detection → `GpuDriverProfile::fp64_strategy()`

**What:** Extend the existing `GpuDriverProfile` to include an `fp64_strategy`
field that selects between native f64 and the DF64 hybrid based on hardware.

```rust
pub enum Fp64Strategy {
    Native,     // Titan V, V100, A100, MI250X — use f64 everywhere
    Hybrid,     // Consumer NVIDIA/AMD — DF64 bulk + f64 critical
}

impl GpuDriverProfile {
    pub fn fp64_strategy(&self) -> Fp64Strategy {
        match self.fp64_rate {
            Fp64Rate::Full     => Fp64Strategy::Native,
            Fp64Rate::Throttled(_) => Fp64Strategy::Hybrid,
            _                  => Fp64Strategy::Hybrid,
        }
    }
}
```

**Why absorb:** Every Spring's `ShaderTemplate` compilation can automatically
select the optimal precision path. Consumer GPUs get DF64 bulk + f64 critical;
compute GPUs get native f64 throughout.

### 1.3 bench_fp64_ratio Pattern → `bench/gpu_throughput.rs`

**What:** The FMA chain micro-benchmark pattern for measuring raw GPU ALU
throughput across FP32, FP64, and DF64 precision paths. Includes anti-
optimization techniques (buffer-read constants, dual output writes) that
prevent compiler constant-folding.

**Why absorb:** Any Spring can characterize new GPUs with `cargo run --release
--bin bench_fp64_ratio`. The pattern is hardware-agnostic — it measures
whatever the Vulkan adapter reports.

**Location in hotSpring:** `barracuda/src/bin/bench_fp64_ratio.rs`

---

## Part 2: Maturing — Not Yet Ready

### 2.1 Production Beta-Scan Infrastructure

`production_beta_scan.rs` implements a multi-β-point quenched lattice QCD
temperature scan with checkpointing, JSON output, and streaming Omelyan HMC.
Still being validated during the biomeGate campaign.

**Status:** Runs correctly but needs more production validation before absorption.
Keep local in hotSpring for now.

### 2.2 CUDA Comparison Benchmark

`barracuda/cuda/bench_fp64_ratio.cu` provides a CUDA baseline for FP64
throughput comparison. This stays in hotSpring since toadStool is Rust/wgpu only.

---

## Part 3: What We Learned (Relevant to toadStool Evolution)

### 3.1 The FP64 Correction

**Old claim:** Consumer GPUs run fp64 at 1:2 via wgpu/Vulkan.
**Reality:** Hardware ~1:64 on consumer Ampere/Ada. Both CUDA and Vulkan
give the same fp64 throughput (~0.3 TFLOPS on RTX 3090).

**Impact on toadStool:** Any documentation or code comments claiming "1:2 via
wgpu" should be corrected. The actual advantage is:
- SHADER_F64 builtins work (sqrt, exp, etc.) — not throttled
- DF64 delivers 9.9× native f64 on FP32 cores — genuine innovation
- Titan V gives compute-class fp64 via NVK — CUDA abandoned Volta

### 3.2 NVK/NAK Compiler Efficiency

The Titan V achieves only ~8% of peak FP64 throughput through NVK/NAK.
Specific issues:

| Issue | Impact | Upstream bug? |
|-------|--------|---------------|
| Poor pipeline scheduling | All workloads at 8-10% peak | Needs SM70 latency tables in NAK |
| `exp(f64)` / `log(f64)` assertion crash | Can't use native transcendentals | Yes — file against Mesa |
| Device loss at 31^4+ lattice | PTE fault, nouveau virtual memory bug | Yes — kernel module issue |
| No loop unrolling for FP64 chains | Iterative kernels underperform | NAK scheduling opportunity |

**Action for toadStool:** If contributing to Mesa/NAK, SM70 latency tables
would give the highest leverage improvement across all Springs.

### 3.3 Cross-Spring Shader Evolution Tracker

The biomeGate campaign validated the full cross-spring ecosystem:

| From | To | Shader/Module | Validated |
|------|----|--------------|-----------|
| wetSpring | hotSpring | `batch_pair_reduce_f64` FMA fix | BCS bisection correct on NVK |
| neuralSpring | hotSpring | `xoshiro128ss.wgsl` GPU RNG | HMC momentum generation |
| hotSpring | toadStool | CG solver (5 shaders) | 15,360× readback reduction |
| hotSpring | toadStool | Lattice QCD (12 shaders) | Full SU(3) on GPU |
| wetSpring | toadStool | `BatchedOdeRK4Generic` | Replaces 5 domain ODEs |
| neuralSpring | toadStool | 38 GPU dispatch ops | Unified tensor→GPU path |

### 3.4 Validation Suite Performance (biomeGate RTX 3090)

The 39/39 validation took 6542.6 seconds. Heaviest suites:

| Suite | Time | Bottleneck |
|-------|------|-----------|
| Stanton-Murillo Transport | 2108.7s | CPU-bound N=10k DSF (9 cases) |
| Transport CPU/GPU Parity | 1577.2s | GPU transport at full N |
| GPU Transport (Paper 5) | 1260.0s | GPU-resident N=10k |
| GPU Streaming Dynamical | 770.3s | Full dynamical fermion HMC |
| GPU Streaming HMC | 441.1s | Quenched HMC with streaming CG |

The transport suites dominate (83% of total time). The core-streaming strategy
would reduce the GPU-bound suites significantly — estimated 5-7× for the
GPU transport and streaming HMC suites.

---

## Part 4: AMD Testing Opportunity

ToadStool has both AMD and NVIDIA consumer GPUs. The core-streaming strategy
should be characterized on AMD:

```bash
HOTSPRING_GPU_ADAPTER=amd cargo run --release --bin bench_fp64_ratio
```

AMD RDNA 2/3 has ~1:16 FP64 hardware (4× better than Ampere's 1:64). Expected
DF64 benefit is lower (~4× vs 10× on NVIDIA) but the native FP64 baseline is
4× higher. Worth measuring.

AMD's RADV/ACO compiler is more mature than NVK/NAK — expect higher % of peak.

---

## Summary: Absorption Checklist

| Item | Priority | Target Location | Ready? |
|------|----------|----------------|--------|
| df64_core.wgsl | **HIGH** | shaders/math/ | YES |
| Fp64Strategy enum | **HIGH** | device/ or GpuDriverProfile | YES |
| bench_fp64_ratio pattern | MEDIUM | bench/ or examples/ | YES |
| production_beta_scan | LOW | lattice/ | NOT YET |
| CUDA benchmark | N/A | Keep in hotSpring | N/A |

---

*39/39 validation suites pass on biomeGate (RTX 3090 + Titan V + Akida,
Threadripper 3970X, toadStool S53). Zero compile errors, zero warnings.*
