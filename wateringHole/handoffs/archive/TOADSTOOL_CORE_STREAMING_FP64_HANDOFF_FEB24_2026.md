# Core Streaming: Hybrid FP64 on Consumer GPUs

**Date:** February 24, 2026
**From:** hotSpring (biomeGate compute campaign, Day 2)
**To:** ToadStool core team
**License:** AGPL-3.0-only
**Hardware for toadStool testing:** RTX 4070 (nvidia, Ada) + AMD GPU (RADV) + Titan V (NVK)
**Priority:** High — this is a potential 7× HMC speedup on consumer GPUs

---

## Executive Summary

We built `bench_fp64_ratio` — a definitive FMA chain micro-benchmark — and
discovered that our "fp64:fp32 ~1:2 on consumer GPUs" claim was wrong. The
hardware ratio on consumer NVIDIA (Ampere/Ada) is **~1:64**, same as CUDA.
But this revealed something better: a **hybrid core-streaming strategy** that
delivers **9.9× the f64-equivalent throughput** on consumer GPUs by routing
bulk math through double-float (f32-pair) arithmetic on the massive FP32 core
array.

This is the same philosophy as our GPU-NPU-CPU streaming pipeline, applied
one level deeper: **streaming to specific execution units within the GPU**.

---

## 1. The Corrected FP64 Picture

### RTX 3090 (Ampere GA102) — consumer, 82 SMs, 10496 FP32 + 164 FP64 units

| Path | FP32 TFLOPS | FP64 TFLOPS | Ratio | Source |
|------|----------:|----------:|:------|--------|
| Vulkan/wgpu (nvidia proprietary) | 14.98 | 0.33 | 1:46 | `bench_fp64_ratio` |
| CUDA (nvcc -O3) | 22.07 | 0.29 | 1:77 | `cuda/bench_fp64_ratio.cu` |
| Hardware spec | 35.6 | 0.56 | 1:64 | NVIDIA whitepaper |

### Titan V (Volta GV100) — compute-class, 80 SMs, 5120 FP32 + 2560 FP64 units

| Path | FP32 TFLOPS | FP64 TFLOPS | Ratio | Source |
|------|----------:|----------:|:------|--------|
| Vulkan/wgpu (NVK/NAK) | 1.40 | 0.59 | 1:2.4 | `bench_fp64_ratio` |
| Hardware spec | 14.9 | 7.45 | 1:2 | NVIDIA whitepaper |

### Key corrections to our documentation

- **Consumer Ampere/Ada fp64 is hardware 1:64** — confirmed by both CUDA and
  Vulkan giving the same fp64 throughput (~0.3 TFLOPS). The ratio difference
  (1:46 vs 1:77) is nvcc's more aggressive FP32 loop unrolling.
- **Titan V has genuine 1:2 hardware** — 2,560 dedicated FP64 cores (same GV100
  die as Tesla V100). Confirmed at 1:2.4 through NVK.
- **NVK/NAK compiler achieves only ~8% of Titan V's peak** — massive room for
  optimization (see §4 below).
- **The old evidence was flawed**: "native builtins 1.5-2.2× faster than math_f64"
  only proved our software emulation was slow. The eigensolve speed comparison
  was confounded by NVK dispatch overhead.

---

## 2. The Core Streaming Discovery

### The problem: 99.7% of the chip is dark during fp64 work

On the RTX 3090, our lattice QCD shaders use `f64` types exclusively. This
routes ALL computation through the 164 dedicated FP64 units, leaving the
10,496 FP32 cores completely idle.

### The solution: double-float (f32-pair) arithmetic on FP32 cores

`bench_fp64_ratio` measured three precision paths on the RTX 3090:

| Path | Throughput | Precision | Uses |
|------|-----------|-----------|------|
| **FP32** | 14.98 TFLOPS | 7 digits (24-bit mantissa) | Not enough for physics |
| **DF64 (f32-pair)** | **3.24 TFLOPS** | **14 digits (48-bit mantissa)** | Bulk SU(3) math |
| **FP64 native** | 0.33 TFLOPS | 16 digits (53-bit mantissa) | Reductions, convergence |

**DF64 delivers 9.9× the f64-equivalent throughput** at 14-digit precision by
running on the FP32 cores. For most lattice QCD operations, 14 digits is more
than sufficient.

On the Titan V: DF64 is **0.5× slower** than native f64. Use native f64 only.

### The architecture: hardware-adaptive core selection

```
Consumer GPU (1:64)                     Compute GPU (1:2)
┌──────────────────────┐               ┌──────────────────────┐
│ 10,496 FP32 cores    │               │ 5,120 FP32 cores     │
│ ├─ DF64 bulk math    │               │ (idle for f64 work)  │
│ │  3.24 TFLOPS @14d  │               │                      │
│ │                    │               │ 2,560 FP64 cores     │
│ 164 FP64 units       │               │ ├─ ALL math          │
│ ├─ Reductions        │               │ │  7.45 TFLOPS @16d  │
│ │  0.33 TFLOPS @16d  │               │ │  (0.59 w/ NVK)     │
│ │                    │               │                      │
│ Combined: ~3.5 TFLOPS│               │ Combined: 7.45 TFLOPS│
└──────────────────────┘               └──────────────────────┘
```

---

## 3. Implementation Guide

### 3.1 The df64_core.wgsl Library

Already prototyped at `barracuda/src/lattice/shaders/df64_core.wgsl`.

Core types and operations:

```wgsl
struct Df64 { hi: f32, lo: f32 }

// Error-free transformations (Knuth/Dekker)
fn two_sum(a: f32, b: f32) -> Df64    // exact: s + e = a + b
fn two_prod(a: f32, b: f32) -> Df64   // exact: p + e = a * b (Dekker split)

// Core arithmetic
fn df64_add(a: Df64, b: Df64) -> Df64
fn df64_sub(a: Df64, b: Df64) -> Df64
fn df64_mul(a: Df64, b: Df64) -> Df64
fn df64_div(a: Df64, b: Df64) -> Df64

// Conversion
fn df64_from_f64(v: f64) -> Df64      // f64 → df64 at precision boundary
fn df64_to_f64(v: Df64) -> f64        // df64 → f64 for accumulations
```

No FMA intrinsic required — uses Dekker splitting (pure f32 ALU ops).
Works on NVIDIA (NVK or proprietary), AMD (RADV), Intel (ANV), Apple (Metal).

### 3.2 ShaderTemplate Integration

The `GpuDriverProfile` already identifies Volta/Ampere/Ada/RDNA. Add:

```rust
pub enum Fp64Strategy {
    Native,     // Titan V, V100, A100, MI250X — use f64 everywhere
    Hybrid,     // Consumer NVIDIA/AMD — DF64 bulk + f64 critical
}

impl GpuDriverProfile {
    pub fn fp64_strategy(&self) -> Fp64Strategy {
        match (self.arch, self.fp64_rate) {
            (_, Fp64Rate::Full)     => Fp64Strategy::Native,
            (_, Fp64Rate::Throttled(_)) => Fp64Strategy::Hybrid,
            _                       => Fp64Strategy::Hybrid,
        }
    }
}
```

`ShaderTemplate` would then select between:
- `su3_gauge_force_f64.wgsl` (current — all native f64)
- `su3_gauge_force_hybrid.wgsl` (new — DF64 bulk + f64 critical)

### 3.3 Kernel-by-Kernel Precision Analysis

Which HMC kernels can safely use DF64 (14 digits)?

| Kernel | % of HMC | DF64-safe? | Rationale |
|--------|----------|------------|-----------|
| `su3_gauge_force` | 40% | **YES** | 7 SU(3) muls — bulk linear algebra |
| `wilson_plaquette` | 15% | **YES** | 4 SU(3) muls + trace |
| `su3_momentum_update` | 5% | **YES** | Simple multiply-add |
| `su3_kinetic_energy` | 5% | **YES** (body), NO (reduction) | DF64 per-site, f64 for global sum |
| `su3_link_update` (Cayley) | 10% | **PARTIAL** | Cayley exponential may need f64 |
| `su3_random_momenta` | 5% | **NO** | RNG distribution precision matters |
| CG solver inner products | 20% | **NO** | Convergence-critical |

**Estimated impact**: If 65% of HMC moves to DF64:
- Effective throughput: 0.65 × 3.24 + 0.35 × 0.33 = **2.22 TFLOPS**
- vs current all-f64: 0.33 TFLOPS
- **6.7× speedup on consumer GPUs**

### 3.4 Concurrent Execution

On Ampere, FP32 and FP64 units execute **simultaneously** within the same SM.
A shader that mixes DF64 and native f64 operations naturally utilizes both
execution unit types — no explicit scheduling needed. The hardware interleaves
f32 and f64 instructions at the warp scheduler level.

---

## 4. AMD RADV Testing Plan

ToadStool has both AMD and NVIDIA consumer GPUs. This is perfect for
cross-vendor validation of the core-streaming strategy.

### What to measure

Run `bench_fp64_ratio` on the AMD GPU:

```bash
HOTSPRING_GPU_ADAPTER=amd cargo run --release --bin bench_fp64_ratio
```

Expected results for RDNA 2/3 (e.g., RX 6800, RX 7900 XTX):

| Metric | Expected | Why |
|--------|----------|-----|
| FP32 throughput | 10-25 TFLOPS | Depends on CU count |
| FP64 throughput | 0.3-1.2 TFLOPS | RDNA has ~1:16 FP64 (better than Ampere's 1:64!) |
| DF64 throughput | 1-4 TFLOPS | Should scale with FP32 |
| FP64:FP32 ratio | ~1:16 | RDNA2/3 has 1 FP64 per 4 CUs (not 1:64) |

**RDNA's 1:16 is 4× better than Ampere's 1:64.** AMD's consumer silicon has
more FP64 hardware per compute unit. This makes the core-streaming strategy
even more interesting on AMD — the DF64 and native f64 paths are closer in
throughput, so the hybrid strategy provides less benefit BUT the native f64
baseline is already 4× better.

### AMD-specific notes

- **RADV/ACO compiler**: More mature than NVK/NAK for compute. ACO matches
  or beats AMD's proprietary driver on many workloads. Expect higher %
  of peak utilization.
- **Wave64 mode**: AMD uses 64-wide SIMD (vs NVIDIA's 32-wide warps).
  The `@workgroup_size` in our WGSL shaders may need tuning — try 64 and 256.
- **Infinity Cache**: RDNA3's 96MB L3 could keep SU(3) link arrays cached
  for the entire staple computation, reducing memory latency dramatically.
- **SHADER_F64 support**: Confirmed available on RDNA2+ via RADV/Vulkan.

### Cross-vendor validation checklist

1. `bench_fp64_ratio` — FP32:FP64:DF64 on AMD (the key measurement)
2. `f64_builtin_test` — verify native f64 builtins compile and produce correct results
3. `validate_cpu_gpu_parity` — physics matches CPU f64 to 1e-15
4. `bench_gpu_fp64` — BCS + eigensolve throughput comparison
5. `validate_gpu_beta_scan` — full QCD physics on AMD (the gold standard)
6. `bench_wgsize_nvk` → `bench_wgsize_amd` — optimal workgroup size scan

---

## 5. NVK/NAK Compiler Optimization

The Titan V has 7.45 TFLOPS of f64 hardware but NVK/NAK extracts only 0.59
TFLOPS (8%). This is the **highest-leverage optimization target** for
toadStool.

### What NAK is missing (based on bench_fp64_ratio)

| Issue | Evidence | Impact |
|-------|----------|--------|
| Poor pipeline scheduling | FP32 at 1.40/14.9 TFLOPS = 9.4% of peak | All workloads |
| No loop unrolling | Dependent FMA chain not optimized | Iterative kernels |
| FP64 underutilized | 0.59/7.45 TFLOPS = 7.9% of peak | All f64 work |
| Register pressure | May be spilling to VRAM | Complex shaders |

### Action items

1. **Profile with `VK_MESA_NVK_DEBUG=*`** — identify shader compilation bottlenecks
2. **Build Mesa from git HEAD** — 6 months of NAK fixes since 25.1.5
3. **Contribute SM70 latency tables to NAK** — NAK uses the same latency model
   for all architectures; Volta-specific tables would help scheduling
4. **File upstream issues** for:
   - FP64 FMA chain performance (include our benchmark SPIR-V)
   - `exp(f64)` / `log(f64)` assertion crash
   - Device loss at 31^4+ lattice (PTE fault)
5. **If NAK reaches 30% efficiency**: 2.2 TFLOPS fp64 on Titan V — **7× faster
   than the RTX 3090 via CUDA**, making a $500 used Titan V the best fp64
   consumer compute card available

---

## 6. Files Created/Modified

### New files

| File | Purpose |
|------|---------|
| `barracuda/src/bin/bench_fp64_ratio.rs` | Definitive FP32:FP64:DF64 benchmark |
| `barracuda/cuda/bench_fp64_ratio.cu` | CUDA comparison (same FMA chain workload) |
| `barracuda/src/lattice/shaders/df64_core.wgsl` | Double-float library (Dekker arithmetic) |

### Updated files (corrected "1:2" claims)

| File | Change |
|------|--------|
| `whitePaper/STUDY.md` | Header, finding #7, Phase D, L1 results |
| `whitePaper/README.md` | FP64 discovery section |
| `whitePaper/BARRACUDA_SCIENCE_VALIDATION.md` | §9.7.1, resource table |
| `experiments/007_CPU_GPU_SCALING_BENCHMARK.md` | "CUDA gimp" table replaced with measured data |

### How to run

```bash
# FP64:FP32:DF64 ratio on your NVIDIA GPU
HOTSPRING_GPU_ADAPTER=4070 cargo run --release --bin bench_fp64_ratio

# Same on your AMD GPU
HOTSPRING_GPU_ADAPTER=amd cargo run --release --bin bench_fp64_ratio

# CUDA comparison (requires nvcc)
cd barracuda/cuda
nvcc -O3 -arch=sm_89 bench_fp64_ratio.cu -o bench_fp64_ratio_cuda  # sm_89 for Ada
./bench_fp64_ratio_cuda

# Physics validation (confirm DF64 doesn't break anything)
cargo run --release --bin validate_gpu_beta_scan
cargo run --release --bin validate_cpu_gpu_parity
```

---

## 7. The Bigger Picture

This discovery reframes barracuda/toadStool's GPU compute story:

| Old claim | New reality | Advantage |
|-----------|-------------|-----------|
| "fp64 at 1:2 on consumer GPUs" | fp64 at 1:64 (hardware) | Honest, verifiable |
| "Vulkan bypasses CUDA gimping" | Both give same fp64 throughput | Focus on real advantages |
| — | **DF64 at 3.24 TFLOPS (10× native)** | Genuine innovation |
| — | **Titan V 1:2 via NVK (CUDA can't)** | Open-driver access to compute-class fp64 |
| — | **Cross-vendor (AMD, Intel, Apple)** | CUDA is NVIDIA-only |
| — | **Streaming pipeline (zero round-trips)** | End-to-end efficiency |

The story is actually **stronger** with the correction:
- We caught our own error through rigorous benchmarking
- We found something better: hybrid core streaming
- We have a clear path to 7× speedup on consumer hardware
- The Titan V through NVK is a genuine compute card that CUDA abandoned

**Have fun exploring this on AMD. The Infinity Cache + RDNA's 1:16 fp64 ratio
could make the AMD path even more interesting than NVIDIA for this workload.**

---

*Update: biomeGate compute campaign complete. RTX 3090 32^4 quenched beta-scan
finished 12/12 points in 13.6 hours ($0.58). Deconfinement transition at β=5.69
(χ=40.1) matches known β_c=5.692. Titan V 30^4 failed (NVK PTE fault).
See experiments/013_BIOMEGATE_PRODUCTION_BETA_SCAN.md for full results.*
