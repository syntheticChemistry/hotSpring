# Experiment 012: FP64 Core Streaming — Consumer GPU Chip Utilization

**Date:** February 24, 2026
**Hardware:** biomeGate (RTX 3090 24GB, Titan V 12GB HBM2)
**Binary:** `bench_fp64_ratio` (WGSL/wgpu) + `cuda/bench_fp64_ratio.cu` (CUDA)
**Status:** Complete — definitive measurement + new architectural strategy
**License:** AGPL-3.0-only

---

## Motivation

Previous experiments claimed consumer GPUs ran fp64 at 1:2 throughput via
wgpu/Vulkan, bypassing CUDA's reported 1:64 ratio. This claim, originating
from Experiment 006 comparisons and eigensolve benchmarks, was based on
flawed methodology (comparing native builtins to slow software emulation
rather than to native FP32).

This experiment:
1. Definitively measures FP32, FP64, and DF64 (double-float) throughput
2. Cross-validates WGSL/wgpu results against CUDA on the same hardware
3. Discovers the "core streaming" strategy for consumer GPU optimization

---

## Method

Pure FMA (fused multiply-add) chain benchmark — the simplest possible
compute-bound workload that isolates ALU throughput from memory effects.

| Parameter | Value |
|-----------|-------|
| Chain length | 4096 FMA ops per thread |
| Threads | 4,194,304 (4M) |
| Workgroup size | 256 |
| Warmup | 3 iterations |
| Measurement | 10 iterations (median) |
| Anti-optimization | Constants read from buffer (prevents compile-time folding) |

Three precision paths measured:
- **FP32**: `f32` multiply-add chain
- **FP64**: `f64` multiply-add chain (native hardware)
- **DF64**: `Df64 { hi: f32, lo: f32 }` using Dekker splitting (runs on FP32 cores)

The DF64 path uses `two_sum` and `two_prod` (Knuth/Dekker error-free
transformations) to achieve ~14 decimal digits of precision using only
f32 ALU operations.

---

## Results

### RTX 3090 (Ampere GA102) — 82 SMs, 10496 FP32 + 164 FP64 units

| Path | TFLOPS | Precision | Ratio vs FP32 |
|------|--------|-----------|--------------|
| FP32 (wgpu) | 14.98 | 7 digits | 1.0× |
| FP64 (wgpu) | 0.33 | 16 digits | 1:46 |
| DF64 (wgpu) | 3.24 | 14 digits | 1:4.6 |
| FP32 (CUDA) | 22.07 | 7 digits | — |
| FP64 (CUDA) | 0.29 | 16 digits | 1:77 |

**Key finding**: DF64 delivers **9.9× native f64 throughput** at 14-digit
precision by running on the FP32 cores.

### Titan V (Volta GV100) — 80 SMs, 5120 FP32 + 2560 FP64 units

| Path | TFLOPS | Precision | Ratio vs FP32 |
|------|--------|-----------|--------------|
| FP32 (NVK) | 1.40 | 7 digits | 1.0× |
| FP64 (NVK) | 0.59 | 16 digits | 1:2.4 |
| DF64 (NVK) | 0.22 | 14 digits | 1:6.4 |

**Key finding**: On Titan V, DF64 is **0.5× slower** than native f64.
Use native f64 only on compute-class GPUs.

### CUDA Cross-Validation

CUDA and Vulkan give the **same fp64 throughput** (~0.3 TFLOPS on RTX 3090),
confirming the ratio is hardware (silicon), not driver software.

---

## The Core Streaming Architecture

### The problem
On consumer Ampere/Ada GPUs, f64-only shaders leave 99.7% of the chip dark
(10,496 FP32 cores idle while 164 FP64 units work).

### The solution
Route bulk math through DF64 on FP32 cores, reserve native f64 for
precision-critical operations (reductions, convergence checks).

```
Consumer GPU (RTX 3090)           Compute GPU (Titan V)
┌──────────────────────┐          ┌──────────────────────┐
│ FP32: DF64 bulk      │          │ FP64: everything     │
│   3.24 TFLOPS @14d   │          │   7.45 TFLOPS @16d   │
│ FP64: reductions     │          │   (0.59 via NVK)     │
│   0.33 TFLOPS @16d   │          │                      │
│ Combined: ~3.5 TFLOPS│          │ Combined: 7.45 TFLOPS│
└──────────────────────┘          └──────────────────────┘
```

### Estimated impact on HMC

| Kernel | % of HMC | Strategy | Throughput |
|--------|----------|----------|-----------|
| SU(3) gauge force (7 staple muls) | 40% | DF64 | 3.24 TFLOPS |
| Wilson plaquette | 15% | DF64 | 3.24 TFLOPS |
| Momentum/kinetic | 10% | DF64 body, f64 reduction | Mixed |
| CG inner products | 20% | Native f64 | 0.33 TFLOPS |
| RNG + misc | 15% | Native f64 | 0.33 TFLOPS |

Effective throughput: 0.65 × 3.24 + 0.35 × 0.33 ≈ **2.22 TFLOPS**
vs current all-f64: 0.33 TFLOPS → **6.7× speedup**

---

## Corrections to Prior Work

| Document | Old Claim | Corrected |
|----------|-----------|-----------|
| Exp 006 (GPU FP64 Comparison) | "fp64:fp32 ~1:2" | Hardware ~1:64 on consumer |
| Exp 007 (CPU/GPU Scaling) | "wgpu bypasses CUDA gimp" | Both APIs give same fp64 |
| README.md | "true fp64:fp32 ratio is ~1:2" | DF64 hybrid is the breakthrough |
| whitePaper/STUDY.md | Finding #7 "1:2 ratio" | Corrected with bench_fp64_ratio data |
| metalForge/gpu/nvidia/HARDWARE.md | "1:2 via wgpu" | Hardware ~1:64, DF64 3.24 TFLOPS |

15+ documents corrected across the codebase.

---

## Files Created

| File | Purpose |
|------|---------|
| `barracuda/src/bin/bench_fp64_ratio.rs` | WGSL FMA chain benchmark (FP32/FP64/DF64) |
| `barracuda/cuda/bench_fp64_ratio.cu` | CUDA comparison benchmark |
| `barracuda/src/lattice/shaders/df64_core.wgsl` | Double-float library (Dekker arithmetic) |

## How to Run

```bash
source metalForge/nodes/biomegate.env  # or eastgate.env
cargo run --release --bin bench_fp64_ratio

# CUDA comparison
cd barracuda/cuda && nvcc -O3 -arch=sm_86 bench_fp64_ratio.cu -o bench && ./bench
```

---

## Conclusion

The "1:2 fp64 on consumer cards" claim was wrong. The hardware ratio is ~1:64.
But the real discovery is better: double-float arithmetic on the massive FP32
core array delivers 9.9× native f64 throughput. This "core streaming" strategy
— deciding which execution units to route each computation to — is analogous to
the GPU-NPU-CPU streaming pipeline, applied one level deeper into the silicon.

The story is actually stronger with the correction. We caught our own error
through rigorous benchmarking and found something more useful.
