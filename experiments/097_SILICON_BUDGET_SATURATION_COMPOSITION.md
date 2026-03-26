# Experiment 097: Silicon Budget, Saturation, and Composition

**Date**: 2026-03-26
**Status**: READY TO RUN
**Binary**: `bench_silicon_budget`, `bench_silicon_saturation`, `bench_silicon_composition`
**Hardware**: RTX 3090, RX 6950 XT, llvmpipe

## Goal

Characterize the theoretical and actual performance of every accessible silicon
unit on each GPU, then measure the compound effect when multiple units operate
on different sub-problems simultaneously.

## Phase 1: Silicon Budget Calculator (`bench_silicon_budget`)

Computes per-GPU theoretical peak from known specs:
- FP32/DF64/FP64 shader TFLOPS
- Memory bandwidth (GB/s)
- TMU texel rate (GT/s)
- ROP pixel rate (GP/s)
- Tensor core TFLOPS (inaccessible via WGSL — future coralReef path)
- QCD working-set analysis: which lattice sizes fit in L2, Infinity Cache, VRAM
- Precision tier throughput table (int2 → qf128)

## Phase 2: Silicon Saturation (`bench_silicon_saturation`)

Six micro-experiments, each saturating one unit:

| Exp | Target | Method | Output |
|-----|--------|--------|--------|
| 2a  | Shader ALU (FP32) | 512×4 FMA chain | Actual TFLOPS |
| 2a' | Shader ALU (DF64) | 256× Dekker chain | Actual DF64 TFLOPS |
| 2b  | Memory controller | Sequential vec4 read at 16-512 MB | Actual GB/s |
| 2c  | Cache hierarchy | Working-set sweep 64 KB → 256 MB | L2/IC boundaries |
| 2d  | Texture units | textureLoad flood (1024² R32Float) | Actual GT/s |
| 2e  | Shared memory | Tree reduction at WG=1024 | LDS bandwidth |
| 2f  | Atomic throughput | atomicAdd on 256 buckets | Gatom/s |

## Phase 3: Silicon Composition (`bench_silicon_composition`)

Three composition experiments measuring compound throughput:

| Exp | Units | QCD Analog | Metric |
|-----|-------|------------|--------|
| 3a  | ALU + TMU | Force compute + EOS table lookup | Composition multiplier |
| 3b  | ALU + BW  | Force (compute) + mom update (BW) | Overlap ratio |
| 3c  | ALU + Reduce | Dirac (ALU) + dot product (reduce) | CG pipeline efficiency |

Composition multiplier > 1.0 means the GPU successfully overlaps work on different
silicon units. 2.0 would mean full parallel execution (theoretical maximum for two
independent units).

## Expected Results

### RTX 3090 (Ampere)
- FP32 saturation: ~30-35 TFLOPS (85-98% of 35.6T theoretical)
- Memory bandwidth: ~700-850 GB/s (75-90% of 936 GB/s)
- TMU: ~400-500 GT/s (72-90% of 558 GT/s theoretical)
- DF64: ~2.5-3.2 TFLOPS
- Cache boundary at ~6 MB (L2 size)

### RX 6950 XT (RDNA 2)
- FP32 saturation: ~20-23 TFLOPS
- Memory bandwidth: ~400-500 GB/s (with Infinity Cache boost at small working sets)
- Cache boundary at ~4 MB (L2) then again at ~128 MB (Infinity Cache)
- DF64: ~4-5.5 TFLOPS (AMD 1:16 advantage)

### Cross-GPU
- AMD wins on DF64 and cache-resident workloads (≤16^4 lattice)
- NVIDIA wins on pure FP32 ALU and raw memory bandwidth
- Composition multiplier likely 1.1-1.3x (limited by shared warp scheduler)

## How to Run

```bash
# Phase 1
cargo run --release --bin bench_silicon_budget

# Phase 2
cargo run --release --bin bench_silicon_saturation

# Phase 3
cargo run --release --bin bench_silicon_composition
```

## Reports To

toadStool `compute.performance_surface.report` with:
- `theoretical.*` measurements (budget)
- `saturation.*` measurements (actual peaks)
- `composition.*` measurements (compound multipliers)
