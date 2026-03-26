# Experiment 098: QCD Silicon Benchmark v2

**Date**: 2026-03-26
**Status**: READY TO RUN
**Binary**: `bench_qcd_silicon` (evolved from v1)
**Hardware**: RTX 3090, RX 6950 XT, llvmpipe

## Goal

Profile every QCD HMC kernel — quenched through dynamical — at FP32 and DF64
precision tiers, across lattice sizes from 4^4 to 32^4 (production scale).
This replaces the v1 proxy-only benchmark with comprehensive coverage.

## Evolution from v1

| Feature | v1 | v2 |
|---------|----|----|
| Kernels | 6 (quenched only) | 14 (quenched + dynamical + observables + DF64) |
| Lattice sizes | 4^4 → 16^4 | 4^4 → 32^4 (1M sites) |
| Precision | FP32 only | FP32 + DF64 |
| Phase coverage | Quenched | Quenched + dynamical + observable |
| Cost model | None | HMC trajectory cost estimate at 32^4 |
| Silicon mapping | Simple F/B heuristic | Per-kernel opportunity analysis |

## Kernel Coverage

### FP32 Tier (11 kernels)

| Kernel | Phase | FLOP/site | Bottleneck | Silicon Opportunity |
|--------|-------|-----------|------------|---------------------|
| Gauge force | Quenched | 864 | ALU | Shader core peak |
| Plaquette | Quenched | 1296 | ALU | 6 plane matmuls |
| SU(3) matmul | Quenched | 216 | ALU | Tensor core MMA candidate |
| Link update | Quenched | 400 | ALU | Cayley exp + sqrt |
| Mom update | Quenched | 72 | Memory | P += dt*F |
| CG dot+reduce | Dynamical | 8 | Shared mem | CG bottleneck |
| Dirac stencil | Dynamical | 288 | Balanced | Heart of CG solver |
| PF force | Dynamical | 1000 | ALU | Most expensive dynamical kernel |
| PRNG heat bath | Dynamical | 360 | Transcendental | TMU LUT candidate |
| Polyakov loop | Observable | 576 | Latency | Serial Nt chain |
| Grad flow acc | Observable | 288 | Balanced | Algebra accumulation |

### DF64 Tier (3 kernels)

| Kernel | FLOP/site | Notes |
|--------|-----------|-------|
| Force (DF64) | 3456 | ~4× FP32 for Dekker arithmetic |
| Plaquette (DF64) | 5184 | Higher precision trace |
| CG dot (DF64) | 32 | Error-compensated accumulation |

## Scales

| Volume | Sites | Working Set | Notes |
|--------|-------|-------------|-------|
| 4^4 | 256 | ~150 KB | Fits L2 on all cards |
| 8^4 | 4096 | ~2.4 MB | Fits L2, fully cached |
| 8^3×16 | 8192 | ~4.7 MB | Fits AMD IC |
| 16^4 | 65536 | ~38 MB | AMD IC advantage |
| 32^4 | 1,048,576 | ~600 MB | VRAM only, production scale |

## How to Run

```bash
cargo run --release --bin bench_qcd_silicon
```

## Reports To

toadStool with per-(kernel, volume, GPU) measurements at both FP32 and DF64.
