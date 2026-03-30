# Experiment 049: Precision Brain + Heterogeneous GPU Evaluation

**Track:** cross-substrate
**Phase:** v0.6.25
**Status:** COMPLETE — bench_precision_eval clean exit (3.6s, both GPUs)
**Binary:** `bench_precision_eval`

## Purpose

Evaluate all 3 tiers of precision (F32, F64 native, DF64 emulated) across
both available GPUs (Titan V NVK + RTX 3090 proprietary). Implement a
self-routing precision brain that discovers hardware capabilities at startup
and routes physics workloads to the best safe tier without static heuristics.

## What It Tests

1. **Hardware calibration** — safe per-tier probing (F32 → F64 → F64Precise → DF64)
   without device poisoning, even on GPUs where DF64 compilation kills the device
2. **Transfer profiles** — upload/readback bandwidth at multiple buffer sizes per GPU
3. **Shader tier matrix** — each test shader × each precision tier, with NVVM-gated
   transcendental skipping
4. **Numerical accuracy** — ULP error analysis (exp/log shader: 0 ULP on Titan V)
5. **Physics pipeline E2E** — HMC, BCS bisection, dielectric Mermin across both GPUs
6. **Dual-card cooperative patterns** — Split BCS, Split HMC, Redundant HMC, PCIe roundtrip
7. **Brain routing** — domain-to-tier routing table derived from calibration data

## Key Findings

### NVIDIA NVVM Device Poisoning

A single failed DF64 or F64Precise shader compilation on the proprietary NVIDIA
driver permanently invalidates the wgpu device — all subsequent operations panic
with `"Buffer is invalid"`. This is unrecoverable within the same process. NVK
(Mesa) handles all tiers correctly.

The brain works around this by:
- Probing safest tiers first (F32 → F64 → F64Precise → DF64)
- Inferring transcendental safety from adapter name (NVK = safe)
- Gating the exp/log shader entirely on GPUs with `nvvm_transcendental_risk`

### Measured Capabilities

| GPU | F32 | F64 | F64Precise | DF64 |
|-----|-----|-----|------------|------|
| Titan V (NVK) | ✓ full | ✓ full | ✓ full | ✓ full |
| RTX 3090 (proprietary) | ✓ full | ✓ full | △ arith only | △ arith only |

### Transfer Profiles

| Metric | Titan V (PCIe 3.0) | RTX 3090 (PCIe 4.0) |
|--------|---------------------|----------------------|
| Upload peak | 2.2 GB/s @ 1MB | 6.6 GB/s @ 1MB |
| Readback peak | 4.4 GB/s @ 16MB | 4.3 GB/s @ 4MB |
| Dispatch overhead | 6 μs | 3 μs |
| Reduce scalar | 1915 μs | 470 μs |

### Brain Routing

| Domain | Titan V | RTX 3090 |
|--------|---------|----------|
| LatticeQcd | F64 | F64 |
| Dielectric | F64Precise | F64 |
| Eigensolve | F64Precise | F64 |
| MolecularDynamics | F64 | F64 |
| NuclearEos | F64 | F64 |
| GradientFlow | F64 | F64 |
| KineticFluid | F64 | F64 |

### Dual-Card Cooperative

| Pattern | Wall | vs Best Single | Notes |
|---------|------|----------------|-------|
| Split BCS (30/70) | 4.4 ms | — | precise validates, throughput computes |
| Split HMC | 45 ms | 2.2× | force on throughput, validation on precise |
| Redundant HMC | 228 ms | — | max plaquette diff: 2.88e-9 |
| PCIe roundtrip | 1.7 ms | — | 512KB, 1.2 GB/s effective |

## New Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `hardware_calibration.rs` | 487 | Safe per-tier probe + capability mask |
| `precision_brain.rs` | 341 | Self-routing brain from calibration data |
| `precision_eval.rs` | ~300 | Per-shader precision/throughput profiler |
| `transfer_eval.rs` | ~200 | PCIe bandwidth profiler |
| `pipeline_eval.rs` | ~250 | Full physics pipeline E2E profiler |
| `dual_pipeline_eval.rs` | ~300 | Cooperative dual-card patterns |
| `bench_precision_eval` | ~400 | Master benchmark binary (6 phases) |

## Metrics

- 840 lib tests, 0 failures, 6 ignored
- 0 clippy warnings (lib + bins)
- bench_precision_eval: 3.6s release, exit 0, no device poisoning
