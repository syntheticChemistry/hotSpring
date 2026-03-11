# hotSpring v0.6.29 → Upstream Sync v4

**Date:** March 11, 2026
**From:** hotSpring v0.6.29 (847 lib tests, 0 failures, 0 clippy warnings)
**To:** barraCuda, toadStool, coralReef teams
**Supersedes:** `HOTSPRING_V0628_KOKKOS_PARITY_DF64_FIX_HANDOFF_MAR11_2026.md` (archived)
**License:** AGPL-3.0-only

---

## Executive Summary

- **Rewired to barraCuda v0.3.5** (`8d63c77`): health module, pharma ops, stable specials, FMA policy, 36 tolerances, tridiag eigensolver, HMM batch shader, hydrology extensions. Clean compile, 847/847 tests pass.
- **Acknowledges toadStool S146** (`751b3849`): `nvvm_transcendental_risk` in `gpu.info`, PrecisionBrain in `compile_wgsl_multi`, VRAM-aware routing, 19 SpringDomain variants, HealthSpring.
- **Acknowledges coralReef Iter 31** (`9d63b72`): **all 9 cross-spring shader compilation gaps resolved** (46/46), FMA lowering, f64 log2 fix, vec3\<f64\> scalarization, SU3 preamble, Nouveau UAPI migration, UVM 0x1F fix, 1509 tests.
- **Kokkos parity benchmark** (from v0.6.28): 12.4× gap persists — primary blocker remains NVVM-safe DF64 exp path.
- **hotSpring precision modules stay local**: PrecisionBrain, HardwareCalibration, PrecisionRouting work with `GpuF64` and do actual dispatch probes. Upstream versions acknowledged; doc comments updated to reference v0.3.5/S146/Iter 31.

---

## Part 1: barraCuda v0.3.5 Acknowledgment

### New Upstream Capabilities Available to hotSpring

| Module | Location | hotSpring Use |
|--------|----------|---------------|
| `PrecisionBrain::from_device()` | `device::precision_brain` | hotSpring keeps local (GpuF64-based, dispatch-probed) |
| `HardwareCalibration::from_device()` | `device::hardware_calibration` | hotSpring keeps local (richer probe) |
| `FmaPolicy` + `domain_requires_separate_fma()` | `device::fma_policy` | Available — hotSpring can import for FMA-sensitive domains |
| `PhysicsDomain` (12 variants) | `device::precision_tier` | hotSpring mirrors all 12 locally |
| `health::{biosignal, microbiome, pkpd}` | `health/` | New — not yet used by hotSpring |
| `ops::pharma::{FoceGradientGpu, VpcSimulateGpu}` | `ops/pharma/` | New — not yet used by hotSpring |
| `special::stable_gpu` (log1p, expm1, erfc, bessel_j0) | `special/` | Available for cancellation-sensitive physics |
| `spectral::tridiag_eigh_gpu` | `spectral/` | Available for eigensolve pipelines |
| 36 tolerances + introspection | `tolerances` | Available via `by_name()`, `all_tolerances()` |
| `stats::hydrology::fao56_et0_with_ea` | `stats/hydrology/` | New — not yet used by hotSpring |

### No API Breaks

barraCuda v0.3.5 is fully backward-compatible with v0.3.4. hotSpring compiles cleanly with zero changes to source code beyond the pin update.

---

## Part 2: toadStool S146 Acknowledgment

| Feature | Status in hotSpring |
|---------|-------------------|
| `nvvm_transcendental_risk` in `gpu.info` | Available for runtime query when toadStool daemon is running |
| PrecisionBrain in `compile_wgsl_multi` | Delegation seam documented — can route through toadStool for multi-device compilation |
| VRAM-aware routing (`route_with_vram`) | Available for large-N MD simulations |
| 19 SpringDomain variants | hotSpring uses 12 PhysicsDomain locally; toadStool's 19 are a superset |
| PcieTopologyGraph (stable) | Available for multi-GPU placement decisions |
| HealthSpring in `Spring` enum | Acknowledged — hotSpring does not produce health workloads |

---

## Part 3: coralReef Iter 31 Acknowledgment

### All Cross-Spring Shader Gaps Resolved

| Gap | Resolution |
|-----|-----------|
| `torsion_angles_f64` | repair_ssa: unreachable block elimination, critical-edge phi |
| `hill_dose_response_f64` | f64 log2 widening fix for `pow` lowering |
| `euler_hll_f64` | vec3\<f64\> SM70 componentwise scalarization |
| SU3 preamble | `su3_f64_preamble.wgsl` auto-prepend with dependency chaining |
| SPIR-V roundtrip | Relational, non-literal const init, Compose/Splat |
| AMD FRnd | `V_TRUNC/FLOOR/CEIL/RNDNE` for f32 (VOP1) and f64 (VOP3) |
| AMD literal materialization | `s_mov` materialization pass |
| AMD Discriminant encoding | Fixed in `emit_discriminant` |
| emit_f64_cmp | Defensive 1→2 component widening |

**Sovereign compile: 46/46** (was 45/46). `complex_f64` gap resolved.

### Nouveau UAPI Migration

coralReef now has `VM_INIT` / `VM_BIND` / `EXEC` pipeline for kernel 6.6+. This is the path to sovereign dispatch on Titan V via nouveau. hotSpring Experiment 051 data (channel alloc EINVAL) was the root cause discovery.

### UVM Fix

`Nv0080AllocParams` with `device_id` fixes the `NV_ERR_OPERATING_SYSTEM` (0x1F) on RTX 3090 proprietary. Needs re-test.

---

## Part 4: Outstanding Action Items

### For barraCuda (P1)

1. **NVVM-safe DF64 exp** — Taylor series with Cody-Waite range reduction. This is the single biggest hotSpring performance unlock (4-8× on Ampere). See Exp 053 handoff.
2. **ReduceScalarPipeline f64 fix** — `sum_f64()` returns zero for large buffers. Physics is correct but energy reporting is broken.

### For toadStool (P2)

1. **Acknowledge hotSpring Exp 053 data** — 12.4× Kokkos gap, DF64 transcendental poisoning root cause.
2. **Spring pin update** — hotSpring v0.6.29, barraCuda v0.3.5.

### For coralReef (P1)

1. **Nouveau E2E validation** — run the new UAPI path on Titan V with kernel 6.17.
2. **UVM re-test** — `Nv0080AllocParams` fix on RTX 3090 proprietary.

---

## Metrics

| Metric | v0.6.28 | v0.6.29 |
|--------|---------|---------|
| barraCuda | v0.3.4 (`a012076`) | v0.3.5 (`8d63c77`) |
| toadStool | S145 (`969341cd`) | S146 (`751b3849`) |
| coralReef | Iter 30 (`c84137c`) | Iter 31 (`9d63b72`) |
| Sovereign compile | 45/46 | **46/46** |
| Lib tests | 847 | 847 |
| Clippy warnings | 0 | 0 |
| Kokkos gap | 12.4× | 12.4× (unchanged — awaiting DF64 exp fix) |
