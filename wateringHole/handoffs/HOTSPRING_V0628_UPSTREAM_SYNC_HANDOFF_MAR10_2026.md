# hotSpring v0.6.28 → Upstream Primal Sync v3

**Date:** March 10, 2026
**From:** hotSpring v0.6.28
**To:** barraCuda, toadStool, coralReef teams
**Supersedes:** `HOTSPRING_V0627_UPSTREAM_SYNC_HANDOFF_MAR10_2026.md` (archived)

---

## Summary

hotSpring v0.6.28 syncs to barraCuda `a012076`, toadStool S145 (`969341cd`),
and coralReef Iteration 30 (`472e5b8`). This cycle acknowledges the major
cross-spring absorption of hotSpring's precision infrastructure: PrecisionBrain,
HardwareCalibration, and PrecisionTier are now upstream in both barraCuda and
toadStool. hotSpring adds 5 new PhysicsDomain variants for upstream parity
and documents the integration seams for future delegation.

---

## To: barraCuda Team

### Validated

- **barraCuda `a012076`** (v0.3.4 expanded): clean `cargo check`, 842 lib tests passing
- **`enable f64;` stripping fix** (`c4a0f2b`): hotSpring has zero WGSL files with `enable f64;` — no local impact, but the fix is picked up via the pin
- **PrecisionBrain/HardwareCalibration/PrecisionTier absorption**: Acknowledged. hotSpring's local versions are richer (actual dispatch probes, sovereign detection, `GpuF64` integration) but enum definitions now mirror upstream's 12 domains

### Enum Parity Achieved

hotSpring's `PhysicsDomain` now has 12 variants matching upstream:

| Variant | Precision Category | Routing |
|---------|-------------------|---------|
| LatticeQcd | Throughput-bound | F64 → DF64 (if throttled) → F32 |
| GradientFlow | Moderate | F64 → DF64 → F32 |
| Dielectric | Precision-critical | F64Precise → F64 → DF64 → F32 |
| KineticFluid | Throughput-bound | F64 → DF64 (if throttled) → F32 |
| Eigensolve | Precision-critical | F64Precise → F64 → DF64 → F32 |
| MolecularDynamics | Throughput-bound | F64 → DF64 (if throttled) → F32 |
| NuclearEos | Moderate | F64 → DF64 → F32 |
| **PopulationPk** | Moderate (new) | F64 → DF64 → F32 |
| **Bioinformatics** | Throughput-bound (new) | F64 → DF64 (if throttled) → F32 |
| **Hydrology** | Moderate (new) | F64 → DF64 → F32 |
| **Statistics** | Throughput-bound (new) | F64 → DF64 (if throttled) → F32 |
| **General** | Throughput-bound (new) | F64 → DF64 (if throttled) → F32 |

### Future Delegation

hotSpring keeps local `PrecisionBrain`, `HardwareCalibration`, `PrecisionTier`
because:
- Local brain works with `GpuF64` (not `WgpuDevice`)
- Local calibration does actual dispatch probes (upstream synthesizes from profile)
- Local routing includes sovereign bypass detection

Once barraCuda's device abstraction stabilizes, hotSpring can replace local
implementations with thin wrappers around upstream.

---

## To: toadStool Team

### Acknowledged from S145

| Feature | Status in hotSpring |
|---------|-------------------|
| `PrecisionBrain` with `PrecisionHint` routing | Documented — future delegation seam when toadStool runtime integrates |
| `NvkZeroGuard` (zero-output detection) | Not yet wired — hotSpring has no NVK zero-output detection; can delegate to toadStool |
| `dispatch_latency_ratio` on `TierCapability` | Documented — hotSpring uses `is_f64_throttled()` (dispatch_us ratio); can delegate to toadStool's field |
| 8 new `WorkloadPatterns` | Documented in device_pair.rs and workload_planner.rs |
| `ProviderRegistry` | Documented — spring-as-provider socket resolution available when toadStool runtime integrates |
| 5 new capability domains | Matched — hotSpring added corresponding PhysicsDomain variants |

### Integration Seams

1. **NvkZeroGuard**: hotSpring's `HardwareCalibration::probe()` currently doesn't check for zero-output on NVK. toadStool's `NvkZeroGuard` provides `ZeroGuardVerdict` (f64 + f32, NaN contamination check). Wire when toadStool runtime is available.

2. **dispatch_latency_ratio**: hotSpring computes F64:F32 dispatch ratio locally in `is_f64_throttled()`. toadStool's `TierCapability.dispatch_latency_ratio` provides the same data from runtime calibration probes. Can replace local computation.

3. **ProviderRegistry**: When hotSpring runs alongside toadStool, it can register as a provider for `gpu.dispatch.precision_brain` capabilities via the socket resolution system.

---

## To: coralReef Team

### Validated from Iter 30

- **45/46** shaders compile to native SM70 + SM86 SASS (unchanged from Iter 29)
- **`complex_f64`** remains sole failure (utility include, not standalone entry point)
- **FMA lowering** (`FmaPolicy::Separate`): validated in sovereign compile pipeline

### FMA Lowering Impact

coralReef Iter 30's `lower_fma` pass is the key unlock for F64Precise through
sovereign compilation. Previously, F64Precise used a WGSL-text path (skipping
sovereign SPIR-V). With `FmaPolicy::Separate`:
- FFma → FMul + FAdd (separate rounding)
- DFma → DMul + DAdd (separate rounding)
- Newton-Raphson sequences retain internal FMA (lowering runs before transcendental lowering)

When coral-driver dispatch matures, hotSpring's `create_pipeline_f64_precise()`
can route through sovereign compilation with `FmaPolicy::Separate`, gaining
native SASS performance for precision-critical domains (dielectric, eigensolve).

### Hardware Data

We have the Titan V + RTX 3090 test rig you need. Experiment 051 documents
the exact commands from your `docs/HARDWARE_TESTING.md` (`c84137c`). Data
capture will be executed when the test rig is available.

**Data coralReef needs from us:**
1. `nouveau_diag.log` — Titan V channel alloc EINVAL debugging (critical)
2. Environment data — kernel, driver, sysfs, firmware, dmesg
3. `nv_probe.log` — device detection on both GPUs
4. `uvm_diag.log` — UVM RM client status (RTX 3090 proprietary)
5. `parity_nouveau.log` — E2E dispatch attempt on Titan V

---

## Metrics

| Metric | v0.6.27 | v0.6.28 |
|--------|---------|---------|
| barraCuda pin | `59c8ec5` | `a012076` |
| toadStool | S144 | S145 |
| coralReef | Iter 29 | Iter 30 |
| PhysicsDomain variants | 7 | 12 |
| Lib tests | 842 | 842 |
| Clippy warnings | 0 | 0 |
| Sovereign compile (SM70) | 45/46 | 45/46 |
| Sovereign compile (SM86) | 45/46 | 45/46 |
| F64Precise sovereign path | Blocked (WGSL-text only) | **Unlocked** (FmaPolicy::Separate) |
