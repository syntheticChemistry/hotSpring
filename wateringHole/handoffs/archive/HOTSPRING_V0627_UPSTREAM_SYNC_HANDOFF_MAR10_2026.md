SPDX-License-Identifier: AGPL-3.0-only

# hotSpring v0.6.27 — Upstream Sync Handoff

**Date:** March 10, 2026
**From:** hotSpring (validation primal)
**To:** toadStool, barraCuda, coralReef teams
**Covers:** v0.6.26 → v0.6.27 (barraCuda `83aa08a` → `59c8ec5`, toadStool S142 → S144)

---

## Executive Summary

- **barraCuda v0.3.4 (`59c8ec5`) integrated**: Fp64Strategy routing fix validated — 842/842 tests pass. DF64 reduce ops (SumReduceF64, VarianceReduceF64, NormReduceF64, ProdReduceF64) now correctly call `.df64()` on Hybrid devices. hotSpring uses these in ~20 call sites across `md/`, `lattice/`, `physics/`
- **toadStool S144 acknowledged**: NVVM poisoning absorption into `nvvm_safety.rs` confirmed (credited to "hotSpring v0.6.25 handoff"). PCIe switch topology, `gpu_guards`, and `compile_wgsl_multi` noted in hotSpring source comments for future integration
- **No breaking API changes**: `cargo check --all-targets` clean on first attempt. All 842 lib tests pass, zero clippy warnings, sovereign compile validation 45/46 unchanged
- **Handoff lifecycle**: The v0.6.26 upstream rewire handoff (`HOTSPRING_V0626_UPSTREAM_REWIRE_HANDOFF_MAR10_2026.md`) is now archived — its toadStool S142 content is superseded by S144's direct absorption

---

## Part 1: barraCuda v0.3.4 Validation

### Fp64Strategy Routing Fix (Critical)

The most impactful change. In v0.3.3, `SumReduceF64` and related reduce operations called `.f64()` on Hybrid devices — producing incorrect results when the device should use DF64 shader compilation. v0.3.4 correctly calls `.df64()`.

**hotSpring impact**: ~20 call sites across:
- `md/simulation/mod.rs`, `md/simulation/verlet.rs` (kinetic energy reduction)
- `md/observables/transport_gpu.rs` (Green-Kubo transport coefficients)
- `md/celllist.rs` (neighbor list statistics)
- `lattice/cg.rs`, `lattice/gpu_hmc/resident_cg_pipelines.rs` (CG solver dot products)
- Multiple validation binaries

**Validation**: 842/842 lib tests pass. No regressions. DF64 reduce results confirmed correct.

### Other v0.3.4 Changes

| Feature | Impact on hotSpring |
|---------|-------------------|
| PCIe topology (`PcieBridge`, `PcieLinkInfo`) | Not directly used — hotSpring's `DevicePair` uses `BandwidthTier::detect_from_adapter_name()`. Future: delegate to sysfs-probed topology |
| VRAM quota (`QuotaTracker`) | Not directly used — future: integrate for proactive OOM prevention in multi-GPU pools |
| `BglBuilder` | Not directly used — future: replace manual BGL construction |
| `discover_coralreef()` → `discover_shader_compiler()` rename | No impact — hotSpring never called this internal function |
| License headers AGPL-3.0-or-later → AGPL-3.0-only | Aligned with hotSpring's existing AGPL-3.0-only |

---

## Part 2: toadStool S144 Integration Points

### NVVM Poisoning Absorption (Confirmed)

toadStool S144 created `crates/runtime/universal/src/backends/nvvm_safety.rs` with:
- `NvvmPoisoningRisk` — mirrors hotSpring's `nvvm_transcendental_risk` field
- `PrecisionTier` — mirrors hotSpring's `PrecisionTier` enum
- `TierCapability` — mirrors hotSpring's `TierCapability` struct

**Credited to**: "hotSpring v0.6.25 handoff" in the source comments.

**hotSpring response**: Updated `HardwareCalibration` doc comments to note that when toadStool's runtime layer is integrated, `HardwareCalibration` can delegate to upstream's native NVVM defense rather than maintaining its own probe logic.

### PCIe Switch Topology

toadStool S144 evolved `PcieTransport` into full switch-level topology:
- `PciBridge`, `GpuPairTopology`, `PcieTopologyGraph` (sysfs-probed)
- `WorkloadRouter::route_multi_gpu()` for topology-aware multi-GPU placement
- `MultiGpuPlacement` struct

**hotSpring response**: Updated `DevicePair` and `WorkloadPlanner` doc comments from S142 to S144 references. When API stabilizes, `DevicePair` should delegate to `PcieTopologyGraph` and `WorkloadRouter`.

### gpu_guards Module

toadStool S144 added `crates/testing/src/gpu_guards.rs`:
- `is_wgpu_safe()` — checks if wgpu can be safely used
- `detect_nvidia_proprietary()` — detects NVIDIA proprietary driver

**hotSpring response**: Noted in `PrecisionBrain` doc comments. hotSpring's probe-based approach (try-compile, catch_unwind on failure) is compatible with toadStool's guard-based approach.

### Multi-Device coralReef Compilation

toadStool S144 added multi-device sovereign compilation:
- `MultiDeviceCompileRequest`, `DeviceTarget`
- `compile_wgsl_multi` function
- `shader.compile.wgsl.multi` JSON-RPC method

**hotSpring response**: Noted in `PrecisionBrain` doc comments. When toadStool's runtime is integrated, sovereign detection should delegate to toadStool's `NvvmPoisoningRisk` assessment and use `compile_wgsl_multi` for heterogeneous dispatch.

---

## Part 3: Sovereign Compilation Status

Unchanged from v0.6.26:
- **45/46 shaders** compile to native SM70/SM86 SASS (coralReef Iter 29)
- **12/12 NVVM bypass** patterns pass (all 3 poisoning patterns × 6 GPU targets)
- **1 compilation gap**: `complex_f64` shader fails in `CoralReefDevice`
- **DRM dispatch**: compile-ready, dispatch-blocked (NVIDIA UVM pending, AMD E2E ready)

---

## Part 4: What Upstream Should Evolve Next

### barraCuda

1. **Absorb `HardwareCalibration` pattern** — hotSpring's per-tier probing could become a `barracuda::calibration` module. The `probe → record → route` pattern is spring-agnostic
2. **Absorb `PrecisionBrain` as `Router`** — domain-aware routing from calibration data is reusable across any spring doing heterogeneous GPU compute
3. **Absorb tolerance pattern** — 170+ centralized constants with physical justification (`tolerances/` module tree)

### toadStool

1. **Stabilize `PcieTopologyGraph` API** — hotSpring's `DevicePair` is ready to delegate when the API is stable
2. **Expose `nvvm_transcendental_risk` in runtime discovery** — springs need this at probe time, not just test time
3. **Integrate `compile_wgsl_multi` with `PrecisionBrain`** — per-GPU tier routing combined with multi-device compilation enables optimal shader selection across heterogeneous hardware

### coralReef

1. **Fix `complex_f64` compilation** — last remaining gap (1/46)
2. **NVIDIA DRM dispatch** — compile path works, dispatch needs UVM integration
3. **Iteration 30+ targets**: SM70 `vec3<f64>` encoding, f64 `log2` edge case, AMD `Discriminant` encoding

---

## Appendix: Files Changed in v0.6.27

| File | Change |
|------|--------|
| `Cargo.toml` | rev `83aa08a` → `59c8ec5`, version `0.6.26` → `0.6.27` |
| `src/device_pair.rs` | toadStool S142 → S144, `PcieTopologyGraph` + `WorkloadRouter` refs |
| `src/workload_planner.rs` | toadStool S142 → S144, `MultiGpuPlacement` ref |
| `src/hardware_calibration.rs` | Iter 28 → 29, note `nvvm_safety.rs` absorption + `gpu_guards` |
| `src/precision_brain.rs` | Iter 28 → 29, note `compile_wgsl_multi` + `gpu_guards` |
| `CHANGELOG.md` | v0.6.27 entry |
| All docs | Version bumps to v0.6.27, `59c8ec5`, S144 |

*AGPL-3.0-only*
