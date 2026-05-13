# hotSpring → VFIO Sovereign Dispatch Wiring Handoff

**Date:** May 12, 2026
**GAP:** GAP-HS-095
**Trigger:** wgpu cannot enumerate VFIO-bound GPUs; toadStool IPC dispatch miswired
**Status:** Option A (in-process VFIO dispatch) wired; Option B (IPC path) blocked upstream

---

## Summary

Three dispatch paths exist for sovereign GPU compute, but none were fully connected
for VFIO-bound GPUs (Titan V, K80). This handoff documents the immediate fix
(Option A: enable `coral-gpu` VFIO feature for in-process dispatch) and the
upstream gaps blocking the IPC path (Option B: toadStool Phase C/D).

## The Problem

| Path | Status | Blocker |
|------|--------|---------|
| wgpu | Dead end | No Vulkan ICD for VFIO GPUs; `wgpu::Instance::enumerate_adapters()` returns empty |
| coral-gpu VFIO | **NOW WIRED** | `vfio` feature was not enabled in hotSpring's dependency |
| toadStool IPC | Broken | `compute.dispatch.submit` forwards to `compute.dispatch.execute` which coralReef doesn't serve |

## What Was Done (Option A — Immediate Fix)

### 1. Cargo.toml Feature Enable

```toml
# Before
coral-gpu = { path = "...", optional = true }

# After
coral-gpu = { path = "...", optional = true, features = ["vfio"] }
```

This unlocks:
- `GpuContext::from_vfio(bdf)` — auto-detect SM from sysfs, open NvVfioComputeDevice
- `GpuContext::from_vfio_with_sm(bdf, sm)` — explicit SM version
- `GpuContext::auto()` VFIO preference — selects VFIO when available
- `discover_vfio_nvidia_bdf()` — sysfs scan for NVIDIA on vfio-pci

### 2. Validation Binary: `validate_vfio_sovereign`

New binary exercising VFIO dispatch on all biomeGate GPUs:

- **Phase 1:** VFIO GPU discovery via sysfs
- **Phase 2:** Per-GPU validation loop:
  - Titan V (SM70) at `0000:02:00.0`
  - K80 die0 (SM37) at `0000:4b:00.0`
  - K80 die1 (SM37) at `0000:4c:00.0`
- **Per-GPU tests:**
  - T1: WGSL compile (write_constant) — SASS binary generation
  - T2: Dispatch + readback — sentinel → dispatch → readback → verify `42`
  - T3: QCD shader compile (wilson_plaquette_f64)
  - T4: QCD shader compile (su3_gauge_force_f64)
- **CLI:** `--bdf 02:00.0 --sm 70` for targeting specific hardware

Usage:
```sh
cargo run --release --features sovereign-dispatch --bin validate_vfio_sovereign
```

### 3. Validation Scenario: `s_vfio_dispatch`

New `GpuCompute`-track, `Live`-tier scenario in the harness:

- Checks VFIO driver presence
- Per-target: sysfs presence → vfio-pci binding → device open → compile → dispatch → readback
- Graceful degradation without `sovereign-dispatch` feature
- Registered in `build_registry()` — 17 total scenarios

### 4. Compilation Verification

- `cargo check --lib` — clean (default features)
- `cargo check --lib --features sovereign-dispatch` — clean
- `cargo check --bin validate_vfio_sovereign --features sovereign-dispatch` — clean
- `cargo test --lib` — 590/590 pass

## Hardware State (biomeGate)

| GPU | BDF | SM | Warm Boot | FECS | GPCs | VFIO Dispatch Ready |
|-----|-----|----|-----------|------|------|---------------------|
| Titan V (GV100) | 02:00.0 | 70 | Binary-patched nouveau (Exp 190) | RUNNING (0x0c060006) | 1 active | **YES** — `from_vfio("0000:02:00.0")` |
| K80 die0 (GK210) | 4b:00.0 | 37 | Binary-patched nouveau (Exp 190) | RUNNING (0x00060005) | 5 active | **YES** — `from_vfio("0000:4b:00.0")` |
| K80 die1 (GK210) | 4c:00.0 | 37 | Not yet warm-caught | — | — | Pending warm-catch |
| RTX 5060 (GB206) | — | 120 | nvidia shared driver | All engines | — | Already PROVEN (DRM/wgpu) |

## Upstream Gaps — Phase C/D (Option B: IPC Path for Other Springs)

These gaps block **other springs** from dispatching on VFIO GPUs via toadStool IPC.
hotSpring's Option A bypass is not available to springs that use `compute.dispatch.submit`.

### Gap 1: toadStool `compute.dispatch.execute` miswiring

**Where:** toadStool's `compute.dispatch.submit` handler
**Problem:** Handler forwards to `compute.dispatch.execute` on coralReef, but
coralReef's `REGISTERED_METHODS` does not include that method.
**Fix:** Phase C should absorb `NvVfioComputeDevice` from `coral-driver` into
toadStool so dispatch executes locally without IPC forwarding.

### Gap 2: toadStool `VfioResourceHandle` is metadata-only

**Where:** `crates/core/ember/src/vfio_handle.rs`
**Problem:** Stores BDF, IOMMU group, and resource token but does not open real
VFIO file descriptors or create compute contexts. Cannot dispatch.
**Fix:** Phase D should integrate `NvVfioComputeDevice::open()` stack:
VFIO group fd → device fd → BAR0 mmap → DMA context → GPFIFO → compute dispatch.

### Gap 3: coralReef `enumerate_all()` is DRM-only

**Where:** `coral-gpu/src/context.rs` — `enumerate_all()`
**Problem:** Even with `vfio` feature enabled, enumeration uses DRM render nodes
(`/dev/dri/renderD*`). VFIO-bound GPUs have no DRM node and are invisible.
**Fix:** Either create `enumerate_all_with_vfio()` or merge VFIO discovery
into the unified enumeration path so `auto()` sees all available backends.

### Gap 4: Dispatch ownership convergence

**Problem:** hotSpring uses in-process `coral-gpu` VFIO (Option A). Other springs
expect `compute.dispatch.submit` via toadStool IPC (Option B). Both paths must
converge post-Phase D so dispatch works uniformly for all springs.
**Fix:** After Phase C absorbs `coral-driver` and Phase D wires local dispatch,
toadStool's `compute.dispatch.submit` should execute in-process using the
absorbed VFIO stack — eliminating the forwarding to coralReef entirely.

## Metrics

| Metric | Before | After |
|--------|--------|-------|
| VFIO dispatch paths for Titan V/K80 | 0 | 1 (in-process coral-gpu) |
| Validation binaries | 155 | 156 |
| Validation scenarios | 16 | 17 |
| Library tests | 590 | 590 |
| Clippy warnings | 0 | 0 |

## Next Steps

1. **Run `validate_vfio_sovereign` on warm-caught GPUs** — prove E2E dispatch
2. **Capture structured results** — first dispatch readback values from Titan V and K80
3. **Hand upstream Phase C/D gaps** — this handoff is the source of truth
4. **Cold boot validation** — once warm dispatch proven, extend to cold boot sentinel
