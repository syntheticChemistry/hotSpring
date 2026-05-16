# hotSpring → Warm VFIO Dispatch Evolution Handoff

**Date:** May 12, 2026
**GAP:** GAP-HS-095 (continued)
**Trigger:** Cold vs warm init mismatch discovered; kernel-module evolution roadmap
**Status:** Warm API wired; hardware validation blocked on ember subprocess crash

---

## Critical Discovery: Cold vs Warm Init Mismatch

`GpuContext::from_vfio()` called `NvVfioComputeDevice::open()` which runs
`seq.cold_init(&bar0)` — resetting PLLs, clocks, PGRAPH, and FECS firmware.
This destroys the warm state carefully preserved by `coralctl warm-catch`
(patched nouveau → FECS loaded → vfio-pci swap with reset_method disabled).

The warm open APIs (`open_warm_direct`, `open_warm_legacy`) existed in
coral-driver but had no `GpuContext` wrapper in coral-gpu.

## What Was Done

### 1. Warm API in coral-gpu (upstream fix)

Added to `coral-gpu/src/context.rs`:

| Method | Target | Behavior |
|--------|--------|----------|
| `from_vfio_warm(bdf)` | Titan V, general | Auto SM detect, `open_warm_direct`, deferred bus-master |
| `from_vfio_warm_with_sm(bdf, sm)` | Titan V, explicit SM | Same as above with caller SM |
| `from_vfio_warm_legacy(bdf, sm)` | K80 (PLX 8747) | Legacy VFIO group path, no iommufd FLR |

### 2. Validation Binary Updated

`validate_vfio_sovereign` now:
- **Defaults to warm mode** (expected state after `coralctl warm-catch`)
- `--cold` flag for cold-init path (testing/bare-metal only)
- `--legacy` flag for explicit legacy VFIO path
- K80 targets automatically use `WarmLegacy` mode
- Titan V targets use `Warm` mode

### 3. Scenario Updated

`s_vfio_dispatch` (`GpuCompute`-track Live-tier):
- `VfioTarget` gains `use_legacy: bool` field
- K80 targets: `from_vfio_warm_legacy()` (safe for PLX bridge)
- Titan V target: `from_vfio_warm_with_sm()` (deferred bus-master)

### 4. Hardware Validation Attempted

**Result: BLOCKED**

```
VFIO open (warm): FAILED — driver error: command submission failed:
FECS unreachable (PRI timeout) — GPU is cold
```

Root cause: The `coral-ember` subprocess spawned by `coral-glowplug` cylinders
crashes immediately on startup, leaving zombie `[coral-ember] <defunct>` processes.
`coralctl warm-fecs` cannot relay swap commands without a live ember. GPUs are
in cold state (BAR0 reads return 0xbadf sentinel / 0xffffffff).

**Upstream action needed:** Fix ember subprocess startup in coral-glowplug
cylinder spawning. The installed `/usr/local/bin/coral-ember` may be stale
vs the source. Rebuilding and reinstalling the binaries should resolve this.

## Upstream Gaps Summary

### Immediate (blocks hardware validation)

| Gap | Owner | Description |
|-----|-------|-------------|
| Ember subprocess crash | coralReef (coral-ember) | Cylinder-spawned ember goes `<defunct>` immediately; blocks warm-catch pipeline |
| `coral-gpu` had no warm API | coralReef (coral-gpu) | **FIXED** — `from_vfio_warm*` methods added |

### Phase C/D (blocks IPC path for other springs)

| Gap | Owner | Description |
|-----|-------|-------------|
| `compute.dispatch.execute` miswiring | toadStool | Handler forwards to coralReef method that doesn't exist |
| `VfioResourceHandle` metadata-only | toadStool | No real VFIO fds or compute contexts |
| `enumerate_all()` DRM-only | coralReef | VFIO GPUs invisible to unified enumeration |

### Long-term (kernel-module evolution)

| Stage | Description |
|-------|-------------|
| **Now:** VFIO userspace | Warm boot proves dispatch pipeline (WGSL→SASS→GPFIFO→DMA→readback) in pure Rust |
| **Next:** `coral-gpu-kmod` | Graduate `vfio_compute/` init+dispatch from userspace MMIO into kernel module |
| **Then:** `/dev/coral-gpu{N}` | Module binds PCI device directly; GPU visible without VFIO isolation |
| **Pattern** | hotSpring solves locally on hardware → hands patterns upstream → primals absorb and abstract → hotSpring resolves with their new abstraction |

Existing `coral-kmod` (`/dev/coral-rm`) already proves the kernel-module pattern
for Blackwell RM proxy. The evolution is: expand from "RM proxy for nvidia driver"
to "standalone GPU compute driver" that replaces both nvidia and VFIO.

## Metrics

| Metric | Value |
|--------|-------|
| Library tests | 590/590 pass |
| Validation scenarios | 17 (14 default + 3 feature-gated) |
| Validation binaries | 167 |
| Warm API methods added to coral-gpu | 3 |
| Clippy warnings | 0 |
| Hardware dispatch proven | RTX 5060 only (Titan V/K80 blocked on ember) |

## Next Steps

1. **Fix ember subprocess** — rebuild coral-ember + coral-glowplug from source, reinstall
2. **Re-run warm-catch** — `coralctl warm-fecs 0000:02:00.0 --settle 14`
3. **Run validation** — `validate_vfio_sovereign` with warm-caught GPUs
4. **Hand dispatch results upstream** — first sovereign dispatch readback from Titan V and K80
