# Experiment 167: Warm Handoff — Sovereign Compute via Nouveau HBM2 Training

**Date:** 2026-04-16
**Status:** Ready for execution (post-reboot)
**Target:** Titan V (GV100, 0000:03:00.0)
**Depends on:** Experiment 166 (Sovereign Boot Wiring)

## Objective

Validate the Phase 1 warm handoff path: use nouveau's native HBM2 training,
then steal the warm state for sovereign VFIO compute. This is the fastest path
to GPU compute — no proprietary drivers, no custom PHY firmware.

## Architecture

```
vfio-pci (cold)
  ↓  ember.swap → nouveau (PCI rescan, skip_sysfs_unbind)
nouveau (warm — PMU trains HBM2 during probe)
  ↓  ember.swap → vfio-pci (PCI rescan preserves warm BAR0/HBM2 state)
vfio-pci (warm)
  ↓  ember.vfio_fds → SCM_RIGHTS
  ↓  NvVfioComputeDevice::open_warm() → GPFIFO channel → dispatch
```

## Critical Fix: AdaptiveLifecycle Forwarding

The initial swap attempt (Apr 16) triggered a D-state because
`AdaptiveLifecycle` (the journal-aware wrapper around `NvidiaLifecycle`)
did not override `skip_sysfs_unbind()`, causing it to return the trait
default `false` instead of delegating to the inner lifecycle's `true`.

**Root cause:** `coral-ember/src/adaptive.rs` — `impl VendorLifecycle for
AdaptiveLifecycle` forwarded all methods except `skip_sysfs_unbind()`.

**Fix:** Added delegation:
```rust
fn skip_sysfs_unbind(&self) -> bool {
    self.inner.skip_sysfs_unbind()
}
```

**Test:** Two regression tests added to `adaptive::tests`:
- `delegates_skip_sysfs_unbind_false_by_default` — StubLifecycle returns false
- `delegates_skip_sysfs_unbind_from_inner` — VoltaStubLifecycle returns true

## Experiment Phases

### Phase 1: Pre-Swap Baseline
Read device state from glowplug + ember. Record personality, vram_alive,
domain count.

### Phase 2: Swap vfio → nouveau
Calls `glowplug.device_swap(bdf, "nouveau", true)`. This triggers:
1. `ember.swap` → `handle_swap_device_with_journal`
2. Lifecycle detection: `NvidiaLifecycle` (Volta+, skip_sysfs_unbind=true)
3. Wrapped in `AdaptiveLifecycle` (now properly forwards skip_sysfs_unbind)
4. `driver/unbind` skipped → PCI remove+rescan tears down vfio-pci
5. `driver_override` set to nouveau → re-enumeration binds nouveau
6. Nouveau probes GV100 → PMU falcon executes DEVINIT → HBM2 trained

Wait 15s for nouveau settle (Volta needs extra time for full DRM init).

### Phase 3: Verify Nouveau Probe
Read device status — confirm personality=nouveau, check vram_alive.

### Phase 4: Swap nouveau → vfio-pci
Same PCI rescan path. The key invariant: PCI remove+rescan should preserve
the warm BAR0 / HBM2 state that nouveau set up. The DRM subsystem is torn
down during re-enumeration (not via sysfs unbind, which D-states).

### Phase 5: Verify Warm State
Confirm ember re-acquired VFIO fds, device is on vfio-pci, and most
importantly: `vram_alive=true` indicating HBM2 training survived the
swap round-trip.

## Expected Results

| Check | Expected |
|-------|----------|
| glowplug reachable | true |
| ember reachable | true |
| swap vfio → nouveau (no D-state) | true |
| device on nouveau after swap | true |
| HBM2 trained by nouveau | true (PMU trains during probe) |
| swap nouveau → vfio-pci (PCI rescan) | true |
| ember re-acquired device | true |
| device back on vfio | true |
| VRAM alive after warm handoff | **TBD** (key experiment result) |

## Risks

- **VRAM may appear dead after rescan:** PCI remove+rescan re-enumerates the
  device which may power-cycle it, destroying HBM2 training. Mitigation:
  Volta's `reset_method` is disabled, so rescan shouldn't trigger FLR.
- **Nouveau may not fully train HBM2:** On GV100, nouveau relies on the PMU
  falcon to execute DEVINIT scripts. If the VBIOS is missing or PMU init
  fails, HBM2 won't be trained. Mitigation: the device was previously
  warm (booted with UEFI POST), so training state may persist in the
  memory controller even across driver swaps.
- **Domain health check may report false negatives:** Glowplug's passive
  health checker may not accurately reflect BAR0 state after a swap.
  The `vram_alive` field is the most reliable indicator.

## Next Steps (Phase B)

After confirming warm state survives the round-trip:

1. Add `coral-driver` with `vfio` feature as a dependency
2. Implement SCM_RIGHTS fd reception for `ember.vfio_fds`
3. Call `NvVfioComputeDevice::open_warm()` with received fds
4. Create warm GPFIFO channel + dispatch test shader
5. This completes the full sovereign compute path

## Usage

```bash
cargo run --release --bin exp167_warm_handoff -- --bdf 0000:03:00.0
```
