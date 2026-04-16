# Experiment 166: Sovereign Boot Wiring — Training Capture → Recipe Replay → Unbind Fix

**Date**: 2026-04-16
**GPU**: NVIDIA Titan V (GV100, SM70, 10de:1d81), Tesla K80 (GK210, SM37)
**Driver**: None (pure Rust + VFIO + firmware blobs)
**Depends on**: Exp165 (SovereignInit Pipeline)
**Status**: **IN PROGRESS** — unbind fix coded, needs reboot to deploy

## Objective

Wire the full sovereign boot pipeline end-to-end: training capture → recipe
replay → sovereign init. Diagnose and fix the system lock during sovereign
boot that was caused by stale daemon binaries missing fork isolation.

## Results

### 1. Training Capture (SUCCESS)

Captured GV100 training recipe via `coralctl capture training 0000:03:00.0 --driver nouveau`:

```
Chip:           gv100
Warm driver:    nouveau
Total writes:   6,241
Domains:        10

FBPA0              :     2 registers  (0x009a0210 - 0x009a0584)
FBPA1              :     3 registers  (0x009a402c - 0x009a4540)
LTC0               :    60 registers  (0x0017e204 - 0x0017ebb0)
LTC1               :  2048 registers  (0x00180000 - 0x00181ffc)
LTC3               :  2048 registers  (0x00184000 - 0x00185ffc)
LTC5               :  2048 registers  (0x00188000 - 0x00189ffc)
PCLOCK             :     2 registers  (0x00137654 - 0x00137658)
PFB                :    14 registers  (0x00100a24 - 0x00100e48)
PFB_NISO           :     8 registers  (0x00100c10 - 0x00100cd8)
PMU                :     8 registers  (0x0010a024 - 0x0010aab8)
```

Recipe saved to `/var/lib/coralreef/training/gv100.json` (363KB).

**Key finding**: 98.4% of writes (6,144/6,241) are LTC tag RAM — L2 cache
configuration, not actual HBM2 PHY training. The PHY training happens
internally on the PMU falcon via firmware, not through host BAR0 writes.

### 2. Sovereign Init on Cold GPU (PARTIAL)

After recipe replay, the pipeline halted cleanly at `wake_mc`:

```
[  OK ] probe      PMC_ENABLE=0x5fecdff1 (23 bits) warm=true devinit=true
[FAIL ] hbm2       writes: 6241 applied, 1 failed (706076us)
[  OK ] pmc        writes: 6 applied, 0 failed (57505us)
[  OK ] topology   GPC:1 SM:16 FBP:12 PBDMA:4
[  OK ] pfb        writes: 2 applied, 0 failed (51us)
[FAIL ] wake_mc    writes: 1 applied, 1 failed (15254us)
```

**Root cause**: The recipe diff captures *results* of training (LTC tags,
FBPA config, PFB settings) but not the actual HBM2 PHY initialization
sequence. The PHY training runs as PMU falcon firmware — a microcode
program that drives the HBM2 I/O and sets timing parameters. The host
sees only the aftermath in BAR0 register space.

### 3. System Lock Diagnosis & Fix (COMPLETED)

The initial sovereign boot attempt locked the system because the installed
`coral-ember` binary (33MB, dated Apr 15 15:14) was outdated and lacked
per-stage fork isolation. The freshly built binary (4MB) had the fix.

**Fix**: Rebuilt and installed both daemons. Confirmed fork isolation works —
subsequent sovereign init halted cleanly without system lock.

### 4. Unbind D-state Fix (CODED, NEEDS REBOOT)

Driver unbind (both `vfio-pci/unbind` and `nouveau/unbind`) D-states on
the Titan V, blocking the swap path. Root cause: `NvidiaLifecycle` used
`SimpleBind` strategy and didn't skip sysfs unbind, unlike the Kepler
lifecycle which correctly uses `PciRescan` + `skip_sysfs_unbind`.

**Fix applied to** `coral-ember/src/vendor_lifecycle/nvidia.rs`:
- `NvidiaLifecycle.skip_sysfs_unbind()` → `true`
- `NvidiaLifecycle.rebind_strategy()` → `SimpleWithRescanFallback` for DRM targets
- `bind_vfio()` → PCI remove+rescan when `skip_sysfs_unbind` is true and old driver bound

All 179 coral-ember lib tests pass. Binaries rebuilt and installed.
Stuck processes from current session require reboot to clear.

### 5. mmiotrace Analysis

Full nouveau init sequence from mmiotrace (1.3MB, 7,364 BAR0 writes):

```
Init order (first write per domain):
  +   0.0ms  PMC       — PMC enable, interrupt masks
  +   0.0ms  PRAMIN    — falcon firmware upload channel
  +   0.0ms  PPCI      — PCI config space setup
  + 116.9ms  PFB_PRI   — framebuffer PRI config
  + 116.9ms  PFB       — framebuffer controller
  + 148.5ms  [0x04xx]  — PTIMER
  + 148.6ms  PCONN     — connector/display port
  + 153.2ms  HOST      — host interface
  + 193.9ms  DISP_SOR  — display serial output
  + 194.3ms  PDISPLAY  — display engine
```

The recipe (cold-vs-warm BAR0 diff) captures 6,241 changes, but nouveau
makes 17,905 writes total and 14,555 reads. The diff misses the PMC init,
PRAMIN firmware upload, and PFB controller setup that are identical in
both cold and warm snapshots.

## Architecture Insight

```
                Training Capture Flow
┌──────────┐    ┌──────────┐    ┌─────────┐
│ Cold BAR0│───→│ Diff     │───→│ Recipe  │
│ Snapshot │    │ Engine   │    │ (6241)  │
└──────────┘    └──────────┘    └─────────┘
     ↑               ↑              │
     │          ┌──────────┐        │ Replay
     │          │ Warm BAR0│        ↓
     │          │ Snapshot │   ┌─────────┐
     │          └──────────┘   │ Sov Init│ ← Only applies diff
     │               ↑         └─────────┘   (LTC tags + FBPA)
     │          ┌──────────┐        │
     │          │ nouveau  │        ↓ Missing!
     │          │ init     │   ┌─────────┐
     │          │ (17905   │   │ PMU     │ ← Falcon firmware
     │          │  writes) │   │ DEVINIT │   does actual PHY
     │          └──────────┘   └─────────┘   training
```

The diff-based recipe misses the PMU falcon's internal memory training.
Three paths forward:

1. **PMU DEVINIT firmware upload** (Phase 2 in init_hbm2): Upload the VBIOS
   DEVINIT app to the PMU falcon and let it train memory natively. Requires
   PROM or sysfs VBIOS access while on vfio-pci.

2. **Warm handoff**: Bind nouveau → let it train → swap to vfio (PCI
   remove+rescan preserving warm state) → sovereign init skips HBM2.
   Now possible with the unbind fix.

3. **mmiotrace-ordered replay**: Instead of the BAR0 diff, replay the
   full 17,905-write sequence from the mmiotrace in order, including
   PMC/PRAMIN writes. This is the "full init recipe" approach.

### 6. Swap Orchestrator Unbind Fix (Apr 16 — Second Pass)

After pulling upstream coralReef and rebooting, the swap path D-stated again.
Root cause: `handle_swap_device_with_journal` (swap/mod.rs:192–200) ALWAYS
writes to `driver/unbind` regardless of `VendorLifecycle::skip_sysfs_unbind()`.
The trait method existed on NvidiaLifecycle locally but was never added to the
upstream VendorLifecycle trait, and the swap orchestrator never consulted it.

**Three-part fix:**
1. Added `skip_sysfs_unbind()` to the `VendorLifecycle` trait (default: false)
2. `handle_swap_device` skips `driver/unbind` write when `skip_sysfs_unbind()` is true
3. `bind_vfio` and `bind_native` use PCI rescan directly when `skip_sysfs_unbind`
   is true and old driver is still bound (simple bind would fail anyway)

All 172 coral-ember lib tests pass. Binaries staged in /tmp for post-reboot install.

## AdaptiveLifecycle Forwarding Bug (Apr 16 — Third Pass)

After the second reboot, `ember.swap` STILL D-stated on the Titan V.
Binary strings confirmed both `"skipping sysfs unbind"` and `"unbinding
current driver"` were present, so the fix was compiled in.

**Root cause:** `AdaptiveLifecycle` (journal-aware wrapper) implements
`VendorLifecycle` but did NOT override `skip_sysfs_unbind()`. The trait
default returns `false`, so the swap orchestrator always took the
`driver/unbind` branch even though the inner `NvidiaLifecycle` returns `true`.

The swap handler ALWAYS creates an `AdaptiveLifecycle` when a journal
exists (which it always does), so `skip_sysfs_unbind()` always returned
`false` regardless of the inner lifecycle.

**Fix:** Added one line to `impl VendorLifecycle for AdaptiveLifecycle`:
```rust
fn skip_sysfs_unbind(&self) -> bool {
    self.inner.skip_sysfs_unbind()
}
```

**Tests:** Two regression tests in `adaptive::tests`:
- `delegates_skip_sysfs_unbind_false_by_default`
- `delegates_skip_sysfs_unbind_from_inner`

All 174 coral-ember unit tests + 8 swap integration + 6 lifecycle pass.
Fixed binary installed at `/usr/local/bin/coral-ember`.

## Next Steps

1. Reboot to clear D-state zombies from the failed swap attempt
2. Run exp167_warm_handoff: vfio → nouveau (HBM2) → vfio → verify warm state
3. Phase B: wire VFIO fd passing + open_warm() + shader dispatch
4. Capture K80 (GK210/Kepler) recipe for cross-gen comparison
5. Build MMIO Gateway handlers in ember for sovereign init (Phase 2b)

## Files Modified

- `coralReef/crates/coral-ember/src/vendor_lifecycle/types.rs` — added `skip_sysfs_unbind()` to trait
- `coralReef/crates/coral-ember/src/vendor_lifecycle/nvidia.rs` — NvidiaLifecycle overrides: skip_sysfs_unbind=true, SimpleWithRescanFallback for DRM, settle 15s
- `coralReef/crates/coral-ember/src/swap/mod.rs` — skip driver/unbind when lifecycle says so
- `coralReef/crates/coral-ember/src/swap/swap_bind.rs` — bind_vfio + bind_native PCI rescan when old driver still bound
- `coralReef/crates/coral-ember/src/vendor_lifecycle/tests.rs` — test updates for new behavior
- `coralReef/crates/coral-ember/src/adaptive.rs` — forward skip_sysfs_unbind to inner lifecycle + 2 regression tests
