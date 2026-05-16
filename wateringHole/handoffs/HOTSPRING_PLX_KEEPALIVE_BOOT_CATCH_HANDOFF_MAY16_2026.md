# PLX Keepalive Boot-Catch + Event-Driven Evolution Handoff

**Date:** May 16, 2026
**Status:** VALIDATED — PLX keepalive catches K80 at boot, prevents D3cold

---

## Root Cause

The Tesla K80 was entering D3cold despite a `PlxKeepalive` implementation in
`toadstool-ember`. Two bugs conspired:

### Bug 1: PCI class code extraction (the real killer)

All discovery functions in `pcie_keepalive.rs` used `(class >> 8) & 0xFF_FFFF` to
extract a 24-bit value, then compared against 16-bit constants like `0x0604`. A PLX
PEX 8747 bridge with raw class register `0x060400ca` produced `class_code = 0x060400`,
which never equals `0x0604`. **Bridge and GPU class checks never matched.**

**Fix:** `(class >> 16) & 0xFFFF` extracts base_class:subclass (16-bit). New helper
`pci_base_subclass()` with named constants `PCI_CLASS_BRIDGE_PCI` (0x0604),
`PCI_CLASS_VGA` (0x0300), `PCI_CLASS_3D` (0x0302).

### Bug 2: Ancestry walk including PCI domain roots

The sysfs ancestry walk checked `name.contains(':')` to identify PCI BDFs, but
`pci0000:40` (a PCI domain root) also contains a colon. This polluted the bridge
list with non-device entries.

**Fix:** New `is_pci_bdf(name)` function checks for both `:` AND `.` — real BDFs
like `0000:49:00.0` have both; domain roots like `pci0000:40` have only a colon.

---

## Changes Made

### `pcie_keepalive.rs` (server)

- **Three-phase discovery**: class scan → GPU ancestry walk (handles dead config
  space at boot) → retry with 1s delays (3 attempts). Logs each discovered bridge
  with BDF + vendor.
- **`pci_base_subclass()` helper**: Correct class extraction with named constants.
- **`tokio::time::interval`** with `MissedTickBehavior::Skip`: First heartbeat at
  t=0, drift-free scheduling.
- **Activity-aware backpressure**: Uses ember's `ActivityTracker` via `OnceLock`
  instead of duplicated static + `record_pcie_activity()`. Skips synthetic heartbeat
  when real PCIe traffic occurred within one interval.
- **Dead-bridge recovery**: Ancestry walk adds parent bridges returning `0xFFFF`
  as keepalive targets and pins their power immediately.
- **17 new unit tests** covering class extraction, discovery, BDF validation,
  activity tracking, swap guard refcount.

### `plx_keepalive.rs` (ember)

- **`ActivityTracker`**: Shared `Arc<AtomicU64>` using `epoch_ms()` from
  `observation.rs` (no duplicated timestamp logic).
- **`is_pci_bdf()`**: Shared BDF format validator exported from ember.
- **`PlxKeepalive::with_activity_tracker()`**: Builder method for backpressure.
- **`tokio::time::interval`** with `MissedTickBehavior::Skip`.
- **`detect_bridge_chain`**: Fixed to use `is_pci_bdf()` instead of colon-only
  check.
- **7 new unit tests** for `ActivityTracker` and `is_pci_bdf`.

---

## Validation

Boot test (power cycle → service start → log check):

```
discovered PLX bridge via class scan  bdf=0000:4a:08.0  vendor=0x10b5
discovered PLX bridge via class scan  bdf=0000:4a:10.0  vendor=0x10b5
discovered PLX bridge via class scan  bdf=0000:49:00.0  vendor=0x10b5
pinned GPU bridge hierarchy at startup  bdf=0000:4c:00.0  bridges_pinned=4
pinned GPU bridge hierarchy at startup  bdf=0000:4b:00.0  bridges_pinned=4
PCIe bridge keepalive started  bridge_count=3  gpu_count=2  hierarchies_pinned=8
```

K80 remains alive with config space reads succeeding every 3 seconds.

---

## Key Files

| File | Role |
|------|------|
| `toadStool/crates/server/src/background/pcie_keepalive.rs` | Server keepalive loop, discovery, interval, activity tracking |
| `toadStool/crates/core/ember/src/plx_keepalive.rs` | Per-device keepalive, ActivityTracker, is_pci_bdf, bridge chain detection |
| `toadStool/crates/core/glowplug/src/plx.rs` | PlxGuardian device lifecycle integration |
| `toadStool/crates/core/ember/src/sysfs.rs` | pin_power, pin_bridge_hierarchy helpers |

---

## Upstream Patterns

- **`pci_base_subclass()`**: Any code parsing PCI class registers should use this
  instead of inline bit shifts. The 24-bit vs 16-bit confusion is a common trap.
- **`is_pci_bdf()`**: Use for sysfs path walking to avoid including PCI domain roots.
- **`ActivityTracker`**: Reusable for any periodic task that should back off when
  real activity is occurring. Clone-safe via `Arc`.
- **`tokio::time::interval` over `tokio::time::sleep`**: Immediate first tick,
  drift-free scheduling, `MissedTickBehavior::Skip` prevents burst-catching.
