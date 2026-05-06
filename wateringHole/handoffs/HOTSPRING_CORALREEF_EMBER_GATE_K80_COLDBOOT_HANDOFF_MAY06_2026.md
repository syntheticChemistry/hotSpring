# hotSpring → coralReef Handoff: Ember Exclusive Device Gate + K80 Cold-Boot Sovereign

> **Date:** May 6, 2026
> **Experiments:** 179 (complete), 180 (complete), 181 (in progress)
> **coralReef iteration:** 89+ (pending push)
> **Status:** K80 cold-boot sovereign validated. Ember gate live. Push ready.

---

## Summary

Two major deliverables from this sprint:

1. **Ember Exclusive Device Gate** — When `coral-ember` holds a device, all direct hardware
   access paths in `coral-driver` are blocked and must route through ember's safety perimeter.
   Prevents accidental probing that kills fragile GPUs (K80 through PLX bridge).

2. **K80 Cold-Boot Sovereign** — Tesla K80 (GK210B) now boots cleanly to `vfio-pci` without
   PCIe link death. Root cause was redundant `drivers_probe` in udev rules triggering a reset
   through the PLX PEX 8747 switch.

---

## Ember Exclusive Device Gate

### Architecture

```
coral-driver call site (VFIO open, sysfs BAR0, oracle)
    │
    ▼
ember_gate::check_driver(bdf) / check_channel(bdf)
    │
    ├─ CORALREEF_EMBER_GATE=off → pass through (debugging escape hatch)
    ├─ ember socket unreachable → pass through (fail-open)
    └─ ember holds BDF → Err(DeviceHeldByEmber { bdf })
```

### Key Design Decisions

- **Fail-open**: If ember is not running or socket is unreachable, direct access proceeds
  normally. Standalone usage without ember is unaffected.
- **1-second timeout**: Socket query has a hard 1s read timeout to prevent blocking.
- **Gate points**: All direct hardware access paths are guarded:
  - `VfioDevice::open(bdf)` and `open_no_busmaster(bdf)`
  - `SysfsBar0::open(bdf, size)`
  - `Bar0Access::from_sysfs_device(sysfs_device)` (DRM path)
  - `nouveau_oracle::Bar0Rw::open(bdf)`
  - `mmu_oracle::Bar0Rw::open(bdf)`

### Files Modified/Created

- **NEW:** `crates/coral-driver/src/vfio/ember_gate.rs`
- `crates/coral-driver/src/vfio/mod.rs` — added `pub mod ember_gate`
- `crates/coral-driver/src/vfio/ember_client.rs` — `pub(super) fn default_socket()`
- `crates/coral-driver/src/error.rs` — `DeviceHeldByEmber` variants on both `DriverError` and `ChannelError`
- `crates/coral-driver/src/vfio/device/open.rs` — gate checks
- `crates/coral-driver/src/vfio/sysfs_bar0.rs` — gate check
- `crates/coral-driver/src/nv/bar0.rs` — gate check (DRM path, `#[cfg(feature = "vfio")]`)
- `crates/coral-driver/src/vfio/channel/nouveau_oracle.rs` — gate check
- `crates/coral-driver/src/vfio/channel/mmu_oracle/capture.rs` — gate check

### Tests

Unit tests in `ember_gate.rs`:
- `query_returns_false_when_socket_missing` — fail-open
- `query_returns_true_when_bdf_in_list` — mock socket, gate blocks
- `query_returns_false_when_bdf_not_in_list` — mock socket, gate passes
- `check_driver_errors_when_held` / `check_channel_returns_correct_variant` — error types

All tests pass without environment variable races (refactored to explicit socket paths).

---

## K80 Cold-Boot Sovereign

### Root Cause

The K80 (GK210B) sits behind a PLX PEX 8747 PCIe switch. This switch is extremely fragile:
- Any device reset (FLR, SBR) kills the PCIe link → config space reads all `0xFF` (`rev ff`)
- Once dead, requires full PSU power drain (capacitor discharge through PLX) to recover
- Linux `D3cold` power management can trigger implicit resets

### Boot Timeline (Before Fix)

1. Kernel boots, `vfio-pci.ids=10de:102d` claims K80 ✅
2. Udev rule fires: `echo 0000:4b:00.0 > /sys/bus/pci/drivers_probe` ❌
3. Re-probe triggers device reset through PLX bridge
4. PCIe link dies → `rev ff`

### Fix Applied

**`/etc/udev/rules.d/99-coralreef-vfio.rules`:**
- Removed `drivers_probe` writes for both K80 dies (`4b:00.0`, `4c:00.0`)
- Added `ATTR{d3cold_allowed}="0"` for both K80 dies
- Kept `driver_override`, `power/control=on`, `reset_method` clearing

**`/etc/kernelstub/configuration`:**
- `vfio-pci.ids=10de:1d81,10de:10f2,10de:102d` (Titan V + K80)

### Validation (Full Power Drain + Reboot)

- `lspci -s 4b:00.0` → `rev a1` (healthy)
- `lspci -s 4c:00.0` → `rev a1` (healthy)
- PLX bridge `49:00.0` → `rev ca` (healthy)
- PCIe link: Gen3 x16 (full speed)
- `/dev/vfio/35`, `/dev/vfio/36` → present, `0666` permissions
- dmesg: no `drivers_probe` writes, no reset events

---

## NvidiaKeplerLifecycle Changes

In `coral-ember/src/vendor_lifecycle/nvidia.rs`:

- `skip_sysfs_unbind() → true` — prevents direct sysfs `driver/unbind` writes that cause
  D-state hangs through the PLX bridge
- `rebind_strategy()` for DRM targets → `RebindStrategy::PciRescan` (was `SimpleWithRescanFallback`)

These changes route all K80 driver management through ember's process-isolated, timeout-guarded
sysfs write helpers.

---

## Isolation D-State Fix

In `coral-driver/src/vfio/isolation.rs`:

`fork_isolated_raw` previously blocked indefinitely on `waitpid` if the child entered D-state
(uninterruptible sleep from sysfs writes to a dead PCIe device). Now uses a non-blocking reap
loop with 2-second timeout, abandoning the zombie if it remains stuck.

---

## Remaining Blockers

| GPU | Status | Blocker |
|-----|--------|---------|
| RTX 5060 (SM120) | ✅ 8/8 dispatch proven | None — sovereign compute live |
| Titan V (GV100) | 🔴 Blocked | SEC2/ACR boot chain, no PMU firmware in nouveau |
| Tesla K80 (GK210B) | 🟡 Cold-boot OK | PGOB: PRI ring has 0 GPCs enrolled, GPC power gating |

### K80 Next Steps

1. PGOB (Power Gating Off Block) — need to ungate GPCs so FECS can address them
2. Options: PMU firmware approach (complex), or direct PRI ring GPC enrollment
3. `scripts/gpu-solve/titan-v-module-swap.sh` available for nvidia-470 testing path

### Titan V Next Steps

1. SEC2/ACR boot chain requires PMU firmware (not available in nouveau for GV100)
2. Warm handoff path (nouveau → vfio) proven but FECS cannot exit HS-mode halt
3. Long-term: pure Rust ACR implementation or RE of firmware protocol

---

## For Other Teams

- **coralReef push pending** — 38 modified + 1 new file, needs rebase on BTSP Phase 3
- **Ember gate is fail-open** — no behavior change unless ember is actively running
- **`CORALREEF_EMBER_GATE=off`** — escape hatch for debugging
- **udev rules synced** to `scripts/boot/99-coralreef-vfio.rules` in both hotSpring and coralReef
