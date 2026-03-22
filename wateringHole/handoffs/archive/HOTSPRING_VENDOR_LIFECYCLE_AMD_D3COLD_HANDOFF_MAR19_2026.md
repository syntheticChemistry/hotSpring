# Handoff: VendorLifecycle Trait + AMD D3cold Discovery

**Date:** March 19, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** VendorLifecycle trait, AMD Vega 20 D3cold analysis, vendor-agnostic swap abstraction, Intel Xe stubs, PCI remove/rescan rebind strategy

---

## Executive Summary

Empirical testing of the AMD Radeon VII (Vega 20 / GFX906) through the VFIO swap pipeline revealed a **critical vendor-specific lifecycle difference**: AMD Vega 20 enters D3cold when vfio-pci performs a bus reset during unbind, killing SMU firmware state. NVIDIA GV100 survives the same operation with HBM2 state intact.

This led to:

1. **`VendorLifecycle` trait** in `coral-ember` — vendor-specific hooks injected into the swap orchestrator
2. **AMD fix**: disable `reset_method` before vfio-pci unbind + PCI remove/rescan for native rebind
3. **Intel Xe/Arc stubs** with conservative defaults
4. **Personality system** expanded with `Xe` and `I915` variants
5. **Clean AMD round-trip validated**: amdgpu → vfio-pci → amdgpu with full DRM/VRAM/hwmon recovery

---

## Part 1: The AMD D3cold Problem

### What Happened

```
amdgpu → vfio-pci:  OK (ember acquires VFIO fds, BAR0 mapped)
vfio-pci → amdgpu:  FAILED
  - vfio-pci unbind triggers bus reset (only reset method on Vega 20)
  - Bus reset puts card into D3cold
  - SMU firmware loses state, cannot re-train HBM2
  - dmesg: "trn=2 ACK should not assert" (SMU training failure)
  - Card unresponsive (rev ff), requires full system reboot
```

### Root Cause

| Property | NVIDIA GV100 | AMD Vega 20 |
|----------|-------------|-------------|
| Reset methods | `bus` | `bus` (only) |
| Bus reset effect | D0 maintained, HBM2 alive | D3cold, SMU dead |
| FLR support | No | No |
| Recovery from bus reset | Automatic | Requires power cycle |

### The Fix (Validated)

Three-step mitigation, now automated in `VendorLifecycle`:

1. **Disable reset_method**: `echo "" > /sys/bus/pci/devices/{bdf}/reset_method`
2. **Pin power**: `d3cold_allowed=0`, `power/control=on`
3. **PCI remove/rescan** for native rebind (avoids sysfs `EEXIST` from stale kobjects)

---

## Part 2: VendorLifecycle Trait

### Location

`coralReef/crates/coral-ember/src/vendor_lifecycle.rs`

### Trait Definition

```rust
pub trait VendorLifecycle: Send + Sync + fmt::Debug {
    fn description(&self) -> &str;
    fn prepare_for_unbind(&self, bdf: &str, current_driver: &str) -> Result<(), String>;
    fn rebind_strategy(&self, target_driver: &str) -> RebindStrategy;
    fn settle_secs(&self, target_driver: &str) -> u64;
    fn verify_health(&self, bdf: &str, target_driver: &str) -> Result<(), String>;
}

pub enum RebindStrategy {
    SimpleBind,  // Standard sysfs driver_override + bind
    PciRescan,   // PCI remove + bus rescan (cleans stale sysfs)
}
```

### Implementations

| Struct | Vendor | Key Behavior |
|--------|--------|-------------|
| `NvidiaLifecycle` | 0x10de | SimpleBind always, nouveau gets 10s settle |
| `AmdVega20Lifecycle` | 0x1002 (Vega 20 IDs) | Disable reset_method on vfio-pci unbind, PciRescan for native, hwmon health check |
| `AmdRdnaLifecycle` | 0x1002 (other) | Conservative: same as Vega 20 until validated |
| `IntelXeLifecycle` | 0x8086 | SimpleBind (FLR expected), 5s settle |
| `GenericLifecycle` | unknown | Conservative: disable reset, PciRescan, 10s settle |

### Factory

```rust
pub fn detect_lifecycle(bdf: &str) -> Box<dyn VendorLifecycle>
```

Reads PCI vendor/device from sysfs, dispatches to appropriate implementation.

### Integration in swap.rs

`handle_swap_device` now:
1. Calls `detect_lifecycle(bdf)` at the start
2. Calls `lifecycle.prepare_for_unbind(bdf, current_driver)` before any unbind
3. Calls `lifecycle.rebind_strategy(target)` to choose SimpleBind vs PciRescan
4. Uses `lifecycle.settle_secs(target)` for wait duration
5. Calls `lifecycle.verify_health(bdf, target)` after bind succeeds

---

## Part 3: Ember sysfs Additions

New helpers in `crates/coral-ember/src/sysfs.rs`:

| Function | Purpose |
|----------|---------|
| `read_pci_id(bdf, field)` | Read vendor/device/subsystem IDs from sysfs |
| `read_power_state(bdf)` | Read PCIe power state (D0/D3hot/D3cold) |
| `pci_remove(bdf)` | Write `1` to device `remove` sysfs node |
| `pci_rescan()` | Write `1` to `/sys/bus/pci/rescan` |

---

## Part 4: Personality System Expansion

### New Personalities (coral-glowplug)

| Personality | Driver Module | Vendor |
|------------|---------------|--------|
| `XePersonality` | `xe` | Intel (Arc, Battlemage discrete) |
| `I915Personality` | `i915` | Intel (integrated + early discrete) |

### Personality Enum

`Personality` enum gained `Xe { drm_card }` and `I915 { drm_card }` variants. `PersonalityRegistry::default_linux()` now has 7 entries.

### PCI IDs

- `INTEL_VENDOR_ID = 0x8086` added
- `hbm2_training_driver(0x8086)` returns `Some("xe")`
- `native_compute_driver()` added for all 3 vendors

---

## Part 5: Test Results

- **6 ember tests**: VendorLifecycle unit tests (rebind strategy, settle times, ID recognition)
- **88 glowplug tests**: All pass including new Intel personality and registry entries
- **Live validation**: AMD Radeon VII full round-trip (amdgpu → vfio → amdgpu) with DRM/VRAM/hwmon intact

---

## Part 6: Adding a New Vendor

To add support for a new GPU vendor (e.g., Qualcomm Adreno, future Arm Mali):

1. **coral-ember**: Add `struct NewVendorLifecycle` implementing `VendorLifecycle` in `vendor_lifecycle.rs`
2. **coral-ember**: Add vendor match arm in `detect_lifecycle()`
3. **coral-glowplug**: Add `NEW_VENDOR_ID` to `pci_ids.rs`
4. **coral-glowplug**: Add personality struct (e.g., `AdrenoPersonality`) to `personality.rs`
5. **coral-glowplug**: Register in `PersonalityRegistry` and `Personality` enum
6. **coral-glowplug**: Add match arm in `device.rs` `swap()` post-swap state update

Start with `GenericLifecycle` defaults (conservative: disable reset, PciRescan, 10s settle) then refine based on empirical testing. The trait's design ensures bad defaults are safe-slow rather than destructive.

---

## Part 7: Action Items

### For coralReef

1. **Deploy updated binaries**: `coral-ember` and `coral-glowplug` with VendorLifecycle
2. **RDNA validation**: Test RX 5000/6000/7000 series with `AmdRdnaLifecycle` — may need refinement
3. **Consider `reset_method` persistence**: Currently disabled per-swap; could be set at service startup for known AMD cards

### For toadStool

1. **Lifecycle-aware dispatch**: When dispatching work to a GPU, toadStool should be aware of vendor lifecycle costs (AMD round-trips are slower due to PCI rescan)
2. **Health verification API**: `verify_health()` results should propagate through the dispatch layer

### For barraCuda

1. **RegisterMap + VendorLifecycle convergence**: Both traits are vendor-dispatched from PCI IDs. Consider a unified `VendorProfile` that includes register maps, lifecycle hooks, and shader ISA metadata.

---

## Supersedes

Updates the **Vendor-Agnostic Hardened GlowPlug Evolution handoff (Mar 18)** with:
- AMD MI50/Radeon VII swap path: ~~IN PROGRESS~~ → **VALIDATED** (with VendorLifecycle fix)
- Intel support: added personality stubs and lifecycle

---

## License

AGPL-3.0-only. Part of the ecoPrimals sovereign compute fossil record.
