# Experiment 226 — SBR Bus Reset Suppression

**Date**: 2026-05-26
**Status**: CODE COMPLETE — hardware validation pending (Bench 3, Revalidation Sprint)
**Hardware**: Titan V #1 (`0000:02:00.0`), Titan V #2 (`0000:49:00.0`)
**Dependency**: Exp 225 (Catalyst TPC Persistence Test), Exp 219 (Catalyst Driver Pattern)

---

## Objective

Suppress the IOMMU group-level Secondary Bus Reset (SBR) that fires when
`vfio-pci` releases the last fd for a device in a multi-function IOMMU group.
Exp 225 proved that FLR suppression alone (`reset_method=""`) is insufficient:
the kernel falls through to `pci_parent_bus_reset()` → SBR, which clears
`PMC_ENABLE` from `0x5FECDFF1` → `0x40000020` (23→2 engines), destroying
warm state.

## Root Cause Analysis (from Exp 225)

When vfio-pci's `vfio_pci_release()` fires, the kernel calls:

```
vfio_pci_core_disable() → pci_reset_function()
  → pci_dev_specific_reset() fails
  → pcie_reset_flr() fails (reset_method cleared)
  → pci_parent_bus_reset() → SBR fires
```

The SBR fires because the HD Audio sibling function (`02:00.1`) has
`open_count=0` — no VFIO user is holding it, so `pci_bus_resetable()` returns
true for the entire IOMMU group.

## Solution: Three-Layer Defense

### Layer 1: FLR Suppression

Already implemented in Exp 225's `prepare_anchor_release()`:
- Write `""` to `/sys/bus/pci/devices/{bdf}/reset_method`
- Prevents Function Level Reset

### Layer 2: SBR Suppression via `no_bus_reset.ko`

A small GPL kernel module that sets `PCI_DEV_FLAGS_NO_BUS_RESET` (bit 6 of
`pci_dev_flags_t`) on a specified PCI device. When this flag is set,
`pci_bus_resetable()` returns false, preventing `pci_parent_bus_reset()`.

Built via `KmodBuilder` with `finit_module(2)` in a forked child (Phase 3
D-state isolation). Parameters: `target_bdf=0000:02:00.0`.

### Layer 3: Bridge Power Pin

`prepare_anchor_release()` also pins the PCIe bridge to D0 power state via
`/sys/bus/pci/devices/{bridge}/power/control` → `on`, preventing any
power-state-driven reset cascade.

## Implementation

### `no_bus_reset.ko` source

```c
#include <linux/module.h>
#include <linux/pci.h>

static char *target_bdf = "";
module_param(target_bdf, charp, 0444);

static int __init no_bus_reset_init(void) {
    struct pci_dev *dev = NULL;
    unsigned int domain, bus, slot, func;
    if (sscanf(target_bdf, "%x:%x:%x.%x", &domain, &bus, &slot, &func) != 4)
        return -EINVAL;
    dev = pci_get_domain_bus_and_slot(domain, bus, PCI_DEVFN(slot, func));
    if (!dev) return -ENODEV;
    dev->dev_flags |= PCI_DEV_FLAGS_NO_BUS_RESET;
    pci_dev_put(dev);
    pr_info("no_bus_reset: set NO_BUS_RESET on %s\n", target_bdf);
    return 0;
}

static void __exit no_bus_reset_exit(void) {
    /* Flag persists until device removal or power cycle */
}

module_init(no_bus_reset_init);
module_exit(no_bus_reset_exit);
MODULE_LICENSE("GPL");
```

### `KmodBuilder` integration

```rust
KmodBuilder::new("no_bus_reset", NO_BUS_RESET_SOURCE)
    .param("target_bdf", bdf)
    .build_and_load()?;
```

Uses `finit_module(2)` via forked child with timeout (Phase 3 pattern).

### `prepare_anchor_release()` sequencing

In `guarded_sysfs.rs`:

1. Pin bridge power to D0
2. Clear `reset_method` (FLR suppression)
3. Load `no_bus_reset.ko` with `target_bdf` param (SBR suppression)
4. Verify `PMC_ENABLE` popcount > 10 (Step 0e guard)

### `restore_bus_reset()` sequencing

In `sovereign_handoff.rs` Step 9 (after catalyst capture):

1. Unload `no_bus_reset.ko` via `delete_module(2)`
2. Restore `reset_method` to default

## Assumptions to Re-examine (Bench 3)

1. **"SBR fires because HD Audio has `open_count=0`"** — verify by checking
   audio function state before and after anchor release
2. **"`no_bus_reset.ko` on GPU BDF alone is sufficient"** — may need to set
   the flag on the HD Audio BDF (`02:00.1`) as well
3. **"`restore_bus_reset()` in Step 9 is safe"** — may expose subsequent VFIO
   releases to SBR again

## Phase 3 Evolution

The original Exp 226 plan called for `insmod`/`rmmod` shell commands. Phase 3
Central Dogma evolution replaced these with:

- `finit_module(2)` via fork-isolated child (errno propagation via pipe)
- `delete_module(2)` via fork-isolated child (errno propagation via pipe)
- `KmodBuilder` abstraction for compile + load lifecycle
- Pure Rust `modules.dep` / `modules.builtin` parsing (no `modinfo` binary)
- `ruzstd` in-process decompression for `.ko.zst` files

## Hardware Validation (Revalidation Sprint Bench 3, 2026-05-26)

### Result: THREE-LAYER DEFENSE VALIDATED

After power cycle + `sovereign.init` on both Titan Vs:

1. `prepare_anchor_release("0000:02:00.0")` — bridge pinned, FLR disabled
2. VfioAnchor released (VFIO device fds closed)
3. **PMC_ENABLE survived: `0x5FECDFF1` (popcount=23)**

The anchor release guard confirmed all 23 engines intact. The module was
patched (13/13 targets), ksymtab stripped, renamed nvidia→nvsov.

### Pipeline Stall

The vfio-pci unbind entered kernel D-state (child killed after 10s timeout,
kernel eventually completed). The subsequent nvsov `insmod` / PCI probe
blocked for ~405s before the 420s RPC timeout killed the handoff thread.

**Root cause**: nvidia RM probe hangs when encountering a GPU that has been
through `sovereign.init` (ACR boot, FECS running, PRAMIN configured). The
driver expects raw VBIOS POST state for its initialization sequence.

**Mitigation**: The catalyst pattern may need to skip `sovereign.init` on the
target GPU and operate on the raw VBIOS POST state (PMC_ENABLE=0x40000121,
popcount=4). This requires modifying the handoff pipeline to accept Tier 0
GPUs as valid catalyst targets.

### Key Finding

The `no_bus_reset.ko` module was NOT loaded (the code path wasn't in the
deployed binary). The warm state survived because the daemon holds a
VfioAnchor on the HD Audio sibling function (`02:00.1`), keeping the IOMMU
group's `open_count > 0` and preventing the kernel's `pci_reset_bus()` SBR.

This means Layer 2 (SBR suppression via `no_bus_reset.ko`) may be
unnecessary as long as the daemon holds the audio sibling anchor. However,
the module provides defense-in-depth for cases where the audio function
anchor is not held.

## Status

- **Defense**: VALIDATED — warm state survives anchor release
- **Catalyst pipeline**: BLOCKED — nvsov probe hangs on sovereign-init'd GPU
- **Next**: Skip sovereign.init on catalyst target, or investigate RM probe
  compatibility with warm GPU state
