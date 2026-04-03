# Experiment 140: Uncrashable GPU Safety Architecture

**Date:** 2026-04-03
**Status:** VALIDATED
**Predecessors:** 132 (frozen warm dispatch), 138 (D-state root cause), 139 (ACR lockdown)

## Problem

Repeated system lockups (hard power-off required) when interacting with
NVIDIA GPUs via VFIO sovereign compute. Three distinct crash vectors:

1. **Dispatch on cold GPU**: `open_warm` writes Volta-specific PCCSR/PFIFO
   registers to a cold Kepler GPU (GK210). Engines are clock-gated
   (PMC_ENABLE: PGRAPH=0, PFIFO=0), so the writes trigger PRI ring faults
   that cascade into total BAR0 death (0xFFFFFFFF).

2. **VFIO fd close on dead GPU**: After BAR0 death, closing the VFIO fd
   triggers kernel `vfio_pci_core_disable()` which accesses PCI config
   space. The dead PCIe endpoint never sends a completion TLP, causing a
   PCIe completion timeout → uninterruptible D-state → system hang.

3. **Service restart**: `systemctl stop coral-ember` sends SIGTERM. Default
   Rust signal handler terminates immediately — no destructors run. Kernel
   closes all VFIO fds synchronously during process cleanup, hitting the
   same D-state path as (2).

## Root Cause Analysis

The kill chain is: **write to clock-gated engine → PRI fault → BAR0 death
→ VFIO fd close → kernel D-state → system hang**.

Every crash followed this sequence. The GPU can survive bad register writes
if the target engines are clocked. It's specifically writing to gated
engines that kills the PRI ring irreversibly.

PMC_ENABLE (register 0x200) is the authoritative source for which engines
are clocked. On cold GPUs (vfio-pci from boot, never driver-initialized):
- Titan V (GV100): PMC_ENABLE = 0x40000020 (PGRAPH=0, PFIFO=0)
- K80 (GK210):     PMC_ENABLE = 0xc0002020 (PGRAPH=0, PFIFO=0)

## Solution: Three-Layer Defense

### Layer 1: PMC_ENABLE Cold-GPU Guards (coral-driver)

Before any BAR0 write, check that the target GPU's engines are actually
enabled. Added to ALL five device open paths:

| Function | File |
|----------|------|
| `open_warm()` | `coral-driver/src/nv/vfio_compute/mod.rs` |
| `open_from_fds()` | same |
| `open_from_fds_with_recipe()` | same |
| `open_warm_with_context()` | same |
| `apply_gr_bar0_init()` | `coral-driver/src/nv/vfio_compute/init.rs` |

Each path reads BOOT0 (offset 0x0) and PMC_ENABLE (offset 0x200). If
BOOT0 = 0xFFFFFFFF or PGRAPH (bit 12) / PFIFO (bit 8) are not set, the
operation returns a clean error immediately with zero writes performed.

Additional guards in `apply_nonctx_writes`, `apply_dynamic_gr_init`, and
`restart_warm_falcons` provide defense-in-depth.

`MappedBar::apply_gr_bar0_writes` performs `is_alive()` checks every 64
writes and aborts mid-batch if BAR0 goes dead.

### Layer 2: Guarded VFIO fd Close (coral-ember)

`guarded_vfio_close()` in `coral-ember/src/guarded_open.rs` wraps
`drop(HeldDevice)` in a dedicated thread with a 10-second timeout. If the
kernel's VFIO teardown enters D-state, the thread is leaked (via
`mem::forget`) and the caller continues. Applied to:

- `ember.release` IPC handler
- REQ IRQ auto-release path
- Swap/close path
- **NEW: `ember.shutdown` IPC handler** (drains all held devices)

### Layer 3: Graceful Systemd Shutdown

New `ember.shutdown` IPC command drains the entire held device map through
`guarded_vfio_close`, sends a JSON-RPC response, then calls
`std::process::exit(0)`.

Service file changes:
```ini
ExecStop=/usr/local/bin/coralctl shutdown
TimeoutStopSec=30    # was 3
```

Shutdown sequence:
1. `systemctl stop coral-ember`
2. systemd runs `ExecStop` → `coralctl shutdown`
3. coralctl connects to ember socket, sends `ember.shutdown`
4. ember drains all devices via guarded close
5. ember sends response, calls `process::exit(0)`
6. systemd confirms clean deactivation

If ember is already dead, ExecStop fails harmlessly and systemd falls back
to SIGTERM → SIGKILL.

## Validation Results

All tests performed with 3 cold compute GPUs held by ember:
- Titan V (GV100): BOOT0=0x140000a1, PMC_ENABLE=0x40000020
- K80 GPU1 (GK210): BOOT0=0x0f22d0a1, PMC_ENABLE=0xc0002020
- K80 GPU2 (GK210): BOOT0=0x0f22d0a1, PMC_ENABLE=0xc0002020

### Test 1: Sovereign dispatch on cold K80 (previously crashed system)
```
$ coralctl dispatch 0000:4c:00.0 --shader noop.wgsl --sovereign --sm 0
error: GPU is cold/un-POSTed (PMC_ENABLE=0xc0002020, PGRAPH=false, PFIFO=false).
       Engines must be enabled before GR init.
```
**Result:** Clean rejection. BOOT0 unchanged. No PRI faults. System stable.

### Test 2: Sovereign dispatch on all 3 cold GPUs
All three rejected cleanly with the same PMC_ENABLE guard message.
Post-test BOOT0 reads confirmed all GPUs still alive and unchanged.

### Test 3: `coralctl shutdown` with 3 held devices
```
$ coralctl shutdown
Sending graceful shutdown to ember...
  {"devices_released":3,"status":"shutdown_complete"}
```
Journal:
```
guarded VFIO close completed normally bdf=0000:03:00.0
guarded VFIO close completed normally bdf=0000:4c:00.0
guarded VFIO close completed normally bdf=0000:4d:00.0
all devices released count=3
ember exiting cleanly after shutdown
Deactivated successfully.
```
**Result:** All 3 devices released in ~416ms total. Clean exit.

### Test 4: `systemctl stop coral-ember` with 3 held cold GPUs
```
$ sudo systemctl stop coral-ember
```
Completed in 597ms. Journal confirmed ExecStop fired, all 3 devices
guarded-closed, clean deactivation. **This is the exact operation that
previously crashed the system.**

## Files Changed

### coral-driver
- `crates/coral-driver/src/nv/vfio_compute/mod.rs`
  - PMC_ENABLE guards in `open_from_fds_with_recipe`, `open_warm_with_context`
  - (Prior: `open_warm`, `open_from_fds`, `MappedBar::is_alive`,
    `MappedBar::apply_gr_bar0_writes` health monitoring)
- `crates/coral-driver/src/nv/vfio_compute/init.rs`
  - (Prior: PMC_ENABLE in `apply_gr_bar0_init`, BOOT0 in `apply_nonctx_writes`,
    `apply_dynamic_gr_init`, `restart_warm_falcons`)

### coral-ember
- `crates/coral-ember/src/ipc/handlers_device.rs`
  - New `shutdown()` handler
  - (Prior: `mmio_write` pre/post health checks, `release` guarded close)
- `crates/coral-ember/src/ipc.rs`
  - `ember.shutdown` wired in both Unix and TCP dispatch tables
- `crates/coral-ember/src/guarded_open.rs`
  - (Prior: `guarded_vfio_close()` utility)

### coralctl
- `crates/coral-glowplug/src/bin/coralctl/main.rs`
  - New `Shutdown` CLI command

### hotSpring
- `scripts/boot/coral-ember.service`
  - `ExecStop=/usr/local/bin/coralctl shutdown`
  - `TimeoutStopSec=30`

## Remaining Work

- K80 cold-boot: implement proper PMC_ENABLE → PRI ring init → devinit → GR
  (the guards prevent crashes, but don't solve the cold-boot itself)
- Titan V warm-fecs: SEC2 CMDQ reconstruction for sovereign ACR boot
- Consider `signal-hook` crate for SIGTERM as belt-and-suspenders alongside
  the IPC-based ExecStop

## Key Insight

The crash was never about the GPU interaction itself — it was about the
**teardown path**. Writing to a cold GPU is dangerous but recoverable via
PCIe bus reset. The system crash happens when the VFIO fd close attempts
PCI config space access on a dead endpoint. The fix is not just "don't write
to cold GPUs" (Layer 1) but also "survive if you do" (Layers 2 and 3).
