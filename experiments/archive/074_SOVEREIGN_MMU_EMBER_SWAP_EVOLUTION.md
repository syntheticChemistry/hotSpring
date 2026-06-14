# Experiment 074 — Sovereign MMU Sprint: Ember Swap Pipeline + D-State Resilience

**Date:** March 22, 2026  
**Hardware:** biomeGate — 2× Titan V (GV100, `0000:03:00.0` + `0000:4b:00.0`), RTX 5060 (display head); kernel 6.17.9  
**Status:** COMPLETE — swap pipeline hardened; HW validated post-reboot  
**Crates:** coral-driver, coral-ember, coral-glowplug (coralReef)

---

## Context

After achieving iommufd/cdev backend (Exp 073), hotSpring needed a reliable **ember/glowplug swap pipeline** to iterate quickly on hardware configurations. The existing path had three blockers:

1. **Ember daemon D-state:** Risky sysfs writes (`driver/unbind`, `bind`, `remove`, `rescan`) put the ember process into uninterruptible kernel sleep. The Unix IPC socket became unresponsive; clients saw hangs rather than clean errors.

2. **IOMMU group peers:** Audio devices sharing the GPU’s IOMMU group were not released when swapping to native (nouveau) drivers, leaving the group in an inconsistent state for later VFIO binds.

3. **EmberClient fragility:** No retry path — `EAGAIN` on a temporarily hung socket was treated as fatal, amplifying transient failures.

**Hardware change:** AMD MI50 removed; second Titan V added. Fleet is now **2× Titan V + RTX 5060** as display head.

---

## Results (hardware-proven after reboot)

### 1. Process-isolated sysfs watchdog

Risky sysfs writes spawn a **child process** via `/bin/sh`. The parent polls with `try_wait()` and a **10s timeout**. If the child enters D-state, the parent **kills** it; the daemon stays responsive. Safe config-space attributes (`power/control`, `reset_method`, etc.) still use **direct writes** in-process.

### 2. IOMMU group peer release

New `release_iommu_group_from_vfio()`, symmetric to `bind_iommu_group_to_vfio()`. Example: audio on `03:00.1` is **auto-released** on nouveau swap and **auto-rebound** on VFIO swap.

### 3. EmberClient retry + full response read

**3 retries** with exponential backoff for `EAGAIN` / `EINTR`. `read_full_response()` loops until a complete JSON line is received (no truncated-RPC false failures).

### 4. DRM isolation auto-generation

New `drm_isolation.rs` generates **udev rules** and **Xorg config** from the device list at startup; **`udevadm` reload** is triggered automatically when rules change.

### 5. Ember socket group

Socket **`chgrp coralreef`** so non-root clients can connect.

### 6. iommufd boot module

`/etc/modules-load.d/iommufd.conf` ensures `iommufd` loads at boot.

### 7. Config auto-upgrade

Staged **`.next`** binaries + config, activated via **`ExecStartPre`** in the systemd unit.

### 8. MMU fault diagnostics

New `mmu_fault.rs` with structured fault decoding; `bench_mmu_fault_diagnostic` example. **PRIV_RING `0xbad00100`** identified as the real blocker (not generic MMU page-table setup alone).

---

## Validation

| Check | Outcome |
|-------|---------|
| Both Titan Vs opened via iommufd/cdev at boot | No `EBUSY` |
| `coralctl swap 0000:03:00.0 nouveau` | ~6s |
| `coralctl swap 0000:03:00.0 vfio` | ~6s |
| Full round-trip: vfio → nouveau → vfio with VRAM | OK |
| Ember process state | **S-state** throughout — **never D-state** |
| Audio IOMMU peer | Handled both directions |

---

## Files changed (coralReef)

### coral-driver

| File | Change |
|------|--------|
| `mmu_fault.rs` | **New** — structured MMU fault decode |
| `submission.rs` | Wired fault decode into submission path |
| `channel/mod.rs` | Integration for fault reporting |
| `registers.rs` | Supporting defs / wiring as needed |
| `examples/bench_mmu_fault_diagnostic.rs` | **New** — diagnostic bench |

### coral-ember

| File | Change |
|------|--------|
| `sysfs.rs` | Guarded risky writes + direct safe writes |
| `swap.rs` | IOMMU peer release on swap |
| `lib.rs` | DRM isolation hookup + socket group (`chgrp coralreef`) |
| `drm_isolation.rs` | **New** — udev + Xorg generation, reload |
| `vendor_lifecycle.rs` | `sysfs_write_direct` for safe attributes |

### coral-glowplug

| File | Change |
|------|--------|
| `ember.rs` | Retry + full-response read loop |

---

## Lessons learned

1. **D-state is a process property, not a socket bug.** Isolating risky sysfs in a child lets the parent keep polling the IPC loop; killing the child is preferable to wedging the whole daemon.

2. **VFIO ↔ native swaps are group-aware.** Symmetric bind/release for IOMMU peers avoids “half-released” audio devices that block later `vfio-pci` binds.

3. **Clients must assume streams, not datagrams.** JSON-RPC over a stream needs bounded retries and line framing; `EAGAIN` is recoverable, not fatal.

4. **Diagnostics should name the fault source.** Structured `mmu_fault` decoding narrowed the blocker to PRIV_RING (`0xbad00100`) instead of chasing generic PTE fill first.

---

## System files

| Path | Role |
|------|------|
| `/etc/coralreef/glowplug.toml` | Updated for 2× Titan V |
| `/etc/udev/rules.d/61-coralreef-drm-ignore.rules` | Auto-generated for both Titans |
| `/etc/modules-load.d/iommufd.conf` | Load `iommufd` at boot |
| `/etc/systemd/system/coral-ember.service` | `ExecStartPre` upgrade hook for staged `.next` |

---

## Next

1. **Nouveau oracle:** Swap a Titan to nouveau, read BAR0 PRAMIN, compare PDE/PTE encoding with coralReef expectations.  
2. **TLB invalidation** — align with hardware behavior after oracle data.  
3. **NOP GPFIFO dispatch** through the corrected MMU path once PRIV_RING / fault semantics are aligned.

---

*Experiment 074 hardens the ember swap pipeline against D-state sysfs hangs, releases IOMMU peers correctly, and makes glowplug resilient to transient socket conditions — enabling rapid sovereign-MMU iteration on dual Titan V + display without losing the daemon or the iommufd session.*
