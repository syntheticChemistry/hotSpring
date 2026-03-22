# hotSpring → Compute Trio: D-State Resilient Ember + Swap Pipeline Proven

**Date:** March 22, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** D-state–safe sysfs I/O, IOMMU group symmetry, EmberClient resilience, DRM isolation automation, fleet hardware (2× Titan V + RTX 5060)

---

## Executive Summary

- **Ember swap pipeline:** nouveau ↔ vfio round-trip proven on Titan V end-to-end.
- **D-state resilience:** Risky sysfs writes isolated in forked children; parent polls with timeout so the daemon never blocks the whole process in uninterruptible sleep.
- **IOMMU group symmetry:** Bind and release paths both handle all peers in the group (primary + companion devices such as audio).
- **EmberClient:** Retry with backoff, full-line JSON-RPC reads, longer timeouts for swap workloads.
- **DRM isolation:** Auto-generated udev rules and Xorg `ServerFlags` from `EmberDeviceConfig` at startup (`write_if_changed`, udev reload on update).
- **Hardware:** 2× Titan V + RTX 5060 (MI50 removed from this host). **iommufd/cdev** backend active on both Titans at boot.

### Validation snapshot

- Swap path exercised: **nouveau → vfio** and **vfio → nouveau** on **Titan V** with the watchdog + symmetric group release in the loop.
- Boot: **iommufd** module present via modules-load; both Titan V devices usable through the **cdev** backend without manual post-boot module load.

---

## Part 1: The D-State Problem and Solution

### Why sysfs writes can hang

Some sysfs writes (especially **unbind** / **driver** paths) can block in **uninterruptible kernel sleep (D-state)** while the kernel waits on driver or PCI teardown. From userspace that looks like a stuck syscall — the thread cannot be interrupted with a normal signal until the kernel completes the operation.

### Why D-state poisons the whole process

In a single-threaded or lightly threaded daemon, **one thread stuck in D-state** blocks progress for work tied to that process: other operations may queue behind locks, health checks stall, and the supervisor may conclude the service is dead. Isolating risky I/O avoids “one bad write freezes everything.”

### Process isolation pattern

For each **risky** sysfs write: **fork a child** that performs the write; the **parent** waits with a **timeout** (poll / `try_wait` style loop). If the child does not exit in time, the parent **SIGKILL**s it and treats the operation as failed — the **parent process** never enters D-state for that path.

### Safe vs risky classification

| Class | Examples | Strategy |
|-------|----------|----------|
| **Safe** | `power/control`-style knobs where behavior is known synchronous | Direct write in-process |
| **Risky** | `unbind`, `driver`, bind/unbind heavy paths | **guarded_sysfs_write()** — child + timeout |

### guarded_sysfs_write() (summary)

- Spawns **`/bin/sh -c`** (or equivalent) in a **child** to perform the sysfs write.
- Parent runs a **try_wait** loop with a **10s** timeout.
- On hang: **SIGKILL** child, return failure upstream (caller can log and retry policy as appropriate).

### Design rationale (short)

Child isolation contains **D-state risk to the helper**; the parent only waits on **pid** exit status and wall-clock timeout. **Zombie** children are reaped by the usual `waitpid` path; **SIGKILL** guarantees teardown of a wedged shell even when the kernel will not return from the syscall in the child.

---

## Part 2: IOMMU Group Symmetry

### Previous bug

`bind_iommu_group_to_vfio()` bound the full group to vfio, but **`bind_native()`** only handled the **primary BDF**. A **peer** (e.g. audio at `03:00.1`) could remain on **vfio-pci**, keeping the group claimed and breaking native/GPU stack expectations.

### Fix

- **`release_iommu_group_from_vfio()`** runs **before** native bind: **unbind peers** from vfio and clear **`driver_override`** as needed so the whole group can return to the native stack consistently.
- **`for_each_iommu_peer()`** (or equivalent refactor) **centralizes** iteration over group members so bind and release stay **symmetric**.

---

## Part 3: EmberClient Retry + Full-Response Read

### EAGAIN clarification

**EAGAIN (os error 11)** on the client was often **socket read timeout / non-blocking WouldBlock**, not an Ember sysfs error string. Treating it as a **transient I/O** condition avoids mis-attributing failures.

### Retry policy

- **3 retries** with **500ms × attempt** backoff for **`WouldBlock` / `Interrupted`** (and related transient cases via **`is_transient_io()`**).

### Protocol read

- **`read_full_response()`** loops until a **newline-delimited** line is read (JSON-RPC line protocol) — no partial JSON parse on chunk boundaries.

### Timeouts

- **60s** timeout for swap-class operations (**up from 30s**), aligned with long-running kernel/device transitions.

### is_transient_io() classifier

Centralizes “should retry this read/write error?” so **`WouldBlock`**, **`Interrupted`**, and similar **non-fatal** socket conditions do not abort the first good line of JSON-RPC. Permanent errors (e.g. **broken pipe**, **EOF** where inappropriate) still fail fast after retries exhaust.

---

## Part 4: DRM Isolation Auto-Generation

New **`drm_isolation.rs`** module in **coral-ember**:

- Reads the managed **device list** from **`EmberDeviceConfig`** at startup.
- Emits **udev** rules per device: e.g. strip seat tagging (`TAG-="seat"`, `ENV{ID_SEAT}=""`) so the display stack does not auto-seat secondary GPUs as intended by policy.
- Emits **Xorg `ServerFlags`**: **`AutoAddGPU=false`**, **`AutoBindGPU=false`** to reduce accidental hotplug binding of managed devices.
- **`write_if_changed()`** skips rewriting files when content is **byte-identical**.
- On update: **`udevadm control --reload-rules`** (and related reload as implemented) so changes apply without manual edits.

### Operational notes

- Generated artifacts live under paths chosen by coral-ember (same directory conventions as existing Ember config); **idempotent** updates avoid unnecessary **udev** churn and log spam.
- Xorg flags are **defensive**: they reduce auto-seat/auto-bind surprises for **managed** GPUs; local desktop policy still applies for primary display devices.

---

## Part 5: System Configuration

- **`/etc/modules-load.d/iommufd.conf`**: ensures **iommufd** loads at boot where required for the cdev path.
- **`glowplug.toml`**: updated for **2× Titan V** topology and IDs.
- **Systemd `ExecStartPre`**: upgrade hook for **staged binary activation** before the main service runs.
- **Socket ownership:** **`chgrp coralreef`** on the Ember/GlowPlug socket for **non-root** access per policy.

### Boot-time expectations

After `systemctl` start, verify **iommufd** is loaded (`/sys/module/iommufd` or equivalent) and Ember logs show **cdev** backend on both Titan V instances. **GlowPlug** should consume the same **two-fd** iommufd layout documented in the iommufd evolution handoff.

---

## Part 6: Per-Primal Action Items

### coralReef

1. **Already integrated** — relevant changes are on **`main` at `6b2202f`**. Review as the canonical reference for this wave.
2. **Pattern adoption:** Any daemon doing **kernel-adjacent sysfs** should consider **`guarded_sysfs_write`** (or the same fork + timeout + SIGKILL pattern) for **risky** paths; keep **safe** paths direct.
3. **DRM isolation:** Prefer **generated** udev + Xorg snippets from config over **hand-maintained** rules on each host.

### toadStool

1. **GlowPlug client stub:** When implemented, adopt **EmberClient-style** retry/backoff and **full-line reads** for JSON-RPC over the socket.
2. If toadStool ever writes **sysfs** directly, use the **process-isolation** pattern for **unbind/driver** class operations — do not perform those in the main process thread without a guard.

### barraCuda

1. **Fleet note:** This host is **2× Titan V**, **no MI50** — refresh any **MI50-specific** benchmark baselines or capability assumptions.
2. **IPC clients:** The **retry / backoff / transient I/O** classifier pattern applies to any **fleet client** talking to long-running services, not only Ember.

---

## Part 7: Architecture — Swap + Watchdog (Conceptual)

```
Config (EmberDeviceConfig)
    │
    ├─► drm_isolation.rs ──► udev rules + Xorg ServerFlags (write_if_changed)
    │
    └─► swap / bind paths
            │
            ├─► bind_iommu_group_to_vfio()     ──► full group → vfio
            ├─► release_iommu_group_from_vfio() ──► peers unbind + driver_override cleared
            └─► guarded_sysfs_write()          ──► risky sysfs only (child + 10s + SIGKILL)

EmberClient (JSON-RPC over Unix socket)
    │
    ├─► read_full_response() ──► newline-delimited line
    ├─► is_transient_io() + 3 retries × (500ms × attempt)
    └─► 60s timeout for swap-class RPC
```

---

*D-state isolation keeps the daemon responsive; symmetric IOMMU group release keeps nouveau ↔ vfio swaps honest; EmberClient and DRM generation close the last operational gaps. AGPL-3.0-only.*
