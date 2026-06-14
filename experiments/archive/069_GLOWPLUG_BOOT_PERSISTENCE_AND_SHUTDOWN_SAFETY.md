# Experiment 069: GlowPlug Boot Persistence and Shutdown Safety

**Date**: March 16, 2026
**Status**: COMPLETE — boot persistence proven, shutdown kernel oops diagnosed and resolved
**Context**: Evolved from experiment 065. coral-glowplug now starts at boot via systemd,
survives reboots, and shuts down cleanly without kernel panics.
**Hardware**: biomeGate — 2× Titan V (GV100), RTX 5060 (display), TRX40 AORUS MASTER

---

## The Problem

coral-glowplug (Exp 064-065) worked as a running daemon but had two critical gaps:

1. **No boot persistence** — daemon did not start after reboot
2. **Shutdown kernel oops** — system shutdown triggered persistent kernel panics

Both had to be solved before coral-glowplug could serve as a reliable hardware broker.

---

## Phase 1: Boot Persistence

### Diagnosis

After reboot, coral-glowplug was not running. Both Titan V cards reverted to their
kernel default drivers (nouveau on 03:00.0, nouveau on 4a:00.0) instead of vfio-pci.

### Solution

1. **System-level systemd service** (`/etc/systemd/system/coral-glowplug.service`)

```ini
[Unit]
Description=coral-glowplug — Sovereign PCIe Device Lifecycle Broker
After=local-fs.target systemd-modules-load.service
Wants=systemd-modules-load.service
Before=display-manager.service

[Service]
Type=simple
ExecStartPre=/bin/sh -c 'echo Y > /sys/module/vfio_pci/parameters/disable_idle_d3 2>/dev/null || true'
ExecStart=/usr/local/bin/coral-glowplug --config /etc/coralreef/glowplug.toml
Restart=on-failure
RestartSec=5
TimeoutStopSec=10
KillMode=mixed
KillSignal=SIGTERM
Environment=RUST_LOG=info
RuntimeDirectory=coralreef
RuntimeDirectoryMode=0755

[Install]
WantedBy=multi-user.target
```

2. **System config** at `/etc/coralreef/glowplug.toml`
3. **Binary** installed to `/usr/local/bin/coral-glowplug`
4. **Modprobe config** at `/etc/modprobe.d/coralreef-vfio.conf`: `options vfio-pci disable_idle_d3=1`

### Auto-Discovery

Added `Config::auto_discover()` to scan `/sys/bus/pci/devices/` for discrete GPUs
(vendor IDs: NVIDIA 0x10de, AMD 0x1002, Intel 0x8086). Skips VGA display cards
(class 0x030000) and assigns vfio-pci or the existing bound driver.

### IOMMU Group Handling

Added `bind_iommu_group_to_vfio()` — when binding a GPU to vfio-pci, ALL devices
in its IOMMU group must also be bound. For Titan V, the companion audio device
(e.g., 03:00.1) must be unbound from snd_hda_intel and bound to vfio-pci first.

### Result

After `systemctl enable coral-glowplug.service` and reboot:

```
║ 0000:03:00.0 GV100 (Titan V) (vfio (group 69)) VRAM ✓ D0
║ 0000:4a:00.0 GV100 (Titan V) (vfio (group 34)) VRAM ✓ D0
```

Both Titans bound to vfio-pci at boot, VRAM alive, D0 power state.

---

## Phase 2: Shutdown Kernel Oops — Diagnosis

### Symptoms

Three consecutive reboots produced the same kernel panic:

```
PID: XXXX  Comm: Cursor  Tainted: G W OE
Call Trace:
  dump_stack_lvl
  __schedule_bug
  do_task_dead
  make_task_dead
  rewind_stack_and_make_dead
```

Key observations from the panic:
- **Comm: Cursor** — the dying process was always Cursor (the IDE), not coral-glowplug
- **ORIG_RAX: 0xca** (futex) in early crashes, **0xe7** (exit_group) in the final crash
- **RIP** pointed to userspace addresses — a userspace thread stuck in kernel D-state

### Root Cause

`lsof /dev/dri/*` revealed that **Cursor held open `/dev/dri/renderD129`** —
the DRM render node for the nouveau-bound Titan V (0000:03:00.0).

The chain of failure:

```
1. Boot: glowplug binds 03:00.0 to nouveau (oracle), 4a:00.0 to vfio
2. Xorg starts, nouveau exposes /dev/dri/card1 + /dev/dri/renderD129
3. Cursor opens renderD129 for GPU-accelerated rendering
4. Shutdown: systemd stops coral-glowplug (SIGTERM)
   → glowplug unbinds nouveau from 03:00.0
   → nouveau DRM device disappears while Cursor still has fd open
   → Cursor's GPU thread is stuck in nouveau kernel ioctl (D-state)
5. systemd sends SIGKILL to Cursor
   → kernel can't reap the stuck thread
   → do_task_dead → __schedule_bug → kernel oops
```

The GV100's nouveau teardown path has a known issue with render node cleanup —
when the DRM device is yanked while a client has an active GPU context, the
kernel thread hangs in an uninterruptible state.

### Investigation Path

| Attempt | What We Tried | Result |
|---------|--------------|--------|
| 1 | SIGTERM handler + explicit fd drop | Oops persisted — not glowplug's fds |
| 2 | Disable PCI reset_method before fd close | Oops persisted — Cursor, not glowplug |
| 3 | disable_idle_d3=1 for vfio-pci | Good practice but didn't fix nouveau issue |
| 4 | TimeoutStopSec=10, KillMode=mixed | Faster SIGKILL but Cursor still stuck |
| 5 | **Boot both Titans on vfio-pci** | **FIXED** — no renderD129, no Cursor grab |

### The Fix

Changed `/etc/coralreef/glowplug.toml`:

```toml
# CRITICAL: do NOT boot on nouveau — desktop apps (Cursor/Xorg) will grab
# renderD129 and block during shutdown, causing kernel oops on GV100.
[[device]]
bdf = "0000:03:00.0"
name = "titan-oracle"
boot_personality = "vfio"      # was "nouveau"
```

With both Titans on vfio-pci at boot:
- Only `/dev/dri/card0` + `/dev/dri/renderD128` exist (RTX 5060 on nvidia driver)
- Cursor, Xorg, GNOME Shell, Firefox all use only renderD128
- No desktop app can touch any Titan V
- Shutdown is clean — nvidia driver handles RTX 5060 teardown gracefully

### HBM2 Impact

The oracle card no longer boots on nouveau, so HBM2 is not automatically trained
at boot by nouveau. However:

- D3hot→D0 VRAM recovery still works (BIOS-trained HBM2 survives D3hot)
- `resurrect_hbm2()` in coral-glowplug can temporarily bind nouveau, wait for
  HBM2 training, then unbind — but ONLY when no desktop apps have DRM fds open
- The health monitor's auto-resurrection cycle handles this

---

## Phase 3: Graceful Shutdown Protocol

The daemon's shutdown handler now performs:

```
SIGTERM received →
  1. Disable PCI reset_method (echo "" > /sys/bus/pci/devices/{BDF}/reset_method)
     — for both GPU and companion audio device in IOMMU group
  2. Pin D0 (echo "on" > power/control) + disable d3cold
  3. Snapshot registers (state vault)
  4. Drop device slots (closes VFIO fds — safe because reset_method is empty)
  5. Abort socket accept loop
  6. Exit cleanly
```

The `reset_method` disable is critical: without it, VFIO's fd close triggers
a PCI PM reset that blocks indefinitely on GV100, causing the same rcu_stall
→ do_task_dead kernel oops.

---

## Lessons Learned

### 1. DRM Render Node Fencing

**Never unbind a DRM driver while desktop apps have its render node open.**

Desktop GPU compositors (Xorg, mutter/GNOME Shell) and apps (Cursor, Firefox)
aggressively open every `/dev/dri/renderD*` they find. If nouveau exposes a
render node for a GV100 Titan V, Cursor WILL use it. Unbinding nouveau while
Cursor holds the fd causes an unrecoverable kernel hang.

**Mitigation**: Boot non-display GPUs on vfio-pci. Only bind to nouveau/amdgpu
when there are no DRM consumers — i.e., during controlled resurrection cycles,
not at system boot when the desktop is starting.

### 2. VFIO PM Reset on GV100

VFIO's automatic PCI PM reset on fd close blocks indefinitely on GV100.
Always disable `reset_method` before closing VFIO file descriptors.

### 3. Systemd Service Design

- `Before=display-manager.service` — ensures coral-glowplug shuts down
  AFTER the display manager (not before), preventing driver yank during
  active desktop sessions
- `Type=simple` (not `notify`) — avoids watchdog timeout during slow
  GPU binding
- `TimeoutStopSec=10` — enough time for graceful shutdown but not so
  long that systemd holds up the entire shutdown sequence

### 4. IOMMU Group Completeness

VFIO requires ALL devices in an IOMMU group to be bound to vfio-pci.
For Titan V, the companion HDA audio device shares the IOMMU group.
Must unbind snd_hda_intel from the audio device before vfio-pci binding
succeeds on the GPU.

---

## Reproducibility Checklist

For the next GPU (MI50, another Titan V, V100, etc.):

1. **Install binary**: `cargo build --release -p coral-glowplug && sudo cp target/release/coral-glowplug /usr/local/bin/`
2. **Create config**: Add device BDF + name + role to `/etc/coralreef/glowplug.toml`
3. **Set boot_personality**: Use `vfio` for non-display GPUs (prevents DRM consumer grab)
4. **Install service**: `sudo cp coral-glowplug.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable coral-glowplug`
5. **Modprobe config**: `echo 'options vfio-pci disable_idle_d3=1' | sudo tee /etc/modprobe.d/coralreef-vfio.conf`
6. **Reboot and verify**: `systemctl status coral-glowplug` + `lspci -ks {BDF}` should show vfio-pci
7. **Verify no DRM leak**: `lsof /dev/dri/*` should show only display GPU's renderD128

---

## Current State

```
coral-glowplug.service: active (running)
  0000:03:00.0  GV100 (Titan V)  vfio-pci (group 69)  VRAM ✓  D0
  0000:4a:00.0  GV100 (Titan V)  vfio-pci (group 34)  VRAM ✓  D0
  0000:21:00.0  RTX 5060          nvidia (display)
```

No kernel oops. Clean shutdown. Boot persistent. Ready for sovereign compute.
