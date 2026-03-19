# Handoff: Ember Architecture + DRM Isolation

**Date:** March 19, 2026
**From:** hotSpring (Exp 070)
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** coral-ember v0.1.0, coral-glowplug (Ember integration), DRM isolation

---

## Executive Summary

- **coral-ember** is an immortal systemd service holding VFIO fds for both Titan V GPUs. Passes duplicated fds to coral-glowplug via `SCM_RIGHTS`. Glowplug crashes no longer trigger PCIe PM resets.
- **`swap_device` RPC** is the single atomic driver swap orchestrator. All sysfs unbind/bind operations happen exclusively inside Ember. Glowplug delegates.
- **DRM isolation** prevents Xorg/GDM from reacting to DRM device creation during nouveau/nvidia swaps. Xorg `AutoAddGPU=false` + udev 61-prefix seat tag removal.
- **External fd safety check** prevents deadlocks: Ember refuses swaps if any external process still holds VFIO group fds. Forces swaps through glowplug RPC (which drops its fds first).
- **SCM_RIGHTS and DRM consumer fence** are now **DELIVERED** — these were P2 and P4 on the PIN handoff.

---

## Part 1: Ember Architecture

### Why Ember Exists

When coral-glowplug restarts (systemd watchdog, upgrade, crash), dropping
VFIO file descriptors triggers a kernel PCIe PM reset on GV100. This resets
HBM2 training state and can cause kernel panics if the device is mid-DMA.

Ember splits the lifecycle:

```
coral-ember (PID immortal, holds original fds)
    ↕ SCM_RIGHTS via /run/coralreef/ember.sock
coral-glowplug (restartable, holds dup'd fds)
    ↕ JSON-RPC via /run/coralreef/glowplug.sock
clients (socat, toadStool, experiments)
```

### Key Files

| File | Purpose |
|------|---------|
| `coralReef/crates/coral-glowplug/src/bin/coral_ember.rs` | Ember daemon — fd holder + swap orchestrator |
| `coralReef/crates/coral-glowplug/src/ember.rs` | EmberClient — glowplug's connection to Ember |
| `coralReef/crates/coral-glowplug/src/device.rs` | DeviceSlot::swap() — delegates to Ember |
| `coralReef/scripts/boot/11-coralreef-gpu-isolation.conf` | Xorg isolation config |
| `coralReef/scripts/boot/61-coralreef-drm-ignore.rules` | udev DRM isolation rules |

### Ember RPC Methods

| Method | Input | Output |
|--------|-------|--------|
| `vfio_fds` | `{bdf}` | SCM_RIGHTS: container_fd, group_fd, device_fd |
| `swap_device` | `{bdf, target}` | `{ok, personality}` — atomic driver swap |
| `release_device` | `{bdf}` | Drops fds (debugging only) |
| `reacquire_device` | `{bdf}` | Opens new VFIO fds (debugging only) |
| `list` | — | `{devices: [bdf...]}` |
| `status` | — | `{devices, uptime_secs}` |

### swap_device Flow

```
1. Check for external VFIO fd holders → ABORT if found
2. Drop held VFIO fds (if target != vfio)
3. Pin power (D0, no D3cold)
4. sysfs unbind current driver
5. sysfs bind target driver
6. Wait for driver init (DRM up for nouveau/nvidia)
7. If target == vfio: reacquire VFIO fds
8. Return personality string
```

### DRM Isolation Preflight

Before binding any DRM-creating driver (nouveau, nvidia, amdgpu), Ember
verifies:

1. `/etc/X11/xorg.conf.d/11-coralreef-gpu-isolation.conf` exists with `AutoAddGPU false`
2. `/etc/udev/rules.d/61-coralreef-drm-ignore.rules` exists and covers the target BDF

If either is missing, the swap is **blocked** with a clear error message.

---

## Part 2: DRM Isolation

### The Problem

When nouveau binds to a GPU, the kernel creates `/dev/dri/cardN`. Xorg detects
this via its `AutoAddGPU` mechanism, adds the GPU as a RandR provider, and
gnome-shell crashes on the CRT-less reconfiguration event (the compute GPU has
no display attached).

### The Fix (Two Layers)

**Layer 1: Xorg** — `11-coralreef-gpu-isolation.conf`

```
Section "ServerFlags"
    Option "AutoAddGPU" "false"
    Option "AutoBindGPU" "false"
EndSection
```

Prevents Xorg from hotplugging any new GPU. The display GPU (RTX 5060) is
already configured via `10-nvidia.conf` OutputClass.

**Layer 2: udev** — `61-coralreef-drm-ignore.rules`

```
SUBSYSTEM=="drm", KERNELS=="0000:03:00.0", ENV{ID_SEAT}="", ENV{ID_FOR_SEAT}="", TAG-="seat", TAG-="master-of-seat", TAG-="uaccess"
SUBSYSTEM=="drm", KERNELS=="0000:4a:00.0", ENV{ID_SEAT}="", ENV{ID_FOR_SEAT}="", TAG-="seat", TAG-="master-of-seat", TAG-="uaccess"
```

Runs at priority 61 (before logind at 70) to prevent seat assignment entirely.

### Installation

```bash
sudo cp scripts/boot/11-coralreef-gpu-isolation.conf /etc/X11/xorg.conf.d/
sudo cp scripts/boot/61-coralreef-drm-ignore.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
# Requires Xorg restart (reboot) for AutoAddGPU to take effect
```

---

## Part 3: Stability Invariants

### DO NOT

- Change `boot_personality = "vfio"` in `/etc/coralreef/glowplug.toml`
- Remove `disable_idle_d3` from vfio-pci options
- Send `swap_device` directly to Ember socket (bypasses glowplug fd drop → deadlock)
- Remove the DRM isolation configs while Ember is running
- PMC-toggle GR bit 12 on GV100

### DO

- Always swap via glowplug JSON-RPC: `device.swap` → glowplug drops fds → calls Ember
- Return to `vfio` personality before shutdown
- Deploy new Ember binary via atomic rename (`cp new /usr/local/bin/coral-ember.new && mv .new coral-ember`)
- Reboot for new Ember binary to take effect (it's immortal — can't restart without dropping fds)

---

## Part 4: Action Items

### For coralReef

1. Version-control the Ember preflight checks (currently inline in `coral_ember.rs`)
2. Add `nvidia` to the swap_device target list (currently `nouveau | amdgpu | nvidia`)
3. Complete GP_PUT DMA read — cache flush experiment is in Iter 57

### For toadStool

1. Wire GlowPlug socket client using `device.swap` method (not direct Ember calls)
2. Feed health data from `device.health` into hw-learn
3. Add Ember status check to sysmon (ember socket at `/run/coralreef/ember.sock`)

### For barraCuda

No action needed. When toadStool wires the GlowPlug client, barraCuda's
`sovereign-dispatch` feature gate will automatically see VFIO devices.

---

## License

AGPL-3.0-only. Part of the ecoPrimals sovereign compute fossil record.
