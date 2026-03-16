# Handoff: GlowPlug Boot Persistence & Sovereign GPU Pipeline Status

**Date:** March 16, 2026
**From:** hotSpring (experiments 060-069)
**To:** toadStool, barraCuda, coralReef
**License:** AGPL-3.0-only
**Covers:** coral-glowplug v0.1.0, coralReef Iter 47, hotSpring Exp 060-069

---

## Executive Summary

- **coral-glowplug is production-grade**: boot-persistent systemd daemon, VFIO-first GPU binding, graceful shutdown with register snapshots
- **Kernel oops resolved**: DRM render node fencing prevents desktop apps from grabbing non-display GPU render nodes during shutdown
- **FECS firmware direct execution confirmed** (Exp 068): LS security bypass on clean falcon — the path to sovereign compute without vendor-signed firmware
- **Remaining blockers**: GPCCS address, FECS halt at PC=0x2835, DMA instance block, AMD Vega metal stub, SCM_RIGHTS fd passing
- **Reproducibility proven**: checklist for adding any new NVIDIA GPU to the fleet in <10 minutes

---

## Part 1: coral-glowplug Production Architecture

### What coral-glowplug Does

A persistent systemd daemon that owns GPU lifecycle from boot to shutdown:

| Feature | Status | Notes |
|---------|--------|-------|
| TOML config + auto-discovery | Complete | `/etc/coralreef/glowplug.toml` + `--auto` flag |
| Device activation (vfio, nouveau, amdgpu) | Complete | Explicit unbind→rebind with IOMMU group handling |
| Personality hot-swap | Complete | vfio→nouveau 4.1s, nouveau→vfio 1.5s |
| Health monitor (VRAM, power, domains) | Complete | 5s interval, auto-resurrection after 3 dead checks |
| HBM2 resurrection (nouveau warm cycle) | Complete | Temporary nouveau bind for HBM2 training |
| Auto-D0 recovery | Complete | Pins power_policy=always_on devices to D0 |
| Unix socket API | Complete | ListDevices, Swap, Resurrect, Health, Status, Shutdown |
| Register snapshot (state vault) | Complete | Captured before every transition and at shutdown |
| Boot persistence (systemd) | Complete | `coral-glowplug.service`, enabled, starts before display-manager |
| Graceful shutdown | Complete | Disable reset_method → pin D0 → snapshot → drop fds |
| IOMMU group binding | Complete | Auto-binds companion audio devices to vfio-pci |
| SCM_RIGHTS fd passing | Not implemented | Socket returns JSON metadata only |

### Systemd Integration

```ini
[Unit]
After=local-fs.target systemd-modules-load.service
Before=display-manager.service

[Service]
Type=simple
ExecStartPre=/bin/sh -c 'echo Y > /sys/module/vfio_pci/parameters/disable_idle_d3'
ExecStart=/usr/local/bin/coral-glowplug --config /etc/coralreef/glowplug.toml
TimeoutStopSec=10
KillMode=mixed
```

Key design decisions:
- `Before=display-manager.service` — shuts down AFTER display manager stops, preventing driver yank during active sessions
- `ExecStartPre` disables VFIO idle D3 transitions (prevents GV100 PM reset hangs)
- `Type=simple` avoids notify watchdog timeout during slow GPU binding

### Config Example (biomeGate)

```toml
[daemon]
socket = "/run/coralreef/glowplug.sock"
log_level = "info"
health_interval_ms = 5000

[[device]]
bdf = "0000:03:00.0"
name = "titan-oracle"
boot_personality = "vfio"    # NOT nouveau — see DRM fencing lesson
power_policy = "always_on"
role = "oracle"

[[device]]
bdf = "0000:4a:00.0"
name = "titan-target"
boot_personality = "vfio"
power_policy = "always_on"
role = "compute"
```

---

## Part 2: Critical Lesson — DRM Render Node Fencing

### The Problem

When Titan V #1 booted on nouveau, the kernel exposed `/dev/dri/renderD129`.
Desktop applications (Cursor IDE, Xorg, GNOME Shell, Firefox) immediately opened
this render node for GPU-accelerated compositing. During system shutdown:

1. coral-glowplug received SIGTERM, unbound nouveau from 03:00.0
2. DRM device disappeared while Cursor still held renderD129 open
3. Cursor's GPU thread stuck in nouveau kernel ioctl (uninterruptible D-state)
4. Kernel couldn't reap the stuck thread → `do_task_dead` → kernel oops

Three consecutive reboots produced the same panic.

### The Fix

**Boot ALL non-display GPUs on vfio-pci.** This prevents any DRM render node
from appearing, so no desktop app can grab it.

### toadStool Action

When implementing GlowPlug socket client:
- **Never request `Swap(bdf, "nouveau")` on a GPU while display-manager is running**
  unless you first verify `lsof /dev/dri/*` shows no consumers on the target card
- The `resurrect_hbm2()` function should be called only during controlled maintenance
  windows, not during normal desktop operation
- Future: implement `DrmConsumerFence` that checks for open render nodes before
  any nouveau/amdgpu bind

### barraCuda Action

No code changes needed. barraCuda accesses GPUs through wgpu/Vulkan (renderD128 on
the display card) or through toadStool's sovereign dispatch (which uses GlowPlug's
VFIO path). Neither path touches non-display GPUs directly.

---

## Part 3: Sovereign GPU Pipeline — Unsolved Blockers

### Critical (blocks sovereign compute dispatch)

| # | Blocker | Status | Next Step |
|---|---------|--------|-----------|
| 1 | GPCCS falcon address on GV100 | Unknown | Scan BAR0 0x400000-0x500000 for falcon signatures |
| 2 | FECS halt at PC=0x2835 | Stalled | Likely needs GPCCS running or channel context configured |
| 3 | DMA instance block (SEC2+0x480) | Not host-writable | Find alternative bind register or configure via FECS firmware |
| 4 | PRIVRING fault from PMC GR toggle | Documented | NEVER toggle PMC bit 12 on Volta — GR engine must be left as-is from BIOS |
| 5 | Sovereign HBM2 training | Workaround | Currently uses nouveau warm cycle; needs reverse-engineering of PHY init |

### Infrastructure (blocks next-GPU readiness)

| # | Gap | Owner | Next Step |
|---|-----|-------|-----------|
| 6 | AMD Vega metal stub | coralReef | Implement `amd_metal.rs` (6 TODOs: SMC, GRBM, UMC, GFX, registers, power-on) |
| 7 | SCM_RIGHTS fd passing | coralReef | Socket should pass VFIO container fds to toadStool clients |
| 8 | DRM consumer fencing | coralReef | Check `/dev/dri/renderD*` consumers before nouveau bind |
| 9 | GP_PUT DMA read (Exp 058) | coralReef | 6/7 PBDMA tests pass; last mile for VFIO channel dispatch |
| 10 | coral-driver ioctl gap | coralReef | NVIDIA DRM dispatch compile-ready but blocked at ioctl level |

### What WORKS Today

| Capability | Proven On | Reproducible? |
|-----------|-----------|---------------|
| D3hot→D0 VRAM recovery | GV100 (Titan V) | Yes — PCIe PM spec, vendor-agnostic |
| VFIO BAR0 register access | GV100 (Titan V) | Yes — standard VFIO |
| Health monitoring (VRAM, power, domains) | GV100 (Titan V) | Yes |
| Personality hot-swap | GV100 (Titan V) | Yes — but must fence DRM consumers |
| FECS firmware direct execution | GV100 (Titan V) | Yes — requires D3hot→D0 clean falcon |
| Boot persistence | biomeGate system | Yes — systemd + modprobe config |
| Graceful shutdown | biomeGate system | Yes — disable reset_method before fd close |
| Auto-discovery (PCI scan) | Any system | Yes — scans sysfs for GPU vendor IDs |

---

## Part 4: Per-Primal Action Items

### coralReef

1. **Implement AMD Vega metal** (`amd_metal.rs`) — register map, power domains, engine map for MI50/GFX906
2. **Add SCM_RIGHTS fd passing** to socket server — toadStool needs VFIO container fds
3. **Implement DRM consumer fence** — `lsof /dev/dri/renderD*` check before nouveau/amdgpu bind in `resurrect_hbm2()`
4. **Continue GPCCS discovery** — scan Volta BAR0 for falcon signatures outside known addresses
5. **GP_PUT DMA read** — complete Exp 058 PBDMA channel dispatch

### toadStool

1. **Wire GlowPlug socket client** — connect to `/run/coralreef/glowplug.sock`, implement `ListDevices`, `Health`, `Swap`
2. **Add VFIO device capability** to `toadstool-sysmon` — detect vfio-pci bound devices, read IOMMU group
3. **Respect DRM consumer fence** — never request nouveau/amdgpu swap while display-manager is active
4. **hw-learn: observe GlowPlug health** — feed VRAM/power/domain health data into learning pipeline

### barraCuda

1. **No code changes required** — barraCuda's IPC-first design means it already works through toadStool
2. **Full sovereign pipeline unblocked once FECS/GPCCS are running** — WGSL→SM70 SASS compilation already works via coralReef

---

## Part 5: Register Quick Reference

### GV100 Falcon Addresses

| Falcon | BAR0 Base | SCTL Offset | Role |
|--------|----------|-------------|------|
| SEC2 | 0x087000 | +0x240 | Heavy Secure — runs ACR, EMEM always writable |
| FECS | 0x409000 | +0x240 | Light Secure — context switch, host-loadable when clean |
| GPCCS | ??? | +0x240 | GPC context switch — address TBD |
| PMU | 0x10A000 | +0x240 | Power management — fully HS locked |

### Key Sysfs Paths

| Path | Purpose |
|------|---------|
| `/sys/bus/pci/devices/{BDF}/reset_method` | Write "" to disable PM reset before VFIO fd close |
| `/sys/bus/pci/devices/{BDF}/power/control` | Write "on" to pin D0 |
| `/sys/bus/pci/devices/{BDF}/d3cold_allowed` | Write "0" to prevent D3cold |
| `/sys/module/vfio_pci/parameters/disable_idle_d3` | Write "Y" to prevent idle D3 transitions |
| `/sys/kernel/iommu_groups/{N}/devices/` | List all devices in IOMMU group |

---

## License

AGPL-3.0-only. Part of the ecoPrimals sovereign compute fossil record.
