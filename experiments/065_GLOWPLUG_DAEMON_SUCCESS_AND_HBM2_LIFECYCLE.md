# Experiment 065: GlowPlug Daemon Success & HBM2 Lifecycle Management

**Date**: 2026-03-16  
**Status**: Active  
**Cards**: 2× Titan V (GV100, 0x1d81), 1× RTX 5060 (GB206, 0x2d05)  
**Setup**: Oracle (0000:03:00.0, nouveau) · Target (0000:4a:00.0, vfio-pci) · Display (0000:21:00.0, nvidia)

---

## Part 1: Milestones Achieved

### 1.1 Passwordless GPU Access (Zero Friction)

The single largest drag on iteration speed was `pkexec`/`sudo` prompts hanging
inside the Cursor IDE terminal. Solved permanently:

| Component | Path | Purpose |
|-----------|------|---------|
| Sudoers rules | `/etc/sudoers.d/coralreef` | Passwordless `tee` to sysfs, `modprobe`, driver bind/unbind |
| Udev rules | `/etc/udev/rules.d/99-coralreef-permissions.rules` | Device permissions persist across reboots |
| `gpu-ctl` | `/usr/local/bin/gpu-ctl` | CLI for `status`, `bind`, `unbind`, `swap`, `d0`, `setup` |

**Verified**: Survives reboot. All GPU operations — including driver hot-swap — work
without any authentication prompt.

### 1.2 coral-glowplug Daemon

New crate: `crates/coral-glowplug/` — a persistent PCIe device lifecycle broker.

**Architecture**:
```
coral-glowplug (daemon)
├── config.rs     — TOML config, multi-device, XDG-aware socket paths
├── device.rs     — DeviceSlot, Personality enum, register snapshots, health probes
├── health.rs     — Periodic background monitor, auto-D0 recovery
├── socket.rs     — Unix socket server, JSON protocol, spawn_blocking for swaps
└── main.rs       — CLI (--config / --bdf), systemd-notify, multi-device activation
```

**Capabilities**:
- Binds GPUs at startup, holds VFIO fds open persistently
- Personality system: Vfio, Nouveau, Amdgpu, NvidiaProprietary, Unbound
- Hot-swap between personalities via socket API (vfio→nouveau: 4.1s, nouveau→vfio: 1.5s)
- Health monitor: BOOT0, PMC_ENABLE, PRAMIN sentinel, 9-domain probe
- Auto-recovery: detects D3hot drift, forces D0 when policy=always_on
- Socket API: ListDevices, Health, Swap, Status, Shutdown

**Key design lesson**: During hot-swap, VFIO fds must be *closed* (not leaked) before
unbind. Leaking prevents the kernel from releasing the VFIO group, causing indefinite
sysfs block. The PM reset from fd close is accepted — the state vault snapshot preserves
register state for restoration.

### 1.3 Test Suite: 24/26 Pass Without Root

```
24 passed, 2 failed (known), 0 regressions
```

| Test | Status | Notes |
|------|--------|-------|
| vfio_open_and_bar0_read | PASS | VFIO open + BAR0 mmap, no root needed |
| vfio_alloc_and_free | PASS | DMA buffer allocation |
| vfio_multiple_buffers | PASS | Multi-buffer DMA |
| vfio_upload_and_readback | PASS | DMA data transfer |
| vfio_free_invalid_handle | PASS | Error handling |
| vfio_readback_invalid_handle | PASS | Error handling |
| vfio_pci_discovery | PASS | PCI enumeration |
| vfio_metal_cartography | PASS | Full BAR0 domain map |
| vfio_metal_glowplug | PASS | GlowPlug warm-up sequence |
| vfio_sovereign_glowplug_full | PASS | Full sovereign warm with oracle |
| vfio_power_bounds | PASS | Power state probing |
| vfio_interpreter_probe | PASS | VBIOS script parsing |
| vfio_devinit_pmu_probe | PASS | PMU FALCON probing |
| vfio_hbm2_phy_probe | PASS | HBM2 PHY register reads |
| vfio_hbm2_timing_capture | PASS | Memory timing extraction |
| vfio_hbm2_falcon_diagnostic | PASS | FALCON state machine analysis |
| vfio_hbm2_training_attempt | PASS | HBM2 training sequence |
| vfio_pri_backpressure_probe | PASS | PRI bus error detection |
| vfio_pfifo_diagnostic_matrix | PASS | PFIFO engine diagnostics |
| vfio_pclock_deep_probe | PASS | Root PLL / PCLOCK analysis |
| vfio_oracle_root_pll_programming | PASS | Oracle-driven PLL programming |
| vfio_digital_pmu_full | PASS | Full digital PMU emulation |
| vfio_boot_follower_diff | PASS | Oracle vs cold register diff |
| vbios_script_scanner | PASS | VBIOS script parsing (no HW) |
| vfio_cross_card_fb_init_oracle | FAIL | Needs root for oracle sysfs resource0 |
| vfio_dispatch_nop_shader | FAIL | Needs GR firmware (secure boot barrier) |

### 1.4 Driver Hot-Swap (Verified Live)

```
Before:  0000:4a:00.0 → vfio-pci (D0)
Swap:    vfio-pci → nouveau    (4.1s, nouveau does HBM2 training)
Swap:    nouveau  → vfio-pci   (1.5s)
After:   0000:4a:00.0 → vfio-pci (D0), BOOT0=0x140000A1 ✓
```

Full cycle completes in <6 seconds with no manual intervention, no password prompts.

---

## Part 2: The HBM2 Problem

### 2.1 The Core Issue

HBM2 training state on GV100 is **fragile**. It dies under multiple conditions:

| Trigger | HBM2 Result | Recovery |
|---------|-------------|----------|
| BIOS POST completes | **Trained** ✓ | N/A — this is the golden state |
| VFIO bind (D3hot→D0 pin) | **Survives** ✓ | Just pin power |
| VFIO fd close (PM reset) | **Destroyed** ✗ | Reboot or nouveau re-warm |
| Driver unbind (no fd leak) | **Destroyed** ✗ | Reboot or nouveau re-warm |
| nouveau unbind | **Destroyed** ✗ | Reboot or nouveau re-bind |
| Extended idle (D3hot drift) | **Destroyed** ✗ | Reboot or nouveau re-warm |

**Symptom**: PRAMIN (0x700000) returns `0xBADx_xxxx` prefix, FBPA/FBHUB/LTC domains
return faulted values. 5/9 domains alive (PMC, PFIFO, PMU, NVPLL, BOOT0), 4/9 dead
(PFB, FBHUB, LTC, FBPA, PRAMIN).

### 2.2 What We Need GlowPlug To Do

1. **Detect** HBM2 death in real-time (health monitor already probes PRAMIN)
2. **Resurrect** HBM2 by cycling through nouveau (which does full FBPA/HBM2 init)
3. **Maintain** HBM2 by preventing the conditions that kill it
4. **Verify** resurrection succeeded before returning to sovereign VFIO mode

Target state: GlowPlug keeps VRAM alive indefinitely, automatically recovering from
any HBM2 loss event without human intervention or reboot.

### 2.3 Resurrection Strategy

```
GlowPlug health monitor detects PRAMIN dead
  → snapshot registers (state vault)
  → swap to nouveau (triggers full HBM2 training: PMU FALCON, FBPA, PHY)
  → wait for nouveau init (~3-4s)
  → verify VRAM via nouveau DRM (optional)
  → swap back to vfio-pci
  → verify PRAMIN alive
  → restore register snapshot (oracle values for PLLs, PCLOCK)
  → report resurrection success/failure
```

---

## Part 3: Experiments — Results

### 3.1 HBM2 Death Timing (COMPLETED)

**Finding**: The killer is the VFIO fd close. When the fd is closed, the kernel
performs a PM reset that wipes PMC_ENABLE, stops PLLs, and kills FBPA/LTC/PRAMIN.

Register delta at death:

| Register | Alive (nouveau-warmed) | Dead (after PM reset) |
|----------|------------------------|----------------------|
| PMC_ENABLE | `0x5fecdff1` (many engines) | `0x40000020` (bare minimum) |
| PFIFO | `0x0020000e` | `0xbad0da00` |
| LTC0 | `0x00000000` (active) | `0xbadf3000` (faulted) |
| FBPA0 | `0x00000043` (active) | `0xbadf3000` (faulted) |
| NVPLL | `0x00000009` (running) | `0x00000008` (stopped) |
| MEMPLL | `0x00000004` (locked) | `0x00000000` (off) |
| PRAMIN | `0x8cdddac0` (live data) | `0xbad0ac00` (dead) |

The PM reset turns off the memory PLLs, which kills FBPA (HBM2 controller),
which makes LTC (L2 cache) and PRAMIN (VRAM window) inaccessible.

### 3.2 Nouveau Warm → VFIO Handoff (COMPLETED — SUCCESS)

**Result**: Full resurrection in 5.1 seconds:

```
PRE:  vram=false  domains=5/9  PMC=0x40000020
      → close VFIO fd (PM reset)
      → clear driver_override to (null) via \n write
      → bind nouveau (HBM2 training: 1.4s)
      → nouveau detects 12288 MiB, DRM card created
      → unbind nouveau
      → rebind vfio-pci, pin D0
POST: vram=true   domains=9/9  PMC=0x5fecdff1
```

**Key discovery**: Writing empty string to `driver_override` via `tee` does NOT clear
it (stays as previous value). Writing `\n` (newline) clears it to `(null)`. This was
the root cause of all previous nouveau bind failures.

**Sentinel test**: PRAMIN write `0xC0EE1EEF`, readback matched — full VRAM
read/write capability confirmed after resurrection.

### 3.3 Auto-Resurrection (IMPLEMENTED)

The health monitor now tracks consecutive dead VRAM readings. After 3 consecutive
checks (default 15 seconds at 5s interval), it automatically triggers resurrection:

```rust
// In health.rs
if *dead_count >= 3 && slot.has_vfio() && slot.config.power_policy == "always_on" {
    slot.resurrect_hbm2()?;
}
```

Socket API also supports on-demand resurrection:
```json
{"Resurrect": {"bdf": "0000:4a:00.0"}}
→ {"Resurrected": {"bdf": "0000:4a:00.0", "vram_alive": true, "domains_alive": 9}}
```

### 3.4 Test Suite Impact

After resurrection implementation: **25/27 tests pass** (up from 24/26).
The new `vfio_hbm2_lifecycle_probe` test validates the full death→resurrection cycle.

---

## Part 4: Architecture Summary

### The HBM2 Lifecycle State Machine

```
┌─────────┐     BIOS POST      ┌──────────┐
│  COLD   │ ──────────────────→ │ TRAINED  │
│ (power  │                     │ (HBM2 OK │
│  off)   │                     │  12 GiB) │
└─────────┘                     └────┬─────┘
                                     │
                              vfio-pci bind
                              (D3hot→D0 pin)
                                     │
                                     ▼
                               ┌──────────┐
                               │  ALIVE   │ ←──── GlowPlug resurrect_hbm2()
                               │ (9/9     │       (nouveau warm cycle, 5.1s)
                               │  domains)│
                               └────┬─────┘
                                    │
                             VFIO fd close
                             (PM reset)
                                    │
                                    ▼
                               ┌──────────┐
                               │   DEAD   │
                               │ (5/9     │
                               │  domains)│
                               └──────────┘
```

### What GlowPlug Provides

1. **Persistent fd ownership** — daemon holds VFIO fd open, preventing PM reset
2. **Health monitoring** — periodic PRAMIN/domain probe detects death
3. **Auto-resurrection** — nouveau warm cycle restores 9/9 domains in 5.1s
4. **On-demand resurrection** — socket API for toadStool to trigger manually
5. **Hot-swap** — seamless driver transitions (vfio↔nouveau↔amdgpu)
6. **Zero friction** — no sudo/pkexec, passwordless sysfs access
