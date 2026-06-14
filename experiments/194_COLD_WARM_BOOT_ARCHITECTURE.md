# Experiment 194 — Cold/Warm Boot Architecture: PRI Ring, Falcon Security, No-FLR Swap

**Date:** May 15, 2026
**Status:** ✅ Complete (architectural findings validated on hardware)
**Hardware:** Titan V (GV100), Tesla K80 (GK210B), RTX 5060 (SM120)
**Site:** biomeGate

## Problem Statement

With PLX D3cold prevention validated (Exp 193), explore cold and warm boot paths across three GPU generations to understand what blocks sovereign compute initialization and what preserves it.

## Key Findings

### 1. Cold Boot PRI Ring Barrier

After VFIO bind (which triggers FLR), the GPU's internal register bus (PRI Ring) is dead on both Kepler and Volta:

- **PRI Ring (0x122xxx)** returns `0xbadf3000` — the ring's clock tree wasn't initialized by DEVINIT/VBIOS
- **PGRAPH HUB (0x400xxx)** returns `0xbadf1002` on K80 — PRI timeout, HUB sub-block not clocked
- **GPC registers (0x418xxx+)** return `0xbadf3000` — behind dead PRI ring

Only registers with **direct BAR0 mapping** (bypass PRI ring) are accessible from cold:
- PMC (0x000xxx), PBUS (0x088xxx), PTIMER (0x020xxx)
- FECS falcon core (0x409xxx) — has dedicated BAR0 port
- PRI Master Control (0x120xxx) — top-level controller

**Root Cause:** FLR resets the GPU to power-on state but doesn't re-run VBIOS DEVINIT scripts. Without DEVINIT, the PLL clock trees and PRI ring remain unconfigured. Blind writes to 0x122xxx (PRI ring config) don't help because the ring's clock domain itself isn't enabled.

### 2. Volta Falcon Security (HS Mode) Barrier

On the Titan V (GV100), even with some PGRAPH registers accessible from cold:

| Register | Cold Value | Meaning |
|----------|-----------|---------|
| FECS CPUCTL | 0x00000010 | HRESET active |
| FECS CPUCTL_ALIAS | 0x00000000 | No alias write |
| FECS SCTL | 0x00003000 | HS-capable, security locked |
| FECS SSTAT | 0x40000000 | HS capable |

- **PIO writes to IMEM/DMEM work** — verified with multi-word write+readback
- **Falcon will NOT execute** — CPUCTL writes (STARTCPU, HRESET) have no effect
- **SCTL=0x3000** bits [13:12]=0x3 indicates falcon security prevents unsigned code execution
- Even the bootloader (`fecs_bl.bin`, 576B, fits in 8KB IMEM) cannot execute via PIO

The HS (Heavy Secure) mode requires SEC2/ACR authentication before FECS can run. SEC2 is behind the PRI ring (dead from cold).

### 3. No-FLR Warm Swap — BREAKTHROUGH

**Pattern:** nouveau (or other driver) initializes GPU → disable FLR → swap to vfio-pci → warm state preserved

```
Pin parent bridge: echo on > power/control, echo 0 > d3cold_allowed
Disable FLR:       echo "" > reset_method
Unbind nouveau:    echo BDF > drivers/nouveau/unbind
Set override:      echo vfio-pci > driver_override
Probe:             echo BDF > drivers_probe
```

**Titan V Results (27/27 registers alive, zero faults):**

| Sub-system | Cold State | Warm State (after swap) |
|-----------|-----------|----------------------|
| PRI Ring (0x122000) | 0xbadf3000 | 0x00000000 ✅ |
| PRI Ring HUB_NR | 0 | 1 ✅ |
| PRI Ring GPC_NR | 0 | 6 ✅ |
| PGRAPH STATUS | 0xbadf5040 | 0x00000000 ✅ |
| GPC0 SETUP | 0xbadf3000 | 0x7006863a ✅ |
| FECS CPUCTL | 0x10 (HRESET) | 0x10 (HRESET) |
| PMC_ENABLE | 0x400011e0 | 0x5fecdff1 ✅ |
| PFB MMU | 0x00208001 | 0x00208001 ✅ |
| PTHERM | 0xbadf3000 | 0x0000000c ✅ |

**PMC_ENABLE writes work** — bit 12 (GR) toggle verified. Full read-modify-write capability on warm GPU.

**Critical:** FECS remains in HRESET because nouveau couldn't boot it (missing PMU firmware for GV100 ACR bootstrap). The nvidia-580 driver also rejected GV100 (dropped Volta support). An older nvidia driver (470/535) would fully initialize FECS.

### 4. K80 PLX D3cold During Manual Swap

The K80 warm boot via nouveau succeeded (`fb: 12288 MiB GDDR5`, GK110B detected), but the subsequent swap to vfio-pci killed the PLX bridge despite manual bridge pinning. Key lesson: **manual sysfs pinning is insufficient** — the kernel's PM subsystem overrides it during driver transitions.

This confirms that toadStool's diesel engine (`SwapGuard` + burst keepalive + `pin_bridge_hierarchy()`) is required for K80 swaps. Direct sysfs manipulation races with the kernel.

### 5. GPU Topology Extraction

**Titan V (GV100)** — warm state:
- 6 GPCs (all accessible)
- 40 TPCs (7+7+7+6+7+6 per GPC)
- 6 ROPs
- IMEM: 8 KB, DMEM: 32 KB (FECS falcon)
- 12,288 MiB HBM2

**K80 (GK210B)** — warm state (before PLX death):
- 5 GPCs
- 6 TPCs per GPC (30 total?)
- 6 ROPs
- 12,288 MiB GDDR5

## Architecture Diagram

```
Cold Boot Path (both GPUs):
  FLR → DEVINIT lost → PRI Ring dead → engines inaccessible
  Only: PMC, PBUS, PTIMER, FECS falcon core (direct BAR0)
  Kepler: FECS PIO works but PGRAPH clock-gated (PRI ring needed)
  Volta: FECS PIO writes work but HS security blocks execution

Warm Boot Path:
  vendor_driver.init() → PRI Ring alive → engines configured
  disable FLR (reset_method="") → unbind → bind vfio-pci
  Result: full register access, writes work, PMC controllable
  Remaining: FECS not running (needs ACR/PMU firmware path)

Future: Full Warm Boot (with FECS):
  nvidia-470 or patched nouveau → full FECS/GPCCS boot
  disable FLR → swap to vfio-pci
  FECS running → compute context → golden context → dispatch
```

## Code Impact

### toadStool Diesel Engine Patterns Validated

1. **`pin_bridge_hierarchy()`** — confirmed necessary but insufficient alone for manual swaps; must be combined with `SwapGuard` burst keepalive
2. **No-FLR swap** — `reset_method=""` before swap prevents state destruction
3. **Warm init seeding** — vendor driver provides PRI ring + clock tree + PFB init; sovereign code takes over after swap
4. **PLX bridge keepalive** — K80 requires continuous CfgRd during swap window (toadStool `SwapGuard`)

### New Capabilities Discovered

- **PMC_ENABLE writable from VFIO** on warm GPU — engine enable/disable/reset from userspace
- **PRI Ring topology readable** — GPC/ROP/HUB counts for architectural discovery
- **FECS IMEM/DMEM PIO verified** on Volta — writes work, readback confirmed
- **Falcon security state readable** — SCTL/SSTAT decode for boot strategy selection

### 6. Full BAR Access from VFIO (May 15 Addendum)

After warm swap with FLR disabled, all three BARs are accessible for sovereign access:

| BAR | Size | Purpose | Verification |
|-----|------|---------|-------------|
| BAR0 (resource0) | 16 MB | MMIO registers | 13/15 critical registers alive |
| BAR1 (resource1) | 256 MB | Direct VRAM | 1 MB write+verify, 0 errors |
| BAR3 (resource3) | 32 MB | RAMIN (instance memory) | 4-word write+verify, 0 errors |

**VRAM bandwidth** (Python, single-threaded BAR1 access):
- Write: ~6.4 MB/s (256K words verified)
- Read: ~1.3 MB/s (PCIe read round-trips)
- Native Rust with mmap would be significantly faster

**SEC2 PRI Route Dead**: SEC2 at 0x840000 returns 0xbadf1100. PMC_ENABLE bit probing didn't help — SEC2's PRI ring station was never configured by nouveau (no PMU firmware → can't bootstrap SEC2). PMU DMEM contains 0xdead5ec2 ("DEAD SEC2") — nouveau's sentinel.

**All Falcons in HS Mode**: PMU, FECS, GPCCS, NVENC0 — all have SCTL=0x3000 (Heavy Secure). Only nvidia-470 (or equivalent) can break through HS via the SEC2→ACR authentication chain.

### 7. WarmInitPlan — Diesel Engine Injection (May 15 Addendum)

Implemented `WarmInitPlan` in glowplug (`warm_init.rs`) — a multi-stage driver injection abstraction with **two containment strategies**:

```rust
// Bare-metal seeder (host-safe, no module conflicts)
let plan = WarmInitPlan::nouveau_titanv("0000:02:00.0");
assert!(plan.is_bare_metal());

// Contained seeder (HAZARDOUS, runs in agentReagents VM)
let plan = WarmInitPlan::nvidia470_titanv("0000:02:00.0");
assert!(plan.requires_containment());
assert_eq!(plan.reagent_template(), Some("reagent-nvidia470-titanv"));
```

**Architecture decision**: Conflicting drivers (nvidia-470 vs nvidia-580) are **hazardous material** — they share the same `nvidia.ko` module name and MUST be contained in agentReagents VMs. The host DRM (RTX 5060) is sacred and never disturbed. The diesel engine spins up additional isolation layers (VMs via benchScale/agentReagents) rather than fighting the kernel's driver model.

| Strategy | Containment | Host Impact | Example |
|----------|------------|-------------|---------|
| Bare-metal | Host kernel swap | None (non-conflicting driver) | nouveau → vfio-pci |
| Contained | agentReagents VM | None (VFIO passthrough) | nvidia-470 in VM |

`SysfsSwapExecutor::execute_warm_init()` **only handles bare-metal plans**. It panics on contained plans — those must be dispatched through agentReagents/benchScale VM lifecycle.

## Next Steps

1. **Full FECS via contained seeder**: Launch `reagent-nvidia470-titanv` VM via agentReagents → FECS running inside VM → compute dispatch via VM IPC
2. **K80 warm boot via diesel engine**: Wire `SwapGuard` into `execute_warm_init` for PLX-safe nouveau→vfio-pci
3. **CE DMA dispatch**: Use warm PFIFO + PBDMA (all 4 alive) to submit Copy Engine operations via BAR3 RAMIN
4. **Golden context capture**: With FECS running in VM, capture golden state → extract init sequence
5. **Sovereign driver evolution**: Use captured init sequences to evolve cylinder `sovereign_init` stages → reduce driver dependency over time

## Relation to Previous Experiments

- **Exp 193**: PLX D3cold prevention — validated, but manual swaps still race with kernel PM
- **Exp 192**: HW validation sprint — Titan V FECS protocol identified, K80 PLX blocked
- **Exp 188**: K80 warm-catch breakthrough — nouveau GK210→GK110B patch
- **Exp 169**: Titan V warm handoff — HBM2 preservation
- **Exp 167**: Warm handoff pattern — `skip_sysfs_unbind` PCI rescan
- **Exp 164**: Sovereign compute dispatch proven — NOP dispatch via DRM
