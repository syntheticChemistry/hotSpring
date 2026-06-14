# Experiment 195: Driver Lab — Mesa vs Vendor Initialization Comparison

**Date**: May 15, 2026
**Hardware**: NVIDIA Titan V (GV100, 0000:02:00.0)
**Status**: IN PROGRESS — nouveau (mesa) trial complete, nvidia-470 (vendor/VM) pending

## Hypothesis

By cycling different drivers through the same GPU via glowplug's containment
architecture, we can precisely identify which registers each driver initializes
and map the security boundary that separates mesa (nouveau) from vendor (nvidia-470).

## Method

```text
DriverLabPlan::standard_titanv("0000:02:00.0")

Trial 1: cold/vfio  → BAR0 snapshot (baseline after FLR)
Trial 2: nouveau    → BAR0 snapshot (mesa/open driver warm swap)
Trial 3: nvidia-470 → BAR0 snapshot (vendor driver in agentReagents VM)

Diff: trial 1→2 = what nouveau initializes
Diff: trial 1→3 = what nvidia-470 initializes  
Diff: trial 2→3 = what nvidia-470 adds beyond nouveau (FECS, SEC2, ACR)
```

Containment architecture:
- nouveau → bare-metal swap (no module conflict with nvidia-580)
- nvidia-470 → agentReagents VM (HAZARDOUS: conflicts with nvidia-580)
- RTX 5060 (0000:21:00.0) on nvidia-580 → NEVER TOUCHED

## Results — Trial 1→2: cold/vfio vs nouveau-warm

### Domain-level diff

| Domain | Woke Up | Went Dead | Changed | Same | Total |
|--------|---------|-----------|---------|------|-------|
| PMC | 0 | 0 | 0 | 38 | 1024 |
| PBUS | 0 | 0 | 1 | 135 | 1024 |
| PFIFO | 0 | 0 | 0 | 236 | 2048 |
| PTIMER | 1 | 0 | 5 | 10 | 1024 |
| PFB | 0 | 0 | 4 | 160 | 1024 |
| PMU | 0 | 0 | 7 | 765 | 1024 |
| SEC2 | 0 | 0 | 0 | 192 | 256 |
| PRI_RING | 0 | 0 | 2 | 2303 | 4096 |
| PGRAPH | 0 | 0 | 0 | 40 | 1024 |
| FECS | 0 | 0 | 2 | 603 | 1024 |
| PGRAPH_GPC | 0 | 0 | 92 | 4658 | 16384 |
| PBDMA0 | 0 | 0 | 2 | 109 | 1024 |

### Key Observations

1. **Zero registers woke up from dead** — the GPU's cold/vfio state already had
   partial responsiveness (PMC, PFB, PMU, PRI_RING were already partially alive
   after the previous boot). This suggests FLR doesn't fully reset NVIDIA GPUs
   to a true cold state.

2. **92 PGRAPH_GPC registers changed** — this is nouveau's GPC cluster
   configuration (TPC setup, SM routing, zcull/rasterizer config). This is the
   bulk of what nouveau contributes.

3. **SEC2 = zero changes** — nouveau does not touch SEC2 at all. SEC2 registers
   read `CPUCTL=0, SCTL=0, PC=0` — completely uninitiated. This is the root
   blocker for FECS boot.

4. **FECS/GPCCS in HS lockdown** — confirmed by security registers:
   - `FECS_SCTL = 0x20204080` (HS mode, production fuse, debug disabled)
   - `FECS_SSTAT = 0x00000001` (ACR lockdown active)
   - `FECS_PC = 0` (never executed)
   - `FECS_IMEMC/DMEMD = 0` (no firmware loaded)

5. **PGRAPH hub split: FECS-gated vs hub-only** —
   - ALIVE: `GR_STATUS=0`, `GR_INTR=0`, `GR_ACTIVITY=1`
   - DEAD: `GR_FECS_CTXSW=0xbadf5040`, `GR_PRI_STATUS=0xbadf5040`
   - This proves certain PGRAPH registers are gated behind FECS firmware.
     They will come alive when nvidia-470 boots FECS via SEC2→ACR.

### FECS Security Decode

```
FECS_SCTL = 0x20204080
  bit 7  = 1  → HS (Heavy Secure) mode ENABLED by hardware fuse
  bit 14 = 1  → PIO access policy: blocked for unsigned code
  bit 21 = 1  → production fuse blown (no debug override possible)
  bit 29 = 1  → debug interface disabled

PMU_SCTL = 0x400e0100
  bit 8    = 1  → security level 1
  bit 30   = 1  → production mode

SEC2_SCTL = 0x00000000 → not initialized (root blocker)
```

The FECS HS mode is a **hardware fuse state** — it cannot be changed by software.
The only path to FECS execution is signed firmware loaded through SEC2→ACR,
which requires the nvidia proprietary driver's firmware blobs.

## Architecture Insight

The driver lab reveals a clear **security boundary** in GV100:

```text
┌─────────────────────────────────────────────┐
│ nouveau can reach (bare-metal, no security) │
│  ✅ PMC, PBUS, PFB, PTIMER                  │
│  ✅ PRI Ring (internal register bus)         │
│  ✅ PGRAPH hub (STATUS, INTR, ACTIVITY)     │
│  ✅ PGRAPH_GPC (92 registers configured)    │
│  ✅ PBDMA (channel infrastructure)          │
│  ✅ BAR0/BAR1/BAR3 (full R/W access)        │
├─────────────────────────────────────────────┤
│ Security boundary (HS fuse gate)            │
├─────────────────────────────────────────────┤
│ nvidia-470 can reach (contained VM only)    │
│  🔒 SEC2 → ACR authentication chain        │
│  🔒 FECS firmware execution                │
│  🔒 GPCCS firmware execution               │
│  🔒 PMU firmware                            │
│  🔒 GR_FECS_CTXSW, GR_PRI_STATUS          │
│  🔒 Compute context scheduling              │
│  🔒 SM warp dispatch                        │
└─────────────────────────────────────────────┘
```

## Deep Exploration Findings (May 15, continued)

### PFIFO Never Enabled by nouveau (GV100)

PFIFO_ENABLE (`0x2200`) reads `0x00000000` even during nouveau's active phase.
PFIFO RUNLIST, FB_BASE, and CHANNEL registers all return `0xbad00200` (PRI hub
timeout for PFIFO engine). This means nouveau on GV100 **never enables PFIFO**
— it lacks the PMU firmware needed to bring up the full PFIFO infrastructure.

| Register | During nouveau | After warm swap |
|----------|---------------|-----------------|
| PFIFO_ENABLE | 0x00000000 | 0x00000000 |
| PFIFO_FB_BASE | 0xbad00200 | 0xbad00200 |
| PFIFO_RUNLIST_BASE | 0xbad00200 | 0xbad00200 |
| PBDMA0_CHANNEL | 0xbad00200 | 0xbad00200 |

**Implication**: CE DMA through PBDMA channels is impossible without PFIFO.
nvidia-470 (in VM) is required for full PFIFO + PBDMA + CE pipeline.

### PBDMA Partial State

PBDMA0, PBDMA2, PBDMA4 show partial register responsiveness, but without PFIFO
enabled they cannot process channel commands. PBDMA0 GP_PUT is writable but
GP_BASE_LO is read-only — the channel configuration path is blocked.

### BAR1/BAR3 Capabilities

| BAR | Size | R/W | Content |
|-----|------|-----|---------|
| BAR1 (VRAM) | 256 MB | Yes | Data persists through warm swaps |
| BAR3 (RAMIN) | 32 MB | Yes | Instance memory, data persists |

PRAMIN window (`0x1700`) is writable — can slide BAR3 view across VRAM.

### Copy Engine Status

| CE | Offset | Status | Notes |
|----|--------|--------|-------|
| CE0 | 0x104000 | DEAD (0xbadf5040) | PRI-gated |
| CE1 | 0x105000 | ALIVE (status=1) | Responsive, but needs PFIFO for DMA |
| CE2 | 0x106000 | DEAD (0xbadf1100) | PRI-gated |

### Cold Boot Hazard — Register Writes Can Brick

**CRITICAL**: Writing to PMC_ENABLE (bit toggle), PFIFO_ENABLE, and PRAMIN
window registers during exploration corrupted the GPU state. nouveau then
failed to bind (`preinit failed with -110`). Recovery required PCI remove +
rescan, which triggered FLR, which killed the PRI ring (back to cold state).
nouveau cannot recover from cold state without VBIOS DEVINIT.

**Lesson**: Read-only exploration is safe. Write exploration must be done
with extreme care — save/restore all registers, and never toggle PMC_ENABLE
bits on a warm GPU. A power cycle is the only recovery from a bricked state.

## Next Steps

1. **Power cycle** to restore VBIOS DEVINIT state (PRI ring alive).
2. **Trial 3**: Launch `reagent-nvidia470-titanv` VM via agentReagents, capture
   BAR0 snapshot from inside VM, diff against Trial 2.
3. **Focus on VM-based nvidia-470**: PFIFO + CE DMA + FECS all require
   nvidia-470's full init. The bare-metal nouveau path gives us BAR access
   and PGRAPH state, but not compute capability.
4. **K80 recovery**: Power cycle to restore PLX bridge, then nouveau warm swap
   with SwapGuard for Kepler (where FECS boots without ACR).
5. **K80 continuous keepalive**: Root cause analysis (Exp 193 Phase 2) confirmed
   PLX D3cold is caused by **inactivity**, not swap events. The `PlxKeepalive`
   (ember) and `PlxGuardian` (glowplug) now provide continuous config space
   heartbeats. After power cycle, K80 must be protected by `PlxGuardian`
   before any operations.

## Code

- `glowplug/src/warm_init.rs`: `DriverLabPlan`, `DriverTrial`, `NV_BAR0_DOMAINS`
- `cylinder/src/vfio/bar_cartography.rs`: `scan_bar0`, `diff_bar_maps`, `BarMapDiff`
- `glowplug/src/sysfs_executor.rs`: `execute_warm_init()` (bare-metal path)
