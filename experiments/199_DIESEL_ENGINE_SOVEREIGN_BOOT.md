# Experiment 199 — Diesel Engine Sovereign Boot (Ember-Integrated)

> **Date:** May 16, 2026
> **Hardware:** Titan V (0000:02:00.0), Tesla K80 die 0 (0000:4b:00.0), K80 die 1 (0000:4c:00.0)
> **Method:** `sovereign.init` with `bar0_source=ember` via diesel engine (`DispatchHandler`)
> **Branch:** hotSpring main

## Objective

Route `sovereign.init` through the diesel engine's existing VFIO lifecycle
(`cached_devices`) instead of opening devices independently, and validate the
existing `boot_falcon_hs` ACR path for GV100 cold boot.

## Changes Made

### 1. `ComputeDevice` trait extensions (`lib.rs`)

Added `bar0()` and `dma_backend()` default-None methods to expose cached
BAR0 mapping and DMA backend from VFIO devices.

### 2. `NvVfioComputeDevice` implementations (`compute_device.rs`)

Implemented `bar0()` → `&VfioDispatchState.bar0` and `dma_backend()` →
`&VfioDispatchState.dma_backend` for linux targets.

### 3. `DispatchHandler::sovereign_init_ember()` (`dispatch/mod.rs`)

New method that borrows `MappedBar` + `DmaBackend` from `cached_devices`
and runs the sovereign pipeline. Dynamically selects `NvGspBridge` vs
`StubGspBridge` based on firmware availability.

### 4. `sovereign.init` routing (`handler/mod.rs`)

When `bar0_source=ember`, routes to `DispatchHandler::sovereign_init_ember()`
instead of the stateless `sovereign::sovereign_init()`.

### 5. Real ACR boot in `NvGspBridge` (`nv_gsp_bridge.rs`)

Replaced stub `acr_boot` with real implementation using `boot_falcon_hs`
for GPCCS then FECS when DMA backend is available. Set `supports_acr()=true`.

## Results

### coral-ember / VFIO ownership

`coral-ember` (PID 1610) holds VFIO device cdev fds (`/dev/vfio/devices/vfio{0,2,3}`)
as the "Immortal VFIO fd Holder". When coral-ember is running, toadstool's factory
gets EBUSY on all three GPUs and falls back to "caps-only mode" (no BAR0, no DMA,
no falcon boot). Toadstool correctly identifies chips via sysfs BOOT0 and caches
devices with correct capabilities even in caps-only mode.

**Resolution:** Stopped `coral-ember` and `coral-glowplug` services before
restarting `toadstool-ember`. With coral-ember stopped, all three VFIO devices
open successfully via iommufd/cdev.

### Titan V (0000:02:00.0) — GV100 Volta

| Stage | Status | Detail |
|-------|--------|--------|
| bar0_probe | OK | boot0=0x140000a1 chip=0x140 (GV100) |
| pmc_enable | OK | before=0x5fecdff1 after=0x5fecdff1 (no change) |
| memory_training | FAILED | PRAMIN PRI faults (0xbad0ac*) — VRAM inaccessible |

**PMC_ENABLE analysis:** 0x5fecdff1 = bits 0,4-12,14-15,18-19,21-28,30.
GR (bit 12) IS enabled, but PGRAPH sub-domain has additional clock gating
beyond PMC_ENABLE on Volta.

**Diesel engine factory (open_vfio) state:**
- VFIO FLR wiped FECS: `fecs_cpuctl_alias=0x00000000, pc=0x00000000, pmc_was_cold=true`
- GPCCS HS boot: DMA loaded (bl=576B, inst=12643B, data=2128B, sig=192B)
  but FBIF reads as PRI fault (`0xbadf3000`). GPCCS domain gated.
- FECS HS boot: **Timeout** — `cpuctl=0x00000000, pc=0x00000008, exci=0x00070000, sctl=0x00003000`
  Exception at PC=8 in HS mode. Bootloader starts but cannot DMA because FBIF is PRI-faulted.
- INIT_CTXSW failed (FECS not running)
- Device cached as "PBDMA dispatch ready" despite FECS failure

**Root cause:** PGRAPH clock gating. On Volta, PGRAPH has CG registers beyond
PMC_ENABLE that gate the GR domain. FECS/GPCCS falcon FBIF registers read as
PRI faults (0xbadf*), preventing DMA boot. Chicken-and-egg: FECS needs PGRAPH
ungated, but PGRAPH ungating needs GR init which needs FECS.

**Nouveau sequence:** Ungate PGRAPH CG → boot SEC2 → ACR load FECS/GPCCS →
INIT_CTXSW → GR init. Our pipeline attempts falcon boot before CG ungating.

### K80 die 0 (0000:4b:00.0) — GK210 Kepler

| Stage | Status | Detail |
|-------|--------|--------|
| bar0_probe | OK | boot0=0x0f22d0a1 chip=0x0f2 (GK210) |
| pmc_enable | OK | before=0xc0002020 after=0xfc37b1ef |
| memory_training | FAILED | DEVINIT not needed per register but PRAMIN is dead |

**PMC_ENABLE improvement:** Changed from 0xc0002020 (2 engines) to 0xfc37b1ef
(~25 engines). Significant engine enablement via PMC_ENABLE write.

**VBIOS interpreter execution:**
- Script 0: 352 ops, 222 writes, 4 PRI skips, 28 unknown opcodes — completed
- Script 1: Hit 100+ unknown opcodes at 0xb34f (opcode 0x0a) — aborted
- Scripts 2-5: Minimal ops, completed
- Total: 6 scripts, 519 ops, 268 writes, 130 unknown opcodes
- Net result: VRAM "still dead"

**PMU FALCON state:** `needs_post=true`, FALCON devinit failed with
"BIT 'I' short form: no PMU firmware table". PMU firmware upload path
unavailable — GK210 uses non-standard PMU firmware location.

### K80 die 1 (0000:4c:00.0) — GK210 Kepler

Identical results to die 0. Both dies require same VBIOS interpreter
improvements for cold boot GDDR5 training.

## Blockers Identified

### Titan V Cold Boot (Priority)

1. **PGRAPH CG ungating before falcon boot** — Need to write Volta PGRAPH
   clock gating control registers (NV_PGRAPH_PRI_FE_CLK_GATING_CTRL,
   NV_PGRAPH_PRI_GPC_CLK_GATING_CTRL, etc.) to ungate the GR domain
   before attempting DMA-based falcon boot.

2. **HBM2 VRAM accessibility** — After FLR, PRAMIN reads return PRI faults.
   HBM2 controller needs initialization before VRAM is accessible.
   VBIOS devinit script for GV100 expects PMU firmware pre-conditions.

### K80 Cold Boot

1. **VBIOS interpreter opcode coverage** — Script 1 (likely GDDR5 training
   script) aborts on unknown opcode 0x0a. Need to implement additional
   Kepler-era VBIOS opcodes for complete devinit.

2. **PMU firmware loading** — GK210 PMU firmware is in non-standard BIT
   table format. Either implement this loader or use host-side VBIOS
   interpreter as the primary devinit path.

## Key Architectural Validation

1. **`bar0_source=ember` pipeline works** — sovereign.init successfully
   borrows BAR0 and DMA from diesel engine's cached devices. No EBUSY.

2. **NvGspBridge ACR boot wired** — Real `acr_boot` implementation calls
   `boot_falcon_hs` for GPCCS then FECS with DMA backend. The mechanism
   works but PGRAPH gating prevents falcon execution.

3. **Diesel engine device identity** — K80 correctly identified as GK210/SM37
   when opened through VFIO (vs 0xFFFFFFFF via sysfs). Titan V correctly
   identified as GV100/SM70.

4. **PMC_ENABLE via ember path** — K80 PMC_ENABLE successfully expanded
   from 0xc0002020 to 0xfc37b1ef through the sovereign pipeline.

## Next Steps

1. Implement PGRAPH CG ungating for Volta (before falcon boot stage)
2. Expand VBIOS interpreter opcode coverage for GK210 script 1
3. Investigate HBM2 controller bring-up for GV100 post-FLR
4. Consider nouveau-derived PGRAPH init sequence capture
