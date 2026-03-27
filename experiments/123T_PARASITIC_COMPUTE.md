# Experiment 123-T: Parasitic Compute via sysfs BAR0

**Date**: 2026-03-25 (design)
**GPU**: Titan V #1 and #2 (GV100)
**Hypothesis**: Sovereign compute dispatch possible via sysfs BAR0 while nouveau is active

## Strategy

Instead of fighting the FWSEC/WPR2/ACR security chain, piggyback on nouveau's
working GPU state. Nouveau has already authenticated FECS/GPCCS via its own ACR
boot path — use sysfs BAR0 to interact with the running falcons directly.

## Prerequisites

- GPU bound to `nouveau` (driver managing FECS/GPCCS, PFIFO, etc.)
- sysfs BAR0 access (`/sys/class/drm/{node}/device/resource0`) via `Bar0Access`
- Root or coralreef group permissions on resource0

## Why This Might Work

1. nouveau already authenticated FECS/GPCCS — they're running
2. sysfs BAR0 mmap is independent of driver (kernel allows it)
3. FECS method interface (0x409500/504/800) is just MMIO registers
4. We can potentially create a "shadow" channel that nouveau doesn't know about

## Why This Might NOT Work

1. Exp 122B showed FECS/GPCCS in HRESET on Titan #2 under nouveau (PMU trapped)
2. Titan #1 may be different — needs probing
3. nouveau's interrupt handler could interfere with our FECS methods
4. Channel ID conflict could corrupt nouveau's state
5. nouveau holds the GR context golden image — we need it but can't access it

## Sub-Experiments

### 123-T0: Falcon State Probe Under Nouveau
**Goal**: Determine if FECS/GPCCS are alive on each Titan while nouveau is active.

Read for both Titans (via sysfs BAR0):
- FECS: cpuctl (0x409100), sctl (0x409240), PC (0x409110 or 0x409400)
- GPCCS: cpuctl (0x41A100), sctl (0x41A240), PC (0x41A110 or 0x41A400)
- FECS mailbox: status (0x409800), scratch (0x409500)
- PMU: cpuctl (0x10A100), sctl (0x10A240), exci (0x10A04C)
- SEC2: cpuctl (0x087100), sctl (0x087240)

**Decision gate**: If both FECS and GPCCS are running (cpuctl bit 4 not set, PC != 0),
proceed to 123-T1. If in HRESET, this path is dead for that Titan.

### 123-T1: FECS Method Interface Test
**Goal**: Send commands to FECS and check for responses.

Test sequence:
1. Read FECS_STATUS (0x409800) — should be non-zero if FECS is in idle loop
2. Write FECS_SCRATCH0 (0x409500) = 0 (clear)
3. Write FECS_SCRATCH1 (0x409504) = command
4. Read FECS_STATUS (0x409800) — check for response
5. Try commands: DISCOVER_IMAGE_SIZE, SET_WATCHDOG_TIMEOUT

**FECS method protocol** (from nouveau gf100_gr_fecs_*):
- Write arg to 0x409500
- Write method to 0x409504
- Poll 0x409800 for completion (bit-specific to method)

### 123-T2: PFIFO Channel Discovery
**Goal**: Discover which channels nouveau is using and find an unused slot.

Read PFIFO state:
- PFIFO_CTRL (0x2200) — is PFIFO enabled?
- Channel table scan (0x800000 + chid*8) — which channels have bit 31 set?
- Runlist state (0x2270/0x2274)

### 123-T3: Shadow Channel Attempt (if 123-T1 succeeds)
**Goal**: Create a compute channel behind nouveau's back.

This is the ambitious part:
1. Allocate a sysmem buffer for channel instance (RAMFC + USERD + GPFIFO)
2. Build GV100-style page tables pointing at sysmem compute buffers
3. Bind a new channel at an unused channel ID
4. Submit a trivial compute dispatch via GPFIFO

**Risk**: High — could crash nouveau or hang the GPU.

## Critical Question

Is nouveau's FECS/GPCCS actually alive? Exp 122B showed Titan #2's FECS in
HRESET under nouveau. This needs to be re-verified on both Titans, as the
Titans may differ (Titan #1 was the primary card in earlier experiments).

If FECS is alive on Titan #1 → proceed. If both are HRESET → parasitic path is dead.

## Fallback

If parasitic compute doesn't work on Titan V, the K80 becomes the sole priority.
Once K80 sovereign compute is validated, we can re-approach Titan V with the
complete knowledge of what the compute dispatch path looks like.
