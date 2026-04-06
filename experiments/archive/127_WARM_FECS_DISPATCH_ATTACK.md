# Experiment 127: Warm FECS Dispatch Attack

**Date**: 2026-03-30
**GPU**: Titan V (GV100, SM70) at 0000:03:00.0
**Goal**: Achieve Layer 7 sovereign compute dispatch via warm FECS handoff from nouveau.
**Result**: FECS firmware preserved in IMEM but **cannot be woken** from HS+ halt state.

## Background

Experiment 125 demonstrated that the kernel livepatch (`livepatch_nvkm_mc_reset`)
successfully preserves FECS/GPCCS firmware through the nouveau→vfio-pci driver swap.
This experiment attempts the next step: restarting the preserved falcons and dispatching
compute work through the GPFIFO pipeline.

## Execution Flow

### Phase 0: Deploy + Verify

- Built and deployed latest `coralReef` binaries via `deploy_all.sh`
- Confirmed `coralctl onboard` available, daemons running, livepatch enabled

### Phase 1: Warm FECS Cycle

```
coralctl warm-fecs 0000:03:00.0 --settle 20
```

Steps 0-2b completed successfully:
1. Livepatch disabled (direct write fallback — `coralreef-sysfs-write` rejects `/sys/kernel/livepatch/` path)
2. Swapped to nouveau — ACR chain boots FECS/GPCCS into HS+ mode
3. 20s settle for GR init
4. Livepatch re-enabled (4-function NOP: `mc_reset`, `gr_fini`, `falcon_fini`, `runl_commit`)

Step 3 failed: nouveau unbind timed out (10s D-state). Manual recovery:
```
coralreef-sysfs-write driver_override vfio-pci
coralreef-sysfs-write drivers/vfio-pci/bind 0000:03:00.0
coralctl swap 0000:03:00.0 vfio
```

### Phase 2: Post-Warm Register Snapshot

**FECS firmware survived the swap:**

| Register | Cold (pre-warm) | Warm (post-swap) |
|---|---|---|
| PMC_ENABLE (0x200) | `0x40000020` | `0x5fecdff1` |
| FECS CPUCTL (0x409100) | `0xbadf1201` (dead) | `0x00000010` (halted) |
| FECS PC (0x409030) | `0xbadf1201` | `0x00003621` |
| FECS SCTL (0x409240) | `0xbadf1201` | `0x00003000` (HS+) |
| GPCCS CPUCTL (0x41a100) | `0xbadf3000` (dead) | `0x00000010` (halted) |
| PFIFO (0x2004) | `0xbad0da00` (clock-gated) | `0x0020000e` (alive) |
| SEC2 CPUCTL (0x87100) | `0x00000010` | `0x00000010` |

23 engines powered (PMC_ENABLE bit count). FECS and GPCCS reachable via PRI.

### Phase 3: Warm Dispatch Test

```
CORALREEF_VFIO_BDF=0000:03:00.0 cargo test vfio_dispatch_warm_handoff
```

#### PFIFO Init (warm handoff config)
- PMC glow plug: skipped (engines already powered)
- PMC PFIFO reset: skipped (bit 8 already set)
- Runlist preempt: skipped (preserve FECS scheduling state)
- PBDMA force-clear: skipped (interrupt-only clear)
- Empty runlist flush: skipped
- PFIFO alive: ✅ preempt ACK received
- Channel created: ID=0, gpfifo=0x1000, userd=0x2000, instance=0x3000

#### FECS Wake Attempts

All wake strategies failed. FECS remained halted after each attempt.

**Strategy 1: SWGEN0 Software Interrupt**
```
IRQSCLR = 0xFFFFFFFF  (clear pending)
IRQMSET = 0xFFFFFFFF  (enable all interrupt sources)
IRQMODE = 0xFC24       (routing config from nouveau)
IRQSSET = 1 << 6      (trigger SWGEN0)
```
Result: `fecs_irq=0x00000040` (interrupt pending) but `fecs_cpuctl=0x00000012`, `fecs_pc` unchanged.
The interrupt was latched in IRQSTAT but the CPU did not wake.

**Strategy 2: STARTCPU via CPUCTL + CPUCTL_ALIAS**
```
ENGCTL = 0x00 (release engine reset — was already 0)
IRQSCLR = 0xFFFFFFFF
MAILBOX0/1 = 0
CPUCTL = IINVAL | STARTCPU (0x03)
CPUCTL_ALIAS = IINVAL | STARTCPU (0x03)
```
Result: CPUCTL readback = `0x00000012` (bit 1 latched = STARTCPU written, bit 4 = halted).
STARTCPU bit accepted but falcon did not start. PC unchanged, MAILBOX0 = 0.

#### FECS Method Interface
```
FECS method 0x0010 (discover image size) → timeout after 2000ms
MTHD_STATUS = 0x00000000, MTHD_STATUS2 = 0x00000000
```
The method write was never picked up. FECS firmware is not running its command loop.

#### Dispatch Result
Fence timeout after 5000ms. GPFIFO stalled at gp_get=0, gp_put=1.
PCCSR channel status: PENDING, enabled, busy — never reached RUNNING.
PBDMA GP_BASE = 0 (GPFIFO never bound to PBDMA by scheduler).

## Root Cause Analysis

### CPUCTL Bit 4: Firmware HALT, Not Hardware Reset

Our register constants had bit 4 labeled as `CPUCTL_HRESET` and bit 5 as
`CPUCTL_HALTED`. Per envytools falcon v4+ documentation:
- Bit 4 = HALTED (CPU is halted)
- Bit 5 = STOPPED (CPU idle/stopped)

FECS CPUCTL = 0x10 means the falcon CPU is HALTED. With ENGCTL = 0 (no engine
reset) and PMC enabled, this halt is from the **firmware executing a HALT
instruction** during its idle loop.

### HS+ Lockdown Prevents Host Wake

FECS SCTL = 0x3000 → bits[13:12] = 0b11 = HS+ (highest security mode).

On Volta GV100 HS+ falcons:
- CPUCTL is partially locked — STARTCPU writes latch (bit 1 appears in readback)
  but do not actually start the CPU
- Software interrupts (SWGEN0 via IRQSSET) are latched in IRQSTAT but do not
  wake the CPU from halt
- The ACR security engine controls the falcon lifecycle; host-initiated restart
  is blocked by design

### FECS Was Already Halted Before Teardown

The livepatch preserves the falcon state during nouveau unbind. But FECS enters
its idle HALT loop during normal nouveau operation when no compute workload is
active. After the 20s settle period, FECS is halted in its idle loop.

The livepatch correctly prevents:
- `nvkm_mc_reset` — PMC engine reset
- `gf100_gr_fini` — GR engine cleanup
- `nvkm_falcon_fini` — falcon ENGCTL/CPUCTL shutdown
- `gk104_runl_commit` — runlist modification

But it cannot prevent the pre-existing firmware HALT that occurred during normal
idle operation.

### PBDMA Cascade

With FECS halted, the PFIFO scheduler cannot process runlist updates for the GR
engine. Our channel submission reaches PCCSR (PENDING state) but FECS never loads
it into a PBDMA. PBDMA method/data registers return `0xbad00200` (PRI timeout) and
PBDMA[1] accumulates interrupt `0x2040c028` (DEVICE error from trying to contact
halted GR engine).

## Confirmed Facts

| Item | Status |
|---|---|
| FECS IMEM preserved across swap | ✅ Confirmed |
| GPCCS IMEM preserved across swap | ✅ Confirmed |
| PMC_ENABLE engines preserved | ✅ 23 engines powered |
| PFIFO alive (preempt ACK) | ✅ Confirmed |
| Channel created + runlist submitted | ✅ PCCSR shows PENDING |
| FECS restartable via STARTCPU | ❌ HS+ lockdown blocks |
| FECS wakeable via SWGEN0 interrupt | ❌ Interrupt latched but CPU frozen |
| FECS method interface responsive | ❌ Status = 0 (never picked up) |
| Compute dispatch | ❌ Fence timeout |

## Implications + Next Steps

### The Problem Is Not Preservation — It's Resumption

The livepatch approach successfully preserves all falcon state (IMEM, DMEM,
registers, PMC_ENABLE). The problem is that **HS+ security mode prevents the host
from resuming a halted falcon**. This is an architectural security feature, not a bug.

### Candidate Solutions

1. **Keep FECS running during swap**: Instead of the current approach where FECS
   halts during idle and we try to resume it, find a way to prevent FECS from
   entering its idle HALT. Options:
   - Submit a continuous no-op workload to FECS before the swap
   - Modify the FECS firmware's halt behavior (blocked by HS+ signed firmware)
   - Use a kernel timer to periodically poke FECS via methods

2. **Re-run ACR secure boot**: After the swap, restart the full ACR chain
   (SEC2 → HS bootloader → FECS firmware load). Requires:
   - SEC2 to be restartable (also HS+, same lockdown issue)
   - ACR payload accessible in VRAM
   - Full secure boot protocol implementation in coral-driver

3. **Hybrid approach — nouveau stays partially loaded**: Instead of fully unbinding
   nouveau, keep it loaded but with BAR0 access shared. FECS continues running
   under nouveau's control while coral-driver accesses BAR0 for dispatch.

4. **Target pre-HS+ generations**: K80 (Kepler) uses PIO-mode falcons without
   HS lockdown. FECS can be directly loaded and started via IMEMC/IMEMD writes.
   Cold boot dispatch on K80 sidesteps the HS+ issue entirely.

5. **PMC GR reset + ACR reboot**: Toggle PMC_ENABLE bit 12 (GR) to fully reset
   the GR engine, then re-run the ACR boot sequence to reload FECS from scratch.
   The firmware is preserved in VRAM's WPR2 region — if we can trigger SEC2 to
   re-authenticate and re-load it, FECS would boot fresh.

### Code Changes Made

- `init.rs:restart_warm_falcons()` — integrated `warm_start_falcon()` call
  (was dead code, never called)
- Added SWGEN0 interrupt wake strategy before STARTCPU fallback
- Added PBDMA interrupt clearing after falcon wake attempts
- Added IRQMSET (not IRQMCLR) for interrupt mask configuration
- Added diagnostic logging for all wake strategy results
