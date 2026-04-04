# Experiment 090: Layer 10 PARTIAL — cpuctl=0x00 Was Misleading

**Date:** 2026-03-25
**Layer:** L10 (GPCCS bootstrap)
**Status:** **PARTIAL** — cpuctl=0x00 but GPCCS faulted at PC=0x0000
**Titan:** #1 (0000:03:00.0), post-nouveau warmup via GlowPlug

## Problem

Exp 089b identified GPCCS stuck at `cpuctl=0x12, PC=0x0000, exci=0x00070000` with
LS-mode locked CPUCTL. Three paths were proposed. Path A (disable FLR) was closed
by Exp 089c — nouveau teardown kills falcons, not FLR. The key insight was that
Exp 088 had already SOLVED this but the finding was obscured by later diagnostic
tests that examined residual state without running our ACR.

## Approach — Reproduce Exp 088

1. Fresh reboot (clean state after D-state hang from 089c script)
2. `coralctl swap 0000:03:00.0 nouveau` → nouveau ACR bootstraps falcons
3. Wait 10s for full GR init
4. `coralctl swap 0000:03:00.0 vfio` → GlowPlug swap back (Ember clears reset_method)
5. `cargo test vfio_falcon_boot_solver` → our ACR (strategy_mailbox.rs) re-bootstraps

## Results

### GlowPlug Swap Sequence
```
coralctl swap 0000:03:00.0 nouveau → ok (6.5s, /dev/dri/card1)
[10s settle]
coralctl swap 0000:03:00.0 vfio → ok (6.0s, vram_alive=true)
```

### Falcon Boot Solver Output
```
BOOTSTRAP_FALCON mask=0x000c (FECS+GPCCS)
After BOOTSTRAP(mask): mb0=0x00000001 mb1=0x0000000c FECS=0x12 GPCCS=0x12
Pre-start: FECS cpuctl=0x12 GPCCS cpuctl=0x12
GPCCS after STARTCPU: cpuctl=0x00000000   ← RUNNING!
FECS RUNNING: cpuctl=0x00000000 (10ms)    ← RUNNING!
```

### Post-Solver State
```
FECS @ 0x00409000: IDLE
  cpuctl=0x00000000 secure=false irqstat=0x00000000 curctx=0x00000002
GPCCS @ 0x0041a000: IDLE
  cpuctl=0x00000000 secure=false irqstat=0x00000000 curctx=0x00000002
SEC2 @ 0x00087000: RUNNING
  cpuctl=0x00000000 mb0=0x00000001 mb1=0x00000003
pgraph_status=0x00000081 (GR engine ACTIVE)
```

### CRITICAL: cpuctl=0x00 Was Misleading

Phase 2 diagnostics (same device, immediately after boot solver) reveal:
```
FECS:  cpuctl=0x00000000 pc=0x0000023c exci=0x08070000  ← STUCK at idle loop
GPCCS: cpuctl=0x00000000 pc=0x00000000 exci=0x08070000  ← FAULTED at PC=0
  GPCCS bootvec=0x00000000 dmactl=0x00000000 sctl=0x00003000 (LS mode)
  GPCCS IMEM[0..16]: 0x001400d0 0x0004fe00 0x957ea4bd 0x02f8002f (code present)
```

**GPCCS cpuctl=0x00 only means "not in HRESET/HALTED."** The falcon actually
faulted immediately at PC=0x0000 with exception 0x08070000. It never executed.

**FECS is stuck at PC=0x023c** (idle loop). It can't complete initialization
because GPCCS is dead. The 0x409800 ready signal never fires.

**FECS methods ALL timeout:**
```
Context image size: FAILED — FECS method 0x0010 timeout (2000ms)
Zcull image size:   FAILED — FECS method 0x0016 timeout (2000ms)
PM image size:      FAILED — FECS method 0x0025 timeout (2000ms)
Watchdog:           FAILED — FECS method 0x0021 timeout (2000ms)
```

**Confirmed on BOTH Titans (clean states).** Layer 10 is NOT solved.

### Dispatch Test
```
vfio_dispatch_nop_shader: FAILED — fence timeout (gp_get=0, gp_put=1)
```
Expected — channel can't run without FECS method interface.

## Why Exp 088's "SUCCESS" Was Misleading

Exp 088's falcon_probe only checks cpuctl, mailbox, and hwcfg — it does NOT
check PC or EXCI. The "IDLE (no mailbox)" status for GPCCS was based solely on
cpuctl=0x00. GPCCS was likely faulted at PC=0 even in Exp 088, but we never
verified the PC register. This error carried forward to Exp 090's initial
"SOLVED" assessment.

## GPCCS PC=0 Fault Analysis

The GPCCS exception (exci=0x08070000) at PC=0x0000 is the core unsolved problem.

**Key observations:**
1. BOOTVEC=0x0000 after STARTCPU — ACR should have set BOOTVEC=0x3400 (bl_imem_off for GPCCS)
2. IMEM[0] has code (from nouveau's previous load) — but this is APP code, not BL
3. sctl=0x3000 (LS mode) — HS authentication required for execution
4. exci=0x08070000 — bits 16-18 set (exception cause field)

**Hypotheses:**
- **H1: BL never ran** — ACR loaded BL to IMEM[0x3400] and set BOOTVEC=0x3400,
  but our STARTCPU cleared BOOTVEC to 0. GPCCS started at IMEM[0] (old APP code)
  which can't execute in the current authentication context.
- **H2: BL ran and loaded APP** — BL at 0x3400 executed correctly, loaded APP code
  to IMEM[0], jumped to APP entry. But APP code faults because LS authentication
  context is broken (WPR region invalid after nouveau teardown).
- **H3: IMEM[0x3400] is empty** — ACR's WPR copy (status=1) didn't actually load
  the BL to IMEM. GPCCS started at 0 because there was nothing at 0x3400.

**Next diagnostic: Read GPCCS TRACEPC and IMEM[0x3400..] to distinguish.**

## What Worked (Partial Success)

| Step | Status |
|------|--------|
| Nouveau warmup → GlowPlug swap → vram_alive=true | ✅ |
| SEC2 boots ACR firmware, reaches idle loop | ✅ |
| BOOTSTRAP_FALCON(FECS+GPCCS) via mailbox | ✅ (cpuctl 0x12) |
| Host STARTCPU → cpuctl 0x00 | ✅ (misleading) |
| GPCCS actually executes firmware | ❌ (PC=0, exci) |
| FECS completes init (0x409800 ready) | ❌ (stuck in idle) |
| FECS method interface responds | ❌ (all timeout) |

## Layer Status Update

| Layer | Status |
|-------|--------|
| L1-L5 | SOLVED |
| L6 | PARTIAL (GP_PUT DMA read — may be resolved by L11 GR context) |
| L7 | **SOLVED** (Exp 085, B1-B7) |
| L8 | **SOLVED** (Exp 087, W1-W7) |
| L9 | **PARTIAL** (Exp 088 — cpuctl=0x00 was misleading, see Phase 2 above) |
| L10 | **PARTIAL** (Exp 090 — GPCCS faults at PC=0, exci=0x08070000) |
| L11 | **BLOCKED** by L10 — FECS stuck at idle loop, GPCCS must be alive |

## Next: Debug GPCCS PC=0 Fault

GPCCS is NOT running despite cpuctl=0x00. The GR context init path requires:
1. **FECS method interface** — `0x409500-0x409804` should respond to commands
2. **DISCOVER_IMAGE_SIZE** → get golden context buffer size
3. **Golden context generation** — `WFI_GOLDEN_SAVE` or nouveau capture
4. **Channel context binding** — FECS `BIND_POINTER`
5. **GPFIFO dispatch** — nop shader through SM

## Safety Note: D-State Hang (Gap 13)

Exp 089c's raw sysfs swap script caused a D-state hang requiring forced power-off.
GlowPlug's atomic swap lifecycle prevented this in Exp 090. All future swaps must
go through GlowPlug/Ember, never raw sysfs scripts. See Gap 13 in gap tracker.
