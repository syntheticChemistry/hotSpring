# Experiment 089c: Warm Swap Confirmation — Path A Closed

**Date:** 2026-03-24
**Layer:** L10 (GPCCS bootstrap)
**Status:** **CONFIRMED** — nouveau teardown kills GPCCS (Path A closed)
**Titan:** #2 (0000:4a:00.0), cold→nouveau→bare

## Problem

Path A hypothesis: GPCCS failure at `PC=0x0000, exci=0x00070000` is caused by
PCI FLR during VFIO bind. If we preserve state across the transition, GPCCS
should survive. Ember already clears `reset_method` to prevent FLR.

## Experiment Design

1. Capture Titan #2 BARE-COLD state (never had nouveau, no driver bound)
2. Bind to nouveau → ACR bootstraps FECS/GPCCS
3. Unbind from nouveau → capture BARE-WARM state
4. Compare: did GPCCS survive nouveau's teardown?

## Results

### BARE-COLD (via VFIO test, prior session)
```
FECS:  cpuctl=0xbadf1201 pc=0xbadf1201  ← PRI timeout, GR engine not powered
GPCCS: cpuctl=0xbadf1200 pc=0xbadf1200  ← PRI timeout
SEC2:  cpuctl=0x00000010 pc=0x000005e4  ← STOPPED
```

### BARE-WARM (after nouveau ran and unbound, direct BAR0 mmap)
```
FECS:  cpuctl=0x00000010 pc=0x000006ca sctl=0x00003000
GPCCS: cpuctl=0x00000012 pc=0x00000000 sctl=0x00003000 exci=0x00070000
SEC2:  cpuctl=0x00000010 pc=0x000006ca mb0=0x00000000 mb1=0x00000000
PMC:   enable=0x5fecdff1 intr=0x40000100
```

## Analysis

**GPCCS fails identically on both Titans after nouveau unbind.** The failure
pattern matches Titan #1 exactly:
- `cpuctl=0x12`: STOPPED + STARTCPU sticky (ACR started it, but it halted)
- `pc=0x0000`: Faulted immediately
- `sctl=0x3000`: LS mode (set by ACR)
- `exci=0x00070000`: HS authentication exception

This **independently confirms Exp 079's finding**: nouveau's `gf100_gr_fini()`
explicitly halts all falcons during unbind. Ember's `reset_method` clearing
prevents PCIe FLR but cannot prevent the driver-level teardown.

**Path A is definitively closed.** The FLR hypothesis was wrong — the damage
happens at the driver level, not the PCI level.

### State Comparison: Cold vs Post-Nouveau

| Falcon | Cold (no nouveau) | After nouveau unbind |
|--------|-------------------|---------------------|
| FECS   | PRI timeout (engine off) | cpuctl=0x10, PC=0x6ca (halted mid-exec) |
| GPCCS  | PRI timeout (engine off) | cpuctl=0x12, PC=0x0, exci=0x00070000 |
| SEC2   | cpuctl=0x10, PC=0x5e4 | cpuctl=0x10, PC=0x6ca |
| GR engine | Off (PMC disabled) | On (PMC enable=0x5fecd...) |

Notable: nouveau's teardown does NOT fully HRESET the falcons (unlike Exp 079
which saw clean HRESET). PC values are nonzero for FECS/SEC2, suggesting
firmware is still in IMEM but halted. GPCCS is the exception — it faulted
at PC=0 before it could advance.

### Key Difference from Exp 079

Exp 079 saw clean HRESET (cpuctl=0x10, bootvec=0x0). Now we see cpuctl=0x12
(STARTCPU bit persists) and PC=0x6ca. This could be due to:
1. Different kernel version (different nouveau teardown path)
2. Ember's reset_method clearing preventing a secondary reset
3. Race condition in teardown timing

The difference doesn't matter for Path A — GPCCS is dead either way.

## Critical Insight: Exp 088 Worked

In Exp 088, our OWN ACR boot (strategy_mailbox.rs) transitioned GPCCS cpuctl to 0x00:
```
GPCCS after STARTCPU: cpuctl=0x00000000
GPCCS final: cpuctl=0x00000000
```

> **CAVEAT (Exp 090):** cpuctl=0x00 does NOT mean "RUNNING." Exp 090 Phase 2
> revealed GPCCS immediately faults at PC=0x0000 with exci=0x08070000 despite
> cpuctl=0x00. The falcon left HRESET but never successfully executed firmware.
> The "successful start" claim here was based solely on cpuctl, which is an
> insufficient diagnostic. PC and EXCI must always be verified.

## Path Forward

1. **Reproduce Exp 088:** Fresh nouveau warmup → VFIO → our ACR → STARTCPU.
   If GPCCS runs, the solution is already in strategy_mailbox.rs.

2. **Path B (SEC2 full boot):** Fix bind_stat strategies 1-4 for truly
   sovereign GPCCS boot without nouveau dependency.

## Script Fix

The `exp089b_warm_swap_test.sh` script had a BAR0 mmap size bug (used 32MB
instead of actual 16MB). Fixed to use `os.fstat(fd).st_size` for dynamic sizing.
