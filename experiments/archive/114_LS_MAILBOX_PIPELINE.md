# Exp 114: LS-Mode FECS/GPCCS Activation (Path Y)

**Date:** 2026-03-26
**Hardware:** Titan V (GV100), BDF 0000:03:00.0
**Depends on:** Exp 113 (PMU dependency), Exp 091 (BOOTVEC fix)

## Objective

After Exp 113 proved the HS ACR chain is blocked by PMU dependency, shift to
Path Y: LS-mode mailbox path to activate FECS/GPCCS without HS mode.

Pipeline: Nouveau cycle → Sysmem ACR boot (correct PDEs, full init) →
BOOTSTRAP_FALCON(FECS+GPCCS) via mailbox → BOOTVEC fix → STARTCPU

## Key Results

| Phase | Outcome |
|-------|---------|
| Nouveau cycle | Clean state, all engines powered |
| Sysmem ACR boot | SEC2 running in LS mode (PC=0x64e2), 31 TRACEPC entries |
| BOOTSTRAP_FALCON | SEC2 **acknowledged** command (MB0 changed to 1) |
| FECS/GPCCS state | **Still in HRESET** (cpuctl=0x10), PC=0x0000, EXCI=0x03070000 |
| WPR copy status | **STALLED at 1** (COPY initiated, not 0xFF=DONE) |

## Detailed Observations

1. **SEC2 ACR runs extensively**: PC=0x64e2, not halted, not trapped. 31 TRACEPC
   entries showing full BL init + firmware execution. But MB0 never changes from
   sentinel (0xdeada5a5) — ACR didn't signal success or failure.

2. **BOOTSTRAP_FALCON acknowledged**: MB0 changed from sentinel to 0x1 after the
   mailbox command, confirming SEC2 processed the request. MB1=0x0c (FECS|GPCCS mask).

3. **WPR copy stalled**: Both FECS and GPCCS WPR header status = 1 (COPY initiated).
   Shadow copy also shows status=1. Copy initiated but never completed.

4. **GPCCS CPUCTL locked**: After BOOTSTRAP_FALCON, GPCCS CPUCTL writes are silently
   dropped (ACR established ownership). Host STARTCPU has no effect.

5. **Direct GPCCS boot fallback**: Attempted `fecs_boot::boot_gpccs` but failed due to
   CPUCTL lock — hardware security enforcement prevents host-loaded code execution.

## Configuration Used

```
BootConfig::full_init():
  pde_upper: true       (correct GV100 PDEs)
  acr_vram_pte: true    (VRAM PTEs for ACR code pages)
  blob_size_zero: false (full init — process WPR blob)
  bind_vram: false      (system memory bind)
  tlb_invalidate: true
```

## Analysis

The ACR firmware runs, processes the WPR blob, and acknowledges the BOOTSTRAP_FALCON
command — but the actual copy-to-target stalls. This is the same status=1 pattern
observed in HS mode (Exp 098, 112, 113), now confirmed in LS mode too.

Hypothesis: `acr_vram_pte: true` might misdirect ACR DMA to VRAM (garbage) instead
of system memory (where the actual WPR blob resides). → **TESTED in Exp 115E: BUSTED.
Setting acr_vram_pte=false had zero effect.**

## Status: WPR COPY STALLS IN ALL MODES

See Exp 115 for comprehensive analysis across all remaining paths.
