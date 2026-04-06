# Exp 115: PMC GR Reset + Direct Falcon Upload / WPR Copy Analysis

**Date:** 2026-03-26
**Hardware:** Titan V (GV100), BDF 0000:03:00.0
**Depends on:** Exp 114 (LS mailbox stall), Exp 113 (HS PMU block), Exp 089b (PMC reset)

## Objective

Three-pronged attack on the FECS/GPCCS activation problem:
1. Bypass ACR entirely — direct PIO firmware upload after PMC GR reset
2. Test `acr_vram_pte=false` hypothesis for the WPR copy stall
3. Comprehensive analysis of all remaining paths

## Results Summary

| Variant | Method | GPCCS Result | FECS Result |
|---------|--------|-------------|-------------|
| A | PIO upload, BOOTVEC=0 | HRESET stays (cpuctl=0x12) | HRESET stays |
| B | PIO upload + BL, BOOTVEC=0x3400 | HRESET stays (cpuctl=0x12) | HRESET stays |
| C | `attempt_direct_falcon_upload` | HRESET stays | HRESET stays |
| D | `fecs_boot::boot_gpccs/fecs` | HRESET stays (timeout) | HRESET stays |
| E | LS ACR + acr_vram_pte=false | WPR stalled (status=1) | WPR stalled |

## Key Discovery 1: Hardware Security Enforcement (Variants A-D)

**GV100 FECS/GPCCS have hardware-level code authentication enforcement.**

After PMC GR reset:
- CPUCTL is writable (PMC reset breaks ACR CPUCTL lock — Exp 089b confirmed)
- Writing STARTCPU (bit 1) is accepted: CPUCTL goes from 0x10 to 0x12
- But **HRESET (bit 4) never clears** — the falcon refuses to start executing
- GPCCS PC stays at 0x0000 — no execution occurs
- EXCI changes from 0x00070000 to 0x03070000 — subtle internal state change

This proves PIO-uploaded code cannot execute on GV100 GR falcons. The hardware
validates authentication context (set by ACR during DMA-based loading) before
allowing execution. STARTCPU without proper authentication context is silently
rejected at the hardware level.

## Key Discovery 2: acr_vram_pte Has Zero Effect (Variant E)

Tested `acr_vram_pte=false` (all PTEs point to system memory, no VRAM PTEs):
- **Same result as Exp 114** (which used acr_vram_pte=true)
- WPR FECS=1 GPCCS=1 | Shadow FECS=1 GPCCS=1
- SEC2 running (PC=0x659d), BOOTSTRAP_FALCON acknowledged (mb0→1)
- But FECS/GPCCS remain in HRESET

## Key Discovery 3: WPR Status May Be Stale

Our `build_wpr()` initializes status=1 (COPY) for both FECS and GPCCS. We read
status from our own DMA buffer. The ACR firmware might update status in an internal
copy (VRAM WPR2 region) rather than writing back to our system memory buffer.

However, both the WPR buffer AND the shadow buffer show status=1, and DMA is
bidirectional in our IOMMU setup. If the ACR attempted a write-back, it would
reach our buffer. So the ACR likely never updated the status at all.

## Key Discovery 4: SEC2 Runs in LS Mode Even Under Nouveau

From Exp 089b, post-nouveau SEC2: `sctl=0x00003000` (LS mode). Yet nouveau's
ACR successfully copies firmware to FECS/GPCCS. **LS mode IS sufficient for
the WPR copy — our implementation has a different problem.**

## Complete Path Analysis

| Path | HS | WPR Copy | Start | Blocker |
|------|-----|----------|-------|---------|
| Legacy PDEs, blob=0 (Exp 095) | Yes | N/A | N/A | One-shot exit |
| Legacy PDEs, full init (Exp 098) | Yes | STALLED | No | TRAP during copy |
| Correct PDEs, full init (Exp 110) | No | STALLED | No | No HS |
| VRAM-native (Exp 111) | No | STALLED | No | No HS |
| Dual-phase (Exp 112/113) | Yes | STALLED | No | TRAP (PMU dep) |
| LS mailbox, vram_pte=true (Exp 114) | No | STALLED | No | Copy stalls |
| LS mailbox, vram_pte=false (Exp 115E) | No | STALLED | No | Copy stalls |
| Direct PIO upload (Exp 115A-D) | N/A | N/A | No | HW security |

**WPR copy stalls universally, regardless of HS/LS, VRAM/sysmem, PDE type.**

## Root Cause Hypothesis

The ACR WPR copy-to-target writes firmware to FECS/GPCCS IMEM via the PRIV bus.
The critical difference between our setup and nouveau's may be:

1. **WPR location**: Nouveau builds WPR in VRAM; we build it in system memory.
   Both the system memory and VRAM paths stall, so this alone isn't the cause.

2. **BL initialization context**: Nouveau's SEC2 boot path includes proper DMA
   context initialization via FWSEC from VBIOS. Our BL boot may lack the
   PRIV bus DMA context needed for cross-falcon writes.

3. **FWSEC/WPR2 state**: FWSEC carves WPR2 during GPU reset (via FRTS command).
   If WPR2 was invalidated during the nouveau→vfio swap, the ACR's copy mechanism
   might fail validation checks.

4. **Missing PMU**: On desktop GV100, PMU is loaded by FWSEC (no PMU firmware in
   `/lib/firmware/nvidia/gv100/`). If PMU was stopped during nouveau teardown,
   its WPR2 management may have been disrupted.

## Next Steps

1. **Exp 116: Investigate PRIV bus state** — Can SEC2 actually write to FECS IMEM
   via DMA? Test by checking if ACR modified any FECS/GPCCS registers.

2. **Exp 117: WPR in VRAM via PRIV** — Instead of system memory, build WPR entirely
   in VRAM via PRAMIN and boot with VRAM-only DMA path.

3. **Exp 118: Partial nouveau preservation** — Use GlowPlug to observe FECS/GPCCS
   state DURING nouveau (before teardown), then try to preserve state across swap.

4. **Research: nouveau WPR construction** — Compare our build_wpr() byte-for-byte
   with nouveau's gp102_acr_wpr_build() output. There may be a remaining format bug.
