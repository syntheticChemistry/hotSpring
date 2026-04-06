# Exp 120: Sovereign DEVINIT + ACR Boot (Cold Boot Path)

**Date:** 2026-03-27
**Hardware:** Titan V #1 (GV100), BDF 0000:03:00.0
**Depends on:** Exp 119 (cold boot state), Exp 114-118 (WPR stall investigation)

## Hypothesis

With DEVINIT already complete (confirmed by devinit_reg bit 1), run ACR
boot on a clean GPU that nouveau never touched. If WPR copy stall was
caused by nouveau teardown or invalid WPR2, it should work now.

## Result: HYPOTHESIS REJECTED — WPR copy stalls REGARDLESS of boot path

### Key Findings

#### 1. DEVINIT Already Complete at Boot

```
devinit_reg = 0x00000002 (bit 1 set)
needs_post = false
HBM2 training: ALREADY DONE (FWSEC/PMU handled it)
```

The GPU's VBIOS ROM runs DEVINIT during PCI power-on. Memory is trained.
**We don't need nouveau for DEVINIT.** Our sovereign `execute_devinit()`
correctly detects "not needed" and skips.

#### 2. WPR Copy Stalls IDENTICALLY (PARADIGM SHIFT)

| Metric | Post-Nouveau (Exp 114-116) | Cold Boot (Exp 120) |
|--------|---------------------------|---------------------|
| bind_stat | 5 OK | 5 OK (66µs) |
| SEC2 running | Yes (PC=idle) | Yes (PC=0x00da) |
| SCTL | 0x3000 (LS) | 0x3000 (LS) |
| WPR FECS | 1 (copy initiated) | 1 (copy initiated) |
| WPR GPCCS | 1 (copy initiated) | 1 (copy initiated) |
| BOOTSTRAP ack | Yes | Yes (mb1=0x0c) |
| FECS/GPCCS | HRESET | HRESET |

**The WPR copy stall is IDENTICAL regardless of boot path.** This means:
- Invalid WPR2 is NOT the cause
- Nouveau teardown is NOT the cause
- The stall is intrinsic to our ACR boot configuration

#### 3. EMEM Contains Random Data

Cold boot EMEM: `0x2b531286 0xf2d3cc85 0x00c03080...` — uninitialized.
Post-nouveau EMEM: `0x00230406...` — structured ACR state.

This means EMEM content (previously thought to be ACR output) may have been
leftover from nouveau's SEC2, not from our ACR boot.

#### 4. SEC2 PC Trace Shows VRAM-Like Addresses

TRACEPC: `0xfd75 0xfd62 0xfd0a 0x0000 0xb82cf 0x207468 0x1fc90b...`

The first 3 values are BL code addresses. After `0x0000`, the remaining 27
entries contain values > 0x10000 — these exceed IMEM size (3 KB). They might
be:
- EMEM virtual addresses (DMA-mapped)
- VRAM physical addresses being accessed
- Stale TRACEPC register data from FWSEC's PMU

## Paradigm Shift: Root Cause Reanalysis

The WPR copy-to-target mechanism in the ACR firmware writes authenticated
firmware images from the WPR buffer to target falcon IMEM/DMEM. This write
uses Falcon DMA, which goes through the FBHUB MMU.

**Hypothesis A: DMA path from SEC2 to target falcons is not configured.**

The ACR firmware copies data via DMA. The DMA target for FECS IMEM/GPCCS
IMEM may require specific FBHUB/PRIV configuration that we haven't set up.
In nouveau, the full `gf100_gr_init()` sequence configures the GR engine
including PRIV ring access and falcon DMA contexts.

**Hypothesis B: PRIV ring errors block inter-falcon DMA.**

The `priv_ring_intr=0xbad00100` seen during PFIFO init suggests PRI ring
issues. The ACR firmware's copy-to-target may use PRI (privileged) ring for
falcon-to-falcon writes, which would fail if the PRI ring is faulted.

**Hypothesis C: Missing GR engine reset/enable sequence.**

The `open_vfio()` path does GR BAR0 init (sw_nonctx + dynamic), but the
cold boot FECS channel submission timed out ("expected on cold VFIO — GR
engine requires falcon firmware"). The GR engine may need to be in a
specific state for ACR to write to FECS/GPCCS IMEM.

## Next Steps

1. **Clear PRIV ring faults before ACR boot** — the 0xbad00100 interrupt
   might be blocking inter-engine DMA

2. **PMC GR reset before ACR** — reset the GR engine to clear any stale
   state, then attempt ACR

3. **Investigate the DMA path** — trace what the ACR firmware actually tries
   to write during copy-to-target. FBHUB fault registers may reveal the
   target address and failure mode.

4. **Minimal init** — Skip open_vfio() device init (PFIFO, channels) and
   just open BAR0 + IOMMU. The PFIFO init triggers PRIV ring faults that
   might poison ACR's DMA path.

## Closed Investigations

- **WPR2 boundaries (Exp 116-118)**: NOT the cause. Stall occurs with AND
  without valid WPR2.
- **nouveau teardown (Exp 117-118)**: NOT the cause. Same behavior on cold
  boot.
- **Cold vfio-pci boot (Exp 119)**: DEVINIT already runs at power-on.
  No additional action needed.
- **blob_size (Exp 116)**: Firmware binary already has blob_size=0. Our
  patch is correct.

---

## CORRECTION (April 3, 2026 — Exp 141)

**The conclusion "We don't need DEVINIT" is CONTEXT-DEPENDENT and partially wrong.**

Exp 120 tested on a BIOS-POSTed GPU (no SBR). In that context, `devinit_reg`
showed bit 1 set and `needs_post=false` — the VBIOS ROM had already run DEVINIT
during system POST. This finding is correct for warm GPUs.

However, **Exp 141 proved that after SBR (Subsystem Bridge Reset), VBIOS DEVINIT
IS the root cause of ACR HS authentication failure.** SBR resets the GPU to cold
state — PLLs, clocks, crypto engine, and memory controllers are uninitialized.
The `sovereign_boot()` recipe replay can maintain register state but cannot
establish it from cold. Without VBIOS DEVINIT, the crypto engine used for HS
signature verification is never initialized, causing the 0x2d78 auth loop.

**Updated conclusion**: VBIOS DEVINIT is not needed when the GPU is warm
(BIOS-POSTed). It IS needed after SBR or any cold boot. `sovereign_boot()` now
runs VBIOS DEVINIT as Phase 0 before recipe replay (see Exp 142).
