# Exp 098: Full Init Conversation (Path O)

**Date:** 2026-03-26
**Status:** COMPLETE — DMA trap identified, WPR copy partial
**Depends on:** Exp 097 (EMEM discovery), Exp 095 (HS mode)

## Objective

Remove `blob_size=0` optimization and attempt full ACR init to keep SEC2
RUNNING with CMDQ/MSGQ active.

## Key Discovery: `patch_acr_desc` Already Sets blob_base/blob_size

```
Original ACR desc: blob_size=0xcd00 blob_base=0x70000
```

- blob_size = 0xCD00 = 52,480 bytes (exactly the WPR data size)
- blob_base = 0x70000 = WPR IOVA (set by `patch_acr_desc`)

The `patch_acr_desc` function already patches the ACR descriptor so that
blob_base points to the WPR in system memory. The `blob_size=0` override
was just a safety measure to skip the DMA that trapped.

## Results

### Run 1: Double-boot (after boot solver) — FAILURE

Running the full-init strategy AFTER the boot solver produced:
- bind_stat TIMEOUT (0x008e043f) — HS lockdown blocks re-binding
- EMEM wiped to zero — SEC2 reset clears EMEM
- Firmware still exits to HRESET

**Root cause:** Cannot re-bind an already-HS falcon.

### Run 2: Clean boot (nouveau cycle → direct full-init) — PARTIAL SUCCESS

Starting from clean LS state via GlowPlug nouveau cycle:

| Metric | Value | Notes |
|--------|-------|-------|
| Pre-boot SCTL | 0x3000 (LS) | Clean start |
| bind_stat→5 | OK in 66µs | Binding works from LS! |
| Post-boot SCTL | 0x3002 (HS) | Entered HS |
| EXCI | 0x201F0000 | DMA access violation (0x20) + breakpoint (0x1F) |
| cpuctl | 0x10 (HRESET) | Firmware exited |
| WPR FECS | 1 (copy started) | Not 0xFF (done) |
| WPR GPCCS | 1 (copy started) | Not 0xFF (done) |
| TRACEPC | All 0x0500 | HS boot loop |
| EMEM init msg | Present at 0x80 | Confirms firmware writes it during boot |

### Run 3: blob_base patched to ACR IOVA — SAME RESULT

Changed blob_base from 0x70000 (WPR) to 0x46000 (ACR payload).
Same EXCI=0x201F0000, different PC (0x4CD8 vs 0x4CBE). Firmware reads the
patched value but still traps.

## Analysis

The ACR firmware on GV100/Volta is a **one-shot loader**, not a persistent daemon:

1. BL enters HS at BOOTVEC
2. BL validates HS environment (0x0500 loop)
3. BL reads blob from DMA (blob_base → WPR at IOVA 0x70000)
4. BL processes WPR headers, initiates FECS/GPCCS copy (byte=1)
5. **DMA FAILS** during the copy — EXCI=0x201F0000
6. Firmware traps and enters HRESET

The DMA failure occurs when the ACR firmware tries to copy authenticated images
from the WPR buffer to the FECS/GPCCS falcons. This is a DMA transfer within
the GPU — SEC2's DMA engine writing to FECS/GPCCS IMEM.

**Why the DMA fails:**

The SEC2 falcon's DMA engine translates addresses through:
1. Falcon MMU (VA → IOVA via page tables)
2. System IOMMU (IOVA → host physical)

Our page tables identity-map VA 0..2MiB → IOVA. All our DMA buffers (WPR at
0x70000, shadow at 0x60000) are within this range. However, the ACR firmware may
need to DMA to addresses outside this range:
- FECS IMEM might be at a different VA/IOVA
- The ACR might use a higher VA for scratch space
- The WPR copy might go through a VRAM path that our FBHUB-dead setup can't serve

## Next Steps → Path Q

**Path Q: Investigate the DMA path from SEC2 → FECS/GPCCS.**

1. Check IOMMU fault log after boot to see what VA/IOVA the DMA tried to access
2. Extend page tables beyond 2 MiB to cover potential DMA targets
3. Check if ACR uses PRI (BAR0 registers) to write FECS/GPCCS IMEM, or DMA
4. If PRI: ensure FECS/GPCCS are in a state that accepts IMEM writes
5. If DMA: map the target falcon IMEM addresses in our page tables

**Alternative: Let nouveau handle the full ACR bootstrap, then just use the
post-ACR state.** Nouveau's 3-second bind window already runs the full ACR.
After swap back, FECS/GPCCS might already be loaded. Check their IMEM content
after the nouveau cycle.

## Files Changed

- `coralReef/.../strategy_sysmem.rs` — Added `attempt_sysmem_acr_boot_full()`,
  blob_size logging, conditional blob patching
- `coralReef/.../mod.rs` — Exported new function
- `coralReef/tests/.../sec2_emem_discovery.rs` — Added Exp 098 test
