# Experiment 105: TLB Invalidation Fix — First -omics Alignment Result

**Date:** 2026-03-25
**Status:** CODE READY — awaiting hardware run
**Method:** Register trace alignment (COMPUTATIONAL_OMICS.md Phase 1)

## Discovery

Aligned nouveau's 32,507-operation mmiotrace against our sovereign boot register
writes. The alignment revealed:

### Missing Gene: TLB Invalidation

Nouveau performs a 3-register TLB invalidation sequence **54 times** during boot.
We performed it **zero times** in our ACR boot path.

```
0x100CB8 = PDB address for invalidation (addr >> 12 << 4)
0x100CEC = high 32 bits of invalidation address (always 0)
0x100CBC = trigger (0x80000005 = PAGE_ALL | HUB_ONLY | bit31)
```

### Substrate Confusion: Register 0x100CEC

We were treating `0x100CEC` as `NV_PFB_PRI_MMU_WPR2_ADDR_LO` and writing our
WPR base address to it. But the mmiotrace and our own `pfifo.rs` code both use
this register as the **high 32-bit address field for TLB invalidation**.

Nouveau ALWAYS writes 0 to `0x100CEC`. We were writing non-zero WPR values,
corrupting the TLB invalidation mechanism. When firmware internally tries to
invalidate TLB entries, it would read our WPR base as the high address, targeting
the wrong physical memory region.

### Alignment Statistics

| Block | Nouveau Writes | Our Writes | Gap |
|-------|---------------|-----------|-----|
| NV_PMC (0x000xxx) | 4,129 | ~20 | Mostly interrupt handling |
| NV_PFB (0x100xxx) | 162 (all TLB) | WPR corruptions | **FIXED** |
| SEC2 (0x087xxx) | 0 | ~50 | Both DMA-driven — OK |
| NV_PMU (0x070xxx) | 137 | 0 | PMU not running — expected |
| PRIV_RING (0x120xxx) | 945 | 0 | PRI topology — inherited from nouveau |

## Changes

### `strategy_sysmem.rs`

1. **Added TLB invalidation after instance block binding** (both simple and full paths)
   - Uses the correct 3-register sequence: CB8 (PDB addr), CEC (high=0), CBC (trigger)
   - Polls 0x100C80 for flush acknowledgment
   - PDB address derived from our instance block IOVA

2. **Removed WPR2 writes to 0x100CEC/CF0** (both paths)
   - 0x100CEC is NOT WPR2_ADDR_LO — it's MMU_INVALIDATE_ALL high address
   - WPR2 boundaries are carried in the ACR descriptor via DMEM, not via these registers
   - Changed to read-only diagnostic

## Hypothesis

With correct TLB state:
- The falcon's MMU walker sees current page table entries (not stale TLB cache)
- DMA during HS authentication resolves to the correct physical addresses
- The BL can verify the HS code signature from clean data
- HS mode transition succeeds → ACR bootstraps FECS/GPCCS

## How This Was Found

This is the first result from the COMPUTATIONAL_OMICS framework (specs/COMPUTATIONAL_OMICS.md).
Register trace alignment treated the mmiotrace as "genome A" and our sovereign boot
as "genome C", identified conserved regions (TLB invalidation = present in A, absent in C),
and a substrate confusion (0x100CEC dual identity = reading frame error).
