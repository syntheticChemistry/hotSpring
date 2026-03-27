# Experiment 100: ACR DMA IOMMU Fix (Path Q)

**Date:** 2026-03-25
**Status:** COMPLETED — IOMMU faults eliminated, DMA trap persists at TRACEPC=0x0500

## Objective

Fix the DMA trap (`EXCI=0x201F0000`) blocking ACR firmware in HS mode. IOMMU
fault log revealed IO_PAGE_FAULT at 0x26000-0x28000 — IOVAs without IOMMU backing.

## Changes

1. **LOW_CATCH buffer** (0x0..0x40000): 256 KiB catch-all IOMMU backing for low VAs
2. **HIGH_CATCH buffer** (WPR_end..2MiB): covers all remaining unmapped ranges
3. **Mid-gap buffers**: fills holes between ACR, shadow, and WPR regions
4. **PT0 page 0 mapping**: included page 0 in identity map

## Results

- IOMMU faults eliminated (no more IO_PAGE_FAULT in dmesg)
- DMA trap persisted: EXCI=0x201F0000, TRACEPC stuck at 0x0500
- Hypothesis: internal falcon MMU fault, not IOMMU

## Sub-experiments

- **v1-v5**: Progressive IOVA coverage improvements
- **v6**: Confirmed SEC2 not starting from clean HRESET in early runs
- **VRAM mirror integration**: WPR + shadow mirrored to VRAM via PRAMIN

## Conclusion

IOMMU layer is clean. The DMA trap is INTERNAL to the falcon's MMU walker.
Page tables in sysmem + HS mode = possible circular dependency for MMU walker.
