# Experiment 102: Fix DMEM Data Loading

**Date:** 2026-03-25
**Status:** COMPLETED — Exhaustive DMA configuration sweep; DMA trap invariant to all changes

## Objective

Investigate why DMEM at 0x200 reads `0xDEAD5EC2` after boot, suggesting the
bootloader's data DMA (xdld) fails in HS mode.

## Variants

### v1: data_size=0 (skip data xdld)
- Set data_size=0 in BL descriptor to prevent BL from DMA-loading data
- Pre-boot DMEM confirmed correct (our pre-loaded data present)
- Post-boot DMEM still `0xDEAD5EC2` — confirms the wipe is from HS transition
  or HS read protection, NOT from failed BL xdld

### v2: No FBIF Override + DMA State Diagnostics
- FBIF stays at 0x110 (natural default) through HS transition
- All DMA configurations persist — HS transition does NOT modify FBIF/ITFEN/DMACTL
- DMA trap still at TRACEPC=0x0500

### v3: Dual-Resident Page Tables (sysmem + VRAM mirrors)
- Mirrored all page table structures to VRAM at their IOVA addresses
- All VRAM mirrors confirmed successful
- DMA trap persisted

### v4: VRAM Instance Block Binding (target=0)
- Changed bind_inst target from sysmem (2) to VRAM (0)
- VRAM binding accepted, HS mode achieved
- DMA trap still at TRACEPC=0x0500

### v5: PHYS DMA + FBIF PHYS_VID
- Set FBIF to 0x190 and ctx_dma to 0 (PHYS)
- DMA trap persisted at TRACEPC=0x0500

## Conclusion

The DMA trap at TRACEPC=0x0500 is **invariant** to: page table location,
binding target, FBIF mode, ctx_dma index, and data_size. This pointed to
a deeper issue: either GPU memory subsystem state or fundamental page table
format error.
