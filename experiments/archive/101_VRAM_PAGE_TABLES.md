# Experiment 101: VRAM Page Tables + Instance Block

**Date:** 2026-03-25
**Status:** COMPLETED — VRAM page tables enable extensive code execution but prevent HS authentication

## Objective

Move the entire page table chain (PD3→PD2→PD1→PD0→PT0) and instance block
to VRAM, eliminating the circular dependency where the MMU walker can't reach
sysmem page tables in HS mode.

## Variants

### v1: Full VRAM (inst + PTs + ACR payload)
- 31 unique PCs executed (most progress ever at the time)
- **Failed to enter HS mode** (SCTL=0x3000)
- HS authentication appears to require sysmem instance block

### v2: Hybrid (sysmem inst, VRAM PTs)
- Same result as v1 — no HS mode
- PDB aperture (VRAM vs sysmem) is critical for authentication

### v3: FBIF Maintenance Loop
- Attempted to continuously re-apply FBIF_PHYSICAL_OVERRIDE after STARTCPU
- Hardware rejected PHYSICAL_OVERRIDE (bit 7) from BAR0 post-STARTCPU

### v4: FBIF Probe Before STARTCPU
- FBIF_TRANSCFG successfully set to 0x190 before STARTCPU
- Value persisted after STARTCPU and into crash
- HS mode achieved BUT DMA trap still at TRACEPC=0x0500
- **Disproved the FBIF circular dependency theory**

## Conclusion

The DMA trap is not caused by FBIF configuration or page table location.
DMEM at 0x200 showed `0xDEAD5EC2` (wiped or protected). Focus shifted to
DMEM data loading and BL descriptor configuration.
