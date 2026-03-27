# Experiment 099: Post-Nouveau Falcon State (Path R)

**Date:** 2026-03-25
**Status:** COMPLETED — FLR wipes all falcon memory, ruling out inheriting nouveau's work

## Objective

Check if nouveau's firmware loading survives the VFIO handoff. If authenticated
FECS/GPCCS firmware persists after FLR, we can skip our ACR boot entirely.

## Method

1. Bind GPU to nouveau via GlowPlug (full devinit, firmware load)
2. Swap to vfio-pci (triggers FLR)
3. Probe SEC2, FECS, GPCCS IMEM and DMEM for residual firmware

## Results

- All falcon IMEM/DMEM reads zero after FLR
- SEC2 in HRESET state, no residual HS firmware
- FECS/GPCCS fully wiped

## Conclusion

**Path R dead.** FLR performs a complete memory scrub on all falcons. Our ACR boot
is necessary. Pivoted to Path Q (fix DMA mapping for ACR boot).
