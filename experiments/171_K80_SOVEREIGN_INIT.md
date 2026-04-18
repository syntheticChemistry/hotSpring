# Experiment 171 — K80 Sovereign Init (Kepler GK210)

**Date**: 2026-04-16
**Status**: PARTIAL — BAR0 probe + PMC enable OK, GDDR5 training BLOCKED
**GPU**: Tesla K80 (GK210 × 2, 0000:4c:00.0 + 0000:4d:00.0)

## Goal

Validate the sovereign init pipeline on Kepler architecture (SM 3.5/3.7, GDDR5) to
identify differences from the Volta/HBM2 path.

## Setup

- K80s were unbound after reboot (clean — no previous EBUSY issues)
- Manually bound to vfio-pci via sysfs driver_override
- Ember successfully holds both K80 VFIO groups (37, 38) via legacy path
- No kernel cmdline `vfio-pci.ids` for K80 devices

## Results

Both K80 GPUs (0000:4c:00.0, 0000:4d:00.0) show identical behavior:

| Stage | Status | Detail |
|-------|--------|--------|
| bar0_probe | OK | boot0=0x0f22d0a1, chip=0x0f2 (GK210) |
| pmc_enable | OK | before=0xc0002020, after=0xfc37b1ef |
| hbm2_training | FAILED | PRAMIN returns 0xbad0fb0* (PCIe completion timeout) |

## Analysis

- **BAR0 is alive**: boot0=0x0f22d0a1 confirms GK210 silicon, PMC write succeeds
- **Memory controller is cold**: UEFI didn't POST the K80 (secondary GPU). PRAMIN returns
  PCIe completion timeout markers (0xbad0fbXX) — memory is completely uninitialized
- **PMC_ENABLE expanded**: from 0xc0002020 (2 bits) to 0xfc37b1ef (many engines on),
  meaning the hardware responds to PMC writes even cold
- **GDDR5 vs HBM2**: K80 uses GDDR5, which has a different memory controller than
  HBM2. The current typestate pipeline targets HBM2 register domains (FBPA, LTC).
  K80 needs Kepler-specific memory init (PDISP/PFB/PMU DEVINIT)
- **nouveau cannot warm K80**: nouveau rejects cold GK210 with "unknown chipset"
  because BOOT0=0x0f22d0a1 maps to an unrecognized variant. The warm-cycle path
  via nouveau is NOT available for K80

## K80-Specific Init Path

K80 needs one of:
1. **DEVINIT via VBIOS**: Execute the VBIOS init script tables on the PMU FALCON
   to initialize GDDR5. The VBIOS is in PROM at BAR0+0x300000
2. **nvidia-470 via VM**: Pass K80 through to a VM with nvidia-470 driver, which
   will POST and train GDDR5, then reclaim via vfio-pci
3. **Manual register programming**: Reverse-engineer the GK210 GDDR5 init sequence
   and replay it. Most complex but fully sovereign

## Next Steps

- Read K80 VBIOS from PROM (ember.vbios.read)
- Attempt DEVINIT execution on K80 via ember.devinit.execute
- If DEVINIT works: K80 GDDR5 should initialize, then re-run sovereign init
- If not: fall back to nvidia-470 VM warm path (agentReagents)
