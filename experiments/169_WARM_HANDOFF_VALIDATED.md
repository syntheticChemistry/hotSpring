# Experiment 169 — Warm Handoff Validated (Titan V)

**Date**: 2026-04-16
**Status**: PASS (stages 1-3), EXPECTED FAIL (falcon_boot)
**GPU**: Titan V (GV100, 0000:03:00.0)

## Goal

Validate the full warm handoff cycle using ember's driver swap: vfio-pci → nouveau → vfio-pci, then run sovereign_init on the warm GPU.

## Procedure

1. Start ember with glowplug.toml config → holds Titan V on VFIO (iommufd)
2. `ember.swap(bdf, "nouveau")` — 7.6s, successful PCI rescan
3. nouveau probes GV100: 12288 MiB HBM2 detected, PMU firmware unavailable (expected)
4. Wait 5s for stabilization
5. `ember.swap(bdf, "vfio-pci")` — 14.3s, successful PCI rescan + vfio-pci re-bind
6. `ember.list` confirms device re-held
7. `mmio.bar0.probe` confirms warm state:
   - boot0 = 0x140000a1 (GV100)
   - pmc_enable = 0x5fecdff1 (engines warm)
   - ptimer counting (HBM2 alive)
8. `ember.sovereign.init` — 5.3s total

## Results

| Stage | Status | Detail |
|-------|--------|--------|
| bar0_probe | OK | boot0=0x140000a1 chip=0x140 |
| pmc_enable | OK | before=0x5fecdff1 after=0x5fecdff1 (already warm) |
| hbm2_training | OK | 5 register writes (warm GPU, minimal) |
| falcon_boot | FAILED | FECS cpuctl=0x00000012 mb0=0x00000000 running=false |

## Key Findings

- **Warm handoff works**: nouveau trains HBM2, swap back preserves warm state
- **No system locks**: PCI rescan path (not driver/unbind) is stable on Volta+
- **HBM2 persists**: pmc_enable unchanged after swap, ptimer alive
- **Falcon is the blocker**: FECS is not running after nouveau teardown. cpuctl=0x12 means falcon is halted. nouveau doesn't leave FECS in a useful state for sovereign dispatch
- **ember survived full cycle**: no crashes, no D-state

## Next Steps

- Investigate FECS bootstrap: need to load ucode and kick FECS via ctxsw manager
- Check if nouveau's gr_init can be replayed, or if we need ACR (SEC2 → FECS)
- Consider capturing FECS ucode from nouveau's loaded state before swap
