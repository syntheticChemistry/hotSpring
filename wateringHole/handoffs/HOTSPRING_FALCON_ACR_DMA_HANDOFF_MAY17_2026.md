# hotSpring Handoff — Falcon ACR DMA Boot Solved

**Date:** May 17, 2026 (PM)
**Experiment:** 206
**Status:** Falcon boot blocker SOLVED — reboot needed for full warm pipeline

## What Changed

The `falcon_boot` sovereign stage has been blocked since Experiment 080:
`"ACR HS boot requires DMA backend — not provided"`. The fix involved
three code changes and two infrastructure fixes in toadStool.

### Code Changes (toadStool upstream)

1. **`sovereign.rs`** — Stateless handler now extracts `DmaBackend` from
   VfioDevice and wires it into `SovereignInitOptions.dma_backend`.
   Also enters `EmberGateBypass` so the daemon's own VFIO opens aren't
   self-rejected by the ember gate.

2. **`sovereign_init.rs`** — `warm_fecs_preserved` detection now includes
   `"ACR boot OK"` to prevent `gr_init` from conflicting PIO re-upload
   after successful ACR firmware loading.

3. **`open.rs`** — iommufd/cdev failure log promoted from `debug` to
   `warn` for diagnostic visibility.

### Infrastructure Fixes

- **02:00.0 bound to vfio-pci** — Was previously unbound (no driver
  symlink). Used `driver_override` + `drivers_probe` to bind, enabling
  iommufd cdev (`vfio0`).

- **Stale coral-ember killed** — Legacy `coral-ember.service` and
  `coral-glowplug.service` were holding VFIO cdev fds, causing
  `DEVICE_BIND_IOMMUFD` to fail with EINVAL. Services disabled.

## Validation Result

TV2 (49:00.0), warm state:
```
falcon_boot: ok (3117ms)
  ACR boot OK: FECS cpuctl=0x00000010 mb0=0x00000000 (2 strategies)
```

Both cards acquired DMA backends via iommufd cdev after infrastructure
cleanup. DMA `boot_falcon_hs` confirmed working:
- Firmware blobs loaded from `/lib/firmware/nvidia/gv100/gr/`
- DMA buffers mapped via iommufd IOAS at fixed IOVAs
- FBIF TRANSCFG configured (stride=4, PHYS_SYS_COH)
- Bootloader uploaded to IMEM, descriptor to DMEM
- STARTCPU → falcon DMA-reads firmware → FECS enters command-wait

## Current State

Both GPUs are cold (PMC_ENABLE=0x40000020) after VFIO cdev opens
disturbed the warm UEFI-POSTed state. HBM2 uninitialized.

**Reboot required** to restore warm state. Expected post-reboot
pipeline:
```
identity_probe → pmc_enable → cg_sweep → pri_recovery → pgob →
memory_training(skip:warm) → falcon_boot(ACR+DMA:OK) →
gr_init(skip:ACR) → verify
```

## Next Frontier

After the warm pipeline completes:

1. **GR context setup** — FECS method init, context switch
2. **Compute dispatch** — PFIFO channel + GPFIFO → shader execution
3. **Cold boot path** — HBM2 training without UEFI POST (requires
   VBIOS devinit via secured PMU falcon — separate track)

## Files Changed

- `toadStool/crates/server/src/pure_jsonrpc/handler/sovereign.rs`
- `toadStool/crates/core/cylinder/src/vfio/sovereign_init.rs`
- `toadStool/crates/core/cylinder/src/vfio/device/open.rs`
- `hotSpring/experiments/206_FALCON_ACR_DMA_BOOT_SOLVED.md`
- `hotSpring/EXPERIMENT_INDEX.md`
- `hotSpring/CHANGELOG.md`
- `hotSpring/experiments/README.md`
- `hotSpring/README.md`
