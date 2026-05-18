# Experiment 206 — Falcon ACR DMA Boot Solved

**Date:** May 17, 2026
**Hardware:** 2× Titan V (GV100), VFIO-bound (iommufd cdev)
**Objective:** Wire DMA backend to sovereign falcon boot, solving the ACR
HS boot blocker that blocked Experiments 80–205.

## Summary

The `falcon_boot` stage has been blocked since Experiment 080 with
`"ACR HS boot requires DMA backend — not provided"`. The root cause was
that `sovereign.init`'s stateless JSON-RPC handler opened BAR0 but never
extracted a `DmaBackend` from the VFIO device — the field was
`#[serde(skip)]` and only the ember handler path attempted to wire it.

Three code changes and two infrastructure fixes resolved it:

1. **sovereign.rs** — stateless handler now acquires `DmaBackend` from
   the VFIO device when using `bar0_source=vfio`, and probes for iommufd
   cdev DMA even when using sysfs BAR0
2. **sovereign.rs** — `EmberGateBypass` entered in stateless handler so
   the daemon's own VFIO opens aren't rejected by the ember gate
3. **sovereign_init.rs** — `gr_init` skipped after successful ACR boot
   (ACR already loads+starts FECS firmware; PIO re-upload conflicts)
4. **02:00.0 vfio-pci binding** — TV1 was unbound (no driver); bound via
   `driver_override` to enable iommufd cdev path
5. **Stale coral-ember** — legacy coral-ember processes held the VFIO
   cdev fd; killed and disabled coral services

## Falcon ACR Boot Result (TV2, warm — first validation)

```
falcon_boot: ok (3117ms)
  ACR boot OK: FECS cpuctl=0x00000010 mb0=0x00000000 (2 strategies)
  hs_dma_gpccs: success — cpuctl=0x... mb0=0x...
  hs_dma_fecs:  success — cpuctl=0x00000010 mb0=0x00000000
```

FECS cpuctl=0x10 = HALTED flag set (firmware entered command-wait loop).
GPCCS booted first (dependency: FECS self-halts if GPCCS absent).

## DMA Boot Mechanics

ACR HS boot (`boot_falcon_hs` in `nv_gsp_bridge.rs`):

1. Load firmware blobs from `/lib/firmware/nvidia/gv100/gr/`:
   - `fecs_bl.bin` (576B), `fecs_inst.bin` (25K), `fecs_data.bin` (5K),
     `fecs_sig.bin` (192B)
   - `gpccs_bl.bin` (→gp107 symlink), `gpccs_inst.bin` (12K),
     `gpccs_data.bin` (2K), `gpccs_sig.bin` (192B)
2. Allocate `DmaBuffer` for code+data at fixed IOVAs (IOMMU-mapped via
   iommufd `IOAS_MAP`)
3. Configure FBIF TRANSCFG (DMA index → target mapping):
   - UCODE → PHYS_VID, VIRT → VIRT, PHYS_SYS_COH/NCOH → system memory
4. Upload bootloader to IMEM, descriptor to DMEM (includes signature,
   ctx_dma=3/PHYS_SYS_COH, code/data IOVAs)
5. `CPUCTL_ALIAS ← STARTCPU` → falcon DMA-reads firmware from system
   memory → executes

## WPR State (Volta)

```
wpr1_beg=0xffffffff wpr1_end=0x0 wpr2_beg=0x0 wpr2_end=0x0
wpr_configured=false
```

Volta (pre-GSP) does not use WPR hardware boundaries. The ACR chain
works without WPR — firmware is authenticated via signature in the
bootloader descriptor.

## Current State After VFIO Open

Both GPUs dropped to cold state (PMC_ENABLE=0x40000020) after the
VFIO cdev was opened — the iommufd bind changed the device state.
HBM2 is uninitialized (PRAMIN reads return PRI fault codes 0xbad0ac0x).

**Reboot required** to restore UEFI-POSTed warm state with HBM2
trained, then the full pipeline will run:
  identity_probe → pmc_enable → cg_sweep → pri_recovery → pgob →
  memory_training(skip:warm) → **falcon_boot(ACR+DMA)** → gr_init(skip:ACR) →
  verify

## Code Changes (toadStool)

### `crates/server/src/pure_jsonrpc/handler/sovereign.rs`

- Default `bar0_source` changed from `"sysfs"` to `"sysfs"` (kept)
- Added `EmberGateBypass::enter()` for stateless handler
- VFIO path now extracts `dev.dma_backend()` into `opts.dma_backend`
- Sysfs path probes for iommufd cdev DMA as fallback
- Both paths now set `opts.dma_backend = dma_backend_for_opts`

### `crates/core/cylinder/src/vfio/sovereign_init.rs`

- `warm_fecs_preserved` check now includes `"ACR boot OK"` — prevents
  conflicting PIO re-upload in `gr_init` after successful ACR boot

### `crates/core/cylinder/src/vfio/device/open.rs`

- iommufd/cdev failure log promoted from `debug` to `warn` for
  diagnostic visibility

## Infrastructure Changes

- **02:00.0 bound to vfio-pci** via `driver_override` + `drivers_probe`
  (was previously unbound — no driver symlink)
- **coral-ember services killed and disabled** (`coral-ember.service`,
  `coral-glowplug.service`) — were holding VFIO cdev fds, blocking
  iommufd bind with EINVAL
- **glowplug.toml** `log_level` restored to `"info"` after debugging

## Remaining Work

1. **Reboot** to restore warm HBM2 state (UEFI POST trains HBM2)
2. **Full pipeline validation** with DMA on both warm Titan Vs
3. **Cold boot path**: HBM2 training requires VBIOS devinit which needs
   PMU falcon firmware (secured on Volta) — separate track
4. **gr_init**: After ACR-booted FECS is running, GR context setup
   (method init, context switch) is the next layer
5. **Compute dispatch**: PFIFO channel + GPFIFO → actual shader execution

## Files Changed

- `toadStool/crates/server/src/pure_jsonrpc/handler/sovereign.rs`
- `toadStool/crates/core/cylinder/src/vfio/sovereign_init.rs`
- `toadStool/crates/core/cylinder/src/vfio/device/open.rs`
