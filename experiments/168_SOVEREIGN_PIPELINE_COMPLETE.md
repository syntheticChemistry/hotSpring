# Experiment 168: Sovereign Pipeline Complete

**Date**: 2026-04-16
**Status**: IMPLEMENTED
**GPU**: Titan V (GV100, 0000:03:00.0), Tesla K80 (GK210, deferred)

## Objective

Complete the sovereign GPU initialization pipeline from cold/warm device to
compute-ready state, with full fork isolation for system safety.

## Architecture

```text
glowplug (orchestrator)
  └─ sovereign_boot() ─── JSON-RPC ──→ ember
       │                                  │
       ├─ detect_driver                   │
       ├─ swap_to_vfio ─────────────────→ ember.swap
       ├─ sovereign_init ───────────────→ ember.sovereign.init
       │                                  │
       │    ┌─────────────────────────────┘
       │    │
       │    ├─ Stage 1: bar0_probe      (fork-isolated BAR0 read)
       │    ├─ Stage 2: pmc_enable      (master engine enable)
       │    ├─ Stage 3: hbm2_training   (typestate controller)
       │    │   ├─ Untrained → PhyUp → LinkTrained → DramReady → Verified
       │    │   └─ Backends: VbiosInterpreter | DifferentialReplay | FalconUpload
       │    ├─ Stage 4: falcon_boot     (GR BAR0 init + FECS/GPCCS)
       │    ├─ Stage 5: gr_init         (FECS channel verification)
       │    └─ Stage 6: verify          (PTIMER + VRAM sentinel)
       │
       └─ result: { all_ok, compute_ready, halted_at, stages[] }
```

## Modules Created/Modified

### coral-driver (crates/coral-driver)

| File | Status | Description |
|------|--------|-------------|
| `src/vfio/isolation.rs` | NEW | Fork-isolated MMIO: `fork_isolated_raw`, `*_read`, `*_write`, `*_batch` |
| `src/vfio/device/mapped_bar.rs` | MOD | Safe wrappers: `isolated_read_u32`, `isolated_write_u32`, `isolated_batch` |
| `src/vfio/sovereign_init.rs` | NEW | 6-stage pipeline with `SovereignInitResult` matching glowplug contract |
| `src/vfio/mod.rs` | MOD | Added `pub mod isolation`, `pub mod sovereign_init` |
| `src/nv/vfio_compute/init.rs` | MOD | Widened `apply_gr_bar0_init` to `pub(crate)` |

### coral-ember (crates/coral-ember)

| File | Status | Description |
|------|--------|-------------|
| `src/ipc/handlers_mmio.rs` | NEW | Layer 1: `mmio.read32/write32/batch`, `mmio.pramin.read32`; Layer 2: `mmio.bar0.probe`, `mmio.falcon.status` |
| `src/ipc/handlers_sovereign.rs` | NEW | `ember.sovereign.init` RPC handler |
| `src/ipc/handlers_devinit.rs` | NEW | `ember.devinit.status`, `ember.devinit.execute`, `ember.vbios.read` |
| `src/ipc.rs` | MOD | Added all new handlers to both dispatch tables |
| `src/adaptive.rs` | MOD | Delegated `skip_sysfs_unbind` to inner lifecycle |
| `src/vendor_lifecycle/nvidia.rs` | MOD | Best-effort `reset_method`, `skip_sysfs_unbind=true` for Volta+ |
| `src/sysfs.rs` | MOD | Handle `vfio-pci.ids` kernel parameter in `pci_remove_rescan_inner` |

## RPC Surface (new methods)

| Method | Layer | Description |
|--------|-------|-------------|
| `mmio.read32` | MMIO | Fork-isolated BAR0 register read |
| `mmio.write32` | MMIO | Fork-isolated BAR0 register write |
| `mmio.batch` | MMIO | Fork-isolated batch read/write |
| `mmio.pramin.read32` | MMIO | PRAMIN window read (VRAM via BAR0) |
| `mmio.bar0.probe` | MMIO | Read BOOT0, PMC, PTIMER, PCI_NV registers |
| `mmio.falcon.status` | MMIO | Read falcon CPUCTL/MBOX/SCTL/BOOTVEC |
| `ember.sovereign.init` | Pipeline | Full 6-stage sovereign init |
| `ember.devinit.status` | DEVINIT | PMU falcon state + VBIOS source diagnostic |
| `ember.devinit.execute` | DEVINIT | Execute DEVINIT with diagnostics |
| `ember.vbios.read` | DEVINIT | Read VBIOS from PROM/sysfs (metadata only) |

## Test Results

- **coral-driver**: 680 lib tests passed, 0 failed, 15 ignored
- **coral-ember**: 174 lib + 17 config + 17 dispatch + 8 swap + 6 lifecycle + 6 doc = 228 passed, 0 failed

## Bugs Fixed (Phase 1)

1. **AdaptiveLifecycle delegation**: `skip_sysfs_unbind` not forwarded to inner lifecycle → D-state
2. **reset_method Permission Denied**: `prepare_for_unbind` propagated non-critical error → swap failure
3. **vfio-pci.ids kernel parameter**: `pci_remove_rescan` didn't handle cmdline-forced vfio-pci binding

## Known Limitations

- K80 VFIO groups stuck in EBUSY (iommufd cdev reference leak) — needs reboot
- FECS/GPCCS may require signed firmware on some cards (secure boot bit in HWCFG)
- HBM2 training with captured golden state not yet tested end-to-end via RPC

## Next Steps

- Reboot to clear K80 VFIO state, then validate K80 path
- End-to-end test: `coralctl sovereign-boot 0000:03:00.0` via glowplug
- Capture golden HBM2 state from warm nouveau, use as training seed
- Wire `ember.devinit` into `sovereign_init` Stage 3 as fallback for cold-boot
