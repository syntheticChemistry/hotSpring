# Experiment 079: Warm Handoff via GlowPlug/Ember

**Date:** 2026-03-23
**Goal:** Prove end-to-end compute dispatch by leveraging nouveau to load FECS firmware, then swapping back to VFIO with firmware preserved.
**Status:** FAILED — nouveau teardown halts falcons before unbind. FECS IMEM does not survive the swap regardless of `reset_method` state.

## Approach

1. `coralctl warm-fecs 0000:03:00.0` → swap to nouveau, wait 15s for GR init, swap back to vfio
2. Ember's `NvidiaLifecycle` clears `reset_method` to prevent PCIe FLR
3. Run `vfio_dispatch_warm_handoff` test — check if FECS is running, attempt dispatch

## Infrastructure Verification

- `coralctl warm-fecs` executed successfully (24.7s total)
- nouveau bound and created `/dev/dri/card1` (GR init happened)
- Ember swapped back to vfio-pci (group 69) with `vram_alive=true`
- `reset_method` sysfs: **empty** (Ember successfully disabled all reset methods)
- No PCIe FLR occurred during the swap

## Results

### FECS State After Warm Handoff

```
FECS @ 0x00409000: HRESET
  cpuctl=0x00000010 bootvec=0x00000000 hwcfg=0x20204080
  mailbox0=0x00000000 mailbox1=0x00000000
```

**FECS is back in HRESET** — identical to cold VFIO. Firmware was loaded by nouveau but did not survive the swap.

### PMU State (Supporting Evidence)

| State | Before (cold) | After (warm) |
|-------|---------------|--------------|
| cpuctl | `0x00000020` (HALTED) | `0x00000010` (HRESET) |
| bootvec | `0x00000000` | `0x00010000` |
| mailbox0 | `0x00000300` | `0x00000000` |
| irqstat | `0x00000000` | `0x00000010` |

PMU changed from HALTED to HRESET with a new bootvec (`0x10000`), indicating nouveau interacted with the PMU falcon during init (loaded boot vector) but teardown reset it. The different HRESET state compared to cold VFIO confirms nouveau's teardown is the cause, not FLR.

## Root Cause Analysis

**Nouveau's unbind sequence explicitly halts and resets all GPU falcons.**

The swap sequence is:
1. `coralctl swap 03:00.0 nouveau` → nouveau probes, loads ACR → FECS/GPCCS firmware runs
2. 15s settle → GR engine fully initialized
3. `coralctl swap 03:00.0 vfio` → Ember disables `reset_method` (no FLR)
   - But: `nouveau_drm_device_fini()` → `gf100_gr_fini()` → falcon CPUCTL halt
   - All falcon IMEM/DMEM contents are lost when the falcon is hard-reset
4. `vfio-pci` binds to a device with halted falcons

Ember's `reset_method` trick prevents the PCIe-level FLR, but it cannot prevent the **driver-level teardown** that nouveau performs during `sysfs unbind`. This is by design — a well-behaved driver cleans up its hardware state before releasing the device.

### Why `reset_method` Alone Cannot Preserve Falcon State

| Reset Type | Source | `reset_method` Prevents? | Effect on Falcons |
|-----------|--------|-------------------------|-------------------|
| PCIe FLR | VFIO bind | YES | Full reset including IMEM |
| Driver teardown | nouveau unbind | **NO** | CPUCTL halt + engine reset |
| PMC ENABLE toggle | Our PFIFO init | **NO** | Clock-gates falcon blocks |

## Implications for Sovereign Pipeline

The warm handoff strategy is **not viable** as a production path. The only paths to GR engine functionality on cold VFIO are:

1. **Sovereign ACR boot (Exp 080):** Load FECS/GPCCS firmware directly via DMA.
   - FECS `secure=false` (from Exp 078) suggests direct IMEM upload may work without ACR.
   - PMU `secure=true` suggests PMU requires signed firmware if we need it.

2. **Kernel-level state preservation:** Hypothetically, a custom kernel module could keep nouveau loaded but suspend-to-idle instead of full teardown. Not practical for sovereign pipeline.

3. **Direct falcon DMA boot:** Upload FECS firmware from `/lib/firmware/nvidia/gv100/gr/fecs_*.bin` directly to falcon IMEM/DMEM, set BOOTVEC, release HRESET. This is Exp 080.

## Code Changes

- `crates/coral-glowplug/src/bin/coralctl.rs` — `warm-fecs` subcommand
- `crates/coral-driver/src/nv/vfio_compute/init.rs` — skip-FECS-if-running guard
- `crates/coral-driver/tests/hw_nv_vfio.rs` — `vfio_dispatch_warm_handoff` test

## Skip-FECS Guard

Added a guard in `apply_fecs_channel_init` that checks FECS falcon state before submitting GR init methods. If FECS is already running (warm handoff succeeded), the guard skips the channel init to avoid conflicting with the running firmware's context management. This guard will be exercised in Exp 080 if sovereign boot succeeds.
