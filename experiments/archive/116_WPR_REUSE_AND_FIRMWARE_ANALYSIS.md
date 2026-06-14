# Exp 116: WPR2 Discovery + blob_size=0 + BOOTSTRAP_FALCON

**Date:** 2026-03-26
**Hardware:** Titan V (GV100), BDF 0000:03:00.0
**Depends on:** Exp 114, 115, nouveau source analysis

## Motivation

Deep comparison of nouveau's ACR code vs ours revealed two potential root causes:

1. **`ucode_blob_size` bug**: Our `patch_acr_desc` set blob_size = WPR size. Nouveau
   leaves it at 0 (firmware binary default). Non-zero blob_size may trigger a "Tegra
   DMA path" that stalls trying to write to hardware WPR2.

2. **WPR location**: Nouveau puts WPR in VRAM (NVKM_MEM_TARGET_INST). We use system
   memory DMA buffers + VRAM mirror.

## Firmware Binary Analysis

**Critical finding**: The firmware's built-in ACR descriptor is ALL ZEROS:
```
FW original ACR desc: blob_size=0x0 blob_base=0x0 wpr_region_id=0 no_regions=0
```

This means:
- Nouveau patches ONLY region_props (wpr_start, wpr_end, shadow_start)
- blob_size stays at 0 — the firmware natively expects NO blob DMA
- Our previous code overwriting blob_size to non-zero was WRONG

**Fixed**: `strategy_sysmem.rs` now logs original values BEFORE patching.

## Bug Fix: `patch_acr_desc` logging

The "Original ACR desc" log was reading AFTER `patch_acr_desc()` had already
overwritten the values. Fixed to read the firmware binary's original fields first.

## Results

### Variant A: WPR2 Hardware Discovery

**WPR2 boundaries after nouveau → vfio swap:**
```
WPR2: start=0xfffe0000 end=0x20000 (INVALID: start > end)
```

The hardware WPR2 register (0x100CD4 indexed) returns:
- Index 2 (start): raw=0x00fffe02 → decoded=0xFFFE0000
- Index 3 (end):   raw=0x00000003 → decoded=0x00020000

**WPR2 is NOT set up** after the nouveau → vfio swap. VRAM at our mirror
addresses (0x70000, 0x60000) is all zeros before our ACR boot.

### Variant B: blob_size=0 + correct PDEs (LS mode) + BOOTSTRAP

- SEC2 boots in LS mode (sctl=0x3000), reaches idle loop (PC=0x671d)
- BOOTSTRAP_FALCON acknowledged (mb0→1, mb1→3)
- **FECS/GPCCS remain in HRESET** — WPR status stays at 1 (COPY)
- Queue discovery fails: "SEC2 init message not found in DMEM"

**Conclusion**: blob_size=0 is correct but doesn't fix the WPR processing issue.

### Variant C: blob_size=0 + legacy PDEs (HS mode)

- SEC2 achieves HS (sctl=0x3002) then **TRAPS** at PC 0x500
- EXCI cause=0x20 (TRAP), all 31 TRACEPC entries at 0x0500
- Same as Exp 112/113 — PMU dependency in HS path

### Variant D: WPR at WPR2 addresses

Skipped — WPR2 boundaries invalid.

## Root Cause Analysis (Updated)

The WPR copy stall is NOT caused by blob_size. It's caused by the ACR firmware
not processing the WPR at all. Three hypotheses remain:

### Hypothesis 1: FWSEC/WPR2 Dependency

The ACR firmware may check hardware WPR2 boundaries before processing the WPR.
If WPR2 is not set (as we observe), the firmware skips WPR processing and enters
idle immediately. FWSEC (VBIOS ROM) sets WPR2 during GPU power-on, but this state
may be lost during the nouveau → vfio swap.

Evidence:
- WPR2 registers return invalid values after swap
- VRAM at low addresses is zeroed (nouveau's WPR was at higher addresses)
- The firmware enters idle without processing WPR

### Hypothesis 2: VRAM-only WPR Access

The ACR firmware may access the WPR via direct VRAM physical addresses (not
through the falcon's page tables). Even though we set region_props to IOVAs
that match our VRAM mirror, the firmware might use a different DMA path that
doesn't go through the page tables.

Evidence:
- Our VRAM mirror at 0x70000 IS populated after the ACR boot
- But the firmware might not read from these addresses

### Hypothesis 3: Missing Initialization Sequence

Nouveau's full GPU init includes steps we skip:
1. DEVINIT (VBIOS parsing + init scripts)
2. FWSEC execution (sets up WPR2, loads PMU)
3. Instmem allocation (for WPR in VRAM)
4. ACR descriptor in a VMA (GPU virtual address space)

We skip all of these and go directly to SEC2 boot. The firmware may depend
on state set by earlier init steps.

## Next Steps

1. **Exp 117: WPR2 State Tracking** — COMPLETED. See 117_WPR2_STATE_TRACKING.md.
   BREAKTHROUGH: WPR2 IS VALID at 0x2FFE00000..0x2FFE40000 (256 KiB) while nouveau
   is running. All falcons running with SCTL=0x7021 (fully authenticated by FWSEC).
   Everything is KILLED by the driver swap.

2. **Exp 118: Capture + Replay nouveau's WPR** — NEXT. Read WPR2 content from
   VRAM while nouveau is active, then write it back after swap.

3. **Research: FWSEC re-trigger** — How to make FWSEC re-establish WPR2 after swap.
