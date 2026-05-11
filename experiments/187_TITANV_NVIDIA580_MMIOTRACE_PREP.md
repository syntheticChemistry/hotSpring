# Experiment 187: Titan V nvidia-580 mmiotrace Preparation

**Date:** 2026-05-10
**GPU:** Titan V (GV100, PCI ID 10de:1d81, BDF 0000:02:00.0)
**Driver:** nvidia-580.126.18 (open kernel module, proprietary userspace)
**Status:** Capture script prepared. Awaiting execution window.

## Objective

Capture an mmiotrace of the nvidia-580 proprietary driver initializing
the Titan V (GV100) to determine:

1. Whether WPR (Write-Protected Region) is actually configured on GV100
2. The exact SEC2/ACR boot sequence used by the proprietary driver
3. How FECS is loaded and started (HS mode vs LS mode)
4. Whether PMU firmware is loaded and from where
5. How HBM2 memory training is triggered

## Context: Existing Nouveau Trace Analysis

The existing nouveau trace (`titanv_0000-02-00_nouveau_init_20260507_210210.mmiotrace`,
16 MB, 371K lines) shows:

- BOOT0 read: `0x140000a1` (GV100 confirmed)
- 251,919 write operations (mostly DEVINIT, memory controller, PRI ring)
- Only 2 FECS register reads (`0x409108`, `0x40912c`) — nouveau failed before GR
- Only 1 SEC2 register access — nouveau never reached security boot
- Heavy writes to `0x100xxx` (PMC/memory controller region)
- No WPR setup visible (nouveau doesn't implement it for GV100)

The nouveau trace is useful as a reference for BAR0 layout but does NOT
show the actual working boot sequence. The nvidia-580 trace is essential.

## Capture Script

`scripts/lab/capture_titanv_nvidia580_mmiotrace.sh`

Steps:
1. Unbind Titan V from vfio-pci
2. Clear driver_override
3. Enable kernel mmiotrace
4. Load nvidia module (triggers the init we want to capture)
5. Probe device to trigger driver init
6. Capture trace to `wateringHole/mmiotraces/`
7. Disable mmiotrace, rebind to vfio-pci

## Key Registers to Analyze in Captured Trace

| Register Range | Engine | What to look for |
|---------------|--------|-----------------|
| `0x409xxx` | FECS | CPUCTL writes, IMEM/DMEM loads, BOOTVEC, mailbox |
| `0x41axxx` | GPCCS | GPC context switch microcode load |
| `0x840xxx` | SEC2 | ACR bootloader load, WPR setup, auth chain |
| `0x10axxx` | PMU | PMU falcon load, DEVINIT execution, power mgmt |
| `0x100xxx` | PMC | Engine enable/disable, WPR config, VRAM config |
| `0x002xxx` | Top/Fuse | Device topology, fuse reads |

## Risks

- **System hang**: nvidia module load on GV100 may interact with vfio
  bindings for the K80 (behind PLX). K80 should remain on vfio-pci.
- **Display corruption**: Titan V may try to claim display resources if
  nvidia-drm loads. The RTX 5060 is the display GPU.
- **Incomplete trace**: If nvidia GSP-RM handles the actual boot (not
  visible via mmiotrace), the trace may show only GSP communication.

## Expected Outcome

The nvidia-580 open kernel module uses GSP-RM for Turing+ but may fall
back to legacy (non-GSP) mode for Volta. If the trace shows:

1. **Direct FECS IMEM/DMEM writes** → nvidia loads FECS directly (no ACR)
2. **SEC2 ACR sequence** → nvidia uses authenticated code loading
3. **WPR register writes** → WPR is configured, FalconBootSolver needs WPR support
4. **No WPR writes** → GAP-HS-030 confirmed, Volta uses non-WPR path
5. **Only GSP RPCs** → need to intercept at GSP firmware level instead

## References

- Exp 173: VM reagent WPR capture (GV100 VFIO, no WPR observed)
- Exp 169: Warm handoff validated (nouveau → VFIO transition)
- GAP-HS-030: GV100 WPR not used by closed driver (hypothesis)
- GAP-HS-047: Titan V PMU firmware extraction
