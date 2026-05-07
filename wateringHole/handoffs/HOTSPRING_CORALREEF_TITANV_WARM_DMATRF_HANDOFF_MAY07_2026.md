# Titan V Warm Handoff: DMATRF Breakthrough + ROM Security Gate

**Date:** 2026-05-07  
**coralReef iteration:** 90+  
**Status:** DMATRF to FECS IMEM proven. Falcon v5 HS ROM security gate identified. PMU firmware extraction needed.

---

## Summary

Warm handoff pipeline for Titan V (GV100) achieves direct FECS IMEM loading via DMATRF
at 192µs for 25632 bytes (101 blocks). The falcon v5 HS ROM intercepts all startups
and requires WPR-authenticated firmware, which depends on SEC2 ACR completing — blocked
by missing PMU firmware in linux-firmware.

## Key Results

### DMATRF to FECS: PROVEN

| Metric | Value |
|--------|-------|
| Firmware staged | `fecs_inst.bin` (25632B) via PRAMIN to VRAM@0x60000 |
| DMATRF blocks | 101 (256B each) |
| DMATRF time | 192µs |
| PRAMIN verify | `0x002000d0 == 0x002000d0` (ok) |
| DMEM verify | `0x20677541 == 0x20677541` (ok, PIO works in HS mode for DMEM) |
| FECS post-start | `cpuctl=0x00000000 pc=0x1161 mb0=0xCAFE0000 exci=0x04070000` |

MAILBOX0 sentinel `0xCAFE0000` NOT consumed — loaded firmware never executes.
PC=0x1161 is ROM address space. `exci=0x04070000` = security trap.

### Falcon v5 HS ROM Security Gate

All falcon v5 falcons (SEC2, FECS, GPCCS) on GV100 have an on-die ROM that:
1. Intercepts STARTCPU
2. Runs security validation before user code
3. Checks IMEM contents against WPR-authenticated signatures
4. If invalid → stays in ROM loop at PC ~0x1161

The ROM cannot be bypassed when `sctl=0x3000` (HS mode 3).

### SEC2 ACR Warm State

| Register | Value | Meaning |
|----------|-------|---------|
| cpuctl | 0x00000000 | Running (not halted, not in HRESET) |
| pc | 0x1161 | ROM address space — ACR BL started but stalled |
| sctl | 0x00003000 | HS mode 3 |
| mb0 | 0x00000001 | ACR BL started |
| mb1 | 0x00000000 | No error reported |
| dmactl | 0x00000001 | DMA enabled |
| itfen | 0x00000005 | Transfer + access enabled |
| TRACEPC | 0xfd75..0x4e5a | Deep ACR execution trace (ROM+BL) |
| DMEM | ALL ZEROS (64KB) | Queues never initialized |
| CMDQ head | 0x8010 | Non-zero but no matching DMEM content |

SEC2 is alive but stuck. ACR BL started (mb0=1), executed extensively through
ROM code (TRACEPC shows 0x2D07→0x4E5a), but never completed initialization.
DMEM queues never materialized — CMDQ/MSGQ ring protocol never established.

### Root Blocker: PMU Firmware

GV100 PMU firmware does **not** exist in linux-firmware. Only Tegra chips
(gm20b, gp10b) have PMU FW. Desktop Volta relies on the proprietary driver
(nvidia-470) which embeds PMU FW in its kernel module binary (`nv-kernel.o_binary`,
40MB, obfuscated symbols).

PMU manages power/clock domains. Without PMU:
- SEC2 ACR BL starts but cannot complete authentication
- WPR never gets configured with authenticated falcon images
- FECS ROM has no valid firmware to load from WPR

## Warm Handoff Flow (Working)

```
nouveau modprobe → HBM2 trained + SEC2 ACR starts
  ↓
livepatch_nvkm_mc_reset.ko → NOP teardown functions
  ↓
nouveau unbind (state preserved)
  ↓
direct resource0 mmap → BAR0 access (no vfio-pci reset)
  ↓
DMATRF to FECS IMEM ← PROVEN (101 blocks, 192µs)
  ↓
FECS STARTCPU → ROM security gate ← BLOCKED (no WPR)
```

## Next Steps

| Priority | Action | Expected Outcome |
|----------|--------|------------------|
| **P0** | Extract PMU firmware from nvidia-470 binary | Unblock SEC2 ACR completion |
| **P1** | Install nvidia-470 for GV100, mmiotrace full boot | Capture complete init sequence including PMU |
| **P2** | If PMU FW extracted: feed to SEC2 ACR BL | ACR completes → WPR configured → FECS boots |
| **P3** | If nvidia-470 boots GV100: warm handoff with PMU state | Full compute dispatch on warm Titan V |

## Files Changed

| File | Change |
|------|--------|
| `examples/volta_warm_pipeline.rs` | Approach C: full fecs_inst.bin DMATRF (replaces BL-only), ENGCTL reset, PRAMIN multi-page staging, DMEM verify, sentinel tracking |
| `src/vfio/device/mapped_bar.rs` | `from_resource0` constructor for direct BAR0 mapping |
| `scripts/livepatch/livepatch_nvkm_mc_reset.c` | NOP 4 nouveau teardown functions |

## Hardware Status

| GPU | BDF | State | Blocker |
|-----|-----|-------|---------|
| Titan V | 0000:02:00.0 | Warm (nouveau POSTed, livepatch preserved) | PMU FW → SEC2 ACR → WPR → FECS ROM |
| K80 die0 | 0000:4b:00.0 | Needs reboot (PCIe link dead) | GPC PLL HW write-protected |
| K80 die1 | 0000:4c:00.0 | vfio-pci | Awaiting die0 |
| RTX 5060 | 0000:21:00.0 | nvidia-580 (display) | Production — no changes |
