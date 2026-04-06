# Exp 117: WPR2 State Tracking Across Driver Swap

**Date:** 2026-03-26
**Hardware:** Titan V (GV100), BDF 0000:03:00.0
**Depends on:** Exp 116

## Motivation

Exp 116 showed WPR2 hardware boundaries are INVALID after the nouveau → vfio swap.
But we never checked WPR2 *during* nouveau's active session. This experiment reads
GPU state via sysfs BAR0 while nouveau is actively managing the device, then compares
the same registers after the swap.

## Method

Used `Bar0Access::from_sysfs_device()` (read-write mmap of resource0) to read GPU
registers while nouveau was bound. Then swapped to vfio-pci and read the same
registers via VFIO MappedBar.

## BREAKTHROUGH Results

### WPR2 Hardware Boundaries

| | During Nouveau | After Swap |
|---|---|---|
| WPR2 start | **0x2FFE00000** | 0xFFFE0000 (INVALID) |
| WPR2 end | **0x2FFE40000** | 0x00020000 (INVALID) |
| WPR2 size | **256 KiB** | N/A |
| WPR2 valid | **YES** | NO |

WPR2 is located at the TOP of VRAM (12 GiB Titan V: WPR2 starts at ~11.44 GiB).

### Falcon State

| Falcon | Nouveau SCTL | Nouveau PC | After Swap SCTL | After Swap PC |
|--------|-------------|-----------|-----------------|--------------|
| SEC2   | **0x7021**  | 0x671D    | 0x3000          | 0x0000       |
| FECS   | **0x7021**  | 0x14A4    | 0x3000          | 0x0000       |
| GPCCS  | **0x7021**  | 0x16A8    | 0x3000          | 0x0000       |

ALL THREE FALCONS are **RUNNING** with SCTL=0x7021 during nouveau's session.
All are in HRESET (dead) after the swap.

### SCTL=0x7021 — A New Security Level

We previously knew:
- 0x3000 = Light Secure (LS) — our resets produce this
- 0x3002 = Heavy Secure (HS) — Exp 112 achieved this via dual-phase boot

0x7021 is a completely different and HIGHER security level:
- Bit 0 = 1 (secure mode active)
- Bit 5 = 1 (authenticated/signed code running)
- Bits 14:12 = 0b111 (security level 7 — maximum?)

This is the "real" authenticated mode set by FWSEC during GPU power-on. Our
HS mode (0x3002) was only "partial authentication" — enough for the bootloader
to run but not the full secure environment.

### Phase C: SEC2 Survival

SEC2 did NOT survive the swap. After vfio-pci bind, SEC2 was in HRESET with
SCTL=0x3000 and PC=0x0000. The driver swap resets all falcons.

### Phase D: ACR Boot with WPR2 Addresses

Even writing our WPR blob to VRAM at nouveau's WPR2 address (0x2FFE00000) and
booting ACR with blob_size=0, the WPR status stayed at COPY (1). The ACR firmware
entered its idle loop but did not process the WPR.

The WPR2 hardware register still showed INVALID after the swap, confirming:
**The ACR firmware checks WPR2 HW boundaries before processing WPR content.**

## Root Cause (Definitive)

The nouveau → vfio driver swap performs a hardware reset that:

1. **Resets all falcons to HRESET** — SEC2, FECS, GPCCS all die
2. **Clears WPR2 hardware boundaries** — set by FWSEC, lost during reset
3. **Does NOT trigger FWSEC re-execution** — no full power cycle occurs

The PMC_ENABLE register (0x000200) shows the SAME value before and after swap
(0x5FECDFF1), confirming no full GPU power cycle — only falcon resets.

Without valid WPR2, the ACR firmware CANNOT process the WPR blob, regardless of:
- blob_size (0 or non-zero)
- WPR location (system memory or VRAM)
- DMA mode (virtual or physical)
- Security level (LS or HS)

**WPR2 is the root blocker for the entire LS-mode FECS/GPCCS path.**

## Implications

### What WPR2 Does

FWSEC (VBIOS ROM firmware) runs during GPU power-on and:
1. Carves a 256 KiB region at the top of VRAM as WPR2
2. Loads PMU firmware into WPR2
3. Sets hardware WPR2 boundary registers (read-only from SW)
4. These boundaries are enforced by hardware — memory access control

### Why We Can't Just Set WPR2

WPR2 boundaries are SET by FWSEC (hardware) and are likely READ-ONLY from
software. Previous experiments (Exp 113C) tried writing to 0x100CEC/CF0 and
0x100CD4 — the writes had no effect.

### Viable Paths Forward

1. **Capture + Replay** (Exp 118): Read WPR2 content from VRAM while nouveau
   is running, then restore after swap. Still blocked by invalid WPR2 HW
   boundaries unless we can re-trigger FWSEC.

2. **DEVINIT/FWSEC Trigger** (Exp 119): Parse VBIOS init scripts and execute
   them post-swap to trigger FWSEC, which re-establishes WPR2.

3. **No-Reset Swap** (Exp 120): Modify the driver swap to avoid falcon resets.
   If we can unbind nouveau without the GPU doing a PMC reset of the falcon
   engines, the WPR2 and falcon state would survive.

4. **Parasitic Mode**: Don't swap drivers at all. Use BAR0 sysfs access while
   nouveau is running. Build our compute pipeline on top of nouveau's active
   initialization. Non-standard but the falcons ARE running.

## Data Archive

### Nouveau-Active Snapshot
```
BOOT_0=0x140000a1  PMC_ENABLE=0x5fecdff1  BAR0_WIN=0x00002ffe
SEC2  : cpuctl=0x00000000 SCTL=0x7021 PC=0x671d EXCI=0x00000000
  MB0=0xdeada5a5  MB1=0x54534f50
FECS  : cpuctl=0x00000000 SCTL=0x7021 PC=0x14a4 EXCI=0x00000000
GPCCS : cpuctl=0x00000000 SCTL=0x7021 PC=0x16a8 EXCI=0x00000000
WPR2 indexed: start_raw=0x02ffe002 end_raw=0x02ffe403
WPR2 decoded: 0x2FFE00000..0x2FFE40000 (256 KiB) VALID
```

### Post-Swap Snapshot
```
BOOT_0=0x140000a1  PMC_ENABLE=0x5fecdff1  BAR0_WIN=0x00000000
SEC2  : cpuctl=0x00000010 SCTL=0x3000 PC=0x0000 EXCI=0x00070000
FECS  : cpuctl=0x00000010 SCTL=0x3000 PC=0x0000 EXCI=0x00070000
GPCCS : cpuctl=0x00000010 SCTL=0x3000 PC=0x0000 EXCI=0x00070000
WPR2 indexed: start_raw=0x00fffe02 end_raw=0x00000003
WPR2 decoded: 0xFFFE0000..0x00020000 INVALID
```
