# Exp 119: Cold-Boot WPR2 — Direct VFIO Without Nouveau

**Date:** 2026-03-27
**Hardware:** Titan V #1 (GV100), BDF 0000:03:00.0
**Depends on:** Exp 117, 118

## Hypothesis

With vfio-pci bound at boot (no nouveau), FWSEC's boot-time WPR2 is never
destroyed by a driver swap. WPR2 should be valid and ACR should work.

## Result: HYPOTHESIS REJECTED — nouveau DEVINIT required

### Cold Boot State (no nouveau, direct vfio-pci)

| Falcon | CPUCTL   | SCTL   | PC     | EXCI       | MB0    | Status |
|--------|----------|--------|--------|------------|--------|--------|
| SEC2   | 0x10     | 0x3000 | 0x007b | 0x001f0000 | 0      | HRESET, never loaded |
| FECS   | 0x10     | 0x3000 | 0x007b | 0x00070000 | 0      | HRESET, never loaded |
| GPCCS  | 0x10     | 0x3000 | 0x0000 | 0x00070000 | 0      | HRESET, never loaded |
| **PMU**| **0x20** | **0x3002**| **0x007b** | **0x201f0000** | **0x300** | **HS, HALTED, TRAPPED** |

WPR2: `start=0x1FFFFE0000 end=0x0 valid=false`

## Analysis

### FWSEC Did Run

PMU is in HS mode (0x3002) — **FWSEC successfully loaded PMU firmware from
VBIOS ROM into the PMU falcon and started it in authenticated mode**. This
confirms FWSEC executes during PCI power-on, independent of any driver.

### PMU TRAPPED — Needs DEVINIT

PMU halted at PC=0x007b with EXCI=0x201f0000 (software trap) and MB0=0x300.
Without nouveau's `gm200_devinit_post()`, the VBIOS init tables (memory
training, clock setup, FBHUB configuration) are never uploaded to PMU.
PMU can't complete its initialization without these tables and traps.

### DEVINIT Dependency Chain

```
Hardware power-on
  → FWSEC runs (VBIOS ROM) → loads PMU → PMU enters HS (0x3002)
    → PMU needs DEVINIT tables from driver
      → nouveau's devinit.post() uploads tables + boot scripts
        → PMU completes: memory init, WPR2 carve, clock setup
          → WPR2 valid → SEC2 ACR can proceed
```

Without nouveau: PMU TRAPS → WPR2 never carved → everything downstream fails.

### nouveau's gm200_devinit_post() Does:

1. Upload DEVINIT app (type 0x04) from VBIOS to PMU IMEM/DMEM
2. Upload tables from BIT entry 'I' (offsets 0x14-0x1A)
3. Upload boot scripts from VBIOS
4. Write 0x5000 to PMU mailbox (0x10a040)
5. Execute DEVINIT (BOOTVEC + CPUCTL)
6. Wait for PMU mailbox bit 13 (completion)
7. Run PRE_OS app (type 0x01) for fan control

### WPR2 Address Anomaly

Cold boot WPR2 start=0x1FFFFE0000 — this is 128 GiB range, clearly invalid
for a 12 GiB HBM2 GPU. With nouveau active (Exp 117), WPR2 was at
0x2FFE00000..0x2FFE40000 (256 KiB at ~12 GiB, top of VRAM).

The cold-boot value appears to be uninitialized register state, not FWSEC output.

## Approaches Closed

- **Cold vfio-pci boot**: WPR2 invalid without DEVINIT. Dead path.
- **No-reset swap (Exp 118)**: Reset is driver-level, not PCI-level. Dead path.
- **WPR2 content capture (Exp 118)**: BAD0AC poison, HW-protected. Dead path.

## Remaining Approaches (Priority Order)

1. **Parasitic Mode**: Use BAR0 sysfs while nouveau is active. Build compute
   pipeline on top of nouveau's established GPU state. WPR2 stays valid,
   falcons stay at SCTL=0x7021. Challenge: DMA without VFIO.

2. **Custom Nouveau Module**: Build nouveau with falcon reset disabled on
   unbind. Preserves WPR2 + falcon state across swap. Challenge: kernel
   module modification.

3. **DEVINIT Reimplementation**: Implement `gm200_devinit_post()` in Rust
   via VFIO BAR0 MMIO. Upload VBIOS tables to PMU, run DEVINIT, wait for
   completion. Then WPR2 is established and we can run ACR. Challenge:
   parsing VBIOS format, significant implementation effort.
