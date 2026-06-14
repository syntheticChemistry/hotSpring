# Experiment 140: SEC2 PMC Bit Discovery — GV100 SEC2 at Bit 5, Not Bit 22

**Date:** 2026-04-03  
**GPU:** Titan V (GV100, sm_70) at 0000:03:00.0  
**Status:** CONFIRMED  
**Follows:** Exp 139 (ACR Lockdown), SEC2 HAL investigation  
**Branch:** `hotspring-sec2-hal`

## Objective

Identify the correct PMC_ENABLE bit for SEC2 on GV100, after discovering that the
hardcoded fallback (bit 22) was wrong and the PTOP table does not list SEC2 at all.

## Background

The ACR boot path (`sec2_hal.rs`) uses `find_sec2_pmc_bit()` to locate SEC2's
PMC_ENABLE register bit. The function scans the PTOP device info table at 0x22700.
When PTOP lookup fails, it fell back to `unwrap_or(22)`.

Multiple bugs were identified in the previous implementation:
1. ENGINE_TYPE was 0x15 (NVJPG) instead of 0x0d (SEC2)
2. PTOP ENUM entry parsing extracted reset bit from wrong position
3. CPUCTL bit definitions (HALTED/STOPPED) were inverted vs Nouveau
4. Reset ordering in prepare functions (PMC before ENGCTL) was reversed

## Key Discovery: SEC2 Not in PTOP

PTOP table dump (16 entries, Titan V GV100):

| Device | Engine Type | MMIO Addr  | PMC Reset Bit |
|--------|------------|------------|---------------|
| 0      | GR (0x00)  | 0x400000   | 12            |
| 1      | CE3 (0x13) | 0x104000   | 6             |
| 2      | CE3 (0x13) | 0x104000   | 7             |
| 3      | CE3 (0x13) | 0x104000   | 21            |
| 4      | NVDEC (0x10)| 0x084000  | 15            |

**SEC2 (engine type 0x0d) is completely absent from the PTOP table on GV100.**

This means `find_sec2_pmc_bit()` will always return `None` for GV100, regardless
of how correct the PTOP parsing is.

## PMC Bit Identification via Live Probing

### Method
Systematically cleared individual PMC_ENABLE bits and checked if SEC2 falcon
registers (at 0x87000 base) returned PRI errors (`0xbad0da00`).

### PMC_ENABLE initial state (cold VFIO GPU): `0x40000020`
Only bits 5 and 30 were set.

### Result: **SEC2 is at PMC_ENABLE bit 5**

```
Clear bit 5 (0x40400000):
  SEC2 CPUCTL = 0xbad0da00  ← PRI error, engine clock gated
  SEC2 PC    = 0xbad0da00
  SEC2 EXCI  = 0xbad0da00

Restore bit 5 (0x40400020):
  SEC2 CPUCTL = 0x00000010  ← HALTED (ROM re-ran)
  SEC2 PC    = 0x00000087   ← ROM halt point (changed from 0x7B)
  SEC2 EXCI  = 0x001F0000

Clear bit 22 (tested separately):
  SEC2 registers UNCHANGED  ← Bit 22 does NOT control SEC2
```

### Confirmation on warm GPU (post-nouveau):

```
PMC_ENABLE = 0x5FECDFF1 (nouveau enabled many engines)

PMC disable bit 5 → SEC2 CPUCTL = 0xbad0da00 (PRI error)
PMC enable bit 5  → SEC2 CPUCTL = 0x00000010 (HALTED, scrub done)
```

## Impact

The old fallback `unwrap_or(22)` was toggling a bit that does NOT control SEC2.
Every `pmc_reset_sec2()` call was effectively a no-op for SEC2 — the falcon was
never actually being power-cycled. This explains why:

- SEC2 POST-START FAULTs occurred regardless of DEVINIT state
- The falcon couldn't be put into a clean state for firmware upload
- ENGCTL alone (without correct PMC) couldn't fully reset a halted falcon

## Fix Applied

`find_sec2_pmc_bit()` now has a two-tier lookup:
1. **PTOP scan** (for GPUs where SEC2 is listed in topology)
2. **`sec2_pmc_bit_by_chip()`** fallback using BOOT0 chip ID:
   - GV100 (0x140): bit 5 (confirmed)
   - GP10x (0x132-0x138): bit 22 (conventional)
   - TU10x (0x164-0x168): bit 5 (tentative)

Callers (`pmc_enable_sec2`, `pmc_reset_sec2`) now handle `None` gracefully
instead of silently using a wrong fallback.

## Additional Fixes in This Branch

1. **CPUCTL bit definitions**: HALTED = bit 4, STOPPED = bit 5 (matching Nouveau)
2. **Wait loop correction**: `falcon_engine_reset` Step 7 uses CPUCTL_HALTED (bit 4)
3. **Reset ordering**: `sec2_prepare_direct_boot` and `sec2_prepare_physical_first`
   now do ENGCTL before PMC (matching Nouveau's `gm200_flcn_enable`)
4. **PTOP ENGINE_TYPE extraction**: masks out bit 31 (continuation flag)

## Registers Reference

```
SEC2 base: 0x087000 (from PTOP, confirmed accessible)
PC:        base + 0x030
MAILBOX0:  base + 0x040
MAILBOX1:  base + 0x044
ITFEN:     base + 0x048
CPUCTL:    base + 0x100
BOOTVEC:   base + 0x104
HWCFG:     base + 0x108
DMACTL:    base + 0x10C
CPUCTL_ALIAS: base + 0x130
EXCI:      base + 0x148
SCTL:      base + 0x240
ENGCTL:    base + 0x3C0

PMC_ENABLE: 0x200 (bit 5 = SEC2 on GV100)
BOOT0:      0x000 (chip ID in bits [31:20]; GV100 = 0x140)
```

## Next Steps

- Run ACR boot with corrected PMC bit to test if POST-START FAULT is resolved
- Verify SEC2 firmware upload + STARTCPU succeeds with clean PMC reset
- Test on K80 (GK210) to determine that GPU's SEC2 PMC bit (if applicable)
