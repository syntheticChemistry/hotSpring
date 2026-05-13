# Experiment 144: PMC Bit 5 ACR Boot — SEC2 Resets, BL Doesn't Execute

**Date**: 2026-04-05
**GPU**: Titan V (GV100, 0000:03:00.0)
**Parent**: Exp 140 (SEC2 PMC Bit 5 Discovery), Exp 143 (No-SBR Contradiction)
**Status**: PROGRESS — SEC2 power cycle confirmed, bootloader doesn't execute
**Branch**: `hotspring-sec2-hal`

## What Changed

- **PMC bit 5** fix deployed to ember service (Exp 140 discovery, Iter 76 code)
- SEC2 is now actually being power-cycled (bit 22 was a no-op)

## Results

### SEC2 Reset: FIXED

| Metric | Before (bit 22) | After (bit 5) |
|--------|-----------------|----------------|
| PMC reset effect | No-op (wrong engine) | Actual SEC2 power cycle |
| Scrub completion | Timeout (3000ms) | **627µs** |
| ROM halt | Timeout (3000ms) | **1.78µs** |
| POST-START FAULT | `exci=0x091f0000` | `exci=0x041f0000` (different) |

Ember journal confirms: `sec2_bit=5`, `sec2_enabled=true` after re-enable.

### Bootloader: DID NOT EXECUTE

```
BOOTVEC = 0xfd00 (our BL code address)
IMEM upload = 512B → IMEM@0xfe00 via PIO (IMEMC registers)
DMEM upload = 6144B data + 84B descriptor → DMEM@0

After STARTCPU:
  PC = 0x000b (ROM halt point, NOT our BOOTVEC 0xfd00)
  MAILBOX0 = 0xdeada5a5 (sentinel UNCHANGED — BL never ran)
  SCTL = 0x3000 (LS mode — fuse-enforced Light Secure)
  EXCI = 0x041f0000 (trap exception)
  CPUCTL = 0x00000000 (not halted, not stopped — stuck)
```

### Hypothesis: PIO IMEM Writes Rejected in LS Mode

GV100 SEC2 is fuse-locked to LS mode (SCTL=0x3000). In Light Secure mode:

1. PIO IMEM writes via IMEMC registers may be **silently dropped**
2. The ROM runs its own boot sequence on STARTCPU
3. ROM finds no valid authenticated firmware → traps at PC=0x000b

**Nouveau loads the SEC2 bootloader via DMA, not PIO.** The falcon's DMA engine
fetches the bootloader from VRAM/sysmem using the instance block page tables.
PIO is only used for DMEM (data memory), not IMEM (instruction memory).

### Evidence: Different Exception Patterns

- **`0x091f0000`** (old, bit 22): ROM never completed scrub. Falcon was in an
  undefined state — the "fault" was from trying to start a broken engine.
- **`0x041f0000`** (new, bit 5): ROM completed cleanly (scrub + halt). The
  0x04 trap indicates the CPU attempted execution but faulted — likely hitting
  empty/invalid IMEM at our BOOTVEC address.

## Verification Needed

1. **Read back IMEM at 0xfe00** — if zeros, PIO writes were silently dropped
2. **Compare with DMEM readback** — DMEM PIO writes typically work in LS mode

## Follow-Up Tests (same session)

### Test 2: CPUCTL_ALIAS (Nouveau's gm200_flcn_start)

Changed `falcon_start_cpu` to use CPUCTL_ALIAS (0x130) instead of CPUCTL (0x100),
matching Nouveau's `gm200_flcn_start`.

**Result**: CPU didn't start at all. CPUCTL stayed at 0x10 (HALTED), EXCI unchanged.
CPUCTL_ALIAS alone is insufficient without a bound instance block context.

### Test 3: sec2_prepare_v1 (full instance block + page tables)

Switched to `sec2_prepare_v1` which builds VRAM page tables, writes instance block
to register 0x480, and configures FBIF_TRANSCFG for PHYS_VID.

**Result**: Bind stalls at `bind_stat=0x000e003f` (timeout, never reaches state 5).

```
PMC_ENABLE: 0x40000020 → 0x40000121 (PFIFO+PMU enabled)
VRAM page tables: ok=true
Instance block: 0x10000 written to 0x480
FBIF_TRANSCFG: 0x110 → 0x011 (PHYS_VID)
BIND: TIMEOUT at stat=0x000e003f (not reaching state 5)
```

The instance block bind depends on page table walk completing through FBIF.
The bind walker appears to enter an error state (0x000e003f has many bits set).

## Summary: Three Bug Discoveries

1. **PMC bit 5** (CONFIRMED): SEC2 at bit 5 on GV100, not bit 22. All prior ACR
   attempts were toggling the wrong engine. This was the root cause of 3-second
   timeouts and POST-START FAULTs since Exp 141.

2. **CPUCTL vs CPUCTL_ALIAS**: CPUCTL (0x100) starts the CPU but BL doesn't execute
   (possibly empty IMEM). CPUCTL_ALIAS (0x130) has no effect without bound context.

3. **v1 bind stall**: `falcon_v1_bind_context` stalls at stat=0x000e003f on cold
   VFIO GPU. The page table walk infrastructure needs further investigation.

## Diagnostic Results (April 5)

### IMEM PIO: VERIFIED WORKING

IMEM readback at 0xfe00 via IMEMC+IMEMD confirmed all 4 test words match
perfectly. **PIO IMEM writes are NOT blocked by LS mode.** BL code is physically
present in IMEM. DMEM also verified working.

### BOOTVEC: CONFIRMED IGNORED

Tested 4 different BOOTVEC values (0xfd00, 0xfe00, 0xfd, 0xfe). ALL resulted in
PC=0x0056 (ROM execution). The falcon always re-enters the ROM on STARTCPU,
regardless of BOOTVEC value. STARTCPU resumes from ROM halt point, not BOOTVEC.

### ROOT CAUSE FOUND: VRAM IS DEAD

**FBPA (Frame Buffer Partition Array) returns PRI errors (`0xbadf3000`).**
The memory controller was never initialized. ALL PRAMIN reads return `0xbad0ac0X`
(PRI timeout pattern with incrementing sequence number).

This means:
- **ALL prior VRAM writes failed silently** (WPR, ACR payload, page tables, instance block)
- **The falcon v1 bind stalls** because it can't read page tables from VRAM
- **DMA-based firmware loading can't work** because firmware is in VRAM
- **PRAMIN writes succeed at the CPU side** (BAR0 PIO completes) but data never reaches VRAM

```
FBPA0 STATUS = 0xbadf3000 (PRI error — engine unreachable)
LTC STAT     = 0xbadf3000 (L2 cache also dead)
PMC_ENABLE bit 16 (PFB) = rejected (not controllable via PMC on GV100)
PRAMIN[any_addr] = 0xbad0ac0X (PRI timeout, incrementing)
```

### Why VRAM Is Dead

On GV100, FBPA and LTC are not controlled by PMC_ENABLE (0x200). They require
separate initialization by the VBIOS DEVINIT scripts and/or the nvidia driver.
The BIOS POST ran DEVINIT (confirmed: DEVINIT_STATUS bit 1 set), but the full
FBPA initialization requires steps BEYOND DEVINIT that only the nvidia driver
performs.

On this system, the Titan V was VFIO-bound from boot — the nvidia driver never
loaded. The memory controller was never initialized. Glowplug confirms this:
`VRAM ✗` in the startup log.

## Path Forward: Three Options

### Option A: Warm GPU via nvidia driver (fastest)
Load nvidia driver briefly → full FBPA/LTC init → switch back to VFIO.
VRAM becomes accessible, bind works, ACR boot can proceed.
Pro: fastest to validate ACR pipeline. Con: requires driver swap each boot.

### Option B: FBPA init from userspace (sovereign)
Reverse-engineer FBPA initialization from nouveau's `gv100_fb_init`.
Write the init sequence via ember MMIO — making coral fully sovereign.
Pro: no nvidia dependency. Con: complex, error-prone.

### Option C: System memory DMA (bypass VRAM entirely)
Allocate DMA buffers in system memory via VFIO IOMMU.
Write WPR/ACR/page tables to sysmem. Configure falcon DMA for sysmem target.
Pro: no VRAM needed. Con: requires VFIO DMA mapping + IOMMU page tables.

**Recommended**: Option A first (validates the ACR pipeline works once VRAM is live),
then Option B (evolve coral to init FBPA natively for full sovereignty).

## Key Registers

```
PMC_ENABLE = 0x40000020 (bit 5 = SEC2, bit 30 = TOP)
PMC_ENABLE_DEVICE (0x20C) = 0x21ecdedc (many bits — Volta engine enable)
SEC2 CPUCTL = 0x00000010 (HALTED after ROM)
SEC2 SCTL   = 0x00003000 (LS mode, fuse-enforced)
FBPA0 = 0xbadf3000 (PRI error — unreachable)
LTC = 0xbadf3000 (PRI error — unreachable)
PRAMIN[*] = 0xbad0ac0X (PRI timeout — VRAM dead)
ITFEN bit 4 = not writable (v1 DMA enable rejected)
```
