# 081: Falcon Boot Solver — Multi-Strategy SEC2/ACR/FECS Boot Chain

## Status: CRITICAL PROGRESS — SEC2 Correctly Probed, Three Strategies Tested

**Date**: 2026-03-23
**Hardware**: NVIDIA Titan V (GV100) — 03:00.0 on vfio-pci

---

## Summary

Built and tested the Falcon Boot Solver — a multi-strategy system that probes GPU state
and attempts to boot FECS through increasingly aggressive approaches. Three strategies
were tested; none achieved FECS boot, but each produced critical data for the next iteration.

## Key Discoveries

### 1. SEC2_BASE Was Wrong (Critical Fix)

**Previous**: `SEC2_BASE = 0x0084_0000` (legacy address, returned `0xbadf1100`)
**Fixed**: `SEC2_BASE = 0x0008_7000` (GV100 PTOP topology, per Exp 066)

This single fix unlocked all SEC2 diagnostics. Previous experiments showed SEC2 as
"inaccessible" because they were reading from the wrong address.

### 2. SEC2 State: CleanReset (HS Not Locked)

```
SEC2 @ 0x00087000: CleanReset
  cpuctl=0x00000010  HRESET (bit 4)
  sctl=0x00003000    HS NOT locked (bits 0,5 clear)
  bootvec=0x00000000
  hwcfg=0x20420100   IMEM=64KB DMEM=64KB secure=true
```

SCTL=0x3000 means the VFIO-bound card went through a driver reset cycle that cleared
the HS fuse state. Both IMEM and EMEM write paths are available.

### 3. EMEM Write/Read Verified

```
SEC2 EMEM write/read test:
  wrote: [42, 43, 44, 45]
  read:  0x45444342
  match: true
```

EMEM PIO at SEC2_BASE+0xAC0/0xAC4 is fully functional, confirming Exp 067 findings.

### 4. Falcon v4+ CPUCTL Bit Layout Fix

Discovered that our STARTCPU/IINVAL bits were swapped for v4+ falcons:
- **Bit 0 = IINVAL** (instruction cache invalidate)
- **Bit 1 = STARTCPU** (release from HRESET)

Previously we wrote 0x01 for STARTCPU (wrong — that's IINVAL on v4+).
Nouveau always writes 0x02 (`gm200_flcn_fw_boot`).

### 5. nvfw_bin_hdr Format Decoded

All NVIDIA firmware blobs (bl.bin, ucode_load.bin) use `nvfw_bin_hdr`:
```
[0x00] bin_magic      = 0x000010DE (NVIDIA vendor ID)
[0x04] bin_ver        = 1
[0x08] bin_size       = total file size
[0x0C] header_offset  = offset to type-specific sub-header
[0x10] data_offset    = offset to code/data payload
[0x14] data_size      = payload size
```

ACR bl.bin (1280B): sub-header at 0x100, payload at 0x200 (768B)
  Sub-header starts with `start_tag=0xFD` → BOOTVEC=0xFD00

ACR ucode_load.bin (18688B): sub-header at 0x100, payload at 0x200 (18176B)
  Sub-header contains signature/crypto bytes

---

## Strategy Results

### Strategy 1: Direct HRESET Clear (081a)
- Direct CPUCTL write to clear HRESET: **Failed** (ACR-managed)
- PMC GR engine toggle: **Failed** (FECS stays in HRESET)
- SEC2 EMEM accessible: **Confirmed** ✓

### Strategy 2: EMEM-Based SEC2 Boot
- Loaded ACR BL payload (768B) into SEC2 EMEM
- EMEM write verification: **Passed** ✓
- PMC reset SEC2 (toggle bit 14): Performed
- Result: SEC2 cpuctl unchanged at 0x12 (HRESET + STARTCPU sticky)
- **Root cause**: The BL payload needs to be in the exact format the ROM expects
  (signed, with proper EMEM layout). Raw code dump isn't sufficient.

### Strategy 3: IMEM-Based SEC2 Boot
- Uploaded 18176B ACR ucode payload to SEC2 IMEM
- Uploaded 256B sub-header to SEC2 DMEM
- Set BOOTVEC=0, issued STARTCPU (bit 1 = 0x02)
- Result: CPUCTL=0x12 (STARTCPU accepted but HRESET not released)
- EXCI=0x001f0000 (exception cause 0x1F at PC=0x0000)
- **Root cause**: On clean-reset falcon, the internal ROM still executes first,
  shadows IMEM, checks EMEM for signed BL, and halts if invalid.

---

## Architecture Built

### Falcon Boot Solver (`acr_boot.rs`)
- `FalconProbe` — captures FECS/GPCCS/SEC2 state
- `Sec2Probe` — detailed SEC2 state with `Sec2State` enum (HsLocked/CleanReset/Running/Inaccessible)
- `AcrFirmwareSet` — loads all firmware files for the boot chain
- `NvFwBinHeader` / `HsBlDescriptor` — proper firmware header parsing
- `FalconBootSolver::boot()` — tries strategies in order of cost
- `sec2_emem_write/read/verify` — SEC2 EMEM PIO interface
- `pmc_reset_sec2` — PMC engine toggle

### Register Infrastructure
- Fixed `SEC2_BASE` to `0x087000`
- Added `CPUCTL_IINVAL` (bit 0) and `CPUCTL_STARTCPU` (bit 1) for v4+
- Added SEC2-specific registers: `SCTL`, `EXCI`, `TRACEPC`, `EMEMC0`, `EMEMD0`, DMA transfer regs

---

## Next Steps (Prioritized)

### Immediate: Proper EMEM BL Loading (081b)
Study nouveau's `gp102_flcn_fw_load()` / `nvkm_falcon_fw_load()` to understand:
1. How the BL descriptor fields map to EMEM layout
2. Where the signature (sec2/sig.bin or embedded) goes
3. Whether the full BL file or just the payload goes into EMEM
4. The BL data descriptor (DMA base address for main ACR ucode) formatting

### Medium: WPR Construction (080b)
Once SEC2 boots the BL, the BL DMA-loads the main ACR firmware. Need:
1. DMA-accessible memory region with ACR ucode
2. WPR header format for LS falcon descriptors
3. FECS/GPCCS firmware images in WPR layout

### Ongoing: Solver Evolution
Each iteration refines the solver — failed strategies still produce data
that informs the next attempt. The solver architecture handles fallback
chains and strategy selection automatically.

---

## Register Quick Reference (SEC2 @ 0x087000)

```
CPUCTL:   +0x100  (bit 0=IINVAL, bit 1=STARTCPU, bit 4=HRESET, bit 5=HALTED)
BOOTVEC:  +0x104
HWCFG:    +0x108  (IMEM/DMEM sizes, security mode)
SCTL:     +0x240  (security control — bit 0=HS_ENABLED, bit 5=HS_AUTH_DONE)
EXCI:     +0x148  (exception: [31:16]=cause, [15:0]=PC)
TRACEPC:  +0x14C  (write index to EXCI, read here)
IMEMC:    +0x180  (BIT(24)=write, BIT(25)=read)
DMEMC:    +0x1C0  (BIT(24)=write, BIT(25)=read)
EMEMC0:   +0xAC0  (BIT(24)=write, BIT(25)=read) — always writable
EMEMD0:   +0xAC4
MAILBOX0: +0x040
MAILBOX1: +0x044
PMC_ENABLE: 0x200 (bit 14 = SEC2 engine power)
```
