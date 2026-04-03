# Experiment 136: SEC2 DMA Path Analysis + FBHUB/FBPA Discovery

**Date**: 2026-04-03
**GPU**: Titan V (GV100), BDF 0000:03:00.0
**Context**: Sovereign boot ACR strategy debugging

## Key Discoveries

### 1. SEC2 HS+ Mode Locks DMA Registers
- After PMC reset, SEC2's ROM enters HS+ mode (`sctl=0x3000`)
- **DMACTL (0x10c)**: Writes from host silently dropped (0x02 → reads 0x00)
- **FBIF_TRANSCFG (0x624)**: Target bits [1:0] locked. Only physical override (bit 7) writable
  - Write 0x192 (SYS_MEM+PHYS) → reads 0x190 (VID_MEM+PHYS)
- ENGCTL reset does NOT clear HS+ mode
- PMC reset does NOT clear HS+ mode (ROM re-enters HS+ instantly)

### 2. Per-Index FBIF Registers ARE Writable
- Registers at 0x600-0x61C (8 entries) accept writes even in HS+ mode
- All set to 0x002 (SYS_MEM_COH) successfully
- However, register 0x604 is also DMAIDX — gets overwritten by bind sequence
- **These writes did NOT change DMA behavior** — BL TRACEPC identical

### 3. PMC_ENABLE is Hardware-Locked
- Value: `0x5fecdff1` — cannot be modified from host
- Writing 0xffffffff reads back as `0x5fecdff1`
- Disabled bits: 1, 2, 3, 13, 16, 17, 20, 29, 31
- **FBPA (0x9a000) returns 0xbadf1100 = "engine not enabled"**

### 4. FBHUB Clock Domain is Gated
- FBHUB registers (0x100010, 0x100200) return 0xbadf5040 (PRI timeout)
- PFB_NISO domain (0x100c+) works — partially initialized by PMU boot ROM
- FBPA disabled → FBHUB can't route DMA to VRAM
- Root cause: PMU boot ROM did minimal init (VFIO grabbed GPU before full VBIOS)

### 5. PMU Also Locked
- cpuctl=0x20 (STOPPED), sctl=0x3002 (HS+)
- ENGCTL reset doesn't change state
- MAILBOX writes silently dropped

## Root Cause
The GPU was bound to vfio-pci at kernel boot, preventing the full VBIOS POST sequence.
PMU boot ROM executed minimal power-on, leaving FBPA disabled and PMC_ENABLE locked.
Without FBPA, FBHUB can't route DMA → SEC2 can't DMA from VRAM → ACR boot fails.

## Resolution Path
Bridge-SBR reset → full VBIOS POST → FBPA enabled → FBHUB functional → ACR chain boot viable.
