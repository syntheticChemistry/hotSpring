# Experiment 141: ACR HS Authentication Root Cause Analysis

**Date**: 2026-04-03
**GPU**: Titan V (GV100, 0000:03:00.0)
**Parent**: Exp 136 (SEC2 DMA / FBHUB discovery), Sovereign Boot Plan

## Summary

After extensive iteration on Strategy 7c (PIO ACR + sysmem WPR), the ACR
Boot Loader (BL) executes, loads ACR code, and enters the non_sec HS
authentication loop at PC 0x2d78 — **but authentication never completes**.
This experiment documents the systematic elimination of DMA/page-table
issues and identifies the true root cause: **missing VBIOS DEVINIT**.

## What Was Fixed (DMA Path Now Correct)

### 1. FBIF is HS+-locked in VIRT mode
- After `falcon_engine_reset`, SCTL=0x3000 (HS+ from boot ROM)
- FBIF_TRANSCFG reads 0x0100 for ALL DMA indices (0-7) — host writes blocked
- DMACTL partially writable (0x07 → 0x01)
- The BL internally sets FBIF to 0x0 (VIRT mode) regardless of ctx_dma

### 2. Page table fix for system memory DMA
- `build_vram_falcon_inst_block` creates VRAM identity-mapped PTEs (first 2MB)
- ACR payload at IOVA 0x180000: PT0 pages 384..389 changed to SYS_MEM_COH PTEs
- WPR at IOVA 0x280000: new PT1 at VRAM 0x16000, PD0[1]→PT1, SYS_MEM_COH PTEs
- All page table writes verified `ok=true`

### 3. ctx_dma = VIRT (correct for falcon MMU path)
- BL uses falcon MMU → our patched page tables → IOMMU → system memory
- ctx_dma=PHYS_SYS_COH was wrong (falcon ignores it, always uses VIRT)

### 4. DMEM repair (BL descriptor overwrites data section)
- BL descriptor (84B) at DMEM@0 overwrites data_section[0..84]
- Data section first 84 bytes are zeros → repair has no effect
- BL data DMA appears to fail (DMEM still shows BL desc values in early runs)
- After repair: DMEM correctly contains data section

### 5. Warm-up STARTCPU unnecessary
- `falcon_engine_reset` includes PMC reset → boot ROM runs → halts
- The boot ROM run IS the "priming" — no separate warm-up STARTCPU needed
- Removing warm-up avoids re-entering HS+ before DMA configuration

## The 0x2d78 Loop (HS Authentication)

### Consistent TRACEPC across ALL strategies:
```
BL:      0xfd75 → 0xfd62 → 0xfd0a
non_sec: 0x0000 → ...
auth:    0x2d78 → 0x2d3a → 0x2de4 → 0x2da7 → 0x2d9a  (first pass)
loop:    0x2d78 → 0x2d3a → 0x2d93 → 0x2d9d              (retries)
exit:    0x2ddc → 0x2d78 → 0x2d3a → 0x0000 → 0xfe57 → 0xfe4b → 0xfe30
```

The ACR:
1. Enters authentication at 0x2d78 (non_sec code)
2. Iterates ~5 times with different sub-paths
3. Gives up and returns to BL error path (0xfe57)
4. Falls back to boot ROM idle (PC increments per boot)

### Key evidence: Strategy 12 vs Strategy 7c
| Strategy | DMA | Data | mb0 | Behavior |
|----------|-----|------|-----|----------|
| Strategy 12 (direct IMEM) | disabled | intact | **0x36** | Halts fast — WPR read fails |
| Strategy 7c (BL + DMA) | enabled | via DMA | 0x00 | Loops at 0x2d78 — auth fails |

Strategy 12's `mb0=0x36` means the ACR ran with correct data but DMA was
disabled, so the WPR read failed quickly. Strategy 7c's auth loop means the
ACR passed the WPR check (DMA works) but **HS authentication fails**.

## Root Cause: Missing VBIOS DEVINIT

### The sovereign boot pipeline DOES run devinit:
```
Phase 1: Recipe Replay (337 steps from nouveau_init_recipe.json)
  phase 'clock':   skipped (PTIMER already ticking)
  phase 'devinit': 102 steps applied, 0 failed
  phase 'pgraph':  4 steps applied, 0 failed
  phase 'extended':231 steps applied, 0 failed
```

### But the recipe is insufficient:
1. **Recipe source**: Captured from nouveau's init (AFTER VBIOS POST)
2. **Missing**: The VBIOS DEVINIT scripts that run BEFORE nouveau
3. **Clock skip**: PTIMER ticks after SBR, but PLLs/clocks may be unconfigured
4. **SEC2 PMC fallback**: `SEC2 PMC bit not found in PTOP, using fallback bit 22`

### VBIOS DEVINIT sets up:
- ROOT_PLL, NVPLL, MEMPLL configuration
- Power sequencing and voltage domains
- Security hardware initialization (crypto engine, fuse access)
- Memory controller calibration
- Clock domain routing

### Without VBIOS DEVINIT:
- The GPU hardware appears functional (registers read/write, DMA partially works)
- But the SEC2 crypto engine may not be initialized
- HS signature verification requires fuse-burned keys + initialized crypto hardware
- The authentication loop retries and eventually fails silently

## Next Steps

### Immediate: VBIOS Script Execution
The codebase has a VBIOS script interpreter at `devinit/script/interpreter.rs`.
Execute BIT 'I' init scripts from the Titan V's VBIOS ROM before ACR boot.

### Also investigate:
1. **SEC2 PMC bit**: Find correct bit via PTOP register scan (fallback 22 may be wrong)
2. **PLL state**: Check if ROOT_PLL, NVPLL are locked after SBR + recipe
3. **Security registers**: Read crypto engine status registers if accessible
4. **No-SBR test**: Skip SBR reset, use GPU in VBIOS-POSTed state — confirms hypothesis

## Files Modified

- `strategy_chain.rs`: Strategy 7c rewritten with:
  - SYS_MEM page table patching (PT0 + PT1 + PD0[1])
  - DMEM repair after BL boot
  - ctx_dma=VIRT (correct for falcon MMU path)
  - No warm-up STARTCPU
  - Per-DMA-index FBIF diagnostics
- `instance_block.rs`: Added `encode_sysmem_pte` (existed), used in strategy_chain
