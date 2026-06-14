# Experiment 104: PDE Slot Fix — Page Table Format Breakthrough

**Date:** 2026-03-25
**Status:** FULLY EXPLAINED by Exp 110 — PDE slot is sole HS determinant. Correct PDEs → working MMU but no HS.

## Root Cause Found

**Critical bug in page table construction:** GV100 MMU v2 uses 16-byte PDE entries
at ALL directory levels. The directory pointer goes in the **upper** 8 bytes
(offset 8..16), with the lower 8 bytes zeroed.

Our `build_vram_falcon_inst_block` (reference implementation) correctly used
offset 8. But `strategy_sysmem.rs` was writing to offset 0 in ALL FOUR directory
levels (PD3, PD2, PD1, PD0) — in both the sysmem DMA buffers AND the VRAM mirrors.

### Why Previous Experiments Seemed to Work

With PDEs in the wrong slot (lower 8 bytes), the MMU walker likely found an
entry that looked valid in a non-standard way. HS mode was achieved because
authentication works on IMEM code (not page tables), but ALL virtual DMA in
HS mode failed because the walker couldn't properly traverse the page table
chain. This caused the invariant crash at TRACEPC=0x0500.

## Fix Applied

```rust
// BEFORE (wrong): PDE in lower 8 bytes
pd3_dma.as_mut_slice()[0..8].copy_from_slice(&pde.to_le_bytes());

// AFTER (correct): PDE in upper 8 bytes, lower zeroed
pd3_dma.as_mut_slice()[0..8].copy_from_slice(&0u64.to_le_bytes());
pd3_dma.as_mut_slice()[8..16].copy_from_slice(&pde.to_le_bytes());
```

Fixed in:
- `attempt_sysmem_acr_boot` (simple path)
- `attempt_sysmem_acr_boot_inner` (full path) — both sysmem DMA and VRAM mirror
- All 4 directory levels: PD3, PD2, PD1, PD0

## Hardware Results

### Before Fix (Exp 100-103)
- HS mode achieved (SCTL=0x3002)
- Crash at PC 0x0500 with HALT (2 trace entries)
- DMEM unreadable (0xDEAD5EC2 — HS protection)
- No EMEM data
- MAILBOX0 never written

### After Fix (Exp 104)
- **31 unique trace PCs** — extensive code execution through BL and ACR
- **CPU alive at idle loop** (cpuctl=0x00000000, not halted)
- **DMEM fully readable and correct** — ACR descriptor intact at 0x200
- **EMEM queue initialization data present** — firmware set up CMDQ/MSGQ
- **No GPU MMU faults** — clean FBHUB
- HS mode NOT achieved (SCTL=0x3000) — under investigation

### Key Diagnostics
```
TRACEPC: 0xfd75 → 0xfd62 → 0xfd0a → 0x0000 → 0x2d07 → 0x2cf8 → ...
         → 0x4e5a → 0x11c6 → 0x3d98 → 0x3bfb → 0x2903 → 0x2ab6 → 0x271c
EMEM: [0x000]=0x00230406 [0x080]=0x00042001 [0x084]=0x026c0200 ...
DMEM ACR desc: wpr_region_id=1 no_regions=2 start=0x700 end=0x7cd
```

## WPR2 Indexed Register Discovery

Added diagnostic for nouveau's WPR2 indexed register at 0x100CD4:
```
WPR2 indexed: start_raw=0x00fffe02 end_raw=0x00000003
```
The indexed register does NOT reflect our direct writes to 0x100CEC/CF0.
This may affect firmware's WPR boundary validation.

## Analysis

The PDE fix fundamentally changed the DMA behavior:
1. Virtual DMA through page tables now works correctly
2. BL successfully loads both code and data via VIRT DMA
3. ACR firmware initializes, sets up EMEM queues, enters idle loop
4. But HS authentication may be failing due to VRAM PTEs loading code
   from VRAM mirror instead of sysmem original

## Next Steps → Answered by Exp 110

1. ~~Test with all SYS_MEM PTEs~~ — **ANSWERED** (Exp 110): VRAM PTEs have zero effect on HS.
2. ~~WPR2 indexed register~~ — Deferred, may revisit in Exp 111 if VRAM-native PTs alone don't work.
3. **Exp 111: VRAM-native page tables** — Build entire PT chain in VRAM with correct
   upper PDEs + VRAM PTEs + VRAM instance block. Theory: correct MMU walk + VRAM code
   source → HS auth + working DMA.
