# Exp 112: Dual-Phase Boot — HS Achieved

**Status**: HS MODE ACHIEVED — firmware traps post-auth (EXCI cause=0x20)  
**Date**: 2026-03-25  
**Predecessors**: Exp 110 (PDE-only HS determinant), Exp 111 (VRAM-native LS-only)

## Hypothesis

If we start SEC2 with **legacy PDEs** (lower 8-byte slot → physical fallback → HS auth),
then **immediately hot-swap** to correct PDEs (upper 8-byte slot → virtual DMA) +
TLB invalidate, post-auth firmware operations should use the corrected virtual path.

## What's New vs Exp 111

| Feature | Exp 111 | Exp 112 |
|---------|---------|---------|
| Initial PDEs | Correct (upper slot) | Legacy (lower slot) |
| Expected HS | No (confirmed) | Yes (confirmed!) |
| Post-start mutation | None | Hot-swap PDEs + TLB invalidate |
| DMA path for auth | Virtual MMU | Physical fallback |

## Implementation

Three new functions in `strategy_vram.rs`:

1. **`build_vram_legacy_pde_tables`** — writes PDEs in lower 8-byte slot (triggers MMU fallback)
2. **`hotswap_pdes_to_correct`** — overwrites lower PDEs with zeros, writes correct upper PDEs, TLB invalidate
3. **`attempt_dual_phase_boot`** — orchestrates: write payload → legacy PDEs → start falcon → hot-swap → poll

## Results

```
  HS=true  SCTL=0x00003002  EXCI=0x201f0000  PC=0x6392  MB0=0xdeada5a5
```

| Signal | Value | Interpretation |
|--------|-------|----------------|
| SCTL | 0x3002 | **HS mode confirmed** |
| PC | 0x6392 | Extensive execution before trap |
| EXCI | 0x201f0000 | cause=0x20 (TRAP), tracepc_cnt=31 |
| MB0 | 0xdeada5a5 | ACR sequence not completed |
| FAULT_STATUS | 0x00000000 | **No MMU faults** |
| TRACEPC | 31x 0x0600 | BL error loop after trap |
| DMEM | All DEAD/5EC2 | DMEM not populated via DMA |
| IMEM | All 0xbadf5447 | IMEM not fully loaded via DMA |

### Key Findings

1. **HS mode is achievable with dual-phase boot** — the legacy PDE physical fallback
   correctly induces HS authentication, and the hot-swap doesn't undo it.

2. **No MMU faults after hot-swap** — FAULT_STATUS=0 means the TLB invalidate worked
   and the new correct PDEs are structurally valid.

3. **Firmware traps (cause=0x20)** — The BL hits a software TRAP instruction at some
   point during execution. This is not an MMU fault but a firmware-level validation
   failure.

4. **DMEM/IMEM not populated** — The dead patterns in DMEM/IMEM suggest the BL never
   successfully performed its DMA transfers for BL descriptor or ACR data. The trap
   likely occurs early in the DMA setup phase.

5. **PC=0x6392 with all TRACEPC=0x0600** — The BL reached 0x6392 before trapping,
   then looped in an error handler at 0x0600. The 31 trace entries all at 0x0600
   means it was stuck in the error loop when we sampled.

## Mechanism Analysis

The TRAP (cause=0x20) most likely fires because:

### Theory A: Hot-Swap Timing
The BL may have already attempted DMA between falcon start and our hot-swap.
With legacy PDEs, DMA goes through physical fallback to VRAM — but the virtual
addresses in the BL descriptor (0x50000 etc.) would resolve differently through
physical vs virtual paths. If the BL initiated DMA before the hot-swap, the
physical fallback would read wrong data or wrong addresses.

### Theory B: WPR2 Boundary Check
The BL in HS mode validates WPR2 hardware boundaries (set by PMU, which we
haven't initialized). Virtual DMA works fine for reading, but the WPR2 check
is a separate hardware register validation that fails regardless of DMA path.

### Theory C: DMA Context Mismatch
The BL descriptor specifies ctx_dma=VIRT (virtual DMA), but during the legacy
PDE phase, the hardware was doing physical fallback. After hot-swap, virtual
DMA activates — but the BL may have cached old TLB entries or DMA state.

## Next Steps

### Exp 113A: Pre-Loaded DMA + Timing Variants
- Pre-load the BL descriptor and ACR data into DMEM/EMEM *before* starting falcon
- Try different hot-swap delays (immediate, 1ms, 10ms, after HS detected)
- Test with `skip_blob_dma=true` to see if the zero-blob path avoids the trap

### Exp 113B: FBIF Physical Override
- Instead of relying on MMU page table physical fallback, use FBIF_TRANSCFG to
  force physical DMA targeting (register 0x100C18 bit patterns)
- This separates "HS auth path" from "MMU correctness"

### Exp 113C: WPR2 Analysis
- Read WPR2 hardware boundary registers before and after boot
- Check if BL is validating WPR2 or if the trap has a different cause
- May need to initialize PMU or fake WPR2 boundaries

## Code Changes

- `strategy_vram.rs`: +3 functions (~200 lines)
- `instance_block.rs`: Made `FALCON_PDx_VRAM` + `encode_vram_pde` public
- `mod.rs`: Exported `attempt_dual_phase_boot`
- `exp111_vram_native.rs`: Added `exp112_dual_phase_boot` test
