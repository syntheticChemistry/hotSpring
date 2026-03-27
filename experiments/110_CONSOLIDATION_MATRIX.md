# Exp 110: Consolidation Matrix

**Date:** 2026-03-26
**Hardware:** Titan V (GV100, 10de:1d81) @ 0000:03:00.0
**Firmware:** `/lib/firmware/nvidia/gv100/`

## Purpose

Consolidate experiments 095–109 into a single definitive truth table.
Six binary ACR boot variables swept across 12 combinations to determine
which achieve HS mode (SCTL=0x3002) on GV100 SEC2.

## Variables

| Variable        | Values                | Controls                                |
|-----------------|-----------------------|-----------------------------------------|
| `pde_upper`     | true/false            | PDE pointer slot in 16-byte MMU entry   |
| `acr_vram_pte`  | true/false            | VRAM aperture PTEs for ACR code pages   |
| `blob_size_zero`| true/false            | Skip firmware internal blob DMA         |
| `bind_vram`     | true/false            | Instance block bind target (VRAM vs SYS)|
| `imem_preload`  | true/false            | PIO pre-load ACR code to IMEM           |
| `tlb_invalidate`| true/false            | TLB flush after instance block bind     |

## Results

| #  | pde   | vram_pte | blob0 | bind | imem  | tlb   | HS  | SCTL       | EXCI       | MB0=0 | Why                                  |
|----|-------|----------|-------|------|-------|-------|-----|------------|------------|-------|--------------------------------------|
|  1 | lower | false    | true  | SYS  | false | false | YES | 0x00003002 | 0x1f200000 | no    | Exp 095 baseline                     |
|  2 | lower | false    | true  | SYS  | false | true  | YES | 0x00003002 | 0x1f200000 | no    | + TLB                                |
|  3 | upper | false    | true  | SYS  | false | true  | no  | 0x00003000 | 0x001f001e | no    | Correct PDEs, skip blob              |
|  4 | upper | true     | true  | SYS  | false | true  | no  | 0x00003000 | 0x001f001e | no    | + VRAM code PTEs                     |
|  5 | lower | false    | false | SYS  | false | false | YES | 0x00003002 | 0x1f200000 | no    | Old PDEs, full init                  |
|  6 | lower | false    | false | SYS  | false | true  | YES | 0x00003002 | 0x1f200000 | no    | + TLB                                |
|  7 | upper | false    | false | SYS  | false | true  | no  | 0x00003000 | 0x001f001e | no    | Correct PDEs, full init              |
|  8 | upper | true     | false | SYS  | false | true  | no  | 0x00003000 | 0x001f001e | no    | + VRAM code PTEs, full init          |
|  9 | upper | true     | false | VRAM | false | true  | no  | 0x00003000 | 0x001f001e | no    | All-VRAM path                        |
| 10 | upper | false    | true  | SYS  | true  | true  | no  | 0x00003000 | 0x001f001e | no    | Pre-load interference                |
| 11 | lower | false    | true  | SYS  | true  | false | YES | 0x00003002 | 0x1f200000 | no    | Pre-load + old PDEs                  |
| 12 | upper | true     | true  | SYS  | false | true  | no  | 0x00003000 | 0x001f001e | no    | Correct PDEs + VRAM code + skip blob |

**HS achieved: 5/12 | MB0 cleared: 0/12**

## Key Finding: PDE Slot Position Is the Only HS Determinant

The single variable that determines HS mode is `pde_upper`:

- **`pde_upper=false` (legacy lower-8-byte slot) → always HS** regardless of all other variables
- **`pde_upper=true` (correct upper-8-byte slot) → never HS** regardless of all other variables

All other variables (VRAM PTEs, bind target, blob_size, IMEM preload, TLB flush)
have **zero effect** on whether HS mode is achieved.

## Mechanism Analysis

GV100 MMU v2 uses 16-byte PDE entries. Per the hardware spec, the directory
pointer belongs in the **upper 8 bytes** (offset 8..16) with the lower 8 bytes
zeroed.

When we place the PDE pointer in the **lower 8 bytes** (wrong position):
1. The MMU walker reads the upper 8 bytes of the PDE, finds zeros → **invalid PDE**
2. Hardware falls back to **physical VRAM addressing** for subsequent DMA
3. The bootloader's code DMA resolves through VRAM → **HS authentication succeeds**
4. Post-authentication firmware DMA also attempts VRAM physical → crashes at 0x0500

When we place the PDE pointer in the **upper 8 bytes** (correct position):
1. The MMU walker follows the PDE chain → resolves to **system memory PTEs**
2. The bootloader's code DMA reads from system memory → **HS authentication fails**
3. Firmware runs in LS mode → idle loops but no crash

## Implications for Sovereign Pipeline

The HS/LS mode gate is entirely about **where the bootloader reads its code from**
during the authentication check. The hardware requires the code to come from VRAM
(or a VRAM-equivalent path). System memory code fails authentication.

### Next Steps

1. **VRAM-only page tables**: Build a complete page table chain in VRAM (not system
   memory) with correct upper-8-byte PDEs. All PTEs point to VRAM. This should give
   both correct MMU structure AND VRAM-sourced code DMA.

2. **Instance block in VRAM**: Move the instance block itself to VRAM so the MMU
   walker doesn't need to reach system memory at all during authentication.

3. **Dual-phase boot**: Boot with legacy PDEs (HS), then rewrite page tables to
   correct format and re-bind before the firmware attempts post-auth DMA.

## Deprecated Code Paths

| Function                           | Status     | Reason                                      |
|------------------------------------|------------|---------------------------------------------|
| `attempt_hybrid_sysmem_vram_boot`  | **Deleted** | Zero callers, 382 lines, redundant with matrix |
| `attempt_sysmem_physical_boot`     | Deprecated | Never achieved HS; FBIF physical mode bypass |
| IMEM pre-load block                | Removed    | Exp 108 confirmed it breaks HS auth          |

## Code Changes

- `strategy_sysmem.rs`: 1766 → 1310 lines (−26%)
- New `boot_diagnostics.rs`: 157 lines (extracted from inline diagnostics)
- New `BootConfig` struct: parameterizes all 6 variables
- New `attempt_sysmem_acr_boot_with_config`: accepts `BootConfig` for matrix testing
- New `exp110_matrix.rs`: test harness with auto-discovery and 12-combo sweep
- New `scripts/hw-test`: convenience script with auto-discovery and permission checks
