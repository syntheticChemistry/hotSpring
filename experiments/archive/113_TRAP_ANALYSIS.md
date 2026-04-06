# Exp 113: TRAP Analysis — PMU Dependency Discovered

**Status**: ALL variants trap. Root cause: HS-authenticated BL code path needs PMU.  
**Date**: 2026-03-26  
**Predecessor**: Exp 112 (dual-phase HS achieved, TRAP at copy-to-target)

## Hypothesis

The TRAP in Exp 112 might be caused by hot-swap timing, WPR2 boundaries, blob
processing, or DMA configuration. Test 5 variants to isolate the cause.

## Variants

| Variant | Hot-swap | blob_size | WPR2 | Delay |
|---------|----------|-----------|------|-------|
| A | OFF | full | off | 0 |
| B | ON | 0 (skip) | off | 0 |
| C | ON | full | SET | 0 |
| D | ON | full | off | 100ms |
| E | OFF | full | SET | 0 |

## Results

**ALL five variants are identical:**

| Signal | Value | Meaning |
|--------|-------|---------|
| HS | true | SCTL=0x3002 in all 5 |
| EXCI | 0x201f0000 | cause=0x20 (TRAP), tracepc_cnt=31 |
| TRACEPC | 31× 0x0600 | BL error loop |
| MB0 | 0xdeada5a5 | Unchanged |
| WPR | FECS=1, GPCCS=1 | COPY started, not completed |
| FAULT_STATUS | 0 | No MMU faults |

### Variables That Have Zero Effect

- **Hot-swap**: Variant A (no swap) traps identically
- **blob_size**: Variant B (blob_size=0) traps identically
- **WPR2 setting**: Variants C/E trap identically
- **Timing**: Variant D (100ms delay) traps identically

## Root Cause Analysis: PMU Dependency

### The Exp 095 "Success" Was a Graceful Failure

In Exp 095/096 (sysmem + legacy PDEs + blob_size=0):
- `code_dma_base` = sysmem IOVA (e.g., 0x200000000)
- Physical fallback DMA reads from VRAM at that high address → garbage/zero
- BL code verification **FAILS** (code doesn't match)
- BL enters **graceful exit** path → writes init message, halts → **no trap**
- HS was set by hardware before BL's own verification → SCTL=0x3002

In Exp 113 (VRAM + legacy PDEs):
- `code_dma_base` = 0x50000 (VRAM)
- Physical fallback DMA reads from VRAM 0x50000 → **correct ACR code**
- BL code verification **PASSES** (code matches)
- BL enters **fully-authenticated** boot path → needs PMU → **TRAP**

### The Full Auth Path Needs PMU

When the BL's code verification passes, it enters the fully-authenticated ACR
code path. This path requires:
1. PMU-configured WPR2 hardware boundaries
2. Possibly other PMU-initialized state
3. PRIV bus access to target falcons

Without PMU initialization, these checks fail and the BL executes a `trap`
instruction (cause=0x20).

### WPR2 Hardware State

```
WPR1: [0xffffffff..0x00000000]  (inverted → disabled)
WPR2: [0x00000000..0x00000000]  (zero → unset)
```

Attempts to set WPR2 via 0x100CEC/CF0 appeared to write but had no effect on
firmware behavior. These registers may be TLB fields, not WPR2.

## Implications

### HS ACR Chain Is Blocked by PMU

The full ACR chain (HS → authenticate → copy to FECS/GPCCS) requires PMU to
set WPR2 boundaries first. This is a fundamental NVIDIA security architecture
requirement:

```
PMU init → set WPR2 → SEC2 boot → ACR authenticate → FECS/GPCCS bootstrap
```

Without PMU, we can get HS mode but can't complete the authenticated sequence.
PMU initialization itself requires similar firmware authentication — chicken-and-egg.

### Alternative: LS-Mode Mailbox Path

The LS-mode path ALREADY works for loading FECS/GPCCS firmware:
- Exp 087: ACR processes WPR, bootstraps FECS/GPCCS → cpuctl=0x12
- Exp 091: BOOTVEC fix applied for GPCCS

The question is whether FECS/GPCCS can FUNCTION from LS mode. If they can,
the HS chain is unnecessary for sovereign compute.

## Next Steps

### Path Y: LS-Mode FECS/GPCCS Activation (Priority)

1. Re-run mailbox path (correct PDEs, LS mode, full init)
2. Apply BOOTVEC fix (GPCCS=0x3400, FECS=0x7E00)
3. Apply ITFEN + INTR_ENABLE
4. Check if FECS enters idle loop (PC~0x058f), GPCCS starts from 0x3400
5. Send GR init commands via FECS method interface
6. If GPCCS responds → sovereign compute path opens without HS

### Path Z: PMU Initialization Research

If LS-mode doesn't work:
1. Research PMU firmware boot requirements
2. Check if PMU has simpler auth requirements than SEC2
3. May be able to initialize PMU from LS mode, then use PMU-set WPR2 for SEC2 HS

## Code Changes

- `strategy_vram.rs`: Added `DualPhaseConfig` struct, `attempt_dual_phase_boot_cfg`
- `mod.rs`: Exported `DualPhaseConfig`, `attempt_dual_phase_boot_cfg`
- New test: `exp113_trap_analysis.rs` with 5-variant matrix
