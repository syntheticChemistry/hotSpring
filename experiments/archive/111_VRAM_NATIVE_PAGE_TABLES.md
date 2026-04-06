# Exp 111: VRAM-Native Page Tables

**Date:** 2026-03-26
**Hardware:** Titan V (GV100, 10de:1d81) @ 0000:03:00.0
**Firmware:** `/lib/firmware/nvidia/gv100/`
**Status:** COMPLETE — HS mechanism fully characterized

## Hypothesis

Exp 110 showed legacy PDEs → HS via VRAM physical fallback, correct PDEs → no HS
because DMA routed to sysmem. Place the ENTIRE page table chain + payload in VRAM
with correct upper-8-byte PDEs so the MMU walker resolves to VRAM PTEs → code DMA
from VRAM → HS auth succeeds.

## What's New vs Exp 110

Exp 110 combo #9 ("all-VRAM path") was **broken**: it used `bind_vram=true` but
the instance block was still in sysmem at IOVA 0x40000. Setting bind target=VRAM
told the falcon to look for the instance block at VRAM physical 0x40000 where
no valid instance block existed.

Exp 111 builds a **truly VRAM-native** setup:
- Instance block at VRAM 0x10000 (via `build_vram_falcon_inst_block`)
- PD3→PD2→PD1→PD0→PT0 all in VRAM with correct upper-8-byte PDEs
- PT0 identity-maps 512 pages (2 MiB) of VRAM
- ACR payload at VRAM 0x50000, WPR at 0x70000, shadow at 0x60000
- Bind target = VRAM (0), address = FALCON_INST_VRAM
- ctx_dma = VIRT (1)

## Results

| Run | skip_blob | HS  | SCTL       | EXCI       | PC     | MB0        | Bind  | TRACEPC |
|-----|-----------|-----|------------|------------|--------|------------|-------|---------|
| 1   | true      | no  | 0x00003000 | 0x001f001e | 0x6300 | 0xdeada5a5 | OK    | 31      |
| 2   | false     | no  | 0x00003000 | 0x001f001e | 0x6302 | 0xdeada5a5 | OK    | 31      |

### What Worked

- **VRAM bind succeeded**: bind_stat→5 in 1.08µs, bind_stat→0 in 1.07µs
- **TLB invalidate acknowledged**: PDB=0x100, ack=true
- **Firmware runs extensively**: 31 unique trace PCs, identical to Exp 104 correct-PDE pattern
- **DMEM fully readable**: ACR descriptor intact at 0x200, BL descriptor at 0x000
- **WPR copy initiated**: FECS status=COPY, GPCCS status=COPY
- **No GPU MMU faults**: FBHUB FAULT_STATUS=0x0
- **VRAM data verified**: ACR payload readback matches written data

### What Did Not Work

- **No HS mode**: SCTL=0x3000 (LS mode) in both runs
- **MB0 unchanged**: 0xdeada5a5 (sentinel never cleared by firmware)

## Key Finding: HS Auth Is Not About Physical Memory Location

The BL's HS authentication mechanism is **not** about where the code physically
resides (VRAM vs system memory). It's about the **DMA path type** used to fetch it:

| DMA Path              | Result | Evidence                          |
|-----------------------|--------|-----------------------------------|
| Physical fallback     | **HS** | Exp 110: legacy PDEs → MMU fails → physical |
| Virtual → sysmem PTEs | no HS  | Exp 110: correct PDEs → sysmem resolve |
| Virtual → VRAM PTEs   | no HS  | **Exp 111**: correct PDEs → VRAM resolve |

The physical fallback is triggered by invalid PDEs (upper 8 bytes = 0). This
activates a hardware-level physical DMA mode that bypasses the MMU entirely.
The BL's HS signature verification succeeds ONLY when code arrives through this
physical bypass path.

## Mechanism Analysis: WPR Security Gate

The most likely explanation: the BL checks whether the code was fetched from
within a **WPR (Write Protection Region)** hardware boundary. WPR2 boundaries
are set by PMU/BIOS firmware and are not accessible to the host driver:

```
WPR2 indexed: start_raw=0x00fffe02 end_raw=0x00000003  (from Exp 104)
```

These indexed register values do NOT reflect our direct writes. Without proper
WPR2 boundaries:
- **Virtual DMA**: BL's security check sees code from outside WPR → rejects HS
- **Physical fallback**: Bypasses the WPR check entirely (or the broken MMU state
  makes the WPR check always pass)

## Trace Analysis

Run 1 trace (identical to Exp 104 correct-PDE pattern):
```
0xfd75 → 0xfd62 → 0xfd0a → 0x0000 → 0x0000 → 0x346d → 0x35ee → 0x4e5a
→ 0x11c6 → 0x3d98 → 0x11c6 → 0x35de → 0x11c6 → 0x3bfb → 0x2903 → 0x2ab6
→ 0x2757 → 0x2a1e → 0x271c → 0x2a1c → 0x2753 → 0x2d9c → 0x2d15 → 0x2d00
→ 0x2d07 → 0x2cf8 → 0x2d07 → 0x2cf8 → 0x2d07 → 0x1c32 → 0x1c32
```

This confirms the firmware is executing the FULL ACR code path in LS mode.
Virtual DMA through VRAM page tables is functionally correct.

## Next Steps

### Path W: Dual-Phase Boot (PRIMARY)

Since HS requires the physical fallback path but post-auth DMA needs correct
page tables, combine both in a single boot:

1. Write legacy PDEs (lower 8-byte) to VRAM
2. Write correct PDEs (upper 8-byte) to the SAME PDE pages
3. Bind VRAM instance block
4. Start falcon → BL enters physical fallback → HS auth
5. Immediately hot-swap PDEs via PRAMIN (zero the lower 8 bytes)
6. When firmware attempts post-auth DMA, MMU walker finds correct upper PDEs
7. Virtual DMA resolves to VRAM → code load succeeds

Challenge: timing is microseconds. May need concurrent PRAMIN writes.

### Path X: WPR2 Boundary Investigation

If the WPR hypothesis is correct, setting WPR2 boundaries correctly would
make virtual DMA paths pass HS auth. This requires either:
- Finding the correct PMU mechanism to set WPR2
- Direct hardware register manipulation
- Understanding what nouveau's DEVINIT does to WPR2

### Path Y: FBIF Physical Override + VRAM Bind

Hybrid approach: use FBIF physical VRAM mode for the BL (physical DMA → HS),
but also bind the VRAM instance block so post-auth ACR code uses virtual DMA.

## Code Changes

- `strategy_vram.rs`: Added `attempt_vram_native_acr_boot()` — VRAM page tables
  + virtual DMA via instance block bind
- `mod.rs`: Exported new function
- `exp111_vram_native.rs`: Test harness with 2-run sweep (skip blob / full init)
