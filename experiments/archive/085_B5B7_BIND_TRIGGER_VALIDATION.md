# Exp 085: B5-B7 Bind Trigger Hardware Validation

**Date:** 2026-03-24
**Type:** Hardware test (both Titan V GPUs)
**Goal:** Add missing trigger writes (0x090, 0x0a4) from nouveau source analysis
**Result:** SUCCESS — bind_stat reaches 5 in ~1µs. SEC2 DMA now active.

---

## Discovery

Cross-driver source analysis (nouveau `gm200_flcn_bind_inst` in
`nvkm/falcon/gm200.c`) revealed that after writing to CHANNEL_NEXT (0x054),
nouveau writes TWO additional trigger registers:

1. `nvkm_falcon_mask(falcon, 0x090, 0x00010000, 0x00010000)` — UNK090 bit 16
2. `nvkm_falcon_mask(falcon, 0x0a4, 0x00000008, 0x00000008)` — ENG_CONTROL bit 3

These trigger the falcon's internal bind state machine. Without them, the
CHANNEL_NEXT value sits in the register but the binding never starts — explaining
why Exp 084 saw writes accepted but bind_stat stuck at 0.

Additionally, after bind_stat reaches 5, nouveau does:
3. `mask(0x004, 0x8, 0x8)` — ack interrupt bit 3
4. `mask(0x058, 0x2, 0x2)` — CHANNEL_TRIGGER LOAD bit 1
5. Poll bind_stat → 0 (channel loaded)

## Results

### bind_stat: SOLVED

| GPU | Strategy | bind_stat | Time |
|-----|----------|-----------|------|
| Titan #2 | SysMem | **5 (OK)** | 1.14µs |
| Titan #2 | VRAM | **5 (OK)** | 1.17µs |
| Titan #2 | Hybrid | **5 (OK)** | 1.12µs |
| Titan #2 | Chain | **5 (OK)** | 1.1µs |
| Titan #1 | SysMem | **5 (OK)** | 1.06µs |
| Titan #1 | VRAM | **5 (OK)** | 1.06µs |
| Titan #1 | Hybrid | **5 (OK)** | 1.05µs |
| Titan #1 | Chain | **5 (OK)** | 1.07µs |

100% success rate on both GPUs across all strategies.

### SEC2 Firmware Execution: Major Improvement

| Metric | Exp 084 | Exp 085 | Interpretation |
|--------|---------|---------|----------------|
| PC max | 0x0072 | **0x0138** | Nearly 2x code coverage |
| sctl | 0x3000 | **0x3002** | Bit 1 set = DMA subsystem active |
| EXCI | 0x001f0000 | 0x201f0000 | Different exception state |
| ACR detection | "not in code range" | **"ACR appears active"** | ACR code IS running |
| WPR status | 1 (COPY) | 1 (COPY) | ACR started but didn't finish |

### What's Working Now

1. **Instance block binding** — fully operational, correct registers, triggers active
2. **DMA subsystem** — sctl bit 1 confirms DMA is functioning
3. **ACR firmware execution** — SEC2 runs the ACR payload, enters the copy loop
4. **Register readback** — all trigger writes confirmed via read-back

### What's Still Blocked

1. **FECS/GPCCS still in HRESET** — cpuctl=0x00000010 on both
2. **WPR status stuck at 1 (COPY)** — ACR starts copying but never finishes
3. **ACR firmware traps at PC 0x138/0x139** — stuck in a loop or hit an error

This is now a **Layer 8 problem** (WPR payload correctness / ACR descriptor
format), not a Layer 7 binding issue.

## Bugs Fixed (Cumulative B1-B7)

| # | Bug | Fix | Status |
|---|-----|-----|--------|
| B1 | Wrong register (0x668) | Changed to 0x054 | ✓ Validated |
| B2 | Missing bit 30 | Added enable flag | ✓ Validated |
| B3 | Wrong target (3→2) | Fixed to coherent | ✓ Validated |
| B4 | Missing DMAIDX clear | Added 0x604 mask | ✓ Validated |
| **B5** | **Missing UNK090 trigger** | **Set bit 16 of 0x090** | **✓ KEY FIX** |
| **B6** | **Missing ENG_CONTROL trigger** | **Set bit 3 of 0x0a4** | **✓ KEY FIX** |
| **B7** | **Missing CHANNEL_TRIGGER** | **Set bit 1 of 0x058 after bind_stat=5** | **✓ Validated** |

## Next Steps

1. **Debug WPR/ACR payload** — why does COPY start but not complete?
   - Check ACR descriptor format matches what firmware expects
   - Verify WPR region alignment and size
   - Check if FECS/GPCCS LS image headers are correct
2. **Analyze PC 0x138/0x139** — what instruction is SEC2 stuck on?
   - Disassemble the BL/ACR firmware around this offset
   - Check DMEM state for error codes
3. **mmiotrace comparison** — capture nvidia's ACR sequence for WPR format reference

## Raw Logs

`hotSpring/data/085/titan{1,2}_boot_solver.txt`
