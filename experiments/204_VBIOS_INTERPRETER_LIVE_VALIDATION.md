# Experiment 204 — VBIOS Interpreter Live Hardware Validation

**Date:** May 17, 2026
**Hardware:** Titan V (GV100, `0000:02:00.0`), VFIO-bound, cold boot
**Objective:** Validate Experiment 203's VBIOS PLL opcode activation on real
hardware, fix discovered stride bugs, and add undocumented Volta opcodes.

## Summary

First live cold-boot execution of the sovereign VBIOS interpreter on a Titan V.
The interpreter progressed from **270 ops / 118 writes** (initial run) to
**422 ops / 231 BAR0 writes** through iterative opcode and stride fixes.
Three critical stride bugs were found and fixed, four undocumented Volta-specific
opcodes were added, and graceful desync recovery was implemented.

## Iteration Log

| Run | Ops  | BAR0 Writes | Unknown | Blocker                           | Fix Applied                          |
|-----|------|-------------|---------|-----------------------------------|--------------------------------------|
| 1   | 270  | 118         | 101     | `0xff` at `0x7b9f`               | Consecutive `0xFF` → end-of-script   |
| 2   | 383  | 211         | 101     | `0xac` at `0x825e`               | Added `0xAC` stride 13               |
| 3   | 391  | 213         | 101     | `0xb0`/`0xb1` at `0x7965`/`0x89b6` | Added `0xB0` stride 10, `0xB1` stride 3 |
| 4   | 421  | 228         | 101     | `0x00` at `0x8253` (stride bug)  | Fixed `0x56` stride: 5 → 3           |
| 5   | 422  | 231         | 101     | `0x9e` at `0x8c2c`               | Added `0x9E` stride 1 + graceful recovery |
| 6   | 422  | 231         | 100     | Graceful termination              | Pipeline continues past VBIOS failure |

## Critical Stride Fixes

### `0x56` (INIT_CONDITION_TIME): 5 → 3
Our code used stride 5 for Maxwell+ (assuming an extended `delay_us:u16` field).
Nouveau uses stride 3 for ALL generations. The "delay" bytes were actually the
next opcodes (INIT_RESUME + INIT_ZM_REG). This single fix eliminated a cascade
of 3 unknown `0x00` bytes at `0x8251–0x8253`.

### `0x3A` (INIT_GENERIC_CONDITION): always `3+size` → context-dependent
Nouveau's `default:` handler does `init->offset += size` (skip data block) for
unrecognized conditions. In sovereign mode (no display connector info), all
conditions are "unknown", so the correct stride is `3 + size`.

### `0x4F` (INIT_TMDS): 9 → 5
Nouveau: `tmds(u8) + addr(u8) + mask(u8) + data(u8) = stride 5`.

## New Volta Opcodes

| Opcode | Stride | Name (empirical)           | Evidence                                              |
|--------|--------|----------------------------|-------------------------------------------------------|
| `0xAC` | 13     | Extended clock config       | Two instances at `0x825e`, `0x829d`; INIT_NV_REG at +13 |
| `0xB0` | 10     | Extended PLL write          | Instance at `0x7965`; INIT_ZM_REG_GROUP at +10        |
| `0xB1` | 3      | Clock/flag selector         | Repeating pattern `b1 XX 01` with INIT_RESUME at +3   |
| `0x9E` | 1      | Unknown Volta prefix/gate   | Not in nouveau or envytools. Stride 1 as prefix.      |

## Remaining Blocker: Opcode `0x9E`

Opcode `0x9E` at ROM offset `0x8c2c` enters a dense block of undocumented
Volta-specific init logic (no INIT_DONE terminator, no known opcode alignment
for any trial stride). The region `0x8c2c–0x8d00` is impenetrable with current
opcode knowledge.

**Mitigation:** After 100 unknown opcodes, the interpreter gracefully terminates
the script (sets offset=0) and returns `Ok(())`, allowing the pipeline to
continue. The 231 BAR0 writes already applied include PLL programming, which
is the critical prerequisite for HBM2 training.

## Hardware Observations

- **PMC_ENABLE**: `0x00000000` confirmed cold (no prior driver)
- **PRI faults**: 17–25 faults, 3–4 recoveries — normal for cold init writes
  hitting ungated engines
- **PGRAPH_STATUS**: `0x00000000` after PGOB (improved from `0x00000081` in
  earlier runs — more engines responding after PLL init)
- **PROM read**: 126KB ROM loaded from BAR0+0x300000

## Methodology

ROM bytes were read directly from BAR0 via Python mmap for opcode analysis.
Stride determination used two techniques:
1. **Successor alignment**: Find known opcodes at trial offsets after the unknown
2. **Cross-reference**: Compare multiple instances of the same opcode in the ROM

## Files Changed

- `crates/core/cylinder/src/vfio/channel/devinit/script/interpreter/opcodes.rs`
  - Fixed `0x56` stride (5→3), `0x3A` stride (3→3+size), `0x4F` stride (9→5)
  - Added `0xAC` (stride 13), `0xB0` (stride 10), `0xB1` (stride 3), `0x9E` (stride 1)
  - Consecutive `0xFF` → end-of-script terminator
  - Graceful desync recovery (100 unknowns → clean script termination)

## Next Steps

1. **Decode `0x9E`**: Likely a PMU or clock-domain opcode. Requires deeper
   reverse engineering of GV100 VBIOS structure (envytools fork or trace-based
   analysis with nvidia driver).
2. **Test warm boot**: The GPU has been partially initialized. A warm sovereign
   boot should show different behavior — falcon warm detection, PFIFO config
   selection via `FalconWarmState`.
3. **Power cycle + second Titan V**: Install second Titan V in the K80 slot
   for parallel experimentation (cold vs warm, different ROM dumps).
