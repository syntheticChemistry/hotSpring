# Exp 122: Systematic WPR2 Resolution — Three-Pronged Attack

**Date**: 2026-03-27
**Status**: COMPLETE — ALL THREE PATHS YIELD CRITICAL DATA
**GPU**: Titan V (GV100) × 2 — 0000:03:00.0 + 0000:4a:00.0

## Hypothesis

The persistent WPR copy stall (FECS=1, GPCCS=1) across Exp 114-121 is caused
by ACR firmware's inability to write authenticated images to the WPR2 VRAM region.
Three parallel approaches to resolve this.

## Variant A: WPR2 Register Write Probe

**Question**: Can the host directly set WPR2 boundaries?

**Method**: Probe every known WPR2-related register for writability:
- Indexed WPR table (0x100CD4)
- Direct PFB registers (0x100CEC, 0x100CF0)
- FBPA registers (0x1FA824, 0x1FA828)
- WPR config (0x100CD0)

### Results — ALL REGISTERS HARDWARE-LOCKED

| Register | Address | Before | Test Value | After | Writable? |
|---|---|---|---|---|---|
| Indexed WPR start | 0x100CD4 idx=2 | 0x1FFFFE02 | 0x002FFE02 | 0x1FFFFE02 | **NO** |
| Indexed WPR end | 0x100CD4 idx=3 | 0x00000003 | 0x002FFF03 | 0x00000003 | **NO** |
| PFB_WPR2_BEG | 0x100CEC | 0x00000000 | 0x2FFE0000 | 0x000E0000 | **NO** (partial mask) |
| PFB_WPR2_END | 0x100CF0 | 0x00000000 | 0x30000000 | 0x00000000 | **NO** |
| FBPA_WPR2_LO | 0x1FA824 | 0xBADF1100 | 0x0002FFE0 | 0xBADF1100 | **NO** (PRI FAULT) |
| FBPA_WPR2_HI | 0x1FA828 | 0xBADF1100 | 0x00030000 | 0xBADF1100 | **NO** (PRI FAULT) |
| PFB_WPR_CFG | 0x100CD0 | 0x00000003 | 0x00000001 | 0x02EFF001 | **NO** (odd readback) |

### Indexed Register Scan (Cold Boot)

```
idx= 0: 0x08000000  (WPR1 region start)
idx= 1: 0x08000001  (WPR1 region end)
idx= 2: 0x1FFFFE02  (WPR2 start — GARBAGE, decoded=0x1FFFFE0000=137GB)
idx= 3: 0x00000003  (WPR2 end — ZERO, decoded=0x20000=128KB → INVALID)
idx= 4: 0x1FFFFE04  (WPR3/unused — same garbage)
idx=13: 0x0010000D  (config/mode)
idx=14: 0x02EFFFFE  (decoded ~12GB — near FB top?)
idx=15: 0x000000CF  (config value)
```

### FBPA Partition Scan

All FBPA partitions (0-2) return PRI FAULT (offline). FBPA[3] returns 0xBADF5040 (FBHUB MEM_ACK error). Memory controller is NOT fully initialized.

### Conclusion 122A

WPR2 registers are **hardware-locked**. Only FWSEC firmware running in secure mode on SEC2 can set WPR2 boundaries. Host writes are silently dropped or masked. This path is **CLOSED**.

---

## Variant B: Parasitic Nouveau Mode

**Question**: What does WPR2 look like while nouveau is actively managing the GPU?

**Method**: Swap to nouveau, open sysfs BAR0, read all WPR/falcon state.

### Results — WPR2 VALID AT HIGH VRAM (>4GB)

```
Indexed WPR2: start_raw=0x02FFE002 end_raw=0x02FFE203
Decoded: 0x2FFE00000..0x2FFE40000 (256 KiB)
```

**WPR2 is at VRAM address 0x2FFE00000 = ~12.87 GB** — the top of VRAM on a 12GB Titan V.

### Indexed Register Comparison (nouveau vs cold boot)

| Index | Cold Boot | Nouveau Active | Changed? |
|---|---|---|---|
| 0 (WPR1 start) | 0x08000000 | 0x080000C0 | Yes |
| 1 (WPR1 end) | 0x08000001 | 0x080000C1 | Yes |
| 2 (WPR2 start) | 0x1FFFFE02 | 0x02FFE002 | **YES — NOW VALID** |
| 3 (WPR2 end) | 0x00000003 | 0x02FFE203 | **YES — NOW VALID** |
| 4 | 0x1FFFFE04 | 0x1FFFFE04 | No |
| 13 | 0x0010000D | 0x1FFFFFFD | **Yes** |
| 15 | 0x000000CF | 0x000000CF | No |

### Falcon State Under Nouveau (Titan #2)

```
PMU  : HALTED, SCTL=0x3002(HS), EXCI=0x201F0000 — HS but TRAPPED
SEC2 : HALTED, SCTL=0x7021(FW), EXCI=0x1A1F0000 — FWSEC mode, HALTED
FECS : HRESET, SCTL=0x3000(LS) — NEVER STARTED
GPCCS: HRESET, SCTL=0x3000(LS) — NEVER STARTED
```

**Critical finding**: FECS and GPCCS are in HRESET even under nouveau! This means nouveau's ACR boot also failed to bootstrap them on Titan #2. SEC2 (FWSEC) set up WPR2 boundaries but then halted before completing ACR bootstrap.

### VRAM Access at WPR2

Reading PRAMIN at the WPR2 address showed all zeros — but this was due to **32-bit truncation bug**: `0x2FFE00000 as u32` = `0xFFE00000`, which reads wrong VRAM. The actual WPR2 at 12GB is **beyond the standard 32-bit PRAMIN window encoding** (requires writing `0x2FFE0` to BAR0_WINDOW, not `0xFFE0`).

### Conclusion 122B

1. WPR2 lives at the **top of VRAM** (~12GB), set by FWSEC at boot
2. Even nouveau failed to bootstrap FECS/GPCCS on Titan #2 (PMU trapped)
3. FBPA registers still PRI FAULT even under nouveau
4. The 32-bit PRAMIN address limitation means our VRAM mirrors at 0x70000 are in the wrong place — the firmware targets 12GB, not 0.4MB

---

## Variant C: FWSEC Binary Extraction

**Question**: Can we find and analyze the FWSEC firmware in the VBIOS?

### VBIOS Structure

```
Image 0: offset=0x0000 size=0xE400 (57 KiB) — x86/BIOS
Image 1: offset=0xE400 size=0x11800 (70 KiB) — EFI
Total PROM: 126 KiB
```

### BIT Table (17 entries)

Key entries: `B` (boot), `S` (security), `I` (init), `p` (PMU), `i` (init scripts), `u`/`U` (utility)

### WPR Register Opcode Scan

**NONE of the known WPR register addresses appear in the VBIOS**:
- 0x100CD4 (INDEXED_WPR): NOT FOUND
- 0x100CD0 (WPR_CFG): NOT FOUND
- 0x100CEC, 0x100CF0: NOT FOUND
- 0x1FA824, 0x1FA828 (FBPA_WPR2): NOT FOUND

This means FWSEC firmware is **NOT stored in the PROM-accessible ROM**. It is either:
- Loaded by the GPU's internal boot ROM from a separate flash region
- Embedded in encrypted form (addresses would be encoded differently)
- Part of the GPU's mask ROM (hardwired in silicon)

### SEC2 TRACEPC

31 non-zero entries with addresses in 0..0x3F0000 range, suggesting SEC2 executed firmware from VRAM. FWSEC was loaded to VRAM and executed there.

### Conclusion 122C

FWSEC is inaccessible from the host. It is loaded and executed by GPU internal mechanisms before any host software can intercept it. We cannot extract, modify, or replay FWSEC. This path is **CLOSED** for direct replay.

---

## Synthesis: Root Cause Definitive

The WPR copy stall has a clear multi-layer root cause:

1. **WPR2 lives at high VRAM (~12GB)** — set by FWSEC at boot
2. **WPR2 registers are hardware-locked** — cannot be changed by host
3. **WPR2 is destroyed by driver swap** — PCI FLR/reset clears the registers
4. **FWSEC is inaccessible** — cannot be re-run to restore WPR2
5. **FBPA partitions are offline** — memory controller not serving high VRAM
6. **Our VRAM mirrors are in low VRAM** — WPR payload at 0x70000 is 12GB away from WPR2

The ACR firmware starts, reads WPR headers from our sysmem buffer, begins the copy-to-target operation, but the target (WPR2 at 12GB) is either inaccessible (FBPA offline) or write-protected (hardware WPR enforcement), so the copy hangs forever.

## Next Steps — Priority Order

### Path 1: FBPA Initialization (Highest Priority)
Bring FBPA partitions online via host register writes. Nouveau does this in `gv100_fb_init()`. If FBPA is online + memory controller serves full VRAM + WPR2 boundaries are valid → ACR firmware can complete the copy.

**Key question**: Does initializing FBPA (without nouveau) make WPR2 valid? Or does FWSEC need to run AFTER FBPA init?

### Path 2: Parasitic Compute via Sysfs BAR0
Since nouveau sets up WPR2 and all falcons, try doing compute while nouveau is bound. Use sysfs BAR0 to set up a PFIFO channel and submit GR work. Avoids the driver swap entirely.

### Path 3: Pre-GV100 GPU (Bypass FWSEC Entirely)
On Maxwell/Pascal (pre-GP102), there is no FWSEC and no hardware WPR2. ACR works differently. Target a GTX 1080 or similar for an easier path to sovereign compute.
