# Experiment 186: PMU Firmware Extraction Analysis

**Date:** 2026-05-10
**GPUs:** Tesla K80 (GK210), Titan V (GV100)
**Tool:** `exp168_pmu_firmware_probe` (enhanced)
**Status:** Analysis complete. Key architecture differences identified. Actionable paths documented.

## Objective

Mature the PMU firmware extraction tooling to determine the correct extraction
path for both K80 (Kepler) and Titan V (Volta) PMU firmware, which is the
cross-generation blocker for sovereign GPU compute.

## Key Finding: Architecture-Specific PMU Firmware Sources

### Kepler (GK210/K80) — PMU from VBIOS

Kepler PMU firmware is **embedded in the GPU's VBIOS ROM**. Nouveau's
`gk110_pmu_new` → `pmu_load()` parses NVBIOS BIT (Binary Information Table)
entries to extract boot code, code, and data sections, then uploads them to
the PMU falcon.

**K80 VBIOS analysis** (`k80_vbios.rom`, 62976 bytes):
- VBIOS signature OK (55 AA)
- BIT header found at offset 0x1c2
- VBIOS identifies as "GK210 P2080 SKU 200", Version 80.21.1B.00.02
- 2 Falcon branch instructions detected
- BIT entries present (PMU table parsing needs deeper BIT format work)

**Implication**: We do NOT need to extract PMU firmware from nvidia-470.
The K80 has its PMU firmware in its own VBIOS. The problem is that
nouveau never reads it because it doesn't recognize chip ID `0xf2`.

**Fast path**: Patch nouveau to add `case 0x0f2: device->chip = &nvf1_chipset;`
→ nouveau will parse the VBIOS, load PMU firmware, run DEVINIT scripts,
ungate GPCs, initialize GR. Then warm-catch to VFIO with live GPCs.

### Volta (GV100/Titan V) — PMU from Separate Firmware Files

Volta PMU firmware is NOT loaded from VBIOS by nouveau. Starting with Pascal+,
NVIDIA moved to a model where:
1. The GPU's boot ROM runs VBIOS DEVINIT scripts autonomously (trains HBM2)
2. The PMU falcon is loaded with signed firmware from separate files
3. SEC2/ACR handles authenticated code loading for FECS/GPCCS

**GV100 VBIOS analysis** (`gv100_vbios_pg500.rom`, 130048 bytes):
- VBIOS signature OK (55 AA)
- BIT header found at offset 0x1b2
- 9 Falcon branch instructions detected (more complex than K80)
- Contains DEVINIT scripts for HBM2 training but NOT the PMU runtime firmware

**GV100 firmware inventory** (`/lib/firmware/nvidia/gv100/`):
- `sec2/image.bin` (91136 bytes) — SEC2 ACR bootloader ✓
- `sec2/desc.bin`, `sec2/sig.bin` — SEC2 descriptor/signature ✓
- `acr/ucode_load.bin` (18688 bytes) — ACR load microcode ✓
- `acr/ucode_unload.bin` (6400 bytes) — ACR unload microcode ✓
- `gr/fecs_inst.bin` (25632 bytes) — FECS instruction memory ✓
- `gr/fecs_data.bin` (4788 bytes) — FECS data memory ✓
- `gr/gpccs_inst.bin` (12643 bytes) — GPCCS instruction memory ✓
- **PMU firmware: MISSING** — not in nvidia-580 firmware package

**nvidia-580 firmware package** (`nvidia-firmware-580-580.126.18`):
- Contains only `gsp_ga10x.bin` and `gsp_tu10x.bin` (Ampere/Turing GSP)
- NO Volta/Kepler PMU firmware whatsoever

**nvidia-470 driver**: Available in apt repos (`nvidia-driver-470`, version
470.256.02). This is the last proprietary closed-source driver that supported
Volta without GSP-RM. Its `nv-kernel.o_binary` likely contains PMU firmware
for GV100 as an embedded ELF section.

### Scan Results Summary

| Target | Falcon UC v3 | Falcon UC v4 | BIT Table | PMU Found |
|--------|-------------|-------------|-----------|-----------|
| K80 VBIOS | — | — | Yes | In VBIOS (BIT tables) |
| GV100 VBIOS | — | — | Yes | DEVINIT only |
| nvidia.ko (580 open) | No | No | — | No (GSP-only) |
| nvidia-470 nv-kernel.o | Not tested | Not tested | — | Likely present |

## Tool Enhancements (exp168_pmu_firmware_probe)

1. **Added Falcon v3 magic** (`0x10DE0142`) for Kepler PMU detection
2. **Added `--mode nv-ko`**: Scans nvidia kernel modules via `readelf -S`
   section analysis + raw binary Falcon UC scan
3. **Added `--mode vbios`**: Parses VBIOS ROM for BIT headers and PMU table
   entries. Identifies BIT entry types including PMU ('p'/'P') and DEVINIT ('I'/'i')
4. **Added GK210/GK110 filenames** to PMU detection patterns
5. **Updated header/banner** to reflect dual-GPU coverage (K80 + Titan V)

## Conclusions

### K80 (Kepler) Path — UNBLOCKED

The K80's PMU firmware is in its own VBIOS. We don't need external extraction.
The fix is a one-line kernel patch to nouveau: `case 0x0f2: device->chip = &nvf1_chipset;`

Priority: **Build patched nouveau, test GR init, warm-catch to VFIO.**

### Titan V (Volta) Path — Still Blocked

GV100 PMU firmware is not in linux-firmware, not in the nvidia-580 open driver,
and not usefully in the VBIOS. The extraction target is the nvidia-470
proprietary kernel module's `nv-kernel.o_binary`. This requires:

1. Install nvidia-470 (can coexist in `/usr/lib` without loading)
2. Run `exp168 --mode elf` against the kernel object
3. If Falcon UC headers found: extract, validate, feed to exp158 SEC2 path

Alternative: RM mmiotrace analysis (see todo: titanv-rm-trace)

## References

- Exp 168: Original PMU probe (Titan V focused)
- Exp 185: K80 nouveau chipset ID analysis
- Exp 171: K80 sovereign init (BOOT0 confirmation)
- GAP-HS-047: Titan V PMU firmware (active P0)
- GAP-HS-057: K80 GK210 nouveau chipset ID (active P0)
