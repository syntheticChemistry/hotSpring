# Experiment 158: SEC2 Real Firmware Upload

## Date
2026-04-07

## Hypothesis
Uploading NVIDIA's signed SEC2/ACR firmware (from linux-firmware) to the Titan V's SEC2 falcon 
should enable HS (Heavy Secure) mode, unblocking ACR authentication of FECS/GPCCS.

## Background
- Previous exp154 used NOP/halt firmware — SEC2 started but never reached HS mode
- `/lib/firmware/nvidia/gv100/` contains signed firmware:
  - `acr/bl.bin`: ACR bootloader (768 bytes code at offset 0x200)
  - `sec2/image.bin`: SEC2 HS image (91136 bytes, code: 64768 bytes)
  - `sec2/sig.bin`: RSA signature (192 bytes)
  - `gr/fecs_inst.bin`: FECS instruction memory (25632 bytes)
  - `gr/fecs_data.bin`: FECS data memory (4788 bytes)
- Titan V VBIOS dumped (130KB, GV100 PG500 SKU 0, dated 2018-02-02)
- SEC2 registers PRI-fault on reads (0xbadf1201) but WRITES succeed (PRIV_LOCKDOWN)

## Key Findings

### Result: 10/11 checks passed

| Check | Result |
|-------|--------|
| Ember reachable | PASS |
| Firmware loaded | PASS |
| ACR BL IMEM upload (768 bytes) | PASS |
| SEC2 code upload (64768 bytes) | PASS |
| SEC2 DMEM upload (512 bytes) | PASS |
| SEC2 start | PASS |
| SEC2 poll completed | PASS |
| **SEC2 HS mode reached** | **FAIL** |
| FECS IMEM upload (25632 bytes) | PASS |
| FECS DMEM upload (4788 bytes) | PASS |
| FECS start | PASS |

### SEC2 State After Real Firmware
- **PC = 682 (0x2AA)** — bootloader executed ~170 instructions
- **cpuctl = 0x12** — halted + start issued  
- **SCTL = 0x3000** — bits 12-13 set (new! was 0x0000 with NOP firmware)
- **mailbox0 = 0** — no error code reported
- **HS mode bit (SCTL bit 1) = 0** — not reached

### SCTL Analysis
The SCTL change from 0x0000 (NOP) to 0x3000 (real firmware) proves the ACR bootloader IS executing 
and making progress. Bits 12-13 may indicate:
- DMA stall waiting for VRAM (likely)
- Secure boot verification in progress
- Bootloader reached DMA stage but can't access VRAM

### Why HS Mode Fails
The ACR bootloader sequence on Volta:
1. Upload ACR BL to IMEM via PIO ✓
2. BL starts executing ✓
3. BL attempts DMA to load main HS ucode from VRAM ← **FAILS (VRAM dead)**
4. BL halts at PC=682

**Root cause**: HBM2 DRAM not trained → no working VRAM → DMA from VRAM impossible → 
ACR bootloader cannot load HS payload.

### FECS Real Firmware
- 25632 bytes of instruction memory uploaded successfully
- 4788 bytes of data memory uploaded successfully
- FECS started: pc=682, cpuctl=0x12 (halted)
- Firmware executed but halted (likely needs ACR authentication first)

## Dependency Chain
```
PMU DEVINIT (VBIOS scripts) → HBM2 trained → VRAM alive
     → SEC2 ACR DMA loads HS ucode → SEC2 enters HS mode
          → ACR authenticates FECS/GPCCS firmware
               → FECS/GPCCS start in secure mode
                    → PGRAPH fully operational
                         → Compute dispatch ready
```

## Files
- Binary: `barracuda/src/bin/exp158_sec2_real_firmware.rs`
- Firmware: `/lib/firmware/nvidia/gv100/`
- VBIOS dump: `/tmp/titanv_vbios.rom`
- Log: `/tmp/exp158_sec2_real.log`

## Next Steps
1. Parse Titan V VBIOS for PMU DEVINIT scripts (same approach as K80 gk210_devinit_recipe.json)
2. Upload PMU firmware + DEVINIT data to PMU falcon
3. Start PMU DEVINIT to train HBM2
4. Re-run exp158 with working VRAM → SEC2 HS should succeed
