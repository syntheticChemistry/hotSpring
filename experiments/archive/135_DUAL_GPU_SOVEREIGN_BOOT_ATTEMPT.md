# Experiment 135: Dual GPU Sovereign Boot Attempt

**Date**: 2026-03-30
**GPUs**: Tesla K80 (GK210, 0000:4c:00.0) + Titan V (GV100, 0000:03:00.0)
**Goal**: Achieve sovereign boot on both local GPUs in parallel

## Executive Summary

Both GPUs hit fundamental initialization barriers. The K80 needs VBIOS POST
(memory training) that neither nouveau nor nvidia can provide over VFIO. The
Titan V's SEC2 ROM rejects the ACR bootloader — PMU firmware must establish
the WPR chain before ACR can authenticate FECS.

## K80 Tesla (GK210 Kepler) — SM37

### What Worked
- **Clock init**: nvidia470 cold-warm diff recipe successfully applies 258
  clock registers (ROOT_PLL, PCLOCK, CLK). PTIMER starts ticking.
- **Devinit**: 315 PMC/PBDMA registers applied successfully.
- **FECS PIO upload**: Firmware uploaded and FECS falcon starts running
  (CPUCTL=0x00). The falcon core registers (0x409000-0x409500) are accessible.
- **FECS scratch0**: Firmware wrote `0x802FEF0B` to scratch0 — a status code
  indicating it ran but encountered errors.

### What Failed
- **PGRAPH CTXSW domain** (0x409504+) PRI-faults. Boundary at exactly
  0x409504: everything below accessible, everything above PRI-faults.
  - `FECS_STATUS` (0x409800) = `0xBADF1020` — firmware can't signal boot
    completion because this register is in the PRI-faulted CTXSW domain.
  - GPCCS (0x41Axxx), GR hub (0x400xxx), GPC (0x419xxx) all PRI-fault.
- **VRAM**: Completely dead. `0xFFFFFFFF` everywhere. Write-readback fails.
- **PFB** (0x100000): PRI-faults (`0xBADF1100`).
- **Memory controller**: Never initialized. GDDR5 memory training was not
  performed because the GPU was never VBIOS-POSTed (claimed by vfio-pci at
  boot before any driver loaded).
- **Nouveau cold-post**: Cannot POST the K80 over VFIO. Leaves PMC_ENABLE at
  `0xC0002020` (minimal engines).
- **VBIOS devinit**: "Partial completion" — the interpreter can execute some
  VBIOS scripts but lacks support for memory training opcodes.

### Root Cause
The K80 was bound to `vfio-pci` at boot time, preventing VBIOS POST. GDDR5
memory training requires either VBIOS POST (at boot, before any OS driver) or
a complete devinit interpreter with memory training support. Without VRAM, the
PGRAPH context switch domain can't operate, and FECS can't signal boot.

### Next Steps — K80
1. **Boot reconfiguration** (quick win): Temporarily bind K80 to nouveau at
   boot (remove from vfio-pci.ids), let VBIOS POST + nouveau init, then swap
   to VFIO. This gives us a warm K80 with live VRAM + FECS.
2. **Devinit evolution**: Capture the GDDR5 training sequence from a warm K80
   and implement it in the devinit interpreter.

## Titan V (GV100 Volta) — SM70

### What Worked
- **VRAM**: Initialized successfully by nouveau warm-fecs round-trip. HBM2
  alive, write-readback passes.
- **PGRAPH**: GR hub (0x400100) accessible. GPC0 (0x419000) accessible with
  data (0x08110780). GPCCS (0x41A100) accessible and halted (0x10).
- **FECS**: Reachable, halted (0x10), ready for ACR boot.
- **SEC2**: Reachable, halted (0x10).
- **PMU**: Reachable, halted (0x10).
- **ACR solver strategy 2**: VRAM-based ACR verified — payload written and
  verified in VRAM at 0x50000. WPR headers correctly built.
- **ACR solver strategy 6**: FECS briefly ran (cpuctl=0x00 for 10ms) after
  BOOTSTRAP_FALCON mailbox command. GPCCS also started (cpuctl=0x00).

### What Failed
- **SEC2 BL signature verification**: Across ALL 12 strategies, the SEC2 BL
  crashes after ~3 instructions (ROM trace: fd0a → fd62 → fd75 → trap).
  SCTL=0x3000 (HS=false) confirms BL runs in NS mode — ROM didn't promote it.
- **WPR2 not configured**: `start_raw=0x00FFFE02 end_raw=0x00000003` — WPR2
  registers contain stale/garbage values. Without proper WPR, ACR DMA fails.
- **SEC2 queue discovery**: "SEC2 init message not found in DMEM" — SEC2
  firmware's command queue infrastructure never initialized.
- **Direct FECS/GPCCS upload**: Firmware uploads verified (IMEM match=true),
  but falcons start in HRESET (0x12) — HS lockdown prevents NS code execution.

### Root Cause
The Titan V has Falcon v2 with HS (Hub Sequencer) security. The ACR boot
chain requires:
1. PMU firmware sets up WPR (Write-Protected Region) in VRAM
2. PMU loads SEC2 LS (Light Secure) code into WPR
3. SEC2 LS authenticates FECS/GPCCS firmware
4. Only then can FECS execute in HS mode

We're missing step 1-2. Without PMU establishing the WPR chain, SEC2 ROM
can't validate the BL (firmware signature mismatch against fuse-burned keys).
The BL from `/lib/firmware/nvidia/gv100/acr/bl.bin` (symlinked from GP102)
may be incompatible with this GV100 revision's fuse configuration.

### Next Steps — Titan V
1. **PMU bootstrap**: Investigate PMU firmware loading — PMU_CPUCTL=0x10
   (halted, reachable). If we can boot PMU, it can set up WPR.
2. **WPR capture**: Use mmiotrace during a successful nouveau session to
   capture the exact WPR register programming and SEC2 command sequence.
3. **Firmware version audit**: Verify the installed firmware version matches
   the kernel nouveau version. Check `dmesg | grep nouveau` for firmware
   loading messages during warm-fecs.
4. **FWSEC path**: Investigate the GV100 FWSEC (Firmware Security) mechanism
   — may be needed to unlock the HS boot chain.

## Architectural Observations

### Evolutionary Pattern Confirmed
Both GPUs validate the evolutionary approach:
- **K80 (Kepler)**: No crypto barriers, just needs VBIOS POST for memory.
  Pure-Rust devinit with memory training is achievable.
- **Titan V (Volta)**: Crypto chain (FWSEC→PMU→WPR→SEC2→ACR→FECS) requires
  understanding each link. Each solved link transfers to all Volta+ GPUs.

### DomainMap + BootSequence Framework
The new `DomainMap` and `BootSequence` traits correctly model the domain
accessibility patterns observed in live testing:
- K80: FECS falcon core accessible, CTXSW domain requires PGRAPH power
- Titan V: PGRAPH accessible after PMC_ENABLE, GPCs accessible after nouveau

### New Tools Created
- `coralctl acr-boot <BDF>`: Runs all 12 ACR boot strategies with detailed
  diagnostics. Essential for Volta+ sovereign boot experimentation.
- Fixed `coralctl devinit replay` to use ember FDs (prevents "Device busy").

## Register State Reference

### K80 (post cold-boot, FECS running)
| Register | Value | Status |
|----------|-------|--------|
| PMC_ENABLE (0x200) | 0xE011312C | GR bit 12 set |
| FECS_CPUCTL (0x409100) | 0x00000000 | Running |
| FECS_MAILBOX0 (0x409040) | 0x00000000 | Accessible |
| FECS_SCRATCH0 (0x409500) | 0x802FEF0B | FW error code |
| FECS_STATUS (0x409800) | 0xBADF1020 | **PRI FAULT** |
| PFB (0x100000) | 0xBADF1100 | **PRI FAULT** |
| PTIMER (0x9400) | ticking | OK |

### Titan V (post nouveau warm-fecs)
| Register | Value | Status |
|----------|-------|--------|
| PMC_ENABLE (0x200) | 0x5FECDFF1 | All engines |
| FECS_CPUCTL (0x409100) | 0x00000010 | Halted |
| GPCCS_CPUCTL (0x41A100) | 0x00000010 | Halted |
| SEC2_CPUCTL (0x87100) | 0x00000010 | Halted |
| PMU_CPUCTL (0x10A100) | 0x00000010 | Halted |
| GR_HUB (0x400100) | 0x00000000 | Accessible |
| GPC0 (0x419000) | 0x08110780 | Accessible |
| PFB (0x100000) | 0x0000FFFF | Accessible |
| VRAM | ALIVE | Write-readback OK |
