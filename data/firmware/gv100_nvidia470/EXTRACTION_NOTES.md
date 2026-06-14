# GV100 (Titan V) Falcon Firmware Extraction
# Source: nvidia-470.256.02 running inside benchScale VM
# Date: 2026-05-11
# VBIOS: 88.00.41.00.18
# Serial: 0321418083807

## PMU Falcon (0x10A000) — RUNNING
- CPUCTL: 0x00000020 (running, not halted)
- HWCFG: 0x400e0100
- IMEM: 64 KB (16384 words, ALL non-trivial) → pmu_imem.bin
- DMEM: 64 KB (16384 words, ALL non-trivial) → pmu_dmem.bin
- MAILBOX0: 0x300 (communication active)
- Role: Runs ACR (Authenticated Code Read) loader
- Note: This is the key firmware needed for sovereign cold-boot

## FECS Falcon (0x409000) — HALTED (init complete)
- CPUCTL: 0x00000010 (halted, waiting for ctxsw commands)
- HWCFG: 0x20204080
- IMEM: 32 KB — reads as zeros (HS mode protects IMEM readback)
- DMEM: 8 KB — reads as zeros (HS mode)
- Role: Front-End Context Switcher, manages GR engine context
- Note: HS-protected, firmware loaded by ACR but not readable from BAR0

## GPCCS Falcon (0x41A000) — HALTED (init complete)
- CPUCTL: 0x00000010 (halted)
- HWCFG: 0x20102840
- IMEM: 16 KB — reads as zeros (HS mode)
- DMEM: 5 KB — reads as zeros (HS mode)

## SEC2 Falcon (0x840000) — NOT ACTIVE
- All registers read 0xbadf1100 (not powered / not accessible)
- On Volta, PMU handles ACR (SEC2 used on Turing+)

## Boot Sequence
1. nvidia-470 uploads PMU firmware via DMA to VRAM
2. PMU firmware is loaded from VRAM into PMU IMEM/DMEM
3. PMU starts running, executes ACR loader
4. ACR reads signed FECS/GPCCS firmware from VRAM
5. ACR authenticates and loads FECS/GPCCS into their falcons (HS mode)
6. FECS initializes GR engine, sets up context switching
7. GPU is ready for compute

## System State with nvidia-470 Running
- PMC_BOOT_0:   0x140000a1 (GV100)
- PMC_ENABLE:   0x42001120
- GR_STATUS:    0x00000000 (idle, ready)
- PTIMER:       ticking (GPU clock active)
- Memory:       12066 MB HBM2 fully accessible
- Compute:      sm_70, 80 SMs, CUDA 11.4 verified
- Bandwidth:    PCIe H2D 6.5 GB/s, D2H 6.4 GB/s (x8 Gen3)

## Sovereign Cold-Boot Path (Future Work)
1. Extract signed FECS/GPCCS firmware from nvidia.ko .rodata section
2. Implement PMU firmware upload in coral-driver (DMA to VRAM, load to PMU falcon)
3. Implement ACR trigger sequence (PMU mailbox protocol)
4. ACR boots FECS/GPCCS autonomously
5. After ACR completes, HBM2 training follows standard coral-driver path
