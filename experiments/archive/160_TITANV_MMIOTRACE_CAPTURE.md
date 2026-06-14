# Experiment 160: Titan V nvidia-535 Full Init mmiotrace Capture

## Date: 2026-04-07

## Hypothesis
Capture the complete MMIO register sequence nvidia-535 uses to initialize
the Titan V (GV100), including HBM2 training, FECS firmware loading, and
all engine enablement. This recipe can be replayed from sovereign tools.

## Method
1. Boot reagent VM with Titan V VFIO passthrough
2. Enable kernel mmiotrace before nvidia driver load
3. Load nvidia-535 under trace (captures every BAR0 read/write)
4. Save trace to `data/titanv/mmiotrace/nvidia535_init_full.log`

## Results

### Trace Statistics
- **1,163,388 total MMIO operations** (970,703 writes, 192,605 reads)
- **401,449 BAR0 operations** (GPU register accesses)
- 7 ioremap MAP entries (BAR0, BAR1, BAR3, auxiliary)

### Operations by Register Range

| Range          | Reads   | Writes  | Total   |
|----------------|---------|---------|---------|
| PRAMIN (VRAM)  |       9 | 198,924 | 198,933 |
| OTHER          | 136,950 |   6,542 | 143,492 |
| PFB (MemCtrl)  |  43,002 |   1,670 |  44,672 |
| PMC            |   5,210 |     421 |   5,631 |
| PGRAPH_GLOBAL  |   2,331 |   2,565 |   4,896 |
| PTIMER         |   1,426 |      16 |   1,442 |
| NVDEC          |      18 |   1,001 |   1,019 |
| FECS           |     115 |      55 |     170 |
| FBPA           |      66 |      91 |     157 |

### Key Finding 1: nvidia-535 Does NOT Use SEC2 on GV100
**Zero SEC2 operations in the entire trace.** nvidia-535's Resource Manager (RM)
handles all falcon firmware loading directly from the CPU side, without the
SEC2/ACR authentication chain that nouveau uses.

SEC2 remains power-gated throughout nvidia-535's lifecycle. The BADF1100
fault pattern on SEC2 registers is not a bug — it's by design.

### Key Finding 2: FECS BootROM Method Interface
nvidia-535 loads FECS firmware through a proprietary silicon BootROM:

1. **Interrupt setup**: Write FECS+0x010=0xFFD2, FECS+0x01C=0xFFF2
2. **Start BootROM**: Write CPUCTL_ALIAS (0x130) = 0x02
3. **CPUCTL reads 0x50** (HALT+SRESET → running BootROM)
4. **Method commands** via FECS+0x500 (data) and FECS+0x504 (cmd):
   - `0x840=0x30, 0x500=0x802FD458, 0x504=0x03` → Load firmware from VRAM
   - Poll 0x800 until 0x10 (accepted)
   - `0x840=0x03, 0x500=0x802FD458, 0x504=0x09` → Execute firmware
   - Poll 0x800 until 0x01 (done)
5. **Firmware loaded into IMEM via internal DMA** (bypasses external IMEM lock)

The address 0x802FD458 is a VRAM physical address (bit 31 = FB flag).

### Key Finding 3: FECS IMEM is Hardware-Locked
Direct IMEM writes via IMEMC0/IMEMD0 always return zeros on readback.
DMA transfers to IMEM time out. PGRAPH reset and Falcon HRESET do not
unlock IMEM. This is Volta's ACR silicon enforcement — only the BootROM's
internal DMA can populate IMEM.

### Key Finding 4: NVDEC IMEM is Directly Writable
Unlike FECS, the NVDEC Falcon accepts direct IMEMC0/IMEMD0 writes.
The mmiotrace shows nvidia-535 uploading the scrubber firmware directly
to NVDEC IMEM (offset 0x084180/0x084184). NVDEC is a potential target
for sovereign falcon firmware.

### Key Finding 5: PMC_ENABLE Sequence
nvidia-535 toggles PMC_ENABLE 43 times during init, cycling through
engine enables: 0x5FECDFF1 → 0xFFFFFFFF → various toggled states.
Final warm state: 0x42001120 (the value ember sees after nvidia unload).

### Key Finding 6: HBM2 Survives nvidia Unload
After `rmmod nvidia`:
- PRAMIN read/write: VERIFIED (DEADBEEF, CAFEBABE)
- PMC_ENABLE: 0x42001120 (preserved)
- PFB_NISO_CFG0: 0xFFE00000 (memory controller active)
- All engines accessible except SEC2
- FECS accessible but IMEM empty (nvidia cleared firmware)

HBM2 training does NOT survive VFIO device release (FLR/PM reset).

## Architecture Implications

### For Titan V Sovereign Compute
1. **SEC2/ACR path is a dead end** — nvidia-535 doesn't use it, and
   SEC2 is power-gated with no known enable mechanism.
2. **FECS BootROM method interface** is the key to firmware loading.
   To use it, we need:
   a. Stage firmware in VRAM at a known address
   b. Reset PGRAPH to put FECS in SRESET+HALT state (CPUCTL=0x50)
   c. Start BootROM via CPUCTL_ALIAS
   d. Send method commands with VRAM firmware address
3. **NVDEC** is immediately usable for sovereign falcon code.
4. **Compute via nvidia-535** works inside the VM (12GB HBM2, CUDA 12.2).

### For K80 (Kepler) — Full Sovereignty Possible
Kepler does not have ACR silicon enforcement. FECS IMEM should be
directly writable. After VM-POST with nvidia-470:
1. `rmmod nvidia` preserves initialization
2. Upload open-source FECS firmware directly
3. Run compute dispatch via PGRAPH

## Files
- `data/titanv/mmiotrace/nvidia535_init_full.log` — 46MB, 1.16M MMIO ops
- `data/titanv/gv100_vbios_pg500.rom` — Titan V VBIOS dump
- Host nouveau firmware: `/lib/firmware/nvidia/gv100/{acr,gr,sec2,nvdec}/`

## Next Steps
1. **Titan V**: Use NVDEC as sovereign falcon target (IMEM writable)
2. **Titan V**: Investigate PGRAPH reset → FECS CPUCTL=0x50 → BootROM method replay
3. **K80**: Power cycle, then VM-POST with nvidia-470 for full sovereign FECS
4. **Both**: Extract PRAMIN firmware staging sequence from mmiotrace for replay
