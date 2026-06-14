# Experiment 174: K80 Sovereign Boot via VM Warm + Direct FECS PIO

**Date:** 2026-04-18
**GPU:** Tesla K80 (GK210, SM35) at 0000:4c:00.0
**Goal:** First sovereign compute on Kepler using VM-warmed GDDR5 + direct PIO falcon upload (no ACR)

## Infrastructure Changes

### coral-driver: sovereign_init.rs
1. **Kepler chip_id mapping**: Added `0x0F0..=0x0FF => 35` (GK210 reports chip_id=0x0F2, was defaulting to SM 70)
2. **Kepler falcon boot path**: New `kepler_falcon_boot()` function:
   - Skips ACR solver entirely (Kepler has no ACR/WPR)
   - Loads firmware from `/lib/firmware/nvidia/gk210/` (inst+data, no bootloader)
   - Boots GPCCS first, then FECS (correct Kepler ordering)
   - Uses direct PIO upload via existing `falcon_upload_imem`/`falcon_upload_dmem`
3. **HBM2 skip for Kepler**: `is_kepler(sm)` forces warm detection (GDDR5, not HBM2)
4. **GR init skip for Kepler**: FECS already booted in falcon_boot stage
5. **PGRAPH clock gating**: Write PMC 0x260=1 before falcon load (nouveau pmc_unk260)
6. **GR BAR0 init**: Apply sw_nonctx/sw_bundle_init register writes for Kepler

### Firmware symlinks
Created `/lib/firmware/nvidia/gk210/gr/` symlinks for GrFirmwareBlobs compatibility.

## VM Warm Phase

1. Bound K80 to vfio-pci (IOMMU group 37, clean isolation)
2. Created disposable VM with K80 passthrough (`managed='yes'`)
3. Installed nvidia-470.256.02 inside VM
4. nvidia-smi confirmed: K80 functional, 12206 MiB GDDR5, 43C, 58W
5. Captured warm BAR0 state: 255,578 non-zero registers
6. FECS CPUCTL=0x10 (HALTED), HWCFG=0x20402050, 282 FECS regs
7. After `rmmod nvidia`: all registers preserved (255,578 unchanged)
8. VM shutdown returned K80 to host vfio-pci

## Sovereign Init Results

### Attempt 1 (before chip_id fix)
- chip_id=0x0F2 decoded to SM 70 (wrong, default)
- Ran HBM2 training pipeline on GDDR5 → PRAMIN sentinel test failed (0xbad0fb01)
- **Root cause**: chip_id_to_sm lacked GK210 range

### Attempt 2 (after chip_id fix + PGRAPH clock gating)
- chip_id=0x0F2 → SM 35 (correct)
- HBM2 training skipped (Kepler detected)
- GR BAR0 init: 1316 writes loaded, 11 BAR0 + 170 sw_nonctx applied
- GPCCS at 0x41A000: returns `0xbadf3000` (PRI error — GPC not powered)
- FECS at 0x409000: CPUCTL=0x10 (accessible, halted from nvidia-470)
- FECS PIO upload: 15,356 bytes inst + 1,920 bytes data
- FECS start: CPUCTL→0x12 (STARTCPU attempted, remained in reset)
- **FECS boot: TIMEOUT** — falcon never responded via mailbox

## Key Finding

**The VM shutdown reset killed the GDDR5 training.** Even though Kepler has no FLR,
`managed='yes'` passthrough triggers an SBR or D3hot→D0 transition on VM release
that resets the memory controller. The GPU is partially alive (PMC responds,
BOOT0=0x0F22D0A1, FECS registers accessible) but GDDR5 is untrained:
- PRAMIN reads return `0xbad0fb0X` (bad FB sentinel)
- GPC registers at 0x41A000 return `0xbadf3000` (PRI error — GPCs need MC)
- FECS firmware halts immediately because it can't set up GR context without VRAM

## Paths Forward

1. **VM with `managed='no'` hot-attach/detach** — prevent libvirt from resetting GPU on release
2. **GDDR5 init via VBIOS DEVINIT replay** — parse GK210 VBIOS for memory init sequence
3. **Nouveau warm handoff** — if nouveau can safely probe GK210 without locking USB
   (Kepler root complex 0x40 shares with USB controller, PCIe completion timeouts
   cascade to USB on cold hardware)

## Delivered

- Kepler sovereign init path in coral-driver (chip_id detection, no-ACR falcon boot)
- nvidia-470 VM warm capture artifacts at `/var/lib/coralreef/reagent-artifacts/exp174-k80/`
- GK210 firmware symlinks at `/lib/firmware/nvidia/gk210/gr/`
