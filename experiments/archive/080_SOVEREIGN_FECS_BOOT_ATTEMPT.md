# Experiment 080: Sovereign FECS Boot — Initial Attempt

**Date:** 2026-03-23
**Goal:** Load FECS firmware directly into falcon IMEM/DMEM and start execution, bypassing ACR.
**Status:** BLOCKED — Direct IMEM upload succeeds but falcon remains in HRESET. ACR-managed boot required.

## Approach

Based on Exp 078 finding that FECS `HWCFG.SECURITY_MODE = 0` (secure=false), we attempted direct firmware upload:

1. Read firmware from `/lib/firmware/nvidia/gv100/gr/`:
   - `fecs_bl.bin` (576 bytes) — bootloader → IMEM at 0x0
   - `fecs_inst.bin` (25632 bytes) — instruction code → IMEM at 0x300 (256-aligned after BL)
   - `fecs_data.bin` (4788 bytes) — data → DMEM at 0x0
2. IINVAL (invalidate IMEM tags)
3. STARTCPU (release from HRESET)
4. Poll mailbox0 for handshake

## Results

### Boot Attempt 1 (CPUCTL = 0x02 → IINVAL)

```
Before: cpuctl=0x00000010 (HRESET)
After:  cpuctl=0x00000012 (HRESET + IINVAL sticky)
```
Bug: wrote IINVAL instead of STARTCPU. Fixed.

### Boot Attempt 2 (IINVAL then STARTCPU = 0x01)

```
Before: cpuctl=0x00000012
After:  cpuctl=0x00000010 (HRESET, STARTCPU trigger cleared)
```

STARTCPU bit 0 was accepted (trigger bit cleared after write), but the falcon remained in HRESET. No mailbox response after 2 seconds.

### GPCCS — Same Result

```
GPCCS cpuctl: 0x00000010 → 0x00000012 → 0x00000010
No mailbox response.
```

## Analysis

### Why Direct Upload Fails

GV100 uses **ACR-managed falcon boot** (Lazy Secure model):

1. FECS and GPCCS are "managed falcons" — they cannot self-boot from host IMEM uploads
2. The ACR (Application Context Resource) firmware running on a "Heavy Secure" (HS) falcon (PMU or SEC2) must:
   - DMA-load firmware from a WPR (Write-Protected Region) in system memory
   - Verify firmware signatures
   - Release the managed falcon from HRESET
3. `HWCFG.SECURITY_MODE = 0` means FECS doesn't require **signed** firmware, but it still requires **ACR-managed boot** — the host cannot directly release it from HRESET

### What `secure=false` Actually Means

| Aspect | secure=true (PMU) | secure=false (FECS) |
|--------|-------------------|---------------------|
| Signed firmware required | Yes | No |
| ACR-managed boot | Yes | **Yes** (still managed) |
| Host IMEM upload | Blocked | Accepted (but ineffective) |
| HRESET release | ACR only | ACR only |
| DMA firmware load | ACR only | ACR only |

The GV100 falcon architecture separates "security" (signature verification) from "management" (who controls HRESET). Even non-secure falcons on Volta are ACR-managed.

## Path Forward (Exp 080a-d)

To achieve sovereign FECS boot, we need the full ACR chain:

### 080a: ACR Blob Parser
- Parse `/lib/firmware/nvidia/gv100/acr/bl.bin` and `ucode_load.bin`
- Extract LS (Lazy Secure) falcon image descriptors
- Map the WPR header format

### 080b: WPR Region Construction
- Allocate DMA memory for WPR
- Build WPR header with LS falcon descriptors (FECS, GPCCS)
- Embed firmware images in WPR layout

### 080c: ACR HS Falcon Boot
- Load ACR bootloader to PMU/SEC2 falcon IMEM
- Configure ACR to boot from our WPR region
- Start ACR execution
- ACR loads FECS/GPCCS and releases HRESET

### 080d: Full Sovereign Dispatch
- Verify FECS running (mailbox0 != 0)
- Submit compute shader via GPFIFO
- Sync — if successful, full sovereignty achieved

## Code Added

- `crates/coral-driver/src/nv/vfio_compute/fecs_boot.rs` — generic falcon IMEM/DMEM upload + boot
  - `FecsFirmware::load()` — reads firmware from `/lib/firmware/`
  - `falcon_boot()` — uploads BL+inst+data and attempts start
  - `boot_fecs()` / `boot_gpccs()` / `boot_gr_falcons()` — high-level boot functions
- `crates/coral-driver/tests/hw_nv_vfio.rs` — `vfio_sovereign_fecs_boot` test

## Hardware Observations

The IMEM/DMEM upload itself succeeded (no errors, registers accepted writes). The falcon infrastructure (upload ports, mailbox registers) is fully functional. The blocker is specifically the HRESET release mechanism being locked to ACR.
