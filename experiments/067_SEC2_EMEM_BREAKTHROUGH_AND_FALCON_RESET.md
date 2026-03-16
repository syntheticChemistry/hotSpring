# 067: SEC2 EMEM Breakthrough & Falcon Reset State Machine

## Status: CRITICAL BREAKTHROUGHS — Two Attack Vectors Confirmed

**Date**: 2026-03-16 (continued from 066)  
**Hardware**: 2x NVIDIA Titan V (GV100) — Titan #1 (03:00.0) on nouveau, Titan #2 (4a:00.0) on vfio-pci

---

## Summary of Discoveries

This experiment session proved three fundamental things about GV100 SEC2 falcon control:

1. **SEC2 EMEM (Extended Memory) is fully host-writable** via PIO at BAR0+0x87AC0/0x87AC4
2. **After nouveau's driver reset, SEC2 IMEM/DMEM/BOOTVEC become host-writable** (HS protection lifted)
3. **The VFIO card retains BIOS POST HS lockdown** — a PCI Function-Level Reset or driver reset is required to clear it

These findings define two viable paths to sovereign SEC2 boot.

---

## Discovery 1: SEC2 EMEM Is Writable (Both Cards)

The EMEM (Extended Memory) interface on SEC2 uses PIO registers:
- **EMEMC** (control): `SEC2_BASE + 0xAC0 + (port * 8)` → port 0 = `0x087AC0`
- **EMEMD** (data): `SEC2_BASE + 0xAC4 + (port * 8)` → port 0 = `0x087AC4`

Write mode: `EMEMC = BIT(24) | byte_address`  
Read mode: `EMEMC = BIT(25) | byte_address`

### Test Results (VFIO card, HS locked, SCTL=0x7021)

```
Write 0xDEADBEEF → readback = 0xDEADBEEF  MATCH!
Sequential auto-increment (8 words) → ALL MATCH!
```

**EMEM is writable even when the falcon is in full HS lockdown.** This is by design —
EMEM is the host's interface for providing the HS bootloader to the falcon's internal ROM.

### Significance

This is the exact mechanism nouveau uses (`gp102_flcn_pio_emem_wr` in `falcon/gp102.c`).
The internal ROM at BOOTVEC=0xFD00 reads EMEM, verifies the bootloader signature,
copies it to IMEM, and begins execution. EMEM is the intended "mailbox" between host
and the HS boot ROM.

---

## Discovery 2: Two Falcon States — HS Locked vs Reset Clean

### VFIO Card (03:00.0) — BIOS POST State (HS Locked)

After BIOS POST, the HS internal ROM has already executed:

| Register | Value | Meaning |
|----------|-------|---------|
| CPUCTL   | 0x60  | Bits 5+6: STOPPED + secure mode |
| BOOTVEC  | 0xFD00 | ROM's own boot vector (read-only from host) |
| SCTL     | 0x7021 | HS mode active (bits 0,5,12,13,14) |
| IMEM     | Protected | Writes ignored, reads return stale data |
| DMEM     | Protected | Writes return FAULT-like patterns |
| EMEM     | **Writable** | PIO interface always accessible |

**PMC_ENABLE** = 0x5fecdff1 (many engines enabled by BIOS)

### Nouveau Card (4a:00.0) — Driver Reset State (Clean)

After nouveau's driver bind/reset cycle:

| Register | Value | Meaning |
|----------|-------|---------|
| CPUCTL   | 0x10  | Bit 4: HALTED (ROM ran and halted, no valid FW) |
| BOOTVEC  | 0x00  | **Host-writable!** |
| SCTL     | 0x3000 | HS NOT fully locked (only bits 12,13) |
| IMEM     | **Writable!** | Full host access confirmed |
| DMEM     | **Writable!** | Full host access confirmed |
| EMEM     | **Writable** | PIO interface always accessible |

**PMC_ENABLE** = 0x40000121 (minimal engines: bits 0,5,8,30)

### Key Difference

The SCTL register tells the story:
- **0x7021**: Bits 0 (HS enabled) and 5 (HS auth done) are SET → full lockdown
- **0x3000**: Bits 0 and 5 are CLEAR → HS mode not fully engaged

Nouveau's driver binding triggers a comprehensive PCI/GPU reset that clears the HS ROM
state. VFIO-pci preserves the BIOS POST state by design.

---

## Discovery 3: Nouveau Card D3hot Power Management

The nouveau card enters D3hot (suspended) via runtime PM when idle. BAR0 reads return
all 0xFFFFFFFF in this state.

**Fix**: `echo on > /sys/bus/pci/devices/0000:4a:00.0/power/control`

After wake-up: BAR0 fully accessible, SEC2 in clean reset state.

---

## Discovery 4: PMC Engine Control

PMC_ENABLE (BAR0 + 0x200) controls engine power:

| Bit | Engine | Effect when cleared |
|-----|--------|-------------------|
| 5   | SEC2   | Full power-off (registers return FAULT) |
| 11  | (related) | Changes CPUCTL from 0x60 → 0x40 |
| 28  | (related) | Same as bit 11 |

**PMC toggle does NOT clear HS state on the VFIO card.** After power-off/on, the HS
ROM re-executes immediately and re-locks everything. The ROM boot is sub-millisecond.

On the nouveau card, PMC toggle also triggers the ROM, which halts (CPUCTL=0x12)
because no valid firmware is in EMEM. But IMEM/DMEM remain writable because the
nouveau driver's initial reset cleared the HS fuse state.

---

## Discovery 5: Falcon Execution Attempted

On the nouveau card (clean state), we successfully:

1. Loaded ACR `bl.bin` code (512 bytes) into IMEM at address 0xFE00 with tag 0xFD
2. Loaded BL data (256 bytes) into DMEM at offset 0
3. Set BOOTVEC = 0xFD00 (matching `start_tag << 8` from bl.bin descriptor)
4. Issued CPUCTL START (0x02)

**Result**: Falcon immediately halted (CPUCTL=0x12). TRACEPC showed execution at
ROM addresses (0x0955-0x4E5A range), NOT our loaded code at 0xFD00.

**Root cause**: The internal ROM executes FIRST from its own hardware-mapped memory,
which shadows the user IMEM addresses. The ROM checks EMEM for a valid signed
bootloader. Without a properly signed image in EMEM, the ROM halts.

### IINVAL Clears Halt Latch

Writing CPUCTL = 0x01 (IINVAL — instruction cache invalidate) clears the HALTED
bit (4), changing CPUCTL from 0x12 → 0x10. This doesn't restart execution but
confirms the host has write access to CPUCTL after reset.

---

## Discovery 6: Falcon Info Registers

SEC2 on GV100 (from register 0x12C):

| Field | Value | Meaning |
|-------|-------|---------|
| version | 6 | Falcon v6 (not v5 as previously assumed) |
| secret | 3 | HS (Heavy Secure) — highest security level |
| code_ports | 1 | Single IMEM port |
| data_ports | 1 | Single DMEM port |
| code.limit | 0x10000 | 64 KB IMEM |
| data.limit | 0x10000 | 64 KB DMEM |

---

## Discovery 7: PTOP Device Table Decoded

The GV100 topology table at BAR0 + 0x22700 contains valid entries. SEC2 entry at
indices [18-20]:

```
[18] 0x8c679c3e  → type 2: reset/engine info
[19] 0x80000037  → type 2: continuation
[20] 0x00087075  → type 0: BAR0=0x087000, type=0x35
```

Full table has 54 entries covering all engines (GR, CE, NVDEC, NVENC, PMU, SEC2, etc.).

---

## Two Paths Forward

### Path A: Sovereign EMEM-Based Boot (Preferred)

The designed loading mechanism:

1. **Load signed ACR bootloader into SEC2 EMEM** (via PIO at 0x87AC0/0x87AC4)
2. **PMC reset SEC2** (toggle bit 5 in PMC_ENABLE)
3. **ROM reads EMEM**, verifies NVIDIA signature, copies to IMEM, jumps to BL
4. **BL loads main ACR firmware** from VRAM via DMA
5. **ACR loads FECS/GPCCS** LS firmwares and starts them

**Blocker**: The ROM verifies NVIDIA's cryptographic signature. We have the signed
firmware files (`/lib/firmware/nvidia/gv100/sec2/{desc,image,sig}.bin`), but we need
to present them in the exact format the ROM expects in EMEM.

**Next step**: Study how nouveau's `nvkm_falcon_fw_load()` formats the EMEM payload.
The `nvfw_bl_desc` from bl.bin gives: `start_tag=0xFD`, `code_off=0`, `code_size=0x200`,
`data_off=0x200`, `data_size=0x100`. The signature patching happens via
`nvkm_falcon_fw_patch()` which writes `sig_prod` data at `sig_base_img` offset in
the firmware image.

### Path B: Post-Reset Direct Loading (Nouveau Oracle)

Since the nouveau card has fully writable IMEM/DMEM/BOOTVEC:

1. **Use nouveau as oracle** to get SEC2 into clean reset state
2. **Load firmware directly into IMEM/DMEM** (bypassing EMEM/ROM entirely)
3. **Set BOOTVEC** to the firmware entry point
4. **Start CPU**

**Blocker**: The internal ROM still executes first and halts before our code runs.
Even with BOOTVEC set to our code, the ROM's halt preempts execution.

**Potential fix**: Load the firmware into EMEM in the correct format, then PMC reset.
The ROM will find valid signed content, copy it to IMEM, and execute the bootloader.
Our IMEM writes are overwritten by the ROM's copy, but the BL then takes over.

### Path C: Hybrid — EMEM Load on Clean Card

Combine both paths:

1. **Wake nouveau card** (set power/control to "on")
2. **Load signed firmware into EMEM** (already proven writable)
3. **PMC reset** SEC2 (toggle bit 5)
4. **ROM boots from EMEM** with NVIDIA-signed firmware
5. **ACR starts FECS/GPCCS**
6. **GR engine becomes available** for sovereign compute dispatch

This is the most promising path — it uses NVIDIA's own signed firmware through the
designed loading mechanism, just orchestrated from Rust instead of nouveau's kernel code.

---

## Discovery 8: BL and ACR Code Successfully Executed

### BL Execution (via IMEM PIO load)

After loading ACR `bl.bin` into IMEM at 0xFE00 (tag 0xFD) and setting BOOTVEC=0xFD00:

```
TRACEPC: 0x00fd02 0x00fd02 0x001a96 0x00fd00 0x000000
                                      ^^^^^^^^
                                  BL ENTRY REACHED!
```

The falcon jumped to 0xFD00 and began executing the bootloader. It halted at 0xFD02
(2nd instruction) because the BLD descriptor had null DMA addresses.

### Direct ACR Firmware Execution (via IMEM+DMEM PIO load)

Loaded the full `ucode_load.bin` data section:
- Non-sec code (256B) → IMEM @ 0x0
- App code (11,776B) → IMEM @ 0x100 (secure tag)
- Data (4,096B) → DMEM @ 0

```
TRACEPC: 0x000012 0x24bd11 0x000012 0x00fd02
         ^^^^^^^^
     ACR CODE RUNNING AT PC=0x12!
```

The ACR firmware executed its first ~4-5 instructions before halting. The halt is
due to missing WPR (Write-Protected Region) + DMA setup — the firmware tries to
DMA-read LS falcon images from VRAM.

### What This Proves

1. **Code execution from host-loaded IMEM works** on the clean falcon (SCTL=0x3000)
2. **The BL is functional** — it starts and tries to DMA
3. **The ACR firmware starts** — non-secure entry point at 0x0 executes
4. **The only remaining blocker is DMA/WPR setup** — providing the LS firmware images
   in a DMA-accessible location

### Remaining Steps for Full ACR Boot

1. Set up falcon instance block (GPU page table) for DMA access
2. Allocate WPR region in VRAM or DMA-mapped system memory
3. Build WPR layout: headers + FECS/GPCCS firmware images + signatures
4. Write WPR region descriptor into ACR's DMEM data section
5. Boot → ACR loads FECS/GPCCS → GR engine available

---

## Key Source Code References

| File | Function | Purpose |
|------|----------|---------|
| `falcon/fw.c` | `nvkm_falcon_fw_boot()` | Full boot sequence: reset→patch→load→boot |
| `falcon/fw.c` | `nvkm_falcon_fw_ctor_hs()` | Parses bl.bin + ucode_load.bin headers |
| `falcon/gm200.c` | `gm200_flcn_fw_boot()` | Writes BOOTVEC, START, waits for HALTED |
| `falcon/gm200.c` | `gm200_flcn_fw_load()` | IMEM/DMEM PIO loading, BL to code.limit |
| `falcon/gm200.c` | `gm200_flcn_enable()` | PMC toggle + mem scrub wait + BOOT_0 write |
| `falcon/gp102.c` | `gp102_flcn_pio_emem_wr_init()` | EMEM PIO: `wr32(0xAC0, BIT(24) \| base)` |
| `subdev/acr/gp102.c` | `gp102_acr_load_setup()` | WPR region descriptor setup |
| `subdev/acr/gv100.c` | `gv100_acr_load_fwif[]` | GV100 uses `gp108_acr_load_0` on SEC2 |

---

## Firmware Files Required

```
/lib/firmware/nvidia/gv100/acr/bl.bin           →  1,280 bytes (ACR HS bootloader)
/lib/firmware/nvidia/gv100/acr/ucode_load.bin   → 18,688 bytes (ACR load firmware)
/lib/firmware/nvidia/gv100/sec2/desc.bin         →    656 bytes (SEC2 HS descriptor)
/lib/firmware/nvidia/gv100/sec2/image.bin        → 91,136 bytes (SEC2 HS firmware)
/lib/firmware/nvidia/gv100/sec2/sig.bin          →    192 bytes (SEC2 HS signature)
/lib/firmware/nvidia/gv100/gr/fecs_*.bin         → FECS LS firmware (inst+data+sig+bl)
/lib/firmware/nvidia/gv100/gr/gpccs_*.bin        → GPCCS LS firmware
```

## Register Quick Reference

```
SEC2 Base: 0x087000
  CPUCTL:   +0x100  (bit 0=IINVAL, bit 1=START, bit 4=HALTED, bit 5=STOPPED)
  BOOTVEC:  +0x104  (entry point address)
  HWCFG:    +0x108  (code/data limits)
  0x10C:    +0x10C  (mem scrub status, bits [2:1])
  0x12C:    +0x12C  (version, secret, ports)
  MAILBOX0: +0x130
  MAILBOX1: +0x134
  EXCI:     +0x148  (exception info, [31:16]=cause, [15:0]=PC)
  TRACEPC:  +0x14C  (read after writing index to 0x148)
  IMEMC:    +0x180  (BIT(24)=write, BIT(25)=read, BIT(28)=secure)
  IMEMD:    +0x184
  IMEMT:    +0x188  (page tag)
  DMEMC:    +0x1C0  (BIT(24)=write, BIT(25)=read)
  DMEMD:    +0x1C4
  SCTL:     +0x240  (security control — read-only in HS mode)
  EMEMC:    +0xAC0  (BIT(24)=write, BIT(25)=read)
  EMEMD:    +0xAC4

PMC_ENABLE: 0x200 (bit 5 = SEC2 power)
PMC_BOOT_0: 0x000 (chip ID, write to falcon 0x084 after reset)
```
