# 068: FECS Direct Execution & PRIVRING Recovery Lessons

## Status: MAJOR BREAKTHROUGH — LS Security Bypass Confirmed on Clean Falcon

**Date**: 2026-03-16 (continued from 067)  
**Hardware**: 2× NVIDIA Titan V (GV100) — Titan #1 (03:00.0) on nouveau, Titan #2 (4a:00.0) on vfio-pci

---

## Summary

Building on 067's SEC2 breakthroughs, this session proved:

1. **FECS firmware executes from host-loaded IMEM** on a clean falcon (SCTL=0x3000)
2. **LS (Light Secure) boot protection is NOT enforced** when SCTL bits 0,5 are clear
3. **ACR bootloader and ACR firmware both execute** from host-loaded code
4. **SEC2 mailbox register written by firmware** — confirming real code execution
5. **PMC toggle of GR engine (bit 12) causes fatal PRIVRING fault** on GV100
6. **D3hot→D0 cycle produces the clean falcon state** needed for direct firmware loading

---

## Discovery 1: ACR Bootloader Executes from Host IMEM

On the VFIO card after D3hot→D0 (SEC2 SCTL=0x3000, IMEM/DMEM writable):

- Loaded ACR `bl.bin` (512 bytes) into IMEM at 0xFE00 (tag 0xFD)
- Set BOOTVEC = 0xFD00
- Issued START

```
TRACEPC: 0x00fd02  0x00fd02  0x001a96  0x00fd00  0x000000
                                         ^^^^^^^^
                                     BL ENTRY HIT!
```

The falcon jumped to our code. BL halted at 0xFD02 because the BLD descriptor
had no valid DMA target — no instance block was bound (SEC2+0x480 = 0).

---

## Discovery 2: ACR Firmware Executes via Direct PIO Load

Loaded the full `ucode_load.bin` directly into SEC2 IMEM+DMEM (no BL, no DMA):

- Non-sec code (256B) → IMEM @ 0x0 (tag 0x0)
- App code (11,776B) → IMEM @ 0x100 (secure tag)
- Data (4,096B) → DMEM @ 0

```
TRACEPC: 0x000012  0x24bd11  0x000012  0x00fd02
         ^^^^^^^^
     ACR CODE AT PC=0x12!
```

**Mailbox register changed from 0xCAFEBEEF to 0x00000000** — the firmware wrote
to the mailbox, confirming real execution. Halted due to missing WPR/DMA context.

---

## Discovery 3: FECS Direct Load — LS Security Bypass

The defining breakthrough of this session.

**FECS falcon state** (VFIO card after D3hot→D0 with BIOS PMC_ENABLE):
- CPUCTL = 0x10 (HALTED)
- SCTL = 0x3000 (CLEAN — bits 0,5 are CLEAR)
- IMEM: writable
- DMEM: writable
- BOOTVEC: writable
- Code limit: 0x8000 (32KB), Data limit: 0x2000 (8KB)

**Loaded FECS firmware directly** (fecs_inst.bin + fecs_data.bin) via PIO:

```
TRACEPC: 0x002835  0x002844  0x002835  0x0063ee  0x00608d  0x005fc5  0x00296c
         ^^^^^^^^                       ^^^^^^^^
     DEEP EXECUTION               NEAR END OF 25KB CODE
```

**Highest PC: 0x63EE** — that is offset 25,582 into a 25,632-byte firmware image.
The FECS firmware executed nearly to completion before halting.

Mailbox changed from 0xCAFEBEEF to 0x00000000 — firmware wrote to it.

### Why This Matters

On the BIOS POST card (SCTL=0x7021), host-loaded FECS code halts immediately
at PC=0 with no execution. The LS protection checks the SCTL register to decide
whether to trust host-loaded code. When SCTL bits 0,5 are clear (the "clean"
state from D3hot→D0), the protection is NOT enforced.

**This means we can bypass the entire ACR→FECS boot chain** by:
1. D3hot→D0 cycle to get clean falcon state
2. Enable BIOS PMC (0x5fecdff1)
3. Load FECS firmware directly into IMEM/DMEM
4. Start FECS

The remaining halt at PC=0x2835 is likely due to FECS waiting for GPCCS or
attempting a DMA operation that requires an instance block or channel context.

---

## Discovery 4: DMA Context Blocker

All DMA-based approaches failed because:

1. **Instance block register (SEC2+0x480) not host-writable** in the clean state
2. **Host-side DMA transfer registers** (DMATRFBASE, DMATRFCMD) accepted writes
   but did not execute transfers
3. **VFIO IOMMU mapping works** (verified) but falcon DMA engine ignores commands
4. **VRAM only 76KB accessible** (HBM2 not trained on the VFIO card)

The BL's DMA context (FALCON_DMAIDX_VIRT=1) requires a bound instance block.
Physical system contexts (2,3,4) also failed — likely need instance block binding
that only the HS ROM or BIOS can perform.

### Instance Block on BIOS POST Card

The nouveau card (BIOS POST state) has SEC2+0x480 = 0x00100000, set by the BIOS
during POST when HS protection was not yet active. This value cannot be changed
once the HS ROM has executed and locked down.

---

## Discovery 5: PRIVRING Fault — Lessons Learned

**PMC toggle of GR engine (bit 12) causes a fatal PRIVRING bus fault** on GV100.

After `PMC_ENABLE &= ~(1<<12)` followed by `PMC_ENABLE |= (1<<12)`:
- ALL GR registers return 0xBAD00100 (PRIVRING timeout)
- SEC2 registers also become inaccessible
- PRIVRING fault-clearing registers (0x12004C) are themselves behind PRIVRING
- D3hot→D0 cycle does NOT recover the bus
- PCI remove+rescan does NOT recover the bus
- Only a full system reboot recovers

**Root cause**: GV100's GR engine requires a specific initialization sequence after
power-on. A raw PMC toggle without the full init sequence leaves the PRIV ring
bus in an unrecoverable fault state.

**Lesson**: NEVER PMC-toggle bit 12 (GR) on GV100 without the full nouveau-style
init sequence. Use existing BIOS POST state instead.

---

## D3hot→D0 State Summary

The D3hot→D0 transition on a VFIO card produces this state:

| Resource | State | Writable? |
|----------|-------|-----------|
| PMC_ENABLE | 0x40000121 (minimal) | Yes → can set to BIOS 0x5fecdff1 |
| SEC2 SCTL | 0x3000 (clean) | Read-only |
| SEC2 IMEM | Empty | Yes |
| SEC2 DMEM | Empty | Yes |
| SEC2 BOOTVEC | 0x0 | Yes |
| SEC2 EMEM | Empty | Yes |
| FECS SCTL | 0x3000 (clean) | Read-only |
| FECS IMEM | Empty | Yes |
| FECS DMEM | Empty | Yes |
| FECS BOOTVEC | 0x0 | Yes |
| VRAM | ~76KB accessible at offset 0 | Yes (via PRAMIN) |
| HBM2 | NOT trained | N/A |

**Critical constraint**: Setting PMC_ENABLE to BIOS value unlocks FECS/GPCCS
access but does NOT toggle them — avoiding the PRIVRING fault.

---

## Paths Forward

### Path A: FECS+GPCCS Direct Load (Highest Priority)

1. D3hot→D0 → set BIOS PMC (no GR toggle)
2. Load GPCCS firmware (need correct BAR0 address — not at 0x41A000 on GV100)
3. Load FECS firmware
4. Start GPCCS, then FECS
5. FECS halt at 0x2835 may resolve once GPCCS is running

**Blocker**: GPCCS address not found in scan of 0x400000-0x520000. May require
GPC-specific addressing or additional engine enables.

### Path B: Instance Block Recovery

1. Use nouveau card (has 0x480 = 0x00100000 from BIOS POST)
2. Read instance block contents from VRAM via PRAMIN
3. Reverse-engineer the page table format
4. Replicate on clean card if a method to write 0x480 is found

### Path C: Nouveau Oracle — Let Nouveau Complete Init

1. Rebind nouveau on one Titan (triggers full init including ACR→FECS)
2. Capture the post-init state
3. Hot-swap to VFIO
4. GlowPlug maintains the warmed state

---

## Firmware Files Used

```
/lib/firmware/nvidia/gv100/acr/bl.bin           —  1,280B (ACR HS bootloader)
/lib/firmware/nvidia/gv100/acr/ucode_load.bin   — 18,688B (ACR load firmware)
/lib/firmware/nvidia/gv100/gr/fecs_inst.bin      — 25,632B (FECS instructions)
/lib/firmware/nvidia/gv100/gr/fecs_data.bin      —  4,788B (FECS data)
/lib/firmware/nvidia/gv100/gr/fecs_sig.bin       —    192B (FECS signature)
/lib/firmware/nvidia/gv100/gr/gpccs_inst.bin     — 12,640B (GPCCS instructions)
/lib/firmware/nvidia/gv100/gr/gpccs_data.bin     —  3,160B (GPCCS data)
```
