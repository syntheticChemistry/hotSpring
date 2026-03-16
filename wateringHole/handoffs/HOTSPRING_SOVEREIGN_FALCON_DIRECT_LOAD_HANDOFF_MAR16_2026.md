# Sovereign Falcon Direct Load — FECS Executes from Host IMEM

**Date:** March 16, 2026  
**From:** hotSpring (ecoPrimals)  
**To:** toadStool, coralReef, barraCuda  
**License:** AGPL-3.0-only  
**Experiments:** 066, 067, 068

---

## Executive Summary

- **FECS firmware ran from host-loaded IMEM** on a GV100 Titan V — PC reached
  0x63EE (25,582 of 25,632 bytes). LS security protection bypassed on clean falcon.
- **ACR bootloader and ACR firmware both execute** from PIO-loaded IMEM/DMEM on
  SEC2. Mailbox register confirms real code execution.
- **D3hot→D0 produces a "clean" falcon state** (SCTL=0x3000) where IMEM, DMEM,
  BOOTVEC, and EMEM are all host-writable. This is the key to direct loading.
- **SEC2 EMEM is always host-writable** — even in full HS lockdown (SCTL=0x7021).
- **PRIVRING FATAL**: PMC toggle of GR bit 12 on GV100 causes unrecoverable bus
  fault. Never toggle GR power without full init sequence.
- **GlowPlug daemon operational**: personality hot-swap, health monitor, auto-D0
  recovery. Ready for integration with toadStool socket API.

---

## Part 1: The Falcon Boot Chain (What We Learned)

```
BIOS POST → HBM2 trained, all engines on, all falcons HS-locked
  ↓
D3hot → D0 = clean falcon state (SCTL=0x3000, IMEM/DMEM writable)
  ↓
Host PIO → Load firmware into IMEM/DMEM
  ↓
SET BOOTVEC, START → Falcon executes loaded code
```

Key registers per falcon:

| Register | Offset | Purpose |
|----------|--------|---------|
| CPUCTL | +0x100 | bit 1=START, bit 4=HALTED |
| BOOTVEC | +0x104 | Entry point address |
| SCTL | +0x240 | Security: bit 0=HS active, bit 5=HS auth done |
| IMEMC | +0x180 | IMEM control (bit 24=write, bit 25=read) |
| IMEMD | +0x184 | IMEM data (auto-increment per write) |
| IMEMT | +0x188 | IMEM page tag |
| DMEMC | +0x1C0 | DMEM control |
| DMEMD | +0x1C4 | DMEM data |

---

## Part 2: Action Items by Primal

### coralReef

1. **Implement D3hot→D0 clean cycle** in `coral-driver/src/vfio/`:
   - `echo auto > power/control` → wait D3hot → `echo on > power/control`
   - Verify SCTL=0x3000 on SEC2/FECS after wake
   - Set PMC_ENABLE to BIOS value (0x5fecdff1) WITHOUT toggling individual bits

2. **Implement FECS PIO firmware loader** in `coral-driver/src/nv/`:
   - Read `fecs_inst.bin` and `fecs_data.bin` from `/lib/firmware/nvidia/gv100/gr/`
   - Load IMEM page-by-page (64 words per page, tag per page)
   - Load DMEM with auto-increment from offset 0
   - Set BOOTVEC=0, issue START

3. **Find GPCCS address on GV100**: Not at legacy 0x41A000. Check GPC-specific
   addressing (may be at 0x502000+offset per GPC, or require PTOP decoding).

4. **NEVER PMC-toggle bit 12** (GR engine) on GV100. Fatal PRIVRING fault.

### toadStool

1. **Wire GlowPlug socket into ResourceOrchestrator**:
   - `ListDevices` → populate GPU inventory
   - `Health` → feed sysmon telemetry
   - `Swap` → orchestrate oracle warm-up cycles

2. **FECS readiness probe**: After GlowPlug reports D0, check FECS CPUCTL for
   RUNNING (bit 1 set, bit 4 clear). This indicates GR engine is ready.

3. **HBM2 lifecycle**: GlowPlug auto-D0 prevents HBM2 training loss. VFIO fd
   close triggers PM reset that destroys training — GlowPlug must hold fds.

### barraCuda

1. **No code changes needed** — barraCuda writes WGSL shaders, coralReef compiles
   them, toadStool dispatches. The sovereign pipeline is:
   ```
   barraCuda shader → coralReef compile → toadStool dispatch → GlowPlug GPU
   ```

2. **For testing**: Once FECS is running, the NOP shader dispatch test
   (`vfio_dispatch_nop_shader`) should pass. This unblocks the full pipeline.

---

## Part 3: Register Quick Reference

```
SEC2:    0x087000 (CPUCTL +0x100, EMEM +0xAC0/0xAC4)
FECS:    0x409000 (CPUCTL +0x100, IMEM +0x180/0x184)
PMU:     0x10A000 (HS locked, not useful)
GPCCS:   ??? (not 0x41A000 on GV100)
PMC:     0x000200 (enable bits — bit 5=SEC2, bit 12=GR DANGER)
PRAMIN:  0x700000 (1MB window, controlled by 0x001700)
```

---

## Part 4: Firmware Files

All in `/lib/firmware/nvidia/gv100/`:

| File | Size | Target |
|------|------|--------|
| `gr/fecs_inst.bin` | 25,632B | FECS IMEM |
| `gr/fecs_data.bin` | 4,788B | FECS DMEM |
| `gr/fecs_sig.bin` | 192B | FECS signature (may not be needed for clean falcon) |
| `gr/gpccs_inst.bin` | 12,640B | GPCCS IMEM |
| `gr/gpccs_data.bin` | 3,160B | GPCCS DMEM |
| `acr/bl.bin` | 1,280B | ACR bootloader (loads into IMEM at code_limit-512) |
| `acr/ucode_load.bin` | 18,688B | ACR firmware (needs DMA — not yet working) |

---

## Part 5: What Does NOT Work

| Attempt | Result | Why |
|---------|--------|-----|
| BL DMA from system memory | Stuck at PC=0xFD02 | IOMMU blocks, no instance block |
| BL DMA from VRAM | Stuck at PC=0xFD02 | VRAM not trained (76KB accessible) |
| Host-side DMA registers | No transfer | Registers writable but engine ignores |
| Instance block bind (0x480) | Register clears to 0 | Not host-writable in clean state |
| PMC toggle GR bit 12 | Fatal PRIVRING fault | Unrecoverable without reboot |
| PCI remove+rescan after fault | Still faulted | PRIVRING state survives |
