# 066: SEC2/ACR Falcon Boot Chain — Complete Analysis

## Status: BREAKTHROUGH — Path to Sovereign Compute Identified

**Date**: 2026-03-16  
**Hardware**: 2x NVIDIA Titan V (GV100), RTX 5060

---

## What We Learned Today

### 1. FECS Falcon IMEM Is Host-Writable (But Won't Execute)

We successfully loaded `fecs_inst.bin` (25,632 bytes) and `fecs_data.bin` (4,788 bytes)
into the FECS falcon's IMEM/DMEM via BAR0 at 0x409000. **Verified byte-for-byte.**

The IMEMC register uses word-addressed format: `IMEMC = word_index << 2`, with IMEMT
(tag) set every 64 words (256-byte block boundary).

**Result**: Falcon halts at PC=0 before executing a single instruction. This is
**hardware-enforced LS (Light Secure) boot protection** — the v5 falcon CPU physically
refuses to run code not loaded through ACR. SCTL register (0x3000) is read-only hardware.

### 2. PMU Falcon Is Fully HS-Locked

PMU at 0x10A000: IMEM returns constant 0x004000d0 regardless of writes. DMEM returns
0xDEAD5EC2. DMA registers are also read-only. This is **HS (Heavy Secure) protection** —
the PMU has internal ROM (BOOTVEC=0x10000) and the host cannot modify its memory.

### 3. SEC2 Was At The Wrong Address

We were reading SEC2 at 0x840000 (legacy address) and getting FAULT. The GV100 topology
table (PTOP at 0x022700) reveals:

```
SEC2 → BAR0 address 0x087000, reset bit 14, PMC_ENABLE bit 14 IS SET
```

**SEC2 is alive and enabled** — we just weren't looking in the right place.

### 4. SEC2 State (at 0x087000)

| Register   | Value       | Meaning                                    |
|-----------|-------------|---------------------------------------------|
| CPUCTL    | 0x00000060  | STOPPED + HALTED (waiting for firmware)     |
| BOOTVEC   | 0x0000FD00  | Pre-set boot vector (internal ROM)          |
| HWCFG     | 0x20420100  | 64 KiB IMEM, 64 KiB DMEM                  |
| SCTL      | 0x00007021  | HS-capable, bit 0 set (secure state)        |
| MAILBOX1  | 0x0003FFFE  | Non-zero status (pre-init handshake?)       |

SEC2 IMEM/DMEM are HS-protected from direct host writes (like PMU). But **DMA transfer
registers are partially writable** (DMATRFBASE, DMATRFMOFFS, DMATRFCMD, DMATRFFBOFFS).

### 5. "pmu: firmware unavailable" Is A Red Herring

From kernel source (`gm200.c`):
```c
int gm200_pmu_nofw(struct nvkm_pmu *pmu, ...) {
    nvkm_warn(&pmu->subdev, "firmware unavailable\n");
    return 0;  // SUCCESS — just a warning
}
```

PMU firmware is intentionally unavailable for GP102+. The PMU subdevice initializes
successfully without firmware. This is NOT the blocker.

### 6. The Complete Boot Chain Exists in Nouveau

From kernel source analysis of the GV100 chipset definition (`base.c`):

```
.pmu  = gp102_pmu_new    → nofw (intentional, returns success)
.sec2 = gp108_sec2_new   → loads nvidia/gv100/sec2/{desc,image,sig}.bin
.acr  = gv100_acr_new    → loads nvidia/gv100/acr/{bl,ucode_load}.bin on SEC2
.gr   = gv100_gr_new     → loads nvidia/gv100/gr/{fecs,gpccs}_{inst,data,bl,sig}.bin
```

The intended init sequence:
1. SEC2 engine loads its LS firmware (sec2/*.bin) as an ACR WPR entry
2. ACR HS firmware (acr/bl.bin + acr/ucode_load.bin) loads onto SEC2
3. SEC2 hardware verifies the HS bootloader against chip fuses
4. SEC2 enters HS mode, runs ACR
5. ACR builds Write Protected Region (WPR) in VRAM with FECS/GPCCS firmware
6. ACR bootstraps FECS, then GPCCS
7. GR engine initializes with running FECS/GPCCS

**All firmware files exist. All code paths exist. But something fails silently.**

### 7. The Silent Failure

Kernel logs show ZERO messages about SEC2, ACR, FECS, or GPCCS. Only the harmless
PMU warning appears. The most telling log entry:

```
bus: MMIO read of 00000000 FAULT at 000600 [ PRIVRING TIMEOUT ]
```

Register 0x600 (PMC_DEVICE_ENABLE on Volta) doesn't exist at that address on GV100.
Nouveau's MC (Master Control) code uses gm200-era register addresses that moved on Volta.
This PRIVRING fault during init may cascade and prevent SEC2/ACR from completing.

### 8. GSP-RM Is NOT Active For GV100

```c
static struct nvkm_gsp_fwif gv100_gsps[] = {
    { -1, gv100_gsp_nofw, &gv100_gsp },  // no-op
    {}
};
```

The traditional init path IS active. GSP-RM (which would bypass everything) is not used.

---

## GV100 Topology Map

From PTOP register scan at 0x022700:

| Engine  | BAR0 Base  | Reset Bit | PMC Enabled? | Status         |
|---------|-----------|-----------|--------------|----------------|
| GR      | 0x400000  | 12        | YES          | FECS empty     |
| CE0     | 0x104000  | 6         | YES          | Working        |
| CE1     | 0x104000  | 7         | YES          | Working        |
| NVDEC   | 0x084000  | 15        | YES          | Clock-gated?   |
| **SEC2**| **0x087000**| **14**  | **YES**      | **Stopped, waiting** |
| NVENC0  | 0x1C8000  | 18        | YES          | Available      |
| GSP     | 0x110000  | -         | -            | No-op          |

---

## Path Forward: Three Attack Vectors

### Vector A: Fix Nouveau's GV100 Init (Kernel Patch)

The PRIVRING TIMEOUT at 0x600 suggests nouveau's MC code doesn't handle Volta's register
map correctly. If we patch the kernel module to:
1. Skip the faulting register access at 0x600
2. Or use the correct Volta-era device enable mechanism

Then SEC2/ACR/FECS init should proceed naturally. This is the **least invasive** fix.

### Vector B: Sovereign DMA-Based SEC2 Loader (Userspace)

Since SEC2's DMA registers ARE writable:
1. Write firmware to VRAM via PRAMIN (needs proper window setup)
2. Configure SEC2 DMA to transfer from VRAM to IMEM
3. Start SEC2 → hardware verifies → ACR boots
4. ACR loads FECS/GPCCS
5. We now have a warm GPU for VFIO compute

This is the **full sovereign** path — no kernel driver needed for GR init.

### Vector C: Hybrid Oracle (Nouveau Warm + VFIO Compute)

If we can make nouveau's SEC2/ACR work (Vector A), then:
1. Boot with nouveau → SEC2/ACR loads FECS/GPCCS
2. GlowPlug captures the warm state
3. Swap to VFIO → dispatch compute through warm channels
4. On state loss, GlowPlug resurrects via nouveau cycle

This leverages both sides: nouveau's knowledge + our sovereign dispatch.

---

## Key Insight (User)

> "Following a delivery driver is a poor way to map the city. We're better off learning
> what open NVIDIA tries to do and ingesting the patterns. We don't need to understand the
> inside of the ACR to utilize it."

The ACR is a **tool**, not an obstacle. It's a function: `f(firmware_blobs) → running_falcons`.
We need to understand its interface (inputs/outputs), not its internals.

**Inputs**: SEC2 LS firmware + ACR HS bootloader + FECS/GPCCS firmware, all placed in a
WPR layout in VRAM.

**Outputs**: Running FECS/GPCCS falcons ready for compute dispatch.

---

## Firmware Inventory

All files present at `/lib/firmware/nvidia/gv100/`:

| File | Size | Purpose |
|------|------|---------|
| sec2/desc.bin | 656 B | SEC2 load descriptor (build: Feb 13 2018) |
| sec2/image.bin | 91,136 B | SEC2 combined code+data |
| sec2/sig.bin | 192 B | SEC2 signature for HS verification |
| acr/bl.bin | 1,280 B | ACR HS bootloader (symlink to gp102) |
| acr/ucode_load.bin | 18,688 B | ACR main firmware |
| gr/fecs_inst.bin | 25,632 B | FECS instruction memory |
| gr/fecs_data.bin | 4,788 B | FECS data memory |
| gr/fecs_sig.bin | 192 B | FECS signature |
| gr/gpccs_inst.bin | 12,643 B | GPCCS instruction memory |
| gr/gpccs_data.bin | 2,128 B | GPCCS data memory |
| gr/gpccs_sig.bin | 192 B | GPCCS signature |

---

## Next Steps

1. **Enable nouveau debug logging** (requires reboot with `nouveau.debug=trace`)
   to capture the exact failure point in SEC2/ACR init

2. **Implement Vector B** (DMA-based SEC2 loader) in Rust as part of coral-driver,
   giving us a sovereign path independent of kernel driver quirks

3. **Test Vector A** by checking if the PRIVRING fault at 0x600 is the root cause
   and whether skipping it allows SEC2/ACR to proceed

4. **Cross-validate** on the second Titan V to confirm reproducibility
