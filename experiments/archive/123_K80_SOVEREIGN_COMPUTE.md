# Experiment 123: K80 Sovereign Compute Pipeline

**Date**: 2026-03-25 (design) / 2026-03-26 (execution — hardware arriving)
**GPU**: Tesla K80 (2× GK210, PCI 10de:102d, Kepler, SM 3.7)
**Hypothesis**: Kepler's lack of firmware security enables end-to-end sovereign compute validation

## Strategic Value

The K80 eliminates the **entire security layer** (FWSEC, WPR2, ACR, signed firmware) that blocks Volta.
If we can dispatch compute on K80, we validate layers L1-L9 and L11 of the sovereign stack.
The ONLY remaining problem for Titan V would then be L10 (authenticated falcon boot).

## Architecture Delta: GK210 vs GV100

| Area | GK210 (Kepler) | GV100 (Volta) |
|------|----------------|---------------|
| FECS/GPCCS boot | Direct PIO upload | ACR → WPR2 → signed boot |
| Compute class | 0xA1C0 (KEPLER_COMPUTE_B) | 0xC3C0 (VOLTA_COMPUTE_A) |
| DMA address space | 40-bit | 47-bit |
| Page tables | GF100-style PDE/PTE (2-level) | 5-level (GP100+ lineage) |
| Channel bind | 0x800000 + chid×8 | Expanded USERD + doorbells |
| Channel class | 0xA16F (KEPLER_CHANNEL_GPFIFO_B) | 0xC36F (VOLTA_CHANNEL_GPFIFO_A) |
| RAMFC signature | 0x0000face | Same, extended fields |
| USERD GPPut | offset 0x8c | offset 0x90 (larger USERD) |
| Firmware source | Built into nouveau (fuc3 compiled) or /lib/firmware/nouveau/ | /lib/firmware/nvidia/gv100/ |

## Sub-Experiments

### 123-K0: Identity Probe (Day 1 arrival)
**Goal**: Read BOOT0 and confirm GK210 chipset ID (envytools has "?" for GK210).
- Read BAR0+0x0000 (PMC.ID) and BAR0+0x0A00 (PMC.NEW_ID)
- Record exact chipset byte for BOOT0→SM mapping
- Read PRI_RING status, MC_ENABLE
- Verify dual-GPU: should see two PCI devices 10de:102d
- **IMPORTANT**: K80 needs external power (2× 8-pin) and may need active cooling

### 123-K1: FECS/GPCCS PIO Boot
**Goal**: Load falcon firmware via direct IMEM/DMEM upload — no ACR.
**Prerequisite**: 123-K0 confirms GK210 identity.

#### Falcon PIO Upload Protocol (v1)
FECS base: 0x409000, GPCCS base: 0x41a000. Offsets from base, port 0:

| Register | Offset | Purpose |
|----------|--------|---------|
| IMEM_CTRL | 0x180 | Write: start_addr | BIT(24) [| BIT(28) if secure] |
| IMEM_DATA | 0x184 | Stream u32 words of instruction memory |
| IMEM_TAG | 0x188 | Write tag every 64 words (tag increments) |
| DMEM_CTRL | 0x1C0 | Write: start_addr | BIT(24) |
| DMEM_DATA | 0x1C4 | Stream u32 words of data memory |
| START_0 | 0x100 | Read first; if bit6 clear, write 0x2 to start |
| START_1 | 0x130 | If bit6 of 0x100 was set, write 0x2 here instead |

#### Upload Sequence
1. **DMEM first**: set cursor (0x1C0), stream data words (0x1C4)
2. **IMEM second**: set cursor (0x180), stream code words (0x184), write tag to 0x188 every 64 words. Pad to multiple of 64 words with zeros.
3. **Start**: read 0x100, branch on bit6, write 0x2 to start register

#### Firmware Source
- Nouveau compiles fuc3 source into hubgk110.fuc3.h / gpcgk110.fuc3.h
- External: /lib/firmware/nouveau/nv[chipset]_fuc409c (FECS code), fuc409d (FECS data), fuc41ac (GPCCS code), fuc41ad (GPCCS data)
- For sovereign compute: extract from nouveau module or compile from fuc3 source

#### Boot Handshake
After starting GPCCS then FECS:
- Clear 0x409800, 0x41a10c, 0x40910c
- Poll 0x409800 until bit0 set (up to 2 seconds)
- Then FECS mailbox: write 0x409500/0x409504, read 0x409800 for status

### 123-K2: Compute Channel Setup
**Goal**: Set up PFIFO channel with GPFIFO for compute dispatch.

#### PFIFO Init
- Enable PFIFO: 0x2200 bit0
- Program PBDMA
- Set USERD BAR1 base: 0x2254

#### Channel Instance (RAMFC — gk104 layout)
| Offset | Value | Purpose |
|--------|-------|---------|
| 0x08-0x0C | USERD VA | Pointer to USERD block |
| 0x10 | 0x0000face | PBDMA validation signature |
| 0x30 | 0xfffff902 | Fixed |
| 0x48-0x4C | GPFIFO base+limit | Ring buffer location + ilog2(len/8) |
| 0x84 | 0x20400000 | |
| 0x94 | 0x30000000 | devm | Engine binding |
| 0x9c | 0x100 | |
| 0xac | 0x0000001f | |
| 0xB8 | 0xf8000000 | |
| 0xE4 | 0x20 (if priv) | |
| 0xE8 | chan_id | |

#### Channel Bind
- Write 0x800000 + chid×8: 0x80000000 | (inst_addr >> 12)

#### MMU (GF100-style)
- PDE: 64-bit, LPT_PRESENT + address fields
- PTE: 64-bit, PRESENT + TARGET (VRAM/sysmem) + ADDRESS
- 40-bit VA space, 4KB small pages
- Simple identity map for initial testing

### 123-K3: Compute Dispatch
**Goal**: Execute a trivial shader kernel.

#### Object Binding
- Subchannel 0: SET_OBJECT with engine=PGRAPH(0), class=0xA1C0

#### Dispatch Methods (from gk104_compute.xml)
- CODE_ADDRESS_HIGH/LOW: 0x1608/0x160C — shader code location
- LAUNCH_DESC_ADDRESS: 0x02B4 — launch descriptor (>>8)
- LAUNCH: 0x02BC — trigger dispatch

#### Launch Descriptor (GK104_COMPUTE_LAUNCH_DESC)
- PROG_START, grid dimensions, block dimensions
- CB configs, local/shared memory, GPR count

## Risk Assessment
- **Low risk**: No security barriers, well-documented in nouveau
- **Medium risk**: Need to extract/compile FECS/GPCCS firmware blobs
- **Medium risk**: K80 cooling — needs active airflow, not passive
- **Low risk**: Dual-GPU should appear as two independent PCI devices

## Success Criteria
- 123-K0: BOOT0 read, chipset ID confirmed → update identity.rs
- 123-K1: FECS/GPCCS running, mailbox handshake complete
- 123-K2: PFIFO channel bound, no PBDMA errors
- 123-K3: Shader executes, results readable from GPU memory

## Impact on Titan V
If K80 sovereign compute works:
1. Validated PFIFO/PBDMA/GR/compute dispatch code (reusable structure)
2. Only ACR/WPR2/FWSEC remains for Volta — attack surface is precisely bounded
3. Can test parasitic compute strategies (sysfs BAR0 while nouveau active) on BOTH GPUs
