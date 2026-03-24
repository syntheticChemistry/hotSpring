# NVIDIA Closed-Source Volta UVM — Reverse Engineering Guide

**Date:** March 18, 2026
**Source:** `nvidia-kernel-source-580` (580.126.18, closed-source variant)
**License note:** All files in this directory are MIT-licensed by NVIDIA

---

## What This Is

The closed-source nvidia kernel module package includes the full UVM (Unified
Virtual Memory) driver source code. This is the **kernel driver that manages
Volta compute dispatch** via RM ioctls — the exact protocol that
`NvUvmComputeDevice` in coral-driver needs to replicate.

The closed-source driver (unlike the open 580.x variant) **supports Volta/GV100**
natively. Its observable actions — register values, memory layouts, ioctl
sequences — are what we need to reverse-engineer.

---

## Key Files and What They Reveal

### Volta GPU Init (`uvm_volta.c`)
- VA space layout: `rm_va_base = 0`, `rm_va_size = 128 TB`
- Max channel VA: `1 << 40` (1 TB) — **not 49-bit for channel buffers**
- GPFIFO can be placed in vidmem
- Page table is Pascal-compatible with NO_ATS extension
- GPC has up to 8 TPCs, each with 1 LTP uTLB

### Volta Host Ops (`uvm_volta_host.c`) — GPFIFO Protocol
- `write_gpu_put`: BAR1 dummy read → write GPPut → wmb → write workSubmissionToken
- Uses class `C36F` (VOLTA_CHANNEL_GPFIFO_A = 0xC36F)
- Semaphore release via `SEM_ADDR_LO/HI` + `SEM_PAYLOAD` + `SEM_EXECUTE`
- TLB invalidate via `MEM_OP_A/B/C/D` (4-word method, targeted)
- Fault replay via `MEM_OP_A/B/C/D` with `TLB_INVALIDATE_REPLAY_START`

### Volta CE Ops (`uvm_volta_ce.c`) — DMA Engine
- Uses class `C3B5` (VOLTA_DMA_COPY_A)
- Semaphore, memcopy, memset operations
- Flush control: `FLUSH_ENABLE` + `FLUSH_TYPE` (GL for GPU membar, SYS for sys)
- PLC (Page Level Cache) mode from HAL

### Volta MMU (`uvm_volta_mmu.c`) — Page Table Format
- Inherits Pascal MMU mode HAL, overrides `make_pte` and `make_pde`
- PTE format (8 bytes): valid[0] | aperture[2:1] | vol[3] | encrypted[4] |
  privilege[5] | read_only[6] | atomic_disable[7] | address[53:8] | kind[63:56]
- PDE: single (depth != 3) or dual (depth == 3, big+small halves)
- `NO_ATS` bit at depth 2 on ATS systems
- Address shift: 12 (4K page granularity)
- COMPTAGLINE field overloaded for 47-bit physical address on NVSwitch

### Channel Class (`clc36f.h`) — GPFIFO Entry Format
- GP entry: 8 bytes. Entry0: fetch[0] | get_addr[31:2]. Entry1: get_hi[7:0] |
  priv[8] | level[9] | length[30:10] | sync[31]
- DMA method format: address[11:0] | subchannel[15:13] | count[28:16] | opcode[31:29]
- Opcode 1 = INC_METHOD, 3 = NON_INC, 4 = IMMD, 5 = ONE_INC, 7 = END_PB_SEGMENT
- Semaphore operations: acquire, release, acq_geq, reduction
- Control structure: GPPut at offset 0x8C, GPGet at 0x88, Put at 0x40, Get at 0x44

### HAL Registration (`uvm_hal.c`) — Function Pointer Tables
- Volta CE HAL inherits from `PASCAL_DMA_COPY_B`
- Volta Host HAL inherits from `PASCAL_CHANNEL_GPFIFO_A`
- Volta Arch HAL inherits from `GP100`
- Volta Fault Buffer HAL inherits from `GP100`

### Fault Buffer (`uvm_volta_fault_buffer.c`)
- Fault entry class: C369 (VOLTA_FAULT_BUFFER_A)
- Engine IDs: GRAPHICS=64, CE0-CE8=15-23, HOST0-HOST13=32-45
- Fault types: PDE(0), PTE(2), RO_VIOLATION(6), ATOMIC(0xF), etc.
- Client types: GPC (TPCs/PEs/GPCCS) or HUB (CE/HOST/FECS/SKED)

---

## Critical Constants for coral-driver

```
VOLTA_CHANNEL_GPFIFO_A = 0xC36F
VOLTA_DMA_COPY_A       = 0xC3B5
VOLTA_FAULT_BUFFER_A   = 0xC369

Max channel VA          = 1 << 40  (1 TB)
Max host VA             = 1 << 40
RM VA size              = 128 TB
Address shift           = 12

GPPut register offset   = 0x8C (in USERD)
GPGet register offset   = 0x88
Put register offset     = 0x40
Get register offset     = 0x44
```

---

## GPPut Write Protocol (Critical for VFIO Path)

From `uvm_volta_host_write_gpu_put`:

```c
// 1. BAR1 read to flush pending CPU writes
if (dummyBar1Mapping)
    READ_ONCE(*dummyBar1Mapping);

// 2. Write new GPPut value
WRITE_ONCE(*gpPut, gpu_put);

// 3. Write memory barrier
wmb();

// 4. Write doorbell token to trigger GPU processing
WRITE_ONCE(*workSubmissionOffset, workSubmissionToken);
```

This 4-step sequence is the **exact doorbell protocol** for submitting work
to the Volta GPFIFO. For the VFIO path, steps 1-3 go through BAR0/BAR1 MMIO
and step 4 is the doorbell write.

---

## Page Table Entry Encoding for VFIO Path

From `make_pte_volta` — the exact bit layout we need for sovereign MMU:

```
Bit  0     : VALID (1 = valid)
Bits 2:1   : APERTURE (0=VID, 1=PEER, 2=SYS_COH, 3=SYS_NCOH)
Bit  3     : VOLATILE (1 = uncached)
Bit  4     : ENCRYPTED (always 0 for us)
Bit  5     : PRIVILEGE (always 0 for us)
Bit  6     : READ_ONLY
Bit  7     : ATOMIC_DISABLE
Bits 53:8  : ADDRESS_SYS (for sysmem, address >> 12)
Bits 32:8  : ADDRESS_VID (for vidmem, address >> 12, lower 25 bits)
Bits 35:33 : PEER_ID (for peer memory)
Bits 63:56 : KIND (0x00 = PITCH for compute)
```

This resolves the `0xbad00200` MMU fault from the sovereign path experiments.

---

## Cross-Reference: Blackwell (GB100/GB206)

Also included are Blackwell files for comparison. Key differences:
- Different TLB invalidation (uses new doorbell mechanism)
- MMU format changes (VER3 page tables)
- Additional ATS capabilities
- Larger physical address space

---

## Usage for Post-Reboot Work

1. **VFIO sovereign path**: Use PTE encoding from `uvm_volta_mmu.c` to fix the
   MMU translation that's currently returning `0xbad00200`
2. **GPPut protocol**: Implement the 4-step doorbell sequence in `NvVfioComputeDevice`
3. **Channel class IDs**: Confirmed `0xC36F` for GPFIFO, validates existing coral-driver constants
4. **TLB invalidation**: Study the `MEM_OP_A/B/C/D` sequence for post-mapping TLB flush
5. **RTX 5060 RM_MAP_MEMORY debug**: Compare Blackwell HAL registration to understand
   what changed in the memory mapping protocol
