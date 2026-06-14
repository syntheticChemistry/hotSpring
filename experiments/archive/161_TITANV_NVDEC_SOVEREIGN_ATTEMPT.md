# Experiment 161: Titan V NVDEC Sovereign Falcon Execution Attempt

## Date: 2026-04-07

## Hypothesis
With NVDEC IMEM confirmed writable (Exp160), upload nouveau's `nvdec/scrubber.bin`
firmware and execute it on the NVDEC Falcon to prove sovereign firmware execution
on Volta GV100.

## Method
1. Inside reagent VM (nvidia-535 previously loaded then `rmmod nvidia`)
2. Enable NVDEC via PMC_ENABLE bit 15 (set 0x5FECDFF1)
3. Upload scrubber firmware to NVDEC IMEM via IMEMC0/IMEMD0
4. Configure security registers to match nvidia-535's sequence
5. Set BOOTVEC=0, enable DMA, start via CPUCTL=0x02

## Results

### Finding 1: IMEM Upload Confirmed (20/20 words verified)

Discovered the NVDEC Falcon v5 IMEMC0 addressing scheme:
- **Writes**: IMEMC0=0x01000000 enables write mode. Each IMEMD0 write auto-increments
  the address pointer by 8 (confirmed via IMEMC0 readback: 0x01000000 → 0x01000008 → 0x01000010).
- **Reads**: No auto-increment. Must set IMEMC0 to `word_index * 8` for each read.
- Firmware binary (`nvdec/scrubber.bin`) has a 512-byte header; actual code starts at
  offset 0x200. nvidia loads 896 words (3584 bytes) of code.
- Verified 20/20 firmware words match expected values after upload.

### Finding 2: DMACTL Bit 1 (DMA_ON) is Hardware-Gated

After PMC_ENABLE reset, NVDEC starts with DMACTL=0x07 (bits 0,1,2 all set, DMA ON).
**Any single write to any NVDEC register** immediately drops DMACTL to 0x01 (only bit 0),
killing the DMA_ON bit permanently. It cannot be re-enabled from host MMIO:

```
Write 0x02 → Readback 0x00 (REJECTED)
Write 0x03 → Readback 0x01 (only bit 0 accepted)
```

nvidia-535 successfully writes DMACTL=0x02 in the mmiotrace because those writes
originate from the RM firmware (running on PMU), not from host BAR0 MMIO.

### Finding 3: CPUCTL START is Blocked (HALT Sticky)

Writing CPUCTL=0x02 (START) results in readback 0x12 (HALT + START latched).
The Falcon acknowledges the START request but HALT bit (0x10) remains asserted.
The Falcon never transitions to RUNNING state (CPUCTL=0x00).

### Finding 4: ACCESS Register is Writable but Insufficient

The ACCESS register (+0x048) starts at 0x04 (bit 2 = HS_LOCK) after reset.
Clearing ACCESS to 0x00 succeeds — but does NOT unlock DMACTL or CPUCTL.
ACCESS controls a different layer of the security model.

### Finding 5: PRI Privilege Level is Hardware-Fused

The root cause is the PRI (Privileged Register Interface) access level:

- **PRIV_LEVEL_MASK (0x12006C) = 0x0C** (bits 2,3): Read-only from host MMIO.
  Cannot be modified. nvidia's driver only READS this register, never writes it.
- **Host BAR0 MMIO**: Privilege level 0 or 1 (insufficient for Falcon control)
- **PMU/RM internal PRI**: Privilege level 2+ (can write DMACTL/CPUCTL)
- **PRI_TIMEOUT_SAVE0 = 0x80084101**: Confirms our NVDEC register writes trigger
  PRI timeout faults (0x84xxx = NVDEC address range, bit 31 = valid fault).

The PRI privilege level is set by GPU fuses, not by software. nvidia's kernel module
does not configure this — it inherits the GPU's internal architecture where the RM
firmware has elevated PRI access.

### Finding 6: All Volta Falcons Have Same Lockout

Surveyed all accessible Falcon engines (with nvidia-535 loaded):

| Engine | CPUCTL | DMACTL | ACCESS | START works | DMACTL writable |
|--------|--------|--------|--------|-------------|-----------------|
| NVDEC  | 0x12   | 0x01   | 0x04   | NO          | NO              |
| FECS   | 0x10   | 0x05   | 0x04   | NO          | NO              |
| GPCCS  | 0x10   | 0x07   | 0x04   | NO          | NO              |
| NVENC  | 0x10   | 0x07   | 0x04   | NO          | NO              |
| PMU    | 0x20*  | 0x80   | 0x04   | NO          | NO              |
| SEC2   | BADF   | BADF   | BADF   | N/A         | N/A (power-gated) |

*PMU CPUCTL=0x20 (STOPPED) — ran RM firmware, then stopped on driver unload.
PMU DMEM filled with 0xDEAD5EC2 (dead-sec2 sentinel pattern).

### Finding 7: PMU State After Driver Unload

PMU Falcon ran nvidia's RM firmware during driver init, then entered STOPPED state
(CPUCTL=0x20) on driver unload. nvidia scrubbed PMU DMEM with 0xDEAD5EC2 sentinel.
PMU has the same MMIO lockout as other Falcons — cannot restart from host.

## Root Cause Analysis

**Volta's hardware security model prevents host CPU MMIO from starting any
Falcon engine.** The mechanism:

1. GPU PRI hub enforces register access levels (fuse-configured, immutable)
2. Host BAR0 MMIO has insufficient PRI privilege for Falcon CPUCTL/DMACTL writes
3. Only the GPU's internal RM firmware (on PMU) has elevated PRI privilege
4. RM grants host access during driver init; access revoked on unload
5. This is enforced at the silicon level — no software bypass exists

The mmiotrace captures both host-initiated and RM-initiated BAR0 operations.
nvidia's "successful" DMACTL/CPUCTL writes in the trace originate from the
RM firmware through the internal PRI ring, not from host MMIO.

## Architecture Implications

### Titan V Sovereign Compute Strategy

Direct Falcon control from host MMIO is **not possible** on Volta. Sovereign
compute must use one of:

1. **FIFO Channel Submission**: Allocate PFIFO channels and submit compute
   pushbuffers directly. The shader engines (SMs) execute compute without
   needing Falcon firmware control. This is how CUDA works.

2. **nvidia-535 as Init Infrastructure**: Use the proprietary driver for
   GPU initialization (POST, HBM2, security context), then sovereign
   dispatch through FIFO for actual compute workloads. The sovereign stack
   replaces CUDA runtime, not the kernel module.

3. **FECS BootROM Method Interface**: The method registers at FECS+0x500/504
   are a defined host-to-Falcon command pathway. With firmware staged in VRAM
   and FECS in CPUCTL=0x50 state, the BootROM may accept load commands
   without needing direct DMACTL/CPUCTL writes. Requires further investigation.

### K80 (Kepler) — Full Sovereignty Remains Viable

Kepler GPUs do NOT have PRI privilege gating on Falcon engines. Direct MMIO
Falcon control (IMEM upload + CPUCTL START) should work without restriction.
The K80 path remains the best target for proving full sovereign Falcon firmware.

## Files
- `data/titanv/mmiotrace/nvidia535_init_full.log` — Reference mmiotrace
- VM firmware: `/tmp/gv100_fw/nvdec/scrubber.bin` (in reagent VM)

### Finding 8: FECS BootROM Method Interface Also PRI-Gated

Attempted nvidia's FECS BootROM startup sequence:
1. CPUCTL=0x40 (SRESET) → CPUCTL=0x50 (HALT+SRESET) — **SUCCESS**
2. Pre-BootROM config: IRQDEST, METHOD_MASK, DMACTL=0, UNK_A20
3. IRQMASK=0xFFD2 → **REJECTED** (reads back 0x00)
4. CPUCTL_ALIAS=0x02 (START) → **REJECTED** (reads back 0x00)
5. Method status 0x800 never signals ready (stuck at 0x00)

CPUCTL_ALIAS, IRQMASK, and IRQDEST are all PRI-gated on FECS, same as
CPUCTL and DMACTL on other Falcons. The BootROM pathway requires the
same elevated PRI privilege as direct Falcon control.

### Finding 9: Compute Path (PFIFO/PGRAPH/SM) is Fully Accessible

While Falcon control registers are PRI-locked, the entire SM compute
dispatch path is accessible from host MMIO:

| Subsystem     | Accessibility | Writable | Notes                    |
|---------------|--------------|----------|--------------------------|
| PRAMIN (VRAM) | YES          | YES      | Read/write verified       |
| PFIFO         | YES          | YES      | Timeslice regs writable   |
| PBDMA         | YES          | -        | 4 engines, status=idle    |
| PGRAPH        | YES          | -        | GPC/TPC/SM regs readable  |
| PCCSR         | YES          | -        | Channel control available |
| USERMODE      | YES          | -        | Doorbell mechanism active |
| BAR1          | YES          | -        | Block pointer valid       |

PBDMA[0-3] all show STATUS=0x10011111 (idle, waiting for commands).
FIFO_STATUS=0x01000000. PRAMIN read/write verified with DEADBEEF pattern.

This means sovereign GPU compute is possible through the PFIFO→PBDMA→PGRAPH→SM
pipeline WITHOUT needing Falcon firmware control.

## Volta (GV100) Security Model Summary

```
┌─────────────────────────────────────────────────────┐
│                    GPU Silicon                       │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  PRI Hub (Fuse-configured privilege levels)   │   │
│  │  PRIV_LEVEL_MASK = 0x0C (read-only)          │   │
│  └──────┬───────────────────────┬───────────────┘   │
│         │ Level 0-1 (Host)      │ Level 2-3 (RM)    │
│         ▼                       ▼                    │
│  ┌─────────────┐        ┌─────────────┐            │
│  │ Accessible: │        │ Restricted: │            │
│  │ - PRAMIN    │        │ - CPUCTL    │            │
│  │ - PFIFO     │        │ - CPUCTL_ALIAS           │
│  │ - PBDMA     │        │ - DMACTL    │            │
│  │ - PGRAPH    │        │ - IRQMASK   │            │
│  │ - IMEM r/w  │        │ - IRQDEST   │            │
│  │ - DMEM r/w  │        │             │            │
│  │ - ACCESS    │        │ (All Falcon │            │
│  │ - MAILBOX   │        │  execution  │            │
│  │ - SM regs   │        │  control)   │            │
│  └─────────────┘        └─────────────┘            │
└─────────────────────────────────────────────────────┘
```

The privilege boundary is set by GPU fuses (immutable). nvidia's RM firmware
(running on PMU) has Level 2+ access and manages Falcon engines on behalf
of the host. After driver unload, RM is stopped and the host has no way
to restart it or regain Falcon control.

## Architecture Implications

### Titan V Sovereign Compute: Three Viable Paths

1. **PFIFO Channel Dispatch** (Maximum sovereignty, high effort):
   Set up PFIFO channels, GPU page tables, and pushbuffers via PRAMIN.
   Submit compute work directly to SM shader engines through PBDMA.
   Requires detailed Volta command format knowledge (nouveau/envytools).
   All required registers are host-accessible.

2. **nvidia-535 Init + Sovereign Runtime** (Practical, medium effort):
   Use nvidia kernel module for GPU initialization (POST, HBM2, security).
   Replace CUDA runtime with sovereign dispatch code that uses the
   nvidia-established FIFO channels for compute workloads.

3. **Open Kernel Module Collaboration** (nvidia open-gpu-kernel-modules):
   nvidia-535 source is available. Fork the open kernel module for Falcon
   management, use sovereign code for everything else. Partial sovereignty
   but practical for Volta-class hardware.

### K80 (Kepler): Full Sovereignty Viable
Kepler has no PRI privilege gating. CPUCTL, DMACTL, IMEM writes should
all work from host MMIO. Direct FECS firmware upload + execution is the
path to fully sovereign GPU compute without any nvidia driver dependency.

## Files
- `data/titanv/mmiotrace/nvidia535_init_full.log` — 46MB reference trace
- This experiment's scripts executed via `ssh -p 2222 reagent@localhost`

## Next Steps
1. **K80 power cycle + sovereign Falcon**: Physical power cycle, then
   nvidia-470 VM POST, rmmod, direct FECS IMEM upload + sovereign execution
   (Kepler has no PRI gates — highest probability of success)
2. **Titan V PFIFO channel**: Set up PRAMIN-based channel descriptor,
   GPU page table, and pushbuffer for direct SM compute dispatch
3. **Titan V open-kernel-module**: Evaluate nvidia open-source kernel module
   as Falcon management layer under sovereign compute runtime
