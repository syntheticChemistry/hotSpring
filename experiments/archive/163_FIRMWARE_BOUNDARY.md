# Experiment 163: Driver vs Firmware Boundary — Sovereign GPU Architecture

**Date**: 2026-04-07
**GPU**: NVIDIA Titan V (GV100, 10de:1d81)
**Driver**: nouveau (open source, host-side)
**Depends on**: Exp159 (VM Post-HBM2), Exp162 (Sovereign Compute Pipeline)
**Status**: Architecture validated, hot-handoff proven

## Objective

Delineate the driver/firmware/hardware boundary for sovereign GPU compute.
Establish that falcon firmware (PMU, SEC2, FECS, GPCCS) is the GPU's internal
operating system — to be interfaced with, not replaced. Prove coexistence with
nouveau via hot-handoff channel injection.

## The Three Layers

### Driver (host CPU, what we write)
- BAR0 MMIO reads/writes
- DMA buffer allocation (VFIO/IOMMU)
- Channel structure layout (instance blocks, runlists, GPFIFO)
- **Talking to falcons via mailbox registers** (the firmware interface)
- Loading firmware blobs into falcon IMEM/DMEM
- Interpreting firmware responses

### Firmware (GPU falcon processors — the GPU's "BIOS")
- **PMU**: Controls PRI gates (register access), engine clocks, power management. Without PMU, most registers are gated (`0xBAD00200`).
- **SEC2**: ACR enforcement. Validates signed firmware before loading FECS/GPCCS. The trust root.
- **FECS**: GR engine context scheduler. Orchestrates context switches on PBDMAs.
- **GPCCS**: GPC-level context switching within GR.
- **GSP** (Turing+): Subsumes PMU+SEC2+FECS. The entire driver runtime moves onto the GPU.

### Hardware (silicon, controlled exclusively by firmware)
- PBDMAs, PFIFO scheduler, Copy Engines, GR Compute, GPU MMU, HBM2

## Evidence: Falcon State Discovery

### FalconProbe output (with nouveau running)

```
Firmware Boundary Probe:
  PMC_ENABLE=0x5fecdff1  UNK260=0xbad00200  PRI_GATES=CLOSED
  PMU   @ 0x10a000: WAITING (cpuctl=0x00000020 mbox0=0x00000300 bootvec=0x00000000)
  SEC2  @ 0x087000: WAITING (cpuctl=0x00000060 mbox0=0x00000000 bootvec=0x0000fd00)
  FECS  @ 0x409000: HALTED  (cpuctl=0x00000010 mbox0=0x00000000 bootvec=0x00000000)
  GPCCS @ 0x41a000: HALTED  (cpuctl=0x00000010 mbox0=0x00000000 bootvec=0x00000000)
  dispatch_viable=true
```

### Key observations

| Falcon | CPUCTL | State | Meaning |
|--------|--------|-------|---------|
| PMU | 0x20 (bit 5) | WAITING | Firmware loaded, idle loop at PC=0x3A5D, processing mailbox commands |
| SEC2 | 0x60 (bits 5+6) | WAITING | Actively cycling (CPUCTL oscillates 0x40↔0x60 at ~574 Hz) |
| FECS | 0x10 (bit 4) | HALTED | Normal: sleeps between scheduling events, woken by scheduler |
| GPCCS | 0x10 (bit 4) | HALTED | Normal: sleeps between GPC context switches |

### After nouveau teardown (from earlier experiments)

| Falcon | CPUCTL | State | Meaning |
|--------|--------|-------|---------|
| PMU | 0x10 (bit 4) | HALTED | Dead — firmware executed HALT, ACR prevents restart |
| SEC2 | 0x10 (bit 4) | HALTED | Dead — completed teardown sequence |
| FECS | 0x10 (bit 4) | HALTED | Dead — scheduler non-functional |
| GPCCS | 0x10 (bit 4) | HALTED | Dead |

The critical difference: PMU `WAITING` (0x20) vs `HALTED` (0x10). When PMU is WAITING, it maintains PRI gates, clocks, and services. When HALTED, everything dies.

## Evidence: Hot Handoff Success

### Test: Channel injection alongside nouveau

Created `hot_handoff_nouveau` example that:
1. Maps BAR0 via sysfs `resource0` (no VFIO, coexists with nouveau)
2. Verifies PMU alive via `FalconProbe`
3. Writes channel structures entirely in VRAM (instance block, GPFIFO, USERD, runlist) via PRAMIN
4. Binds PCCSR and enables channel
5. Submits runlist to live scheduler

### Result

```
▶ Phase 4: PCCSR Channel Bind + Enable
  PCCSR after enable: 0x11000001
    ENABLE=true STATUS=1 PBDMA_FAULTED=false ENG_FAULTED=false

▶ Phase 6: Post-submit Diagnostic
  PCCSR[500] = 0x11000001
    ENABLE=true STATUS=1 BUSY=true
    PBDMA_FAULTED=false ENG_FAULTED=false
  PBDMA1: SIG=0x00003ace (loaded our RAMFC SIGNATURE=0xFACE)

═══ Verdict ═══
✓ Channel 500 is ENABLED, no faults — hot handoff SUCCEEDED.
  The scheduler accepted our channel alongside nouveau.
```

**Channel 500**: ENABLED, BUSY, STATUS=1 (PENDING), no faults. PBDMA1 loaded our RAMFC context (SIG=0x3ACE from our 0xFACE signature). The scheduler accepted an externally injected channel.

## Evidence: PMU Mailbox Protocol

### 5-second register trace (nouveau idle)

| Register | Value | Behavior |
|----------|-------|----------|
| PMU_CPUCTL | 0x00000020 | Stable (WAITING) |
| PMU_MBOX0 | 0x00000300 | Stable — initialization-complete status word |
| PMU_MBOX1 | 0x00000000 | Stable |
| PMU_PC | 0x3A5D-0x3A5E | Tight idle loop (2 instructions) |
| PMU_IRQMODE | 0x0000FC24 | Interrupt routing configured |
| PMU_ITFEN | 0x00000004 | DMA interface enabled |
| PMU_DMACTL | 0x00000080 | DMA control active |
| QUEUE[0-3] HEAD/TAIL | 0xBADF5040 | **BAD READ** — queue offsets don't exist on GV100 |
| MSGQ HEAD/TAIL | 0x00000000 | Empty |
| SEC2_CPUCTL | 0x40↔0x60 | **2871 changes in 5s** (~574 Hz oscillation) |

### Protocol conclusions

1. **GV100 PMU uses register-based mailbox**, not queue-based RPC (Turing+).
2. Queue register offsets (0x4A0/0x4B0) return `0xBADF5040` — not implemented on this PMU version.
3. PMU MBOX0=0x300 is the idle handshake value (initialization complete).
4. SEC2 is the only actively cycling falcon (frequent CPUCTL state changes).
5. Communication is event-driven: host writes command → interrupt → PMU responds.

## What We Got Wrong (Previous Approach)

The `vfio/channel/` code tried to **be** the firmware:

- `init_pfifo_engine_with()` — writing PFIFO_ENABLE, SCHED_EN, UNK260 directly
- `PfifoInitConfig` — modeling "how to init the scheduler" (firmware's job)
- Direct PBDMA programming (Phase 8) — trying to bypass the scheduler
- `gv100_warm()` `discovery_only` — acknowledging registers are dead without understanding **why**

These registers are dead because **PMU controls them via PRI gates**. We were attempting to replace the firmware layer. On Kepler (K80) this worked because there was no firmware layer. On Volta, the firmware layer is mandatory and security-enforced (ACR).

## Architecture: Scaling Across GPU Generations

| Era | Firmware Interface | Driver Role |
|-----|-------------------|-------------|
| Kepler (K80) | None — direct register writes | Full hardware control |
| Volta (GV100) | PMU mailbox + SEC2 ACR + FECS scheduling | Firmware interface + channel structures |
| Turing/Ampere | GSP RPC — host driver becomes thin RPC client | RPC message formatting |
| Hopper/Blackwell | GSP with extended offloaded functionality | Even thinner RPC layer |

Learning the Volta firmware interface is the foundation for ALL modern NVIDIA cards.

## ACR Boot Chain Audit

coral-driver already contains substantial infrastructure:

| Module | Purpose | GV100 Status |
|--------|---------|--------------|
| `acr_boot/` | Boot chain strategies (SEC2 load, FECS boot) | Multiple experimental paths |
| `strategy_mailbox/` | PMU/SEC2 mailbox command interface | Implemented but untested E2E |
| `sec2_hal.rs` | SEC2 hardware abstraction + reset | GV100-specific fallbacks present |
| `fecs_boot.rs` | FECS/GPCCS firmware upload + start | Complete, uses `FalconCapabilities` |
| `falcon_capability.rs` | Runtime state probing + PIO layout | Validated on GV100 |
| `devinit/pmu.rs` | VBIOS-driven PMU initialization | Complete for HBM2/memory init |

**Gap**: No orchestrator chains `devinit → SEC2(ACR) → FECS/GPCCS` in the correct order for a cold boot. The hot-handoff sidesteps this entirely by coexisting with nouveau.

## Evidence: NOP Dispatch via DRM (End-to-End)

### Discovery: PFIFO scheduler is firmware-controlled

Direct BAR0 MMIO control of PFIFO/scheduler is blocked:
- `PFIFO_ENABLE = 0x00000000` (PRI-gated)
- `SCHED_EN = 0xbad00200` (PRI-gated)
- Only RL0/RL1 writable from host BAR0; no engine mapped to them
- RL4 (nouveau's active runlist) read-only from BAR0 userspace

Conclusion: The scheduler is exclusively firmware-controlled. NOP dispatch requires going through the DRM kernel interface.

### NOP dispatch via raw C DRM ioctls

```
=== Nouveau NOP Submit (raw ioctls) ===
  Device: /dev/dri/card1
  Chipset: NV140
  VRAM: 12287 MB
  Channel allocated: id=2, pushbuf_domains=6
  GEM buffer: handle=2, size=4096, offset=15000, domain=4
  NOP dispatch SUCCEEDED!
```

### NOP dispatch via pure Rust DRM ioctls

```
=== NVIDIA Nouveau NOP Dispatch (Pure Rust) ===

  Phase 1: Open nouveau render node... OK (/dev/dri/renderD129)
  Phase 2: VM_INIT... OK (new UAPI active)
  Phase 3: Channel alloc (class 0xC3C0)... OK (channel 2)
  Phase 4: Syncobj create... OK (handle 1)
  Phase 5: GEM alloc (4 KiB, GTT)... OK
  Phase 6: VM_BIND at 0x100000000... OK
  Phase 7: mmap + write NOP... OK (8 bytes: SET_OBJECT 0xC3C0)
  Phase 8: Submit... OK
  Phase 9: Sync... OK (syncobj signaled)
  Phase 10: Cleanup... OK

  NOP dispatch SUCCEEDED via pure Rust DRM ioctls.
  Pipeline: open -> vm_init -> channel -> gem -> vm_bind -> mmap -> exec -> sync
  Zero C, zero libc — sovereign Rust GPU control.
```

Pipeline: `VM_INIT → CHANNEL_ALLOC(VOLTA_COMPUTE_A) → SYNCOBJ → GEM_NEW → VM_BIND → mmap → EXEC → SYNCOBJ_WAIT`. New UAPI (kernel 6.6+) confirmed required and working on kernel 6.17 + GV100.

## Evidence: PmuInterface Created

`pmu_interface.rs` encapsulates the register-based mailbox protocol:
- `PmuState` enum: Running / Waiting / Halted / Reset / Inaccessible
- `PmuSnapshot`: captures CPUCTL, MAILBOX0/1, IRQSTAT, PC
- `mailbox_exchange()`: write MBOX0/MBOX1 → raise IRQSSET → poll response
- `poll_mbox0_bits()`: wait for firmware status flags
- `probe_queues()`: detect Turing+ queue-based RPC availability
- 4 unit tests passing

## Paths Forward

### Path A: Hot Handoff + DRM (proven, NOP dispatch complete)
- ✅ BAR0 coexistence with nouveau works
- ✅ Channel injection accepted by scheduler
- ✅ NOP dispatch via DRM ioctls (both C and pure Rust)
- ✅ New UAPI (VM_INIT/VM_BIND/EXEC) operational on kernel 6.17 + GV100
- ✅ `PmuInterface` created for firmware communication
- Next: full compute dispatch (shader upload, QMD, multi-buffer) — infrastructure exists in `NvDevice`
- Limitation: requires nouveau to be running

### Path B: Proper Firmware Boot (sovereign, generalizable)
- Use `acr_boot/` infrastructure after nouveau teardown + vfio-pci bind
- Boot PMU via VBIOS devinit → SEC2 via ACR → FECS/GPCCS
- Independent of any kernel driver
- Requires: validated WPR + BL descriptors, PRI/PRAMIN ordering fix

### Path C: PMU Mailbox Protocol (strategic, foundation built)
- ✅ GV100 PMU uses simple register-based mailbox (MBOX0/MBOX1 + interrupt)
- ✅ `PmuInterface` struct built — encapsulates the transport layer
- Next: map specific PMU command words (engine enable, clock control)
- Scales to any card with compatible PMU firmware

## toadStool Pattern Alignment

| toadStool Concept | GPU Firmware Equivalent |
|-------------------|------------------------|
| `FirmwareInventory::probe()` | `FalconProbe::discover()` — which falcons are alive/halted? |
| `RegisterAccess` trait | `MappedBar` — uniform BAR0 interface |
| `PowerManager::glow_plug()` | `PmuMailbox::enable_engines()` — ask PMU to enable clocks |
| `needs_software_pmu()` | `FalconProbe::pmu_alive()` — is PMU running? |
| `compute_viable()` | `FalconProbe::dispatch_viable()` — can we submit work? |

## Files Modified/Created

| File | Change |
|------|--------|
| `falcon_capability.rs` | Added `FalconState`, `FalconStatus`, `FalconProbe` for firmware boundary probing |
| `bench_mmu_fault_diagnostic.rs` | Replaced ad-hoc Phase 1.5 with structured `FalconProbe` |
| `hot_handoff_nouveau.rs` | **New** — BAR0 coexistence test, channel injection alongside nouveau |
| `pmu_mailbox_trace.rs` | **New** — PMU/SEC2/FECS register polling tracer |
| `nouveau_nop_submit.c` | **New** — C proof of NOP dispatch via raw DRM ioctls |
| `nvidia_nop_dispatch.rs` | **New** — Pure Rust NOP dispatch via `nv::ioctl` wrappers |
| `pmu_interface.rs` | **New** — `PmuInterface` struct: mailbox protocol, state probing, IRQ signaling |
| `drm.rs` | Promoted `MappedRegion` and methods to `pub` (sovereign API surface) |
| `nv/ioctl/mod.rs` | Promoted `gem_mmap_region` to `pub` |
