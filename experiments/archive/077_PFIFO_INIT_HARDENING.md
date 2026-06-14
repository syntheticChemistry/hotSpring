# Experiment 077 — PFIFO Init Hardening: Five Failure Modes

**Date**: 2026-03-23
**Status**: HARDENED
**Hardware**: Titan V #1 (`0000:4a:00.0`, GV100, 12 GB HBM2)
**Stack**: coralReef `coral-driver` → `coral-ember` (SCM_RIGHTS) → iommufd

## Summary

During the PFIFO dispatch debugging sessions (Exp 071–076), five distinct
failure modes were discovered that would silently corrupt GPU state, flood logs
with false diagnostics, or produce misleading experiment results on any new
hardware. This experiment documents each failure mode, its root cause, and the
hardening applied in the "Sovereign Pipeline Debt Burndown" sprint.

---

## Finding 1: SM Mismatch Corrupts GPU State Without Recovery

### Symptom

After running tests with `CORALREEF_VFIO_SM=86` (GA102 default) on the GV100
Titan V, the GPU entered an unrecoverable corrupted state. All subsequent runs
in the same ember session — including those with correct `SM=70` — observed:

- `PFIFO_ENABLE` stuck at 0
- `SCHED_EN` reads `0xbad00200` (PBUS domain timeout)
- PRIV_RING fault `0x00000100` on every register access
- PBDMA completely idle (`pbdma_1_state=0x00000000`)

### Root Cause

`NvVfioComputeDevice::open()` accepted `sm_version` from the caller without
validation against the actual hardware. When SM=86 was passed to a GV100 GPU:

1. `identity::compute_class(86)` returned `0xC6C0` (AMPERE_COMPUTE_A)
2. The driver loaded 1,269 GA102 GR init register writes into GV100 BAR0
3. Writes to non-existent Ampere-specific registers triggered PRIV_RING faults
4. PRIV_RING fault handler was not cleared, blocking all subsequent MMIO

No amount of PFIFO resets or re-init could recover — only a full system reboot
or PCIe Function Level Reset (FLR) restored the GPU.

### Fix

- Added `boot0_to_sm()` in `identity.rs` to decode the BOOT0 register
  (offset 0x0) into the correct SM version
- `open()` and `open_from_fds()` now auto-detect SM when caller passes
  `sm_version=0`, or validate against BOOT0 when a specific SM is provided
- Mismatch returns `DriverError::OpenFailed` — no writes touch the GPU
- All test defaults changed from `SM=86` to `SM=0` (auto-detect)

### BOOT0 Decode Table

| BOOT0 value    | Chipset bits [31:20] | SM  | Chip    |
|----------------|---------------------|-----|---------|
| `0x140000a1`   | `0x140`             | 70  | GV100   |
| `0x164000a1`   | `0x164`             | 75  | TU102   |
| `0x170000a1`   | `0x170`             | 80  | GA100   |
| `0x172000a1`   | `0x172`             | 86  | GA102   |
| `0x192000a1`   | `0x192`             | 89  | AD102   |

---

## Finding 2: PMC Bit 8 vs Bit 1 for PFIFO Reset on GV100

### Symptom

Early experiments used `PMC_ENABLE` bit 1 (the documented "PFIFO" bit on older
Fermi/Kepler GPUs) for engine reset. On GV100, toggling bit 1 had no effect on
PFIFO state — `PFIFO_ENABLE` remained unchanged, and the PBDMA never activated.

### Root Cause

On Volta (GV100), the PFIFO engine is gated by PMC bit 8, not bit 1.
The `nouveau` driver's `gk104_fifo_init()` confirms this: it toggles bit 8
(`NV_PMC_ENABLE_PFIFO`) via the PMC reset function. Bit 1 controls a different
engine on Volta (likely PTIMER or host interface).

### Fix

- `PfifoInitConfig` added explicit `pmc_glow_plug` flag that writes
  `0xFFFFFFFF` to PMC_ENABLE (enabling all domains including bit 8)
- Diagnostic config uses the existing PMC value without toggling
- The register constant `PMC_ENABLE_PFIFO_BIT = 1 << 8` is used in
  warm-glow-plug paths

### Registers

| Register        | Offset     | GV100 PFIFO Bit |
|----------------|-----------|------------------|
| `PMC_ENABLE`    | `0x000200` | Bit 8            |
| `PMC_DEVICE_EN` | `0x000600` | Bit 8 (mirror)   |

---

## Finding 3: PFIFO_ENABLE Reads 0 But Engine Is Functional

### Symptom

After a successful PFIFO init, `PFIFO_ENABLE` (0x2200) consistently reads 0.
The driver's per-checkpoint health warnings treated this as "PFIFO disabled"
and emitted 12+ warnings per channel creation:

```
WARN  PFIFO disabled at pre-channel-create (PFIFO_ENABLE=0x00000000)
WARN  PFIFO disabled at post-bind (PFIFO_ENABLE=0x00000000)
WARN  PFIFO disabled at post-runlist (PFIFO_ENABLE=0x00000000)
...
```

These warnings obscured real issues (like the SM corruption in Finding 1) by
flooding logs with false negatives.

### Root Cause

On GV100, the `PFIFO_ENABLE` register at 0x2200 is not the authoritative PFIFO
status indicator. Volta moved the PFIFO enable/status to the top-level PMC
domain gating (bit 8 of `PMC_ENABLE`). The 0x2200 register appears to be a
legacy Kepler/Maxwell compatibility stub that reads 0 on Volta regardless of
actual PFIFO state.

The engine is provably functional: runlist preempts ACK correctly, runlist
flushes complete, and PBDMA channels can be created and scheduled.

### Fix

- Replaced per-checkpoint `pfifo_ck` warnings with a single post-init
  **liveness probe**: performs a runlist preempt and checks for ACK
- If the preempt ACK succeeds, PFIFO is declared functionally live
- The 0x2200 readback is logged at `debug` level only
- A single warning is emitted only if the liveness probe itself fails

---

## Finding 4: RAMFC GP_PUT=1 Causing PBDMA to Fetch Empty GPFIFO

### Symptom

After channel creation and INST_BIND, the PBDMA immediately attempted to fetch
a GPFIFO entry — before any work had been submitted. The fetch targeted slot 0
of the GPFIFO ring, which was zero-filled (no valid push buffer pointer). This
caused the PBDMA to stall or fault on invalid data.

### Root Cause

The RAMFC (channel context in instance memory) was initialized with `GP_PUT=1`
and `GP_GET=0`. This told the PBDMA "there is 1 GPFIFO entry to fetch" the
moment the channel context was loaded. But at that point, the GPFIFO ring
contained only zeroes — no valid push buffer entry existed.

The PBDMA fetched the zero entry, interpreted it as an invalid push buffer
descriptor, and either:
- Stalled with `GP_FETCH` stuck at the fetch address, or
- Triggered a spurious fault that masked the real problem

### Fix

- RAMFC now initializes with `GP_PUT=0, GP_GET=0` (empty ring)
- Work submission explicitly sets `GP_PUT` after writing valid GPFIFO entries
- The doorbell notification (`NV_USERMODE_NOTIFY_CHANNEL_PENDING`) is only
  rung after `GP_PUT` is advanced

---

## Finding 5: False-Positive MMU Fault from Fault Buffer Enable Bit

### Symptom

The driver reported MMU faults on every dispatch attempt:

```
MMU fault: PDE invalid, fault_va=0x0, access_type=VIRT_READ, engine=GR, aperture=VRAM
```

This appeared to be a fatal page table configuration error. Hours were spent
investigating the page table walk, PDE/PTE encoding, and IOMMU configuration.

### Root Cause

The fault detection logic compared `FAULT_BUF0_PUT` against `FAULT_BUF0_GET`
to determine if a fault had occurred. But `FAULT_BUF0_PUT` was initialized
to `0x8000_0000` (the enable bit), while `FAULT_BUF0_GET` started at 0.

Since `PUT != GET`, the driver concluded a fault was pending and decoded the
fault status register at offset 0. But `fault_status=0x00000000` — there was
no actual fault. All the "decoded" fields (PDE invalid, VA=0x0, GR engine,
VRAM aperture) were simply the default decode of zero:

| Field           | Value | Meaning of 0           |
|----------------|-------|------------------------|
| `fault_type`    | 0     | PDE invalid            |
| `access_type`   | 0     | VIRT_READ              |
| `engine_id`     | 0     | GR (engine 0)          |
| `aperture`      | 0     | maps to VRAM display   |
| `fault_va`      | 0     | virtual address 0x0    |

### Fix

- Fault buffer comparison now masks the enable bit:
  `has_fault = (put & 0x7FFF_FFFF) != (get & 0x7FFF_FFFF)`
- Fault status is only decoded when `has_fault` is true
- Eliminates the false-positive that consumed significant debugging time

---

## Cross-Cutting Hardening

In addition to the five individual fixes, the sprint delivered:

| Change | Purpose |
|--------|---------|
| `PfifoInitConfig` struct | Parameterizes PFIFO init — prevents future divergence between production and diagnostic paths |
| `GpuCapabilities` in experiment context | Diagnostic matrix is now architecture-aware; experiments can declare `requires_sm` |
| `coralctl reset <BDF>` | PCIe FLR via VFIO — recovers from GPU corruption without reboot |
| `VfioHolder::reset()` | Daemon-accessible FLR through the glowplug RPC (`device.reset`) |

## Lessons Learned

1. **Never trust caller-provided hardware parameters** — always validate against
   the actual device. BOOT0 is the ground truth for NVIDIA GPU identity.

2. **Register semantics change across generations** — a register that means
   "PFIFO enable" on Kepler may be a dead stub on Volta. Always verify with
   a functional probe (runlist preempt ACK) rather than a register readback.

3. **Initialize ring buffers as empty** — any ring with `PUT != GET` at creation
   time will be consumed immediately, before the software has filled it.

4. **Mask enable bits before comparing pointers** — hardware often overloads
   the high bit of PUT/GET registers as an enable flag. Raw comparison produces
   false positives.

5. **FLR is essential for iteration speed** — without a reset path, any
   firmware mismatch or init failure requires a full reboot. At 90+ seconds
   per reboot, this is the single largest time sink during hardware debugging.
