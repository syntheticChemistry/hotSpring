# Exp 097: EMEM Discovery + MSI Wake

**Date:** 2026-03-26
**Status:** COMPLETE — major discovery. MSI/IRQ findings moot for Volta (SEC2 is one-shot loader per Exp 098).
**Depends on:** Exp 096 (unified diagnostics), Exp 095 (HS mode)
**Unlocks:** Path O (Exp 098), then consolidated by Exp 110

## Objective

After Exp 096 revealed DMEM is locked in HS mode, explore the EMEM back-channel
and MSI interrupt path to establish SEC2 conversation.

## Results

### MAJOR DISCOVERY: Init Message Found in EMEM

The SEC2 init message — the exact structure that `Sec2Queues::discover()` scans
DMEM for — exists in EMEM at offset **0x0080**:

```
EMEM@0x0080: 00042001 026c0200 01000000 00000080 01000080 01000080 01000100 a5a51f00

Decoded:
  w0 = 0x00042001 → unit_id=0x01 size=32 ctrl=0x04 seq=0x00
  w1 = 0x026c0200 → msg_type=0x00 num_queues=2 os_debug_entry=0x026c
  Queue 0 (CMDQ, id=0): offset=0x01000000 size=128 bytes
  Queue 1 (MSGQ, id=1): offset=0x01000080 size=128 bytes
```

This perfectly matches the `nv_sec2_init_msg` format: unit_id=0x01, size∈24..48,
msg_type=0x00, num_queues=2. The queue offsets (0x01000000, 0x01000080) are
**falcon virtual addresses**, not raw DMEM offsets. On GP102+ falcons, DMEM is
mapped at VA 0x01000000, meaning:

- **CMDQ ring** = DMEM offset 0x0000, size 128 bytes
- **MSGQ ring** = DMEM offset 0x0080, size 128 bytes

### EMEM Memory Map (after HS boot)

| Region | Content | Interpretation |
|--------|---------|----------------|
| 0x000 | 0x00230406 | ACR status/version word |
| 0x080-0x0A0 | Init message (see above) | Queue layout, os_debug=0x026c |
| 0x200 | 0x000010DE | NVIDIA PCI vendor ID |
| 0x208 | 0x00000500 | HS boot loop address |
| 0x300 | 0x000000FD | BOOTVEC value (0xFD00 >> 8) |
| 0x400-0x5FF | BL code (0x004000d0...) | The bootloader we uploaded |
| 0x700-0x1FFF | Dense binary data | Signed/encrypted firmware or hash tables |
| 0x2000+ | 0xDEAD5ED0 (lockdown) | EMEM lockdown sentinel (distinct from DMEM's 0xDEAD5EC2) |

Total EMEM: ~8KB live data + ~6KB lockdown region. 3,760 non-zero words.

### MSI IRQ: Zero Fires

| Action | MSI Fired | IRQSTAT | Notes |
|--------|-----------|---------|-------|
| Initial arm | no | 0x10 | No pending IRQs |
| Spontaneous (100ms wait) | no | 0x10 | SEC2 generates no interrupts |
| IRQSSET=0x40 poke | no | 0x10 | IRQSTAT unchanged — host can't set IRQ bits |
| IRQMSET=0x40 (enable mask) | no | mask=0x00 | IRQMASK unchanged — host can't modify mask |
| After mask+poke | no | 0x10 | Still nothing |
| After CMDQ write+poke | no | 0x10 | CMDQ head writable but no response |

**Conclusion:** In HS mode, the host cannot control SEC2's IRQ subsystem.
IRQSSET and IRQMSET writes are silently dropped. The firmware has locked
IRQ configuration.

### STARTCPU: No Effect

| Method | cpuctl Before | cpuctl After | PC |
|--------|--------------|-------------|-----|
| CPUCTL write (STARTCPU bit) | 0x10 | 0x10 | 0x4bfd (unchanged) |
| CPUCTL_ALIAS write | 0x10 | 0x10 | 0x4bfd (unchanged) |

SEC2 cpuctl=0x10 means HRESET bit set. The HS firmware completed its ACR job
and entered HRESET. STARTCPU has no effect — the ACR security engine prevents
the host from restarting the falcon in HS mode.

### CMDQ Head Register: Writable But Unresponsive

CMDQ head register at SEC2_BASE+0xA00 IS writable (wrote 0x10, readback=0x10).
But with SEC2 in HRESET, advancing the head pointer + IRQSSET poke produces
no response. The ring buffer mechanism requires SEC2 to be RUNNING.

## Root Cause Analysis

The `blob_size=0` optimization causes the ACR firmware to complete its WPR
processing, then exit/halt without entering the CMDQ idle loop. The normal
ACR flow is:

```
1. Enter HS mode
2. DMA the ACR blob (firmware code)
3. Process WPR headers, bootstrap FECS/GPCCS
4. Initialize CMDQ/MSGQ in DMEM
5. Write init message to DMEM
6. Enter idle loop (RUNNING, polling CMDQ)
```

With `blob_size=0`, step 2 is skipped. The firmware sees "no blob to DMA"
and may short-circuit to an early exit, skipping steps 3-6. The init message
at EMEM offset 0x80 was likely written BEFORE the firmware checked blob_size.

**Critical insight:** In Exp 095 without blob_size=0, SEC2 entered HS but
trapped at EXCI=0x201F0000. That trap happened during step 2 (blob DMA).
The firmware may have already completed steps 3-5 BEFORE the trap. If we
catch the state at the right moment, DMEM might contain the initialized
queues and SEC2 might still be RUNNING.

## Next Steps → Completed via Exp 098, 110

~~Path O tested in Exp 098: full init achieves HS but DMA traps during WPR→falcon copy.~~
~~Path P subsumed: blob is pre-populated but falcon DMA target is the issue.~~

Exp 110 consolidated the full variable space. The HS+MMU paradox (legacy PDEs → HS but
broken DMA; correct PDEs → working DMA but no HS) is the sole remaining gate.
See `experiments/110_CONSOLIDATION_MATRIX.md` → Next Steps (VRAM-native page tables).

## Files Changed

- `coralReef/crates/coral-driver/tests/hw_nv_vfio/sec2_emem_discovery.rs` — new test
- `coralReef/crates/coral-driver/tests/hw_nv_vfio.rs` — registered new module
