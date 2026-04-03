# Experiment 137: SEC2 DMA Context Reconstruction

**Date**: 2026-04-02
**GPU**: Titan V (GV100, 0000:03:00.0)
**Status**: CRITICAL DISCOVERIES — root cause of BOOTSTRAP_FALCON failure confirmed, SEC2 communication protocol identified
**Predecessor**: Exp 136 (Dual GPU Sovereign Boot Iteration)

## Executive Summary

This experiment investigated **why Strategy 8's BOOTSTRAP_FALCON command fails** to load
firmware into FECS/GPCCS. Three critical discoveries change the entire approach:

1. **BIND_INST bit 0 = 0 (NOT VALID)** — SEC2's DMA engine is completely disabled
   because the instance block binding was invalidated by nouveau teardown.
2. **FECS/GPCCS IMEM already has firmware** — contrary to Exp 136's GPCCS IMEM empty
   report, both falcons have valid firmware loaded by nouveau's ACR before the VFIO swap.
3. **SEC2 uses CMDQ/MSGQ protocol, not mailbox** — writing to MB0/MB1 is ignored;
   SEC2's ACR firmware processes commands through DMEM-resident ring buffers.

The actual blocker is: **HS mode hardware lockdown prevents host-initiated STARTCPU**,
and SEC2 (the only entity that can start FECS/GPCCS) has no initialized command queue.

## Starting State (post-Exp 136)

```
PMC_ENABLE:  0x5fecdff1  (all engines up)
FECS_CPUCTL: 0x00000012  (HALTED + STARTCPU pending, HS mode)
GPCCS_CPUCTL: 0x00000012  (same)
SEC2_CPUCTL: 0x00000000  (RUNNING, HS mode, PC at 0x00000000)
SEC2_SCTL:   0x00003000  (HS mode active)
VRAM: ALIVE (HBM2 write-readback OK)
```

## Discovery 1: SEC2 Instance Block Destroyed

### Root Cause

SEC2's BIND_INST register at `0x87090`:

```
BIND_INST = 0x00010040
  bits 31:12 = 0x00010 → VRAM address 0x10000 (64KB)
  bit 6      = 1       → BIND_ALLOW_SYSMEM
  bits 5:4   = 00      → TARGET = VID_MEM
  bit 0      = 0       → BIND_ENABLE = FALSE ← ROOT CAUSE
```

**bit 0 = 0 means SEC2's DMA engine has NO valid binding.** All DMA operations
(including BOOTSTRAP_FALCON's firmware fetch from WPR) silently fail.

### Verification

PRAMIN window at VRAM 0x10000 (the instance block address):

```
VRAM[0x10000..0x1004C] = ALL ZEROS
```

Nouveau completely zeroed the instance block memory during teardown. The page directory
base, page tables, and all engine context pointers are gone.

## Discovery 2: FECS/GPCCS IMEM Has Firmware

Despite Exp 136 reporting "GPCCS IMEM[0x3400] is empty", the IMEM actually contains
valid firmware loaded by nouveau's ACR before the VFIO swap:

```
FECS  IMEM[0x0000] = 0x002000d0  (valid falcon instruction)
FECS  IMEM[0x3400] = 0xd0b405fc  (bootloader region)
GPCCS IMEM[0x0000] = 0x001400d0  (valid falcon instruction)
GPCCS IMEM[0x3400] = 0x000400d0  (bootloader region)
```

BOOTVEC values (set by nouveau's ACR):
```
FECS_BOOTVEC:  0x00007E00  (BL entry point)
GPCCS_BOOTVEC: 0x00003400  (BL entry point)
```

The firmware IS loaded. The problem is starting execution, not loading.

## Discovery 3: HS Mode Hardware Lockdown

### STARTCPU Blocked

Writing STARTCPU (bit 1) to FECS_CPUCTL always results in:
```
cpuctl = 0x00000012  (HALTED + STARTCPU set, falcon never starts)
PC = 0x00000000  (never advances)
EXCI = 0x00000002  (stale ILLEGAL_INSN from prior attempt)
```

Tested via both:
- `FECS_CPUCTL` (0x409100) — blocked
- `FECS_CPUCTL_ALIAS` (0x409130) — also blocked

### SCTL Cannot Be Cleared

```
FECS_SCTL = 0x00003000  (HS mode)
write 0x00000000 → read back 0x00003000  (HARDWARE LOCKED)
```

HS mode is enforced by hardware fuses. Once set by ACR, only a full engine
reset (which also destroys IMEM firmware) can clear it.

**Conclusion**: On GV100, ONLY the ACR running on SEC2 can start FECS/GPCCS.

## Discovery 4: SEC2 Communication Protocol

### Mailbox (MB0/MB1) — Does Not Work

Following the `mailbox_command.rs` protocol (Strategy 8):
```
MB1 ← 0x0000000C  (falcon bitmask: FECS|GPCCS)
MB0 ← 0x00000001  (ACR_CMD_BOOTSTRAP_FALCON)
```

Result: MB0 stays at 0x00000001 for 1+ seconds. **SEC2 does not poll the mailbox
registers for commands.** The GV100 ACR firmware uses the CMDQ/MSGQ ring buffer
protocol instead.

### CMDQ/MSGQ — Queues Not Initialized

```
CMDQ0_HEAD (0x87A00) = 0x00000000
CMDQ0_TAIL (0x87A04) = 0x00000000
MSGQ0_HEAD (0x87A30) = 0x00000000
MSGQ0_TAIL (0x87A34) = 0x00000000
```

The init message (unit_id=0x01, msg_type=0x00, num_queues=2) was NOT found
in a full 4KB DMEM scan. SEC2's ACR firmware either:
- (a) sent the init message, nouveau consumed it, and queues were advanced to
  a state that looks "reset" after teardown, OR
- (b) nouveau's operation left the queues consumed and head==tail (empty)

### SEC2 DMEM Contents

Only 15 non-zero words in first 4KB. The ACR state data:

| Offset | Value      | Interpretation                    |
|--------|------------|-----------------------------------|
| 0x020  | 0x00000001 | Status/result (success)           |
| 0x024  | 0x00180000 | ACR firmware base in VRAM (1.5MB) |
| 0x030  | 0x00000100 | Non-secure code offset            |
| 0x034  | 0x00000100 | Non-secure data offset            |
| 0x038  | 0x00002e00 | Secure code size (11.5KB)         |
| 0x040  | 0x00182f00 | Data section VRAM address         |
| 0x048  | 0x00001000 | Data size (4KB)                   |
| 0xB00  | e931c2b6.. | Crypto data (ACR signature)       |
| 0xD20  | 17a2a953.. | Crypto data (ACR signature)       |

## Attempted Fix: BIND_INST Reconstruction

### Approach

1. Wrote minimal instance block (zeros, valid for physical mode) to VRAM 0x20000 via PRAMIN
2. Set BIND_INST = 0x00020041 (VRAM 0x20000, vidmem target, BIND_ENABLE=1)
3. Verified SEC2 still running after rebind
4. Re-issued BOOTSTRAP_FALCON via mailbox

### Result

SEC2 survived the BIND_INST change but **did not process the mailbox command**.
This confirmed Discovery 4: mailbox is not the right protocol.

The instance block fix was necessary but not sufficient — the CMDQ must also
be reconstructed for SEC2 to receive commands.

## Device State Degradation

Attempted a fresh `warm-fecs` cycle to reset state, but the VFIO unbind hung:

```
error: sysfs write .../driver/unbind: timed out after 10s (child likely in D-state)
```

This caused cascading failures:
- coral-ember spawned D-state children (PIDs 162863, 198112)
- coral-glowplug daemon (PID 1776) crashed
- Restarted glowplug (PID 202165) also entered D-state
- Device left unbound, partially inaccessible
- **System reboot required** to recover

## Architecture of the Solution

The Exp 136-137 investigation chain has fully mapped the problem:

```
                    ┌─────────────────────────┐
                    │   FECS/GPCCS in HS Mode │
                    │   (firmware loaded,     │
                    │    STARTCPU blocked)    │
                    └──────────┬──────────────┘
                               │
                    Only SEC2 can start them
                               │
                    ┌──────────▼──────────────┐
                    │   SEC2 running (HS mode) │
                    │   idle loop at PC=0      │
                    │   NO command queue init  │
                    └──────────┬──────────────┘
                               │
                    Needs CMDQ + IRQ poke
                               │
                    ┌──────────▼──────────────┐
                    │   CMDQ/MSGQ not set up  │
                    │   (init_msg consumed,   │
                    │    queue regs all zero)  │
                    └──────────┬──────────────┘
                               │
                    Need queue offsets from init_msg
                               │
                    ┌──────────▼──────────────┐
                    │   Must capture init_msg │
                    │   DURING nouveau phase  │
                    │   (before teardown)      │
                    └─────────────────────────┘
```

## Next Steps (post-reboot)

### Priority 1: Capture SEC2 DMEM During Nouveau Phase

Add code to `warm_handoff.rs` (or a new `coralctl` subcommand) that:

1. **Before nouveau teardown**: dump SEC2 DMEM (full 64KB)
2. **Capture queue registers**: CMDQ0_HEAD/TAIL, MSGQ0_HEAD/TAIL
3. **Capture BIND_INST**: save the valid instance block address and contents
4. **Capture WPR boundaries**: read PMU WPR registers while they're alive
5. **Run `oracle capture`**: snapshot full MMU page table chain

This gives us the complete DMA configuration to reconstruct after VFIO swap.

### Priority 2: CMDQ Reconstruction Protocol

After VFIO swap, with captured data:

1. **Reconstruct instance block** in VRAM (write via PRAMIN)
2. **Set BIND_INST** with bit 0 = 1 (valid)
3. **Write BOOTSTRAP_FALCON command** to CMDQ at the captured DMEM offset
4. **Advance CMDQ head register** past the command
5. **Poke SEC2 IRQSSET (0x40)** to wake the firmware
6. **Poll MSGQ** for response
7. If successful: start GPCCS then FECS via STARTCPU

### Priority 3: Alternative — Keep Nouveau FECS Running

Instead of the ACR reconstruction approach, explore keeping FECS running
through the driver swap:

- Nouveau's `warm-fecs` flow puts FECS in STOP_CTXSW (frozen but alive)
- If we can swap to VFIO without resetting FECS, it stays running
- This bypasses the entire ACR re-bootstrap problem
- Requires ember livepatch to also preserve FECS engine state

### Priority 4: Guard Against D-state

The driver swap timeout (D-state) is a recurring failure mode. Add:

- Pre-swap health check of PCI link status
- Timeout with graceful fallback (don't leave device unbound)
- Automatic PCI remove/rescan recovery
- SBR (Secondary Bus Reset) as escalation path before D-state occurs

## Register Map Reference

### SEC2 (base 0x87000)

| Register       | Offset  | Purpose                          |
|---------------|---------|----------------------------------|
| MAILBOX0      | 0x040   | MB0 (legacy command interface)   |
| MAILBOX1      | 0x044   | MB1 (legacy parameter)           |
| ITFEN         | 0x048   | Interface enable                 |
| BIND_INST     | 0x090   | Instance block binding           |
| BIND_STAT     | 0x094   | Binding status                   |
| CPUCTL        | 0x100   | CPU control                      |
| BOOTVEC       | 0x104   | Boot vector (IMEM entry)         |
| HWCFG         | 0x108   | Hardware config (IMEM/DMEM size) |
| PC            | 0x110   | Program counter                  |
| EXCI          | 0x118   | Exception cause                  |
| DMEMC         | 0x1C0   | DMEM control (auto-incr bit 25)  |
| DMEMD         | 0x1C4   | DMEM data                        |
| SCTL          | 0x240   | Secure control (HS mode)         |
| FBIF_TRANSCFG | 0xA24   | FBIF transfer config             |
| CMDQ0_HEAD    | 0xA00   | Command queue head               |
| CMDQ0_TAIL    | 0xA04   | Command queue tail               |
| MSGQ0_HEAD    | 0xA30   | Message queue head               |
| MSGQ0_TAIL    | 0xA34   | Message queue tail               |

### FECS (base 0x409000)

| Register       | Offset  | Purpose                          |
|---------------|---------|----------------------------------|
| CPUCTL        | 0x100   | CPU control (HS-locked)          |
| BOOTVEC       | 0x104   | Boot vector = 0x7E00             |
| PC            | 0x110   | Program counter                  |
| EXCI          | 0x118   | Exception cause                  |
| CPUCTL_ALIAS  | 0x130   | Alternate CPUCTL (also blocked)  |
| SCTL          | 0x240   | Secure control = 0x3000          |

## coralReef Code References

- `sec2_queue.rs` — CMDQ/MSGQ ring buffer protocol (the correct interface)
- `mailbox_command.rs` — Legacy mailbox interface (does NOT work on GV100)
- `sec2_hal.rs` — SEC2 probing and diagnostics
- `warm_handoff.rs` — Warm FECS swap flow (needs DMEM capture addition)
