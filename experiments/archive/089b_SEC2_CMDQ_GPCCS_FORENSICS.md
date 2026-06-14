# Exp 089b: SEC2 CMDQ Probe + GPCCS LS-Mode Forensics

**Date:** 2026-03-24
**Hardware:** Titan V (GV100), BDF 0000:03:00.0
**Build:** coralReef b2a3e63 (pinned)

## Objective

Determine why GPCCS stays at PC=0x0000 after SEC2 ACR bootstrap attempts.
Investigate SEC2 CMDQ/MSGQ ring protocol as potential root cause.

## Key Findings

### 1. GPCCS CPUCTL is ACR-Locked

GPCCS is in Light Secure (LS) mode (`sctl=0x3000`). Host writes to `CPUCTL`
(0x41A100) are **silently dropped**. Only SEC2 (running in HS mode) can modify
GPCCS CPUCTL.

Verification:
```
Write 0x20 to GPCCS+0x100, read back 0x10 → LOCKED
Write 0xDEAD to GPCCS+0x104 (BOOTVEC), read back 0xDEAD → WRITABLE
Write 0xCAFE to GPCCS+0x040 (MAILBOX0), read back 0xCAFE → WRITABLE
```

### 2. Raw VFIO State (Before Any Init)

Reading BAR0 immediately after VFIO FD receipt, before `apply_gr_bar0_init`:

```
FECS:  cpuctl=0x00000010 (STOPPED) pc=0x0000058f sctl=0x00003000
GPCCS: cpuctl=0x00000012 (STOPPED+STARTCPU) pc=0x00000000 sctl=0x00003000 exci=0x00070000
SEC2:  cpuctl=0x00000000 (RUNNING) pc=0x0000058f mb0=0x00000001 mb1=0x00000003
```

GPCCS `cpuctl=0x12` means ACR already issued STARTCPU (bit 1), but GPCCS faulted.

### 3. GPCCS Has Valid Firmware

```
IMEM[0..32]: 0x001400d0 0x0004fe00 0x957ea4bd 0x02f8002f 0x90f900f8 0xb0f9a0f9 0xd0f9c0f9 0xf0f9e0f9
DMEM[0..32]: 0x20677541 0x32203820 0x00373130 0x303a3032 0x36333a36 0x00000000 0x00000001 0x00027100
```

DMEM starts with ASCII "Aug  8 2017 20:06:36" — firmware build date. Code is present.
Hardware: `hwcfg=0x20102840` → IMEM=16KB, DMEM=5KB (at offset +0x108, not +0x008).

### 4. SEC2 CMDQ Never Initialized

- `CMDQ[0] head=0x00000000 tail=0x00000000` (registers at SEC2+0xa00/0xa04)
- `MSGQ[0] head=0x00000000 tail=0x00000000` (registers at SEC2+0xa30/0xa34)
- Full 64KB DMEM scan: NO init message found (expected `unit_id=0x01, msg_type=0x00`)
- SEC2 hwcfg=0x20420100 (64KB IMEM, 64KB DMEM) — firmware running but no CMDQ setup

### 5. PMC GR Reset Breaks CPUCTL Lock

```
Before: GPCCS cpuctl=0x00000000 (locked to host)
PMC GR reset (bit 12 toggle)
After:  GPCCS cpuctl=0x00000010 (STOPPED, but host CAN write STARTCPU)
```

LS mode persists (`sctl=0x3000`). STARTCPU after reset: `cpuctl=0x12`, PC stays 0.
Exception changes from `0x08070000` to `0x00070000` — still faults at PC=0.

### 6. HWCFG Offset Correction

Previous reports of "GPCCS hwcfg=0x00000000" were reading IRQSCLR (falcon+0x008).
Actual HWCFG is at falcon+0x108:

```
FECS  hwcfg (0x409108) = 0x20204080  → IMEM=32KB, DMEM=16KB
GPCCS hwcfg (0x41A108) = 0x20102840  → IMEM=16KB, DMEM=5KB
SEC2  hwcfg (0x087108) = 0x20420100  → IMEM=64KB, DMEM=64KB
```

### 7. apply_gr_bar0_init Wakes FECS But Not GPCCS

FECS transitions from STOPPED (cpuctl=0x10) to RUNNING (cpuctl=0x00) after
sw_nonctx.bin + dynamic GR writes are applied. GPCCS stays stuck because it
faulted before it could execute any firmware to respond to register changes.

## nouveau CMDQ Protocol (from source analysis)

SEC2 falcon func (`gp102_sec2_flcn`):
```c
.cmdq = { 0xa00, 0xa04, 8 },  // head, tail, stride
.msgq = { 0xa30, 0xa34, 8 },  // head, tail, stride
```

Init message (`nv_sec2_init_msg`) provides DMEM queue offsets. Without it,
host cannot construct CMDQ entries. The `BOOTSTRAP_FALCON` command via CMDQ:
```
unit_id=0x08, size=16, ctrl_flags=0x03, seq_id=N, cmd_type=0x00
flags=0 (RESET_YES), falcon_id=3 (GPCCS)
```

## Root Cause

GPCCS is trapped in LS mode with an invalid HS authentication context.
SEC2 ACR loaded firmware and issued STARTCPU, but GPCCS can't execute
because the HS code authentication failed during the nouveau→VFIO transition.

## Next Steps

1. **Path A:** Disable FLR during VFIO transition (preserve running GPCCS)
2. **Path B:** Fix bind_stat for SEC2 strategies 1-4 (enable full CMDQ)
3. **Path C:** Capture nouveau GPCCS warm state via GlowPlug swap
