# Experiment 211: PMU Mailbox Protocol — Path to Tier 2 Sovereign Compute

**Date:** 2026-05-19
**Hardware:** 2x NVIDIA Titan V (GV100), vfio-pci
**Spring:** hotSpring
**Status:** Phase A COMPLETE, Phase B COMPLETE (all attempts failed), Phase C PARTIAL
**Depends on:** Exp 210 (GPC boundary analysis, sovereignty tier model)

## Objective

Cross the Tier 1 → Tier 2 boundary by commanding the PMU falcon to
ungate the GPC power domain. This would enable sovereign shader execution
on Titan V without any vendor driver in the host.

## Background

Exp 210 established that all engine domains (GR, CE, NVDEC) are
power-gated after nouveau unbind. The PRI ring to these domains is dead
(reads return `0xbadfXXXX` faults). The chicken-and-egg: need PRI to
write power registers, but PRI to those domains requires power.

The PMU (Power Management Unit) falcon sits outside the gated domain.
It is alive post-unbind — its base registers at `0x10A000+` are
accessible. The PMU's primary function is managing GPU power domains.

## PMU Register Space (GV100)

```
Base: 0x10A000 (PPMU)

Key registers:
  0x10A000  PMU_FALCON_IRQSSET    — interrupt set
  0x10A004  PMU_FALCON_IRQSCLR    — interrupt clear
  0x10A008  PMU_FALCON_IRQSTAT    — interrupt status
  0x10A100  PMU_FALCON_CPUCTL     — CPU control (start/halt)
  0x10A104  PMU_FALCON_CPUSTAT    — CPU status (halted, etc.)
  0x10A108  PMU_FALCON_BOOTVEC    — boot vector
  0x10A110  PMU_FALCON_HWCFG      — hardware config
  0x10A118  PMU_FALCON_DMACTL     — DMA control
  0x10A120  PMU_FALCON_DMATRFBASE — DMA transfer base
  0x10A124  PMU_FALCON_DMATRFMOFFS — DMA transfer MOFFS
  0x10A128  PMU_FALCON_DMATRFCMD  — DMA transfer command
  0x10A12C  PMU_FALCON_DMATRFFBOFFS — DMA transfer FB offset
  0x10A180  PMU_FALCON_OS         — OS signal register

  0x10A450  PMU_FALCON_MAILBOX0   — command data
  0x10A454  PMU_FALCON_MAILBOX1   — command trigger
  0x10A458  PMU_FALCON_IRQMSET    — interrupt mask set
  0x10A460  PMU_FALCON_IRQDEST    — interrupt destination

  0x10A620  PMU_FALCON_PC         — program counter
```

## Investigation Plan

### Phase A: PMU Liveness Probe

Confirm PMU is alive and responsive after nouveau unbind.

1. Read `PMU_FALCON_CPUCTL` (`0x10A100`) — check for running state
2. Read `PMU_FALCON_PC` (`0x10A620`) — confirm PC is advancing
3. Read `PMU_FALCON_MAILBOX0/1` — check for idle/ready state
4. Read `PMU_FALCON_OS` (`0x10A180`) — check OS signal register

Expected: PMU falcon should be running (halted in command-wait loop if
HS firmware was loaded by nouveau before unbind).

### Phase B: Enumerate Known PMU Commands

From nouveau source (`nvkm/subdev/pmu/gv100.c`, `gk20a.c`, `gm200.c`)
and envytools documentation:

```
Known PMU mailbox protocol (Volta HS firmware):
  INIT          — initialize PMU subsystem
  FINI          — finalize / prepare for shutdown
  PG_CTRL       — power gating control (enable/disable per engine)
  CG_CTRL       — clock gating control
  THERM_CTRL    — thermal management
  VOLT_CTRL     — voltage control
  PERF_CTRL     — performance state changes
```

The key command is **PG_CTRL** — power gating control. If we can
construct the correct PG_CTRL message to disable GPC power gating,
the engines should come alive.

### Phase C: PMU Command Injection

Attempt to send a power gating disable command:

1. Write command payload to `MAILBOX0` (`0x10A450`)
2. Write command trigger to `MAILBOX1` (`0x10A454`)
3. Wait for PMU to process (poll `MAILBOX1` for completion)
4. Read back `GPC_ENABLES` at `0x41A004` — expect non-fault value
5. Read CE0 at `0x104000` — expect non-fault value
6. If success: attempt CE DMA copy (Exp 210 pipeline is ready)

### Phase D: Fallback — Direct PGOB Override

If PMU mailbox doesn't work (HS firmware may not respond to unsolicited
commands), try direct PGOB manipulation:

1. Read `PMC_ENABLE` at `0x200` — current engine enable state
2. Try writing PGOB registers directly (may require PRI ring path)
3. Try CG (clock gating) registers at `0x17E` range
4. Monitor for PRI fault changes after each write attempt

### Phase E: Fallback — Kernel Patch

If neither PMU nor PGOB works from userspace:

1. Modify nouveau `nvkm/engine/gr/gv100.c` `gv100_gr_fini()`
2. Skip the power-down sequence during driver unbind
3. Rebuild nouveau module
4. Test: `modprobe nouveau`, let it init, `echo 1 > unbind`, bind vfio-pci
5. Check GPC_ENABLES — should retain powered state

## Success Criteria

| Criterion | Register | Expected Value |
|-----------|----------|----------------|
| GPC domain alive | `0x41A004` (GPC_ENABLES) | Non-fault (not `0xbadfXXXX`) |
| CE domain alive | `0x104000` (CE0 base) | Non-fault (not `0xbadfXXXX`) |
| PBDMA no DEVICE error | `0x04X100` (PBDMA intr_0) | Bit 28 clear |
| CE DMA copy | dest buffer readback | Matches source pattern |
| GR method response | `0x409504` (FECS MTHD_CMD) | Method completion (not fault) |

## Relationship to Evolution Ladder

This experiment directly targets the **vendor-atheistic → silicon-deistic**
transition. If the PMU mailbox works, it means the GPU's own internal
firmware can be commanded to cooperate with sovereign software — no
vendor driver needed. The PMU becomes a servant, not a gatekeeper.

If the kernel patch is required instead, it's still vendor-atheistic
(open-source kernel, auditable patch) but less elegant. The PMU path
is the more aligned solution.

## Phase A Results: PMU Liveness (2026-05-19)

Hardware probe via `sovereign.pmu_investigate` RPC on Titan V (`0000:02:00.0`).

### PMU Core State

| Register | Value | Interpretation |
|----------|-------|----------------|
| CPUCTL | `0x00000020` | Running (bit 5 set = started), not halted |
| BOOTVEC | `0x00000000` | Boot vector zero (normal for HS) |
| PC | `0x00000000` | Static — PC not readable in HS-locked mode |
| SCTL | `0x00003002` | **HS-locked** (bit 1) + ACR locked (bit 12-13) |
| HWCFG | `0x40060100` | IMEM=3KB, DMEM=64KB, **signed firmware required** (bit 8) |
| MAILBOX0 | `0x00000300` | Idle state (nouveau PMU init marker: `0x300` = PMU_INIT_DONE) |
| MAILBOX1 | `0x00000000` | Cleared (no pending trigger) |
| IRQSTAT | `0x00000000` | No pending interrupts |
| IRQMASK | `0x00000000` | All interrupts masked |
| OS | `0x00000000` | No OS-level signal |
| PFIFO_ENABLE | `0x00000000` | **PFIFO disabled** — PMU hasn't unlocked PFIFO |

### PMU Queue State (MSG_QUEUE / CMD_QUEUE)

| Queue | Head | Tail | Status |
|-------|------|------|--------|
| Queue 0 | `0x00000000` | `0xBADF3040` | Tail PRI-faulted — queue in gated domain |
| Queue 1 | `0x00000000` | `0xBADF3040` | Same |
| Queue 2 | `0x00000000` | `0xBADF3040` | Same |
| Queue 3 | `0x00000000` | `0xBADF3040` | Same |

### PMU FBIF (DMA Window)

| Register | Value | Interpretation |
|----------|-------|----------------|
| FBIF_CTL | `0x00000110` | FBIF enabled, aperture configured |
| FBIF_TRANSCFG | `0x00000110` | Transaction config set |

### Key Finding: PMU is HS-Locked but Responsive

The PMU falcon is in **High-Security locked mode** (`SCTL=0x3002`). This means:

1. The ACR (Authenticated Code Runner) loaded signed firmware into PMU
2. The IMEM/DMEM are locked — cannot be read or written via PIO
3. PC reads as 0x0 (HS mode masks the real PC)
4. But MAILBOX0/1 are still accessible (they sit outside the security boundary)

**Critical observation**: MAILBOX0 contains `0x300`, which in nouveau's
PMU protocol means `PMU_INIT_MSG_DONE` — the PMU completed its init
sequence before nouveau unbound. The PFIFO is disabled (0x0), meaning
the PMU either explicitly disabled it during shutdown, or nouveau told
it to during `gv100_pmu_fini()`.

## Phase B Results: Ungating Attempts (2026-05-19)

Six ungating strategies attempted, all failed. GPC and CE remain at `0xBADF3000`.

### Attempt 1: CG Sweep + PRI Recovery

- CG: 0 changes (all targets already faulted), 16 faulted
- PRI: 7 alive stations, 6 faulted, bus recovered
- GPC: unchanged at `0xBADF3000`
- **Conclusion**: Clock gating was not the blocker — domains are power-gated, not clock-gated

### Attempt 2: PMC + PGOB Direct

- PMC_ENABLE = `0x5FECDFF1` (24+ engines enabled at PMC level)
- Wrote GPC broadcast control = `0x110` (ungate request)
- Wrote per-GPC PGOB = `0x0` (disable power gating)
- GPC: unchanged at `0xBADF3000`
- **Conclusion**: PGRAPH broadcast register is itself in the gated domain — writes are silently dropped

### Attempt 3: THERM Gate Override

- THERM_GATE_CTRL was already 0x0 (not gating via thermal subsystem)
- GPC: unchanged
- **Conclusion**: Thermal power gating is not the mechanism

### Attempt 4: PMC GR Engine Toggle

- Toggled GR bit 12 in PMC_ENABLE off then on
- GPC: unchanged
- **Conclusion**: PMC_ENABLE controls top-level engine enable, not GPC power domains. The engine is "enabled" but its internal power domain is gated.

### Attempt 5: PMU Mailbox PG_CMD_ALLOW

- Wrote MBOX0 = `0x308` (unit=PG `0x03`, cmd=PG_ALLOW `0x08`)
- Wrote MBOX1 = `1` (trigger)
- **PMU responded**: MBOX0 reverted to `0x300`, MBOX1 cleared to `0x0`
- GPC: unchanged
- **Conclusion**: PMU IS alive and processing mailbox writes. The simple
  command format was recognized (values changed) but didn't produce the
  desired power gating effect. The HS firmware's command dispatcher likely
  requires the **queue-based message protocol** (structured header + payload
  in DMEM), not simple MBOX0/MBOX1 command words.

### Attempt 5b: PMU Reinit Signal

- Wrote MBOX0 = `0` + MBOX1 = `1`
- MBOX0 restored to `0x300` (PMU re-asserted its init-done state)
- GPC: unchanged
- **Conclusion**: Confirms PMU is actively monitoring MBOX0/1 and restoring
  its state. It's not ignoring writes — it's rejecting our commands.

## Analysis: Why All Attempts Failed

### The Power Domain Architecture

```
PMC_ENABLE (0x200) ─── gates engine clock at top level
  └─ engines are "enabled" (PMC popcount = 24+)

GPC Power Domain ─── separate from PMC, controlled by:
  ├─ PGOB (Power Gating Override Block) ─── requires PRI to PGRAPH ← GATED
  ├─ THERM gate control ─── not the mechanism (already 0)
  ├─ PMU firmware ─── HS-locked, command format unknown
  └─ Boot ROM / cold init ─── only during power-on

The chicken-and-egg is confirmed at register level:
  - PGOB registers are INSIDE the gated domain
  - PMU mailbox is OUTSIDE but HS firmware rejects our commands
  - PMC toggle doesn't affect internal power domains
```

### PMU Mailbox Protocol Gap

The PMU responded to our MBOX0/1 writes (values changed and reverted),
confirming it's alive and processing. But the Volta HS PMU firmware uses
a queue-based message passing protocol, not simple command words:

```
nouveau gv100_pmu.c protocol:
  1. Write message header to CMD_QUEUE (in DMEM via FBIF DMA)
  2. Set queue head pointer (at 0x10A4C0+)
  3. Trigger PMU interrupt via MBOX1
  4. PMU reads from CMD_QUEUE, processes, writes response to MSG_QUEUE
  5. Read response from MSG_QUEUE

Problem: Queue tail registers (0x10A4D0+) return 0xBADF3040 — they're
in the gated domain too. The queue-based protocol may not be usable
from the VFIO side.
```

## Revised Path to Tier 2

Based on live hardware evidence, the paths are now re-prioritized:

### Priority 1: Kernel Patch (nouveau gv100_gr_fini)

The most viable path. Modify nouveau to skip GPC power-down during driver
unbind. This preserves the powered state for VFIO passthrough.

- File: `drivers/gpu/drm/nouveau/nvkm/engine/gr/gv100.c`
- Function: `gv100_gr_fini()` (called during unbind)
- Change: Skip the GPC power gating sequence, leave domains powered
- Risk: Low (only affects unbind path, not normal operation)

### Priority 2: nvidia-470 Warm Handoff

Use the proprietary nvidia-470 driver as the warm-handoff source. The
proprietary driver keeps GPCs powered during its normal operation. If we
unbind nvidia-470 cleanly and bind vfio-pci, the GPC domain may remain
powered.

### Priority 3: PMU Queue Protocol (Needs DMEM Access)

If we can construct a valid queue-based message IN the PMU's DMEM via
the FBIF DMA window (which IS accessible at `0x10AE00+`), we might be
able to send a proper PG command. This requires reverse-engineering the
Volta HS PMU message queue layout.

### Priority 4: K80 Cross-Gen (When Hardware Arrives)

The K80's unsigned falcons bypass all of this. No HS lock, no ACR, no
queue protocol — direct PIO upload and mailbox command.

## Code Changes (Exp 211)

### New Files
- `cylinder/src/vfio/pmu_investigate.rs` — PMU investigation pipeline:
  `PmuInvestigationResult`, `UngatingAttempt`, `investigate_pmu(bar0)`

### Modified Files
- `cylinder/src/vfio/mod.rs` — registered `pmu_investigate` module
- `cylinder/src/vfio/sovereign_stages.rs` — removed stale `#[expect(dead_code)]`
  on `CgSweepResult` fields (now used by PMU investigation)
- `server/handler/mod.rs` — added `sovereign.pmu_investigate` / `pmu.investigate` route
- `server/handler/dispatch/mod.rs` — added `sovereign_pmu_investigate()` handler

### New RPC Method

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"sovereign.pmu_investigate","params":{"bdf":"0000:02:00.0"}}' \
  | nc 127.0.0.1 PORT
```

## Cross-References

- Exp 210: `experiments/210_SOVEREIGN_GPC_BOUNDARY.md` — the wall
- Exp 209: `experiments/209_SOVEREIGN_VFIO_DISPATCH_BRIDGE.md` — PBDMA proven
- Exp 208: `experiments/208_REBOOT_EFFICIENT_SOVEREIGN_EVOLUTION.md` — 183ms warm
- Exp 204: VBIOS interpreter (422 ops, ~100 unknown opcodes)
- GAP-HS-047: Titan V PMU firmware extraction tool (deprioritized, may be relevant)
- GAP-HS-107: Tier 2 sovereign compute blocker (this experiment's gap entry)
- Silicon Deism: `infra/whitePaper/gen4/architecture/SILICON_DEISM.md`
