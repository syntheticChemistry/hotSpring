# Experiment 211: PMU Mailbox Protocol — Path to Tier 2 Sovereign Compute

**Date:** 2026-05-19 (framed)
**Hardware:** 2x NVIDIA Titan V (GV100), vfio-pci
**Spring:** hotSpring
**Status:** NOT STARTED — investigation plan framed from Exp 210 findings
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

## Cross-References

- Exp 210: `experiments/210_SOVEREIGN_GPC_BOUNDARY.md` — the wall
- Exp 209: `experiments/209_SOVEREIGN_VFIO_DISPATCH_BRIDGE.md` — PBDMA proven
- Exp 208: `experiments/208_REBOOT_EFFICIENT_SOVEREIGN_EVOLUTION.md` — 183ms warm
- Exp 204: VBIOS interpreter (422 ops, ~100 unknown opcodes)
- GAP-HS-047: Titan V PMU firmware extraction tool (deprioritized, may be relevant)
- GAP-HS-107: Tier 2 sovereign compute blocker (this experiment's gap entry)
- Silicon Deism: `infra/whitePaper/gen4/architecture/SILICON_DEISM.md`
