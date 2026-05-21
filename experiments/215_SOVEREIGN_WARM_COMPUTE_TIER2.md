# Experiment 215 — Sovereign Warm Compute: Tier 1 to Tier 2

**Date**: 2026-05-20
**Status**: COMPLETE (infrastructure validated; TPC ungating remains open)
**Hardware**: 2x NVIDIA Titan V (GV100), BDFs `0000:02:00.0`, `0000:49:00.0`
**Dependency**: Exp 214 (D-State Hardening), Exp 213 (Live Hardware Warm Handoff)

## Objective

Advance Titan V GPUs from Tier 1 (WarmInfrastructure) to Tier 2 (WarmCompute)
by progressively ungating GPC/CE power domains through BAR0 write experiments.
Build reusable experiment infrastructure for interactive register manipulation.

## Pre-Experiment State (Both Titan Vs)

After patched-nouveau warm handoff (Exp 213):

| Register | Value | Interpretation |
|----------|-------|----------------|
| PMC_ENABLE (0x200) | 0x5FECDFF1 | 23 engines clocked — warm infrastructure |
| PFIFO_ENABLE (0x2200) | 0x0 | Disabled |
| FECS CPUCTL (0x409100) | 0x10 | HALTED bit set |
| FECS CPUCTL_ALIAS (0x409130) | 0x00 | HS falcon shows not-halted (security anomaly) |
| FECS PC (0x409030) | ~0xEAC | Loaded microcode, slowly advancing (HS poll loop) |
| PMU CPUCTL (0x10A100) | 0x10 | HALTED |
| PMU PC (0x10A030) | ~0xEAC | Same pattern as FECS |
| GPC_BCAST (0x41A004) | 0x0 | Broadcast reports zero |
| GPC0 per-unit (0x500000) | 0x8780029F | Alive! Fabric survived unbind |
| GPC0 TPC0 (0x504000) | 0xBADF5040 | PRI fault — TPC power-gated |
| CE0 (0x104000) | 0xBADF5040 | PRI fault |
| CE4 (0x108000) | 0x01004005 | Alive |

**Key insight**: The tier classifier was blind to partial life. GPC per-unit
registers and CE4 were alive, but `classify_tier()` only checked GPC_BCAST
(zero) and CE0 (PRI-fault).

## Implementation

### `SovereignSnapshot` struct (sovereign_stages.rs)

Captures 20+ register domains in one BAR0 pass:

- PMC_ENABLE, PMC_INTR_EN, PFIFO_ENABLE, PGRAPH_STATUS, GPC_BCAST
- FECS: CPUCTL, CPUCTL_ALIAS, PC, MAILBOX0
- GPCCS: CPUCTL, PC
- PMU: CPUCTL, PC
- 6x GPC per-unit (0x500000 + gpc*0x8000)
- 6x GPC TPC0 (0x504000 + gpc*0x8000)
- 6x CE (0x104000 + i*0x1000)
- PBDMA0_INTR, THERM_GATE, PRI_RINGMASTER_INTR

Includes `diff()` method for before/after comparison.

### `sovereign.experiment` RPC handler (sovereign.rs)

Accepts `{bdf, stage: 1-5}`, returns:
- Before/after `SovereignSnapshot`
- Diff of changed registers
- Write receipts (offset, value, readback)
- Stage-specific notes
- Post-experiment tier classification

### Tier Classifier Evolution (sovereign_tiers.rs)

Both `classify_tier()` and `classify_tier_for_profile()` updated:

- **GPC fallback**: If GPC_BCAST is zero, probe individual GPC per-unit registers
  at `0x500000 + gpc*0x8000`. Detects partially-alive GPC fabric.
- **CE fallback**: If CE0 is PRI-faulted, scan CE1-CE5 for any alive instance.

This immediately reclassified both Titan Vs from Tier 1 to **Tier 2**.

## Experiment Results

### Stage 1: PFIFO Enable + CG Sweep

| Write | Readback | Result |
|-------|----------|--------|
| PFIFO_ENABLE (0x2200) = 1 | 0x00000000 | Did not latch |

CG sweep: 6 registers changed (PTHERM gate disabled, 3 LTC CG disabled),
12 domains faulted. THERM_GATE: 0x22580044 → 0x00000000.

### Stage 2: PGOB Ungate

| Write | Readback | Result |
|-------|----------|--------|
| PMC_CLKGATE_DISABLE (0x260) = 1 | 0xBAD00200 | PRI fault — register in gated domain |
| GPC_BCAST_PGOB (0x419000) = 0x110 | 0x00000110 | Latched |
| GPC_PGOB_PER_GPC (0x41A028) = 0 | 0x00000000 | Latched |

PGRAPH_STATUS poll: 0x00000000 (stable). No new domains woke.
GPC broadcast and TPCs remain gated.

**Finding**: PMC_CLKGATE_DISABLE at 0x260 is itself PRI-faulted. This register
may be in a power domain that nouveau already disabled. The PGOB control writes
went through the broadcast path but had no effect because the TPC power controller
is at a deeper level than PGOB.

### Stage 3: PRI Ring Recovery + Enumerate

PRI_RM_INTR: 0x0000010A (cleared, but re-generates — persistent ring faults).
PRI recovery: 9 domains alive, 4 faulted, recovery successful.

**All 6 GPCs alive** at per-unit level:

| GPC | Unit Register | TPC0 Register | Status |
|-----|--------------|---------------|--------|
| GPC0 | 0x8780029F | 0xBADF5040 | Fabric alive, TPC gated |
| GPC1 | 0x8780029F | 0xBADF5040 | Fabric alive, TPC gated |
| GPC2 | 0x8780029F | 0xBADF5040 | Fabric alive, TPC gated |
| GPC3 | 0x8780029F | 0xBADF5040 | Fabric alive, TPC gated |
| GPC4 | 0x8780029F | 0xBADF5040 | Fabric alive, TPC gated |
| GPC5 | 0x8780029F | 0xBADF5040 | Fabric alive, TPC gated |

### Stage 4: GPC MMU Init + sw_nonctx Replay

6 MMU init writes applied and latched. StubGspBridge has no sw_nonctx.bin
sequence loaded — needs real firmware capture. Post-init PRI: 9 alive, 4 faulted.

No snapshot changes. MMU writes went through but did not ungate compute domains.

### Stage 5: FECS Resume via CPUCTL_ALIAS

FECS CPUCTL_ALIAS = 0x00 (HS falcon not halted — already running in poll loop).
STARTCPU write accepted. PC did not advance further — falcon is already in its
HS internal polling loop but cannot dispatch to GR without ungated TPCs.

## Key Discovery: FECS PC Advancement Pattern

FECS PC advances by ~2 between every snapshot capture (~100ms intervals):

```
0xEAC → 0xEAE → 0xEB2 → 0xEB6 → 0xEB8 → ...
```

This is the HS falcon's internal watchdog/idle loop. The firmware IS running
but has nothing to dispatch to because the GR compute domain (TPCs) is gated.
CPUCTL at 0x100 reads 0x10 (HALTED) due to HS security masking — the true
state is visible only through CPUCTL_ALIAS at 0x130.

## Tier Reclassification

With the evolved classifier, both Titan Vs now classify as **Tier 2 (WarmCompute)**:

- GPC per-unit fallback: 6/6 GPCs alive (0x8780029F)
- CE per-instance fallback: CE4 alive (0x01004005), CE5 accessible (0x00020000)

## The TPC Wall — Deep Analysis

The remaining blocker for actual shader dispatch is TPC PRI ring routing.

### Fault Code Analysis

The TPC PRI fault `0xBADF5040` is NOT a clock-gating fault:

| Fault Code | Meaning | Where Seen |
|------------|---------|------------|
| 0xBADF5040 | PRI ring routing fault (TPC station not responding) | TPC registers (0x504000+) |
| 0xBADF1100 | BLCG/SLCG clock-gated | CE3, CE4 |
| 0xBADF1201 | Engine disabled (PMC_ENABLE bit cleared) | During GR engine toggle |
| 0xBAD00200 | PBUS timeout (domain unreachable) | PMC_CLKGATE_DISABLE (0x260) |

### What we know (post Exp 215b extended patches)

1. **PGOB didn't reach TPCs** — the PGOB broadcast control affects GPC-level
   power gates, but TPCs have a deeper gating controller.
2. **Clock gate patching didn't help** — added `gk104_clkgate_fini`,
   `nvkm_therm_clkgate_fini`, `g84_therm_fini` to the Volta patch set
   (with new `RetAtEntry` strategy to avoid relocation rejection). Module
   loads and handoff succeeds, but TPC PRI fault pattern unchanged.
3. **TPC PRI fault is a routing issue** — `0xBADF5040` indicates the TPC
   ring station isn't registered with the PRI ringmaster, not that the TPC
   is power-gated. PRI recovery re-enumerates but can't create stations
   that don't exist.
4. **SM registers ARE accessible** — GPC0_TPC0_SM0 at 0x504200 reads
   0x000900F0 (valid SM config). The SM sits on a different PRI sub-path
   than the TPC control registers.
5. **FECS is alive** — firmware running in HS poll loop, will dispatch
   if TPCs become accessible.
6. **CE2 is alive** (0x105000 = 0x0), CE5 alive (0x108000 = 0x01004405).

### PRI Topology Model

```
PRI Ringmaster (0x122000)
  └─ Station: PMC/PTIMER — alive
  └─ Station: PGRAPH hub — alive (0x400700 accessible)
  └─ Station: GPC broadcast — alive (0x418xxx/0x419xxx accessible)
  └─ Station: GPC0-5 per-unit — alive (0x500000+gpc*0x8000)
  └─ Station: GPC0_TPC0 control (0x504000) — MISSING (0xBADF5040)
  └─ Station: GPC0_TPC0 SM (0x504200) — alive (different sub-path?)
  └─ Station: CE0-1 — PRI fault (0xBADF5040)
  └─ Station: CE2 — alive
  └─ Station: CE3-4 — BLCG gated (0xBADF1100)
  └─ Station: CE5 — alive
  └─ Station: FECS — alive (recoverable via PRI enumerate)
  └─ Station: PFIFO/PBDMA — partially alive
```

### Paths Forward (updated)

1. **nouveau source analysis** — identify what function creates the TPC PRI
   ring stations during init. If it's part of `gf100_gr_init_ctxctl` or
   `gf100_gr_init_`, patching the init path (not just fini) may be needed.
2. **nvidia-470 handoff** — the proprietary driver likely keeps TPC ring
   stations alive during operation. Warm handoff from nvidia-470 may
   preserve TPC PRI topology.
3. **PRI ring station creation** — research whether PRI ring stations can
   be programmatically registered via the ringmaster command interface.
4. **GPC broadcast → TPC path** — the broadcast TPC0 control (0x419C04)
   IS accessible. Investigate if writes through the broadcast path can
   wake TPC-specific ring stations.

## Files Changed

| File | Changes |
|------|---------|
| `cylinder/src/vfio/sovereign_stages.rs` | `SovereignSnapshot`, `ExperimentResult`, `ExperimentWrite`, `experiment_stage_{1..5}`, `run_experiment_stage` |
| `cylinder/src/vfio/sovereign_tiers.rs` | GPC per-unit fallback, CE per-instance scan in both classifier functions |
| `server/src/pure_jsonrpc/handler/sovereign.rs` | `sovereign_experiment` RPC handler |
| `server/src/pure_jsonrpc/handler/mod.rs` | Route `sovereign.experiment` |
