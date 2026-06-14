# Experiment 132: Ember-Frozen Warm Dispatch (Diesel Engine Pattern)

**Date:** 2026-03-30
**Hardware:** Titan V (GV100), Tesla K80 (GK210)
**Status:** IMPLEMENTED — awaiting hardware validation

## Objective

Evolve the warm handoff dispatch to leverage ember as keepalive and glowplug
as passthrough orchestrator (the "diesel engine" pattern):

1. **Glowplug** is the passthrough — orchestrates the driver swap sequence
2. **Ember** is the keepalive — holds VFIO fds, provides BAR0 MMIO access,
   manages livepatch, and now supports `mmio.write` for active intervention

The key insight: nouveau's normal teardown destroys PFIFO infrastructure
(channels, runlists, PBDMA state) even when livepatch prevents `mc_reset`
from resetting falcon IMEM. The previous `warm_handoff()` PFIFO config was
too conservative — it preserved everything, but there was nothing left to
preserve. The new `warm_fecs()` config accepts that PFIFO is destroyed and
rebuilds it while FECS is frozen.

## Architecture: Diesel Engine Analogy

- **Glowplug** = glow plug wire (pre-heats the engine)
  - Disables livepatch → swaps to nouveau → nouveau boots FECS
  - Enables livepatch → freezes teardown paths
  - Sends STOP_CTXSW via ember.mmio.write → freezes FECS scheduling
  - Captures PFIFO snapshot via ember.mmio.read
  - Swaps back to vfio-pci

- **Ember** = keep-alive (maintains the engine state)
  - Holds VFIO fds across driver swaps (no bus reset)
  - BAR0 mmio.read/mmio.write while ANY driver is bound
  - Livepatch enable/disable control
  - FECS state monitoring
  - FD sharing via SCM_RIGHTS

- **Combustion** = open_warm_with_context
  - Receives handoff context (fecs_frozen, pfifo_snapshot)
  - Selects warm_fecs() PFIFO config (rebuilds infrastructure)
  - Sends START_CTXSW to resume FECS scheduling
  - Sets up GR context, dispatches compute shader

## Changes

### 1. Phase 0: Fix Workspace Compilation

Fixed `uvm_compute.rs` test module missing 8 GPU class constant imports:
`VOLTA_CHANNEL_GPFIFO_A`, `VOLTA_COMPUTE_A`, `AMPERE_CHANNEL_GPFIFO_A`,
`AMPERE_COMPUTE_A`, `AMPERE_COMPUTE_B`, `BLACKWELL_COMPUTE_A`,
`BLACKWELL_COMPUTE_B`, `BLACKWELL_CHANNEL_GPFIFO_B`.

### 2. Ember `mmio.write` (Phase 1)

Added `ember.mmio.write` JSON-RPC handler to coral-ember:
- Opens BAR0 `resource0` read-write via `Bar0Access::open_resource`
- Validates 4-byte alignment
- Writes a single u32, reads back for verification
- Returns `{offset, value, readback}` in response
- Routed in both Unix socket and TCP dispatchers

Added `EmberClient::mmio_write(&self, bdf, offset, value)` to coral-glowplug:
- Mirrors `mmio_read` client pattern
- Returns readback value for verification

### 3. Warm Handoff FECS Freeze (Phase 2)

Modified `warm_handoff.rs` to add steps 6b and 6c between FECS poll and
swap-back:

**Step 6b: STOP_CTXSW** — when FECS is running, send method 0x01 via
ember.mmio.write (registers 0x409500/0x409504, poll 0x409804). Freezes
FECS scheduling so it doesn't notice channels being freed during nouveau
teardown.

**Step 6c: PFIFO snapshot** — capture critical PFIFO registers via
ember.mmio.read:
- `PMC_ENABLE` (0x200) — engine clock gating
- `PBDMA_MAP` (0x2004) — present PBDMA engines
- `PFIFO_SCHED_EN` (0x2504) — scheduler state
- `RUNLIST_BASE` (0x2270) — runlist DMA base
- `RUNLIST_SUBMIT` (0x2274) — runlist submit config

Both `fecs_frozen` and `pfifo_snapshot` are included in the handoff result
JSON for consumption by `open_warm_with_context`.

### 4. PfifoInitConfig::warm_fecs() (Phase 3)

New hybrid PFIFO init config for FECS-frozen warm handoff:
- **PMC_ENABLE**: untouched (FECS engine bits stay enabled)
- **PMC PFIFO reset**: skipped (preserves falcon clock domain)
- **PRIV ring**: cleared (swap may leave stale faults)
- **PBDMA**: force-cleared (nouveau's addresses are unmapped)
- **Runlists**: flushed empty (clears stale nouveau entries)
- **Preempt**: skipped (FECS scheduling already frozen)
- **Scheduler**: enabled (needed for dispatch after START_CTXSW)

Added `VfioChannel::create_warm_fecs()` using this config.

### 5. open_warm_with_context (Phase 3)

New `NvVfioComputeDevice::open_warm_with_context(bdf, fds, sm, class, ctx)`
accepting `WarmHandoffContext`:
- `fecs_alive: bool` — FECS was seen running during poll
- `fecs_frozen: bool` — STOP_CTXSW was sent
- `pfifo_snapshot: Option<PfifoSnapshot>` — pre-swap PFIFO state

Strategy selection:
- `fecs_frozen` → `warm_fecs()` config + `restart_frozen_fecs()`
- `fecs_alive` only → conservative `warm_handoff()` + `restart_warm_falcons()`

`restart_frozen_fecs()`:
1. Clears stale PBDMA interrupts
2. Resets FECS method interface status
3. Re-applies GR engine/interrupt configuration
4. Sends START_CTXSW (method 0x02) to resume scheduling
5. Falls back to `restart_warm_falcons()` if START_CTXSW fails
6. Sets up GR context (discover sizes, bind, golden save)

### 6. K80 Cold Boot Audit (Phase 4)

Audited K80 cold boot recipe and Kepler PFIFO init:

**Finding 1**: `k80_cold_boot::cold_boot()` priority ordering is correct.
Clock domains (priority 0-2) are applied before everything else.
PMC writes (priority 3) enable PFIFO clock domain before PFIFO writes
(priority 25) are applied.

**Finding 2**: `exp123k_common::apply_nvidia470_recipe()` skips PMC_ENABLE
(0x200), which means PFIFO clock domain (bit 8) is never enabled. This
causes `init_kepler_pfifo` to detect PRI faults and skip PFIFO init.

**Fix**: Added PMC_ENABLE bit 8 auto-enable as a prerequisite in
`init_kepler_pfifo`. If bit 8 is not set, the function now enables it
before probing PFIFO registers. Added documentation noting that sovereign
K80 dispatch should use the full cold boot path.

### 7. Test Harness (Phase 5)

New `open_vfio_warm_with_context()` test helper:
- Calls glowplug warm_handoff, extracts `fecs_frozen` and `pfifo_snapshot`
- Parses hex snapshot values into `PfifoSnapshot` struct
- Opens device via `open_warm_with_context` with full context

New test `vfio_dispatch_warm_fecs_frozen`:
- Exercises the full diesel engine pipeline end-to-end
- Compiles a NOP shader via coral-reef, dispatches via GPFIFO
- Detailed diagnostic output for failure analysis

Re-exported `WarmHandoffContext` and `PfifoSnapshot` from
`coral_driver::nv` for test and external use.

## Sequence Diagram

```
coralctl warm-fecs BDF
  └→ glowplug: livepatch.disable
  └→ glowplug → ember: swap nouveau
       └→ ember: unbind vfio, bind nouveau
       └→ nouveau: ACR boot → FECS/GPCCS alive
  └→ glowplug: livepatch.enable (freeze mc_reset)
  └→ glowplug: poll FECS via ember.fecs.state
  └→ [NEW] glowplug → ember: mmio.write STOP_CTXSW
       └→ FECS scheduling frozen
  └→ [NEW] glowplug → ember: mmio.read PFIFO snapshot
  └→ glowplug → ember: swap vfio-pci
       └→ nouveau teardown (channels freed, mc_reset NOPed)
       └→ FECS alive+frozen, PFIFO destroyed
  └→ returns: {fecs_frozen: true, pfifo_snapshot: {...}}

test: open_vfio_warm_with_context()
  └→ ember: request_fds (SCM_RIGHTS)
  └→ open_warm_with_context(ctx)
       └→ [warm_fecs] clear PCCSR, clear PFIFO intr
       └→ create_warm_fecs: rebuild PFIFO (force-clear PBDMA, flush runlists)
       └→ restart_frozen_fecs: START_CTXSW → FECS resumes
       └→ setup_gr_context_warm: discover sizes, bind, golden save
  └→ dispatch compute shader via GPFIFO
```

## Expected Outcome

If the PCI bus reset during vfio-pci rebind does NOT destroy falcon IMEM
(this is the key assumption — livepatch blocks mc_reset, but PCIe FLR
could still occur on FLR-capable GPUs), then:

1. FECS survives with scheduling frozen
2. PFIFO rebuilt from warm_fecs config
3. START_CTXSW resumes scheduling with our new channel
4. GR context setup succeeds (method 0x10 responds)
5. Compute dispatch through GPFIFO doorbell works

**Key risk**: Titan V does NOT have FLR capability (confirmed Exp 131),
so vfio-pci rebind should NOT trigger a bus reset. This is the primary
validation target. K80 also lacks FLR and uses the cold boot path.

## Files Modified

- `crates/coral-driver/src/nv/uvm_compute.rs` — test imports
- `crates/coral-ember/src/ipc/handlers_device.rs` — mmio_write handler
- `crates/coral-ember/src/ipc.rs` — mmio.write routing
- `crates/coral-glowplug/src/ember.rs` — EmberClient::mmio_write
- `crates/coral-glowplug/src/socket/handlers/warm_handoff.rs` — FECS freeze + PFIFO snapshot
- `crates/coral-driver/src/vfio/channel/pfifo.rs` — PfifoInitConfig::warm_fecs()
- `crates/coral-driver/src/vfio/channel/mod.rs` — create_warm_fecs
- `crates/coral-driver/src/nv/vfio_compute/mod.rs` — WarmHandoffContext, PfifoSnapshot, open_warm_with_context
- `crates/coral-driver/src/nv/vfio_compute/init.rs` — restart_frozen_fecs
- `crates/coral-driver/src/nv/mod.rs` — re-exports
- `crates/coral-driver/src/vfio/channel/kepler.rs` — PMC bit 8 auto-enable
- `crates/coral-driver/tests/hw_nv_vfio/exp123k_common.rs` — audit note
- `crates/coral-driver/tests/hw_nv_vfio/helpers.rs` — open_vfio_warm_with_context
- `crates/coral-driver/tests/hw_nv_vfio/dispatch.rs` — vfio_dispatch_warm_fecs_frozen
