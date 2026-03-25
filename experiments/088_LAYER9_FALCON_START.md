# Experiment 088: Layer 9 — FECS/GPCCS Post-ACR Start

**Date:** 2026-03-24
**Layer:** L9 (FECS/GPCCS full boot — HS mode release)
**Status:** **PARTIAL** (cpuctl=0x00 was misleading — see Exp 090)
**Titan:** #1 (0000:03:00.0), post-nouveau warmup

## Problem

After Exp 087 solved Layer 8 (WPR construction), ACR successfully bootstraps
FECS and GPCCS via `BOOTSTRAP_FALCON` mailbox commands. However, both falcons
end up at `cpuctl=0x12` — HRESET (bit 4) with STARTCPU sticky (bit 1) — rather
than transitioning to RUNNING (`cpuctl=0x00`).

Nouveau's source reveals that `BOOTSTRAP_FALCON` alone does NOT start the
falcons. The host must perform an explicit post-ACR start sequence.

## Root Cause

Nouveau's `gf100_gr_init_ctxctl_ext()` (gf100.c) performs these steps AFTER
ACR's `BOOTSTRAP_FALCON` returns:

1. `nvkm_mc_unk260(device, 1)` — clock-gating restore
2. Clear registers: `0x409800=0`, `0x41a10c=0`, `0x40910c=0`
3. `nvkm_falcon_start(GPCCS)` — GPCCS first (FECS depends on it)
4. `nvkm_falcon_start(FECS)` — FECS second
5. Poll `0x409800` bit 0 for up to 2000ms — FECS ready signal
6. Set watchdog timeout (`0x7fffffff`)

`nvkm_falcon_v1_start()` writes `0x2` (STARTCPU) to `CPUCTL` or `CPUCTL_ALIAS`
depending on bit 6 (ALIAS_EN).

coralReef already had `falcon_start_cpu()` in `sec2_hal.rs` matching this
exactly. It was only being called for SEC2 — not for FECS/GPCCS after ACR.

## Fix

Modified `strategy_mailbox.rs::attempt_acr_mailbox_command()` to add a
post-bootstrap falcon start sequence:

1. Clear `0x409800`, `0x41a10c`, `0x40910c`
2. `falcon_start_cpu(bar0, GPCCS_BASE)` — GPCCS first
3. `falcon_start_cpu(bar0, FECS_BASE)` — FECS second
4. Poll `0x409800` bit 0 for up to 2000ms (FECS ready)
5. Set FECS watchdog timeout on success

## Results

```
Pre-start: FECS cpuctl=0x00000012 GPCCS cpuctl=0x00000012
GPCCS after STARTCPU: cpuctl=0x00000000   ← RUNNING!
FECS RUNNING (no ready signal yet): 0x409800=0x00000000 cpuctl=0x00000000 (10ms)
Set FECS watchdog timeout 0x7fffffff
FECS final: cpuctl=0x00000000 mb0=0x00000000   ← RUNNING!
GPCCS final: cpuctl=0x00000000
```

Post-solver diagnostics:
```
FECS @ 0x00409000: IDLE (no mailbox)
  cpuctl=0x00000000 secure=false irqstat=0x00000000 curctx=0x00000002
GPCCS @ 0x0041a000: IDLE (no mailbox)
  cpuctl=0x00000000 secure=false irqstat=0x00000000 curctx=0x00000002
SEC2 @ 0x00087000: RUNNING (mailbox active)
  mb0=0x00000001 mb1=0x00000003
```

**Both falcons transitioned from `0x12` (HRESET) to `0x00` immediately upon
STARTCPU. GR engine shows `pgraph_status=0x00000081` (active).**

> **CAVEAT (Exp 090):** `cpuctl=0x00` only means "not in HRESET/HALTED" — it does
> NOT confirm actual instruction execution. Exp 090 Phase 2 revealed GPCCS faults
> at PC=0x0000 with `exci=0x08070000` despite cpuctl=0x00. FECS reaches an idle
> loop at PC=0x023c but cannot complete init without GPCCS. The PC and EXCI
> registers must always be checked alongside cpuctl.

## Observations

1. **`0x409800` ready signal did not fire** — FECS is running but the context
   switch mailbox wasn't set. This register likely requires FECS to have a valid
   context loaded before signaling. The fallback detection (`cpuctl & (HRESET |
   HALTED) == 0`) caught the running state correctly.

2. **GPCCS starts instantly** — no delay between STARTCPU and cpuctl=0x00.

3. **FECS running at 10ms** — effectively instant. The polling loop's first
   iteration detected it.

4. **`curctx=0x00000002`** on both falcons — they have a context pointer set
   (likely from ACR bootstrap).

5. **`pgraph_status=0x00000081`** — GR engine is enabled and active.

## Layer Status Update

| Layer | Status |
|-------|--------|
| L1-L5 | SOLVED |
| L6 | PARTIAL (GP_PUT DMA read) |
| L7 | **SOLVED** (Exp 085, B1-B7) |
| L8 | **SOLVED** (Exp 087, W1-W7) |
| L9 | **PARTIAL** (Exp 088, post-ACR start — cpuctl=0x00 but GPCCS faults at PC=0) |
| L10 | **BLOCKED** — shader dispatch / GR context init |

## Next: Layer 10

With FECS and GPCCS running, the next challenge is GR engine context
initialization and shader dispatch:

1. GR context allocation (golden context image)
2. Context switch to our channel
3. GPFIFO command submission through FECS
4. Shader execution on the SMs

The `PCCSR[0]: en=true busy=true` from the test output suggests the channel
infrastructure is partially there, but the first GPFIFO submission still
triggers a fence timeout (`GPFIFO completion stalled`). L6's GP_PUT DMA issue
and L10's GR context setup are the remaining blockers.

## Files Modified

- `coralReef/crates/coral-driver/src/nv/vfio_compute/acr_boot/strategy_mailbox.rs`
  - Added post-ACR falcon start in `attempt_acr_mailbox_command()`
