# hotSpring → coralReef: K80 Warm FECS/PFIFO Pipeline Handoff

**From:** hotSpring
**To:** coralReef, primalSpring
**Date:** April 30, 2026
**Experiment:** 179

## Summary

Tesla K80 (GK210B, Kepler) warm-catch pipeline: Nouveau performs full GR
initialization, then VFIO rebind hands warm state to coral-driver. FECS firmware
boots and reaches idle. PFIFO runlist completes. GPFIFO dispatch blocked on
scheduler error — active debugging.

## Breakthroughs

### 1. FECS Boot (Internal Firmware Protocol)

GK210B uses internal firmware (extracted from `nouveau.ko`). Key corrections:
- Falcon v3 IMEM tags (AINCW bit 24 per 256-byte block)
- csdata DMEM load bug: `AINCW + star` not `AINCW + starstar`
- Internal protocol: host starts FECS, FECS starts GPCCS
- Context size from `0x409804` (internal firmware populates directly)
- Fire-and-forget channel binding for internal firmware

### 2. PFIFO Pipeline (GK210B Post-VFIO FLR)

**Scheduler sub-block permanently dead** — registers `0x2004`, `0x2200-0x2253`,
`0x22C0`, `0x2300`, `0x2504`, `0x2600` PRI-faulted after FLR. Not recoverable.

**Workaround**: Accessible registers still functional:
- `0x2270` / `0x2274`: RUNLIST_BASE / RUNLIST_SUBMIT
- `0x2390+seq*4`: PBDMA→runlist assignment table (left by Nouveau)
- Read hardware's existing assignment, submit to correct runlist ID

**Critical fix**: PBDMA 0 → runlist **1** (not 0). Previous hardcoded runlist 0
caused silent stall. After fix, runlist completes.

### 3. Kepler-Specific GPFIFO Mechanisms

- Doorbell at `0x3000 + channel_id * 8` (not Volta usermode)
- Runlist entry: `(channel_id, 0x00000004)` — bare channel, no TSG
- PCCSR encoding: `(addr >> 12) | (target << 28)` matching Nouveau `gk104_chan_bind`

## Current Blocker

**SCHED_ERROR code=32** after runlist completion. PBDMA 0 idle with stale Nouveau
context. Channel stuck in PENDING status. Added PBDMA stale state clearing
(GP_BASE, GP_PUT, GP_GET, USERD, STATE, SIGNATURE). Next: validate clearing
effectiveness, check PCCSR binding.

## Files Changed (22 files in coralReef coral-driver)

Core: `kepler_fecs_boot.rs`, `fecs_method.rs`, `warm_channel.rs`, `kepler_csdata.rs`,
`gr_engine_status.rs`, `pfifo.rs`, `channel/mod.rs`, `page_tables.rs`, `registers.rs`,
`submission.rs`, `device_open.rs`

## For Upstream

- GK210B PFIFO accessibility map (register-level) documented in Exp 179
- Internal firmware protocol corrections applicable to any Kepler Falcon v3 target
- PBDMA→runlist dynamic discovery pattern reusable across GPU generations

## Next Steps

1. Resolve SCHED_ERROR code=32 (PBDMA stale state / PCCSR binding)
2. Run `k80_nouveau_warmup_dispatch` test for E2E validation
3. Validate sovereign Rust compute pipeline on K80 (WGSL → SM35 SASS → dispatch)
