# Experiment 179: K80 Warm FECS Dispatch Pipeline

**Date:** April 30, 2026
**Status:** ⚠️ In Progress (FECS boot operational, PFIFO runlist operational, GPFIFO dispatch blocked on SCHED_ERROR)
**Hardware:** Tesla K80 (GK210B, Kepler)
**Primal:** coralReef (coral-driver)
**Predecessor:** Exp 178 (PGOB analysis), Exp 155 (K80 warm FECS), Exp 174 (K80 sovereign boot)

## Objective

Full GR engine initialization on Tesla K80 via nouveau warm-catch: let Nouveau
bring up GPCs, PRI topology, and FECS/GPCCS firmware, then VFIO rebind and
establish sovereign GPFIFO dispatch pipeline for compute workloads.

## Approach: Nouveau Warm-Catch

Instead of cold sovereign boot (blocked by PGOB/PRI GPC enrollment — Exp 178),
leverage Nouveau's proven init path:

1. `modprobe nouveau` — Nouveau initializes K80 (PGOB, PRI, FECS/GPCCS, GR)
2. `echo <bdf> > unbind` from nouveau
3. `echo <bdf> > bind` to vfio-pci
4. coral-driver takes over warm state via `open_warm_legacy(bdf)`

## Key Discoveries

### FECS/GPCCS Boot (Internal Firmware Protocol)

- GK210B uses **internal firmware** (embedded in `nouveau.ko`, extracted as
  `gk110_internal_fecs_code.bin` / `gk110_internal_gpccs_code.bin`)
- Falcon v3 PIO upload: IMEM needs integrity tags per 256-byte block (AINCW bit 24)
- **csdata bug found and fixed**: `AINCW + starstar` was overwriting FECS DMEM;
  corrected to `AINCW + star`
- Internal protocol: host starts FECS only; FECS starts GPCCS internally
- FECS reaches idle state: CPUCTL = 0x20 (HALTED bit = software idle, not hardware halt)
- Context size read from `0x409804` (internal firmware populates directly, no FECS
  method 0x10 needed)
- Fire-and-forget channel binding via `fecs_internal_bind_channel`

### PFIFO Pipeline (GK210B after VFIO FLR)

- **Scheduler sub-block dead**: Registers `0x2004`, `0x2200-0x2253`, `0x22C0`,
  `0x2300`, `0x2504`, `0x2600` permanently PRI-faulted (`0xbad0011f`). Not
  recoverable by PMC resets, PRI ring re-init, or PBUS resets.
- **Accessible registers**: `0x2270` (RUNLIST_BASE), `0x2274` (RUNLIST_SUBMIT),
  `0x2390+seq*4` (PBDMA→runlist assignment table), `0x2100` (PFIFO_INTR)
- **PBDMA→runlist assignment**: `0x2390` shows PBDMA 0 → runlist **1** (left by
  Nouveau). Previous hardcoded runlist 0 was wrong — silent stall.
- **Runlist entry format**: GK104 uses `(channel_id, 0x00000004)`, not instance
  address encoding.
- **Kepler doorbell**: `0x3000 + channel_id * 8` (not Volta usermode doorbell).
- After fixes: **runlist completes** for the first time (`rl_pending = 0x00000000`).

### Current Blocker: SCHED_ERROR code=32

- Runlist completes but scheduler reports `SCHED_ERROR code=32`
- PBDMA 0 remains idle with stale Nouveau state (`gp_base=0x0000802b`)
- Channel shows PCCSR status = PENDING (never transitions to RUNNING)
- Added PBDMA stale state clearing (GP_BASE, GP_PUT, GP_GET, USERD, STATE, SIGNATURE)
- Next investigation: PCCSR binding correctness, PBDMA context handoff

## Files Changed (coralReef coral-driver)

| File | Change |
|------|--------|
| `kepler_fecs_boot.rs` | PMC GR reset + CG disable, MC_UNK260 bracket, Falcon PC fix |
| `fecs_method.rs` | 0x409840 wake trigger, internal firmware method interface |
| `warm_channel.rs` | Internal firmware context size, non-fatal watchdog, fire-and-forget binding |
| `kepler_csdata.rs` | AINCW+star fix (was +starstar) |
| `gr_engine_status.rs` | fecs_halted() CPUCTL interpretation, ctxsw_mailbox0 |
| `pfifo.rs` | PBDMA→runlist from 0x2390, skip faulted regs, stale PBDMA clear |
| `channel/mod.rs` | Use discovered target_runlist, post-bind diagnostics |
| `page_tables.rs` | GK104 runlist entry format |
| `registers.rs` | gk104_doorbell, runlist encoding, PCCSR fields |
| `submission.rs` | Kepler-specific doorbell |
| `device_open.rs` | Fine-grained PFIFO register probe |

## Register Map: GK210B PFIFO After VFIO FLR

| Register | Name | Value | Accessible |
|----------|------|-------|------------|
| 0x2100 | PFIFO_INTR | varies | YES |
| 0x2200 | PFIFO_CACHES | 0xbad0011f | NO |
| 0x2204 | PFIFO_MODE | 0xbad0011f | NO |
| 0x2270 | GK104_RUNLIST_BASE | writable | YES |
| 0x2274 | GK104_RUNLIST_SUBMIT | writable | YES |
| 0x2390 | PBDMA_ASSIGN_SEQ0 | 0x00000001 | YES |
| 0x2504 | SCHED_EN | 0xbad0011f | NO |
| 0x252C | BIND_ERROR | 0xbad0011f | NO |
| 0x254C | SCHED_ERROR | varies | YES (partial) |
| 0x2600 | PBDMA_MAP | 0xbad0011f | NO |

## Next Steps

1. Fix SCHED_ERROR code=32 — investigate PBDMA stale state clearing effectiveness
2. Verify PCCSR instance block binding (IOVA encoding in bits [27:0])
3. Once GPFIFO pipeline operational, run `k80_nouveau_warmup_dispatch` test
4. Validate sovereign Rust compute pipeline on K80 (WGSL → SM35 SASS → dispatch)
