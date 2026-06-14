# Experiment 078: Layer 7 Diagnostic Matrix

**Date:** 2026-03-23
**Goal:** Comprehensive diagnostic capture to confirm FECS-not-running as root cause of Layer 7 (GR engine) GPFIFO completion stalls on cold VFIO.
**Status:** CONFIRMED — FECS is the sole blocker. PBDMA never loads channel context because GR engine is dead.

## Hardware

- GPU: NVIDIA GV100 (Titan V) at `0000:03:00.0`
- VFIO group 69, legacy group mode (iommufd unavailable)
- GlowPlug/Ember active (not used for fd acquisition in this test — direct `/dev/vfio/69`)

## Key Findings

### 1. Falcon State (Root Cause Confirmation)

| Falcon | cpuctl | State | Secure | IMEM | DMEM | Mailbox0 |
|--------|--------|-------|--------|------|------|----------|
| **FECS** | `0x00000010` | **HRESET** | false | 32KiB | 8KiB | `0x00000000` |
| **GPCCS** | `0x00000010` | **HRESET** | false | 16KiB | 5KiB | `0x00000000` |
| **PMU** | `0x00000020` | HALTED | true | 64KiB | 64KiB | `0x00000300` |
| **SEC2** | `0xbadf1100` | INACCESSIBLE | true | — | — | `0xbadf1100` |

**FECS and GPCCS are both in hard reset (HRESET)** — no firmware loaded, zero mailboxes. This is the definitive root cause: without FECS firmware, the GR engine cannot schedule or execute commands.

**Critical discovery: FECS `secure=false`** — the HWCFG register reports no signed-firmware requirement. This opens a potential path for direct DMA firmware upload without ACR secure boot in Exp 080.

**PMU is halted but has mailbox0=0x300** — residual state from a prior driver session. Not relevant to GR dispatch.

**SEC2 reads `0xbadf1100` everywhere** — falcon is powered off or clock-gated. Not accessible on cold VFIO.

### 2. PCCSR Channel Status

```
PCCSR[0]: inst=0xa0000003 chan=0x11000001 status=PENDING en=true busy=true
          pbdma_fault=false eng_fault=false
```

Channel 0 stuck in **PENDING** at every capture point (pre-dispatch, t+1ms through t+1s, post-dispatch, post-timeout). The scheduler accepts the channel but never dispatches it to a PBDMA because the target engine (GR, runlist 1) is not ready.

### 3. PBDMA State (Never Loaded)

PBDMAs 1 and 2 serve GR runlist 1. All operational registers read zero at every time point:

```
PBDMA[1]: gp_base=0:0 gp_put=0 gp_fetch=0 gp_state=0 userd=0:0
          chan_state=0 method0=0xbad00200 intr=0x2040c028
PBDMA[2]: gp_base=0:0 gp_put=0 gp_fetch=0 gp_state=0 userd=0:0
          chan_state=0 method0=0xbad00200 intr=0x00000000
```

The `0xbad00200` on method0/data0 indicates PBDMA registers are in an error/inaccessible state. PBDMA[1] has stale interrupt bits (`0x2040c028`). Neither PBDMA ever loads the channel context — consistent with the scheduler refusing to dispatch.

### 4. Timed Post-Doorbell Captures

All 5 captures (t+1ms, t+10ms, t+100ms, t+500ms, t+1s) show identical state:
- PCCSR: PENDING
- PBDMAs: all zeros + bad00200

The scheduler never transitions the channel. This is not a timing issue — the engine itself is unresponsive.

### 5. Engine Topology

```
GR runlist: 1 (PBDMAs: [1, 2])
CE runlist: 0 (PBDMAs: [])  ← needs investigation
Active PBDMAs: [1, 2, 3, 21]
  PBDMA 1 → runlist 1 (GR)
  PBDMA 2 → runlist 1 (GR)
  PBDMA 3 → runlist 2
  PBDMA 21 → runlist 4 (reads 0xbadf1100 — inaccessible)
```

The ENGN_TABLE parsing needs refinement for GV100's multi-word format (entries showed type=65535 for first entry). The engine status registers do show activity:
- ENGN[0]: `0x13ad1b91` (runlist_bits=1) — GR engine, acknowledges runlist 1
- ENGN[2]: `0x11441418` (runlist_bits=1) — secondary engine on runlist 1

### 6. PFIFO Scheduler

```
intr=0x40000000 (rl_complete=true, chsw_err=false, pbdma_agg=false)
sched_disable=0x00000000  ← scheduler is NOT disabled
chsw_error=0x00000000     ← no channel-switch errors
pmc_enable=0x5fecdff1     ← GR engine enabled in PMC (bit 12 set)
pgraph_status=0x00000000  ← GR engine completely idle
```

The PFIFO scheduler is functional (runlist complete interrupt fires, preempt ACK received during init). It simply refuses to schedule the channel because the target engine is dead.

### 7. CE Isolation (PBDMA Comparison)

CE engines are on runlist 0, but no PBDMAs were mapped to runlist 0 by `find_pbdmas_for_runlist`. This suggests CE shares PBDMAs with another mechanism on GV100, or the runlist-to-PBDMA mapping table at `0x2390` doesn't cover CE. The GR PBDMAs (1, 2) show the same zeroed-out state as in the main diagnostic.

## Diagnosis

The failure chain is:

```
Cold VFIO bind
  → FLR resets all falcon IMEM/DMEM
    → FECS falcon in HRESET (cpuctl=0x10)
      → GR engine dead (pgraph_status=0)
        → PFIFO scheduler holds channel in PENDING
          → PBDMA never loads context (gp_base=0, gp_put=0)
            → GPFIFO completion never fires
              → FenceTimeout
```

## Path Forward

### Exp 079: Warm Handoff
Nouveau loads ACR → FECS/GPCCS firmware. Use `coralctl swap` to transition nouveau→vfio with `reset_method` disabled (Ember's NvidiaLifecycle). If FECS IMEM persists, channel should escape PENDING.

### Exp 080: Sovereign Boot
FECS `secure=false` is a major finding — suggests direct DMA upload to FECS IMEM may work without ACR. This dramatically simplifies the sovereign boot path:
1. Upload FECS firmware to IMEM via DMA (bypass ACR entirely)
2. Set BOOTVEC and start FECS
3. Wait for mailbox0 handshake
4. Proceed with GR engine init

## Files Changed

- `crates/coral-driver/src/vfio/channel/registers.rs` — falcon register constants
- `crates/coral-driver/src/nv/vfio_compute/diagnostics.rs` — new diagnostic structs
- `crates/coral-driver/src/nv/vfio_compute/mod.rs` — diagnostic capture methods
- `crates/coral-driver/src/nv/vfio_compute/submission.rs` — timed diagnostic variant
- `crates/coral-driver/src/nv/vfio_compute/dispatch.rs` — traced dispatch inner
- `crates/coral-driver/src/vfio/channel/mod.rs` — `create_on_runlist` factory
- `crates/coral-driver/tests/hw_nv_vfio.rs` — `vfio_layer7_diagnostic` + `vfio_pbdma_ce_isolation` tests
