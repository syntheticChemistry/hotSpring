# Exp 092: Experiment Loop Infrastructure + First Personality Sweep

**Date:** 2026-03-25
**Status:** COMPLETE
**Depends:** Exp 091, coralReef Iter 66 (ring/mailbox)
**Goal:** Wire the full adaptive experiment loop (journal, observers, ring_meta persistence), run first automated personality sweep on both Titan Vs

## Summary

Closed all "defined vs. wired" gaps from the Exp 091 infrastructure. The system now self-learns from every swap and reset, persists timing data, extracts driver-specific insights from mmiotraces, and provides an automated sweep command for hardware characterization.

## Infrastructure Wired (7 Gaps Closed)

### Gap 1: ember.device_reset Journal Instrumentation
- `ipc.rs` now wraps every reset (SBR, bridge-SBR, remove-rescan) with `Instant` timing
- Creates `ResetObservation` with success/failure/duration
- Appends `JournalEntry::Reset` to persistent JSONL journal
- RPC response includes `duration_ms`
- **Feeds:** `AdaptiveLifecycle` reset method selection

### Gap 2: SwapObservation Captured in DeviceSlot
- New field: `DeviceSlot::last_swap_observation: Option<SwapObservation>`
- Every successful ember swap stores full observation (timing, trace path, health)
- Logged with `total_ms`, `bind_ms`, `trace_path` on capture

### Gap 3: ObserverRegistry Wired into handle_swap
- After swap, `ObserverRegistry::default_observers()` runs `observe_swap` + `observe_trace`
- Insights logged and returned in RPC response JSON
- Concrete observers: NouveauObserver, VfioObserver, NvidiaObserver, NvidiaOpenObserver

### Gap 4: RingMeta Save/Restore Around Swaps
- Before dropping VFIO: `ring_meta_snapshot()` → `ring_meta_set` to ember (version incremented)
- After reacquiring VFIO: `ring_meta_get` → `restore_ring_meta` rebuilds mailboxes and rings
- State survives personality round-trips

### Gap 5: `coralctl experiment sweep` Command
- Iterates list of personalities on a target BDF
- Each swap: traced, timed, observer insights captured, journals everything
- Returns to base personality between tests
- Prints comparison table with TOTAL_MS, BIND_MS, UNBIND_MS, INSIGHTS, TRACE/ERROR
- Usage: `coralctl experiment sweep 0000:03:00.0 --personalities nouveau,nvidia-open`

### Gap 6: VFIO Settle Cap Removed
- Removed `.min(2)` hardcoded cap on VFIO settle time in `bind_vfio`
- `AdaptiveLifecycle`'s learned `settle_secs` now applies without ceiling

### Build Verification
- Full workspace build: 0 errors, 1 pre-existing warning
- Full test suite: **4,065 tests pass, 0 failures**

## First Experiment Sweep Results

### Titan V #1 (0000:03:00.0)

| Personality | Status | Total (ms) | Bind (ms) | Unbind (ms) | Insights | Trace |
|-------------|--------|-----------|----------|------------|----------|-------|
| nouveau     | OK     | 21,916    | 21,084   | 812        | 2        | 1.4MB mmiotrace |
| nvidia-open | OK     | 26,803    | 25,967   | 817        | 2        | 521B (header only) |

### Titan V #2 (0000:4a:00.0)

| Personality | Status | Total (ms) | Bind (ms) | Unbind (ms) | Insights | Trace |
|-------------|--------|-----------|----------|------------|----------|-------|
| nouveau     | OK     | 22,178    | 21,318   | 842        | 2        | 1.4MB mmiotrace |
| nvidia-open | OK     | 26,823    | 25,959   | 846        | 2        | 521B (header only) |

### Cross-Card Consistency
- Nouveau bind: 21,084 vs 21,318ms (234ms / 1.1% variance)
- nvidia-open bind: 25,967 vs 25,959ms (8ms / 0.03% variance)
- Unbind times: ~820ms consistent across all swaps
- **Both cards are functionally identical** — rules out hardware variance

### Journal Aggregate (Post-Sweep + Warm-FECS)

| Personality | Count | Avg Total | Avg Bind | Avg Unbind |
|-------------|-------|-----------|----------|------------|
| vfio        | 6     | 7,504ms   | 7,068ms  | 416ms      |
| nouveau     | 4     | 12,398ms  | 11,551ms | 827ms      |
| nvidia-open | 2     | 26,813ms  | 25,963ms | 831ms      |

## Trace Analysis: Nouveau GV100 Boot Sequence

### Key Finding: GR Init is 100% Firmware-Driven

32,507 MMIO operations in 1.2 seconds. **Zero** writes to:
- 0x409xxx (FECS Falcon)
- 0x41axxx (GPCCS Falcon)
- 0x840xxx (SEC2 Falcon)

Nouveau's entire ACR → FECS → GPCCS chain goes through SEC2 DMA at 0x800000, invisible to mmiotrace.

### Visible MMIO Boot Phases

| Phase | Time (s) | Registers | Operations | Purpose |
|-------|----------|-----------|------------|---------|
| 1 | 7295.543 | 0x12004c, 0x000160, 0x00d054 | ~2,800 | PRIV_RING topology + PMC_ENABLE + PFIFO poll |
| 2 | 7295.730 | 0x009420, 0x070000, 0x100cb8 | ~300 | PTIMER + PRAMIN window + PFB/FBPA config |
| 3 | 7295.753 | 0x000204, 0x040xxx-0x05axxx | ~100 | TOP_ENABLE + 14× GPC interrupt config |
| 4 | 7295.755 | 0x800000 | 4 | SEC2 boot descriptor + DMA setup |
| 5 | 7295.770 | 0x002630 | 2 | PFIFO PBDMA config |
| 6 | 7295.777 | 0x640xxx, 0x610xxx | ~200 | L2 cache/LTC + Display engine |
| 7 | 7295.782 | 0x00dc68 | 6 | TLB flush cascade |
| 8 | 7295.843 | 0x611860, 0x000160 | 3 | Display activate + final PMC_ENABLE |

### nvidia-open Trace: Empty
521 bytes — only mmiotrace header. GSP handles everything. No useful MMIO data captured.

## Post-Sweep Validation

Both Titan Vs returned to VFIO cleanly. After warm-fecs cycle:
- PMC_BOOT0: `0x140000a1` (GV100 confirmed)
- PMC_ENABLE: `0x5fecdff1` (all engines on)
- HBM2: **ALIVE** — write/readback `0xdeadbeef` passes on both cards
- PRAMIN: Accessible via PRAMIN window

## Implications

1. **mmiotrace captures topology + memory init but NOT firmware boot** — for FECS/GPCCS debugging, need falcon register polling, not mmiotrace
2. **nvidia-open is opaque** — GSP firmware means no MMIO observability; nouveau is the only traceable path
3. **Timing is deterministic** — sub-1% variance means timing-based adaptive settle can converge quickly
4. **Warm-fecs second pass is 10× faster** — 2.7s vs 21.9s, driver detects initialized state
5. **Journal data feeds AdaptiveLifecycle** — 12 observations already; settle times and reset methods can now adapt
