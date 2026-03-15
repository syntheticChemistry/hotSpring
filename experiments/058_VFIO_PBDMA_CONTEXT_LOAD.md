# Experiment 058: VFIO PBDMA Context Load on Volta (GV100)

**Date**: March 13-14, 2026
**Hardware**: Titan V (GV100, SM70) at `0000:4b:00.0` on `vfio-pci`
**Software**: coralReef Phase 10, Iteration 44+, `coral-driver` crate
**Test**: `cargo test --test hw_nv_vfio --features vfio -- --ignored --test-threads=1 --nocapture vfio_dispatch_nop_shader`
**Status**: 6/7 pass — PBDMA context loaded, GP_PUT DMA read remaining
**Chat**: [VFIO PBDMA breakthrough](28732f32-750e-4053-a1ae-a8d39a738d7a)

---

## Objective

Achieve full VFIO compute dispatch on Titan V (GV100) by correctly
programming the PFIFO hardware scheduler without any kernel GPU driver.
Starting from Experiment 057's state (6/7 pass, scheduler ignoring runlist
submissions), determine why the hardware scheduler does not dispatch
channels and fix it.

---

## Design: Diagnostic Matrix (Experiments A-Q)

A systematic matrix of experiments was designed to test different
combinations of memory targets, initialization sequences, and scheduler
interactions. Each experiment isolates a single variable.

### Phase 1: Memory Target Variations (A-D)

| Exp | Instance Block | GPFIFO | USERD | Result |
|-----|---------------|--------|-------|--------|
| A (baseline) | SysMem COH | SysMem COH | SysMem COH | PCCSR scheduled, PBDMA idle |
| B | SysMem COH | SysMem COH | SysMem COH + zero runlist flush | Same |
| C | SysMem COH | SysMem COH | SysMem COH + preempt(0x2634) | Channel descheduled |
| D | SysMem COH | SysMem COH | SysMem COH + inst_bind before enable | Same as A |

### Phase 2: Deep Scheduler Probing (E-L)

| Exp | Key Change | Result |
|-----|-----------|--------|
| E | Empty runlist flush before real submit | No change |
| F | PMC_ENABLE toggle (PFIFO reset) | GPU enters cold state — destructive |
| G | PBDMA direct programming (bypass scheduler) | PBDMA doesn't respond |
| H | Verify PBDMA base addresses via BAR0 scan | Mapped PBDMA1/2/3/21 |
| I | Read ENGN0_STATUS for GR runlist | Confirmed runlist 1 = GR engine |
| J | Full PCCSR fault clear sequence | Reduced persistent faults |
| K | VRAM instance block (via PRAMIN) | INST_BIND fault persists |
| L | L2 cache flush after PRAMIN write | No improvement |

### Phase 3: Breakthrough Protocol (M-Q)

| Exp | Key Change | Result |
|-----|-----------|--------|
| M | Full PBDMA register dump (0x00-0x1FF) | Revealed stale nouveau context |
| N | gv100 preempt at 0x002638 (not 0x2634) | Stale context evicted |
| O | gk104 runlist ACK at 0x002A00 | **BREAKTHROUGH**: scheduler dispatches |
| P | SIGNATURE=0xDEAD test | Confirmed fresh context load |
| Q | Full protocol + ectx bind + clean SIG | PBDMA loaded, zero errors |

---

## The Preempt + ACK Protocol

### Discovery: GV100 Runlist Preempt

Experiments A-D used `0x002634` for preempt, which is the per-channel
preempt register. Experiment C showed this actually descheduled the
channel. The correct register for Volta is:

```
0x002638: Per-runlist preempt (GV100+)
Write BIT(runlist_id) to force scheduler re-evaluation
```

Source: `gv100_runl_preempt()` in nouveau `gv100.c`.

### Discovery: Runlist Completion ACK

After runlist submission, PFIFO fires INTR bit 30 (0x40000000). Without
acknowledging, the scheduler stays in a "waiting for software" state:

```
1. Wait for PFIFO_INTR bit 30
2. Read 0x002A00 → completed runlist bitmask
3. Write BIT(runl_id) to 0x002A00 → acknowledge
4. Write 0x40000000 to 0x002100 → clear PFIFO_INTR
```

Source: `gk104_fifo_intr_runlist()` in nouveau `gk104.c`.

---

## PBDMA Register Dump Analysis

### Full 0x00-0x1FF Scan Results (Experiment M)

Before breakthrough, PBDMA2 showed default state:
```
PBDMA2 (base=0x42000):
  SIGNATURE (0x010): 0x00000000  (empty — no context loaded)
  GP_BASE   (0x048): 0x00000000
  STATE     (0x110): 0x00000000  (idle, clean)
  INTR      (0x108): 0x00000000
```

PBDMA1 showed stale nouveau context:
```
PBDMA1 (base=0x41000):
  SIGNATURE (0x010): 0x00003ACE  (nouveau's signature)
  STATE     (0x110): 0x07800000  (halted/frozen)
```

After breakthrough (Experiment Q), PBDMA2 loads our context:
```
PBDMA2 (base=0x42000):
  USERD_LO  (0x008): 0x00002001  (IOVA 0x2000, target=COH)
  SIGNATURE (0x010): 0x0000FACE  (correct — validates OK)
  GP_BASE   (0x048): 0x00001000  (our GPFIFO at IOVA 0x1000)
  GP_BASE_HI(0x04C): 0x00070000  (SYS_MEM_COH, limit=7)
  CHID      (0x0E8): 0x00000000  (channel 0)
  CONFIG    (0x0F4): 0x00001100  (changed from default 0x100)
  CHAN_INFO  (0x0F8): 0x10003080  (changed from default 0x3080)
  STATE     (0x110): loaded      (active context)
  INTR      (0x108): 0x00000000  (zero errors)
```

---

## SIGNATURE Validation (Experiment P)

To confirm the scheduler was loading a **fresh** context (not stale state),
we wrote `0x0000DEAD` to the RAMFC SIGNATURE field at instance block offset
0x010 (in VRAM via PRAMIN).

Result: PBDMA2 INTR = 0x80000000 — bit 31 fires.

Lookup in `gk104_runq_intr_0_names`:
```
{ 0x80000000, "SIGNATURE" }   ← This bit
```

This is a SIGNATURE mismatch error (PBDMA enforces SIGNATURE=0xFACE).
This conclusively proves:
1. The scheduler loaded our RAMFC context (it saw 0xDEAD, not stale data)
2. PBDMA validates the SIGNATURE field during context load
3. The SIGNATURE field must be 0xFACE for clean operation

Reverting to 0xFACE → PBDMA2 INTR = 0x00000000 (clean).

---

## Engine Context Binding

From nouveau `gv100_ectx_bind()`:

```
inst[0x210] = lower_32(ctx_addr | 4)    // engine context, bit 2 set
inst[0x214] = upper_32(ctx_addr | 4)
inst[0x0AC] |= 0x00010000               // bind flag (bit 16)
```

Without this, the PBDMA fires CTXNOTVALID via HCE INTR (offset 0x148,
bit 31). We used PT0_IOVA as a placeholder for the context address.
After adding the binding, all PBDMA interrupts clear.

---

## Current Blocker: USERD GP_PUT DMA Read

### Evidence

PBDMA2 has our context loaded with zero errors:
- GP_BASE = 0x1000 (correct GPFIFO address)
- USERD_LO = 0x2001 (IOVA 0x2000, SYS_MEM_COH)
- SIGNATURE = 0xFACE (validated)
- INTR = 0x00000000 (no errors)
- GP_FETCH remains at 0 — PBDMA is not fetching

Host memory at USERD + 0x8C (GP_PUT offset) contains value 1.
The doorbell at 0x810090 was rung with channel_id=0.
But the PBDMA does not advance GP_FETCH.

### Hypothesis

The PBDMA's DMA reads from system memory may not traverse the IOMMU.
VFIO maps host pages at IOVA addresses for DMA, and explicit DMA
transfers (upload/readback test) work. But the PBDMA's USERD polling
may use a different internal DMA path that doesn't go through the
standard IOMMU translation.

### Test: VRAM USERD

Write USERD (specifically GP_PUT=1 at offset 0x8C) to VRAM via PRAMIN
at offset 0x0000. Update RAMFC USERD_LO to point to VRAM (target=0).
If the PBDMA reads GP_PUT from VRAM, this confirms the IOMMU hypothesis.

---

## Nouveau Source References

| Function | Source | Relevance |
|----------|--------|-----------|
| `gv100_runl_preempt` | gv100.c | 0x002638 with BIT(runl_id) |
| `gk104_fifo_intr_runlist` | gk104.c | 0x002A00 ACK loop |
| `gk104_runl_commit` | gk104.c | 0x002270/0x002274 format |
| `gv100_chan_ramfc_write` | gv100.c | RAMFC offset definitions |
| `gv100_ectx_bind` | gv100.c | inst[0x210/0x214/0x0AC] |
| `gk104_runq_intr_0_names` | gk104.c | PBDMA INTR bit definitions |
| `gv100_runq_intr_1_ctxnotvalid` | gv100.c | HCE CTXNOTVALID handling |

---

## Infrastructure Built

| Component | Status | Location |
|-----------|--------|----------|
| Experiment Q (VramFullDispatch) | Working | `channel.rs` |
| Preempt + ACK loop | Working | `channel.rs` |
| Full PBDMA register dump | Working | `channel.rs` |
| PRAMIN VRAM write | Working | `channel.rs` |
| warm_and_test.sh | Working | `scripts/` |
| capture_nouveau_mmiotrace.sh v1/v2 | Ready | `scripts/` |
| nouveau_reference_bar0.txt | Captured | `scripts/` |

---

## BREAKTHROUGH: RUNLIST_SUBMIT Register Fix (March 15, 2026)

### Root Cause Found

GV100 uses `gk104_runl_commit()`, which has a **two-register** global format:

```c
nvkm_wr32(device, 0x002270, (target << 28) | (addr >> 12));  // RUNLIST_BASE
nvkm_wr32(device, 0x002274, (runl->id << 20) | count);       // RUNLIST_SUBMIT trigger
```

We were treating `0x002274` as `RUNLIST_BASE_HI` (writing 0) and writing
count to a non-existent register at `0x002278`. The write of 0 to `0x002274`
accidentally triggered an empty submission (count=0), and **the actual count=2
submission never reached the hardware**.

### Additional Fix: Channel Entry Encoding

`inst_target` was placed in DW0[5:4] (reserved bits) instead of DW2[21:20]:
```
DW0: [31:8] USERD_ADDR, [7:6] USERD_TARGET, [1] RUNQ  ← removed inst_target from [5:4]
DW2: [31:12] INST_ADDR, [21:20] INST_TARGET, [11:0] CHID  ← added inst_target here
```

### Results After Fix

| Metric | Before | After |
|--------|--------|-------|
| Scheduled experiments | 0 | **8** |
| PBDMA context loaded from RAMFC | No | **Yes** |
| PBDMA fetches GPFIFO | No | **Yes** (GP_FETCH=0x1000) |
| PCCSR BUSY bit | Never set | **Set** (0x15000003) |

Scheduled experiments: `I_activate_sched_{coh,ncoh}`, `T_sched_doorbell_{coh,ncoh}`,
`R_ramfc_sched_{coh,ncoh}`, `S_both_sched_{coh,ncoh}`.

The R experiment proves end-to-end RAMFC loading:
```
CTX: USERD=0x00002001 SIG=0x0000FACE GP_PUT=1
DIRECT: GP_FETCH=4096 (0x1000 = GPFIFO_IOVA)
```

### Remaining Issue

All 8 scheduled experiments fault (PBDMA_FAULTED | ENG_FAULTED in PCCSR).
PBDMA attempts GPFIFO fetch but faults, likely because:
- GPFIFO buffer has no valid push buffer entries (GP_PUT=1 but no commands)
- MMU page tables may need verification for PBDMA virtual address translation
- PBDMA CONFIG/CHANNEL_INFO may need additional fields set

---

## Forward Plan

1. Investigate PBDMA fault type: read detailed interrupt/fault registers
2. Populate valid GPFIFO entries (NOP command) to prevent invalid command faults
3. Verify MMU page table setup for GPU virtual → IOVA translation
4. Achieve clean scheduling (SCHED + no fault) → 7/7 dispatch
5. Cross-test on RTX 5060 to validate portability of diagnostic matrix

---

*The scheduler dispatches. The PBDMA loads our context and fetches our GPFIFO. The last mile is valid commands.*
