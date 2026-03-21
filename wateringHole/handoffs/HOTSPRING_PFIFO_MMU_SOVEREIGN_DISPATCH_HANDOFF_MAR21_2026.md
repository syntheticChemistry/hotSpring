# Handoff: Sovereign PFIFO Dispatch — MMU Page Table Cracking, Diagnostic Matrix Results

**Date:** March 21, 2026
**From:** hotSpring
**To:** coralReef, toadStool, barraCuda
**License:** AGPL-3.0-only
**Covers:** 54-config PFIFO diagnostic matrix, warm-state GPU transfer, PFIFO re-initialization sequence, PBDMA/MMU root cause analysis, sovereign command submission architecture
**Experiment:** 071

---

## Executive Summary

- **54-configuration diagnostic matrix** tests every channel initialization strategy on GV100 (Titan V) via VFIO; 12 winning configs achieve scheduler acceptance, zero faults
- **PFIFO re-initialization solved**: PMC reset → soft enable → preempt all runlists → clear PBDMAs produces clean `STATE=0x00000000` in ~25ms
- **Root cause identified**: PBDMA loads GPFIFO base address (0x1000) but `0xbad00200` (PBUS timeout) on fetch — the GPU MMU cannot translate page table entries to reach host memory
- **All prior layers validated**: warm-state transfer, PFB/MMU alive, PFIFO engine, scheduler dispatch, runlist submission, channel binding, PBDMA context loading — only MMU page table translation remains
- **Parallel-safe**: This work runs in `coralReef/crates/coral-driver` and does not touch barraCuda or toadStool source. Teams can evolve in parallel

---

## Part 1: PFIFO Re-Initialization Sequence (SOLVED)

After nouveau unbind, the PFIFO engine is disabled and PBDMAs hold stale contexts.
The following 5-phase sequence reliably produces a clean PFIFO state:

```
Phase 1: PMC-level PFIFO reset (bit 8 of pmc::ENABLE 0x200)
  - Clear bit → 2ms delay → set bit → 5ms delay
Phase 2: PFIFO soft enable (0x2200 = 0 → 1)
  - 10ms delay for scheduler init
Phase 3: Preempt ALL active runlists
  - Read PBDMA_MAP, decode runlist assignments
  - Write bitmask to GV100_PREEMPT (0x2638)
  - Poll INTR for BIT30 (INTR_RL_COMPLETE)
  - ACK by writing BIT30 to INTR
Phase 4: Empty-flush each discovered runlist
  - Submit empty runlist (count=0) to each runlist register
  - Wait for BIT30 ACK per flush
Phase 5: Force-clear PBDMA registers
  - Zero context save area (0x000-0x1FC)
  - Zero operational registers (GP_BASE, GP_FETCH, GP_PUT, USERD, STATUS)
  - Clear interrupts (INTR=0xFFFFFFFF, INTR_EN=0)
```

**Timing**: ~25ms total. PBDMA STATE transitions from stale (e.g., `0x80801222`)
to clean (`0x00000000`).

---

## Part 2: Diagnostic Matrix Results

| Metric | Value |
|--------|-------|
| Total configs | 54 |
| Faulted | 0 |
| Scheduler-accepted | 12 |
| Clean (no fault + scheduled) | 12 |
| PBDMA registers changed | 40 |

### Winning Configurations

All 12 show `PCCSR=0x11000003` (ENABLE | NEXT | STATUS=PENDING | BUSY):

| Config | Key Behavior |
|--------|-------------|
| I_activate_sched | Bind → enable → runlist → doorbell: scheduler accepts |
| T_sched_doorbell | Scheduler + doorbell: scheduler accepts |
| R_ramfc_sched | RAMFC context → scheduler: GP_FETCH=0x1000 loaded |
| S_both_sched | Direct PBDMA + RAMFC: GP_FETCH=0x1000, SIG=0xFACE |
| U_cleanSched | Clean scheduler: GP_FETCH=0x1000 |
| U2_nopPushbuf | NOP pushbuffer: GP_FETCH=0x1000, GP_PUT=1 |

### Root Cause: MMU Translation Failure

In configs R, S, U, U2 the PBDMA successfully loads:
- `GP_BASE=0x00001000` (correct GPFIFO IOVA)
- `USERD=0x00002002` (correct USERD IOVA + coherent target)
- `SIG=0x0000FACE` (correct channel signature)
- `GP_PUT=1` (one entry pending)
- `GP_FETCH=4096` (0x1000 — loaded base but NOT advancing)

Config R additionally shows `METHOD=0xbad00200 DATA=0xbad00200` — PBUS timeout.
The PBDMA tried to fetch GPFIFO entries from GPU VA 0x1000 through the FBHUB MMU
and the translation failed.

---

## Part 3: MMU Page Table Architecture

### Current Setup (System Memory Identity Map)

```
Instance Block (IOVA 0x3000):
  ├── RAMFC (0x000-0x1FF): GPFIFO base, USERD, signature, GP_PUT/GET
  └── RAMIN (0x200+): Page Directory Base → PD3 at IOVA 0x5000
       ├── V2 format (bit 10), big_page=64K (bit 11), VOL (bit 2)
       └── Aperture: SYS_MEM_COHERENT (target=2)

5-Level Page Table Chain:
  PD3 (0x5000) → PD2 (0x6000) → PD1 (0x7000) → PD0 (0x8000) → PT0 (0x9000)

PT0: 512 entries × 8 bytes = 4KB, identity-maps first 2MB:
  GPU VA 0x1000 → IOVA 0x1000 (GPFIFO ring)
  GPU VA 0x2000 → IOVA 0x2000 (USERD page)
  GPU VA 0x3000 → IOVA 0x3000 (instance block)
  ...
  GPU VA 0x9000 → IOVA 0x9000 (PT0 itself)
```

### PDE/PTE Encoding (V2 Format)

```
PDE: (target_iova >> 4) | (aperture << 1) | (VOL << 3)
  aperture: 2 = SYS_MEM_COHERENT
  VOL: 1 = volatile

PTE: (phys_addr >> 4) | VALID | (aperture << 1) | (VOL << 3)
  VALID: bit 0 = 1
  aperture: 2 = SYS_MEM_COHERENT
```

### Suspected Issues

1. **IOMMU mapping gap**: Page table buffers at IOVAs 0x5000-0x9000 may not be
   IOMMU-mapped (they're DMA buffers, but need verification)
2. **BAR2 requirement**: PBDMA may access page tables through BAR2 (VRAM window)
   rather than directly through system memory — the FBHUB MMU path may require
   BAR2 to be configured for the page table chain
3. **Subcontext format**: GV100 uses subcontexts (SC0/SC1 page directory base at
   0x2A0/0x2B0) — the subcontext PDB may need different encoding
4. **TLB invalidation**: After programming page tables, TLB may need explicit
   invalidation via `pfb::MMU_INVALIDATE`

---

## Part 4: Action Items

### For coralReef (immediate)

1. **Verify DMA buffer IOMMU mapping** for page table IOVAs (0x5000-0x9000).
   Each page table must be a `DmaBuffer` with a valid IOMMU mapping.
2. **Add MMU fault buffer decode** — read `mmu::FAULT_BUF0_GET/PUT` after failed
   dispatch to get exact fault address, type, and instance.
3. **Try explicit TLB invalidation** via `pfb::MMU_INVALIDATE` after page table setup.
4. **Test BAR2-resident page tables** — write page tables to VRAM via PRAMIN,
   reference from instance block with `aperture=VRAM(1)`.

### For toadStool

1. **No blocking action** — the sovereign pipeline is a coralReef/hotSpring concern.
2. **GlowPlug socket client** remains the primary integration target.
3. **hw-learn**: When sovereign dispatch works, expose PBDMA health metrics
   (STATE register, GP_FETCH progress, fault counts) as learning signals.

### For barraCuda

1. **No code changes needed** — the diagnostic matrix lives in coral-driver.
2. **RegisterMap convergence**: The PBDMA register layout discovered here
   (CTX_GP_FETCH_BYTE at 0x050, not GP_FETCH at 0x048) should be captured
   in any shared register map.
3. **Future**: Once sovereign dispatch works, barraCuda's `MdEngine<B>` can
   route through a `SovereignBackend` that uses coral-driver for GPFIFO submission.

---

## Part 5: Register Reference (Compact)

### PFIFO Initialization Registers

| Register | Offset | Purpose |
|----------|--------|---------|
| PMC_ENABLE | 0x0200 | Bit 8 = PFIFO domain enable |
| PFIFO_ENABLE | 0x2200 | Soft enable (0→1 transition) |
| PBDMA_MAP | 0x2004 | Bitmask of active PBDMAs |
| GV100_PREEMPT | 0x2638 | Runlist-level preempt trigger |
| PFIFO_INTR | 0x2100 | Interrupt status (BIT30 = RL_COMPLETE) |
| RUNLIST_BASE(id) | 0x2270 + id*0x10 | Per-runlist base address |
| RUNLIST_SUBMIT(id) | 0x2274 + id*0x10 | Per-runlist submit trigger |

### PBDMA Registers (per-PBDMA, stride 0x2000 from 0x40000)

| Register | Offset | Notes |
|----------|--------|-------|
| CTX_GP_BASE_LO | 0x048 | Context save: GPFIFO base (NOT operational GP_FETCH) |
| CTX_GP_FETCH_BYTE | 0x050 | Real GP_FETCH (byte-granular fetch pointer) |
| CTX_GP_PUT | 0x054 | Context save: GP_PUT index |
| CHANNEL_STATE | 0x0B0 | PBDMA operational state |
| USERD_LO/HI | 0x0D0/0D4 | User data page address |
| INTR | 0x108 | PBDMA interrupt status |

### PCCSR Channel Status Decode

| Value | Name | Meaning |
|-------|------|---------|
| 0 | IDLE | Not on any runlist |
| 1 | PENDING | On runlist, waiting for PBDMA assignment |
| 5 | ON_PBDMA | Dispatched to PBDMA, fetching commands |
| 6 | ON_PBDMA+ENG | On PBDMA and engine simultaneously |
| 7 | ON_ENG | Running on compute engine |

---

## Part 6: Sovereign Pipeline Layer Status

| Layer | Component | Status |
|-------|-----------|--------|
| 0 | PCIe / VFIO | ✅ Working — BAR0 MMIO, DMA buffers, IOMMU |
| 1 | PFB / MMU | ✅ Alive via warm-state transfer from nouveau |
| 2 | PFIFO Engine | ✅ Fully re-initialized (PMC + soft enable + preempt) |
| 3 | Scheduler | ✅ Processes runlists, dispatches channels (empty flush BIT30) |
| 4 | Channel | ✅ Accepted by scheduler (PCCSR STATUS=PENDING) |
| 5 | PBDMA Context | ✅ Loaded (GP_BASE, USERD, SIG all correct) |
| 6 | MMU Translation | ❌ **BLOCKING** — PBDMA cannot fetch through GPU MMU |
| 7 | GPFIFO Fetch | Blocked by Layer 6 |
| 8 | Command Execution | Blocked by Layer 7 |
| 9 | FECS/GPCCS | Pending (firmware load after dispatch works) |
| 10 | Shader Dispatch | Pending (after FECS/GPCCS) |

**Six of ten layers are proven working. The MMU page table translation is the single remaining blocker for PBDMA command fetch.**
