# Experiment 071: PFIFO Diagnostic Matrix — HBM2 Pipeline Cracking

**Date:** March 18-21, 2026
**Hardware:** Titan V (GV100, 0000:03:00.0) on biomeGate, vfio-pci via warm-state transfer
**Status:** 🔄 Active — MMU page table mapping identified as root cause of PBDMA fetch failure
**Depends:** Exp 058 (VFIO PBDMA context load), Exp 060 (BAR2 self-warm), Exp 062 (D3hot VRAM), Exp 069-070 (GlowPlug/Ember)

---

## Objective

Achieve sovereign GPU command submission on GV100: host-written GPFIFO entries
fetched by PBDMA, executed by the scheduler, without any kernel driver assistance.
This is the final gate before shader dispatch (FECS/GPCCS firmware load).

## Method: 54-Configuration Diagnostic Matrix

A systematic experiment matrix testing every combination of channel initialization
strategies across two coherency modes (coherent / non-coherent system memory):

| Category | Experiments | What It Tests |
|----------|------------|---------------|
| A-D: Bind ordering | 8 configs | PCCSR inst bind → enable → runlist sequencing |
| E-I: Direct PBDMA | 10 configs | Programming PBDMA registers directly vs via scheduler |
| J-L: VRAM instance | 6 configs | Instance block in VRAM vs system memory |
| N-P: Full dispatch | 6 configs | Complete channel lifecycle with fault analysis |
| R-T: RAMFC mirror | 6 configs | Context save area vs direct register programming |
| U-U2: Clean sched | 4 configs | Fresh scheduler with NOP pushbuffer entries |
| V-Z7: Reinit variants | 14 configs | PMC reset, preempt, runlist ACK, bind protocols |

### Warm-State Transfer Protocol

The GPU must be "warm" (HBM2 trained, PFB/MMU enabled) for PFIFO experiments.
Sequence:

1. Unbind vfio-pci from Titan V
2. Clear `reset_method`, bind nouveau → HBM2 PHY training executes
3. Clear `reset_method`, unbind nouveau (PFIFO disabled, but PFB/MMU alive)
4. Bind vfio-pci → VFIO BAR0 access with warm PFB/MMU state

The runner detects warm state via `PFB=0x0000ffff` (alive) and **skips**
GlowPlug cold-start initialization to preserve nouveau's MMU/BAR2 setup.

## Results (March 21, 2026)

### PFIFO Initialization — SOLVED

After nouveau unbind leaves PFIFO disabled (`ENABLE=0`), the following
sequence successfully re-initializes the engine:

1. **PMC-level PFIFO reset** (bit 8 of `pmc::ENABLE` 0x200)
2. **PFIFO soft enable** (`pfifo::ENABLE` 0→1 transition)
3. **Preempt ALL active runlists** via `pfifo::GV100_PREEMPT` bitmask
4. **Wait for `INTR_RL_COMPLETE` (BIT30)** — acknowledge interrupt
5. **Force-clear PBDMA registers** for target runlist PBDMAs

This sequence takes ~25ms and produces clean PBDMA states (`STATE=0x00000000`)
for all PBDMAs on the target runlist.

### Diagnostic Matrix Summary

```
Total:        54
Faulted:      0
Scheduled:    12
Clean:        12 (no fault + scheduled)
PBDMA ours:   40 (registers changed from residual)
```

### 12 Winning Configurations (channel accepted by scheduler)

All show `PCCSR=0x11000003` (ENABLE | NEXT | PENDING | BUSY):

- I_activate_sched (coh/ncoh)
- T_sched_doorbell (coh/ncoh)
- R_ramfc_sched (coh/ncoh)
- S_both_sched (coh/ncoh)
- U_cleanSched (coh/ncoh)
- U2_nopPushbuf (coh/ncoh)

### Current Blocker: MMU Page Table Translation Failure

**Root cause identified:** The PBDMA loads the GPFIFO base address (GPU VA 0x1000)
from the channel context but **cannot fetch entries through the GPU MMU**.

Evidence:
- `GP_FETCH=4096` (0x1000) in R/S/U/U2 experiments — PBDMA knows where GPFIFO is
- `METHOD=0xbad00200` (PBUS timeout) — the bus access timed out during fetch
- `GP_FETCH` does not advance past the base address
- Channel STATUS remains PENDING (1), never transitions to ON_PBDMA (5)
- Empty runlist flushes successfully trigger BIT30 (scheduler is functional)
- Non-empty runlist submissions do NOT trigger BIT30 (scheduler stalls on channel dispatch)

### Architecture: 5-Level GPU Page Table (PD3→PD2→PD1→PD0→PT0)

Identity mapping: GPU VA 0x1000 → IOVA 0x1000 → host physical memory.
Page tables allocated as DMA buffers at IOVAs 0x5000-0x9000.

The PBDMA accesses through the FBHUB MMU using the instance block's page
directory pointer. The `0xbad00200` indicates the MMU translation is failing —
either the page table format is wrong for Volta, or the page table chain
addresses are not reachable by the PBDMA's access path.

### Register Map Discoveries

| Register | Offset | Behavior |
|----------|--------|----------|
| `pfifo::PBDMA_MAP` | 0x2004 | Bitmask of active PBDMAs (0x0020000e = PBDMAs 1,2,3,21) |
| `pfifo::GV100_PREEMPT` | 0x2638 | Runlist-level preempt — write bitmask, wait BIT30 |
| `pfifo::INTR_RL_COMPLETE` | BIT30 | Fires for empty flushes AND preempts, NOT for non-empty runlists |
| `pbdma::CTX_GP_FETCH_BYTE` | 0x050 | Real GP_FETCH register (byte-granular), not 0x048 |
| PCCSR STATUS=1 | bits 27:24 | PENDING — on runlist but not dispatched |
| PCCSR STATUS=5 | bits 27:24 | ON_PBDMA — successfully dispatched (target) |
| `0xbad00200` | PBUS error | Bus timeout — MMU translation failure |

### PFIFO Engine Domain Relationships

```
PMC (0x200)          Controls clock domains (bit 8 = PFIFO)
  └── PFIFO (0x2200)   Scheduler + PBDMA management
       ├── Scheduler   Reads runlists, dispatches to PBDMAs
       │    └── GV100_PREEMPT (0x2638)  Runlist-level preempt
       ├── PBDMA[1-3]  Command fetch engines (stride 0x2000 from 0x40000)
       │    ├── Context save area (0x000-0x1FC, mirrors RAMFC)
       │    ├── GP_BASE (0x040/044), GP_FETCH (0x050), GP_PUT (0x054)
       │    └── CHANNEL_STATE (0x0B0)
       └── PCCSR (0x800000)  Per-channel status registers
            ├── inst(ch)    Instance block pointer + BIND trigger
            └── channel(ch)  Status + enable/disable
```

## Next Steps

1. **Debug MMU page table format** — verify PDE/PTE encoding matches Volta V2 format
2. **Check page table buffer IOMMU reachability** — verify DMA buffers at 0x5000-0x9000 are IOMMU-mapped
3. **Read MMU fault buffer** — decode fault entries for specific VA/aperture/reason
4. **Try BAR2-resident page tables** — use VRAM page tables (nouveau-configured BAR2) instead of system memory
5. **NOP GPFIFO entries** — put valid NOP entries to eliminate empty-buffer faults once fetch works

## Lessons Learned

1. **nouveau disables PFIFO on unbind** — must explicitly re-enable via PMC + soft toggle
2. **PBDMAs hold stale channel contexts** — PMC reset alone insufficient; preempt + clear required
3. **GP_FETCH register is at 0x050, not 0x048** — 0x048 is GP_STATE (context save offset)
4. **Empty runlist flushes are the canary** — BIT30 fires for empty, confirms scheduler is alive
5. **Warm-state preservation is critical** — GlowPlug cold-init destroys nouveau MMU setup
6. **The scheduler PENDING→ON_PBDMA transition requires working MMU** — all prior layers (PFIFO, scheduler, runlist, channel, PBDMA) are functional

## Files Modified

- `coralReef/crates/coral-driver/src/vfio/channel/diagnostic/runner.rs` — warm-state detection, PFIFO init, PBDMA clearing
- `coralReef/crates/coral-driver/src/vfio/channel/diagnostic/types.rs` — `pbdma_gp_fetch_050` field
- `coralReef/crates/coral-driver/src/vfio/channel/diagnostic/experiments/context.rs` — BIT30 ACK in submit_runlist
- `coralReef/crates/coral-driver/src/vfio/channel/registers.rs` — register definitions and constants
