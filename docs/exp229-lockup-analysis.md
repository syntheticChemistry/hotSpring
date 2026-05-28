# Exp 229: Catalyst Channel — System Lockup Analysis

## Status: ROOT CAUSE IDENTIFIED, FIX PENDING TEST

### Lockup Mechanism (confirmed across 5 incidents)

**Kill chain**: When `rm_trigger` exits and closes its nvidia file descriptors,
`nvidia_close` runs the teardown path. Because `rm_disable_adapter` and
`rm_shutdown_adapter` are NOP'd (to keep the GPU warm), the teardown skips GPU
register quiescing but STILL runs `free_irq()` + `pci_disable_msi()`.

Result: GPU is warm with 23 active engines, `INTR_EN=0x7fffffff` (all interrupt
sources enabled), MSI handler removed, MSI disabled at PCI level → GPU falls
back to legacy INTx → level-triggered INTx fires with no handler to ACK →
**infinite interrupt storm** → all CPUs saturated → system lockup requiring
power cycle.

### Evidence Trail

#### Sentinel v2 data (lockup #5, 2026-05-28 08:19)

```
TICK 15: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(6)
TICK 16: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(6)
TICK 17: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(6)
TICK 18: pmc=0x5fecdff1 pop=23 intr_en=0x7fffffff  driver=nvsov  mods=nvsov(0) ← rm_trigger exited
[NO TICK 19 — SYSTEM DEAD]
```

#### Why the quench fix from lockup #4 failed

```
[QUENCH] NV_PMC_INTR_EN_0: 0x7fffffff → 0x7fffffff
```

On Volta (GV100), `NV_PMC_INTR_EN(0)` at BAR0 offset **0x140 is READ-ONLY**.
NVIDIA uses a SET/CLEAR register pair pattern:
- `0x140` — INTR_EN: read current enable mask (READ-ONLY)
- `0x160` — INTR_EN_SET: write 1 bits to enable interrupt sources (WRITE-ONLY)
- `0x180` — INTR_EN_CLEAR: write 1 bits to disable interrupt sources (WRITE-ONLY)

The fix wrote 0 to 0x140 (no-op). Must write `0xFFFFFFFF` to **0x180** instead.

### Fix Applied

In `rm_trigger.rs`, `quench_gpu_interrupts()`:
- Write `0xFFFFFFFF` to `NV_PMC_INTR_EN_CLEAR(0)` at BAR0+0x180
- Read back `NV_PMC_INTR_EN(0)` at 0x140 to verify it becomes 0
- Read `NV_PMC_INTR(0)` at 0x100 to ACK any pending interrupts

### Timeline of all lockups

| # | Date | Cause | Fix attempted | Fix worked? |
|---|------|-------|---------------|-------------|
| 1 | May 28 early | Unknown (pre-sentinel) | Added keepalive exclusion | No |
| 2 | May 28 early | Unknown (pre-sentinel) | Extended exclusion to bridges | No |
| 3 | May 28 early | Unknown (pre-sentinel) | Removed nv_pci_remove NOPs | No |
| 4 | May 28 07:39 | IRQ storm after rm_trigger exit | Write 0 to INTR_EN@0x140 | No — 0x140 is read-only |
| 5 | May 28 08:19 | Same IRQ storm, quench confirmed failed | Write 0xFFFFFFFF to INTR_EN_CLEAR@0x180 | **PENDING TEST** |

### Key registers (Volta GV100)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x000 | NV_PMC_BOOT_0 | RO | Chip identification |
| 0x100 | NV_PMC_INTR(0) | R/W | Pending interrupts (read clears edge) |
| 0x140 | NV_PMC_INTR_EN(0) | RO | Current interrupt enable mask |
| 0x160 | NV_PMC_INTR_EN_SET(0) | WO | Write 1 to enable interrupt source |
| 0x180 | NV_PMC_INTR_EN_CLEAR(0) | WO | Write 1 to disable interrupt source |
| 0x200 | NV_PMC_ENABLE | RW | Engine enable (warmness indicator) |

### Safety architecture (diesel engine)

1. **Primary defense**: `quench_gpu_interrupts()` in rm_trigger writes to
   INTR_EN_CLEAR before closing nvidia fds
2. **Watchdog**: `catalyst-watchdog` thread in toadstool monitors handoff
   liveness, performs emergency quench + process kill if pipeline hangs
3. **Sentinel**: External bash script logs BAR0 state every second (no PCI
   config reads — those take pci_lock and can deadlock)
4. **Keepalive exclusion**: RAII guard excludes target BDF + bridges from
   keepalive config reads during handoff
